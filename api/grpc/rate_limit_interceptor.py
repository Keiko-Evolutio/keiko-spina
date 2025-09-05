"""Rate Limiting Interceptor für KEI-RPC gRPC Server.

Implementiert Token-Bucket-basiertes Rate Limiting pro Peer/Method mit
konfigurierbaren Limits und automatischer Bucket-Bereinigung.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import grpc

from kei_logging import get_logger

from .base_interceptor import BaseInterceptor, ServicerContext, UnaryUnaryHandler
from .constants import ErrorCodes, ErrorMessages, MetadataKeys, RateLimitConfig

logger = get_logger(__name__)


@dataclass
class TokenBucket:
    """Token-Bucket für Rate Limiting.

    Attributes:
        capacity: Maximale Anzahl Tokens
        refill_rate: Tokens pro Sekunde
        tokens: Aktuelle Token-Anzahl
        last_refill: Zeitpunkt der letzten Auffüllung
    """

    capacity: float
    refill_rate: float
    tokens: float
    last_refill: float

    def consume(self, tokens: int = 1) -> bool:
        """Versucht Tokens zu konsumieren.

        Args:
            tokens: Anzahl zu konsumierender Tokens

        Returns:
            True wenn Tokens verfügbar waren
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Füllt Bucket basierend auf verstrichener Zeit auf."""
        now = time.time()
        elapsed = now - self.last_refill

        # Neue Tokens basierend auf Refill-Rate hinzufügen
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def get_remaining(self) -> int:
        """Gibt Anzahl verbleibender Tokens zurück."""
        self._refill()
        return int(self.tokens)

    def get_reset_time(self) -> float:
        """Gibt Zeitpunkt zurück, wann Bucket wieder voll ist."""
        self._refill()
        if self.tokens >= self.capacity:
            return time.time()

        tokens_needed = self.capacity - self.tokens
        seconds_to_full = tokens_needed / self.refill_rate
        return time.time() + seconds_to_full


class RateLimitInterceptor(BaseInterceptor):
    """Rate Limiting Interceptor mit Token-Bucket-Algorithmus.

    Features:
    - Per-Peer und Per-Method Rate Limiting
    - Konfigurierbare Limits über Environment Variables
    - Automatische Bucket-Bereinigung
    - Rate-Limit-Headers in Response
    """

    def __init__(self) -> None:
        """Initialisiert Rate Limit Interceptor."""
        super().__init__("RateLimit")

        # Bucket-Storage: (peer_ip, method) -> TokenBucket
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._cleanup_task: asyncio.Task | None = None

        # Cleanup-Task starten (nur wenn Event Loop läuft)
        try:
            self._start_cleanup_task()
        except RuntimeError:
            # Kein Event Loop verfügbar - Task wird später gestartet
            pass

    def _start_cleanup_task(self) -> None:
        """Startet periodische Bucket-Bereinigung."""

        async def cleanup_buckets():
            while True:
                try:
                    await asyncio.sleep(RateLimitConfig.BUCKET_CLEANUP_INTERVAL_SECONDS)
                    await self._cleanup_expired_buckets()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.exception(f"Fehler bei Bucket-Bereinigung: {e}")

        try:
            # Prüfe ob Event Loop läuft
            asyncio.get_running_loop()
            self._cleanup_task = asyncio.create_task(cleanup_buckets())
        except RuntimeError:
            # Kein Event Loop - Task wird später gestartet
            self._cleanup_task = None

    async def _cleanup_expired_buckets(self) -> None:
        """Entfernt abgelaufene Buckets."""
        now = time.time()
        expired_keys = []

        for key, bucket in self._buckets.items():
            # Bucket als abgelaufen markieren wenn lange nicht verwendet
            if (now - bucket.last_refill) > RateLimitConfig.BUCKET_EXPIRY_SECONDS:
                expired_keys.append(key)

        for key in expired_keys:
            del self._buckets[key]

        if expired_keys:
            self.logger.debug(f"Bereinigt {len(expired_keys)} abgelaufene Rate-Limit-Buckets")

    async def _process_unary_unary(
        self, request: Any, context: ServicerContext, behavior: UnaryUnaryHandler, method_name: str
    ) -> Any:
        """Verarbeitet Unary-Unary Request mit Rate Limiting.

        Args:
            request: gRPC Request
            context: gRPC Service Context
            behavior: Original Handler
            method_name: Name der gRPC-Methode

        Returns:
            Response vom Original Handler

        Raises:
            grpc.RpcError: Bei Rate Limit Überschreitung
        """
        # 1. Peer-IP extrahieren
        peer_ip = self._extract_peer_ip(context)

        # 2. Rate Limit prüfen
        bucket = self._get_or_create_bucket(peer_ip, method_name)

        if not bucket.consume():
            # Rate Limit überschritten
            self._abort_with_rate_limit_error(context, bucket, method_name)

        # 3. Rate-Limit-Headers setzen
        self._set_rate_limit_headers(context, bucket)

        # 4. Original Handler ausführen
        return await behavior(request, context)

    def _extract_peer_ip(self, context: ServicerContext) -> str:
        """Extrahiert Peer-IP aus gRPC Context.

        Args:
            context: gRPC Service Context

        Returns:
            Peer-IP-Adresse
        """
        try:
            peer = context.peer()
            if peer and ":" in peer:
                # Format: "ipv4:127.0.0.1:12345" oder "ipv6:[::1]:12345"
                if peer.startswith("ipv4:"):
                    return peer.split(":")[1]
                if peer.startswith("ipv6:"):
                    # IPv6 kann komplexer sein, vereinfachte Extraktion
                    return peer.split("]")[0].replace("ipv6:[", "")
                # Fallback: alles vor dem letzten Doppelpunkt
                return peer.rsplit(":", 1)[0]
        except Exception as e:
            self.logger.warning(f"Fehler bei Peer-IP-Extraktion: {e}")

        return "unknown"

    def _get_or_create_bucket(self, peer_ip: str, method_name: str) -> TokenBucket:
        """Holt oder erstellt Token-Bucket für Peer/Method.

        Args:
            peer_ip: Peer-IP-Adresse
            method_name: gRPC-Methoden-Name

        Returns:
            Token-Bucket für diese Kombination
        """
        key = (peer_ip, method_name)

        if key not in self._buckets:
            # Limit für diese Methode ermitteln
            limit = RateLimitInterceptor._get_method_limit(method_name)
            refill_rate = limit / 60.0  # Pro Sekunde

            self._buckets[key] = TokenBucket(
                capacity=RateLimitConfig.TOKEN_BUCKET_CAPACITY,
                refill_rate=refill_rate,
                tokens=float(RateLimitConfig.TOKEN_BUCKET_CAPACITY),
                last_refill=time.time(),
            )

        return self._buckets[key]

    @staticmethod
    def _get_method_limit(method_name: str) -> int:
        """Ermittelt Rate-Limit für spezifische Methode.

        Args:
            method_name: gRPC-Methoden-Name

        Returns:
            Requests pro Minute für diese Methode
        """
        # Direkte Methoden-Limits prüfen
        if method_name in RateLimitConfig.METHOD_LIMITS:
            return RateLimitConfig.METHOD_LIMITS[method_name]

        # Fallback auf Wildcard
        return RateLimitConfig.METHOD_LIMITS.get("*", RateLimitConfig.DEFAULT_REQUESTS_PER_MINUTE)

    def _set_rate_limit_headers(self, context: ServicerContext, bucket: TokenBucket) -> None:
        """Setzt Rate-Limit-Headers in Response.

        Args:
            context: gRPC Service Context
            bucket: Token-Bucket mit aktuellen Werten
        """
        try:
            metadata = [
                (MetadataKeys.RATE_LIMIT_REMAINING, str(bucket.get_remaining())),
                (MetadataKeys.RATE_LIMIT_LIMIT, str(bucket.capacity)),
                (MetadataKeys.RATE_LIMIT_RESET, str(int(bucket.get_reset_time()))),
            ]

            context.set_trailing_metadata(metadata)

        except Exception as e:
            self.logger.warning(f"Fehler beim Setzen der Rate-Limit-Headers: {e}")

    def _abort_with_rate_limit_error(
        self, context: ServicerContext, bucket: TokenBucket, method_name: str
    ) -> None:
        """Bricht Request mit Rate-Limit-Fehler ab.

        Args:
            context: gRPC Service Context
            bucket: Token-Bucket mit aktuellen Werten
            method_name: gRPC-Methoden-Name
        """
        # Rate-Limit-Headers auch bei Fehler setzen
        self._set_rate_limit_headers(context, bucket)

        # Error-Metadata setzen
        context.set_trailing_metadata(
            [
                (MetadataKeys.ERROR_CODE, ErrorCodes.RATE_LIMIT_EXCEEDED),
                (MetadataKeys.ERROR_SEVERITY, "WARNING"),
            ]
        )

        # Detaillierte Fehlermeldung
        reset_time = int(bucket.get_reset_time() - time.time())
        error_message = (
            f"{ErrorMessages.RATE_LIMIT_EXCEEDED}. Versuche in {reset_time} Sekunden erneut."
        )

        self.logger.warning(
            f"Rate Limit überschritten für Methode {method_name}. "
            f"Reset in {reset_time} Sekunden."
        )

        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, error_message)

    async def _before_call(self, request: Any, context: ServicerContext, method_name: str) -> None:
        """Hook vor Handler-Aufruf für Logging."""
        peer_ip = self._extract_peer_ip(context)
        self.logger.debug(f"Rate Limit Check für {peer_ip} -> {method_name}")

    async def _on_error(
        self, request: Any, error: Exception, context: ServicerContext, method_name: str
    ) -> None:
        """Hook bei Fehlern für erweiterte Logging."""
        # Prüfe ob es ein gRPC Rate Limit Fehler ist
        if (isinstance(error, grpc.RpcError) and
            hasattr(error, "code") and
            callable(getattr(error, "code", None))):
            try:
                if error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    # Rate Limit Fehler - bereits geloggt
                    return
            except Exception:
                # Fehler beim Abrufen des Status Codes
                pass

        # Andere Fehler an Parent weiterleiten
        await super()._on_error(request, error, context, method_name)

    def __del__(self) -> None:
        """Cleanup beim Zerstören des Interceptors."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


__all__ = ["RateLimitInterceptor", "TokenBucket"]
