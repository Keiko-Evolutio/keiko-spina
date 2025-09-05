# backend/security/utils/rate_limiter.py
"""Zentrale Rate Limiting für Keiko Personal Assistant

Konsolidiert Rate Limiting-Logik aus verschiedenen Security-Modulen
und bietet einheitliche Rate Limiting-API.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function

from ..constants import RateLimitDefaults, SecurityErrorMessages, SecurityTimeouts

if TYPE_CHECKING:
    import asyncio

logger = get_logger(__name__)


class RateLimitType(str, Enum):
    """Typen von Rate Limits."""
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_TOKEN = "per_token"
    PER_OPERATION = "per_operation"
    GLOBAL = "global"


class RateLimitExceeded(Exception):
    """Exception für überschrittene Rate Limits."""

    def __init__(self, message: str, retry_after: int | None = None):
        """Initialisiert Rate Limit Exception.

        Args:
            message: Fehlermeldung
            retry_after: Sekunden bis nächster Versuch möglich
        """
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitInfo:
    """Informationen über Rate Limit-Status."""
    requests: list[float] = field(default_factory=list)
    blocked_until: float | None = None
    total_requests: int = 0
    blocked_count: int = 0

    def cleanup_old_requests(self, window_seconds: int) -> None:
        """Entfernt alte Requests außerhalb des Zeitfensters.

        Args:
            window_seconds: Zeitfenster in Sekunden
        """
        cutoff_time = time.time() - window_seconds
        self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]


@dataclass
class RateLimitResult:
    """Ergebnis einer Rate Limit-Prüfung."""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: int | None = None
    limit_type: RateLimitType | None = None
    identifier: str | None = None


class RateLimiter:
    """Zentrale Rate Limiting-Implementierung."""

    def __init__(self):
        """Initialisiert Rate Limiter."""
        self._rate_limits: dict[str, RateLimitInfo] = {}
        self._blocked_ips: set[str] = set()
        self._cleanup_task: asyncio.Task | None = None
        self._last_cleanup = time.time()

        # Standard-Limits
        self._limits = {
            RateLimitType.PER_IP: {
                "requests_per_window": RateLimitDefaults.AUTH_ATTEMPTS_PER_HOUR,
                "window_seconds": SecurityTimeouts.RATE_LIMIT_WINDOW_SECONDS,
                "block_duration": 3600  # 1 Stunde
            },
            RateLimitType.PER_USER: {
                "requests_per_window": RateLimitDefaults.DEFAULT_API_CALLS_PER_HOUR,
                "window_seconds": SecurityTimeouts.RATE_LIMIT_WINDOW_SECONDS,
                "block_duration": 1800  # 30 Minuten
            },
            RateLimitType.PER_TOKEN: {
                "requests_per_window": RateLimitDefaults.TOKEN_VALIDATION_PER_MINUTE * 60,
                "window_seconds": SecurityTimeouts.RATE_LIMIT_WINDOW_SECONDS,
                "block_duration": 900  # 15 Minuten
            },
            RateLimitType.PER_OPERATION: {
                "requests_per_window": RateLimitDefaults.AUTH_ATTEMPTS_PER_MINUTE * 60,
                "window_seconds": SecurityTimeouts.RATE_LIMIT_WINDOW_SECONDS,
                "block_duration": 600  # 10 Minuten
            }
        }

    def configure_limit(
        self,
        limit_type: RateLimitType,
        requests_per_window: int,
        window_seconds: int,
        block_duration: int = 3600
    ) -> None:
        """Konfiguriert Rate Limit.

        Args:
            limit_type: Typ des Rate Limits
            requests_per_window: Anzahl Requests pro Zeitfenster
            window_seconds: Zeitfenster in Sekunden
            block_duration: Blockierung-Dauer in Sekunden
        """
        self._limits[limit_type] = {
            "requests_per_window": requests_per_window,
            "window_seconds": window_seconds,
            "block_duration": block_duration
        }
        logger.info(f"Rate Limit konfiguriert: {limit_type} = {requests_per_window}/{window_seconds}s")

    @trace_function("rate_limiter.check_limit")
    async def check_limit(
        self,
        identifier: str,
        limit_type: RateLimitType,
        operation: str = "default"
    ) -> RateLimitResult:
        """Prüft Rate Limit für Identifier.

        Args:
            identifier: Eindeutiger Identifier (IP, User-ID, Token, etc.)
            limit_type: Typ des Rate Limits
            operation: Operation-Name für spezifische Limits

        Returns:
            Rate Limit-Ergebnis

        Raises:
            RateLimitExceeded: Wenn Rate Limit überschritten
        """
        # Cleanup alte Einträge
        await self._cleanup_if_needed()

        # Prüfe ob IP blockiert
        if limit_type == RateLimitType.PER_IP and identifier in self._blocked_ips:
            raise RateLimitExceeded(
                SecurityErrorMessages.RATE_LIMIT_BLOCKED,
                retry_after=3600
            )

        # Rate Limit-Konfiguration abrufen
        limit_config = self._limits.get(limit_type)
        if not limit_config:
            # Kein Limit konfiguriert - erlaube Request
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_time=datetime.now(UTC) + timedelta(hours=1),
                limit_type=limit_type,
                identifier=identifier
            )

        # Eindeutigen Schlüssel erstellen
        key = self._create_key(identifier, limit_type, operation)

        # Rate Limit-Info abrufen oder erstellen
        rate_info = self._rate_limits.get(key, RateLimitInfo())

        # Prüfe ob blockiert
        current_time = time.time()
        if rate_info.blocked_until and current_time < rate_info.blocked_until:
            retry_after = int(rate_info.blocked_until - current_time)
            raise RateLimitExceeded(
                SecurityErrorMessages.RATE_LIMIT_EXCEEDED,
                retry_after=retry_after
            )

        # Cleanup alte Requests
        rate_info.cleanup_old_requests(limit_config["window_seconds"])

        # Prüfe Rate Limit
        requests_in_window = len(rate_info.requests)
        max_requests = limit_config["requests_per_window"]

        if requests_in_window >= max_requests:
            # Rate Limit überschritten
            rate_info.blocked_until = current_time + limit_config["block_duration"]
            rate_info.blocked_count += 1

            # Bei wiederholten Überschreitungen IP blockieren
            if limit_type == RateLimitType.PER_IP and rate_info.blocked_count >= 3:
                self._blocked_ips.add(identifier)
                logger.warning(f"IP {identifier} dauerhaft blockiert nach {rate_info.blocked_count} Überschreitungen")

            self._rate_limits[key] = rate_info

            retry_after = int(rate_info.blocked_until - current_time)
            raise RateLimitExceeded(
                SecurityErrorMessages.RATE_LIMIT_EXCEEDED,
                retry_after=retry_after
            )

        # Request erlaubt - zu Rate Limit-Info hinzufügen
        rate_info.requests.append(current_time)
        rate_info.total_requests += 1
        self._rate_limits[key] = rate_info

        # Reset-Zeit berechnen
        oldest_request = min(rate_info.requests) if rate_info.requests else current_time
        reset_time = datetime.fromtimestamp(
            oldest_request + limit_config["window_seconds"],
            tz=UTC
        )

        return RateLimitResult(
            allowed=True,
            remaining=max_requests - len(rate_info.requests),
            reset_time=reset_time,
            limit_type=limit_type,
            identifier=identifier
        )

    def _create_key(self, identifier: str, limit_type: RateLimitType, operation: str) -> str:
        """Erstellt eindeutigen Schlüssel für Rate Limit.

        Args:
            identifier: Identifier
            limit_type: Rate Limit-Typ
            operation: Operation

        Returns:
            Eindeutiger Schlüssel
        """
        key_parts = [str(limit_type.value), identifier, operation]
        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def _cleanup_if_needed(self) -> None:
        """Führt Cleanup durch wenn nötig."""
        current_time = time.time()
        if current_time - self._last_cleanup > SecurityTimeouts.RATE_LIMIT_CLEANUP_INTERVAL:
            await self._cleanup_old_entries()
            self._last_cleanup = current_time

    async def _cleanup_old_entries(self) -> None:
        """Entfernt alte Rate Limit-Einträge."""
        current_time = time.time()
        keys_to_remove = []

        for key, rate_info in self._rate_limits.items():
            # Entferne Einträge die nicht mehr blockiert sind und keine aktuellen Requests haben
            if (not rate_info.blocked_until or current_time > rate_info.blocked_until) and \
               (not rate_info.requests or max(rate_info.requests) < current_time - 3600):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._rate_limits[key]

        if keys_to_remove:
            logger.debug(f"Rate Limiter Cleanup: {len(keys_to_remove)} alte Einträge entfernt")

    def unblock_ip(self, ip_address: str) -> bool:
        """Entsperrt IP-Adresse.

        Args:
            ip_address: IP-Adresse

        Returns:
            True wenn entsperrt
        """
        if ip_address in self._blocked_ips:
            self._blocked_ips.remove(ip_address)
            logger.info(f"IP {ip_address} entsperrt")
            return True
        return False

    def get_blocked_ips(self) -> set[str]:
        """Gibt blockierte IP-Adressen zurück."""
        return self._blocked_ips.copy()

    def get_rate_limit_stats(self) -> dict[str, Any]:
        """Gibt Rate Limit-Statistiken zurück."""
        total_entries = len(self._rate_limits)
        blocked_entries = sum(1 for info in self._rate_limits.values()
                             if info.blocked_until and time.time() < info.blocked_until)
        total_requests = sum(info.total_requests for info in self._rate_limits.values())
        total_blocks = sum(info.blocked_count for info in self._rate_limits.values())

        return {
            "total_entries": total_entries,
            "blocked_entries": blocked_entries,
            "blocked_ips": len(self._blocked_ips),
            "total_requests": total_requests,
            "total_blocks": total_blocks,
            "configured_limits": len(self._limits)
        }

    def clear_all_limits(self) -> None:
        """Löscht alle Rate Limits (für Tests)."""
        self._rate_limits.clear()
        self._blocked_ips.clear()
        logger.warning("Alle Rate Limits gelöscht")


__all__ = [
    "RateLimitExceeded",
    "RateLimitInfo",
    "RateLimitResult",
    "RateLimitType",
    "RateLimiter",
]
