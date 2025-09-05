"""Replay-Protector für Webhook-Verification.

Schützt vor Replay-Angriffen durch Redis-basiertes Nonce-Tracking
mit konfigurierbaren TTLs und Cleanup-Strategien.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from ..constants import (
    NONCE_TTL_SECONDS,
    SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS,
    get_redis_key,
    get_tenant_normalized,
)

if TYPE_CHECKING:
    from ..utils.redis_manager import RedisManager

logger = get_logger(__name__)


class ReplayProtector:
    """Schützt vor Replay-Angriffen mit Redis-basiertem Nonce-Tracking."""

    def __init__(self, redis_manager: RedisManager) -> None:
        """Initialisiert den Replay-Protector.

        Args:
            redis_manager: Redis-Manager für Nonce-Storage
        """
        self.redis_manager = redis_manager

    async def check_and_record_nonce(
        self,
        payload: bytes,
        timestamp: str,
        target_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """Prüft Nonce und zeichnet ihn auf wenn neu.

        Args:
            payload: Request-Payload
            timestamp: Timestamp-Header
            target_id: Target-ID
            tenant_id: Optional Tenant-ID

        Returns:
            True wenn Nonce neu (kein Replay), False wenn bereits verwendet
        """
        # Validiere Timestamp zuerst
        if not self._is_timestamp_valid(timestamp):
            logger.debug(
                "Invalid timestamp for replay protection: %s (target: %s)",
                timestamp, target_id
            )
            return False

        # Generiere Nonce aus Payload + Timestamp + Target
        nonce = self._generate_nonce(payload, timestamp, target_id)

        # Prüfe und setze Nonce atomisch
        return await self._check_and_set_nonce(nonce, tenant_id)

    def _is_timestamp_valid(self, timestamp: str) -> bool:
        """Validiert Timestamp gegen Toleranz-Fenster.

        Args:
            timestamp: Unix-Timestamp als String

        Returns:
            True wenn Timestamp im gültigen Fenster
        """
        try:
            request_time = int(timestamp)
            current_time = int(datetime.now(UTC).timestamp())

            # Prüfe Toleranz-Fenster
            time_diff = abs(current_time - request_time)
            return time_diff <= SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS

        except (ValueError, TypeError):
            return False

    def _generate_nonce(
        self,
        payload: bytes,
        timestamp: str,
        target_id: str,
    ) -> str:
        """Generiert deterministischen Nonce aus Request-Daten.

        Args:
            payload: Request-Payload
            timestamp: Timestamp-Header
            target_id: Target-ID

        Returns:
            Hex-kodierter SHA256-Hash als Nonce
        """
        # Kombiniere alle relevanten Request-Daten
        nonce_data = b"".join([
            payload,
            timestamp.encode("utf-8"),
            target_id.encode("utf-8"),
        ])

        # Generiere SHA256-Hash
        return hashlib.sha256(nonce_data).hexdigest()

    async def _check_and_set_nonce(
        self,
        nonce: str,
        tenant_id: str | None,
    ) -> bool:
        """Prüft Nonce und setzt ihn atomisch wenn neu.

        Args:
            nonce: Zu prüfender Nonce
            tenant_id: Optional Tenant-ID

        Returns:
            True wenn Nonce neu gesetzt wurde, False wenn bereits existiert
        """
        try:
            # Redis-Key für Nonce
            nonce_key = self._get_nonce_key(nonce, tenant_id)

            # Atomische SET NX Operation mit TTL
            success = await self.redis_manager.safe_set(
                key=nonce_key,
                value="1",
                ex=NONCE_TTL_SECONDS,
                nx=True  # Nur setzen wenn nicht existiert
            )

            if not success:
                logger.debug("Replay attack detected: nonce already exists: %s", nonce[:16])

            return success

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to check/set nonce %s: %s", nonce[:16], exc)
            # Bei Redis-Fehlern erlauben wir den Request (fail-open)
            return True

    def _get_nonce_key(self, nonce: str, tenant_id: str | None) -> str:
        """Generiert Redis-Key für Nonce-Storage.

        Args:
            nonce: Nonce-Wert
            tenant_id: Optional Tenant-ID

        Returns:
            Redis-Key für Nonce
        """
        tenant = get_tenant_normalized(tenant_id)
        return get_redis_key("kei:webhook:nonce", tenant, nonce)

    async def cleanup_expired_nonces(self, tenant_id: str | None = None) -> int:
        """Bereinigt abgelaufene Nonces (optional, Redis TTL macht das automatisch).

        Args:
            tenant_id: Optional Tenant-ID für spezifische Bereinigung

        Returns:
            Anzahl bereinigter Nonces
        """
        # Redis TTL bereinigt automatisch, aber wir können manuell bereinigen
        # für bessere Kontrolle
        try:
            # Für jetzt verlassen wir uns auf Redis TTL
            # Könnte erweitert werden für explizite Cleanup-Operationen
            logger.debug("Nonce cleanup requested for tenant: %s", tenant_id)
            return 0

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Nonce cleanup failed: %s", exc)
            return 0

    def get_protection_info(self, tenant_id: str | None = None) -> dict[str, Any]:
        """Gibt Informationen über Replay-Protection zurück.

        Args:
            tenant_id: Optional Tenant-ID

        Returns:
            Dictionary mit Protection-Informationen
        """
        return {
            "enabled": True,
            "nonce_ttl_seconds": NONCE_TTL_SECONDS,
            "timestamp_tolerance_seconds": SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS,
            "tenant_id": tenant_id,
            "algorithm": "SHA256",
            "storage": "redis",
        }


__all__ = ["ReplayProtector"]
