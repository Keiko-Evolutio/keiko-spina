"""Webhook-Signature-Generator für HMAC-basierte Authentifizierung.

Generiert sichere HMAC-Signaturen für Webhook-Payloads mit automatischer
Secret-Auflösung aus Key Vault oder Legacy-Secrets.
"""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import TYPE_CHECKING

from kei_logging import get_logger

from ..constants import KEI_EVENT_TYPE_HEADER, KEI_SIGNATURE_HEADER, KEI_TIMESTAMP_HEADER
from ..exceptions import WebhookExceptionFactory
from ..secret_manager import get_secret_manager

if TYPE_CHECKING:
    from ..models import WebhookTarget

logger = get_logger(__name__)


class WebhookSignatureGenerator:
    """Generiert HMAC-Signaturen für Webhook-Payloads."""

    def __init__(self) -> None:
        """Initialisiert den Signature-Generator."""
        self._secret_manager = get_secret_manager()

    async def generate_signature(
        self,
        target: WebhookTarget,
        payload: bytes,
    ) -> str:
        """Generiert HMAC-SHA256-Signatur für Webhook-Payload.

        Args:
            target: Webhook-Target mit Secret-Konfiguration
            payload: Payload-Bytes für Signatur-Berechnung

        Returns:
            Hex-kodierte HMAC-Signatur

        Raises:
            WebhookDeliveryException: Wenn kein Secret verfügbar ist
        """
        secret_value = await self._resolve_secret(target)
        if not secret_value:
            raise WebhookExceptionFactory.invalid_signature()

        # HMAC-SHA256 Signatur generieren
        return hmac.new(
            secret_value.encode("utf-8"),
            payload,
            sha256
        ).hexdigest()


    async def create_headers(
        self,
        target: WebhookTarget,
        payload: bytes,
        timestamp: int,
        event_type: str,
    ) -> dict[str, str]:
        """Erstellt vollständige Header für Webhook-Request.

        Args:
            target: Webhook-Target
            payload: Payload-Bytes
            timestamp: Unix-Timestamp
            event_type: Event-Typ

        Returns:
            Dictionary mit allen erforderlichen Headern
        """
        signature = await self.generate_signature(target, payload)

        headers = {
            "content-type": "application/json",
            KEI_SIGNATURE_HEADER: signature,
            KEI_TIMESTAMP_HEADER: str(timestamp),
            KEI_EVENT_TYPE_HEADER: event_type,
        }

        # Target-spezifische Header hinzufügen
        if target.headers:
            headers.update(target.headers)

        return headers

    async def verify_signature(
        self,
        target: WebhookTarget,
        payload: bytes,
        provided_signature: str,
    ) -> bool:
        """Verifiziert eine bereitgestellte Signatur.

        Args:
            target: Webhook-Target mit Secret-Konfiguration
            payload: Payload-Bytes
            provided_signature: Zu verifizierende Signatur

        Returns:
            True wenn Signatur gültig, False sonst
        """
        try:
            expected_signature = await self.generate_signature(target, payload)
            return hmac.compare_digest(expected_signature, provided_signature)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Signature verification failed: %s", exc)
            return False

    async def _resolve_secret(self, target: WebhookTarget) -> str | None:
        """Löst Secret für Target auf (Key Vault bevorzugt, Fallback auf Legacy).

        Args:
            target: Webhook-Target mit Secret-Konfiguration

        Returns:
            Secret-Wert oder None wenn nicht verfügbar
        """
        # Versuche Key Vault Secret zuerst
        if target.secret_key_name:
            try:
                if target.secret_version:
                    secret_value = await self._secret_manager.get_secret_by_version(
                        key_name=target.secret_key_name,
                        version=target.secret_version
                    )
                else:
                    secret_value, _ = await self._secret_manager.get_current_secret(
                        key_name=target.secret_key_name
                    )

                if secret_value:
                    return secret_value

            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug(
                    "Key Vault secret resolution failed for target %s: %s",
                    target.id, exc
                )

        # Fallback auf Legacy Secret
        if target.legacy_secret:
            logger.debug(
                "Using legacy secret for target %s (Key Vault unavailable)",
                target.id
            )
            return target.legacy_secret

        # Kein Secret verfügbar
        logger.warning(
            "No secret available for target %s (key_name=%s, has_legacy=%s)",
            target.id,
            target.secret_key_name,
            bool(target.legacy_secret)
        )
        return None

    def get_signature_info(self, target: WebhookTarget) -> dict[str, str]:
        """Gibt Informationen über die Signatur-Konfiguration zurück.

        Args:
            target: Webhook-Target

        Returns:
            Dictionary mit Signatur-Informationen
        """
        return {
            "algorithm": "HMAC-SHA256",
            "header": KEI_SIGNATURE_HEADER,
            "secret_source": (
                "key_vault" if target.secret_key_name
                else "legacy" if target.legacy_secret
                else "none"
            ),
            "secret_key_name": target.secret_key_name or "",
            "secret_version": target.secret_version or "",
            "has_legacy_secret": bool(target.legacy_secret),
        }


__all__ = ["WebhookSignatureGenerator"]
