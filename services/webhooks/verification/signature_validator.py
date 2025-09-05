"""Inbound-Signature-Validator für Webhook-Verification.

Validiert HMAC-Signaturen von eingehenden Webhooks mit Multi-Version-Support
und automatischer Secret-Auflösung.
"""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from ..constants import KEI_SIGNATURE_HEADER, KEI_TIMESTAMP_HEADER
from ..secret_manager import get_secret_manager

if TYPE_CHECKING:
    from ..models import WebhookTarget

logger = get_logger(__name__)


class InboundSignatureValidator:
    """Validiert HMAC-Signaturen von eingehenden Webhooks."""

    def __init__(self) -> None:
        """Initialisiert den Signature-Validator."""
        self._secret_manager = get_secret_manager()

    async def validate_signature(
        self,
        payload: bytes,
        signature: str,
        target: WebhookTarget,
    ) -> bool:
        """Validiert HMAC-Signatur gegen Target-Secrets.

        Args:
            payload: Request-Payload als Bytes
            signature: Bereitgestellte Signatur
            target: Webhook-Target mit Secret-Konfiguration

        Returns:
            True wenn Signatur gültig, False sonst
        """
        # Hole alle verfügbaren Secrets für das Target
        secrets = await self._get_target_secrets(target)
        if not secrets:
            logger.warning(
                "No secrets available for signature validation (target: %s)",
                target.id
            )
            return False

        # Teste Signatur gegen alle verfügbaren Secrets
        for secret in secrets:
            if self._verify_signature_with_secret(payload, signature, secret):
                return True

        logger.debug(
            "Signature validation failed for target %s (tested %d secrets)",
            target.id, len(secrets)
        )
        return False

    async def _get_target_secrets(self, target: WebhookTarget) -> list[str]:
        """Sammelt alle verfügbaren Secrets für ein Target.

        Args:
            target: Webhook-Target

        Returns:
            Liste verfügbarer Secret-Werte
        """
        secrets = []

        # Key Vault Secrets (aktuell und vorherige Versionen)
        if target.secret_key_name:
            vault_secrets = await self._get_key_vault_secrets(target)
            secrets.extend(vault_secrets)

        # Legacy Secret als Fallback
        if target.legacy_secret:
            secrets.append(target.legacy_secret)

        return secrets

    async def _get_key_vault_secrets(self, target: WebhookTarget) -> list[str]:
        """Holt Key Vault Secrets (aktuell und vorherige Versionen).

        Args:
            target: Webhook-Target

        Returns:
            Liste der Key Vault Secret-Werte
        """
        secrets = []

        try:
            # Aktuelles Secret
            if target.secret_version:
                # Spezifische Version
                secret_value = await self._secret_manager.get_secret_by_version(
                    key_name=target.secret_key_name,
                    version=target.secret_version
                )
                if secret_value:
                    secrets.append(secret_value)
            else:
                # Aktuelle Version
                secret_value, _ = await self._secret_manager.get_current_secret(
                    key_name=target.secret_key_name
                )
                if secret_value:
                    secrets.append(secret_value)

            # Vorherige Versionen für Rotation-Support
            previous_secrets = await self._get_previous_secret_versions(target)
            secrets.extend(previous_secrets)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug(
                "Failed to retrieve Key Vault secrets for target %s: %s",
                target.id, exc
            )

        return secrets

    async def _get_previous_secret_versions(self, target: WebhookTarget) -> list[str]:
        """Holt vorherige Secret-Versionen für Rotation-Support.

        Args:
            target: Webhook-Target

        Returns:
            Liste vorheriger Secret-Versionen
        """
        try:
            # Hole die letzten 3 Versionen für Rotation-Support
            # NOTE: list_secret_versions might not be available in all SecretManager implementations
            if hasattr(self._secret_manager, "list_secret_versions"):
                versions = await self._secret_manager.list_secret_versions(
                    key_name=target.secret_key_name,
                    max_results=3
                )
            else:
                return []

            secrets = []
            for version in versions:
                try:
                    secret_value = await self._secret_manager.get_secret_by_version(
                        key_name=target.secret_key_name,
                        version=version
                    )
                    if secret_value and secret_value not in secrets:
                        secrets.append(secret_value)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.debug(
                        "Failed to get secret version %s for target %s: %s",
                        version, target.id, exc
                    )

            return secrets

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug(
                "Failed to list secret versions for target %s: %s",
                target.id, exc
            )
            return []

    def _verify_signature_with_secret(
        self,
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verifiziert Signatur mit einem spezifischen Secret.

        Args:
            payload: Request-Payload
            signature: Zu verifizierende Signatur
            secret: Secret für HMAC-Berechnung

        Returns:
            True wenn Signatur gültig
        """
        try:
            # Berechne erwartete Signatur
            expected_signature = hmac.new(
                secret.encode("utf-8"),
                payload,
                sha256
            ).hexdigest()

            # Sichere Vergleich mit hmac.compare_digest
            return hmac.compare_digest(expected_signature, signature)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Signature verification failed: %s", exc)
            return False

    def get_validation_info(self, target: WebhookTarget) -> dict[str, Any]:
        """Gibt Informationen über die Validierungs-Konfiguration zurück.

        Args:
            target: Webhook-Target

        Returns:
            Dictionary mit Validierungs-Informationen
        """
        return {
            "algorithm": "HMAC-SHA256",
            "signature_header": KEI_SIGNATURE_HEADER,
            "timestamp_header": KEI_TIMESTAMP_HEADER,
            "secret_sources": {
                "key_vault": {
                    "enabled": bool(target.secret_key_name),
                    "key_name": target.secret_key_name or "",
                    "version": target.secret_version or "latest",
                },
                "legacy": {
                    "enabled": bool(target.legacy_secret),
                },
            },
            "multi_version_support": bool(target.secret_key_name),
        }


__all__ = ["InboundSignatureValidator"]
