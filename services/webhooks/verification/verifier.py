"""Inbound-Signature-Verifier mit verbesserter Architektur.

Koordiniert alle Verification-Komponenten für eingehende Webhooks
mit Dependency Injection und besserer Testbarkeit.
"""

from __future__ import annotations

import json
from typing import Any

from kei_logging import get_logger

from ..targets import TargetRegistry
from ..utils.redis_manager import get_redis_manager
from .config_manager import InboundConfigManager
from .replay_protector import ReplayProtector
from .signature_validator import InboundSignatureValidator

logger = get_logger(__name__)


class InboundSignatureVerifier:
    """Koordiniert alle Verification-Komponenten (Facade-Pattern)."""

    def __init__(
        self,
        *,
        signature_validator: InboundSignatureValidator | None = None,
        replay_protector: ReplayProtector | None = None,
        config_manager: InboundConfigManager | None = None,
        target_registry: TargetRegistry | None = None,
    ) -> None:
        """Initialisiert den Signature-Verifier.

        Args:
            signature_validator: Signature-Validator
            replay_protector: Replay-Protector
            config_manager: Config-Manager
            target_registry: Target-Registry
        """
        # Dependency Injection mit Standard-Implementierungen
        self.signature_validator = signature_validator or InboundSignatureValidator()
        self.replay_protector = replay_protector or ReplayProtector(get_redis_manager())
        self.config_manager = config_manager or InboundConfigManager()
        self.target_registry = target_registry or TargetRegistry()

    async def verify_signature(  # pylint: disable=too-many-arguments
        self,
        *,
        payload: bytes,
        signature: str,
        timestamp: str,
        target_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """Verifiziert Webhook-Signatur mit vollständiger Validation.

        Args:
            payload: Request-Body als Bytes
            signature: Webhook-Signatur
            timestamp: Timestamp-Header
            target_id: Target-ID für Secret-Lookup
            tenant_id: Optionale Tenant-ID

        Returns:
            True wenn Signatur gültig und kein Replay
        """
        config = self.config_manager.get_config()

        try:
            # 1. Target-Lookup
            target = await self.target_registry.get(target_id)
            if not target:
                logger.warning("Target not found for verification: %s", target_id)
                return False

            if not target.enabled:
                logger.debug("Target disabled for verification: %s", target_id)
                return False

            # 2. Payload-Size-Check
            if len(payload) > config.max_payload_size_bytes:
                logger.warning(
                    "Payload too large for verification: %d bytes (max: %d)",
                    len(payload), config.max_payload_size_bytes
                )
                return False

            # 3. Replay-Protection (wenn aktiviert)
            if config.replay_protection_enabled:
                is_new_request = await self.replay_protector.check_and_record_nonce(
                    payload=payload,
                    timestamp=timestamp,
                    target_id=target_id,
                    tenant_id=tenant_id,
                )
                if not is_new_request:
                    if config.log_validation_failures:
                        logger.warning(
                            "Replay attack detected for target %s (tenant: %s)",
                            target_id, tenant_id
                        )
                    return False

            # 4. Signature-Validation
            is_valid = await self.signature_validator.validate_signature(
                payload=payload,
                signature=signature,
                target=target,
            )

            if not is_valid and config.log_validation_failures:
                logger.warning(
                    "Signature validation failed for target %s (tenant: %s)",
                    target_id, tenant_id
                )

            return is_valid

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception(
                "Unexpected error during signature verification for target %s: %s",
                target_id, exc
            )
            # Fail-open oder fail-closed basierend auf Konfiguration
            return config.fail_open_on_redis_error

    async def validate(  # pylint: disable=too-many-arguments
        self,
        *,
        payload: bytes,
        signature: str,
        timestamp: str,
        target_id: str,
        tenant_id: str | None = None,
        event_type: str | None = None,
    ) -> bool:
        """Alias für verify_signature mit zusätzlichen Parametern.

        Args:
            payload: Request-Body als Bytes
            signature: Webhook-Signatur
            timestamp: Timestamp-Header
            target_id: Target-ID
            tenant_id: Optionale Tenant-ID
            event_type: Optionaler Event-Typ (für Logging)

        Returns:
            True wenn Validation erfolgreich
        """
        logger.debug(
            "Validating webhook signature (target: %s, tenant: %s, event_type: %s)",
            target_id, tenant_id, event_type
        )

        return await self.verify_signature(
            payload=payload,
            signature=signature,
            timestamp=timestamp,
            target_id=target_id,
            tenant_id=tenant_id,
        )

    def sanitize_payload_json(
        self,
        payload: bytes,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """Sanitized JSON-Payload für sichere Verarbeitung.

        Args:
            payload: Request-Payload
            event_type: Optionaler Event-Typ

        Returns:
            Sanitized JSON-Dictionary
        """
        config = self.config_manager.get_config()

        try:
            # Parse JSON
            data = json.loads(payload.decode("utf-8"))

            if not config.sanitize_payload:
                return data

            # Basis-Sanitization
            sanitized = self._sanitize_dict(data)

            # Event-Type-Override (falls konfiguriert)
            if event_type:
                sanitized["event_type"] = event_type

            return sanitized

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Failed to parse payload as JSON: %s", exc)
            return {"error": "invalid_json", "raw_payload_size": len(payload)}

    def _sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitized Dictionary rekursiv.

        Args:
            data: Zu sanitized Dictionary

        Returns:
            Sanitized Dictionary
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            # Sanitize Key
            clean_key = str(key)[:100]  # Begrenze Key-Länge

            # Sanitize Value
            if isinstance(value, dict):
                sanitized[clean_key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value[:100]  # Begrenze Array-Länge
                ]
            elif isinstance(value, str):
                sanitized[clean_key] = value[:1000]  # Begrenze String-Länge
            else:
                sanitized[clean_key] = value

        return sanitized

    def get_verification_status(self) -> dict[str, Any]:
        """Gibt Status aller Verification-Komponenten zurück.

        Returns:
            Status-Dictionary
        """
        config = self.config_manager.get_config()

        return {
            "config": self.config_manager.get_config_summary(),
            "components": {
                "signature_validator": {
                    "enabled": True,
                    "algorithm": "HMAC-SHA256",
                },
                "replay_protector": {
                    "enabled": config.replay_protection_enabled,
                    **self.replay_protector.get_protection_info(),
                },
                "target_registry": {
                    "enabled": True,
                    "registry_name": self.target_registry.registry_name,
                },
            },
            "validation_warnings": self.config_manager.validate_config(config),
        }


# =============================================================================
# Factory-Funktion für Standard-Setup
# =============================================================================

def create_inbound_signature_verifier(**kwargs) -> InboundSignatureVerifier:
    """Factory-Funktion für Standard-InboundSignatureVerifier-Setup.

    Args:
        **kwargs: Optionale Komponenten für Dependency Injection

    Returns:
        Konfigurierter InboundSignatureVerifier
    """
    return InboundSignatureVerifier(**kwargs)


__all__ = [
    "InboundSignatureVerifier",
    "create_inbound_signature_verifier",
]
