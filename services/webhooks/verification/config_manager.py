"""Configuration-Manager für Inbound-Webhook-Verification.

Verwaltet Konfiguration für Signature-Validation, Replay-Protection
und andere Verification-Parameter mit Settings-Integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import settings
from kei_logging import get_logger

from ..constants import (
    KEI_EVENT_TYPE_HEADER,
    KEI_SIGNATURE_HEADER,
    KEI_TIMESTAMP_HEADER,
    NONCE_TTL_SECONDS,
    SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS,
)

logger = get_logger(__name__)


@dataclass
class InboundVerificationConfig:  # pylint: disable=too-many-instance-attributes
    """Konfiguration für Inbound-Webhook-Verification."""

    # Signature-Validation
    signature_header: str = KEI_SIGNATURE_HEADER
    timestamp_header: str = KEI_TIMESTAMP_HEADER
    event_type_header: str = KEI_EVENT_TYPE_HEADER
    timestamp_tolerance_seconds: int = SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS

    # Replay-Protection
    replay_protection_enabled: bool = True
    nonce_ttl_seconds: int = NONCE_TTL_SECONDS

    # Payload-Sanitization
    sanitize_payload: bool = True
    max_payload_size_bytes: int = 1024 * 1024  # 1MB

    # Error-Handling
    fail_open_on_redis_error: bool = True
    log_validation_failures: bool = True


class InboundConfigManager:
    """Verwaltet Konfiguration für Inbound-Webhook-Verification."""

    def __init__(self) -> None:
        """Initialisiert den Config-Manager."""
        self._config: InboundVerificationConfig | None = None

    def get_config(self) -> InboundVerificationConfig:
        """Holt aktuelle Verification-Konfiguration.

        Returns:
            Verification-Konfiguration
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def reload_config(self) -> InboundVerificationConfig:
        """Lädt Konfiguration neu aus Settings.

        Returns:
            Neue Verification-Konfiguration
        """
        self._config = self._load_config()
        return self._config

    def _load_config(self) -> InboundVerificationConfig:
        """Lädt Konfiguration aus Settings mit Fallbacks.

        Returns:
            Verification-Konfiguration
        """
        try:
            config = InboundVerificationConfig()

            # Signature-Validation-Settings
            if hasattr(settings, "webhook_signature_tolerance_seconds"):
                config.timestamp_tolerance_seconds = max(
                    60,  # Minimum 1 Minute
                    int(settings.webhook_signature_tolerance_seconds)
                )

            if hasattr(settings, "webhook_signature_header"):
                config.signature_header = str(settings.webhook_signature_header)

            if hasattr(settings, "webhook_timestamp_header"):
                config.timestamp_header = str(settings.webhook_timestamp_header)

            # Replay-Protection-Settings
            if hasattr(settings, "webhook_replay_protection_enabled"):
                config.replay_protection_enabled = bool(
                    settings.webhook_replay_protection_enabled
                )

            if hasattr(settings, "webhook_nonce_ttl_seconds"):
                config.nonce_ttl_seconds = max(
                    300,  # Minimum 5 Minuten
                    int(settings.webhook_nonce_ttl_seconds)
                )

            # Payload-Settings
            if hasattr(settings, "webhook_sanitize_payload"):
                config.sanitize_payload = bool(settings.webhook_sanitize_payload)

            if hasattr(settings, "webhook_max_payload_size_bytes"):
                config.max_payload_size_bytes = max(
                    1024,  # Minimum 1KB
                    int(settings.webhook_max_payload_size_bytes)
                )

            # Error-Handling-Settings
            if hasattr(settings, "webhook_fail_open_on_redis_error"):
                config.fail_open_on_redis_error = bool(
                    settings.webhook_fail_open_on_redis_error
                )

            if hasattr(settings, "webhook_log_validation_failures"):
                config.log_validation_failures = bool(
                    settings.webhook_log_validation_failures
                )

            logger.debug("Loaded inbound verification config: %s", config)
            return config

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to load verification config from settings, using defaults: %s",
                exc
            )
            return InboundVerificationConfig()

    def validate_config(self, config: InboundVerificationConfig) -> list[str]:
        """Validiert Konfiguration und gibt Warnungen zurück.

        Args:
            config: Zu validierende Konfiguration

        Returns:
            Liste von Validierungswarnungen
        """
        warnings = []

        # Timestamp-Toleranz-Validierung
        if config.timestamp_tolerance_seconds < 60:
            warnings.append(
                f"Timestamp tolerance too low: {config.timestamp_tolerance_seconds}s "
                "(minimum recommended: 60s)"
            )

        if config.timestamp_tolerance_seconds > 3600:
            warnings.append(
                f"Timestamp tolerance very high: {config.timestamp_tolerance_seconds}s "
                "(maximum recommended: 3600s)"
            )

        # Nonce-TTL-Validierung
        if config.nonce_ttl_seconds < config.timestamp_tolerance_seconds * 2:
            warnings.append(
                f"Nonce TTL ({config.nonce_ttl_seconds}s) should be at least "
                f"2x timestamp tolerance ({config.timestamp_tolerance_seconds * 2}s)"
            )

        # Payload-Size-Validierung
        if config.max_payload_size_bytes > 10 * 1024 * 1024:  # 10MB
            warnings.append(
                f"Max payload size very large: {config.max_payload_size_bytes} bytes "
                "(consider security implications)"
            )

        # Header-Validierung
        required_headers = [
            config.signature_header,
            config.timestamp_header,
            config.event_type_header,
        ]

        for header in required_headers:
            if not header or not header.strip():
                warnings.append(f"Empty header configuration: {header}")

        return warnings

    def get_config_summary(self) -> dict[str, Any]:
        """Gibt Konfigurationszusammenfassung zurück.

        Returns:
            Dictionary mit Konfigurationsinformationen
        """
        config = self.get_config()

        return {
            "signature_validation": {
                "signature_header": config.signature_header,
                "timestamp_header": config.timestamp_header,
                "timestamp_tolerance_seconds": config.timestamp_tolerance_seconds,
            },
            "replay_protection": {
                "enabled": config.replay_protection_enabled,
                "nonce_ttl_seconds": config.nonce_ttl_seconds,
            },
            "payload_handling": {
                "sanitize_enabled": config.sanitize_payload,
                "max_size_bytes": config.max_payload_size_bytes,
            },
            "error_handling": {
                "fail_open_on_redis_error": config.fail_open_on_redis_error,
                "log_validation_failures": config.log_validation_failures,
            },
        }


__all__ = ["InboundConfigManager", "InboundVerificationConfig"]
