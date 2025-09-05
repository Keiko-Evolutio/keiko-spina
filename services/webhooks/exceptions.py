"""Spezifische Exception-Hierarchie für das KEI‑Webhook System.

Stellt typsichere Fehler inkl. strukturiertem Logging und HTTP‑Status-
Mapping bereit. Sensible Informationen werden aus Kontextdaten entfernt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class WebhookException(Exception):
    """Basisklasse für Webhook‑bezogene Fehler.

    Attributes:
        message: Menschlich lesbare Fehlermeldung
        error_code: Maschinell auswertbarer Fehlercode
        context: Nicht‑sensitive Zusatzinformationen
        correlation_id: Korrelations‑ID für Tracing
        status_code: HTTP‑Statuscode
    """

    message: str
    error_code: str
    context: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    status_code: int | None = None
    # Standard‑Statuscode (durch Subklassen überschreiben)
    default_status_code: int = 500

    def __post_init__(self) -> None:
        if self.status_code is None:
            self.status_code = type(self).default_status_code

    def __str__(self) -> str:
        """Kurzbeschreibung des Fehlers."""
        return f"{self.error_code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Gibt eine redaktierte, strukturierte Fehlerrepräsentation zurück."""
        data: dict[str, Any] = {
            "error": {"code": self.error_code, "message": self.message}
        }
        if self.context:
            data["error"]["details"] = self._redact(self.context)
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        return data

    def log(self, logger: logging.Logger, level: int = logging.ERROR) -> None:
        """Schreibt strukturierte Log‑Einträge ohne PII/Secrets."""
        payload: dict[str, Any] = {"error_code": self.error_code, "message": self.message}
        if self.correlation_id:
            payload["correlation_id"] = self.correlation_id
        if self.context:
            payload["context"] = self._redact(self.context)
        logger.log(level, f"WebhookException: {payload}")

    def _redact(self, data: dict[str, Any]) -> dict[str, Any]:
        """Entfernt sensible Daten aus Context-Dictionary."""
        redacted: dict[str, Any] = {}
        for k, v in data.items():
            redacted[k] = "***REDACTED***" if self._is_sensitive_key(k) else v
        return redacted

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        """Prüft, ob ein Key sensible Daten enthält."""
        sensitive_keys: set[str] = {
            "secret", "authorization", "api_key", "token", "password",
            "key", "auth", "credential", "private", "signature"
        }
        return key.lower() in sensitive_keys or any(
            sensitive in key.lower() for sensitive in sensitive_keys
        )


class WebhookValidationException(WebhookException):
    """Validierungsfehler (z. B. Timestamp/Schema ungültig)."""

    default_status_code = 422


class WebhookDeliveryException(WebhookException):
    """Transport-/HTTP‑Zustellfehler."""

    default_status_code = 502


class WebhookTargetException(WebhookException):
    """Fehler in Zielkonfiguration (z. B. nicht gefunden/disabled)."""

    default_status_code = 400


class WebhookTimeoutException(WebhookException):
    """Timeout während einer Operation (HTTP/Redis)."""

    default_status_code = 504


class WebhookAuthenticationException(WebhookException):
    """Authentifizierungs-/HMAC‑Fehler."""

    default_status_code = 401


class WebhookRateLimitException(WebhookException):
    """Rate‑Limit erreicht."""

    default_status_code = 429


class WebhookSubscriptionException(WebhookException):
    """Fehler im Subscription‑Subsystem (ungültig, nicht erlaubt)."""

    default_status_code = 400


def safe_execute_debug(
    fn: Callable[[], Any],
    *,
    logger: logging.Logger,
    error_code: str,
    context: dict[str, Any] | None = None
) -> Any | None:
    """Helfer zum Ausführen mit Debug‑Fallback ohne harte Fehler.

    Dient dazu, breite except‑Blöcke außerhalb der Kernlogik zu vermeiden.
    """
    try:
        return fn()
    except Exception as exc:  # pragma: no cover  # pylint: disable=broad-exception-caught
        payload = {"error_code": error_code, **(context or {})}
        logger.debug("Webhook safe_execute_debug: %s – %s", payload, exc)
        return None


# =============================================================================
# Exception-Factory für häufige Fehlertypen
# =============================================================================

class WebhookExceptionFactory:
    """Factory für häufig verwendete Webhook-Exceptions."""

    @staticmethod
    def invalid_signature(correlation_id: str | None = None) -> WebhookAuthenticationException:
        """Erstellt Exception für ungültige Signatur."""
        return WebhookAuthenticationException(
            message="Webhook-Signatur ist ungültig oder fehlt",
            error_code="invalid_signature",
            correlation_id=correlation_id,
        )

    @staticmethod
    def replay_attack(correlation_id: str | None = None) -> WebhookAuthenticationException:
        """Erstellt Exception für Replay-Angriff."""
        return WebhookAuthenticationException(
            message="Webhook-Request wurde bereits verarbeitet (Replay-Angriff)",
            error_code="replay_attack",
            correlation_id=correlation_id,
        )

    @staticmethod
    def target_not_found(
        target_id: str, correlation_id: str | None = None
    ) -> WebhookTargetException:
        """Erstellt Exception für nicht gefundenes Target."""
        return WebhookTargetException(
            message=f"Webhook-Target '{target_id}' wurde nicht gefunden",
            error_code="target_not_found",
            context={"target_id": target_id},
            correlation_id=correlation_id,
        )

    @staticmethod
    def delivery_timeout(url: str, correlation_id: str | None = None) -> WebhookTimeoutException:
        """Erstellt Exception für Delivery-Timeout."""
        return WebhookTimeoutException(
            message="Webhook-Zustellung zeitüberschritten",
            error_code="delivery_timeout",
            context={"url": url},
            correlation_id=correlation_id,
        )

    @staticmethod
    def rate_limit_exceeded(
        limit: int,
        window_seconds: int,
        correlation_id: str | None = None
    ) -> WebhookRateLimitException:
        """Erstellt Exception für Rate-Limit-Überschreitung."""
        return WebhookRateLimitException(
            message=f"Rate-Limit überschritten: {limit} Requests pro {window_seconds}s",
            error_code="rate_limit_exceeded",
            context={"limit": limit, "window_seconds": window_seconds},
            correlation_id=correlation_id,
        )

    @staticmethod
    def payload_too_large(size_bytes: int, max_bytes: int) -> WebhookValidationException:
        """Erstellt Exception für zu große Payloads."""
        return WebhookValidationException(
            message=f"Payload zu groß: {size_bytes} Bytes (Maximum: {max_bytes} Bytes)",
            error_code="payload_too_large",
            context={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )

    @staticmethod
    def circuit_breaker_open(target_id: str) -> WebhookDeliveryException:
        """Erstellt Exception für offenen Circuit-Breaker."""
        return WebhookDeliveryException(
            message=f"Circuit-Breaker für Target '{target_id}' ist geöffnet",
            error_code="circuit_breaker_open",
            context={"target_id": target_id},
        )


__all__ = [
    "WebhookAuthenticationException",
    "WebhookDeliveryException",
    "WebhookException",
    "WebhookExceptionFactory",
    "WebhookRateLimitException",
    "WebhookSubscriptionException",
    "WebhookTargetException",
    "WebhookTimeoutException",
    "WebhookValidationException",
    "safe_execute_debug",
]
