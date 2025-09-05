"""Schema-Registry und JSON Schema Validierung für Webhook-Payloads.

Bietet eine zentrale Verwaltung versionierter JSON Schemas je `event_type`
und eine Validierungsfunktion, die fail‑fast bei Ungültigkeit ist.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .exceptions import WebhookValidationException

logger = get_logger(__name__)

try:  # pragma: no cover - optionale Abhängigkeit
    import jsonschema
    _JSONSCHEMA_AVAILABLE = True
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore
    _JSONSCHEMA_AVAILABLE = False


class SchemaRegistry:
    """Einfache In‑Memory Schema‑Registry mit Versionierung."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._bootstrap_defaults()

    def _bootstrap_defaults(self) -> None:
        """Registriert einige Standard‑Schemas für gängige Event‑Typen."""
        v1 = "1.0"
        self.register(
            event_type="task_completed",
            version=v1,
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["task_id", "status"],
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": ["success", "failed"]},
                    "result": {},
                },
                "additionalProperties": True,
            },
        )
        self.register(
            event_type="agent_observation",
            version=v1,
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["agent_id", "observed_at"],
                "properties": {
                    "agent_id": {"type": "string", "minLength": 1},
                    "observed_at": {"type": "string", "format": "date-time"},
                    "payload": {},
                },
                "additionalProperties": True,
            },
        )
        self.register(
            event_type="document_uploaded",
            version=v1,
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["document_id", "mime_type", "size_bytes"],
                "properties": {
                    "document_id": {"type": "string", "minLength": 1},
                    "mime_type": {"type": "string", "minLength": 1},
                    "size_bytes": {"type": "integer", "minimum": 0},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": True,
            },
        )

    def register(self, *, event_type: str, version: str, schema: dict[str, Any]) -> None:
        """Registriert/aktualisiert ein Schema für einen Event‑Typ und Version."""
        self._schemas.setdefault(event_type, {})[version] = schema

    def get_schema(self, *, event_type: str, version: str) -> dict[str, Any] | None:
        """Liefert Schema für Event‑Typ und Version oder None."""
        return self._schemas.get(event_type, {}).get(version)

    def validate(self, *, event_type: str, version: str, payload: dict[str, Any]) -> None:
        """Validiert Payload gegen das registrierte JSON Schema.

        Raises:
            WebhookValidationException: bei fehlendem Validator, Schema oder bei Verstoß.
        """
        if not _JSONSCHEMA_AVAILABLE:
            raise WebhookValidationException(
                message="JSON Schema Validator nicht verfügbar",
                error_code="schema_validator_missing",
                status_code=500,
                context={"event_type": event_type, "version": version},
            )
        schema = self.get_schema(event_type=event_type, version=version)
        if not schema:
            raise WebhookValidationException(
                message="Kein Schema registriert",
                error_code="schema_not_found",
                status_code=422,
                context={"event_type": event_type, "version": version},
            )
        try:
            jsonschema.validate(instance=payload, schema=schema)  # type: ignore[attr-defined]
        except Exception as exc:
            raise WebhookValidationException(
                message="Payload entspricht nicht dem JSON Schema",
                error_code="schema_validation_failed",
                status_code=422,
                context={"event_type": event_type, "version": version, "reason": str(exc)},
            ) from exc


_registry: SchemaRegistry | None = None


def get_schema_registry() -> SchemaRegistry:
    """Singleton‑Zugriff auf die Schema‑Registry."""
    global _registry
    if _registry is None:
        _registry = SchemaRegistry()
    return _registry


def validate_payload_against_schema(*, event_type: str, payload: dict[str, Any], version: str | None) -> None:
    """Hilfsfunktion: Validiert Payload gegen Schema für Event‑Typ/Version.

    Fehlt die Version, wird "1.0" als Default angenommen.
    """
    get_schema_registry().validate(event_type=event_type, version=version or "1.0", payload=payload)


__all__ = ["SchemaRegistry", "get_schema_registry", "validate_payload_against_schema"]
