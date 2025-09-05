"""Generator für AsyncAPI 3.0 Spezifikation.

Erzeugt eine konsolidierte AsyncAPI‑Spezifikation aus Projektartefakten:
- KEI‑Bus Subject‑Patterns
- Webhook Inbound/Outbound Kanäle
- Gemeinsame Nachrichtenschemata (BusEnvelope, WebhookEvent)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from kei_logging import get_logger
from services.messaging.config import bus_settings
from services.webhooks.models import (
    DeliveryRecord,
    DeliveryStatus,
    WebhookEvent,
    WebhookEventMeta,
    WebhookTarget,
)

from .constants import SpecConstants

logger = get_logger(__name__)


class AsyncAPIServer(BaseModel):
    """Serverdefinition für AsyncAPI."""

    host: str
    protocol: str
    description: str | None = None


def _servers_from_settings() -> dict[str, dict[str, Any]]:
    """Leitet Server aus Bus‑Settings und Defaults ab."""
    servers: dict[str, dict[str, Any]] = {}
    try:
        if bus_settings.provider == "nats":
            for idx, s in enumerate(bus_settings.servers):
                servers[f"nats-{idx}"] = {"host": s.replace("nats://", ""), "protocol": "nats"}
        if bus_settings.provider == "kafka":
            for idx, s in enumerate(bus_settings.kafka_bootstrap_servers):
                servers[f"kafka-{idx}"] = {"host": s, "protocol": "kafka"}
    except Exception:
        # Fallback: lokale Dev‑Server
        servers = {
            "dev-nats": {"host": SpecConstants.NATS_DEV_HOST, "protocol": "nats"},
            "dev-kafka": {"host": SpecConstants.KAFKA_DEV_HOST, "protocol": "kafka"},
        }
    return servers


def _bus_envelope_schema() -> dict[str, Any]:
    """Schema für `BusEnvelope` (vereinfachte Darstellung)."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"},
            "subject": {"type": "string"},
            "ts": {"type": "string", "format": "date-time"},
            "corr_id": {"type": "string", "nullable": True},
            "causation_id": {"type": "string", "nullable": True},
            "tenant": {"type": "string", "nullable": True},
            "key": {"type": "string", "nullable": True},
            "headers": {"type": "object", "additionalProperties": True},
            "traceparent": {"type": "string", "nullable": True},
            "schema_ref": {"type": "string", "nullable": True},
            "payload": {"type": "object", "additionalProperties": True},
        },
        "required": ["id", "type", "subject", "ts", "payload"],
    }


def _model_schema(model: Any) -> dict[str, Any]:
    """Erzeugt JSON‑Schema aus einem Pydantic‑Modell."""
    try:
        return model.model_json_schema()  # type: ignore[attr-defined]
    except Exception:
        return {"type": "object"}


def _webhook_schemas() -> dict[str, Any]:
    """Sammelt Schemas für Webhook‑Modelle (automatisch aus Pydantic)."""
    return {
        "WebhookEvent": _model_schema(WebhookEvent),
        "WebhookEventMeta": _model_schema(WebhookEventMeta),
        "WebhookTarget": _model_schema(WebhookTarget),
        "DeliveryRecord": _model_schema(DeliveryRecord),
        "DeliveryStatus": {
            "type": "string",
            "enum": [s.value for s in DeliveryStatus],
            "description": "Zustellstatus",
        },
    }


def build_asyncapi_dict() -> dict[str, Any]:
    """Erzeugt eine AsyncAPI 3.0 Spezifikation als Dictionary.

    Returns:
        AsyncAPI Dokument als verschachteltes Dictionary
    """
    servers = _servers_from_settings()

    channels: dict[str, Any] = {
        # Generische KEI‑Bus Kanäle
        "kei.events.*": {"address": "kei.events.*", "messages": {"DomainEvent": {"$ref": "#/components/messages/BusMessage"}}},
        "kei.rpc.*": {"address": "kei.rpc.*", "messages": {"RpcMessage": {"$ref": "#/components/messages/BusMessage"}}},
        "kei.tasks.*": {"address": "kei.tasks.*", "messages": {"TaskMessage": {"$ref": "#/components/messages/BusMessage"}}},
        "kei.a2a.*": {"address": "kei.a2a.*", "messages": {"AgentToAgent": {"$ref": "#/components/messages/BusMessage"}}},
        # Webhook‑Inbound auf Bus publiziert (vereinheitlicht)
        "kei.webhook.inbound.*.v1": {
            "address": "kei.webhook.inbound.*.v1",
            "messages": {"WebhookInbound": {"name": "WebhookInbound", "contentType": "application/json", "payload": {"$ref": "#/components/schemas/WebhookEvent"}}},
            "traits": [{"security": [{"hmacHeader": []}, {"mtls": []}]}],
        },
        # Outbound Bestätigungen/Tracking (optional)
        "kei.webhook.outbound.status.*": {
            "address": "kei.webhook.outbound.status.*",
            "messages": {
                "DeliveryRecord": {
                    "name": "DeliveryRecord",
                    "contentType": "application/json",
                    "payload": {"$ref": "#/components/schemas/DeliveryRecord"},
                }
            },
        },
        # DLQ/ Parking Subjects
        "kei.dlq.>": {"address": "kei.dlq.>", "messages": {"DeadLetter": {"$ref": "#/components/messages/BusMessage"}}},
        "kei.parking.*": {"address": "kei.parking.*", "messages": {"Parked": {"$ref": "#/components/messages/BusMessage"}}},
    }

    spec: dict[str, Any] = {
        "asyncapi": SpecConstants.ASYNCAPI_VERSION,
        "info": {
            "title": "Keiko AsyncAPI",
            "version": SpecConstants.API_VERSION,
            "description": "Asynchrone Kommunikationsverträge für KEI‑Bus und Webhooks",
        },
        "defaultContentType": "application/json",
        "servers": dict(servers.items()),
        "channels": channels,
        "components": {
            "securitySchemes": {
                "mtls": {"type": "clientCertificate", "description": "mTLS Client‑Zertifikate"},
                # HMAC Prüfsumme in Header `x-kei-signature`
                "hmacHeader": {"type": "httpApiKey", "name": "x-kei-signature", "in": "header", "description": "HMAC‑Signatur des Payloads"},
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
            },
            "messages": {
                "BusMessage": {
                    "name": "BusMessage",
                    "title": "KEI‑Bus Envelope",
                    "contentType": "application/json",
                    "payload": _bus_envelope_schema(),
                }
            },
            "schemas": _webhook_schemas(),
        },
        "tags": [
            {"name": "bus", "description": "KEI‑Bus Kernnachrichten"},
            {"name": "webhook", "description": "Webhook Inbound/Outbound"},
            {"name": "security", "description": "Sicherheit mTLS/JWT"},
        ],
        "security": [{"bearerAuth": []}],
    }
    return spec


__all__ = ["build_asyncapi_dict"]
