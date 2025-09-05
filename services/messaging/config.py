"""Bus-Konfigurationen.

Definiert global nutzbare Einstellungen für den KEI-Bus inkl. Security- und
QoS-Parametern.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class BusSecurityConfig(BaseModel):
    """Sicherheitskonfiguration für Bus-Verbindungen.

    - Unterstützt mTLS, OAuth2/OIDC (JWT in Headers)
    - Optional: Field-Level-Encryption per KMS (nicht in diesem Schritt)
    """

    enable_mtls: bool = Field(default=False)
    ca_cert_path: str | None = Field(default=None)
    client_cert_path: str | None = Field(default=None)
    client_key_path: str | None = Field(default=None)
    enable_oidc: bool = Field(default=False)
    oidc_jwt_required: bool = Field(default=False)
    require_jwt_for_publish: bool = Field(default=True)
    require_jwt_for_consume: bool = Field(default=True)
    # Bus-Scopes (zusätzlich zu subject-spezifischen Scopes)
    require_bus_base_scopes: bool = Field(default=True)
    require_tenant_scope: bool = Field(default=True)
    publish_scope_name: str = Field(default="kei.bus.publish")
    consume_scope_name: str = Field(default="kei.bus.consume")
    tenant_scope_prefix: str = Field(default="kei.bus.tenant.")
    # NATS Credentials (optional)
    nats_username: str | None = Field(default=None)
    nats_password: str | None = Field(default=None)
    nkey_seed: str | None = Field(default=None, description="NATS NKey Seed für NKey-Auth")


class BusQoSConfig(BaseModel):
    """QoS-Parameter für Zustellung und Retry-Strategien."""

    at_least_once: bool = Field(default=True)
    exactly_once: bool = Field(default=False)
    max_redeliveries: int = Field(default=5)
    retry_backoff_initial_ms: int = Field(default=200)
    retry_backoff_max_ms: int = Field(default=5000)
    retry_backoff_jitter_ms: int = Field(default=200)


class BusFlowControlConfig(BaseModel):
    """Flow-Control Konfiguration für Consumer."""

    use_pull_subscriber: bool = Field(default=False, description="Pull statt Push Subscription verwenden")
    pull_batch_size: int = Field(default=20, ge=1, le=1000)
    max_in_flight: int = Field(default=100, ge=1, le=10000)
    ack_wait_ms: int = Field(default=30000, ge=1000)
    max_ack_pending: int = Field(default=1000, ge=1)
    slow_consumer_timeout_ms: int = Field(default=2000, ge=100)


class BusRPCConfig(BaseModel):
    """Konfiguration für Request/Reply über den Bus."""

    default_timeout_seconds: float = Field(default=5.0, ge=0.1)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_ms: int = Field(default=200, ge=0)
    retry_backoff_max_ms: int = Field(default=2000, ge=0)
    retry_jitter_ms: int = Field(default=200, ge=0)


def get_bool_env(key: str, default: bool) -> bool:
    """Hilfsfunktion für Boolean-Environment-Variablen."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


class BusSettings(BaseModel):
    """Globale Bus-Einstellungen."""

    enabled: bool = Field(default_factory=lambda: get_bool_env("KEI_BUS_ENABLED", False))  # Environment-konfigurierbar
    provider: str = Field(default_factory=lambda: os.getenv("KEI_BUS_PROVIDER", "nats"))  # nats|kafka (kafka später)
    servers: list[str] = Field(default_factory=lambda: os.getenv("KEI_BUS_SERVERS", "nats://localhost:4222").split(","))
    namespace_prefix: str = Field(default="kei")
    security: BusSecurityConfig = Field(default_factory=BusSecurityConfig)
    qos: BusQoSConfig = Field(default_factory=BusQoSConfig)
    enable_tracing: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    schema_registry_url: str | None = Field(default=None)

    # Datenschutz / Privacy
    redact_payload_before_send: bool = Field(default=False, description="Maskiert sensible Felder vor dem Versenden")
    redact_fields: list[str] = Field(default_factory=lambda: [
        "password",
        "access_token",
        "id_token",
        "refresh_token",
        "authorization",
        "email",
        "phone",
    ])
    enable_field_encryption: bool = Field(default=False, description="Aktiviert Feld-Verschlüsselung auf Payload-Ebene")
    encryption_fields: list[str] = Field(default_factory=list, description="Feldnamen/JSON-Pfade zur Verschlüsselung")
    kms_key_id: str = Field(default="local-default", description="Aktiver KMS-Schlüssel-Name")
    kms_rotation_days: int = Field(default=30, description="Rotation in Tagen (nur für lokalen KMS-Provider)")
    # ACL Konfiguration: Subject-Pattern -> erforderliche Scopes je Aktion
    required_scopes_by_subject: dict[str, dict[str, list[str]]] = Field(
        default_factory=lambda: {
            "kei.events.>": {"publish": ["events:write"], "consume": ["events:read"]},
            "kei.agents.>": {"publish": ["agents:write"], "consume": ["agents:read"]},
            "kei.tasks.>": {"publish": ["tasks:enqueue"], "consume": ["tasks:dequeue"]},
            "kei.rpc.>": {"publish": ["rpc:invoke"], "consume": ["rpc:serve"]},
        }
    )
    flow: BusFlowControlConfig = Field(default_factory=BusFlowControlConfig)

    # Kafka Einstellungen
    kafka_bootstrap_servers: list[str] = Field(default_factory=lambda: ["localhost:9092"])
    kafka_transactional_id_prefix: str = Field(default="kei-bus-tx")
    kafka_group_id_prefix: str = Field(default="kei-bus-group")
    rpc: BusRPCConfig = Field(default_factory=BusRPCConfig)
    # Replay Store Limit
    replay_store_limit_per_subject: int = Field(default=1000, ge=10, le=100000)
    # Versionierung & Lifecycle
    envelope_version: str = Field(default="1.0.0")
    sdk_versions: dict[str, str] = Field(default_factory=lambda: {
        "python": "1.0.0",
        "typescript": "1.0.0",
        "go": "1.0.0",
    })
    capabilities: dict[str, bool] = Field(default_factory=lambda: {
        "rpc": True,
        "sagas": True,
        "dlq": True,
        "replay": True,
        "chaos": True,
        "encryption": True,
        "oidc": True,
    })
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    # Deprecations: subject pattern -> metadata
    deprecations: dict[str, dict[str, Any]] = Field(default_factory=dict)


# Globale Instanz mit Defaults; kann später aus `config.settings` übersteuert werden
bus_settings = BusSettings()
