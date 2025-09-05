"""Webhook Datenmodelle für KEI-Webhook Service.

Enthält typsichere Pydantic-Modelle für Inbound/Outbound Event‑Payloads
und Delivery‑Status.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class WebhookEventMeta(BaseModel):
    """Metadaten für Webhook‑Events.

    Beschreibt Korrelation, Quelle und Schema‑Version.
    """

    correlation_id: str | None = Field(default=None, description="Korrelations-ID")
    source: str | None = Field(default="keiko", description="Quellsystem")
    schema_version: str = Field(default="1.0", description="Schema-Version")
    tenant: str | None = Field(default=None, description="Tenant/Namespace")
    replay: bool = Field(default=False, description="Kennzeichnet Replay-Events")


class WebhookEvent(BaseModel):
    """Standardisiertes Webhook‑Event.

    Attributes:
        id: Eindeutige Event‑ID
        event_type: Typsicherer Event‑Name
        occurred_at: Zeitpunkt des Auftretens in UTC
        data: Nutzdaten
        meta: Metadaten inkl. Korrelation und Schema
    """

    id: str = Field(..., description="Eindeutige Event-ID")
    event_type: str = Field(..., description="Event-Typ")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Zeitpunkt UTC"
    )
    data: dict[str, Any] = Field(default_factory=dict, description="Event-Daten")
    meta: WebhookEventMeta = Field(default_factory=WebhookEventMeta, description="Metadaten")
    # Optionales Top‑Level Feld für Tenant (dupliziert meta.tenant für einfache Queries)
    tenant_id: str | None = Field(default=None, description="Mandanten‑ID für Isolation")


class DeliveryStatus(str, Enum):
    """Lieferstatus für Outbound Webhooks."""

    pending = "pending"
    retrying = "retrying"
    success = "success"
    failed = "failed"
    dlq = "dlq"


class DeliveryRecord(BaseModel):
    """Lieferdatensatz für Outbound Zustellungen.

    Beinhaltet Retry‑Zähler, Zeitpunkte und letzten Fehler.
    """

    delivery_id: str
    target_id: str
    event_id: str
    tenant_id: str | None = Field(default=None, description="Mandanten‑ID für Isolation")
    correlation_id: str | None = None
    trace_id: str | None = None
    status: DeliveryStatus = Field(default=DeliveryStatus.pending)
    attempt: int = 0
    max_attempts: int = 5
    backoff_seconds: float = 1.0
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None


class TargetTransform(BaseModel):
    """Einfache Transformationsregeln für Payloads."""

    include_fields: list[str] | None = None
    rename_map: dict[str, str] | None = None
    # Erweiterte Regeln
    exclude_fields: list[str] | None = None
    add_fields: dict[str, Any] | None = None
    drop_nulls: bool = False


class WebhookTarget(BaseModel):
    """Zielkonfiguration für Outbound Webhooks.

    Unterstützt optionale mTLS‑Parameter (PEM) und HMAC‑Secrets über Azure Key Vault.
    Für Rückwärtskompatibilität kann ein altes `secret` eingelesen werden, wird jedoch
    nicht mehr persistiert.
    """

    # Pydantic Modellkonfiguration für Aliase/Serialisierung
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }

    id: str
    url: str
    # Legacy: nur zum Einlesen alter Konfigurationen (alias "secret"), wird nicht persistiert
    legacy_secret: str | None = Field(default=None, alias="secret")
    # Key Vault Integration
    secret_key_name: str | None = None
    secret_version: str | None = None
    previous_secret_version: str | None = None
    secret_last_rotated_at: datetime | None = None
    secret_grace_until: datetime | None = None
    enabled: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    transform: TargetTransform | None = None
    mtls_cert_pem: str | None = None
    mtls_key_pem: str | None = None
    max_attempts: int = 5
    backoff_seconds: float = 1.0
    # Circuit Breaker Konfiguration (per Target überschreibbar)
    cb_use_consecutive_failures: bool = Field(default=False, description="Verwendet Schwellen‑Modus (aufeinanderfolgende Fehler)")
    cb_failure_threshold: int = Field(default=5, description="Fehler bis OPEN im Schwellen‑Modus")
    cb_recovery_timeout_seconds: float = Field(default=60.0, description="Wartezeit bis HALF_OPEN im Schwellen‑Modus")
    cb_success_threshold: int = Field(default=3, description="Erfolge in HALF_OPEN bis CLOSED im Schwellen‑Modus")
    # Health Monitoring
    last_health_check: datetime | None = Field(default=None, description="Zeitpunkt des letzten Health‑Checks")
    health_status: str | None = Field(default=None, description="Gesundheitsstatus des Targets (healthy/unhealthy/unknown)")
    # Subscriptions für Topic/Event‑Filter
    subscriptions: list[Subscription] = Field(default_factory=list)
    # Mandant
    tenant_id: str | None = Field(default=None, description="Mandanten‑ID des Targets")


class TopicFilter(BaseModel):
    """Filtert Topics/Events mithilfe von Pattern ("*" und "**")."""

    pattern: str


class EventSubscription(BaseModel):
    """Exakte Subscription für Event‑Typen."""

    event_type: str


class Subscription(BaseModel):
    """Subscription‑Definition je Target.

    Eine Subscription matcht, wenn entweder ein TopicFilter passt oder
    ein EventSubscription exakt übereinstimmt. Mehrere Subscriptions pro
    Target sind erlaubt (OR‑Verknüpfung).
    """

    id: str
    enabled: bool = True
    topics: list[TopicFilter] = Field(default_factory=list)
    events: list[EventSubscription] = Field(default_factory=list)


def topic_matches(pattern: str, topic: str) -> bool:
    r"""Prüft, ob ein Topic ein Pattern mit \"*\"/\"**\" erfüllt.

    Regeln:
    - "*" matcht genau ein Segment
    - "**" matcht beliebig viele Segmente (auch 0)
    - Segmente werden mit '.' getrennt
    """
    p_parts = pattern.split(".") if pattern else []
    t_parts = topic.split(".") if topic else []

    def match(i: int, j: int) -> bool:
        if i == len(p_parts) and j == len(t_parts):
            return True
        if i == len(p_parts):
            return False
        token = p_parts[i]
        if token == "**":
            # '**' kann 0..n Segmente verbrauchen
            # versuche greedy und non‑greedy
            return match(i + 1, j) or (j < len(t_parts) and match(i, j + 1))
        if j == len(t_parts):
            return False
        if token == "*" or token == t_parts[j]:
            return match(i + 1, j + 1)
        return False

    return match(0, 0)


__all__ = [
    "DeliveryRecord",
    "DeliveryStatus",
    "EventSubscription",
    "Subscription",
    "TargetTransform",
    "TopicFilter",
    "WebhookEvent",
    "WebhookEventMeta",
    "WebhookTarget",
    "topic_matches",
]


class WorkerPoolConfig(BaseModel):
    """Konfiguration des Worker‑Pools für das Webhook‑System.

    Diese Klasse beschreibt die Anzahl und Benennung der Shards, Zeitlimits
    für Warteschlangen‑Operationen sowie Supervisions‑Einstellungen. Zur
    Wahrung der Rückwärtskompatibilität werden die bisherigen Felder
    `worker_count`, `queue_name` und `poll_interval` weiterhin unterstützt
    und gegen die neuen Felder abgebildet.

    Attributes:
        pool_size: Anzahl der Worker im Pool (min 1, max 32)
        shard_names: Liste der Queue‑Shard‑Namen (Default: ["default"]) –
            wird automatisch aus `queue_name` und `pool_size` generiert,
            sofern nicht explizit gesetzt.
        queue_timeout_seconds: Timeout in Sekunden für Queue‑Operationen
            beim Shutdown (Default: 30.0)
        supervision_enabled: Aktiviert automatischen Neustart bei Fehlern
            (Default: True)
        max_restart_attempts: Maximale Anzahl automatischer Neustarts je
            Worker (Default: 3)
        queue_name: Basis‑Queue‑Name zur Shard‑Generierung (Kompatibilität)
        poll_interval: Poll‑Intervall in Sekunden (Kompatibilität)
        worker_count: Alter Feldname für `pool_size` (Kompatibilität)
    """

    # Neue Felder
    pool_size: int = Field(
        default=4,
        description="Anzahl der Worker (1..32)",
    )
    shard_names: list[str] = Field(
        default_factory=lambda: ["default"],
        description="Liste der Worker‑Shard‑Namen",
    )
    queue_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout für Queue‑Operationen beim Shutdown (Sekunden)",
    )
    supervision_enabled: bool = Field(
        default=True,
        description="Aktiviert automatische Worker‑Supervision/Neustart",
    )
    max_restart_attempts: int = Field(
        default=3,
        description="Maximale Zahl automatischer Neustarts je Worker",
    )

    # Kompatibilitätsfelder (beibehalten für bestehende Aufrufer)
    queue_name: str = Field(default="default", description="Basis‑Queue‑Name")
    poll_interval: float = Field(default=1.0, description="Poll‑Intervall in Sekunden")
    worker_count: int | None = Field(default=None, description="Kompatibilitätsfeld zu pool_size")

    # Pydantic v2: Validierer
    @classmethod
    def _generate_default_shards(cls, *, base: str, size: int) -> list[str]:
        """Erzeugt deterministische Shard‑Namen aus Basisname und Poolgröße."""
        if size <= 1:
            return [base]
        return [f"{base}:{i}" for i in range(size)]

    @classmethod
    def _normalize_shards(cls, shard_names: list[str]) -> list[str]:
        """Normalisiert Shard‑Namen (trimmen) und entfernt Duplikate, Reihenfolge bleibt stabil."""
        seen = set()
        normalized: list[str] = []
        for name in shard_names:
            nm = (name or "").strip()
            if not nm or nm in seen:
                continue
            seen.add(nm)
            normalized.append(nm)
        return normalized or ["default"]

    @staticmethod
    def _coalesce_int(value: int | None, fallback: int) -> int:
        return int(value) if value is not None else int(fallback)

    @staticmethod
    def _coalesce_float(value: float | None, fallback: float) -> float:
        return float(value) if value is not None else float(fallback)

    @model_validator(mode="before")
    def _map_legacy_and_defaults(cls, data: Any) -> Any:
        """Mappt Legacy‑Felder und ergänzt sinnvolle Defaults vor Feldprüfung."""
        if not isinstance(data, dict):
            return data
        mapped = dict(data)
        if mapped.get("pool_size") is None and mapped.get("worker_count") is not None:
            with contextlib.suppress(Exception):
                mapped["pool_size"] = int(mapped.get("worker_count"))
        if not mapped.get("shard_names"):
            # Nur generieren, wenn Größe EXPLIZIT vorgegeben wurde (pool_size/worker_count im Input)
            explicit_size = None
            if "pool_size" in mapped and mapped.get("pool_size") is not None:
                explicit_size = int(mapped.get("pool_size"))
            elif "worker_count" in mapped and mapped.get("worker_count") is not None:
                explicit_size = int(mapped.get("worker_count"))
            if explicit_size is not None:
                base = (mapped.get("queue_name") or "default")
                mapped["shard_names"] = cls._generate_default_shards(base=base, size=int(explicit_size))
        return mapped

    @property
    def shard_names_property(self) -> list[str]:
        """Kompatibilitäts‑Property, falls Legacy‑Code als Property zugreift."""
        return list(self.shard_names)

    # Einzelfeld‑Validierungen
    @field_validator("pool_size")
    def _validate_pool_size(cls, v: int) -> int:
        """Validiert die Poolgröße (1..32)."""
        if not isinstance(v, int):
            raise TypeError("pool_size muss int sein")
        if v < 1 or v > 32:
            raise ValueError("pool_size muss zwischen 1 und 32 liegen")
        return v

    @field_validator("queue_timeout_seconds")
    def _validate_timeout(cls, v: float) -> float:
        """Validiert das Queue‑Timeout (positiv, max. 600s)."""
        if v <= 0:
            raise ValueError("queue_timeout_seconds muss > 0 sein")
        if v > 600:
            raise ValueError("queue_timeout_seconds zu groß (max 600)")
        return v

    @field_validator("max_restart_attempts")
    def _validate_max_restarts(cls, v: int) -> int:
        """Validiert die maximale Anzahl an Neustarts (>= 0)."""
        if v < 0:
            raise ValueError("max_restart_attempts darf nicht negativ sein")
        return v

    @field_validator("poll_interval")
    def _validate_poll_interval(cls, v: float) -> float:
        """Validiert das Poll‑Intervall (>= 0.05s)."""
        if v < 0.05:
            raise ValueError("poll_interval muss >= 0.05 sein")
        return v

    @field_validator("shard_names")
    def _validate_shards(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Validiert Shard‑Namen: nicht leer, eindeutige Einträge, Anzahl passend zu pool_size."""
        normalized = cls._normalize_shards(v or ["default"])
        # Zugriff auf pool_size aus bereits geparsten Daten (sofern vorhanden)
        pool_size = None
        try:
            pool_size = int((info.data or {}).get("pool_size"))  # type: ignore[union-attr]
        except Exception:
            pool_size = None
        # Prüfe ob pool_size gültig ist und Shard-Anzahl angepasst werden muss
        is_valid_pool_size = isinstance(pool_size, int) and pool_size > 1
        needs_shard_adjustment = is_valid_pool_size and len(normalized) != pool_size

        if needs_shard_adjustment:
            # Erzeuge konsistente Liste, wenn Anzahl nicht passt
            base = "default"
            try:
                base = (info.data or {}).get("queue_name", "default")  # type: ignore[union-attr]
            except Exception:
                base = "default"
            return cls._generate_default_shards(base=base, size=int(pool_size))
        return normalized

    # Nach‑Validierung zur Konsistenz und Rückwärtskompatibilität
    def model_post_init(self, __context: Any) -> None:
        """Stellt Konsistenz zwischen neuen und alten Feldern her."""
        # worker_count stets mit pool_size spiegeln
        object.__setattr__(self, "worker_count", int(self.pool_size))
        # Sicherstellen, dass shard_names konsistent sind
        if not self.shard_names:
            object.__setattr__(self, "shard_names", self._generate_default_shards(base=self.queue_name, size=self.pool_size))


__all__ += ["WorkerPoolConfig"]
