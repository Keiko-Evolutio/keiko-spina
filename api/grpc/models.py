"""KEI-RPC Datenmodelle für standardisierte Agent-Operationen.

Implementiert typisierte Request/Response-Modelle für plan/act/observe/explain
Operationen mit W3C Trace-Propagation und Idempotenz-Unterstützung.

Implementiert verbesserte Type Hints und eliminierte Magic Numbers.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# ENUMS UND KONSTANTEN
# ============================================================================


class OperationType(str, Enum):
    """KEI-RPC Operation-Typen."""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    EXPLAIN = "explain"


class OperationStatus(str, Enum):
    """Status von RPC-Operationen."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PriorityLevel(str, Enum):
    """Prioritätsstufen für RPC-Operationen."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# BASIS-MODELLE
# ============================================================================


class TraceContext(BaseModel):
    """W3C Trace Context für RPC-Operationen."""

    traceparent: str | None = Field(None, description="W3C Traceparent Header")
    tracestate: str | None = Field(None, description="W3C Tracestate Header")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Korrelations-ID"
    )
    causation_id: str | None = Field(None, description="Verursachende Operation-ID")

    @field_validator("traceparent")
    def validate_traceparent(cls, v: str | None) -> str | None:
        """Validiert W3C Traceparent Format."""
        if v is None:
            return v

        # Vereinfachte W3C Traceparent Validierung
        parts = v.split("-")
        if len(parts) != 4:
            raise ValueError("Ungültiges Traceparent-Format")

        return v


class OperationMetadata(BaseModel):
    """Metadaten für RPC-Operationen."""

    operation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Eindeutige Operation-ID"
    )
    idempotency_key: str | None = Field(None, description="Idempotenz-Schlüssel")
    priority: PriorityLevel = Field(default=PriorityLevel.NORMAL, description="Prioritätsstufe")
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="Timeout in Sekunden")
    retry_count: int = Field(default=0, ge=0, le=5, description="Anzahl Wiederholungen")
    tags: dict[str, str] = Field(default_factory=dict, description="Zusätzliche Tags")

    @field_validator("idempotency_key")
    def validate_idempotency_key(cls, v: str | None) -> str | None:
        """Validiert Idempotenz-Schlüssel Format."""
        if v is None:
            return v

        if len(v) < 1 or len(v) > 255:
            raise ValueError("Idempotenz-Schlüssel muss zwischen 1 und 255 Zeichen lang sein")

        return v


class AgentContext(BaseModel):
    """Kontext-Informationen für Agent-Operationen."""

    agent_id: str | None = Field(None, description="Spezifische Agent-ID")
    required_capabilities: list[str] = Field(
        default_factory=list, description="Erforderliche Capabilities"
    )
    preferred_framework: str | None = Field(None, description="Bevorzugtes Framework")
    execution_context: dict[str, Any] = Field(
        default_factory=dict, description="Ausführungskontext"
    )
    constraints: dict[str, Any] = Field(default_factory=dict, description="Ausführungs-Constraints")


# ============================================================================
# REQUEST-MODELLE
# ============================================================================


class BaseRPCRequest(BaseModel):
    """Basis-Request-Modell für alle RPC-Operationen."""

    trace_context: TraceContext = Field(default_factory=TraceContext, description="Trace-Kontext")
    metadata: OperationMetadata = Field(
        default_factory=OperationMetadata, description="Operation-Metadaten"
    )
    agent_context: AgentContext = Field(default_factory=AgentContext, description="Agent-Kontext")


class PlanRequest(BaseRPCRequest):
    """Request-Modell für Plan-Operation."""

    objective: str = Field(..., min_length=1, max_length=10000, description="Zielbeschreibung")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Planungs-Constraints")
    resources: dict[str, Any] = Field(default_factory=dict, description="Verfügbare Ressourcen")
    success_criteria: list[str] = Field(default_factory=list, description="Erfolgskriterien")

    @field_validator("objective")
    def validate_objective(cls, v: str) -> str:
        """Validiert Zielbeschreibung."""
        if not v.strip():
            raise ValueError("Zielbeschreibung darf nicht leer sein")
        return v.strip()


class ActRequest(BaseRPCRequest):
    """Request-Modell für Act-Operation."""

    action: str = Field(..., min_length=1, max_length=5000, description="Auszuführende Aktion")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Aktions-Parameter")
    plan_reference: str | None = Field(None, description="Referenz auf zugehörigen Plan")
    step_number: int | None = Field(None, ge=1, description="Schritt-Nummer im Plan")

    @field_validator("action")
    def validate_action(cls, v: str) -> str:
        """Validiert Aktions-Beschreibung."""
        if not v.strip():
            raise ValueError("Aktions-Beschreibung darf nicht leer sein")
        return v.strip()


class ObserveRequest(BaseRPCRequest):
    """Request-Modell für Observe-Operation."""

    observation_target: str = Field(..., min_length=1, description="Beobachtungsziel")
    observation_type: str = Field(default="general", description="Art der Beobachtung")
    filters: dict[str, Any] = Field(default_factory=dict, description="Beobachtungs-Filter")
    include_history: bool = Field(default=False, description="Historische Daten einschließen")

    @field_validator("observation_target")
    def validate_observation_target(cls, v: str) -> str:
        """Validiert Beobachtungsziel."""
        if not v.strip():
            raise ValueError("Beobachtungsziel darf nicht leer sein")
        return v.strip()


class ExplainRequest(BaseRPCRequest):
    """Request-Modell für Explain-Operation."""

    subject: str = Field(..., min_length=1, description="Zu erklärendes Subjekt")
    explanation_type: str = Field(default="detailed", description="Art der Erklärung")
    audience: str = Field(default="general", description="Zielgruppe")
    context_references: list[str] = Field(default_factory=list, description="Kontext-Referenzen")
    detail_level: str = Field(default="medium", description="Detailgrad")

    @field_validator("subject")
    def validate_subject(cls, v: str) -> str:
        """Validiert Erklärungs-Subjekt."""
        if not v.strip():
            raise ValueError("Erklärungs-Subjekt darf nicht leer sein")
        return v.strip()

    @field_validator("detail_level")
    def validate_detail_level(cls, v: str) -> str:
        """Validiert Detailgrad."""
        valid_levels = ["low", "medium", "high", "expert"]
        if v not in valid_levels:
            raise ValueError(f"Detailgrad muss einer von {valid_levels} sein")
        return v


# ============================================================================
# RESPONSE-MODELLE
# ============================================================================


class OperationTiming(BaseModel):
    """Timing-Informationen für Operationen."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(None, description="Abschluss-Zeitpunkt")
    duration_ms: int | None = Field(None, ge=0, description="Dauer in Millisekunden")

    def mark_completed(self) -> None:
        """Markiert Operation als abgeschlossen."""
        self.completed_at = datetime.now(UTC)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)


class OperationError(BaseModel):
    """Fehler-Informationen für Operationen."""

    error_code: str = Field(..., description="Fehlercode")
    error_message: str = Field(..., description="Fehlermeldung")
    error_type: str = Field(..., description="Fehlertyp")
    details: dict[str, Any] = Field(default_factory=dict, description="Fehler-Details")
    retry_after: int | None = Field(None, description="Retry-After in Sekunden")


class BaseRPCResponse(BaseModel):
    """Basis-Response-Modell für alle RPC-Operationen."""

    operation_type: OperationType = Field(..., description="Operation-Typ")
    operation_id: str = Field(..., description="Operation-ID")
    correlation_id: str = Field(..., description="Korrelations-ID")
    status: OperationStatus = Field(..., description="Operation-Status")

    # Timing
    timing: OperationTiming = Field(
        default_factory=OperationTiming, description="Timing-Informationen"
    )

    # Agent-Informationen
    agent_id: str | None = Field(None, description="Ausführender Agent")
    agent_framework: str | None = Field(None, description="Verwendetes Framework")

    # Fehler-Informationen
    error: OperationError | None = Field(None, description="Fehler-Informationen")

    # Metadaten
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response-Metadaten")


class PlanResponse(BaseRPCResponse):
    """Response-Modell für Plan-Operation."""

    plan: dict[str, Any] | None = Field(None, description="Generierter Plan")
    steps: list[dict[str, Any]] = Field(default_factory=list, description="Plan-Schritte")
    estimated_duration: int | None = Field(None, description="Geschätzte Dauer in Sekunden")
    confidence_score: float | None = Field(None, ge=0.0, le=1.0, description="Konfidenz-Score")


class ActResponse(BaseRPCResponse):
    """Response-Modell für Act-Operation."""

    action_result: dict[str, Any] | None = Field(None, description="Aktions-Ergebnis")
    side_effects: list[dict[str, Any]] = Field(default_factory=list, description="Seiteneffekte")
    next_actions: list[str] = Field(default_factory=list, description="Nächste empfohlene Aktionen")
    completion_status: str = Field(default="partial", description="Vollständigkeits-Status")


class ObserveResponse(BaseRPCResponse):
    """Response-Modell für Observe-Operation."""

    observations: list[dict[str, Any]] = Field(default_factory=list, description="Beobachtungen")
    summary: str | None = Field(None, description="Zusammenfassung")
    anomalies: list[dict[str, Any]] = Field(default_factory=list, description="Erkannte Anomalien")
    confidence_level: float | None = Field(None, ge=0.0, le=1.0, description="Konfidenz-Level")


class ExplainResponse(BaseRPCResponse):
    """Response-Modell für Explain-Operation."""

    explanation: str | None = Field(None, description="Generierte Erklärung")
    key_points: list[str] = Field(default_factory=list, description="Wichtige Punkte")
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list, description="Belege")
    clarity_score: float | None = Field(None, ge=0.0, le=1.0, description="Klarheits-Score")
    references: list[str] = Field(default_factory=list, description="Referenzen")


# ============================================================================
# UTILITY-MODELLE
# ============================================================================


class OperationSummary(BaseModel):
    """Zusammenfassung einer Operation für Monitoring."""

    operation_id: str
    operation_type: OperationType
    status: OperationStatus
    agent_id: str | None
    duration_ms: int | None
    error_code: str | None
    created_at: datetime
