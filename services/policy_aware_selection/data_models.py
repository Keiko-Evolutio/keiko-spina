# backend/services/policy_aware_selection/data_models.py
"""Datenmodelle für Policy-aware Agent Selection.

Definiert alle Datenstrukturen für Policy-Enforcement,
Compliance-Checks und Multi-Tenant Agent-Selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from policy_engine.compliance_framework import ComplianceStandard


class PolicyType(Enum):
    """Policy-Typen für Agent-Selection."""

    SECURITY = "security"                    # Security-Policies
    COMPLIANCE = "compliance"                # Compliance-Anforderungen
    CAPABILITY = "capability"                # Capability-Constraints
    TENANT = "tenant"                       # Tenant-spezifische Policies
    GEOGRAPHIC = "geographic"               # Geografische Beschränkungen
    DATA_CLASSIFICATION = "data_classification"  # Datenklassifizierungs-Policies
    PERFORMANCE = "performance"             # Performance-Anforderungen
    COST = "cost"                          # Kosten-Constraints


class PolicyEffect(Enum):
    """Policy-Effekte."""

    ALLOW = "allow"                        # Explizit erlauben
    DENY = "deny"                          # Explizit verbieten
    REQUIRE = "require"                    # Erforderlich
    PREFER = "prefer"                      # Bevorzugen
    AVOID = "avoid"                        # Vermeiden


class PolicyPriority(Enum):
    """Policy-Prioritäten."""

    CRITICAL = "critical"                  # Kritisch (höchste Priorität)
    HIGH = "high"                         # Hoch
    MEDIUM = "medium"                     # Mittel
    LOW = "low"                          # Niedrig


class SecurityLevel(Enum):
    """Security-Level für Agents."""

    PUBLIC = "public"                     # Öffentlich zugänglich
    INTERNAL = "internal"                 # Intern
    CONFIDENTIAL = "confidential"         # Vertraulich
    RESTRICTED = "restricted"             # Eingeschränkt
    TOP_SECRET = "top_secret"            # Streng geheim


class DataClassification(Enum):
    """Datenklassifizierung."""

    PUBLIC = "public"                     # Öffentliche Daten
    INTERNAL = "internal"                 # Interne Daten
    CONFIDENTIAL = "confidential"         # Vertrauliche Daten
    RESTRICTED = "restricted"             # Eingeschränkte Daten
    PII = "pii"                          # Personenbezogene Daten
    PHI = "phi"                          # Gesundheitsdaten
    FINANCIAL = "financial"               # Finanzdaten


class ComplianceStatus(Enum):
    """Compliance-Status."""

    COMPLIANT = "compliant"               # Konform
    NON_COMPLIANT = "non_compliant"       # Nicht konform
    PENDING = "pending"                   # Prüfung ausstehend
    UNKNOWN = "unknown"                   # Unbekannt
    EXEMPT = "exempt"                     # Ausgenommen


@dataclass
class PolicyConstraint:
    """Policy-Constraint für Agent-Selection."""

    # Constraint-Identifikation
    constraint_id: str
    name: str
    description: str
    policy_type: PolicyType

    # Constraint-Definition
    effect: PolicyEffect
    priority: PolicyPriority

    # Constraint-Bedingungen
    conditions: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)
    forbidden_capabilities: list[str] = field(default_factory=list)

    # Security-Anforderungen
    min_security_level: SecurityLevel | None = None
    max_security_level: SecurityLevel | None = None
    required_clearances: list[str] = field(default_factory=list)

    # Compliance-Anforderungen
    required_compliance_standards: list[ComplianceStandard] = field(default_factory=list)
    data_classification_constraints: list[DataClassification] = field(default_factory=list)

    # Geografische Constraints
    allowed_regions: list[str] = field(default_factory=list)
    forbidden_regions: list[str] = field(default_factory=list)

    # Tenant-Constraints
    tenant_restrictions: dict[str, Any] = field(default_factory=dict)

    # Performance-Constraints
    max_response_time_ms: float | None = None
    min_success_rate: float | None = None
    max_error_rate: float | None = None

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str | None = None
    is_active: bool = True


@dataclass
class AgentPolicy:
    """Policy für Agent-Selection."""

    # Policy-Identifikation
    policy_id: str
    name: str
    description: str
    version: str = "1.0.0"

    # Policy-Scope
    tenant_id: str | None = None
    user_groups: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)

    # Policy-Constraints
    constraints: list[PolicyConstraint] = field(default_factory=list)

    # Policy-Metadaten
    priority: PolicyPriority = PolicyPriority.MEDIUM
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str | None = None

    # Policy-Gültigkeit
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Audit-Informationen
    audit_required: bool = True
    compliance_frameworks: list[ComplianceStandard] = field(default_factory=list)


@dataclass
class AgentSelectionContext:
    """Kontext für Policy-aware Agent-Selection."""

    # Request-Kontext
    request_id: str
    orchestration_id: str
    subtask_id: str
    task_type: str

    # User-Kontext
    user_id: str | None = None
    tenant_id: str | None = None
    user_groups: list[str] = field(default_factory=list)
    user_clearances: list[str] = field(default_factory=list)

    # Task-Kontext
    task_payload: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)

    # Data-Kontext
    data_classification: DataClassification | None = None
    contains_pii: bool = False
    contains_phi: bool = False
    geographic_restrictions: list[str] = field(default_factory=list)

    # Security-Kontext
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    compliance_requirements: list[ComplianceStandard] = field(default_factory=list)

    # Performance-Kontext
    max_execution_time_ms: float | None = None
    priority_level: str = "normal"

    # Metadaten
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str | None = None


@dataclass
class PolicyViolation:
    """Policy-Verletzung bei Agent-Selection."""

    # Violation-Identifikation
    violation_id: str
    policy_id: str
    constraint_id: str

    # Violation-Details
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str

    # Betroffene Entitäten
    agent_id: str
    context: AgentSelectionContext

    # Violation-Daten
    expected_value: Any
    actual_value: Any

    # Remediation
    remediation_suggestions: list[str] = field(default_factory=list)
    can_be_waived: bool = False
    waiver_reason: str | None = None

    # Metadaten
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    resolution_action: str | None = None


@dataclass
class ComplianceResult:
    """Ergebnis einer Compliance-Prüfung."""

    # Compliance-Status
    is_compliant: bool
    compliance_score: float  # 0.0 - 1.0

    # Prüfungs-Details
    checked_policies: list[str]
    passed_constraints: list[str]
    failed_constraints: list[str]

    # Violations
    violations: list[PolicyViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Performance-Metriken
    check_duration_ms: float = 0.0
    policies_evaluated: int = 0

    # Metadaten
    check_timestamp: datetime = field(default_factory=datetime.utcnow)
    checker_version: str = "1.0.0"


@dataclass
class AgentComplianceProfile:
    """Compliance-Profil eines Agents."""

    # Agent-Identifikation
    agent_id: str
    agent_type: str

    # Security-Profil
    security_level: SecurityLevel
    clearances: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)

    # Compliance-Status
    compliance_standards: dict[ComplianceStandard, ComplianceStatus] = field(default_factory=dict)
    last_compliance_check: datetime | None = None
    compliance_expiry: datetime | None = None

    # Capabilities und Constraints
    verified_capabilities: list[str] = field(default_factory=list)
    capability_constraints: dict[str, Any] = field(default_factory=dict)

    # Geografische Informationen
    deployment_region: str | None = None
    allowed_regions: list[str] = field(default_factory=list)
    data_residency_compliant: bool = True

    # Performance-Profil
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0

    # Audit-Informationen
    audit_trail_enabled: bool = True
    last_audit: datetime | None = None

    # Metadaten
    profile_created: datetime = field(default_factory=datetime.utcnow)
    profile_updated: datetime = field(default_factory=datetime.utcnow)
    profile_version: str = "1.0.0"


@dataclass
class PolicyEvaluationResult:
    """Ergebnis einer Policy-Evaluation."""

    # Evaluation-Status
    decision: PolicyEffect  # ALLOW, DENY, etc.
    confidence: float  # 0.0 - 1.0

    # Evaluation-Details
    evaluated_policies: list[str]
    matched_constraints: list[str]
    violated_constraints: list[str]

    # Agent-Ranking
    agent_scores: dict[str, float] = field(default_factory=dict)  # agent_id -> score
    recommended_agents: list[str] = field(default_factory=list)
    excluded_agents: list[str] = field(default_factory=list)

    # Compliance-Informationen
    compliance_results: list[ComplianceResult] = field(default_factory=list)

    # Performance-Metriken
    evaluation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    # Metadaten
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)
    evaluator_version: str = "1.0.0"
    context: AgentSelectionContext | None = None


@dataclass
class AuditEvent:
    """Audit-Event für Policy-aware Agent-Selection."""

    # Event-Identifikation
    event_id: str
    event_type: str  # "policy_check", "agent_selected", "violation_detected", etc.

    # Event-Kontext
    orchestration_id: str
    subtask_id: str | None = None
    agent_id: str | None = None

    # User-Kontext
    user_id: str | None = None
    tenant_id: str | None = None

    # Event-Daten
    event_data: dict[str, Any] = field(default_factory=dict)
    policies_applied: list[str] = field(default_factory=list)
    compliance_results: list[ComplianceResult] = field(default_factory=list)

    # Event-Metadaten
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_component: str = "policy_aware_selection"
    correlation_id: str | None = None

    # Audit-Klassifikation
    sensitivity_level: SecurityLevel = SecurityLevel.INTERNAL
    retention_period_days: int = 365

    # Compliance-Informationen
    compliance_frameworks: list[ComplianceStandard] = field(default_factory=list)
    audit_required: bool = True
