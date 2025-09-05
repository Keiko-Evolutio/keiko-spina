# backend/agents/capabilities/specialized_capabilities.py
"""Spezialisierte Capability-Typen für Agent-Kategorien.

Definiert typisierte Capability-Definitionen für Tools, Skills,
Domains und Policies mit kategorie-spezifischen Validierungsregeln.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from kei_logging import get_logger

from ..metadata.agent_metadata import CapabilityCategory
from .capability_utils import CapabilityConstants
from .enhanced_capabilities import EnhancedCapability

logger = get_logger(__name__)





class ToolType(str, Enum):
    """Typen von Tool-Capabilities."""

    FUNCTION = "function"
    API_ENDPOINT = "api_endpoint"
    WEBHOOK = "webhook"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ToolParameter:
    """Parameter-Definition für Tools.

    Definiert einen einzelnen Parameter für ein Tool mit Validierungsregeln
    und Typ-Informationen gemäß JSON Schema Standard.
    """

    name: str
    type: str
    description: str
    required: bool = False
    default_value: Any = None
    validation_pattern: str | None = None

    def __post_init__(self) -> None:
        """Validiert Parameter-Definition nach der Initialisierung."""
        if not self.name or not self.name.strip():
            raise ValueError("Parameter-Name darf nicht leer sein")

        if not self.description or not self.description.strip():
            raise ValueError("Parameter-Beschreibung darf nicht leer sein")

        valid_types = {"string", "number", "integer", "boolean", "object", "array", "null"}
        if self.type not in valid_types:
            raise ValueError(f"Ungültiger Parameter-Typ: {self.type}. Erlaubt: {valid_types}")

    def to_json_schema(self) -> dict[str, Any]:
        """Konvertiert Parameter zu JSON Schema Format.

        Returns:
            Dictionary im JSON Schema Format
        """
        schema = {
            "type": self.type,
            "description": self.description
        }

        if self.default_value is not None:
            schema["default"] = self.default_value

        if self.validation_pattern:
            schema["pattern"] = self.validation_pattern

        return schema
    enum_values: list[str] | None = None


@dataclass
class ToolEndpoint:
    """Endpunkt-Definition für Tools."""

    url: str
    method: str = "POST"
    headers: dict[str, str] | None = None
    timeout_seconds: int = 30
    retry_count: int = 3

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}


class ToolsCapability(EnhancedCapability):
    """Spezialisierte Capability für Tools."""

    tool_type: ToolType = Field(..., description="Typ des Tools")
    tool_parameters: list[ToolParameter] = Field(default_factory=list, description="Tool-Parameter")
    tool_endpoints: list[ToolEndpoint] = Field(default_factory=list, description="Tool-Endpunkte")

    execution_timeout: int = Field(
        default=CapabilityConstants.DEFAULT_EXECUTION_TIMEOUT,
        description="Ausführungs-Timeout in Sekunden"
    )
    max_concurrent_executions: int = Field(
        default=CapabilityConstants.MAX_CONCURRENT_EXECUTIONS,
        description="Maximale parallele Ausführungen"
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Erfordert Benutzerbestätigung"
    )

    allowed_domains: set[str] = Field(default_factory=set, description="Erlaubte Domains")
    security_level: str = Field(default="medium", description="Sicherheitsstufe")

    def __init__(self, **data: Any) -> None:
        data["category"] = CapabilityCategory.TOOLS
        super().__init__(**data)

    @field_validator("tool_parameters")
    @classmethod
    def validate_tool_parameters(cls, v: list[ToolParameter]) -> list[ToolParameter]:
        """Validiert Tool-Parameter."""
        param_names = [p.name for p in v]
        if len(param_names) != len(set(param_names)):
            raise ValueError("Parameter-Namen müssen eindeutig sein")
        return v

    def get_required_parameters(self) -> list[ToolParameter]:
        """Gibt erforderliche Parameter zurück."""
        return [p for p in self.tool_parameters if p.required]

    def validate_parameter_values(self, values: dict[str, Any]) -> bool:
        """Validiert Parameter-Werte gegen Schema."""
        for param in self.tool_parameters:
            if param.required and param.name not in values:
                return False

            if param.name in values:
                value = values[param.name]

                # Enum-Validierung
                if param.enum_values and value not in param.enum_values:
                    return False

                # Pattern-Validierung (vereinfacht)
                if param.validation_pattern and isinstance(value, str):
                    import re

                    if not re.match(param.validation_pattern, value):
                        return False

        return True


# ============================================================================
# SKILLS CAPABILITIES
# ============================================================================


class SkillLevel(str, Enum):
    """Skill-Level für Fähigkeiten."""

    BEGINNER = "beginner"  # Anfänger-Level
    INTERMEDIATE = "intermediate"  # Fortgeschritten
    ADVANCED = "advanced"  # Experte
    EXPERT = "expert"  # Spezialist


@dataclass
class SkillMetric:
    """Metrik für Skill-Bewertung."""

    name: str
    value: float
    unit: str
    description: str


class SkillsCapability(EnhancedCapability):
    """Spezialisierte Capability für Skills."""

    # Skill-spezifische Felder
    skill_level: SkillLevel = Field(..., description="Level der Fähigkeit")
    skill_domains: set[str] = Field(..., description="Anwendungsdomänen")
    skill_metrics: list[SkillMetric] = Field(default_factory=list, description="Skill-Metriken")

    # Lern-Konfiguration
    can_learn: bool = Field(default=False, description="Kann lernen und sich verbessern")
    learning_rate: float = Field(default=0.1, description="Lernrate")
    training_data_required: bool = Field(default=False, description="Benötigt Trainingsdaten")

    # Performance-Metriken
    accuracy_score: float | None = Field(default=None, description="Genauigkeits-Score")
    confidence_threshold: float = Field(default=0.8, description="Konfidenz-Schwellwert")

    def __init__(self, **data: Any) -> None:
        # Setze Kategorie automatisch
        data["category"] = CapabilityCategory.SKILLS
        super().__init__(**data)

    @field_validator("accuracy_score")
    @classmethod
    def validate_accuracy_score(cls, v: float | None) -> float | None:
        """Validiert Accuracy Score."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Accuracy Score muss zwischen 0.0 und 1.0 liegen")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validiert Konfidenz-Schwellwert."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Konfidenz-Schwellwert muss zwischen 0.0 und 1.0 liegen")
        return v

    def is_confident_enough(self, confidence: float) -> bool:
        """Prüft, ob Konfidenz ausreichend ist."""
        return confidence >= self.confidence_threshold

    def get_skill_metric(self, metric_name: str) -> SkillMetric | None:
        """Gibt spezifische Skill-Metrik zurück."""
        for metric in self.skill_metrics:
            if metric.name == metric_name:
                return metric
        return None


# ============================================================================
# DOMAINS CAPABILITIES
# ============================================================================


class DomainType(str, Enum):
    """Typen von Domain-Capabilities."""

    KNOWLEDGE_BASE = "knowledge_base"  # Wissensbasis
    ONTOLOGY = "ontology"  # Ontologie
    TAXONOMY = "taxonomy"  # Taxonomie
    VOCABULARY = "vocabulary"  # Vokabular
    CONTEXT = "context"  # Kontext-Domain


@dataclass
class DomainEntity:
    """Entität in einer Domain."""

    id: str
    name: str
    type: str
    properties: dict[str, Any]
    relationships: list[str] = None

    def __post_init__(self) -> None:
        if self.relationships is None:
            self.relationships = []


@dataclass
class DomainRelationship:
    """Beziehung zwischen Domain-Entitäten."""

    source_entity: str
    target_entity: str
    relationship_type: str
    properties: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.properties is None:
            self.properties = {}


class DomainsCapability(EnhancedCapability):
    """Spezialisierte Capability für Domains."""

    # Domain-spezifische Felder
    domain_type: DomainType = Field(..., description="Typ der Domain")
    domain_scope: str = Field(..., description="Scope der Domain")
    domain_entities: list[DomainEntity] = Field(
        default_factory=list, description="Domain-Entitäten"
    )
    domain_relationships: list[DomainRelationship] = Field(
        default_factory=list, description="Domain-Beziehungen"
    )

    # Wissens-Konfiguration
    knowledge_source: str | None = Field(default=None, description="Quelle des Wissens")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
    update_frequency: str | None = Field(default=None, description="Update-Frequenz")

    # Qualitäts-Metriken
    completeness_score: float = Field(default=0.0, description="Vollständigkeits-Score")
    consistency_score: float = Field(default=0.0, description="Konsistenz-Score")

    def __init__(self, **data: Any) -> None:
        # Setze Kategorie automatisch
        data["category"] = CapabilityCategory.DOMAINS
        super().__init__(**data)

    @field_validator("completeness_score", "consistency_score")
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Validiert Score-Werte."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Score muss zwischen 0.0 und 1.0 liegen")
        return v

    def get_entity(self, entity_id: str) -> DomainEntity | None:
        """Gibt Domain-Entität zurück."""
        for entity in self.domain_entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_relationships_for_entity(self, entity_id: str) -> list[DomainRelationship]:
        """Gibt Beziehungen für Entität zurück."""
        relationships = []
        for rel in self.domain_relationships:
            if rel.source_entity == entity_id or rel.target_entity == entity_id:
                relationships.append(rel)
        return relationships

    def is_knowledge_current(self, max_age_days: int = 30) -> bool:
        """Prüft, ob Wissen aktuell ist."""
        if not self.update_frequency:
            return True

        age = datetime.now(UTC) - self.last_updated
        return age.days <= max_age_days


# ============================================================================
# POLICIES CAPABILITIES
# ============================================================================


class PolicyType(str, Enum):
    """Typen von Policy-Capabilities."""

    SECURITY = "security"  # Sicherheits-Policies
    COMPLIANCE = "compliance"  # Compliance-Regeln
    GOVERNANCE = "governance"  # Governance-Policies
    PRIVACY = "privacy"  # Datenschutz-Policies
    SAFETY = "safety"  # Sicherheits-Policies


class PolicySeverity(str, Enum):
    """Schweregrad von Policy-Verletzungen."""

    LOW = "low"  # Niedrig
    MEDIUM = "medium"  # Mittel
    HIGH = "high"  # Hoch
    CRITICAL = "critical"  # Kritisch


@dataclass
class PolicyRule:
    """Einzelne Policy-Regel."""

    id: str
    name: str
    description: str
    condition: str  # Regel-Bedingung (z.B. JSON Logic)
    action: str  # Aktion bei Verletzung
    severity: PolicySeverity
    enabled: bool = True


@dataclass
class PolicyViolation:
    """Policy-Verletzung."""

    rule_id: str
    message: str
    severity: PolicySeverity
    timestamp: datetime
    context: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = {}


class PoliciesCapability(EnhancedCapability):
    """Spezialisierte Capability für Policies."""

    # Policy-spezifische Felder
    policy_type: PolicyType = Field(..., description="Typ der Policy")
    policy_rules: list[PolicyRule] = Field(default_factory=list, description="Policy-Regeln")
    enforcement_mode: str = Field(default="enforce", description="Durchsetzungs-Modus")

    # Audit-Konfiguration
    audit_enabled: bool = Field(default=True, description="Audit aktiviert")
    violation_threshold: int = Field(default=10, description="Verletzungs-Schwellwert")

    # Statistiken
    total_evaluations: int = Field(default=0, description="Gesamte Evaluierungen")
    total_violations: int = Field(default=0, description="Gesamte Verletzungen")

    def __init__(self, **data: Any) -> None:
        # Setze Kategorie automatisch
        data["category"] = CapabilityCategory.POLICIES
        super().__init__(**data)

    @field_validator("enforcement_mode")
    @classmethod
    def validate_enforcement_mode(cls, v: str) -> str:
        """Validiert Durchsetzungs-Modus."""
        valid_modes = ["enforce", "warn", "audit", "disabled"]
        if v not in valid_modes:
            raise ValueError(f"Ungültiger Enforcement-Modus: {v}")
        return v

    def get_rule(self, rule_id: str) -> PolicyRule | None:
        """Gibt Policy-Regel zurück."""
        for rule in self.policy_rules:
            if rule.id == rule_id:
                return rule
        return None

    def get_enabled_rules(self) -> list[PolicyRule]:
        """Gibt aktivierte Regeln zurück."""
        return [rule for rule in self.policy_rules if rule.enabled]

    def evaluate_rules(self, context: dict[str, Any]) -> list[PolicyViolation]:
        """Evaluiert alle Regeln gegen Kontext (vereinfacht)."""
        violations = []

        for rule in self.get_enabled_rules():
            # Vereinfachte Regel-Evaluierung
            # In echter Implementierung würde hier JSON Logic oder ähnliches verwendet
            if PoliciesCapability._evaluate_rule_condition(_rule=rule, _context=context):
                violation = PolicyViolation(
                    rule_id=rule.id,
                    message=f"Policy-Verletzung: {rule.name}",
                    severity=rule.severity,
                    timestamp=datetime.now(UTC),
                    context=context,
                )
                violations.append(violation)

        # Statistiken aktualisieren
        self.total_evaluations += 1
        self.total_violations += len(violations)

        return violations

    @staticmethod
    def _evaluate_rule_condition(_rule: PolicyRule, _context: dict[str, Any]) -> bool:
        """Evaluiert Regel-Bedingung.

        Args:
            _rule: Policy-Regel zur Evaluierung
            _context: Kontext-Daten für die Evaluierung

        Returns:
            False für vereinfachte Implementierung (keine Verletzung)
        """
        return False

    @property
    def violation_rate(self) -> float:
        """Berechnet Verletzungsrate."""
        if self.total_evaluations == 0:
            return 0.0
        return self.total_violations / self.total_evaluations
