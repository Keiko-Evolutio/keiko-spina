# backend/api/routes/agent_policies_management.py
"""Agent Policies Management Endpoints für Keiko Personal Assistant

Implementiert Sicherheits- und Compliance-Richtlinien für Agenten mit
Policy-Enforcement und Compliance-Monitoring.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field, field_validator

from kei_logging import (
    BusinessLogicError,
    LogLinkedError,
    ValidationError,
    get_logger,
    with_log_links,
)

from .enhanced_agents_management import (
    PolicyConfiguration,
    PolicyStatusResponse,
    get_agent_or_404,
    validate_agent_access,
)

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/agents", tags=["Agent Policies Management"])


class PolicyUpdateRequest(BaseModel):
    """Request-Model für Policy-Updates."""
    security_level: str | None = Field(None, description="Sicherheitslevel")
    data_retention_days: int | None = Field(None, ge=1, le=3650, description="Datenaufbewahrung in Tagen")
    allowed_domains: list[str] | None = Field(None, description="Erlaubte Domains")
    blocked_domains: list[str] | None = Field(None, description="Blockierte Domains")
    require_encryption: bool | None = Field(None, description="Verschlüsselung erforderlich")
    audit_logging: bool | None = Field(None, description="Audit-Logging aktiviert")
    compliance_frameworks: list[str] | None = Field(None, description="Compliance-Frameworks")

    @field_validator("security_level")
    @classmethod
    def validate_security_level(cls, v):
        """Validiert Sicherheitslevel."""
        if v is not None:
            allowed_levels = ["low", "standard", "high", "critical"]
            if v not in allowed_levels:
                raise ValueError(f"Sicherheitslevel muss einer von {allowed_levels} sein")
        return v

    @field_validator("allowed_domains", "blocked_domains")
    @classmethod
    def validate_domains(cls, v):
        """Validiert Domain-Listen."""
        if v is not None:
            import re
            domain_pattern = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$")
            for domain in v:
                if not domain_pattern.match(domain):
                    raise ValueError(f"Ungültige Domain: {domain}")
        return v

    @field_validator("compliance_frameworks")
    @classmethod
    def validate_compliance_frameworks(cls, v):
        """Validiert Compliance-Frameworks."""
        if v is not None:
            allowed_frameworks = ["gdpr", "hipaa", "sox", "pci_dss", "iso27001", "nist"]
            for framework in v:
                if framework not in allowed_frameworks:
                    raise ValueError(f"Unbekanntes Compliance-Framework: {framework}")
        return v


class PolicyViolation(BaseModel):
    """Policy-Verletzung."""
    violation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Verletzungs-ID")
    policy_name: str = Field(..., description="Name der verletzten Policy")
    violation_type: str = Field(..., description="Typ der Verletzung")
    severity: str = Field(..., description="Schweregrad")
    description: str = Field(..., description="Beschreibung der Verletzung")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Zeitstempel")
    resolved: bool = Field(default=False, description="Verletzung behoben")
    resolution_timestamp: datetime | None = Field(None, description="Zeitstempel der Behebung")
    additional_data: dict[str, Any] = Field(default_factory=dict, description="Zusätzliche Daten")


class ComplianceReport(BaseModel):
    """Compliance-Report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Report-ID")
    agent_id: str = Field(..., description="Agent-ID")
    framework: str = Field(..., description="Compliance-Framework")
    compliance_score: float = Field(..., ge=0.0, le=100.0, description="Compliance-Score in Prozent")
    passed_checks: int = Field(..., description="Bestandene Checks")
    failed_checks: int = Field(..., description="Fehlgeschlagene Checks")
    total_checks: int = Field(..., description="Gesamt-Checks")
    violations: list[PolicyViolation] = Field(default_factory=list, description="Policy-Verletzungen")
    recommendations: list[str] = Field(default_factory=list, description="Empfehlungen")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Generierungszeitpunkt")
    valid_until: datetime = Field(..., description="Gültig bis")


def _get_policy_configuration(agent_id: str) -> PolicyConfiguration:
    """Holt Policy-Konfiguration für Agent."""
    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        # Hole Agent-spezifische Policies
        policies = policy_engine.get_agent_policies(agent_id)

        return PolicyConfiguration(**policies)

    except ImportError:
        policy_engine = None  # type: ignore
        logger.warning("Policy Engine nicht verfügbar - verwende Standard-Policies")
        return PolicyConfiguration(
            security_level="standard",
            data_retention_days=90,
            allowed_domains=[],
            blocked_domains=[],
            require_encryption=True,
            audit_logging=True,
            compliance_frameworks=["gdpr"]
        )


def _save_policy_configuration(agent_id: str, config: PolicyConfiguration) -> None:
    """Speichert Policy-Konfiguration für Agent."""
    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        policy_engine.set_agent_policies(agent_id, config.dict())

    except ImportError:
        policy_engine = None  # type: ignore
        logger.warning("Policy Engine nicht verfügbar - Policy-Speicherung übersprungen")


def _check_policy_compliance(agent_id: str, config: PolicyConfiguration) -> dict[str, bool]:
    """Prüft Policy-Compliance für Agent."""
    compliance_status = {}

    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        # Prüfe verschiedene Compliance-Aspekte
        compliance_status["encryption_enabled"] = config.require_encryption
        compliance_status["audit_logging_enabled"] = config.audit_logging
        compliance_status["data_retention_configured"] = config.data_retention_days is not None
        compliance_status["security_level_appropriate"] = config.security_level in ["standard", "high", "critical"]
        compliance_status["domain_restrictions_configured"] = len(config.allowed_domains) > 0 or len(config.blocked_domains) > 0

        # Prüfe Framework-spezifische Compliance
        for framework in config.compliance_frameworks or []:
            compliance_status[f"{framework}_compliant"] = policy_engine.check_framework_compliance(agent_id, framework)

    except ImportError:
        policy_engine = None  # type: ignore
        logger.warning("Policy Engine nicht verfügbar - verwende Standard-Compliance-Checks")
        compliance_status = {
            "encryption_enabled": config.require_encryption,
            "audit_logging_enabled": config.audit_logging,
            "data_retention_configured": config.data_retention_days is not None,
            "security_level_appropriate": config.security_level in ["standard", "high", "critical"]
        }

    return compliance_status


def _get_policy_violations(agent_id: str) -> list[PolicyViolation]:
    """Holt Policy-Verletzungen für Agent."""
    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        violations_data = policy_engine.get_agent_violations(agent_id)

        violations = []
        for violation_data in violations_data:
            violations.append(PolicyViolation(**violation_data))

        return violations

    except ImportError:
        policy_engine = None  # type: ignore
        logger.warning("Policy Engine nicht verfügbar - keine Verletzungen verfügbar")
        return []


def _generate_compliance_report(agent_id: str, framework: str, config: PolicyConfiguration) -> ComplianceReport:
    """Generiert Compliance-Report für spezifisches Framework."""
    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        report_data = policy_engine.generate_compliance_report(agent_id, framework)

        return ComplianceReport(**report_data)

    except ImportError:
        policy_engine = None  # type: ignore
        logger.warning("Policy Engine nicht verfügbar - generiere Standard-Report")

        # Standard-Compliance-Checks
        total_checks = 10
        passed_checks = 0

        if config.require_encryption:
            passed_checks += 2
        if config.audit_logging:
            passed_checks += 2
        if config.data_retention_days and config.data_retention_days <= 365:
            passed_checks += 2
        if config.security_level in ["high", "critical"]:
            passed_checks += 2
        if len(config.compliance_frameworks or []) > 0:
            passed_checks += 2

        failed_checks = total_checks - passed_checks
        compliance_score = (passed_checks / total_checks) * 100

        return ComplianceReport(
            agent_id=agent_id,
            framework=framework,
            compliance_score=compliance_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            total_checks=total_checks,
            violations=[],
            recommendations=[
                "Aktiviere Verschlüsselung für alle Datenübertragungen",
                "Implementiere umfassendes Audit-Logging",
                "Konfiguriere angemessene Datenaufbewahrungsrichtlinien",
                "Erhöhe Sicherheitslevel auf 'high' oder 'critical'",
                "Definiere spezifische Compliance-Frameworks"
            ][:failed_checks],
            valid_until=datetime.now(UTC) + timedelta(days=30)
        )


# Policies Management Endpoints

@router.get("/{agent_id}/policies", response_model=PolicyStatusResponse)
@with_log_links(component="policy_management", operation="get_policies")
async def get_agent_policies(
    agent_id: str = Path(..., description="Agent-ID"),
    include_violations: bool = Query(default=True, description="Policy-Verletzungen einschließen"),
    include_compliance: bool = Query(default=True, description="Compliance-Status einschließen")
) -> PolicyStatusResponse:
    """Holt Policy-Konfiguration und Compliance-Status für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        include_violations: Policy-Verletzungen in Response einschließen
        include_compliance: Compliance-Status in Response einschließen

    Returns:
        Policy-Konfiguration und Compliance-Informationen

    Raises:
        ValidationError: Bei ungültiger Agent-ID
        AuthorizationError: Bei fehlenden Berechtigungen
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="read")

    try:
        # Hole Policy-Konfiguration
        config = _get_policy_configuration(agent_id)

        # Prüfe Compliance-Status
        compliance_status = {}
        if include_compliance:
            compliance_status = _check_policy_compliance(agent_id, config)

        # Hole Policy-Verletzungen
        violations = []
        if include_violations:
            violations = _get_policy_violations(agent_id)

        # Hole letztes Audit-Datum
        last_audit = None
        try:
            from audit_system import audit_manager
            last_audit = audit_manager.get_last_audit_date(agent_id)
        except ImportError:
            # Definiere audit_manager als None im except Block
            audit_manager = None

        # Erstelle Response
        response_data = {
            "configuration": config,
            "compliance_status": compliance_status,
            "violations": [v.dict() for v in violations],
            "last_audit": last_audit
        }

        logger.info(
            f"Policy-Informationen für Agent {agent_id} abgerufen",
            extra={
                "agent_id": agent_id,
                "violations_count": len(violations),
                "compliance_checks": len(compliance_status),
                "correlation_id": f"policy_get_{uuid.uuid4().hex[:8]}"
            }
        )

        return PolicyStatusResponse(**response_data)

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Policy-Abfrage fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="policy_management",
            operation="get_policies",
            cause=e
        )


@router.put("/{agent_id}/policies")
@with_log_links(component="policy_management", operation="update_policies")
async def update_agent_policies(
    agent_id: str = Path(..., description="Agent-ID"),
    request: PolicyUpdateRequest = ...
) -> dict[str, Any]:
    """Aktualisiert Policy-Konfiguration für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        request: Neue Policy-Konfiguration

    Returns:
        Bestätigung der Policy-Aktualisierung

    Raises:
        ValidationError: Bei ungültigen Policy-Werten
        AuthorizationError: Bei fehlenden Berechtigungen
        BusinessLogicError: Bei Policy-Update-Fehlern
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="admin")

    try:
        # Hole aktuelle Konfiguration
        current_config = _get_policy_configuration(agent_id)

        # Update-Felder anwenden
        update_data = request.dict(exclude_unset=True)

        # Validiere Policy-Konsistenz
        if "allowed_domains" in update_data and "blocked_domains" in update_data:
            allowed = set(update_data["allowed_domains"])
            blocked = set(update_data["blocked_domains"])
            overlap = allowed.intersection(blocked)
            if overlap:
                raise ValidationError(
                    message=f"Domains können nicht gleichzeitig erlaubt und blockiert sein: {list(overlap)}",
                    field="domains",
                    value=list(overlap)
                )

        # Erstelle neue Konfiguration
        new_config_data = current_config.dict()
        new_config_data.update(update_data)
        new_config = PolicyConfiguration(**new_config_data)

        # Speichere neue Konfiguration
        _save_policy_configuration(agent_id, new_config)

        # Prüfe neue Compliance
        new_compliance_status = _check_policy_compliance(agent_id, new_config)

        logger.info(
            f"Policy-Konfiguration für Agent {agent_id} aktualisiert",
            extra={
                "agent_id": agent_id,
                "updated_fields": list(update_data.keys()),
                "compliance_status": new_compliance_status,
                "correlation_id": f"policy_update_{uuid.uuid4().hex[:8]}"
            }
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "updated_at": datetime.now(UTC).isoformat(),
            "updated_fields": list(update_data.keys()),
            "new_configuration": new_config.dict(),
            "compliance_status": new_compliance_status
        }

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Policy-Update fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="policy_management",
            operation="update_policies",
            cause=e
        )


@router.get("/{agent_id}/policies/compliance/{framework}", response_model=ComplianceReport)
@with_log_links(component="policy_management", operation="get_compliance_report")
async def get_compliance_report(
    agent_id: str = Path(..., description="Agent-ID"),
    framework: str = Path(..., description="Compliance-Framework")
) -> ComplianceReport:
    """Generiert detaillierten Compliance-Report für spezifisches Framework.

    Args:
        agent_id: Eindeutige Agent-ID
        framework: Compliance-Framework (gdpr, hipaa, sox, etc.)

    Returns:
        Detaillierter Compliance-Report

    Raises:
        ValidationError: Bei ungültigem Framework
        AuthorizationError: Bei fehlenden Berechtigungen
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="read")

    # Validiere Framework
    allowed_frameworks = ["gdpr", "hipaa", "sox", "pci_dss", "iso27001", "nist"]
    if framework not in allowed_frameworks:
        raise ValidationError(
            message=f"Unbekanntes Compliance-Framework: {framework}",
            field="framework",
            value=framework
        )

    try:
        # Hole Policy-Konfiguration
        config = _get_policy_configuration(agent_id)

        # Generiere Compliance-Report
        report = _generate_compliance_report(agent_id, framework, config)

        logger.info(
            f"Compliance-Report für Agent {agent_id} und Framework {framework} generiert",
            extra={
                "agent_id": agent_id,
                "framework": framework,
                "compliance_score": report.compliance_score,
                "violations_count": len(report.violations),
                "correlation_id": f"compliance_{uuid.uuid4().hex[:8]}"
            }
        )

        return report

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Compliance-Report-Generierung fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            framework=framework,
            component="policy_management",
            operation="get_compliance_report",
            cause=e
        )
