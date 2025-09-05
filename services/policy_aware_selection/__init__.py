# backend/services/policy_aware_selection/__init__.py
"""Policy-aware Agent Selection Package.

Implementiert Policy-Enforcement, Compliance-Checks und Multi-Tenant-Isolation
für intelligente Agent-Selection mit Integration in den Orchestrator Service.
"""

from __future__ import annotations

from .agent_selector import PolicyAwareAgentSelector
from .compliance_checker import AgentComplianceChecker
from .compliance_monitor import ComplianceMonitor
from .data_models import (
    AgentComplianceProfile,
    AgentPolicy,
    AgentSelectionContext,
    AuditEvent,
    ComplianceResult,
    ComplianceStatus,
    DataClassification,
    PolicyConstraint,
    PolicyEffect,
    PolicyEvaluationResult,
    PolicyPriority,
    PolicyType,
    PolicyViolation,
    SecurityLevel,
)
from .orchestrator_integration import PolicyAwareOrchestrationIntegration
from .policy_engine import PolicyEnforcementEngine

__all__ = [
    # Core Components
    "PolicyAwareAgentSelector",
    "AgentComplianceChecker",
    "PolicyEnforcementEngine",
    "ComplianceMonitor",
    "PolicyAwareOrchestrationIntegration",

    # Data Models
    "AgentPolicy",
    "PolicyConstraint",
    "AgentSelectionContext",
    "PolicyViolation",
    "ComplianceResult",
    "AgentComplianceProfile",
    "PolicyEvaluationResult",
    "AuditEvent",

    # Enums
    "PolicyType",
    "PolicyEffect",
    "PolicyPriority",
    "SecurityLevel",
    "DataClassification",
    "ComplianceStatus",

    # Factory Functions
    "create_policy_aware_agent_selector",
    "create_agent_compliance_checker",
    "create_policy_enforcement_engine",
    "create_compliance_monitor",
    "create_orchestrator_integration",
]

__version__ = "1.0.0"


def create_policy_aware_agent_selector(
    agent_registry,
    policy_engine,
    security_manager,
    compliance_checker=None,
    policy_enforcement_engine=None
) -> PolicyAwareAgentSelector:
    """Factory-Funktion für Policy-aware Agent Selector.

    Args:
        agent_registry: Agent Registry Instanz
        policy_engine: Policy Engine Instanz
        security_manager: Security Manager Instanz
        compliance_checker: Compliance Checker (optional)
        policy_enforcement_engine: Policy Enforcement Engine (optional)

    Returns:
        Konfigurierter Policy-aware Agent Selector
    """
    return PolicyAwareAgentSelector(
        agent_registry=agent_registry,
        policy_engine=policy_engine,
        security_manager=security_manager,
        compliance_checker=compliance_checker,
        policy_enforcement_engine=policy_enforcement_engine
    )


def create_agent_compliance_checker(policy_engine) -> AgentComplianceChecker:
    """Factory-Funktion für Agent Compliance Checker.

    Args:
        policy_engine: Policy Engine Instanz

    Returns:
        Konfigurierter Agent Compliance Checker
    """
    return AgentComplianceChecker(policy_engine)


def create_policy_enforcement_engine(policy_engine) -> PolicyEnforcementEngine:
    """Factory-Funktion für Policy Enforcement Engine.

    Args:
        policy_engine: Policy Engine Instanz

    Returns:
        Konfigurierte Policy Enforcement Engine
    """
    return PolicyEnforcementEngine(policy_engine)


def create_compliance_monitor(bus_service=None) -> ComplianceMonitor:
    """Factory-Funktion für Compliance Monitor.

    Args:
        bus_service: Message Bus Service (optional)

    Returns:
        Konfigurierter Compliance Monitor
    """
    return ComplianceMonitor(bus_service)


def create_orchestrator_integration(
    execution_engine,
    policy_engine,
    security_manager,
    enable_policy_enforcement=True
) -> PolicyAwareOrchestrationIntegration:
    """Factory-Funktion für Orchestrator Integration.

    Args:
        execution_engine: Orchestrator Execution Engine
        policy_engine: Policy Engine Instanz
        security_manager: Security Manager Instanz
        enable_policy_enforcement: Policy-Enforcement aktivieren

    Returns:
        Konfigurierte Orchestrator Integration
    """
    return PolicyAwareOrchestrationIntegration(
        execution_engine=execution_engine,
        policy_engine=policy_engine,
        security_manager=security_manager,
        enable_policy_enforcement=enable_policy_enforcement
    )
