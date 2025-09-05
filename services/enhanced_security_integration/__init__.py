# backend/services/enhanced_security_integration/__init__.py
"""Enhanced Security Integration Package.

Implementiert Enterprise-Grade Security-Features mit Plan Persistence,
Multi-Tenant Security-Isolation, Enhanced Authentication/Authorization,
Secure Communication Channels und Security-Event-Monitoring.
"""

from __future__ import annotations

from .data_models import (
    EncryptedState,
    EncryptionAlgorithm,
    SecureCommunicationChannel,
    SecurityCheckResult,
    SecurityContext,
    SecurityEvent,
    SecurityEventType,
    SecurityLevel,
    SecurityPerformanceMetrics,
    StateIntegrityCheck,
    StateIntegrityStatus,
    TenantSecurityBoundary,
    ThreatDetectionResult,
    ThreatLevel,
)
from .plan_persistence_manager import PlanPersistenceManager
from .policy_integration_layer import PolicyIntegrationLayer
from .secure_communication_manager import SecureCommunicationManager
from .security_integration_engine import EnhancedSecurityIntegrationEngine
from .threat_detection_engine import ThreatDetectionEngine

__all__ = [
    # Core Components
    "EnhancedSecurityIntegrationEngine",
    "PlanPersistenceManager",
    "SecureCommunicationManager",
    "ThreatDetectionEngine",
    "PolicyIntegrationLayer",

    # Data Models
    "SecurityContext",
    "SecurityEvent",
    "SecurityCheckResult",
    "ThreatDetectionResult",
    "EncryptedState",
    "StateIntegrityCheck",
    "TenantSecurityBoundary",
    "SecureCommunicationChannel",
    "SecurityPerformanceMetrics",

    # Enums
    "SecurityEventType",
    "ThreatLevel",
    "EncryptionAlgorithm",
    "StateIntegrityStatus",
    "SecurityLevel",

    # Factory Functions
    "create_enhanced_security_integration_engine",
    "create_plan_persistence_manager",
    "create_secure_communication_manager",
    "create_threat_detection_engine",
    "create_policy_integration_layer",
]

__version__ = "1.0.0"


def create_enhanced_security_integration_engine(
    rbac_system,
    tenant_isolation_service,
    mtls_manager,
    plan_persistence_manager=None,
    threat_detection_engine=None,
    secure_communication_manager=None
) -> EnhancedSecurityIntegrationEngine:
    """Factory-Funktion für Enhanced Security Integration Engine.

    Args:
        rbac_system: RBAC/ABAC System
        tenant_isolation_service: Tenant Isolation Service
        mtls_manager: mTLS Manager
        plan_persistence_manager: Plan Persistence Manager (optional)
        threat_detection_engine: Threat Detection Engine (optional)
        secure_communication_manager: Secure Communication Manager (optional)

    Returns:
        Konfigurierte Enhanced Security Integration Engine
    """
    return EnhancedSecurityIntegrationEngine(
        rbac_system=rbac_system,
        tenant_isolation_service=tenant_isolation_service,
        mtls_manager=mtls_manager,
        plan_persistence_manager=plan_persistence_manager,
        threat_detection_engine=threat_detection_engine,
        secure_communication_manager=secure_communication_manager
    )


def create_plan_persistence_manager(encryption_key=None) -> PlanPersistenceManager:
    """Factory-Funktion für Plan Persistence Manager.

    Args:
        encryption_key: Encryption Key (optional)

    Returns:
        Konfigurierter Plan Persistence Manager
    """
    return PlanPersistenceManager(encryption_key=encryption_key)


def create_secure_communication_manager() -> SecureCommunicationManager:
    """Factory-Funktion für Secure Communication Manager.

    Returns:
        Konfigurierter Secure Communication Manager
    """
    return SecureCommunicationManager()


def create_threat_detection_engine() -> ThreatDetectionEngine:
    """Factory-Funktion für Threat Detection Engine.

    Returns:
        Konfigurierte Threat Detection Engine
    """
    return ThreatDetectionEngine()


def create_policy_integration_layer(
    security_integration_engine,
    policy_aware_selector,
    rbac_system
) -> PolicyIntegrationLayer:
    """Factory-Funktion für Policy Integration Layer.

    Args:
        security_integration_engine: Enhanced Security Integration Engine
        policy_aware_selector: Policy-aware Agent Selector
        rbac_system: RBAC Authorization Service

    Returns:
        Konfigurierte Policy Integration Layer
    """
    return PolicyIntegrationLayer(
        security_integration_engine=security_integration_engine,
        policy_aware_selector=policy_aware_selector,
        rbac_system=rbac_system
    )
