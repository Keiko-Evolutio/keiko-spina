# backend/audit_system/__init__.py
"""Vollständiges revisionssicheres Audit System für Keiko Personal Assistant

Implementiert tamper-proof Audit-Trails, kryptographische Signaturen,
Blockchain-ähnliche Verkettung und comprehensive Action Logging.
"""

from __future__ import annotations

from kei_logging import get_logger

# Comprehensive Action Logging
from .action_logger import (
    ActionCategory,
    ActionLogger,
    AgentActionEvent,
    CommunicationEvent,
    PolicyEnforcementEvent,
    QuotaUsageEvent,
    ToolCallEvent,
    action_logger,
)

# Utilities and Constants
from .audit_constants import (
    AuditAlertTypes,
    AuditConstants,
    AuditEnvironmentVariables,
    AuditMessages,
    AuditPaths,
)

# Audit Middleware
from .audit_middleware import (
    AuditConfig,
    AuditEnforcementResult,
    AuditMiddleware,
    audit_decorator,
    require_audit_compliance,
)

# Monitoring and Analytics
from .audit_monitoring import (
    AuditAlert,
    AuditAnalytics,
    AuditDashboard,
    AuditHealthCheck,
    AuditMonitor,
    ComplianceReport,
    audit_monitor,
)

# Performance and Scalability
from .audit_performance import (
    AsyncAuditProcessor,
    AuditEventStreamer,
    AuditPerformanceManager,
    BatchProcessor,
    HorizontalScaler,
    PerformanceMetrics,
    audit_performance_manager,
)

# Enhanced PII Redaction for Audit
from .audit_pii_redaction import (
    AuditAnonymizer,
    AuditPIIRedactor,
    ConsentManager,
    PIIAuditPolicy,
    RedactionLevel,
    ReversibleRedaction,
    audit_pii_redactor,
)
from .audit_utils import (
    AsyncTaskManager,
    RequestMetadata,
    calculate_duration_ms,
    create_error_context,
    create_request_metadata,
    extract_client_ip,
    extract_user_agent,
    generate_correlation_id,
    get_current_timestamp,
    is_path_excluded,
    safe_async_call,
    sanitize_for_logging,
)

# Compliance and Retention
from .compliance_manager import (
    ArchiveManager,
    ComplianceManager,
    ComplianceStandard,
    ExportManager,
    RetentionPolicy,
    RightToBeForgottenHandler,
    compliance_manager,
)

# Core Audit System
from .core_audit_engine import (
    AuditBlock,
    AuditChain,
    AuditContext,
    AuditEngine,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditSignature,
    audit_engine,
)

# Tamper-Proof Audit Trail
from .tamper_proof_trail import (
    AuditHashChain,
    BlockchainAuditChain,
    CryptographicSigner,
    IntegrityVerifier,
    TamperDetectionResult,
    TamperProofAuditTrail,
    tamper_proof_trail,
)

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "ActionCategory",
    # Comprehensive Action Logging
    "ActionLogger",
    "AgentActionEvent",
    "ArchiveManager",
    "AsyncAuditProcessor",
    "AsyncTaskManager",
    "AuditAlert",
    "AuditAlertTypes",
    "AuditAnalytics",
    "AuditAnonymizer",
    "AuditBlock",
    "AuditChain",
    "AuditConfig",
    # Utilities and Constants
    "AuditConstants",
    "AuditContext",
    "AuditDashboard",
    "AuditEnforcementResult",
    # Core Audit System
    "AuditEngine",
    "AuditEnvironmentVariables",
    "AuditEvent",
    "AuditEventStreamer",
    "AuditEventType",
    "AuditHashChain",
    "AuditHealthCheck",
    "AuditMessages",
    # Audit Middleware
    "AuditMiddleware",
    # Monitoring and Analytics
    "AuditMonitor",
    # Enhanced PII Redaction for Audit
    "AuditPIIRedactor",
    "AuditPaths",
    # Performance and Scalability
    "AuditPerformanceManager",
    "AuditSeverity",
    "AuditSignature",
    "BatchProcessor",
    "BlockchainAuditChain",
    "CommunicationEvent",
    # Compliance and Retention
    "ComplianceManager",
    "ComplianceReport",
    "ComplianceStandard",
    "ConsentManager",
    "CryptographicSigner",
    "ExportManager",
    "HorizontalScaler",
    "IntegrityVerifier",
    "PIIAuditPolicy",
    "PerformanceMetrics",
    "PolicyEnforcementEvent",
    "QuotaUsageEvent",
    "RedactionLevel",
    "RequestMetadata",
    "RetentionPolicy",
    "ReversibleRedaction",
    "RightToBeForgottenHandler",
    "TamperDetectionResult",
    # Tamper-Proof Audit Trail
    "TamperProofAuditTrail",
    "ToolCallEvent",
    "action_logger",
    "audit_decorator",
    "audit_engine",
    "audit_monitor",
    "audit_performance_manager",
    "audit_pii_redactor",
    "calculate_duration_ms",
    "compliance_manager",
    "create_error_context",
    "create_request_metadata",
    "extract_client_ip",
    "extract_user_agent",
    "generate_correlation_id",
    "get_current_timestamp",
    "is_path_excluded",
    "require_audit_compliance",
    "safe_async_call",
    "sanitize_for_logging",
    "tamper_proof_trail",
]

# Audit-System Status
def get_audit_system_status() -> dict:
    """Gibt Status des Audit-Systems zurück."""
    return {
        "package": "backend.audit_system",
        "version": "1.0.0",
        "components": {
            "core_audit_engine": True,
            "tamper_proof_trail": True,
            "action_logger": True,
            "audit_pii_redaction": True,
            "compliance_manager": True,
            "audit_performance": True,
            "audit_middleware": True,
            "audit_monitoring": True,
        },
        "features": {
            "tamper_proof_audit_trails": True,
            "cryptographic_signatures": True,
            "blockchain_like_chaining": True,
            "comprehensive_action_logging": True,
            "enhanced_pii_redaction": True,
            "reversible_redaction": True,
            "gdpr_ccpa_compliance": True,
            "consent_management": True,
            "configurable_retention": True,
            "automatic_archiving": True,
            "export_functions": True,
            "right_to_be_forgotten": True,
            "async_audit_logging": True,
            "batch_processing": True,
            "horizontal_scaling": True,
            "real_time_streaming": True,
            "tamper_detection": True,
            "integrity_verification": True,
            "compliance_reporting": True,
            "audit_analytics": True,
        },
        "audit_event_types": [
            "agent_input",
            "agent_output",
            "tool_call",
            "agent_communication",
            "policy_enforcement",
            "quota_usage",
            "authentication",
            "authorization",
            "data_access",
            "configuration_change",
            "system_event"
        ],
        "compliance_standards": [
            "sox",
            "gdpr",
            "ccpa",
            "hipaa",
            "pci_dss",
            "iso_27001",
            "nist_cybersecurity"
        ],
        "retention_periods": {
            "financial_sector": "7_years",
            "healthcare_sector": "6_years",
            "general_business": "3_years",
            "personal_data": "configurable"
        },
        "cryptographic_features": [
            "sha256_hashing",
            "rsa_signatures",
            "ecdsa_signatures",
            "merkle_trees",
            "hash_chains",
            "digital_timestamps"
        ]
    }

logger.info(f"Audit System geladen - Status: {get_audit_system_status()}")
