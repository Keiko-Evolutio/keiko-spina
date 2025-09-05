# backend/policy_engine/__init__.py
"""Vollständige Policy Engine für Keiko Personal Assistant

Implementiert Safety Guardrails, Compliance-Framework, erweiterte PII-Redaction,
Data Minimization, Prompt Guardrails und Integration mit bestehenden Systemen.
"""

from __future__ import annotations

from kei_logging import get_logger

# Compliance Framework
from .compliance_framework import (
    AuditTrailManager,
    ComplianceCheck,
    ComplianceEngine,
    ComplianceStandard,
    ComplianceViolation,
    DataProcessingPurpose,
    DataRetentionPolicy,
    GDPRChecker,
    HIPAAChecker,
    RightToBeForgottenHandler,
    compliance_engine,
)

# Core Policy Engine
from .core_policy_engine import (
    PolicyContext,
    PolicyDecision,
    PolicyEffect,
    PolicyEngine,
    PolicyEvaluationResult,
    PolicyPriority,
    PolicyRule,
    PolicyType,
    policy_engine,
)

# Data Minimization
from .data_minimization import (
    DataLifecycleManager,
    DataMinimizationEngine,
    DataMinimizationPolicy,
    DifferentialPrivacyMechanism,
    PrivacyMechanism,
    PurposeBinder,
    SamplingStrategy,
    SmartSampler,
    data_minimization_engine,
)

# Enhanced PII Redaction
from .enhanced_pii_redaction import (
    ContextualPIIDetector,
    CustomEntityRecognizer,
    EnhancedPIIRedactor,
    PIIDetectionResult,
    PIIEntity,
    PIIEntityType,
    RedactionStrategy,
    RegexPIIDetector,
    enhanced_pii_redactor,
)

# Policy Middleware
from .policy_middleware import (
    PolicyConfig,
    PolicyEnforcementMiddleware,
    PolicyEnforcementResult,
    enforce_data_minimization,
    enforce_safety_guardrails,
    require_policy_compliance,
)

# Prompt Guardrails
from .prompt_guardrails import (
    ContentFilter,
    InjectionType,
    InputValidator,
    PromptGuardrailsEngine,
    PromptInjectionDetector,
    PromptRiskLevel,
    PromptSanitizer,
    PromptValidationResult,
    prompt_guardrails_engine,
)

# Safety Guardrails
from .safety_guardrails import (
    BiasDetector,
    ContentSafetyCheck,
    HarmfulContentFilter,
    SafetyGuardrailsEngine,
    SafetyLevel,
    SafetyViolation,
    ToxicityDetector,
    ViolationType,
    safety_guardrails_engine,
)

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "AuditTrailManager",
    "BiasDetector",
    "ComplianceCheck",
    # Compliance Framework
    "ComplianceEngine",
    "ComplianceStandard",
    "ComplianceViolation",
    "ContentFilter",
    "ContentSafetyCheck",
    "ContextualPIIDetector",
    "CustomEntityRecognizer",
    "DataLifecycleManager",
    # Data Minimization
    "DataMinimizationEngine",
    "DataMinimizationPolicy",
    "DataProcessingPurpose",
    "DataRetentionPolicy",
    "DifferentialPrivacyMechanism",
    # Enhanced PII Redaction
    "EnhancedPIIRedactor",
    "GDPRChecker",
    "HIPAAChecker",
    "HarmfulContentFilter",
    "InjectionType",
    "InputValidator",
    "PIIDetectionResult",
    "PIIEntity",
    "PIIEntityType",
    "PolicyConfig",
    "PolicyContext",
    "PolicyDecision",
    "PolicyEffect",
    # Policy Middleware
    "PolicyEnforcementMiddleware",
    "PolicyEnforcementResult",
    # Core Policy Engine
    "PolicyEngine",
    "PolicyEvaluationResult",
    "PolicyPriority",
    "PolicyRule",
    "PolicyType",
    "PrivacyMechanism",
    # Prompt Guardrails
    "PromptGuardrailsEngine",
    "PromptInjectionDetector",
    "PromptRiskLevel",
    "PromptSanitizer",
    "PromptValidationResult",
    "PurposeBinder",
    "RedactionStrategy",
    "RegexPIIDetector",
    "RightToBeForgottenHandler",
    # Safety Guardrails
    "SafetyGuardrailsEngine",
    "SafetyLevel",
    "SafetyViolation",
    "SamplingStrategy",
    "SmartSampler",
    "ToxicityDetector",
    "ViolationType",
    "compliance_engine",
    "data_minimization_engine",
    "enforce_data_minimization",
    "enforce_safety_guardrails",
    "enhanced_pii_redactor",
    # System Status
    "get_policy_system_status",
    "policy_engine",
    "prompt_guardrails_engine",
    "require_policy_compliance",
    "safety_guardrails_engine",
]

# Policy-System Status
def get_policy_system_status() -> dict:
    """Gibt Status des Policy-Systems zurück."""
    return {
        "package": "backend.policy_engine",
        "version": "1.0.0",
        "components": {
            "core_policy_engine": True,
            "safety_guardrails": True,
            "compliance_framework": True,
            "enhanced_pii_redaction": True,
            "data_minimization": True,
            "prompt_guardrails": True,
            "policy_middleware": True,
        },
        "features": {
            "content_safety_checks": True,
            "toxicity_detection": True,
            "bias_detection": True,
            "harmful_content_filtering": True,
            "gdpr_ccpa_compliance": True,
            "industry_compliance": True,
            "audit_trails": True,
            "data_retention": True,
            "right_to_be_forgotten": True,
            "ml_based_pii_detection": True,
            "contextual_pii_recognition": True,
            "custom_entity_recognition": True,
            "configurable_redaction_strategies": True,
            "data_minimization": True,
            "smart_sampling": True,
            "differential_privacy": True,
            "prompt_injection_detection": True,
            "content_filtering": True,
            "prompt_sanitization": True,
            "real_time_monitoring": True,
            "circuit_breaker_pattern": True,
            "async_evaluation": True,
            "batch_processing": True,
        },
        "policy_types": [
            "safety_guardrails",
            "compliance_checks",
            "pii_redaction",
            "data_minimization",
            "prompt_validation",
            "content_filtering",
            "access_control",
            "audit_logging"
        ],
        "compliance_standards": [
            "gdpr",
            "ccpa",
            "hipaa",
            "sox",
            "pci_dss",
            "iso_27001"
        ]
    }

logger.info(f"Policy Engine geladen - Status: {get_policy_system_status()}")
