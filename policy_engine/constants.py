# backend/policy_engine/constants.py
"""Konsolidierte Konstanten für das Policy-Engine-Modul.

Eliminiert alle Magic Numbers und Hard-coded Strings aus dem policy_engine Modul
und stellt sie als konfigurierbare Konstanten zur Verfügung.
"""

from typing import Final

# =============================================================================
# CACHE-KONFIGURATION
# =============================================================================

# Cache TTL (Time To Live) in Sekunden
DEFAULT_CACHE_TTL: Final[int] = 300  # 5 Minuten
POLICY_CACHE_TTL: Final[int] = 300
TOOL_ACCESS_CACHE_TTL: Final[int] = 300
REDACTION_CACHE_TTL: Final[int] = 600  # 10 Minuten

# Cache-Größen
MAX_CACHE_SIZE: Final[int] = 10000
DEFAULT_CACHE_SIZE: Final[int] = 1000

# =============================================================================
# DETECTION-SCHWELLWERTE
# =============================================================================

# Confidence-Level für Detektionen
CONFIDENCE_LOW: Final[float] = 0.3
CONFIDENCE_MEDIUM: Final[float] = 0.6
CONFIDENCE_HIGH: Final[float] = 0.8
CONFIDENCE_VERY_HIGH: Final[float] = 0.9
CONFIDENCE_CERTAIN: Final[float] = 1.0

# Risk-Level-Schwellwerte
RISK_THRESHOLD_LOW: Final[float] = 0.3
RISK_THRESHOLD_MEDIUM: Final[float] = 0.6
RISK_THRESHOLD_HIGH: Final[float] = 0.8
RISK_THRESHOLD_CRITICAL: Final[float] = 0.9

# Toxicity-Schwellwerte
TOXICITY_THRESHOLD_LOW: Final[float] = 0.3
TOXICITY_THRESHOLD_MEDIUM: Final[float] = 0.6
TOXICITY_THRESHOLD_HIGH: Final[float] = 0.8
TOXICITY_THRESHOLD_UNSAFE: Final[float] = 0.9

# =============================================================================
# REDACTION-KONFIGURATION
# =============================================================================

# Standard-Redaction-Masks
DEFAULT_MASK: Final[str] = "***"
HASH_PREFIX: Final[str] = "[HASH:"
ENCRYPT_PREFIX: Final[str] = "[ENC:"
TOKEN_PREFIX: Final[str] = "["
NEUTRALIZED_PREFIX: Final[str] = "[NEUTRALIZED_ROLEPLAY]"

# Hash-Konfiguration
DEFAULT_HASH_LENGTH: Final[int] = 8
SHORT_HASH_LENGTH: Final[int] = 5
LONG_HASH_LENGTH: Final[int] = 16

# Token-Konfiguration
DEFAULT_TOKEN_MODULO: Final[int] = 10000
LARGE_TOKEN_MODULO: Final[int] = 100000
TOKEN_FORMAT_WIDTH: Final[int] = 4  # für :04d Format

# Encrypt-Konfiguration
DEFAULT_ENCRYPT_MODULO: Final[int] = 100000
ENCRYPT_FORMAT_WIDTH: Final[int] = 5  # für :05d Format

# =============================================================================
# POLICY-ENGINE-KONFIGURATION
# =============================================================================

# Policy-Entscheidungen
POLICY_DECISION_ALLOW: Final[str] = "allow"
POLICY_DECISION_DENY: Final[str] = "deny"
POLICY_DECISION_UNKNOWN: Final[str] = "unknown"

# Circuit-Breaker-Konfiguration
CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 60  # Sekunden
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS: Final[int] = 3

# =============================================================================
# DATA-MINIMIZATION-KONFIGURATION
# =============================================================================

# Sampling-Konfiguration
DEFAULT_STRATA_COUNT: Final[int] = 10
MIN_STRATA_COUNT: Final[int] = 2
MAX_STRATA_COUNT: Final[int] = 50

# Sampling-Raten
DEFAULT_SAMPLING_RATE: Final[float] = 0.1  # 10%
MIN_SAMPLING_RATE: Final[float] = 0.01     # 1%
MAX_SAMPLING_RATE: Final[float] = 1.0      # 100%

# =============================================================================
# PERFORMANCE-KONFIGURATION
# =============================================================================

# Timeouts in Sekunden
DEFAULT_PROCESSING_TIMEOUT: Final[int] = 30
FAST_PROCESSING_TIMEOUT: Final[int] = 5
SLOW_PROCESSING_TIMEOUT: Final[int] = 120

# Performance-Schwellwerte in Millisekunden
PERFORMANCE_WARNING_THRESHOLD: Final[float] = 100.0
PERFORMANCE_ERROR_THRESHOLD: Final[float] = 1000.0

# Batch-Größen
DEFAULT_BATCH_SIZE: Final[int] = 100
MIN_BATCH_SIZE: Final[int] = 1
MAX_BATCH_SIZE: Final[int] = 1000

# =============================================================================
# CONTENT-LIMITS
# =============================================================================

# Text-Längen-Limits
MAX_CONTENT_LENGTH: Final[int] = 100000  # 100KB
MAX_SNIPPET_LENGTH: Final[int] = 100
MAX_DESCRIPTION_LENGTH: Final[int] = 500

# Pattern-Limits
MAX_PATTERNS_PER_CATEGORY: Final[int] = 1000
MAX_KEYWORDS_PER_CATEGORY: Final[int] = 500

# =============================================================================
# LOGGING-KONFIGURATION
# =============================================================================

# Log-Level-Mapping
LOG_LEVEL_MAPPING: Final[dict[str, str]] = {
    "safe": "DEBUG",
    "low_risk": "INFO",
    "medium_risk": "WARNING",
    "high_risk": "ERROR",
    "critical": "CRITICAL",
    "unsafe": "CRITICAL"
}

# =============================================================================
# ACTION-MAPPINGS
# =============================================================================

# Risk-Level zu Action-Mapping
RISK_ACTION_MAPPING: Final[dict[str, str]] = {
    "safe": "none",
    "low_risk": "log",
    "medium_risk": "warn",
    "high_risk": "sanitize",
    "critical": "block",
    "unsafe": "block"
}

# Safety-Level zu Action-Mapping
SAFETY_ACTION_MAPPING: Final[dict[str, str]] = {
    "SAFE": "none",
    "LOW_RISK": "log",
    "MEDIUM_RISK": "warn",
    "HIGH_RISK": "sanitize",
    "UNSAFE": "block"
}

# =============================================================================
# REGEX-PATTERNS
# =============================================================================

# Häufig verwendete Regex-Patterns
EMAIL_PATTERN: Final[str] = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_PATTERN: Final[str] = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
SSN_PATTERN: Final[str] = r"\b\d{3}-\d{2}-\d{4}\b"
CREDIT_CARD_PATTERN: Final[str] = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"

# =============================================================================
# FEATURE-FLAGS
# =============================================================================

# Feature-Toggles
ENABLE_CACHING: Final[bool] = True
ENABLE_CIRCUIT_BREAKER: Final[bool] = True
ENABLE_PERFORMANCE_MONITORING: Final[bool] = True
ENABLE_DETAILED_LOGGING: Final[bool] = False
ENABLE_ASYNC_PROCESSING: Final[bool] = True

# =============================================================================
# VALIDATION-KONFIGURATION
# =============================================================================

# Validierungs-Limits
MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.0
MAX_CONFIDENCE_THRESHOLD: Final[float] = 1.0
MIN_RISK_THRESHOLD: Final[float] = 0.0
MAX_RISK_THRESHOLD: Final[float] = 1.0

# =============================================================================
# EXPORT-LISTE
# =============================================================================

__all__ = [
    "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS",
    "CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
    "CONFIDENCE_CERTAIN",
    "CONFIDENCE_HIGH",
    # Detection-Schwellwerte
    "CONFIDENCE_LOW",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_VERY_HIGH",
    "CREDIT_CARD_PATTERN",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_SIZE",
    # Cache-Konfiguration
    "DEFAULT_CACHE_TTL",
    "DEFAULT_ENCRYPT_MODULO",
    "DEFAULT_HASH_LENGTH",
    # Redaction-Konfiguration
    "DEFAULT_MASK",
    # Performance-Konfiguration
    "DEFAULT_PROCESSING_TIMEOUT",
    "DEFAULT_SAMPLING_RATE",
    # Data-Minimization-Konfiguration
    "DEFAULT_STRATA_COUNT",
    "DEFAULT_TOKEN_MODULO",
    # Regex-Patterns
    "EMAIL_PATTERN",
    "ENABLE_ASYNC_PROCESSING",
    # Feature-Flags
    "ENABLE_CACHING",
    "ENABLE_CIRCUIT_BREAKER",
    "ENABLE_DETAILED_LOGGING",
    "ENABLE_PERFORMANCE_MONITORING",
    "ENCRYPT_FORMAT_WIDTH",
    "ENCRYPT_PREFIX",
    "FAST_PROCESSING_TIMEOUT",
    "HASH_PREFIX",
    "LARGE_TOKEN_MODULO",
    # Mappings
    "LOG_LEVEL_MAPPING",
    "LONG_HASH_LENGTH",
    "MAX_BATCH_SIZE",
    "MAX_CACHE_SIZE",
    "MAX_CONFIDENCE_THRESHOLD",
    # Content-Limits
    "MAX_CONTENT_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_KEYWORDS_PER_CATEGORY",
    "MAX_PATTERNS_PER_CATEGORY",
    "MAX_RISK_THRESHOLD",
    "MAX_SAMPLING_RATE",
    "MAX_SNIPPET_LENGTH",
    "MAX_STRATA_COUNT",
    "MIN_BATCH_SIZE",
    # Validation
    "MIN_CONFIDENCE_THRESHOLD",
    "MIN_RISK_THRESHOLD",
    "MIN_SAMPLING_RATE",
    "MIN_STRATA_COUNT",
    "NEUTRALIZED_PREFIX",
    "PERFORMANCE_ERROR_THRESHOLD",
    "PERFORMANCE_WARNING_THRESHOLD",
    "PHONE_PATTERN",
    "POLICY_CACHE_TTL",
    # Policy-Engine-Konfiguration
    "POLICY_DECISION_ALLOW",
    "POLICY_DECISION_DENY",
    "POLICY_DECISION_UNKNOWN",
    "REDACTION_CACHE_TTL",
    "RISK_ACTION_MAPPING",
    "RISK_THRESHOLD_CRITICAL",
    "RISK_THRESHOLD_HIGH",
    "RISK_THRESHOLD_LOW",
    "RISK_THRESHOLD_MEDIUM",
    "SAFETY_ACTION_MAPPING",
    "SHORT_HASH_LENGTH",
    "SLOW_PROCESSING_TIMEOUT",
    "SSN_PATTERN",
    "TOKEN_FORMAT_WIDTH",
    "TOKEN_PREFIX",
    "TOOL_ACCESS_CACHE_TTL",
    "TOXICITY_THRESHOLD_HIGH",
    "TOXICITY_THRESHOLD_LOW",
    "TOXICITY_THRESHOLD_MEDIUM",
    "TOXICITY_THRESHOLD_UNSAFE"
]
