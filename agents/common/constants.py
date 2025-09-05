# backend/agents/common/constants.py
"""Gemeinsame Konstanten für Resilience-Module

Konsolidiert alle Magic Numbers und Hard-coded Strings in eine zentrale
Konstanten-Datei für bessere Wartbarkeit und Konsistenz.
"""

# Enum wird nicht verwendet - alle Konstanten sind als Klassen-Attribute definiert

# Circuit Breaker-Konstanten
class CircuitBreakerDefaults:
    """Standard-Werte für Circuit Breaker-Konfiguration."""

    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 60.0  # Sekunden
    SUCCESS_THRESHOLD = 3
    TIMEOUT = 30.0  # Sekunden
    SLIDING_WINDOW_SIZE = 100
    SLIDING_WINDOW_DURATION = 300.0  # 5 Minuten
    FAILURE_RATE_THRESHOLD = 0.5  # 50%
    MINIMUM_THROUGHPUT = 10
    HALF_OPEN_MAX_CALLS = 3


# Retry-Konstanten
class RetryDefaults:
    """Standard-Werte für Retry-Konfiguration."""

    MAX_ATTEMPTS = 3
    BASE_DELAY = 1.0  # Sekunden
    MAX_DELAY = 60.0  # Sekunden
    EXPONENTIAL_BASE = 2.0
    JITTER_RANGE = 0.1  # 10%
    PERFORMANCE_WINDOW_SIZE = 100
    PERFORMANCE_THRESHOLD_MS = 1000.0  # 1 Sekunde

    # Standard HTTP-Status-Codes für Retry
    RETRYABLE_HTTP_STATUS_CODES = [429, 502, 503, 504]


# Budget-Konstanten
class BudgetDefaults:
    """Standard-Werte für Budget-Konfiguration."""

    DEFAULT_TIMEOUT = 30.0  # Sekunden
    MAX_CPU_TIME_MS = 5000.0  # 5 Sekunden
    MAX_MEMORY_MB = 512.0  # 512 MB
    MAX_NETWORK_CALLS = 10
    BUDGET_WARNING_THRESHOLD = 0.8  # 80%
    BUDGET_CRITICAL_THRESHOLD = 0.95  # 95%


# Performance-Monitoring-Konstanten
class MonitoringDefaults:
    """Standard-Werte für Performance-Monitoring."""

    MONITORING_INTERVAL = 10.0  # Sekunden
    METRICS_RETENTION_SIZE = 1000
    ALERT_COOLDOWN = 300.0  # 5 Minuten
    ERROR_RATE_THRESHOLD = 0.05  # 5%
    RESPONSE_TIME_THRESHOLD = 5.0  # 5 Sekunden
    CIRCUIT_BREAKER_TRIPS_THRESHOLD = 3
    BUDGET_EXHAUSTIONS_THRESHOLD = 5


# Deadline-Konstanten
class DeadlineDefaults:
    """Standard-Werte für Deadline-Management."""

    DEFAULT_REQUEST_TIMEOUT = 30.0  # Sekunden
    DEFAULT_OPERATION_TIMEOUT = 60.0  # Sekunden
    DEFAULT_BATCH_TIMEOUT = 300.0  # 5 Minuten
    DEADLINE_WARNING_THRESHOLD = 0.8  # 80% der Zeit verbraucht
    DEADLINE_EXTENSION_FACTOR = 1.5  # 50% Verlängerung


# Resilience-Policy-Konstanten
class ResiliencePolicyDefaults:
    """Standard-Werte für Resilience-Policies."""

    CIRCUIT_BREAKER_ENABLED = True
    RETRY_ENABLED = True
    BUDGET_TRACKING_ENABLED = True
    PERFORMANCE_MONITORING_ENABLED = True
    ALERTING_ENABLED = True


# Logging-Konstanten
class LoggingConstants:
    """Konstanten für Logging."""

    # Log-Level
    DEFAULT_LOG_LEVEL = "INFO"
    DEBUG_LOG_LEVEL = "DEBUG"
    ERROR_LOG_LEVEL = "ERROR"

    # Log-Nachrichten
    CIRCUIT_BREAKER_OPENED = "Circuit breaker opened"
    CIRCUIT_BREAKER_CLOSED = "Circuit breaker closed"
    CIRCUIT_BREAKER_HALF_OPEN = "Circuit breaker half-open"
    RETRY_ATTEMPT = "Retry attempt"
    RETRY_EXHAUSTED = "Retry attempts exhausted"
    BUDGET_EXCEEDED = "Budget exceeded"
    DEADLINE_EXCEEDED = "Deadline exceeded"
    ALERT_CREATED = "Alert created"
    ALERT_RESOLVED = "Alert resolved"


# Metrics-Konstanten
class MetricsConstants:
    """Konstanten für Metrics-Collection."""

    # Metric-Namen
    CIRCUIT_BREAKER_STATE_CHANGES = "circuit_breaker.state_changes"
    CIRCUIT_BREAKER_CALLS_REJECTED = "circuit_breaker.calls_rejected"
    CIRCUIT_BREAKER_RESPONSE_TIME = "circuit_breaker.response_time"
    CIRCUIT_BREAKER_FAILURES = "circuit_breaker.failures"

    RETRY_ATTEMPTS = "retry.attempts"
    RETRY_RESPONSE_TIME = "retry.response_time"
    RETRY_EXHAUSTED = "retry.exhausted"

    BUDGET_USAGE = "budget.usage"
    BUDGET_EXCEEDED = "budget.exceeded"

    PERFORMANCE_RESPONSE_TIME = "performance.response_time"
    PERFORMANCE_SUCCESS_RATE = "performance.success_rate"
    PERFORMANCE_REQUESTS_PER_SECOND = "performance.requests_per_second"

    ALERTS_CREATED = "alerts.created"
    ALERTS_RESOLVED = "alerts.resolved"

    # Metric-Tags
    AGENT_ID_TAG = "agent_id"
    CAPABILITY_TAG = "capability"
    STATE_TAG = "state"
    OPERATION_TAG = "operation"
    ATTEMPT_TAG = "attempt"
    EXCEPTION_TYPE_TAG = "exception_type"
    SEVERITY_TAG = "severity"
    METRIC_NAME_TAG = "metric_name"
    IDENTIFIER_TAG = "identifier"
    SUCCESS_TAG = "success"


# Error-Nachrichten
class ErrorMessages:
    """Standard-Fehlermeldungen."""

    # Circuit Breaker-Fehler
    CIRCUIT_BREAKER_OPEN = "Circuit breaker is open"
    CIRCUIT_BREAKER_TIMEOUT = "Circuit breaker timeout"
    CIRCUIT_BREAKER_INVALID_CONFIG = "Invalid circuit breaker configuration"

    # Retry-Fehler
    RETRY_EXHAUSTED = "Retry attempts exhausted"
    RETRY_INVALID_CONFIG = "Invalid retry configuration"

    # Budget-Fehler
    BUDGET_EXHAUSTED = "Budget exhausted"
    DEADLINE_EXCEEDED = "Deadline exceeded"
    BUDGET_INVALID_CONFIG = "Invalid budget configuration"

    # Allgemeine Fehler
    INVALID_PARAMETER = "Invalid parameter"
    OPERATION_FAILED = "Operation failed"
    CONFIGURATION_ERROR = "Configuration error"


# Capability-spezifische Konstanten
class CapabilityConstants:
    """Konstanten für verschiedene Capabilities."""

    # Standard-Capabilities
    CHAT = "chat"
    SEARCH = "search"
    API_CALL = "api_call"
    DB_QUERY = "db_query"
    FILE_OPERATION = "file_operation"
    NETWORK_REQUEST = "network_request"
    COMPUTATION = "computation"

    # Capability-spezifische Timeouts (in Sekunden)
    CAPABILITY_TIMEOUTS = {
        CHAT: 30.0,
        SEARCH: 60.0,
        API_CALL: 45.0,
        DB_QUERY: 30.0,
        FILE_OPERATION: 120.0,
        NETWORK_REQUEST: 60.0,
        COMPUTATION: 300.0,
    }

    # Capability-spezifische Failure-Thresholds
    CAPABILITY_FAILURE_THRESHOLDS = {
        CHAT: 3,
        SEARCH: 5,
        API_CALL: 4,
        DB_QUERY: 3,
        FILE_OPERATION: 2,
        NETWORK_REQUEST: 4,
        COMPUTATION: 2,
    }


# Environment-spezifische Konstanten
class EnvironmentConstants:
    """Konstanten für verschiedene Umgebungen."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    # Environment-spezifische Multipliers
    TIMEOUT_MULTIPLIERS = {
        DEVELOPMENT: 2.0,  # Längere Timeouts für Development
        TESTING: 0.5,     # Kürzere Timeouts für Tests
        STAGING: 1.0,     # Standard-Timeouts
        PRODUCTION: 1.0,  # Standard-Timeouts
    }

    # Environment-spezifische Monitoring-Intervalle
    MONITORING_INTERVALS = {
        DEVELOPMENT: 30.0,  # Weniger häufiges Monitoring
        TESTING: 1.0,       # Häufiges Monitoring für Tests
        STAGING: 10.0,      # Standard-Monitoring
        PRODUCTION: 5.0,    # Häufigeres Monitoring in Production
    }


# Validation-Konstanten
class ValidationConstants:
    """Konstanten für Validierung."""

    # Minimum/Maximum-Werte
    MIN_FAILURE_THRESHOLD = 1
    MAX_FAILURE_THRESHOLD = 100
    MIN_RECOVERY_TIMEOUT = 1.0
    MAX_RECOVERY_TIMEOUT = 3600.0  # 1 Stunde
    MIN_MAX_ATTEMPTS = 1
    MAX_MAX_ATTEMPTS = 20
    MIN_BASE_DELAY = 0.0
    MAX_BASE_DELAY = 300.0  # 5 Minuten
    MIN_TIMEOUT = 0.1
    MAX_TIMEOUT = 3600.0  # 1 Stunde

    # Regex-Patterns
    AGENT_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
    CAPABILITY_PATTERN = r"^[a-zA-Z0-9_-]+$"
    METRIC_NAME_PATTERN = r"^[a-zA-Z0-9._-]+$"


# Feature-Flags
class FeatureFlags:
    """Feature-Flags für experimentelle Features."""

    ADAPTIVE_RETRY_ENABLED = True
    ADVANCED_CIRCUIT_BREAKER_ENABLED = True
    DETAILED_METRICS_ENABLED = True
    ALERT_AGGREGATION_ENABLED = True
    PERFORMANCE_OPTIMIZATION_ENABLED = True
    EXPERIMENTAL_FEATURES_ENABLED = False


# Version-Informationen
class VersionInfo:
    """Version-Informationen für Resilience-Module."""

    MAJOR_VERSION = 1
    MINOR_VERSION = 0
    PATCH_VERSION = 0
    VERSION_STRING = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"

    # Kompatibilitäts-Informationen
    MIN_PYTHON_VERSION = "3.11"
    SUPPORTED_PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]
