# backend/quotas_limits/constants.py
"""Konstanten für das Quotas/Limits System.

Zentrale Sammlung aller Magic Numbers und Konfigurationswerte
für bessere Wartbarkeit und Konsistenz.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

# Cache-Konfiguration
CACHE_TTL_SECONDS: Final[int] = 60
CACHE_TTL_LONG_SECONDS: Final[int] = 300
CACHE_MAX_SIZE: Final[int] = 10000

# Monitoring-Intervalle
MONITORING_INTERVAL_SECONDS: Final[int] = 60
ANALYTICS_INTERVAL_SECONDS: Final[int] = 3600
HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 30

# Rate-Limiting-Defaults
DEFAULT_REQUESTS_PER_SECOND: Final[float] = 10.0
DEFAULT_REQUESTS_PER_MINUTE: Final[float] = 600.0
DEFAULT_REQUESTS_PER_HOUR: Final[float] = 36000.0
DEFAULT_REQUESTS_PER_DAY: Final[float] = 864000.0

# Budget-Defaults
DEFAULT_BASE_COST: Final[Decimal] = Decimal("0.001")
DEFAULT_DATA_COST_PER_KB: Final[Decimal] = Decimal("0.0001")
DEFAULT_BUDGET_ALERT_THRESHOLD: Final[float] = 0.8

# Quota-Defaults
DEFAULT_QUOTA_LIMIT: Final[float] = 1000.0
DEFAULT_QUOTA_WINDOW_SECONDS: Final[int] = 3600

# Performance-Limits
MAX_FUNCTION_LINES: Final[int] = 20
MAX_CYCLOMATIC_COMPLEXITY: Final[int] = 10
MIN_TEST_COVERAGE_PERCENT: Final[int] = 85

# Retry-Konfiguration
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0
DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0

# Timeout-Konfiguration
DEFAULT_OPERATION_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_NETWORK_TIMEOUT_SECONDS: Final[int] = 10
DEFAULT_DATABASE_TIMEOUT_SECONDS: Final[int] = 5

# Batch-Processing
DEFAULT_BATCH_SIZE: Final[int] = 100
MAX_BATCH_SIZE: Final[int] = 1000

# Alert-Konfiguration
DEFAULT_COOLDOWN_MINUTES: Final[int] = 30
DEFAULT_CONSECUTIVE_VIOLATIONS: Final[int] = 1
DEFAULT_EVALUATION_WINDOW_MINUTES: Final[int] = 5

# Scaling-Faktoren
DEFAULT_SCALE_FACTOR: Final[float] = 1.2
MAX_SCALE_FACTOR: Final[float] = 5.0
MIN_SCALE_FACTOR: Final[float] = 0.1

# Datenstrukturen-Limits
MAX_DEQUE_SIZE: Final[int] = 10000
MAX_HISTORY_ENTRIES: Final[int] = 1000
MAX_CACHE_ENTRIES: Final[int] = 5000

# HTTP-Status-Codes
HTTP_TOO_MANY_REQUESTS: Final[int] = 429
HTTP_PAYMENT_REQUIRED: Final[int] = 402
HTTP_FORBIDDEN: Final[int] = 403

# Logging-Konfiguration
LOG_LEVEL_DEBUG: Final[str] = "DEBUG"
LOG_LEVEL_INFO: Final[str] = "INFO"
LOG_LEVEL_WARNING: Final[str] = "WARNING"
LOG_LEVEL_ERROR: Final[str] = "ERROR"

# Metrik-Namen
METRIC_QUOTA_CHECKS: Final[str] = "quota_checks_total"
METRIC_RATE_LIMIT_VIOLATIONS: Final[str] = "rate_limit_violations_total"
METRIC_BUDGET_EXHAUSTIONS: Final[str] = "budget_exhaustions_total"
METRIC_PROCESSING_TIME: Final[str] = "processing_time_seconds"

# Pattern-Detection
MIN_PATTERN_OCCURRENCES: Final[int] = 3
PATTERN_DETECTION_WINDOW_HOURS: Final[int] = 24
ANOMALY_DETECTION_THRESHOLD: Final[float] = 2.0

# Circuit-Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS: Final[int] = 3

# Predictive-Analytics
PREDICTION_HORIZON_HOURS: Final[int] = 24
MIN_DATA_POINTS_FOR_PREDICTION: Final[int] = 10
PREDICTION_CONFIDENCE_THRESHOLD: Final[float] = 0.7

# Quota-Typen
QUOTA_TYPE_REQUEST_RATE: Final[str] = "request_rate"
QUOTA_TYPE_OPERATION_COUNT: Final[str] = "operation_count"
QUOTA_TYPE_DATA_VOLUME: Final[str] = "data_volume"
QUOTA_TYPE_COMPUTE_TIME: Final[str] = "compute_time"
QUOTA_TYPE_MEMORY_USAGE: Final[str] = "memory_usage"
QUOTA_TYPE_STORAGE_QUOTA: Final[str] = "storage_quota"
QUOTA_TYPE_BANDWIDTH_LIMIT: Final[str] = "bandwidth_limit"
QUOTA_TYPE_CONCURRENT_OPERATIONS: Final[str] = "concurrent_operations"

# Budget-Typen
BUDGET_TYPE_API_CALLS: Final[str] = "api_calls"
BUDGET_TYPE_COMPUTE_RESOURCES: Final[str] = "compute_resources"
BUDGET_TYPE_STORAGE: Final[str] = "storage"
BUDGET_TYPE_BANDWIDTH: Final[str] = "bandwidth"

# Scope-Typen
SCOPE_TYPE_GLOBAL: Final[str] = "global"
SCOPE_TYPE_TENANT: Final[str] = "tenant"
SCOPE_TYPE_AGENT: Final[str] = "agent"
SCOPE_TYPE_CAPABILITY: Final[str] = "capability"
SCOPE_TYPE_USER: Final[str] = "user"

# Alert-Severity-Levels
ALERT_SEVERITY_INFO: Final[str] = "info"
ALERT_SEVERITY_WARNING: Final[str] = "warning"
ALERT_SEVERITY_CRITICAL: Final[str] = "critical"
ALERT_SEVERITY_EMERGENCY: Final[str] = "emergency"

# Regex-Patterns
PATTERN_AGENT_ID: Final[str] = r"^agent_[a-zA-Z0-9_-]+$"
PATTERN_QUOTA_ID: Final[str] = r"^quota_[a-zA-Z0-9_-]+$"
PATTERN_BUDGET_ID: Final[str] = r"^budget_[a-zA-Z0-9_-]+$"
PATTERN_UUID: Final[str] = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

# Validierungs-Limits
MIN_QUOTA_LIMIT: Final[float] = 0.1
MAX_QUOTA_LIMIT: Final[float] = 1_000_000.0
MIN_BUDGET_AMOUNT: Final[Decimal] = Decimal("0.01")
MAX_BUDGET_AMOUNT: Final[Decimal] = Decimal("1000000.00")

# Performance-Benchmarks
TARGET_QUOTA_CHECK_LATENCY_MS: Final[float] = 10.0
TARGET_RATE_LIMIT_CHECK_LATENCY_MS: Final[float] = 5.0
TARGET_BUDGET_OPERATION_LATENCY_MS: Final[float] = 20.0

# Cleanup-Konfiguration
CLEANUP_INTERVAL_HOURS: Final[int] = 24
MAX_HISTORY_AGE_DAYS: Final[int] = 30
MAX_LOG_AGE_DAYS: Final[int] = 7

# Feature-Flags
FEATURE_PREDICTIVE_ANALYTICS: Final[bool] = True
FEATURE_AUTO_SCALING: Final[bool] = True
FEATURE_CIRCUIT_BREAKER: Final[bool] = True
FEATURE_CACHING: Final[bool] = True
FEATURE_MONITORING: Final[bool] = True
