# backend/agents/slo_sla/constants.py
"""Zentrale Konstanten für das SLO/SLA Management-System.

Definiert alle Magic Numbers, Default-Werte und Konfigurationskonstanten
für konsistente Verwendung im gesamten SLO/SLA-Modul.
"""

from enum import Enum
from typing import Final

# =============================================================================
# HISTORY UND BUFFER LIMITS
# =============================================================================

# Standard-Limits für History-Management
DEFAULT_HISTORY_LIMIT: Final[int] = 1000
"""Standard-Limit für History-Listen (Breaches, Notifications, etc.)"""

MEASUREMENTS_HISTORY_LIMIT: Final[int] = 10000
"""Limit für SLO-Measurements-History"""

CONSUMPTION_HISTORY_LIMIT: Final[int] = 1000
"""Limit für Error-Budget-Consumption-History"""

RECOMMENDATION_HISTORY_LIMIT: Final[int] = 1000
"""Limit für Scaling-Recommendations-History"""

# =============================================================================
# ZEIT-KONSTANTEN
# =============================================================================

# Basis-Zeiteinheiten
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600
SECONDS_PER_DAY: Final[int] = 86400
MINUTES_PER_HOUR: Final[int] = 60
HOURS_PER_DAY: Final[int] = 24

# Default-Zeitfenster
DEFAULT_MONITORING_INTERVAL_SECONDS: Final[float] = 30.0
"""Standard-Monitoring-Intervall in Sekunden"""

DEFAULT_CAPACITY_PLANNING_INTERVAL_MINUTES: Final[float] = 60.0
"""Standard-Capacity-Planning-Intervall in Minuten"""

DEFAULT_ERROR_RATE_WINDOW_SECONDS: Final[float] = 900.0
"""Standard-Zeitfenster für Error-Rate-Tracking (15 Minuten)"""

DEFAULT_PERFORMANCE_BUDGET_WINDOW_HOURS: Final[float] = 24.0
"""Standard-Zeitfenster für Performance-Budgets (24 Stunden)"""

DEFAULT_GRACE_PERIOD_SECONDS: Final[float] = 60.0
"""Standard-Grace-Period für SLO-Violations"""

# =============================================================================
# SLO/SLA THRESHOLDS UND TARGETS
# =============================================================================

# Standard-SLO-Targets
DEFAULT_SLO_TARGET_PERCENTAGE: Final[float] = 99.0
"""Standard-SLO-Target (99%)"""

DEFAULT_SLA_TARGET_PERCENTAGE: Final[float] = 99.5
"""Standard-SLA-Target (99.5%)"""

# Error-Rate-Thresholds
DEFAULT_ERROR_RATE_THRESHOLD: Final[float] = 0.05
"""Standard-Error-Rate-Threshold (5%)"""

LOW_ERROR_RATE_THRESHOLD: Final[float] = 0.01
"""Niedriger Error-Rate-Threshold (1%)"""

HIGH_ERROR_RATE_THRESHOLD: Final[float] = 0.10
"""Hoher Error-Rate-Threshold (10%)"""

# Latency-Thresholds
DEFAULT_LATENCY_P95_MS: Final[float] = 500.0
"""Standard-P95-Latency-Threshold (500ms)"""

DEFAULT_LATENCY_P99_MS: Final[float] = 1000.0
"""Standard-P99-Latency-Threshold (1000ms)"""

DEFAULT_RESPONSE_TIME_THRESHOLD_SECONDS: Final[float] = 5.0
"""Standard-Response-Time-Threshold (5 Sekunden)"""

# Availability-Thresholds
DEFAULT_AVAILABILITY_THRESHOLD: Final[float] = 0.999
"""Standard-Availability-Threshold (99.9%)"""

HIGH_AVAILABILITY_THRESHOLD: Final[float] = 0.9999
"""Hoher Availability-Threshold (99.99%)"""

# =============================================================================
# ERROR BUDGET KONSTANTEN
# =============================================================================

# Error-Budget-Burn-Rate-Thresholds
DEFAULT_ERROR_BUDGET_BURN_RATE_THRESHOLD: Final[float] = 2.0
"""Standard-Error-Budget-Burn-Rate-Threshold (2x normal)"""

CRITICAL_ERROR_BUDGET_BURN_RATE_THRESHOLD: Final[float] = 5.0
"""Kritischer Error-Budget-Burn-Rate-Threshold (5x normal)"""

# Error-Budget-Alert-Thresholds
ERROR_BUDGET_ALERT_THRESHOLDS: Final[list[float]] = [0.5, 0.8, 0.9]
"""Standard-Alert-Thresholds für Error-Budget-Consumption"""

# =============================================================================
# ALERT UND ESCALATION KONSTANTEN
# =============================================================================

# Circuit-Breaker-Thresholds
DEFAULT_CIRCUIT_BREAKER_TRIPS_THRESHOLD: Final[int] = 3
"""Standard-Threshold für Circuit-Breaker-Trips"""

DEFAULT_BUDGET_EXHAUSTIONS_THRESHOLD: Final[int] = 5
"""Standard-Threshold für Budget-Exhaustions"""

# Escalation-Timeouts
DEFAULT_ESCALATION_TIMEOUT_MINUTES: Final[int] = 15
"""Standard-Timeout für Escalation-Workflows (15 Minuten)"""

CRITICAL_ESCALATION_TIMEOUT_MINUTES: Final[int] = 5
"""Kritischer Timeout für Escalation-Workflows (5 Minuten)"""

# =============================================================================
# CAPACITY PLANNING KONSTANTEN
# =============================================================================

# Trend-Analysis
MINIMUM_DATA_POINTS_FOR_TREND: Final[int] = 14
"""Minimum Datenpunkte für Trend-Analyse (2 Wochen)"""

TREND_ANALYSIS_CONFIDENCE_THRESHOLD: Final[float] = 0.6
"""Confidence-Threshold für Trend-Pattern-Detection"""

# Scaling-Thresholds
DEFAULT_SCALE_UP_THRESHOLD: Final[float] = 0.8
"""Standard-Threshold für Scale-Up-Empfehlungen (80%)"""

DEFAULT_SCALE_DOWN_THRESHOLD: Final[float] = 0.3
"""Standard-Threshold für Scale-Down-Empfehlungen (30%)"""

# Performance-Budget-Defaults
DEFAULT_MAX_RESPONSE_TIME_P95_SECONDS: Final[float] = 0.5
"""Standard-Max-Response-Time für Performance-Budgets (500ms)"""

DEFAULT_MAX_ERROR_RATE_PERCENT: Final[float] = 1.0
"""Standard-Max-Error-Rate für Performance-Budgets (1%)"""

DEFAULT_MIN_AVAILABILITY_PERCENT: Final[float] = 99.9
"""Standard-Min-Availability für Performance-Budgets (99.9%)"""

# =============================================================================
# MONITORING KONSTANTEN
# =============================================================================

# Percentile-Berechnungen
SUPPORTED_PERCENTILES: Final[list[float]] = [50.0, 90.0, 95.0, 99.0, 99.9]
"""Unterstützte Percentile für Latency-Berechnungen"""

# Sliding-Window-Konfiguration
DEFAULT_SLIDING_WINDOW_SIZE: Final[int] = 100
"""Standard-Größe für Sliding-Windows"""

LARGE_SLIDING_WINDOW_SIZE: Final[int] = 1000
"""Große Sliding-Window-Größe für detaillierte Analysen"""

# =============================================================================
# VALIDATION KONSTANTEN
# =============================================================================

# Percentage-Validierung
MIN_PERCENTAGE: Final[float] = 0.0
"""Minimum-Wert für Percentage-Validierung"""

MAX_PERCENTAGE: Final[float] = 100.0
"""Maximum-Wert für Percentage-Validierung"""

# String-Längen-Limits
MAX_NAME_LENGTH: Final[int] = 255
"""Maximum-Länge für Namen (SLO/SLA-Namen, etc.)"""

MAX_DESCRIPTION_LENGTH: Final[int] = 1000
"""Maximum-Länge für Beschreibungen"""

MAX_TAG_KEY_LENGTH: Final[int] = 50
"""Maximum-Länge für Tag-Keys"""

MAX_TAG_VALUE_LENGTH: Final[int] = 200
"""Maximum-Länge für Tag-Values"""

# =============================================================================
# PATTERN DETECTION KONSTANTEN
# =============================================================================

# Seasonal-Pattern-Detection
WEEKLY_PATTERN_VARIANCE_THRESHOLD: Final[float] = 10.0
"""Variance-Threshold für Weekly-Pattern-Detection"""

PATTERN_STRENGTH_NORMALIZATION_FACTOR: Final[float] = 10.0
"""Normalisierungsfaktor für Pattern-Strength-Berechnung"""

# =============================================================================
# ENUM-BASIERTE KONSTANTEN
# =============================================================================

class DefaultTimeWindows(Enum):
    """Standard-Zeitfenster für SLO/SLA-Definitionen."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class AlertSeverityLevels(Enum):
    """Standard-Alert-Severity-Level."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ScalingDirections(Enum):
    """Scaling-Richtungen für Capacity-Planning."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


# =============================================================================
# UTILITY KONSTANTEN
# =============================================================================

# Logging
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
"""Standard-Log-Level für SLO/SLA-Module"""

# Retry-Konfiguration
DEFAULT_MAX_RETRIES: Final[int] = 3
"""Standard-Maximum-Retries für fehlerhafte Operationen"""

DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0
"""Standard-Delay zwischen Retry-Versuchen"""

# Batch-Processing
DEFAULT_BATCH_SIZE: Final[int] = 100
"""Standard-Batch-Größe für Bulk-Operationen"""

LARGE_BATCH_SIZE: Final[int] = 1000
"""Große Batch-Größe für Performance-kritische Operationen"""
