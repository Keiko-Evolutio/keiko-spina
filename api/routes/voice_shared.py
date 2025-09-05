# backend/api/routes/voice_shared.py
"""Gemeinsame Komponenten für Voice-System Module
Konsolidierte Enums, Models und Utilities für Health, Monitoring und Recovery.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from kei_logging import get_logger

logger = get_logger(__name__)

# =====================================================================
# Gemeinsame Enums
# =====================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Voice system component types."""
    WEBSOCKET_ENDPOINT = "websocket_endpoint"
    AZURE_CONNECTIVITY = "azure_connectivity"
    AUDIO_PIPELINE = "audio_pipeline"
    SESSION_MANAGEMENT = "session_management"
    ERROR_HANDLING = "error_handling"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    RECONNECT = "reconnect"
    RESET_SESSION = "reset_session"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


# =====================================================================
# Gemeinsame Dataclasses
# =====================================================================

@dataclass
class HealthCheckResult:
    """Individual health check result."""
    component: ComponentType
    status: HealthStatus
    message: str
    details: dict[str, Any]
    timestamp: datetime
    duration_ms: float
    error: str | None = None


@dataclass
class ErrorEvent:
    """Error event record."""
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    message: str
    user_id: str | None = None
    component: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 30.0
    recovery_actions: list[RecoveryAction] = field(default_factory=list)


# =====================================================================
# Gemeinsame Pydantic Models
# =====================================================================

class VoiceSystemHealth(BaseModel):
    """Complete voice system health status."""
    overall_status: HealthStatus
    components: list[HealthCheckResult]
    startup_validation_passed: bool
    last_check: datetime
    uptime_seconds: float
    error_count: int
    warning_count: int


class VoiceMetrics(BaseModel):
    """Voice system performance metrics."""
    active_sessions: int
    azure_connections: int
    average_response_time_ms: float
    error_rate_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_latency_ms: float
    session_success_rate: float
    average_session_duration_seconds: float
    timestamp: datetime


class AlertLevel(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =====================================================================
# Gemeinsame Utility-Funktionen
# =====================================================================

def create_health_check_result(
    component: ComponentType,
    status: HealthStatus,
    message: str,
    details: dict[str, Any],
    start_time: float,
    error: str | None = None
) -> HealthCheckResult:
    """Helper-Funktion zur Erstellung von HealthCheckResult."""
    return HealthCheckResult(
        component=component,
        status=status,
        message=message,
        details=details,
        timestamp=datetime.now(UTC),
        duration_ms=(time.time() - start_time) * 1000,
        error=error
    )


def determine_overall_status(results: list[HealthCheckResult]) -> HealthStatus:
    """Bestimme Gesamt-Health-Status aus Komponenten-Ergebnissen."""
    if any(r.status == HealthStatus.FAILED for r in results):
        return HealthStatus.FAILED
    if any(r.status == HealthStatus.CRITICAL for r in results):
        return HealthStatus.CRITICAL
    if any(r.status == HealthStatus.WARNING for r in results):
        return HealthStatus.WARNING
    return HealthStatus.HEALTHY


def determine_error_severity(_error_type: str, error: Exception) -> ErrorSeverity:
    """Bestimme Error-Severity basierend auf Typ und Inhalt."""
    error_str = str(error).lower()

    # Critical errors
    if any(keyword in error_str for keyword in ["authentication", "authorization", "security"]):
        return ErrorSeverity.CRITICAL

    # High severity errors
    if any(keyword in error_str for keyword in ["connection", "network", "timeout"]):
        return ErrorSeverity.HIGH

    # Medium severity errors
    if any(keyword in error_str for keyword in ["audio", "processing", "format"]):
        return ErrorSeverity.MEDIUM

    # Default to low severity
    return ErrorSeverity.LOW


def log_error_event(error_event: ErrorEvent):
    """Log error event mit entsprechendem Level."""
    if error_event.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"🚨 CRITICAL: {error_event.error_type} - {error_event.message}")
    elif error_event.severity == ErrorSeverity.HIGH:
        logger.error(f"🔴 HIGH: {error_event.error_type} - {error_event.message}")
    elif error_event.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"🟡 MEDIUM: {error_event.error_type} - {error_event.message}")
    else:
        logger.info(f"🔵 LOW: {error_event.error_type} - {error_event.message}")


# =====================================================================
# Performance Thresholds
# =====================================================================

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "response_time_ms": 1000.0,
    "error_rate_percent": 5.0,
    "memory_usage_mb": 500.0,
    "session_success_rate": 95.0,
    "max_active_connections": 100,
    "max_latency_ms": 200.0,
    "min_recovery_rate": 80.0
}


# =====================================================================
# Vereinfachte Recovery Strategies
# =====================================================================

# Vereinfachte Standard-Recovery-Strategie
DEFAULT_RECOVERY_STRATEGY = RecoveryStrategy(
    max_retries=3,
    retry_delay=1.0,
    backoff_multiplier=2.0,
    timeout_seconds=30.0,
    recovery_actions=[RecoveryAction.RETRY, RecoveryAction.RECONNECT]
)
