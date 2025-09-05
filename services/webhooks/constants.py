"""Zentrale Konstanten für das KEI-Webhook System.

Sammelt alle Magic Numbers, Timeouts, TTLs und Limits an einem Ort für
bessere Wartbarkeit und Konfigurierbarkeit.
"""

from __future__ import annotations

from typing import Final

from config.settings import settings

# =============================================================================
# Timeout-Konstanten
# =============================================================================

# Signatur-Validierung
SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS: Final[int] = getattr(
    settings, "webhook_signature_tolerance_seconds", 300
)

# HTTP-Timeouts
HTTP_REQUEST_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_http_timeout_seconds", 30.0
)
HTTP_CONNECT_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_connect_timeout_seconds", 5.0
)

# Worker-Timeouts
WORKER_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_worker_shutdown_timeout", 10.0
)
WORKER_POLL_INTERVAL_SECONDS: Final[float] = getattr(
    settings, "webhook_worker_poll_interval", 1.0
)

# Health-Check-Timeouts
HEALTH_CHECK_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_health_check_timeout", 5.0
)
HEALTH_CHECK_INTERVAL_SECONDS: Final[float] = getattr(
    settings, "webhook_health_check_interval", 300.0
)

# =============================================================================
# TTL-Konstanten (Time To Live)
# =============================================================================

# Cache-TTLs
NONCE_TTL_SECONDS: Final[int] = getattr(
    settings, "webhook_nonce_ttl_seconds", 300
)
IDEMPOTENCY_TTL_SECONDS: Final[int] = getattr(
    settings, "webhook_idempotency_ttl_seconds", 600
)
DELIVERY_TRACKING_TTL_SECONDS: Final[int] = getattr(
    settings, "webhook_delivery_tracking_ttl_seconds", 30 * 24 * 3600  # 30 Tage
)

# History-Retention
EVENT_HISTORY_RETENTION_DAYS: Final[int] = getattr(
    settings, "webhook_event_history_retention_days", 7
)
AUDIT_LOG_RETENTION_DAYS: Final[int] = getattr(
    settings, "webhook_audit_log_retention_days", 90
)

# =============================================================================
# Retry- und Backoff-Konstanten
# =============================================================================

# Delivery-Retries
DEFAULT_MAX_ATTEMPTS: Final[int] = getattr(
    settings, "webhook_default_max_attempts", 3
)
DEFAULT_BACKOFF_SECONDS: Final[float] = getattr(
    settings, "webhook_default_backoff_seconds", 2.0
)
MAX_BACKOFF_SECONDS: Final[float] = getattr(
    settings, "webhook_max_backoff_seconds", 30.0
)

# Alert-Retries
ALERT_RETRY_MAX_ATTEMPTS: Final[int] = getattr(
    settings, "alert_retry_max_attempts", 3
)
ALERT_RETRY_BACKOFF_SECONDS: Final[float] = getattr(
    settings, "alert_retry_backoff_seconds", 1.0
)

# =============================================================================
# Rate-Limiting-Konstanten
# =============================================================================

# Alerting-Rate-Limits
ALERTING_RATE_LIMIT_TOKENS: Final[int] = getattr(
    settings, "alerting_rate_limit_tokens", 10
)
ALERTING_RATE_LIMIT_WINDOW_SECONDS: Final[int] = getattr(
    settings, "alerting_rate_limit_window_seconds", 60
)

# Retry-Rate-Limits
RETRY_RATE_LIMIT_WINDOW_SECONDS: Final[int] = getattr(
    settings, "webhook_retry_rate_limit_window", 300
)
RETRY_RATE_LIMIT_MAX_ATTEMPTS: Final[int] = getattr(
    settings, "webhook_retry_rate_limit_max_attempts", 5
)

# =============================================================================
# Circuit-Breaker-Konstanten
# =============================================================================

# Standard Circuit-Breaker-Konfiguration
CIRCUIT_BREAKER_WINDOW_SIZE: Final[int] = getattr(
    settings, "webhook_circuit_breaker_window_size", 10
)
CIRCUIT_BREAKER_FAILURE_RATIO: Final[float] = getattr(
    settings, "webhook_circuit_breaker_failure_ratio", 0.5
)
CIRCUIT_BREAKER_OPEN_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_circuit_breaker_open_timeout", 60.0
)
CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_circuit_breaker_half_open_timeout", 30.0
)
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS: Final[int] = getattr(
    settings, "webhook_circuit_breaker_half_open_max_calls", 3
)

# Schwellen-basierte Circuit-Breaker-Konfiguration
CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = getattr(
    settings, "webhook_circuit_breaker_failure_threshold", 5
)
CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Final[int] = getattr(
    settings, "webhook_circuit_breaker_success_threshold", 2
)
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: Final[float] = getattr(
    settings, "webhook_circuit_breaker_recovery_timeout", 300.0
)

# =============================================================================
# Worker-Pool-Konstanten
# =============================================================================

# Standard Worker-Pool-Konfiguration
DEFAULT_WORKER_COUNT: Final[int] = getattr(
    settings, "webhook_default_worker_count", 2
)
DEFAULT_QUEUE_NAME: Final[str] = getattr(
    settings, "webhook_default_queue_name", "default"
)
WORKER_SUPERVISION_INTERVAL_SECONDS: Final[float] = getattr(
    settings, "webhook_worker_supervision_interval", 30.0
)

# =============================================================================
# Pagination-Konstanten
# =============================================================================

# Standard-Pagination
DEFAULT_PAGE_SIZE: Final[int] = getattr(
    settings, "webhook_default_page_size", 50
)
MAX_PAGE_SIZE: Final[int] = getattr(
    settings, "webhook_max_page_size", 1000
)

# =============================================================================
# Secret-Management-Konstanten
# =============================================================================

# Secret-Rotation
SECRET_ROTATION_INTERVAL_HOURS: Final[int] = getattr(
    settings, "webhook_secret_rotation_interval_hours", 24 * 7  # 1 Woche
)
SECRET_GRACE_PERIOD_HOURS: Final[int] = getattr(
    settings, "webhook_secret_grace_period_hours", 24  # 1 Tag
)

# Secret-Caching
SECRET_CACHE_TTL_SECONDS: Final[int] = getattr(
    settings, "webhook_secret_cache_ttl_seconds", 300  # 5 Minuten
)

# =============================================================================
# Monitoring-Konstanten
# =============================================================================

# Prometheus-Metriken
PROMETHEUS_HISTOGRAM_BUCKETS: Final[tuple] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10, 30
)

# Custom-Metriken
METRIC_FLUSH_INTERVAL_SECONDS: Final[float] = getattr(
    settings, "webhook_metric_flush_interval", 60.0
)

# =============================================================================
# Validierungs-Konstanten
# =============================================================================

# Payload-Größen-Limits
MAX_PAYLOAD_SIZE_BYTES: Final[int] = getattr(
    settings, "webhook_max_payload_size_bytes", 1024 * 1024  # 1 MB
)
MAX_HEADER_SIZE_BYTES: Final[int] = getattr(
    settings, "webhook_max_header_size_bytes", 8192  # 8 KB
)

# URL-Validierung
MAX_URL_LENGTH: Final[int] = getattr(
    settings, "webhook_max_url_length", 2048
)

# =============================================================================
# Redis-Key-Präfixe
# =============================================================================

REDIS_KEY_PREFIX: Final[str] = "kei:webhook"
REDIS_NONCE_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:nonce"
REDIS_IDEMPOTENCY_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:idem"
REDIS_DELIVERY_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:delivery"
REDIS_TARGETS_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:targets"
REDIS_OUTBOX_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:outbox"
REDIS_DLQ_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:dlq"
REDIS_HISTORY_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}:history"
REDIS_ALERTING_DLQ_KEY: Final[str] = "kei:alerting:dlq"

# =============================================================================
# HTTP-Header-Konstanten
# =============================================================================

# Standard KEI-Webhook-Headers
KEI_SIGNATURE_HEADER: Final[str] = "x-kei-signature"
KEI_TIMESTAMP_HEADER: Final[str] = "x-kei-timestamp"
KEI_EVENT_TYPE_HEADER: Final[str] = "x-kei-event-type"
KEI_CORRELATION_ID_HEADER: Final[str] = "x-kei-correlation-id"
KEI_DELIVERY_ID_HEADER: Final[str] = "x-kei-delivery-id"

# Content-Type
JSON_CONTENT_TYPE: Final[str] = "application/json"

# =============================================================================
# Redis-Operations-Konstanten
# =============================================================================

# Redis-Scan-Parameter
REDIS_SCAN_COUNT_DEFAULT: Final[int] = 20
REDIS_SCAN_COUNT_WORK_STEALING: Final[int] = 50

# HTTP-Client-Konstanten
HTTP_CLIENT_TIMEOUT_SECONDS: Final[float] = 20.0

# =============================================================================
# Utility-Funktionen
# =============================================================================

def get_tenant_normalized(tenant_id: str | None) -> str:
    """Normalisiert Tenant-ID für Redis-Keys.

    Args:
        tenant_id: Optionale Tenant-ID

    Returns:
        Normalisierte Tenant-ID oder 'default'
    """
    if not tenant_id or not tenant_id.strip():
        return "default"
    return tenant_id.strip()


def get_redis_key(prefix: str, *parts: str | None) -> str:
    """Erstellt Redis-Key aus Präfix und Teilen.

    Args:
        prefix: Key-Präfix
        *parts: Variable Anzahl von Key-Teilen

    Returns:
        Vollständiger Redis-Key
    """
    clean_parts = [str(part) for part in parts if part is not None]
    return ":".join([prefix, *clean_parts])


__all__ = [
    # Rate-Limiting
    "ALERTING_RATE_LIMIT_TOKENS",
    "ALERTING_RATE_LIMIT_WINDOW_SECONDS",
    "ALERT_RETRY_BACKOFF_SECONDS",
    "ALERT_RETRY_MAX_ATTEMPTS",
    "AUDIT_LOG_RETENTION_DAYS",
    "CIRCUIT_BREAKER_FAILURE_RATIO",
    "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS",
    "CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT_SECONDS",
    "CIRCUIT_BREAKER_OPEN_TIMEOUT_SECONDS",
    "CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
    "CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
    # Circuit-Breaker
    "CIRCUIT_BREAKER_WINDOW_SIZE",
    "DEFAULT_BACKOFF_SECONDS",
    # Retry-Konstanten
    "DEFAULT_MAX_ATTEMPTS",
    # Pagination
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_QUEUE_NAME",
    # Worker-Pool
    "DEFAULT_WORKER_COUNT",
    "DELIVERY_TRACKING_TTL_SECONDS",
    "EVENT_HISTORY_RETENTION_DAYS",
    "HEALTH_CHECK_INTERVAL_SECONDS",
    "HEALTH_CHECK_TIMEOUT_SECONDS",
    "HTTP_CONNECT_TIMEOUT_SECONDS",
    "HTTP_REQUEST_TIMEOUT_SECONDS",
    "IDEMPOTENCY_TTL_SECONDS",
    "JSON_CONTENT_TYPE",
    "KEI_CORRELATION_ID_HEADER",
    "KEI_DELIVERY_ID_HEADER",
    "KEI_EVENT_TYPE_HEADER",
    # HTTP-Headers
    "KEI_SIGNATURE_HEADER",
    "KEI_TIMESTAMP_HEADER",
    "MAX_BACKOFF_SECONDS",
    "MAX_HEADER_SIZE_BYTES",
    "MAX_PAGE_SIZE",
    # Validierung
    "MAX_PAYLOAD_SIZE_BYTES",
    "MAX_URL_LENGTH",
    "METRIC_FLUSH_INTERVAL_SECONDS",
    # TTL-Konstanten
    "NONCE_TTL_SECONDS",
    # Monitoring
    "PROMETHEUS_HISTOGRAM_BUCKETS",
    "REDIS_ALERTING_DLQ_KEY",
    "REDIS_DELIVERY_PREFIX",
    "REDIS_DLQ_PREFIX",
    "REDIS_HISTORY_PREFIX",
    "REDIS_IDEMPOTENCY_PREFIX",
    # Redis-Keys
    "REDIS_KEY_PREFIX",
    "REDIS_NONCE_PREFIX",
    "REDIS_OUTBOX_PREFIX",
    "REDIS_TARGETS_PREFIX",
    "RETRY_RATE_LIMIT_MAX_ATTEMPTS",
    "RETRY_RATE_LIMIT_WINDOW_SECONDS",
    "SECRET_CACHE_TTL_SECONDS",
    "SECRET_GRACE_PERIOD_HOURS",
    # Secret-Management
    "SECRET_ROTATION_INTERVAL_HOURS",
    # Timeout-Konstanten
    "SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS",
    "WORKER_POLL_INTERVAL_SECONDS",
    "WORKER_SHUTDOWN_TIMEOUT_SECONDS",
    "WORKER_SUPERVISION_INTERVAL_SECONDS",
    "get_redis_key",
    # Utility-Funktionen
    "get_tenant_normalized",
]
