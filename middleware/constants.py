"""Konstanten für Rate-Limiting-Middleware.

Dieses Modul enthält alle Magic Numbers und Hard-coded Strings,
die in den Rate-Limiting-Modulen verwendet werden.
"""

from typing import Final


class RateLimitConstants:
    """Konstanten für Rate-Limiting-Funktionalität."""

    # Bypass und Fallback-Werte
    BYPASS_LIMIT: Final[int] = 999999
    BYPASS_REMAINING: Final[int] = 999999

    # Zeit-Konstanten (in Sekunden)
    DEFAULT_TTL_SECONDS: Final[int] = 3600  # 1 Stunde
    DEFAULT_TIMEOUT_SECONDS: Final[int] = 60  # 1 Minute
    GRACE_PERIOD_SECONDS: Final[int] = 5
    MAX_BACKOFF_SECONDS: Final[int] = 300  # 5 Minuten
    STREAM_RETRY_SECONDS: Final[int] = 30
    CLEANUP_INTERVAL_SECONDS: Final[int] = 300  # 5 Minuten
    ERROR_RESET_INTERVAL_SECONDS: Final[int] = 60  # 1 Minute

    # Backoff-Konstanten
    MAX_BACKOFF_MULTIPLIER: Final[int] = 16
    DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 1.5
    MIN_RETRY_AFTER_SECONDS: Final[int] = 1

    # Kapazitäts-Konstanten
    DEFAULT_BURST_CAPACITY: Final[int] = 100
    DEFAULT_WINDOW_SIZE_SECONDS: Final[int] = 60

    # Tier-spezifische Kapazitäten
    FREE_TIER_REQUESTS_PER_SECOND: Final[float] = 10.0
    FREE_TIER_BURST_CAPACITY: Final[int] = 20
    FREE_TIER_FRAMES_PER_SECOND: Final[float] = 5.0
    FREE_TIER_MAX_CONCURRENT_STREAMS: Final[int] = 2

    PREMIUM_TIER_REQUESTS_PER_SECOND: Final[float] = 100.0
    PREMIUM_TIER_BURST_CAPACITY: Final[int] = 200
    PREMIUM_TIER_FRAMES_PER_SECOND: Final[float] = 50.0
    PREMIUM_TIER_MAX_CONCURRENT_STREAMS: Final[int] = 10

    ENTERPRISE_TIER_REQUESTS_PER_SECOND: Final[float] = 500.0
    ENTERPRISE_TIER_BURST_CAPACITY: Final[int] = 1000
    ENTERPRISE_TIER_FRAMES_PER_SECOND: Final[float] = 200.0
    ENTERPRISE_TIER_MAX_CONCURRENT_STREAMS: Final[int] = 50

    # Default-Werte für KEI-Stream
    DEFAULT_REQUESTS_PER_SECOND: Final[float] = 50.0
    DEFAULT_FRAMES_PER_SECOND: Final[float] = 20.0
    DEFAULT_MAX_CONCURRENT_STREAMS: Final[int] = 5

    # Token-Bucket-Konstanten
    INITIAL_FRAME_TOKENS: Final[int] = 100
    FRAME_TOKEN_MULTIPLIER: Final[int] = 10  # frames_per_second * 10

    # Redis-Konstanten
    DEFAULT_REDIS_URL: Final[str] = "redis://localhost:6379"
    DEFAULT_KEY_PREFIX: Final[str] = "kei_stream_rate_limit"
    STREAM_KEY_PREFIX: Final[str] = "stream"
    USER_STREAMS_KEY_PREFIX: Final[str] = "user_streams"


class HeaderConstants:
    """HTTP-Header-Konstanten."""

    # Standard-Headers
    AUTHORIZATION: Final[str] = "Authorization"
    BEARER_PREFIX: Final[str] = "Bearer "

    # IP-Extraktion-Headers
    X_FORWARDED_FOR: Final[str] = "X-Forwarded-For"
    X_REAL_IP: Final[str] = "X-Real-IP"

    # KEI-Stream-spezifische Headers
    X_USER_ID: Final[str] = "X-User-ID"
    X_TENANT_ID: Final[str] = "X-Tenant-ID"
    X_SESSION_ID: Final[str] = "X-Session-ID"
    X_STREAM_ID: Final[str] = "X-Stream-ID"
    X_FRAME_TYPE: Final[str] = "X-Frame-Type"
    X_REQUEST_ID: Final[str] = "X-Request-ID"

    # Rate-Limiting-Response-Headers
    X_RATE_LIMIT_LIMIT: Final[str] = "X-RateLimit-Limit"
    X_RATE_LIMIT_REMAINING: Final[str] = "X-RateLimit-Remaining"
    X_RATE_LIMIT_RESET: Final[str] = "X-RateLimit-Reset"
    X_RATE_LIMIT_RESET_AFTER: Final[str] = "X-RateLimit-Reset-After"
    X_RATE_LIMIT_WARNING: Final[str] = "X-RateLimit-Warning"
    RETRY_AFTER: Final[str] = "Retry-After"

    # KEI-Stream-spezifische Rate-Limiting-Headers
    X_RATE_LIMIT_ALGORITHM_STRATEGY: Final[str] = "X-RateLimit-Algorithm-Strategy"
    X_RATE_LIMIT_IDENTIFICATION_STRATEGY: Final[str] = "X-RateLimit-Identification-Strategy"
    X_RATE_LIMIT_ENDPOINT_TYPE: Final[str] = "X-RateLimit-Endpoint-Type"
    X_RATE_LIMIT_FRAMES_PER_SECOND: Final[str] = "X-RateLimit-Frames-Per-Second"
    X_RATE_LIMIT_MAX_CONCURRENT_STREAMS: Final[str] = "X-RateLimit-Max-Concurrent-Streams"
    X_RATE_LIMIT_WINDOW: Final[str] = "X-RateLimit-Window"
    X_RATE_LIMIT_STRATEGY: Final[str] = "X-RateLimit-Strategy"
    X_RATE_LIMIT_REQUESTS_PER_SECOND: Final[str] = "X-RateLimit-Requests-Per-Second"
    X_RATE_LIMIT_BURST_CAPACITY: Final[str] = "X-RateLimit-Burst-Capacity"
    X_RATE_LIMIT_FRAME_TOKENS_REMAINING: Final[str] = "X-RateLimit-Frame-Tokens-Remaining"
    X_RATE_LIMIT_CURRENT_STREAMS: Final[str] = "X-RateLimit-Current-Streams"
    X_RATE_LIMIT_MAX_STREAM_DURATION: Final[str] = "X-RateLimit-Max-Stream-Duration"


class PathConstants:
    """URL-Pfad-Konstanten."""

    # Ausgeschlossene Pfade (keine Rate-Limits)
    EXCLUDED_PATHS: Final[tuple] = (
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/metrics",
        "/favicon.ico"
    )

    # Operation-Mapping-Pfade
    REGISTER_PATHS: Final[tuple] = ("/servers/register",)
    INVOKE_PATHS: Final[tuple] = ("/tools/invoke",)
    DISCOVERY_PATHS: Final[tuple] = ("/tools",)
    STATS_PATHS: Final[tuple] = ("/stats", "/metrics")

    # KEI-Stream-Endpunkt-Pfade
    WEBSOCKET_PATHS: Final[tuple] = ("/stream/ws/",)
    SSE_PATHS: Final[tuple] = ("/stream/sse/",)
    STREAM_MANAGEMENT_PATHS: Final[tuple] = ("/stream/create", "/stream/close")
    TOOL_EXECUTION_PATHS: Final[tuple] = ("/tools/execute",)


class ErrorConstants:
    """Error-Message-Konstanten."""

    # Standard-Error-Messages
    RATE_LIMIT_EXCEEDED: Final[str] = "Rate Limit Exceeded"
    KEI_STREAM_RATE_LIMIT_EXCEEDED: Final[str] = "kei_stream_rate_limit_exceeded"

    # Error-Types
    RATE_LIMIT_ERROR_TYPE: Final[str] = "rate_limit_error"

    # Fallback-Werte
    UNKNOWN_CLIENT: Final[str] = "unknown"
    ANONYMOUS_CLIENT: Final[str] = "anonymous"
    NO_API_KEY: Final[str] = "no_api_key"
    NO_TENANT: Final[str] = "no_tenant"
    NO_SESSION: Final[str] = "no_session"
    NO_STREAM: Final[str] = "no_stream"

    # Client-ID-Prefixes
    IP_PREFIX: Final[str] = "ip"
    API_KEY_PREFIX: Final[str] = "api_key"
    USER_PREFIX: Final[str] = "user"
    TENANT_PREFIX: Final[str] = "tenant"
    SESSION_PREFIX: Final[str] = "session"
    STREAM_PREFIX: Final[str] = "stream"
    COMBINED_PREFIX: Final[str] = "combined"


class MetricsConstants:
    """Metriken-Konstanten."""

    # Basis-Metriken
    REQUESTS_ALLOWED: Final[str] = "requests_allowed"
    REQUESTS_BLOCKED: Final[str] = "requests_blocked"
    FRAMES_ALLOWED: Final[str] = "frames_allowed"
    FRAMES_BLOCKED: Final[str] = "frames_blocked"
    STREAMS_CREATED: Final[str] = "streams_created"
    STREAMS_REJECTED: Final[str] = "streams_rejected"
    REDIS_ERRORS: Final[str] = "redis_errors"
    FALLBACK_USED: Final[str] = "fallback_used"

    # Custom-Metriken-Namen
    KEI_STREAM_RATE_LIMITING_PROCESSING_TIME_MS: Final[str] = "kei_stream_rate_limiting_processing_time_ms"
    KEI_STREAM_RATE_LIMITING_REQUESTS_TOTAL: Final[str] = "kei_stream_rate_limiting_requests_total"
    KEI_STREAM_RATE_LIMITING_REQUESTS_BLOCKED_TOTAL: Final[str] = "kei_stream_rate_limiting_requests_blocked_total"
    KEI_STREAM_RATE_LIMITING_TOKENS_REMAINING: Final[str] = "kei_stream_rate_limiting_tokens_remaining"
    KEI_STREAM_RATE_LIMITING_BUCKET_UTILIZATION: Final[str] = "kei_stream_rate_limiting_bucket_utilization"
    KEI_STREAM_FRAME_RATE_TOKENS_REMAINING: Final[str] = "kei_stream_frame_rate_tokens_remaining"
    KEI_STREAM_CONCURRENT_STREAMS: Final[str] = "kei_stream_concurrent_streams"
    KEI_STREAM_RATE_LIMITING_ERRORS_TOTAL: Final[str] = "kei_stream_rate_limiting_errors_total"


class ConfigConstants:
    """Konfigurations-Konstanten."""

    # API-Key-Länge für Privacy
    API_KEY_DISPLAY_LENGTH: Final[int] = 8

    # Lua-Script-Parameter-Anzahl
    LUA_SCRIPT_KEY_COUNT: Final[int] = 2

    # Endpoint-Pattern-Regex
    ENDPOINT_PATTERN_PLACEHOLDER: Final[str] = r"\{[^}]+\}"
    ENDPOINT_PATTERN_REPLACEMENT: Final[str] = r"[^/]+"

    # Support-URLs
    DOCUMENTATION_URL: Final[str] = "https://docs.services-streaming.com/rate-limiting"
    SUPPORT_EMAIL: Final[str] = "support@services-streaming.com"
    UPGRADE_URL: Final[str] = "https://services-streaming.com/pricing"


class ValidationConstants:
    """Validierungs-Konstanten."""

    # Minimum-Werte für Validierung
    MIN_REQUESTS_PER_SECOND: Final[float] = 0.1
    MIN_BURST_CAPACITY: Final[int] = 1
    MIN_WINDOW_SIZE_SECONDS: Final[int] = 1
    MIN_FRAMES_PER_SECOND: Final[float] = 0.1
    MIN_MAX_CONCURRENT_STREAMS: Final[int] = 1

    # Maximum-Werte für Validierung
    MAX_REQUESTS_PER_SECOND: Final[float] = 10000.0
    MAX_BURST_CAPACITY: Final[int] = 100000
    MAX_WINDOW_SIZE_SECONDS: Final[int] = 86400  # 24 Stunden
    MAX_FRAMES_PER_SECOND: Final[float] = 1000.0
    MAX_MAX_CONCURRENT_STREAMS: Final[int] = 1000


# Convenience-Exports für häufig verwendete Konstanten
DEFAULT_REDIS_URL = RateLimitConstants.DEFAULT_REDIS_URL
DEFAULT_KEY_PREFIX = RateLimitConstants.DEFAULT_KEY_PREFIX
BYPASS_LIMIT = RateLimitConstants.BYPASS_LIMIT
DEFAULT_TTL_SECONDS = RateLimitConstants.DEFAULT_TTL_SECONDS
EXCLUDED_PATHS = PathConstants.EXCLUDED_PATHS
UNKNOWN_CLIENT = ErrorConstants.UNKNOWN_CLIENT


__all__ = [
    "BYPASS_LIMIT",
    "DEFAULT_KEY_PREFIX",
    # Convenience exports
    "DEFAULT_REDIS_URL",
    "DEFAULT_TTL_SECONDS",
    "EXCLUDED_PATHS",
    "UNKNOWN_CLIENT",
    "ConfigConstants",
    "ErrorConstants",
    "HeaderConstants",
    "MetricsConstants",
    "PathConstants",
    "RateLimitConstants",
    "ValidationConstants",
]
