"""Zentrale Konstanten für das Konfigurationsmodul.

Eliminiert Magic Numbers und Hard-coded Strings durch aussagekräftige Konstanten.
Folgt Clean Code Prinzipien und verbessert die Wartbarkeit.
"""


# ============================================================================
# ENVIRONMENT UND DEPLOYMENT KONSTANTEN
# ============================================================================

DEFAULT_ENVIRONMENT: str = "development"
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_AZURE_LOCATION: str = "westeurope"

# ============================================================================
# AZURE KONFIGURATION KONSTANTEN
# ============================================================================

# Azure Storage
DEFAULT_STORAGE_ACCOUNT_URL: str = "https://defaultstorage.blob.core.windows.net"
DEFAULT_KEIKO_STORAGE_CONTAINER: str = "keiko-images"

# Azure AI Services
DEFAULT_DATABASE_NAME: str = "cdb_db_id_keikoPersonalAssistant"
DEFAULT_CONTAINER_NAME: str = "configurations"
DEFAULT_MODEL_DEPLOYMENT_NAME: str = "gpt-4o"
DEFAULT_API_VERSION: str = "2025-05-01"

# ============================================================================
# SECURITY UND AUTHENTICATION KONSTANTEN
# ============================================================================

# Secret Management
DEFAULT_SECRET_ROTATION_INTERVAL_DAYS: int = 30
DEFAULT_SECRET_GRACE_PERIOD_HOURS: int = 24
DEFAULT_SECRET_CACHE_TTL_SECONDS: int = 300  # 5 Minuten

# n8n Webhook Security
DEFAULT_N8N_HMAC_SECRET: str = "n8n_webhook_secret"

# JWT Configuration
DEFAULT_JWT_ALGORITHM: str = "HS256"
DEFAULT_JWT_LEEWAY_SECONDS: int = 30

# ============================================================================
# RATE LIMITING KONSTANTEN
# ============================================================================

# Basic Tier Limits
BASIC_TIER_REQUESTS_PER_MINUTE: int = 60
BASIC_TIER_REQUESTS_PER_HOUR: int = 1000
BASIC_TIER_REQUESTS_PER_DAY: int = 10000
BASIC_TIER_BURST_SIZE: int = 10

# Premium Tier Limits
PREMIUM_TIER_REQUESTS_PER_MINUTE: int = 200
PREMIUM_TIER_REQUESTS_PER_HOUR: int = 5000
PREMIUM_TIER_REQUESTS_PER_DAY: int = 50000
PREMIUM_TIER_BURST_SIZE: int = 30

# Enterprise Tier Limits
ENTERPRISE_TIER_REQUESTS_PER_MINUTE: int = 1000
ENTERPRISE_TIER_REQUESTS_PER_HOUR: int = 20000
ENTERPRISE_TIER_REQUESTS_PER_DAY: int = 200000
ENTERPRISE_TIER_BURST_SIZE: int = 100

# Admin Tier Limits
ADMIN_TIER_REQUESTS_PER_MINUTE: int = 10000
ADMIN_TIER_REQUESTS_PER_HOUR: int = 100000
ADMIN_TIER_REQUESTS_PER_DAY: int = 1000000
ADMIN_TIER_BURST_SIZE: int = 500

# Rate Limiting Configuration
DEFAULT_RATE_LIMIT_WINDOW_SIZE_SECONDS: int = 60
DEFAULT_RATE_LIMIT_SOFT_LIMIT_FACTOR: float = 0.8
DEFAULT_RATE_LIMIT_BURST_REFILL_RATE: float = 1.0

# Redis Configuration
DEFAULT_REDIS_HOST: str = "localhost"
DEFAULT_REDIS_PORT: int = 6379
DEFAULT_REDIS_DB: int = 1
DEFAULT_REDIS_TIMEOUT_SECONDS: float = 5.0

# IP-based Fallback Limits
DEFAULT_IP_REQUESTS_PER_MINUTE: int = 20

# ============================================================================
# DDOS PROTECTION KONSTANTEN
# ============================================================================

DEFAULT_DDOS_IP_BLOCK_DURATION_SECONDS: int = 900  # 15 Minuten
DEFAULT_DDOS_ANOMALY_REQUESTS_PER_MINUTE: int = 600
DEFAULT_DDOS_BURST_THRESHOLD: int = 120

# ============================================================================
# MONITORING UND OBSERVABILITY KONSTANTEN
# ============================================================================

# Health Check Intervals
DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS: float = 60.0
DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS: float = 30.0

# Cache TTL Values
DEFAULT_TOOL_CACHE_TTL_SECONDS: float = 300.0  # 5 Minuten
DEFAULT_OIDC_CACHE_TTL_SECONDS: int = 3600  # 1 Stunde
DEFAULT_JWKS_CACHE_TTL_SECONDS: int = 3600  # 1 Stunde
DEFAULT_DISCOVERY_CACHE_TTL_SECONDS: int = 86400  # 24 Stunden

# Cleanup Intervals
DEFAULT_CLEANUP_INTERVAL_SECONDS: int = 3600  # 1 Stunde

# ============================================================================
# MCP SERVER KONFIGURATION KONSTANTEN
# ============================================================================

# Default Timeouts
DEFAULT_MCP_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MCP_MAX_RETRIES: int = 3

# Specific Service Timeouts
WEATHER_SERVICE_TIMEOUT_SECONDS: float = 15.0
WEATHER_SERVICE_MAX_RETRIES: int = 2

DATABASE_SERVICE_TIMEOUT_SECONDS: float = 45.0
DATABASE_SERVICE_MAX_RETRIES: int = 3

DOCUMENT_SERVICE_TIMEOUT_SECONDS: float = 120.0  # Längerer Timeout für Dokumentenverarbeitung
DOCUMENT_SERVICE_MAX_RETRIES: int = 1

AI_ML_SERVICE_TIMEOUT_SECONDS: float = 60.0
AI_ML_SERVICE_MAX_RETRIES: int = 2

# Connection Pool Configuration
DEFAULT_CONNECTION_POOL_SIZE: int = 10
DEFAULT_MAX_CONCURRENT_REQUESTS: int = 50

# ============================================================================
# MTLS KONFIGURATION KONSTANTEN
# ============================================================================

# SSL Session Configuration
DEFAULT_SSL_SESSION_CACHE_SIZE: int = 1000
DEFAULT_SSL_SESSION_TIMEOUT_SECONDS: int = 3600  # 1 Stunde

# Certificate Headers
DEFAULT_CLIENT_CERT_HEADER: str = "X-Client-Cert"

# ============================================================================
# WEBSOCKET KONFIGURATION KONSTANTEN
# ============================================================================

# WebSocket Auth Configuration
DEFAULT_WS_AUTH_HEADER: str = "Authorization"
DEFAULT_WS_TOKEN_QUERY_PARAM: str = "token"

# WebSocket Paths
DEFAULT_WS_AUTH_PATHS: list[str] = ["/ws/", "/websocket/"]

# ============================================================================
# STREAMING UND KEI-STREAM KONSTANTEN
# ============================================================================

# Free Tier Streaming Limits
FREE_TIER_REQUESTS_PER_SECOND: float = 5.0
FREE_TIER_BURST_CAPACITY: int = 10
FREE_TIER_FRAMES_PER_SECOND: float = 2.0
FREE_TIER_MAX_CONCURRENT_STREAMS: int = 1
FREE_TIER_MAX_STREAM_DURATION_SECONDS: int = 300  # 5 Minuten

# Basic Tier Streaming Limits
BASIC_TIER_REQUESTS_PER_SECOND: float = 20.0
BASIC_TIER_BURST_CAPACITY: int = 40
BASIC_TIER_FRAMES_PER_SECOND: float = 10.0
BASIC_TIER_MAX_CONCURRENT_STREAMS: int = 3
BASIC_TIER_MAX_STREAM_DURATION_SECONDS: int = 1800  # 30 Minuten

# Premium Tier Streaming Limits
PREMIUM_TIER_REQUESTS_PER_SECOND: float = 100.0
PREMIUM_TIER_BURST_CAPACITY: int = 200
PREMIUM_TIER_FRAMES_PER_SECOND: float = 50.0
PREMIUM_TIER_MAX_CONCURRENT_STREAMS: int = 10
PREMIUM_TIER_MAX_STREAM_DURATION_SECONDS: int = 3600  # 1 Stunde

# Enterprise Tier Streaming Limits
ENTERPRISE_TIER_REQUESTS_PER_SECOND: float = 500.0
ENTERPRISE_TIER_BURST_CAPACITY: int = 1000
ENTERPRISE_TIER_FRAMES_PER_SECOND: float = 200.0
ENTERPRISE_TIER_MAX_CONCURRENT_STREAMS: int = 50
ENTERPRISE_TIER_MAX_STREAM_DURATION_SECONDS: int = 7200  # 2 Stunden

# Unlimited Tier Streaming Limits
UNLIMITED_TIER_REQUESTS_PER_SECOND: float = 10000.0
UNLIMITED_TIER_BURST_CAPACITY: int = 20000
UNLIMITED_TIER_FRAMES_PER_SECOND: float = 1000.0
UNLIMITED_TIER_MAX_CONCURRENT_STREAMS: int = 1000
UNLIMITED_TIER_MAX_STREAM_DURATION_SECONDS: int = 86400  # 24 Stunden

# Backoff Configuration
DEFAULT_GRACE_PERIOD_SECONDS: int = 5
DEFAULT_BACKOFF_MULTIPLIER: float = 1.5
DEFAULT_MAX_BACKOFF_SECONDS: int = 300

# ============================================================================
# SLA UND PERFORMANCE KONSTANTEN
# ============================================================================

# SLA Targets
DEFAULT_SLA_AVAILABILITY_TARGET_PCT: float = 99.9
DEFAULT_SLA_LATENCY_TARGET_MS: float = 200.0
DEFAULT_SLA_ERROR_RATE_TARGET_PCT: float = 1.0

# Reporting Configuration
DEFAULT_REPORTING_INTERVAL_MINUTES: int = 60
MIN_REPORTING_INTERVAL_MINUTES: int = 1
MAX_REPORTING_INTERVAL_MINUTES: int = 1440  # 24 Stunden

# Anomaly Detection
DEFAULT_ANOMALY_TRAINING_INTERVAL_MINUTES: int = 120
MIN_ANOMALY_TRAINING_INTERVAL_MINUTES: int = 5
MAX_ANOMALY_TRAINING_INTERVAL_MINUTES: int = 10080  # 1 Woche
DEFAULT_ANOMALY_MODEL_VERSION: str = "v1"

# ============================================================================
# EMAIL UND SMS KONSTANTEN
# ============================================================================

# SMTP Configuration
DEFAULT_SMTP_PORT: int = 587

# ============================================================================
# GRAFANA UND MONITORING KONSTANTEN
# ============================================================================

DEFAULT_GRAFANA_URL: str = "http://localhost:3001"

# ============================================================================
# DEVELOPMENT UND TESTING KONSTANTEN
# ============================================================================

DEFAULT_DEV_TOKEN: str = "dev-token-12345"

# ============================================================================
# TIER MAPPING DICTIONARIES
# ============================================================================

TIER_RATE_LIMITS: dict[str, dict[str, int]] = {
    "basic": {
        "requests_per_minute": BASIC_TIER_REQUESTS_PER_MINUTE,
        "requests_per_hour": BASIC_TIER_REQUESTS_PER_HOUR,
        "requests_per_day": BASIC_TIER_REQUESTS_PER_DAY,
        "burst_size": BASIC_TIER_BURST_SIZE,
    },
    "premium": {
        "requests_per_minute": PREMIUM_TIER_REQUESTS_PER_MINUTE,
        "requests_per_hour": PREMIUM_TIER_REQUESTS_PER_HOUR,
        "requests_per_day": PREMIUM_TIER_REQUESTS_PER_DAY,
        "burst_size": PREMIUM_TIER_BURST_SIZE,
    },
    "enterprise": {
        "requests_per_minute": ENTERPRISE_TIER_REQUESTS_PER_MINUTE,
        "requests_per_hour": ENTERPRISE_TIER_REQUESTS_PER_HOUR,
        "requests_per_day": ENTERPRISE_TIER_REQUESTS_PER_DAY,
        "burst_size": ENTERPRISE_TIER_BURST_SIZE,
    },
    "admin": {
        "requests_per_minute": ADMIN_TIER_REQUESTS_PER_MINUTE,
        "requests_per_hour": ADMIN_TIER_REQUESTS_PER_HOUR,
        "requests_per_day": ADMIN_TIER_REQUESTS_PER_DAY,
        "burst_size": ADMIN_TIER_BURST_SIZE,
    },
}

TIER_STREAMING_LIMITS: dict[str, dict[str, float]] = {
    "free": {
        "requests_per_second": FREE_TIER_REQUESTS_PER_SECOND,
        "burst_capacity": FREE_TIER_BURST_CAPACITY,
        "frames_per_second": FREE_TIER_FRAMES_PER_SECOND,
        "max_concurrent_streams": FREE_TIER_MAX_CONCURRENT_STREAMS,
        "max_stream_duration_seconds": FREE_TIER_MAX_STREAM_DURATION_SECONDS,
    },
    "basic": {
        "requests_per_second": BASIC_TIER_REQUESTS_PER_SECOND,
        "burst_capacity": BASIC_TIER_BURST_CAPACITY,
        "frames_per_second": BASIC_TIER_FRAMES_PER_SECOND,
        "max_concurrent_streams": BASIC_TIER_MAX_CONCURRENT_STREAMS,
        "max_stream_duration_seconds": BASIC_TIER_MAX_STREAM_DURATION_SECONDS,
    },
    "premium": {
        "requests_per_second": PREMIUM_TIER_REQUESTS_PER_SECOND,
        "burst_capacity": PREMIUM_TIER_BURST_CAPACITY,
        "frames_per_second": PREMIUM_TIER_FRAMES_PER_SECOND,
        "max_concurrent_streams": PREMIUM_TIER_MAX_CONCURRENT_STREAMS,
        "max_stream_duration_seconds": PREMIUM_TIER_MAX_STREAM_DURATION_SECONDS,
    },
    "enterprise": {
        "requests_per_second": ENTERPRISE_TIER_REQUESTS_PER_SECOND,
        "burst_capacity": ENTERPRISE_TIER_BURST_CAPACITY,
        "frames_per_second": ENTERPRISE_TIER_FRAMES_PER_SECOND,
        "max_concurrent_streams": ENTERPRISE_TIER_MAX_CONCURRENT_STREAMS,
        "max_stream_duration_seconds": ENTERPRISE_TIER_MAX_STREAM_DURATION_SECONDS,
    },
    "unlimited": {
        "requests_per_second": UNLIMITED_TIER_REQUESTS_PER_SECOND,
        "burst_capacity": UNLIMITED_TIER_BURST_CAPACITY,
        "frames_per_second": UNLIMITED_TIER_FRAMES_PER_SECOND,
        "max_concurrent_streams": UNLIMITED_TIER_MAX_CONCURRENT_STREAMS,
        "max_stream_duration_seconds": UNLIMITED_TIER_MAX_STREAM_DURATION_SECONDS,
    },
}
