"""KEI MCP Constants und Konfigurationswerte.

Zentrale Definition aller Magic Numbers, Hard-coded Strings und
Konfigurationswerte für das KEI MCP System.
"""

from typing import Any, Final

# ============================================================================
# HTTP-KONFIGURATION
# ============================================================================

# Timeouts
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_READ_TIMEOUT_SECONDS: Final[float] = 30.0

# Retries
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0
DEFAULT_RETRY_BACKOFF_FACTOR: Final[float] = 2.0

# Connection Pooling
DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 100
DEFAULT_MAX_KEEPALIVE_CONNECTIONS: Final[int] = 20
DEFAULT_KEEPALIVE_EXPIRY_SECONDS: Final[float] = 5.0

# ============================================================================
# CIRCUIT BREAKER KONFIGURATION
# ============================================================================

DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
DEFAULT_RECOVERY_TIMEOUT: Final[float] = 60.0
DEFAULT_SUCCESS_THRESHOLD: Final[int] = 3
DEFAULT_MONITOR_WINDOW_SECONDS: Final[float] = 300.0  # 5 Minuten

# ============================================================================
# CACHING KONFIGURATION
# ============================================================================

DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600  # 1 Stunde
DEFAULT_MAX_CACHE_SIZE: Final[int] = 1000
DEFAULT_DISCOVERY_INTERVAL_SECONDS: Final[float] = 300.0  # 5 Minuten
DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS: Final[float] = 60.0  # 1 Minute

# ============================================================================
# HTTP-HEADERS
# ============================================================================

HEADERS: Final[dict[str, str]] = {
    "API_KEY": "X-API-Key",
    "CONTENT_TYPE": "application/json",
    "ACCEPT": "application/json",
    "USER_AGENT": "KEI-MCP-Client/1.0",
    "CORRELATION_ID": "X-Correlation-ID",
    "REQUEST_ID": "X-Request-ID",
}

# Standard HTTP Headers
STANDARD_HEADERS: Final[dict[str, str]] = {
    "Content-Type": HEADERS["CONTENT_TYPE"],
    "Accept": HEADERS["ACCEPT"],
    "User-Agent": HEADERS["USER_AGENT"],
}

# ============================================================================
# MCP-ENDPOINTS
# ============================================================================

ENDPOINTS: Final[dict[str, str]] = {
    # Core MCP Endpoints
    "TOOLS": "/mcp/tools",
    "TOOLS_INVOKE": "/mcp/invoke",
    "RESOURCES": "/mcp/resources",
    "PROMPTS": "/mcp/prompts",

    # Server Info
    "INFO": "/mcp/info",
    "HEALTH": "/mcp/health",
    "STATUS": "/mcp/status",

    # Management Endpoints
    "REGISTER": "/mcp/register",
    "UNREGISTER": "/mcp/unregister",
    "METRICS": "/mcp/metrics",
}

# ============================================================================
# ERROR CODES UND MESSAGES
# ============================================================================

ERROR_CODES: Final[dict[str, int]] = {
    "CONNECTION_FAILED": 1001,
    "TIMEOUT": 1002,
    "AUTHENTICATION_FAILED": 1003,
    "VALIDATION_FAILED": 1004,
    "SERVER_ERROR": 1005,
    "CIRCUIT_BREAKER_OPEN": 1006,
    "RATE_LIMIT_EXCEEDED": 1007,
}

ERROR_MESSAGES: Final[dict[str, str]] = {
    "CONNECTION_FAILED": "Verbindung zum MCP Server fehlgeschlagen",
    "TIMEOUT": "Request-Timeout erreicht",
    "AUTHENTICATION_FAILED": "Authentifizierung fehlgeschlagen",
    "VALIDATION_FAILED": "Schema-Validierung fehlgeschlagen",
    "SERVER_ERROR": "Interner Server-Fehler",
    "CIRCUIT_BREAKER_OPEN": "Circuit Breaker ist geöffnet",
    "RATE_LIMIT_EXCEEDED": "Rate Limit überschritten",
}

# ============================================================================
# DISCOVERY KONFIGURATION
# ============================================================================

DISCOVERY_CONFIG: Final[dict[str, Any]] = {
    "BATCH_SIZE": 10,
    "PARALLEL_WORKERS": 5,
    "TIMEOUT_PER_SERVER": 30.0,
    "MAX_DISCOVERY_RETRIES": 2,
}

# Tool-Kategorien
TOOL_CATEGORIES: Final[dict[str, str]] = {
    "DATA_PROCESSING": "data_processing",
    "FILE_OPERATIONS": "file_operations",
    "WEB_SCRAPING": "web_scraping",
    "API_INTEGRATION": "api_integration",
    "ANALYSIS": "analysis",
    "COMMUNICATION": "communication",
    "AUTOMATION": "automation",
    "UTILITY": "utility",
}

# Resource-Typen
RESOURCE_TYPES: Final[dict[str, str]] = {
    "FILE": "file",
    "DATABASE": "database",
    "API": "api",
    "STREAM": "stream",
    "MEMORY": "memory",
    "CACHE": "cache",
}

# Prompt-Kategorien
PROMPT_CATEGORIES: Final[dict[str, str]] = {
    "SUMMARIZATION": "summarization",
    "TRANSLATION": "translation",
    "ANALYSIS": "analysis",
    "GENERATION": "generation",
    "CLASSIFICATION": "classification",
    "EXTRACTION": "extraction",
}

# ============================================================================
# SSL/TLS KONFIGURATION
# ============================================================================

SSL_CONFIG: Final[dict[str, Any]] = {
    "VERIFY_MODE": True,
    "CHECK_HOSTNAME": True,
    "MINIMUM_VERSION": "TLSv1.2",
    "CIPHERS": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
}

# ============================================================================
# LOGGING KONFIGURATION
# ============================================================================

LOG_CONFIG: Final[dict[str, Any]] = {
    "LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "MAX_LOG_SIZE_MB": 10,
    "BACKUP_COUNT": 5,
}

# ============================================================================
# METRICS UND MONITORING
# ============================================================================

METRICS_CONFIG: Final[dict[str, Any]] = {
    "COLLECTION_INTERVAL_SECONDS": 60.0,
    "RETENTION_DAYS": 7,
    "BATCH_SIZE": 100,
    "EXPORT_TIMEOUT_SECONDS": 30.0,
}

# Performance-Thresholds
PERFORMANCE_THRESHOLDS: Final[dict[str, float]] = {
    "MAX_RESPONSE_TIME_MS": 5000.0,
    "MAX_ERROR_RATE_PERCENT": 5.0,
    "MIN_SUCCESS_RATE_PERCENT": 95.0,
    "MAX_MEMORY_USAGE_MB": 512.0,
}

# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

VALIDATION_PATTERNS: Final[dict[str, str]] = {
    "SERVER_NAME": r"^[a-zA-Z0-9_-]+$",
    "TOOL_NAME": r"^[a-zA-Z0-9_.-]+$",
    "RESOURCE_NAME": r"^[a-zA-Z0-9_./:-]+$",
    "URL": r"^https?://[^\s/$.?#].[^\s]*$",
    "API_KEY": r"^[a-zA-Z0-9_-]{16,}$",
}

# ============================================================================
# FEATURE FLAGS
# ============================================================================

FEATURE_FLAGS: Final[dict[str, bool]] = {
    "HTTP2_ENABLED": True,
    "COMPRESSION_ENABLED": True,
    "METRICS_ENABLED": True,
    "TRACING_ENABLED": True,
    "CACHING_ENABLED": True,
    "CIRCUIT_BREAKER_ENABLED": True,
    "RATE_LIMITING_ENABLED": True,
}
