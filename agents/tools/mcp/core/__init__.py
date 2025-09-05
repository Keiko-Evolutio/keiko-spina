"""KEI MCP Core Module.

Zentrale Komponenten für das KEI MCP System:
- BaseHTTPClient: Konsolidierte HTTP-Client-Funktionalität
- Constants: Alle Konstanten und Konfigurationswerte
- Exceptions: Einheitliche Exception-Hierarchie
- Utils: Gemeinsame Utility-Funktionen
"""

from .base_client import BaseHTTPClient, HTTPClientConfig
from .constants import (
    # Caching
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_CONNECTION_POOL_SIZE,
    # Circuit Breaker
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_MAX_CACHE_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECOVERY_TIMEOUT,
    DEFAULT_SUCCESS_THRESHOLD,
    # HTTP-Konfiguration
    DEFAULT_TIMEOUT_SECONDS,
    # MCP-Endpoints
    ENDPOINTS,
    # HTTP-Headers
    HEADERS,
)
from .exceptions import (
    KEIMCPAuthenticationError,
    KEIMCPConnectionError,
    KEIMCPError,
    KEIMCPServerError,
    KEIMCPTimeoutError,
    KEIMCPValidationError,
)
from .utils import (
    batch_items,
    calculate_hash,
    create_cache_key,
    deep_merge_dicts,
    extract_domain_from_url,
    format_duration_ms,
    format_error_message,
    generate_correlation_id,
    generate_request_id,
    is_valid_api_key,
    mask_sensitive_data,
    normalize_url,
    safe_json_loads,
    sanitize_server_name,
    truncate_string,
    validate_url,
)

__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_CONNECTION_POOL_SIZE",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_MAX_CACHE_SIZE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RECOVERY_TIMEOUT",
    "DEFAULT_SUCCESS_THRESHOLD",
    # Constants
    "DEFAULT_TIMEOUT_SECONDS",
    "ENDPOINTS",
    "HEADERS",
    # Base Client
    "BaseHTTPClient",
    "HTTPClientConfig",
    "KEIMCPAuthenticationError",
    "KEIMCPConnectionError",
    # Exceptions
    "KEIMCPError",
    "KEIMCPServerError",
    "KEIMCPTimeoutError",
    "KEIMCPValidationError",
    "batch_items",
    "calculate_hash",
    "create_cache_key",
    "deep_merge_dicts",
    "extract_domain_from_url",
    "format_duration_ms",
    "format_error_message",
    # Utils
    "generate_correlation_id",
    "generate_request_id",
    "is_valid_api_key",
    "mask_sensitive_data",
    "normalize_url",
    "safe_json_loads",
    "sanitize_server_name",
    "truncate_string",
    "validate_url",
]
