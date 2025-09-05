# backend/services/clients/common/__init__.py
"""Common Utilities für Client Services.

Dieses Package enthält wiederverwendbare Komponenten für alle Client Services:
- Konstanten und Konfigurationswerte
- Retry-Logik mit Exponential Backoff
- HTTP Client Konfiguration
- Konsistente Fehlerbehandlung
"""

from __future__ import annotations

# Konstanten
from .constants import (
    # Logging Events
    CLIENT_INIT_EVENT,
    CLIENT_READY_EVENT,
    CONTENT_POLICY_VIOLATION_DETECTED_EVENT,
    # API Versionen
    CONTENT_SAFETY_API_VERSION,
    # Content Safety
    CONTENT_SAFETY_CATEGORIES,
    CONTENT_SAFETY_FALLBACK_CATEGORY,
    CONTENT_SAFETY_REQUEST_EVENT,
    CONTENT_SAFETY_UNAVAILABLE_EVENT,
    CONTENT_SAFETY_UNAVAILABLE_REASON,
    DEEP_RESEARCH_FALLBACK_MESSAGE,
    DEEP_RESEARCH_MAX_ITERATIONS,
    DEEP_RESEARCH_SDK_UNAVAILABLE,
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_CONFIDENCE_SCORE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_IMAGE_API_VERSION,
    DEFAULT_IMAGE_CONTENT_TYPE,
    DEFAULT_IMAGE_DEPLOYMENT,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_IMAGE_RESPONSE_FORMAT,
    # Image Generation
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INITIAL_DELAY,
    # Retry Konfiguration
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    # Timeouts
    DEFAULT_TIMEOUT,
    # Default Values
    DEFAULT_USER_ID,
    HIGH_CONFIDENCE_SCORE,
    IMAGE_GENERATION_NO_CONTENT_ERROR,
    IMAGE_GENERATION_REQUEST_EVENT,
    IMAGE_SERVICE_CLIENT_READY_EVENT,
    MAX_SEVERITY_LEVEL,
    MEDIUM_CONFIDENCE_SCORE,
    OPENAI_IMAGES_GENERATE_CALL_EVENT,
    OPENAI_IMAGES_GENERATE_ERROR_EVENT,
    OPENAI_IMAGES_GENERATE_OK_EVENT,
    OPENAI_IMAGES_NO_CONTENT_EVENT,
    PROMPT_SAFETY_VIOLATION_EVENT,
    PROMPT_SANITIZED_EVENT,
    SAFE_SEVERITY_THRESHOLD,
    # Error Messages
    SERVICE_NOT_CONFIGURED_ERROR,
    SERVICE_UNAVAILABLE_ERROR,
    USING_FALLBACK_PROMPT_EVENT,
)

# Error Handling
from .error_handling import (
    ClientServiceException,
    ContentPolicyViolationException,
    ErrorHandler,
    ServiceError,
    ServiceNotConfiguredException,
    ServiceUnavailableException,
    create_fallback_result,
    handle_http_error,
    is_content_policy_violation,
    log_service_error,
)

# HTTP Configuration
from .http_config import (
    HTTPClientConfig,
    StandardHTTPClientConfig,
    create_aiohttp_connector,
    create_aiohttp_session_config,
    create_azure_headers,
    create_httpx_client_config,
    create_kei_rpc_headers,
    create_openai_headers,
)

# Retry Utilities
from .retry_utils import (
    RetryableClient,
    RetryConfig,
    RetryExhaustedException,
    create_content_safety_retry_config,
    create_http_retry_config,
    create_image_generation_retry_config,
    retry_with_backoff,
    with_retry,
)

__all__ = [
    "CLIENT_INIT_EVENT",
    "CLIENT_READY_EVENT",
    "CONTENT_POLICY_VIOLATION_DETECTED_EVENT",
    "CONTENT_SAFETY_API_VERSION",
    "CONTENT_SAFETY_CATEGORIES",
    "CONTENT_SAFETY_FALLBACK_CATEGORY",
    "CONTENT_SAFETY_REQUEST_EVENT",
    "CONTENT_SAFETY_UNAVAILABLE_EVENT",
    "CONTENT_SAFETY_UNAVAILABLE_REASON",
    "DEEP_RESEARCH_FALLBACK_MESSAGE",
    "DEEP_RESEARCH_MAX_ITERATIONS",
    "DEEP_RESEARCH_SDK_UNAVAILABLE",
    "DEFAULT_BACKOFF_MULTIPLIER",
    "DEFAULT_CONFIDENCE_SCORE",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_IMAGE_API_VERSION",
    "DEFAULT_IMAGE_CONTENT_TYPE",
    "DEFAULT_IMAGE_DEPLOYMENT",
    "DEFAULT_IMAGE_QUALITY",
    "DEFAULT_IMAGE_RESPONSE_FORMAT",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_INITIAL_DELAY",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_REQUEST_TIMEOUT",
    # Konstanten
    "DEFAULT_TIMEOUT",
    "DEFAULT_USER_ID",
    "HIGH_CONFIDENCE_SCORE",
    "IMAGE_GENERATION_NO_CONTENT_ERROR",
    "IMAGE_GENERATION_REQUEST_EVENT",
    "IMAGE_SERVICE_CLIENT_READY_EVENT",
    "MAX_SEVERITY_LEVEL",
    "MEDIUM_CONFIDENCE_SCORE",
    "OPENAI_IMAGES_GENERATE_CALL_EVENT",
    "OPENAI_IMAGES_GENERATE_ERROR_EVENT",
    "OPENAI_IMAGES_GENERATE_OK_EVENT",
    "OPENAI_IMAGES_NO_CONTENT_EVENT",
    "PROMPT_SAFETY_VIOLATION_EVENT",
    "PROMPT_SANITIZED_EVENT",
    "SAFE_SEVERITY_THRESHOLD",
    "SERVICE_NOT_CONFIGURED_ERROR",
    "SERVICE_UNAVAILABLE_ERROR",
    "USING_FALLBACK_PROMPT_EVENT",
    "ClientServiceException",
    "ContentPolicyViolationException",
    "ErrorHandler",
    # HTTP Configuration
    "HTTPClientConfig",
    # Retry Utilities
    "RetryConfig",
    "RetryExhaustedException",
    "RetryableClient",
    # Error Handling
    "ServiceError",
    "ServiceNotConfiguredException",
    "ServiceUnavailableException",
    "StandardHTTPClientConfig",
    "create_aiohttp_connector",
    "create_aiohttp_session_config",
    "create_azure_headers",
    "create_content_safety_retry_config",
    "create_fallback_result",
    "create_http_retry_config",
    "create_httpx_client_config",
    "create_image_generation_retry_config",
    "create_kei_rpc_headers",
    "create_openai_headers",
    "handle_http_error",
    "is_content_policy_violation",
    "log_service_error",
    "retry_with_backoff",
    "with_retry",
]
