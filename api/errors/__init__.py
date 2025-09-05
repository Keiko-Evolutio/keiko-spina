"""Konsolidiertes Error-Handling-Modul für die Keiko-API.

Dieses Modul stellt eine einheitliche, konsolidierte API für alle
Error-Handling-Funktionalitäten bereit und eliminiert Code-Duplikation.

"""

from .agent_errors import (
    AGENT_ERRORS,
    STANDARD_ERRORS,
    # Legacy API (Deprecated)
    AgentError,
    ErrorCategory,
    # Error-Codes und Konstanten
    ErrorCodes,
    HTTPStatusCodes,
    # Error-Definitionen
    StandardError,
    # Utility-Funktionen
    get_error_definition,
    get_errors_by_category,
    get_http_status_for_error,
    get_retryable_errors,
    is_retryable_error,
)
from .error_response_builder import (
    # Response-Models
    ErrorContext,
    ErrorDetail,
    # Response-Builder
    ErrorResponseBuilder,
    RecoveryStrategy,
    StandardErrorResponse,
    # Convenience-Funktionen
    build_error_detail,
    create_error_response,
    handle_standard_exceptions,
)

# ============================================================================
# UNIFIED API EXPORTS
# ============================================================================

__all__ = [
    "AGENT_ERRORS",
    "STANDARD_ERRORS",
    # Legacy API (Deprecated - für Backward-Compatibility)
    "AgentError",
    "ErrorCategory",
    # Error-Codes und Konstanten
    "ErrorCodes",
    # Response-Models
    "ErrorContext",
    "ErrorDetail",
    # Response-Builder
    "ErrorResponseBuilder",
    "HTTPStatusCodes",
    "RecoveryStrategy",
    # Error-Definitionen
    "StandardError",
    "StandardErrorResponse",
    # Convenience-Funktionen
    "build_error_detail",
    "create_error_response",
    # Utility-Funktionen
    "get_error_definition",
    "get_errors_by_category",
    "get_http_status_for_error",
    "get_retryable_errors",
    "handle_standard_exceptions",
    "is_retryable_error",
]


# ============================================================================
# VERSION UND METADATA
# ============================================================================

__version__ = "2.0.0"
__author__ = "Keiko Development Team"
__description__ = "Konsolidiertes Error-Handling für Keiko-API"
