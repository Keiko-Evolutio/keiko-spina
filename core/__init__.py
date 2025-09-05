"""Keiko Core Package: Exceptions, Error Handling, Logging-Strategie."""

# Explicit imports instead of wildcard imports
from .constants import (
    ErrorCode,
    HTTPStatus,
    LoggingConfig,
    OTelAttributes,
    RetryConfig,
    SeverityLevel,
)
from .error_handler import (
    # Data Classes
    ErrorContext,
    ErrorLogger,
    # Core Components
    ExceptionClassifier,
    # Main Handler
    GlobalErrorHandler,
    RecoveryStrategy,
    ResponseBuilder,
    TracingHandler,
)
from .exceptions import (
    # Domain Exceptions
    AgentError,
    AuthError,
    AzureError,
    BadRequestError,
    BudgetExceededError,
    ConflictError,
    DeadlineExceededError,
    DependencyError,
    ExceptionFactory,
    KeikoAuthenticationError,
    KeikoAzureError,
    KeikoBadRequestError,
    KeikoErrorPayload,
    # Base Classes
    KeikoException,
    KeikoNotFoundError,
    KeikoRateLimitError,
    KeikoServiceError,
    KeikoTimeoutError,
    # Backward Compatibility Aliases
    KeikoValidationError,
    NetworkError,
    NotFoundError,
    OperationTimeout,
    PolicyViolationError,
    RateLimitExceeded,
    ServiceUnavailableError,
    ValidationError,
)
from .logging_strategy import (
    StructuredLogger,
    get_audit_logger,
    get_error_logger,
    get_performance_logger,
    get_security_logger,
)

__all__ = [  # explizite Re-Exports für Klarheit
    # Domain Exceptions
    "AgentError",
    "AuthError",
    "AzureError",
    "BadRequestError",
    "BudgetExceededError",
    "ConflictError",
    "DeadlineExceededError",
    "DependencyError",
    "ErrorCode",
    # Error Handler Components
    "ErrorContext",
    "ErrorLogger",
    "ExceptionClassifier",
    "ExceptionFactory",
    "GlobalErrorHandler",
    # Constants
    "HTTPStatus",
    "KeikoAuthenticationError",
    "KeikoAzureError",
    "KeikoBadRequestError",
    "KeikoErrorPayload",
    # Base Classes
    "KeikoException",
    "KeikoNotFoundError",
    "KeikoRateLimitError",
    "KeikoServiceError",
    "KeikoTimeoutError",
    # Backward Compatibility Aliases
    "KeikoValidationError",
    "LoggingConfig",
    "NetworkError",
    "NotFoundError",
    "OTelAttributes",
    "OperationTimeout",
    "PolicyViolationError",
    "RateLimitExceeded",
    "RecoveryStrategy",
    "ResponseBuilder",
    "RetryConfig",
    "ServiceUnavailableError",
    "SeverityLevel",
    # Logging Components
    "StructuredLogger",
    "TracingHandler",
    "ValidationError",
    "get_audit_logger",
    "get_error_logger",
    "get_performance_logger",
    "get_security_logger",
]
