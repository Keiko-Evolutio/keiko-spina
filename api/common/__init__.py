"""Gemeinsame Basis-Komponenten für das API-Modul.

Dieses Paket stellt wiederverwendbare Basis-Abstraktionen, Error-Handler,
Response-Models und Validation-Utilities für alle API-Komponenten bereit.
"""

from .base_middleware import BaseKeikoMiddleware
from .constants import (
    CacheConfig,
    ErrorCodes,
    FeatureFlags,
    HTTPStatusCodes,
    LoggingConfig,
    MiddlewareConfig,
    ValidationPatterns,
)
from .error_handlers import (
    APIError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    StandardErrorHandler,
    ValidationError,
    create_error_response,
    handle_standard_exceptions,
)

# Exception Utilities
from .exceptions import (
    APIExceptionBuilder,
    business_logic_error,
    conflict_error,
    duplicate_name,
    internal_server_error,
    invalid_field,
    name_exists_error,
    not_found_error,
    rate_limit_error,
    resource_not_found,
    validation_error,
)
from .identifiers import (
    IDConstants,
    IDGenerator,
    agent_id_generator,
    configuration_id_generator,
    generate_agent_id,
    generate_config_id,
    generate_human_readable_id,
    generate_prefixed_id,
    generate_secure_id,
    generate_session_id,
    generate_short_uuid,
    generate_timestamped_id,
    generate_uuid,
    generate_webhook_id,
    session_id_generator,
    validate_id_format,
    webhook_id_generator,
)
from .pagination import (
    PaginationCalculator,
    PaginationMeta,
    PaginationMetaModel,
    PaginationParams,
    calculate_pagination_meta,
    paginate_items,
    validate_pagination_params,
)
from .response_models import (
    ErrorResponse,
    PaginatedResponse,
    StandardResponse,
    SuccessResponse,
    create_success_response,
)
from .response_models import create_error_response as create_std_error_response
from .storage import (
    BaseStorage,
    InMemoryStorage,
    NamedInMemoryStorage,
    clear_all_storages,
    get_storage,
)
from .validation_utils import (
    check_required_fields,
    sanitize_input,
    validate_correlation_id,
    validate_email,
    validate_json_payload,
    validate_server_name,
    validate_uuid,
)

__all__ = [
    "APIError",
    # Exception Utilities
    "APIExceptionBuilder",
    # Base Middleware
    "BaseKeikoMiddleware",
    # Storage Utilities
    "BaseStorage",
    "CacheConfig",
    "ConflictError",
    # Constants
    "ErrorCodes",
    "ErrorResponse",
    "FeatureFlags",
    "HTTPStatusCodes",
    # ID Generation Utilities
    "IDConstants",
    "IDGenerator",
    "InMemoryStorage",
    "LoggingConfig",
    "MiddlewareConfig",
    "NamedInMemoryStorage",
    "NotFoundError",
    "PaginatedResponse",
    "PaginationCalculator",
    "PaginationMeta",
    "PaginationMetaModel",
    # Pagination
    "PaginationParams",
    "RateLimitError",
    # Error Handling
    "StandardErrorHandler",
    # Response Models
    "StandardResponse",
    "SuccessResponse",
    "ValidationError",
    "ValidationPatterns",
    "agent_id_generator",
    "business_logic_error",
    "calculate_pagination_meta",
    "check_required_fields",
    "clear_all_storages",
    "configuration_id_generator",
    "conflict_error",
    "create_error_response",
    "create_std_error_response",
    "create_success_response",
    "duplicate_name",
    "generate_agent_id",
    "generate_config_id",
    "generate_human_readable_id",
    "generate_prefixed_id",
    "generate_secure_id",
    "generate_session_id",
    "generate_short_uuid",
    "generate_timestamped_id",
    "generate_uuid",
    "generate_webhook_id",
    "get_storage",
    "handle_standard_exceptions",
    "internal_server_error",
    "invalid_field",
    "name_exists_error",
    "not_found_error",
    "paginate_items",
    "rate_limit_error",
    "resource_not_found",
    "sanitize_input",
    "session_id_generator",
    # Validation Utils
    "validate_correlation_id",
    "validate_email",
    "validate_id_format",
    "validate_json_payload",
    "validate_pagination_params",
    "validate_server_name",
    "validate_uuid",
    "validation_error",
    "webhook_id_generator",
]
