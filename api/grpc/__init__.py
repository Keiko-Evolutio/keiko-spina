"""gRPC API Package.

Migriert aus kei_rpc/ nach api/grpc/ als Teil der kei_*-Module-Konsolidierung.

Implementiert plan/act/observe/explain Operationen mit W3C Trace-Propagation,
Idempotenz-Unterstützung und Capability-basiertem Agent-Routing.

Hauptkomponenten:
- gRPC Server mit Interceptors
- Agent-Integration für plan/act/observe/explain
- Rate Limiting und Authentication
- Error Mapping und Monitoring
"""

from __future__ import annotations

import warnings

from kei_logging import get_logger

from .auth_interceptor import AuthInterceptor

# Base Components
from .base_interceptor import BaseInterceptor, NoOpInterceptor

# Constants
from .constants import (
    AuthConfig,
    DLPConfig,
    ErrorCodes,
    ErrorMessages,
    GRPCServerConfig,
    MetadataKeys,
    RateLimitConfig,
    ServiceNames,
    TracingConfig,
)

# Error Mapping
from .error_mapping import ErrorMappingInterceptor

# Interceptors
from .interceptors import (
    DeadlineInterceptor,
    DLPInterceptor,
    MetricsInterceptor,
    TracingInterceptor,
    create_interceptor_chain,
)

# Models (mit verbesserter Type Safety)
from .models import (  # Enums; Base Models; Request Models; Response Models; Utility Models
    ActRequest,
    ActResponse,
    AgentContext,
    ExplainRequest,
    ExplainResponse,
    ObserveRequest,
    ObserveResponse,
    OperationError,
    OperationMetadata,
    OperationStatus,
    OperationSummary,
    OperationTiming,
    OperationType,
    PlanRequest,
    PlanResponse,
    PriorityLevel,
    TraceContext,
)
from .rate_limit_interceptor import RateLimitInterceptor, TokenBucket

# Server Components
from .server_factory import GRPCServerFactory, GRPCServerManager, serve_grpc, shutdown_grpc

# Core Service
from .service import (
    AgentNotFoundError,
    CapabilityNotAvailableError,
    KEIRPCService,
    KEIRPCServiceError,
    OperationRouter,
    OperationTimeoutError,
    kei_rpc_service,
)

# Utilities
from .utils import (
    create_grpc_error,
    create_timing_info,
    extract_bearer_token,
    extract_correlation_id,
    extract_metadata_value,
    extract_peer_info,
    extract_peer_ip,
    extract_tenant_id,
    handle_common_errors,
    log_operation_end,
    log_operation_start,
    validate_idempotency_key,
    validate_w3c_traceparent,
)

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "ActRequest",
    "ActResponse",
    "AgentContext",
    "AgentNotFoundError",
    "AuthConfig",
    "AuthInterceptor",
    # Interceptors
    "BaseInterceptor",
    "CapabilityNotAvailableError",
    "DLPConfig",
    "DLPInterceptor",
    "DeadlineInterceptor",
    "ErrorCodes",
    "ErrorMappingInterceptor",
    "ErrorMessages",
    "ExplainRequest",
    "ExplainResponse",
    # Constants
    "GRPCServerConfig",
    # Server Components
    "GRPCServerFactory",
    "GRPCServerManager",
    # Core Service
    "KEIRPCService",
    # Exceptions
    "KEIRPCServiceError",
    "MetadataKeys",
    "MetricsInterceptor",
    "NoOpInterceptor",
    "ObserveRequest",
    "ObserveResponse",
    "OperationError",
    "OperationMetadata",
    "OperationRouter",
    "OperationStatus",
    # Utility Models
    "OperationSummary",
    "OperationTimeoutError",
    "OperationTiming",
    # Enums
    "OperationType",
    # Request Models
    "PlanRequest",
    # Response Models
    "PlanResponse",
    "PriorityLevel",
    "RateLimitConfig",
    "RateLimitInterceptor",
    "ServiceNames",
    "TokenBucket",
    # Base Models
    "TraceContext",
    "TracingConfig",
    "TracingInterceptor",
    "create_grpc_error",
    "create_interceptor_chain",
    "create_timing_info",
    "extract_bearer_token",
    "extract_correlation_id",
    # Utilities
    "extract_metadata_value",
    "extract_peer_info",
    "extract_peer_ip",
    "extract_tenant_id",
    "handle_common_errors",
    "kei_rpc_service",
    "log_operation_end",
    "log_operation_start",
    "serve_grpc",
    "shutdown_grpc",
    "validate_idempotency_key",
    "validate_w3c_traceparent",
]


# ============================================================================
# PACKAGE STATUS UND HEALTH
# ============================================================================


def get_kei_rpc_status() -> dict:
    """Gibt Status des KEI-RPC Systems zurück.

    Returns:
        Status-Dictionary
    """
    return {
        "package": "kei_rpc",
        "version": "2.0.0",
        "operations": ["plan", "act", "observe", "explain"],
        "features": {
            "w3c_trace_propagation": True,
            "idempotency_support": True,
            "capability_routing": True,
            "agent_discovery": True,
            "fallback_mechanisms": True,
            "error_handling": True,
            "base_interceptor_pattern": True,
            "consolidated_utilities": True,
            "improved_type_safety": True,
        },
        "components": {
            "service": True,
            "models": True,
            "agent_integration": True,
            "api_routes": True,
            "grpc_server": True,
            "interceptors": True,
            "utilities": True,
            "constants": True,
        },

    }





# ============================================================================
# PACKAGE INITIALIZATION
# ============================================================================

# Log package status
status = get_kei_rpc_status()
logger.info(
    f"KEI-RPC Interface geladen v{status['version']} - "
    f"Features: {len(status['features'])}, "
    f"Components: {len(status['components'])}"
)


# ============================================================================
# BACKWARD COMPATIBILITY WARNINGS
# ============================================================================




def _warn_deprecated_import(old_name: str, new_name: str) -> None:
    """Warnt vor deprecated Imports."""
    warnings.warn(
        f"{old_name} ist deprecated. Verwende {new_name} stattdessen.",
        DeprecationWarning,
        stacklevel=3,
    )


# Beispiel für Backward Compatibility (falls benötigt)
# def get_legacy_interceptors():
#     _warn_deprecated_import("get_legacy_interceptors", "create_interceptor_chain")
#     return create_interceptor_chain()
