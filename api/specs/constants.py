# backend/api/specs/constants.py
"""Konstanten für OpenAPI-Spezifikationen.

Zentralisiert alle Magic Numbers, Hard-coded Strings und Konfigurationswerte
für bessere Wartbarkeit und Konsistenz.
"""

from __future__ import annotations

from typing import Final

# API-Versionierung
API_VERSION: Final[str] = "v1"
KEI_RPC_BASE_PATH: Final[str] = f"/api/{API_VERSION}/api-grpc"

# Timeout-Konfiguration
DEFAULT_TIMEOUT_SECONDS: Final[int] = 60
MIN_TIMEOUT_SECONDS: Final[int] = 1
MAX_TIMEOUT_SECONDS: Final[int] = 300

# Header-Konfiguration
IDEMPOTENCY_KEY_MAX_LENGTH: Final[int] = 255

# Prioritäts-Levels
PRIORITY_LEVELS: Final[tuple[str, ...]] = ("low", "normal", "high", "critical")
DEFAULT_PRIORITY: Final[str] = "normal"

# HTTP-Status-Codes
HTTP_STATUS_OK: Final[str] = "200"
HTTP_STATUS_CREATED: Final[str] = "201"
HTTP_STATUS_BAD_REQUEST: Final[str] = "400"
HTTP_STATUS_UNAUTHORIZED: Final[str] = "401"
HTTP_STATUS_FORBIDDEN: Final[str] = "403"
HTTP_STATUS_TIMEOUT: Final[str] = "408"
HTTP_STATUS_RATE_LIMITED: Final[str] = "429"
HTTP_STATUS_INTERNAL_ERROR: Final[str] = "500"
HTTP_STATUS_SERVICE_UNAVAILABLE: Final[str] = "503"

# OpenAPI-Tags
KEI_RPC_TAG: Final[str] = "api-grpc"
HEALTH_TAG: Final[str] = "health"
MONITORING_TAG: Final[str] = "monitoring"

# Content-Types
CONTENT_TYPE_JSON: Final[str] = "application/json"

# KEI-RPC Operationen
KEI_RPC_OPERATIONS: Final[tuple[str, ...]] = ("plan", "act", "observe", "explain")

# Standard-Header-Namen
HEADER_TRACEPARENT: Final[str] = "traceparent"
HEADER_TRACESTATE: Final[str] = "tracestate"
HEADER_IDEMPOTENCY_KEY: Final[str] = "Idempotency-Key"
HEADER_PRIORITY: Final[str] = "X-Priority"
HEADER_TIMEOUT: Final[str] = "X-Timeout"
HEADER_CORRELATION_ID: Final[str] = "X-Correlation-ID"
HEADER_OPERATION_ID: Final[str] = "X-Operation-ID"
HEADER_AGENT_ID: Final[str] = "X-Agent-ID"
HEADER_DURATION_MS: Final[str] = "X-Duration-MS"

# Schema-Referenzen
SCHEMA_REF_ERROR_RESPONSE: Final[str] = "#/components/schemas/ErrorResponse"
SCHEMA_REF_PLAN_REQUEST: Final[str] = "#/components/schemas/PlanRequest"
SCHEMA_REF_PLAN_RESPONSE: Final[str] = "#/components/schemas/PlanResponse"
SCHEMA_REF_ACT_REQUEST: Final[str] = "#/components/schemas/ActRequest"
SCHEMA_REF_ACT_RESPONSE: Final[str] = "#/components/schemas/ActResponse"
SCHEMA_REF_OBSERVE_REQUEST: Final[str] = "#/components/schemas/ObserveRequest"
SCHEMA_REF_OBSERVE_RESPONSE: Final[str] = "#/components/schemas/ObserveResponse"
SCHEMA_REF_EXPLAIN_REQUEST: Final[str] = "#/components/schemas/ExplainRequest"
SCHEMA_REF_EXPLAIN_RESPONSE: Final[str] = "#/components/schemas/ExplainResponse"

# Component-Referenzen
COMPONENT_REF_BAD_REQUEST: Final[str] = "#/components/responses/BadRequest"
COMPONENT_REF_UNAUTHORIZED: Final[str] = "#/components/responses/Unauthorized"
COMPONENT_REF_FORBIDDEN: Final[str] = "#/components/responses/Forbidden"
COMPONENT_REF_TIMEOUT: Final[str] = "#/components/responses/Timeout"
COMPONENT_REF_RATE_LIMITED: Final[str] = "#/components/responses/RateLimited"
COMPONENT_REF_INTERNAL_ERROR: Final[str] = "#/components/responses/InternalError"

# Parameter-Referenzen
PARAM_REF_TRACEPARENT: Final[str] = "#/components/parameters/TraceparentHeader"
PARAM_REF_TRACESTATE: Final[str] = "#/components/parameters/TracestateHeader"
PARAM_REF_IDEMPOTENCY_KEY: Final[str] = "#/components/parameters/IdempotencyKeyHeader"
PARAM_REF_PRIORITY: Final[str] = "#/components/parameters/PriorityHeader"
PARAM_REF_TIMEOUT: Final[str] = "#/components/parameters/TimeoutHeader"

# Security-Schema-Namen
SECURITY_BEARER_AUTH: Final[str] = "BearerAuth"
SECURITY_MTLS: Final[str] = "mTLS"

# Beispiel-Werte
EXAMPLE_TRACEPARENT: Final[str] = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

# Service-Status
SERVICE_STATUS_HEALTHY: Final[str] = "healthy"
SERVICE_STATUS_UNHEALTHY: Final[str] = "unhealthy"
OPERATION_STATUS_FAILED: Final[str] = "failed"

# Endpunkt-Pfade
def get_kei_rpc_endpoint_path(operation: str) -> str:
    """Generiert KEI-RPC Endpunkt-Pfad für gegebene Operation.

    Args:
        operation: KEI-RPC Operation (plan, act, observe, explain)

    Returns:
        Vollständiger Endpunkt-Pfad

    Raises:
        ValueError: Wenn Operation nicht unterstützt wird
    """
    if operation not in KEI_RPC_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}. Supported: {KEI_RPC_OPERATIONS}")

    return f"{KEI_RPC_BASE_PATH}/{operation}"


# Health/Status Endpunkte
KEI_RPC_HEALTH_PATH: Final[str] = f"{KEI_RPC_BASE_PATH}/health"
KEI_RPC_STATUS_PATH: Final[str] = f"{KEI_RPC_BASE_PATH}/status"
