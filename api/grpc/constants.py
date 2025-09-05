"""Konstanten für das KEI-RPC Modul.

Zentralisiert alle Magic Numbers, Hard-coded Strings und Konfigurationswerte
für bessere Wartbarkeit und Konfigurierbarkeit.
"""

from __future__ import annotations

import os
from typing import Final

# ============================================================================
# GRPC SERVER KONFIGURATION
# ============================================================================


class GRPCServerConfig:
    """Konfiguration für gRPC Server."""

    # Server-Adressen
    DEFAULT_BIND_ADDRESS: Final[str] = "0.0.0.0:50051"
    DEFAULT_HEALTH_CHECK_PORT: Final[int] = 8080

    # Performance-Einstellungen
    DEFAULT_MAX_WORKERS: Final[int] = 10
    DEFAULT_MAX_INFLIGHT_REQUESTS: Final[int] = 10
    DEFAULT_GRACE_PERIOD_SECONDS: Final[int] = 30

    # Timeouts
    DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[int] = 120
    DEFAULT_KEEPALIVE_TIME_MS: Final[int] = 30000
    DEFAULT_KEEPALIVE_TIMEOUT_MS: Final[int] = 5000

    # mTLS Konfiguration
    MTLS_CERT_PATH: Final[str] = os.getenv("KEI_GRPC_CERT_PATH", "/certs/server.crt")
    MTLS_KEY_PATH: Final[str] = os.getenv("KEI_GRPC_KEY_PATH", "/certs/server.key")
    MTLS_CA_PATH: Final[str] = os.getenv("KEI_GRPC_CA_PATH", "/certs/ca.crt")


# ============================================================================
# METADATA KEYS
# ============================================================================


class MetadataKeys:
    """Standard Metadata-Keys für gRPC Requests."""

    # Authentication
    AUTHORIZATION: Final[str] = "authorization"
    BEARER_PREFIX: Final[str] = "Bearer "

    # Scopes und Permissions
    SCOPES: Final[str] = "x-scopes"
    TENANT_ID: Final[str] = "x-tenant-id"
    USER_ID: Final[str] = "x-user-id"

    # Tracing und Monitoring
    TRACE_ID: Final[str] = "x-trace-id"
    SPAN_ID: Final[str] = "x-span-id"
    CORRELATION_ID: Final[str] = "x-correlation-id"

    # Request-Kontrolle
    TIME_BUDGET: Final[str] = "x-time-budget-ms"
    IDEMPOTENCY_KEY: Final[str] = "idempotency-key"
    PRIORITY: Final[str] = "x-priority"

    # Rate Limiting
    RATE_LIMIT_REMAINING: Final[str] = "x-ratelimit-remaining"
    RATE_LIMIT_RESET: Final[str] = "x-ratelimit-reset"
    RATE_LIMIT_LIMIT: Final[str] = "x-ratelimit-limit"

    # Error Handling
    ERROR_CODE: Final[str] = "kei-error-code"
    ERROR_SEVERITY: Final[str] = "kei-error-severity"
    ERROR_CORRELATION_ID: Final[str] = "kei-error-correlation-id"


# ============================================================================
# RATE LIMITING KONFIGURATION
# ============================================================================


class RateLimitConfig:
    """Konfiguration für Rate Limiting."""

    # Standard-Limits
    DEFAULT_REQUESTS_PER_MINUTE: Final[int] = int(
        os.getenv("KEI_GRPC_DEFAULT_LIMIT_PER_MIN", "600")
    )
    DEFAULT_BURST_SIZE: Final[int] = int(os.getenv("KEI_GRPC_BURST_SIZE", "100"))

    # Token Bucket Konfiguration
    TOKEN_REFILL_RATE: Final[float] = DEFAULT_REQUESTS_PER_MINUTE / 60.0  # Pro Sekunde
    TOKEN_BUCKET_CAPACITY: Final[int] = DEFAULT_BURST_SIZE

    # Cleanup-Intervalle
    BUCKET_CLEANUP_INTERVAL_SECONDS: Final[int] = 300  # 5 Minuten
    BUCKET_EXPIRY_SECONDS: Final[int] = 3600  # 1 Stunde

    # Method-spezifische Limits (können über ENV überschrieben werden)
    METHOD_LIMITS: Final[dict[str, int]] = {
        "/kei.rpc.v1.KEIRPCService/Plan": 60,  # 1 pro Sekunde
        "/kei.rpc.v1.KEIRPCService/Act": 120,  # 2 pro Sekunde
        "/kei.rpc.v1.KEIRPCService/Observe": 300,  # 5 pro Sekunde
        "/kei.rpc.v1.KEIRPCService/Explain": 180,  # 3 pro Sekunde
        "*": DEFAULT_REQUESTS_PER_MINUTE,  # Fallback
    }


# ============================================================================
# AUTHENTICATION KONFIGURATION
# ============================================================================


class AuthConfig:
    """Konfiguration für Authentication."""

    # Token-Validierung
    STATIC_TOKENS: Final[set] = {
        os.getenv("KEI_GRPC_STATIC_TOKEN", "dev-token-123"),
        os.getenv("KEI_GRPC_ADMIN_TOKEN", "admin-token-456"),
    }

    # OIDC/JWT Konfiguration
    OIDC_ISSUER: Final[str] = os.getenv("KEI_RPC_OIDC_ISSUER", "")
    OIDC_AUDIENCE: Final[str] = os.getenv("KEI_RPC_OIDC_AUD", "")
    OIDC_JWKS_URI: Final[str] = os.getenv("KEI_RPC_OIDC_JWKS_URI", "")

    # Required Scopes
    REQUIRED_SCOPES: Final[dict[str, str]] = {
        "Plan": "kei.rpc.plan",
        "Act": "kei.rpc.act",
        "Observe": "kei.rpc.observe",
        "Explain": "kei.rpc.explain",
        "CreateResource": "kei.rpc.write",
        "UpdateResource": "kei.rpc.write",
        "DeleteResource": "kei.rpc.write",
        "ListResources": "kei.rpc.read",
        "GetResource": "kei.rpc.read",
    }


# ============================================================================
# SERVICE NAMES UND PATHS
# ============================================================================


class ServiceNames:
    """Service-Namen für gRPC Registration."""

    KEI_RPC_SERVICE: Final[str] = "kei.rpc.v1.KEIRPCService"
    KEI_STREAM_SERVICE: Final[str] = "kei.stream.v1.KEIStreamService"
    HEALTH_SERVICE: Final[str] = "grpc.health.v1.Health"

    # Service-Pfade
    KEI_RPC_PLAN: Final[str] = f"/{KEI_RPC_SERVICE}/Plan"
    KEI_RPC_ACT: Final[str] = f"/{KEI_RPC_SERVICE}/Act"
    KEI_RPC_OBSERVE: Final[str] = f"/{KEI_RPC_SERVICE}/Observe"
    KEI_RPC_EXPLAIN: Final[str] = f"/{KEI_RPC_SERVICE}/Explain"


# ============================================================================
# ERROR CODES UND MESSAGES
# ============================================================================


class ErrorCodes:
    """Standard Error-Codes für KEI-RPC."""

    # Authentication Errors
    AUTH_TOKEN_MISSING: Final[str] = "AUTH_TOKEN_MISSING"
    AUTH_TOKEN_INVALID: Final[str] = "AUTH_TOKEN_INVALID"
    AUTH_SCOPE_INSUFFICIENT: Final[str] = "AUTH_SCOPE_INSUFFICIENT"

    # Rate Limiting Errors
    RATE_LIMIT_EXCEEDED: Final[str] = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED: Final[str] = "QUOTA_EXCEEDED"

    # Agent Errors
    AGENT_NOT_FOUND: Final[str] = "AGENT_NOT_FOUND"
    AGENT_UNAVAILABLE: Final[str] = "AGENT_UNAVAILABLE"
    CAPABILITY_NOT_AVAILABLE: Final[str] = "CAPABILITY_NOT_AVAILABLE"

    # Operation Errors
    OPERATION_TIMEOUT: Final[str] = "OPERATION_TIMEOUT"
    OPERATION_FAILED: Final[str] = "OPERATION_FAILED"
    VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"

    # System Errors
    SERVICE_UNAVAILABLE: Final[str] = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR: Final[str] = "INTERNAL_ERROR"


class ErrorMessages:
    """Standard Error-Messages für KEI-RPC."""

    AUTH_TOKEN_MISSING: Final[str] = "Authorization token fehlt"
    AUTH_TOKEN_INVALID: Final[str] = "Authorization token ist ungültig"
    AUTH_SCOPE_INSUFFICIENT: Final[str] = "Unzureichende Berechtigung für Operation"

    RATE_LIMIT_EXCEEDED: Final[str] = "Rate Limit überschritten"
    QUOTA_EXCEEDED: Final[str] = "Quota überschritten"

    AGENT_NOT_FOUND: Final[str] = "Agent nicht gefunden"
    AGENT_UNAVAILABLE: Final[str] = "Agent nicht verfügbar"
    CAPABILITY_NOT_AVAILABLE: Final[str] = "Capability nicht verfügbar"

    OPERATION_TIMEOUT: Final[str] = "Operation Timeout"
    OPERATION_FAILED: Final[str] = "Operation fehlgeschlagen"
    VALIDATION_ERROR: Final[str] = "Validierungsfehler"

    SERVICE_UNAVAILABLE: Final[str] = "Service nicht verfügbar"
    INTERNAL_ERROR: Final[str] = "Interner Fehler"


# ============================================================================
# TRACING UND MONITORING
# ============================================================================


class TracingConfig:
    """Konfiguration für Tracing und Monitoring."""

    # OpenTelemetry
    OTEL_SERVICE_NAME: Final[str] = "kei-rpc-grpc"
    OTEL_SERVICE_VERSION: Final[str] = "1.0.0"

    # Span-Namen
    SPAN_INTERCEPTOR_AUTH: Final[str] = "grpc.interceptor.auth"
    SPAN_INTERCEPTOR_RATE_LIMIT: Final[str] = "grpc.interceptor.rate_limit"
    SPAN_INTERCEPTOR_TRACING: Final[str] = "grpc.interceptor.tracing"
    SPAN_INTERCEPTOR_DLP: Final[str] = "grpc.interceptor.dlp"
    SPAN_INTERCEPTOR_METRICS: Final[str] = "grpc.interceptor.metrics"

    # Metrics-Namen
    METRIC_REQUESTS_TOTAL: Final[str] = "rpc.grpc.requests"
    METRIC_ERRORS_TOTAL: Final[str] = "rpc.grpc.errors"
    METRIC_DURATION: Final[str] = "rpc.grpc.duration_ms"
    METRIC_RATE_LIMIT_HITS: Final[str] = "rpc.grpc.rate_limit_hits"


# ============================================================================
# DLP UND REDACTION
# ============================================================================


class DLPConfig:
    """Konfiguration für Data Loss Prevention."""

    # PII-Patterns
    PII_PATTERNS: Final[dict[str, str]] = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }

    # Redaction-Einstellungen
    REDACTION_CHAR: Final[str] = "*"
    REDACTION_ENABLED: Final[bool] = os.getenv("KEI_GRPC_DLP_ENABLED", "true").lower() == "true"

    # Field-Masken pro Methode
    FIELD_MASKS: Final[dict[str, list]] = {
        "/kei.rpc.v1.KEIRPCService/Plan": ["objective", "constraints"],
        "/kei.rpc.v1.KEIRPCService/Act": ["action", "parameters"],
        "/kei.rpc.v1.KEIRPCService/Observe": ["observation"],
        "/kei.rpc.v1.KEIRPCService/Explain": ["explanation"],
    }


# ============================================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================================


def get_env_bool(key: str, default: bool = False) -> bool:
    """Hilfsfunktion für Boolean Environment Variables."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int) -> int:
    """Hilfsfunktion für Integer Environment Variables."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """Hilfsfunktion für Float Environment Variables."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AuthConfig",
    "DLPConfig",
    "ErrorCodes",
    "ErrorMessages",
    "GRPCServerConfig",
    "MetadataKeys",
    "RateLimitConfig",
    "ServiceNames",
    "TracingConfig",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
]
