# backend/security/constants.py
"""Security-Konstanten für Keiko Personal Assistant

Zentrale Definition aller Timeouts, Limits, Status Codes und anderen
konstanten Werte für das Security-System.
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# HTTP Status Codes
# =============================================================================

class SecurityHTTPStatus:
    """HTTP Status Codes für Security-Operationen."""

    # Success
    OK: Final[int] = 200
    CREATED: Final[int] = 201
    NO_CONTENT: Final[int] = 204

    # Client Errors
    BAD_REQUEST: Final[int] = 400
    UNAUTHORIZED: Final[int] = 401
    FORBIDDEN: Final[int] = 403
    NOT_FOUND: Final[int] = 404
    METHOD_NOT_ALLOWED: Final[int] = 405
    CONFLICT: Final[int] = 409
    UNPROCESSABLE_ENTITY: Final[int] = 422
    TOO_MANY_REQUESTS: Final[int] = 429

    # Server Errors
    INTERNAL_SERVER_ERROR: Final[int] = 500
    BAD_GATEWAY: Final[int] = 502
    SERVICE_UNAVAILABLE: Final[int] = 503
    GATEWAY_TIMEOUT: Final[int] = 504


# =============================================================================
# Timeouts und TTL-Werte
# =============================================================================

class SecurityTimeouts:
    """Timeout-Konstanten für Security-Operationen."""

    # HTTP-Timeouts (Sekunden)
    HTTP_REQUEST_TIMEOUT: Final[int] = 30
    HTTP_CONNECT_TIMEOUT: Final[int] = 10
    HTTP_READ_TIMEOUT: Final[int] = 30

    # Cache-TTL (Sekunden)
    DISCOVERY_CACHE_TTL: Final[int] = 3600  # 1 Stunde
    JWKS_CACHE_TTL: Final[int] = 3600       # 1 Stunde
    ROLE_HIERARCHY_CACHE_TTL: Final[int] = 300  # 5 Minuten
    PERMISSION_CACHE_TTL: Final[int] = 300      # 5 Minuten
    TOKEN_VALIDATION_CACHE_TTL: Final[int] = 300  # 5 Minuten

    # Token-Timeouts
    TOKEN_REFRESH_THRESHOLD_SECONDS: Final[int] = 300  # 5 Minuten vor Ablauf
    TOKEN_VALIDATION_TIMEOUT: Final[int] = 10

    # Rate Limiting
    RATE_LIMIT_WINDOW_SECONDS: Final[int] = 3600  # 1 Stunde
    RATE_LIMIT_CLEANUP_INTERVAL: Final[int] = 300  # 5 Minuten

    # Certificate Rotation
    CERT_ROTATION_CHECK_INTERVAL_HOURS: Final[int] = 24
    CERT_ROTATION_THRESHOLD_DAYS: Final[int] = 30

    # HMAC Validation
    HMAC_TIMESTAMP_TOLERANCE_SECONDS: Final[int] = 300  # 5 Minuten


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimitDefaults:
    """Standard-Rate-Limits für verschiedene Operationen."""

    # API-Zugriffe pro Stunde
    DEFAULT_API_CALLS_PER_HOUR: Final[int] = 10000
    ADMIN_API_CALLS_PER_HOUR: Final[int] = 100000
    SERVICE_ACCOUNT_API_CALLS_PER_HOUR: Final[int] = 50000

    # Authentifizierungs-Versuche
    AUTH_ATTEMPTS_PER_MINUTE: Final[int] = 10
    AUTH_ATTEMPTS_PER_HOUR: Final[int] = 100

    # Token-Operationen
    TOKEN_REFRESH_PER_HOUR: Final[int] = 100
    TOKEN_VALIDATION_PER_MINUTE: Final[int] = 1000

    # OIDC Discovery
    DISCOVERY_REQUESTS_PER_HOUR: Final[int] = 10

    # mTLS Operations
    MTLS_HANDSHAKES_PER_MINUTE: Final[int] = 100


# =============================================================================
# Ressourcen-Limits
# =============================================================================

class ResourceLimits:
    """Ressourcen-Limits für Tenants und System."""

    # Tenant-Limits (Default)
    DEFAULT_MAX_AGENTS: Final[int] = 100
    DEFAULT_MAX_CONCURRENT_TASKS: Final[int] = 1000
    DEFAULT_MAX_STORAGE_MB: Final[int] = 10240  # 10GB

    # System-Limits
    MAX_AUDIT_ENTRIES: Final[int] = 10000
    MAX_CACHE_ENTRIES: Final[int] = 10000
    MAX_PERMISSION_GRANTS_PER_PRINCIPAL: Final[int] = 1000

    # Admin/Super-User Limits
    ADMIN_MAX_AGENTS: Final[int] = 1000
    ADMIN_MAX_CONCURRENT_TASKS: Final[int] = 10000
    ADMIN_MAX_STORAGE_MB: Final[int] = 102400  # 100GB


# =============================================================================
# Security Headers
# =============================================================================

class SecurityHeaders:
    """Standard Security Headers."""

    # Header Names
    CONTENT_SECURITY_POLICY: Final[str] = "Content-Security-Policy"
    STRICT_TRANSPORT_SECURITY: Final[str] = "Strict-Transport-Security"
    X_CONTENT_TYPE_OPTIONS: Final[str] = "X-Content-Type-Options"
    X_FRAME_OPTIONS: Final[str] = "X-Frame-Options"
    X_XSS_PROTECTION: Final[str] = "X-XSS-Protection"
    REFERRER_POLICY: Final[str] = "Referrer-Policy"

    # Header Values
    CSP_DEFAULT: Final[str] = "default-src 'self'"
    HSTS_DEFAULT: Final[str] = "max-age=31536000; includeSubDomains"
    NOSNIFF: Final[str] = "nosniff"
    DENY_FRAME: Final[str] = "DENY"
    XSS_BLOCK: Final[str] = "1; mode=block"
    REFERRER_STRICT: Final[str] = "strict-origin-when-cross-origin"


# =============================================================================
# Kryptographische Konstanten
# =============================================================================

class CryptoConstants:
    """Kryptographische Konstanten."""

    # Algorithmen
    JWT_ALGORITHM_RS256: Final[str] = "RS256"
    JWT_ALGORITHM_HS256: Final[str] = "HS256"
    JWT_ALGORITHM_ES256: Final[str] = "ES256"

    # Hash-Algorithmen
    HASH_ALGORITHM_SHA256: Final[str] = "sha256"
    HASH_ALGORITHM_SHA512: Final[str] = "sha512"

    # Key-Größen
    RSA_KEY_SIZE_2048: Final[int] = 2048
    RSA_KEY_SIZE_4096: Final[int] = 4096

    # HMAC
    HMAC_ALGORITHM: Final[str] = "sha256"


# =============================================================================
# Standard-Scopes und Permissions
# =============================================================================

class StandardScopes:
    """Standard-Scopes für das System."""

    # Agent-Scopes
    AGENT_READ: Final[str] = "agent:read"
    AGENT_WRITE: Final[str] = "agent:write"
    AGENT_EXECUTE: Final[str] = "agent:execute"
    AGENT_MANAGE: Final[str] = "agent:manage"
    AGENT_ADMIN: Final[str] = "agent:admin"

    # Webhook-Scopes
    WEBHOOK_INBOUND_RECEIVE: Final[str] = "webhook:inbound:receive"
    WEBHOOK_OUTBOUND_SEND: Final[str] = "webhook:outbound:send"
    WEBHOOK_ADMIN: Final[str] = "webhook:admin:*"

    # System-Scopes
    SYSTEM_READ: Final[str] = "system:read"
    SYSTEM_WRITE: Final[str] = "system:write"
    SYSTEM_ADMIN: Final[str] = "system:admin"

    # Tenant-Scopes
    TENANT_READ: Final[str] = "tenant:read"
    TENANT_WRITE: Final[str] = "tenant:write"
    TENANT_MANAGE: Final[str] = "tenant:manage"


# =============================================================================
# Error Messages
# =============================================================================

class SecurityErrorMessages:
    """Standard-Fehlermeldungen für Security-Operationen."""

    # Authentication
    AUTH_REQUIRED: Final[str] = "Authentication required"
    AUTH_INVALID_TOKEN: Final[str] = "Invalid or expired token"
    AUTH_INVALID_CREDENTIALS: Final[str] = "Invalid credentials"
    AUTH_TOKEN_EXPIRED: Final[str] = "Token has expired"

    # Authorization
    AUTHZ_INSUFFICIENT_PERMISSIONS: Final[str] = "Insufficient permissions"
    AUTHZ_TENANT_MISMATCH: Final[str] = "Tenant access denied"
    AUTHZ_SCOPE_REQUIRED: Final[str] = "Required scope missing"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED: Final[str] = "Rate limit exceeded"
    RATE_LIMIT_BLOCKED: Final[str] = "IP address blocked due to rate limiting"

    # mTLS
    MTLS_CERT_INVALID: Final[str] = "Invalid client certificate"
    MTLS_CERT_EXPIRED: Final[str] = "Client certificate expired"
    MTLS_CA_VALIDATION_FAILED: Final[str] = "Certificate authority validation failed"

    # HMAC
    HMAC_SIGNATURE_INVALID: Final[str] = "Invalid HMAC signature"
    HMAC_TIMESTAMP_INVALID: Final[str] = "Invalid or expired timestamp"

    # General
    INTERNAL_ERROR: Final[str] = "Internal security error"
    CONFIGURATION_ERROR: Final[str] = "Security configuration error"


# =============================================================================
# Default Paths und URLs
# =============================================================================

class SecurityPaths:
    """Standard-Pfade für Security-Operationen."""

    # OIDC Discovery
    OIDC_DISCOVERY_PATH: Final[str] = "/.well-known/openid_configuration"
    JWKS_PATH: Final[str] = "/.well-known/jwks.json"

    # Health Checks
    HEALTH_PATH: Final[str] = "/health"
    METRICS_PATH: Final[str] = "/metrics"

    # Documentation
    DOCS_PATH: Final[str] = "/docs"
    OPENAPI_PATH: Final[str] = "/openapi.json"
    REDOC_PATH: Final[str] = "/redoc"

    # Excluded from Auth
    AUTH_EXCLUDED_PATHS: Final[tuple] = (
        "/health", "/metrics", "/docs", "/openapi.json", "/redoc",
        "/.well-known/openid_configuration", "/.well-known/jwks.json"
    )


# =============================================================================
# Environment Variable Names
# =============================================================================

class EnvVarNames:
    """Namen der Umgebungsvariablen."""

    # OIDC/OAuth2
    OIDC_ISSUER_URL: Final[str] = "OIDC_ISSUER_URL"
    OIDC_CLIENT_ID: Final[str] = "OIDC_CLIENT_ID"
    OIDC_CLIENT_SECRET: Final[str] = "OIDC_CLIENT_SECRET"

    # API Tokens
    KEI_MCP_API_TOKEN: Final[str] = "KEI_MCP_API_TOKEN"
    EXTERNAL_MCP_API_TOKEN: Final[str] = "EXTERNAL_MCP_API_TOKEN"
    KEI_API_TOKEN: Final[str] = "KEI_API_TOKEN"

    # mTLS
    MTLS_ENABLED: Final[str] = "MTLS_ENABLED"
    MTLS_CA_CERT_PATH: Final[str] = "MTLS_CA_CERT_PATH"
    MTLS_CLIENT_CERT_PATH: Final[str] = "MTLS_CLIENT_CERT_PATH"
    MTLS_CLIENT_KEY_PATH: Final[str] = "MTLS_CLIENT_KEY_PATH"

    # RBAC
    RBAC_ROLES_JSON: Final[str] = "RBAC_ROLES_JSON"
    RBAC_ASSIGNMENTS_JSON: Final[str] = "RBAC_ASSIGNMENTS_JSON"

    # HMAC
    N8N_HMAC_SECRET: Final[str] = "N8N_HMAC_SECRET"

    # Azure Key Vault
    AZURE_KEY_VAULT_URL: Final[str] = "AZURE_KEY_VAULT_URL"
    AZURE_KEY_VAULT_SECRET_NAME: Final[str] = "AZURE_KEY_VAULT_SECRET_NAME"


__all__ = [
    "CryptoConstants",
    "EnvVarNames",
    "RateLimitDefaults",
    "ResourceLimits",
    "SecurityErrorMessages",
    "SecurityHTTPStatus",
    "SecurityHeaders",
    "SecurityPaths",
    "SecurityTimeouts",
    "StandardScopes",
]
