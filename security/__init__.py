# backend/security/__init__.py
"""Enhanced Security-System für Keiko Personal Assistant

Implementiert vollständige Authentication & Authorization mit OIDC/OAuth2,
RBAC/ABAC, mTLS, Tenant-Isolation und granularen Permission-Controls.
"""

from __future__ import annotations

from kei_logging import get_logger

# Enhanced OIDC/OAuth2
from .enhanced_oidc_client import (
    EnhancedOIDCClient,
    GrantType,
    OIDCDiscoveryDocument,
    ServiceAccountConfig,
    TokenInfo,
    TokenType,
)

# mTLS Support
from .mtls_manager import (
    CertificateInfo,
    CertificateStatus,
    CertificateStore,
    CertificateType,
    MTLSConfig,
    MTLSManager,
)

# RBAC/ABAC System
from .rbac_abac_system import (
    Action,
    AuthorizationContext,
    AuthorizationDecision,
    Permission,
    PermissionEffect,
    PolicyEngine,
    Principal,
    RBACAuthorizationService,
    ResourceType,
    Role,
    RoleRegistry,
    rbac_authorization_service,
)

# Scope & Permission Management
from .scope_permission_manager import (
    DelegationType,
    PermissionEvaluationContext,
    PermissionEvaluationResult,
    PermissionGrant,
    PermissionLevel,
    PermissionManager,
    Scope,
    ScopeRegistry,
    ScopeType,
    permission_manager,
    scope_registry,
)

# Tenant Isolation
from .tenant_isolation import (
    IsolationLevel,
    TenantConfig,
    TenantContext,
    TenantIsolationService,
    TenantRegistry,
    TenantResource,
    TenantStatus,
    TenantUsageStats,
    tenant_isolation_service,
)

# Enhanced Auth Middleware entfernt - ersetzt durch UnifiedAuthMiddleware
# from .enhanced_auth_middleware import (
#     AuthConfig,
#     AuthenticationResult,
#     EnhancedAuthMiddleware,
#     require_permission,
#     require_tenant_access,
# )

logger = get_logger(__name__)

# Package-Level Exports
__all__ = [
    "Action",
    # "AuthConfig",  # Entfernt - ersetzt durch UnifiedAuthMiddleware
    # "AuthenticationResult",  # Entfernt - ersetzt durch UnifiedAuthMiddleware
    "AuthorizationContext",
    "AuthorizationDecision",
    "CertificateInfo",
    "CertificateStatus",
    "CertificateStore",
    "CertificateType",
    "DelegationType",
    # Enhanced Auth Middleware entfernt - ersetzt durch UnifiedAuthMiddleware
    # "EnhancedAuthMiddleware",
    # Enhanced OIDC/OAuth2
    "EnhancedOIDCClient",
    "GrantType",
    "IsolationLevel",
    "MTLSConfig",
    # mTLS Support
    "MTLSManager",
    "OIDCDiscoveryDocument",
    "Permission",
    "PermissionEffect",
    "PermissionEvaluationContext",
    "PermissionEvaluationResult",
    "PermissionGrant",
    "PermissionLevel",
    "PermissionManager",
    "PolicyEngine",
    "Principal",
    "RBACAuthorizationService",
    "ResourceType",
    "Role",
    "RoleRegistry",
    "Scope",
    "ScopeRegistry",
    "ScopeType",
    "ServiceAccountConfig",
    "TenantConfig",
    "TenantContext",
    "TenantIsolationService",
    "TenantRegistry",
    "TenantResource",
    "TenantStatus",
    "TenantUsageStats",
    "TokenInfo",
    "TokenType",
    "permission_manager",
    # RBAC/ABAC System
    "rbac_authorization_service",
    # Scope & Permission Management
    "scope_registry",
    # Tenant Isolation
    "tenant_isolation_service",
]

# Security-System Status
def get_security_system_status() -> dict:
    """Gibt Status des Security-Systems zurück."""
    return {
        "package": "backend.security",
        "version": "1.0.0",
        "components": {
            "enhanced_oidc": True,
            "rbac_abac": True,
            "mtls_support": True,
            "tenant_isolation": True,
            "scope_permission_management": True,
            "enhanced_auth_middleware": True,
        },
        "features": {
            "oidc_oauth2_integration": True,
            "service_account_auth": True,
            "token_refresh_rotation": True,
            "hierarchical_rbac": True,
            "attribute_based_access_control": True,
            "mutual_tls": True,
            "certificate_management": True,
            "multi_tenant_isolation": True,
            "cross_tenant_access_control": True,
            "granular_scopes": True,
            "permission_delegation": True,
            "audit_logging": True,
            "rate_limiting": True,
            "security_headers": True,
        },
        "auth_methods": [
            "oidc_bearer_token",
            "service_account_credentials",
            "mutual_tls_certificates",
            "jwt_bearer_assertion"
        ],
        "authorization_models": [
            "role_based_access_control",
            "attribute_based_access_control",
            "scope_based_permissions",
            "tenant_isolation"
        ]
    }

logger.info(f"Enhanced Security-System geladen - Status: {get_security_system_status()}")
