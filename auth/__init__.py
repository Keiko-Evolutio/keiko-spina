"""Enterprise Authentication System.

Unified Authentication und Authorization für Keiko Personal Assistant.
Konsolidiert alle Authentication-Funktionalitäten in einem Clean Code-konformen Modul.

Usage:
    from auth import auth, require_auth, require_scope, Scope, PrivilegeLevel
    from auth import configure_auth_middleware

    # FastAPI Dependencies
    @app.get("/protected")
    async def protected_endpoint(context: AuthContext = Depends(require_auth)):
        return {"user": context.subject}

    # Scope-basierte Authorization
    @app.post("/admin")
    async def admin_endpoint(context: AuthContext = Depends(require_scope([Scope.AGENTS_ADMIN]))):
        return {"message": "Admin access granted"}

    # Middleware-Konfiguration
    configure_auth_middleware(app)
"""

# Legacy auth_middleware removed - using unified_enterprise_auth
# from .auth_middleware import (
#     # Configuration
#     DEFAULT_PUBLIC_PATHS,
#     DEFAULT_SCOPE_MAP,
#     # Middleware
#     AuthMiddleware,
#     configure_auth_middleware,
# )
from .enterprise_auth import (
    AuthContext,
    AuthResult,
    EnterpriseAuthenticator,
    PrivilegeLevel,
    Scope,
    # Core Classes
    TokenType,
    TokenValidator,
    # Global Instance
    auth,
    # FastAPI Dependencies
    require_auth,
    require_privilege,
    require_scope,
)

# Import unified enterprise auth system
from .unified_enterprise_auth import (
    AuthenticationMode,
    UnifiedAuthConfig,
    UnifiedEnterpriseAuth,
    require_unified_auth,
    require_websocket_auth,
    unified_auth,
)

__all__ = [
    # Legacy (commented out)
    # "DEFAULT_PUBLIC_PATHS",
    # "DEFAULT_SCOPE_MAP",
    # "AuthMiddleware",
    # "configure_auth_middleware",

    # Enterprise Auth
    "AuthContext",
    "AuthResult",
    "EnterpriseAuthenticator",
    "PrivilegeLevel",
    "Scope",
    "TokenType",
    "TokenValidator",
    "auth",
    "require_auth",
    "require_privilege",
    "require_scope",

    # Unified Enterprise Auth
    "UnifiedEnterpriseAuth",
    "UnifiedAuthConfig",
    "AuthenticationMode",
    "unified_auth",
    "require_unified_auth",
    "require_websocket_auth",
]
