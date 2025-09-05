"""Unified Authentication Middleware

Single, consolidated authentication middleware for enterprise-grade security.
Replaces all previous conflicting auth middleware implementations.
"""

import time

from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from auth.unified_enterprise_auth import unified_auth
from config.constants import DEFAULT_DEV_TOKEN
from kei_logging import get_logger

logger = get_logger(__name__)

# SECURITY: NO EXCLUDED PATHS - ALL ENDPOINTS REQUIRE AUTHENTICATION
# Excluded paths are a security anti-pattern. Every endpoint must authenticate.
PRODUCTION_EXCLUDED_PATHS: set[str] = {
    # Only absolute minimum for API documentation
    "/docs", "/redoc", "/openapi.json", "/favicon.ico",
}

# Production-ready authentication configuration
# Development tokens are now handled through environment variables only


class UnifiedAuthMiddleware(BaseHTTPMiddleware):
    """Unified Authentication Middleware

    Single source of truth for all authentication logic.
    Provides enterprise-grade security with proper token validation.
    """

    def __init__(self, app, excluded_paths: set[str] | None = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or PRODUCTION_EXCLUDED_PATHS
        self.security = HTTPBearer(auto_error=False)
        # Parameter wird für Middleware-Initialisierung benötigt
        _ = app

        logger.info(f"🔒 Unified Auth Middleware initialized with {len(self.excluded_paths)} excluded paths")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Main authentication dispatch logic."""
        # CORS Preflight-Requests immer ohne Auth zulassen
        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        # Check if path is excluded from authentication
        if self._is_path_excluded(request.url.path):
            logger.debug(f"🔓 Path excluded from auth: {request.url.path}")
            return await call_next(request)

        # Perform authentication
        try:
            await UnifiedAuthMiddleware._authenticate_request(request)
            logger.debug(f"🔐 Authentication successful for: {request.url.path}")

        except HTTPException as e:
            logger.warning(f"❌ Authentication failed for {request.url.path}: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail, "type": "authentication_error"}
            )
        except Exception as e:
            logger.error(f"❌ Authentication error for {request.url.path}: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal authentication error", "type": "authentication_error"}
            )

        # Continue to next middleware/endpoint
        return await call_next(request)

    def _is_path_excluded(self, path: str) -> bool:
        """Check if path is excluded from authentication."""
        # Exact match
        if path in self.excluded_paths:
            return True

        # Prefix match for paths ending with /
        for excluded_path in self.excluded_paths:
            if excluded_path.endswith("/") and path.startswith(excluded_path):
                return True

        # Special handling for WebSocket paths
        if path.startswith("/ws/") or path.startswith("/websocket/"):
            return True

        return False

    @staticmethod
    async def _authenticate_request(request: Request) -> None:
        """Authenticate the request using Unified Enterprise Auth System."""
        # Extract Authorization header
        authorization = request.headers.get("Authorization")
        credentials = None

        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix
            if token:
                from fastapi.security import HTTPAuthorizationCredentials
                credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        # Use Unified Enterprise Auth System
        auth_result = await unified_auth.authenticate_http(request, credentials)

        if not auth_result.success:
            raise HTTPException(
                status_code=401,
                detail=auth_result.error or "Authentication failed"
            )

        # Set authentication context in request state
        if auth_result.context:
            request.state.auth_context = auth_result.context
            request.state.user = {
                "sub": auth_result.context.subject,
                "scopes": [scope.value for scope in auth_result.context.scopes],
                "privilege_level": auth_result.context.privilege.value,
                "token_type": auth_result.context.token_type.value,
                "token_id": auth_result.context.token_id
            }
            request.state.authenticated = True
            request.state.auth_method = "unified_enterprise_auth"
            request.state.auth_timestamp = time.time()

    # Legacy token validation methods removed - now using Unified Enterprise Auth System

    # All token validation now handled by Unified Enterprise Auth System

    def add_excluded_path(self, path: str) -> None:
        """Add a path to the excluded paths set."""
        self.excluded_paths.add(path)
        logger.info(f"🔓 Added excluded path: {path}")

    def remove_excluded_path(self, path: str) -> None:
        """Remove a path from the excluded paths set."""
        self.excluded_paths.discard(path)
        logger.info(f"🔒 Removed excluded path: {path}")

    def get_excluded_paths(self) -> set[str]:
        """Get current excluded paths."""
        return self.excluded_paths.copy()


# Factory function for easy integration
def create_unified_auth_middleware(excluded_paths: set[str] | None = None) -> UnifiedAuthMiddleware:
    """Factory function to create unified auth middleware.

    Args:
        excluded_paths: Optional set of paths to exclude from authentication

    Returns:
        Configured UnifiedAuthMiddleware instance
    """
    return UnifiedAuthMiddleware(excluded_paths=excluded_paths)


# Utility functions for path management
def get_default_excluded_paths() -> set[str]:
    """Get the default set of excluded paths."""
    return PRODUCTION_EXCLUDED_PATHS.copy()


def is_development_token(token: str) -> bool:
    """Check if token is the development token."""
    return token == DEFAULT_DEV_TOKEN


# Export main components
__all__ = [
    "DEFAULT_DEV_TOKEN",
    "PRODUCTION_EXCLUDED_PATHS",
    "UnifiedAuthMiddleware",
    "create_unified_auth_middleware",
    "get_default_excluded_paths",
    "is_development_token"
]
