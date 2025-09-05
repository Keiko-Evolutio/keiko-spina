"""API Middleware für Keiko."""

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kei_logging import get_logger

logger = get_logger(__name__)

# Middleware-Imports
from .budget_middleware import BudgetMiddleware
from .deadline_middleware import DeadlineMiddleware
from .error_middleware import ErrorHandlingMiddleware, SecurityHeadersMiddleware
from .request_detective_middleware import RequestDetectiveMiddleware
from .scope_middleware import ScopeMiddleware
from .structured_error_middleware import (
    LegacyErrorCompatibilityMiddleware,
    StructuredErrorHandlingMiddleware,
    create_legacy_compatibility_middleware,
    create_structured_error_middleware,
)
from .tracing_middleware import TracingMiddleware
from .unified_auth_middleware import UnifiedAuthMiddleware
from .websocket_middleware import WebSocketMiddleware


class MiddlewareConfig:
    """Vereinfachte Middleware-Konfiguration."""

    def __init__(
        self,
        cors_origins: list = None,
        cors_enabled: bool = True,
        auth_enabled: bool = True,  # Auth standardmäßig aktiviert für Production Security
        environment: str = "development"
    ):
        # Standard CORS Origins mit WebSocket-Support
        default_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "ws://localhost:8000",
            "wss://localhost:8000"
        ]
        self.cors_origins = cors_origins or default_origins
        self.cors_enabled = cors_enabled
        self.auth_enabled = auth_enabled
        self.environment = environment


def setup_middleware(app: FastAPI, config: MiddlewareConfig | None = None) -> FastAPI:
    """Vereinfachte Middleware-Stack Konfiguration."""
    if config is None:
        config = MiddlewareConfig()

    active = []

    # CORS zuerst hinzufügen, damit es Preflight-Requests abfängt
    if config.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=[
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "Accept",
                "Accept-Language",
                "Content-Language",
                "X-Tenant-Id",
                "X-User-Id",
                "x-tenant",
                "x-user-id",
                "X-Trace-Id"
            ],
            max_age=3600
        )
        active.append("CORS")

    # Error Handling + Security
    app.add_middleware(SecurityHeadersMiddleware)

    # Strukturiertes Error Handling (neue Implementation)
    app.add_middleware(
        StructuredErrorHandlingMiddleware,
        include_debug_details=config.environment == "development"
    )

    # Legacy Error Compatibility (für Backward Compatibility)
    app.add_middleware(
        LegacyErrorCompatibilityMiddleware,
        enable_legacy_format=True
    )

    # Fallback: Originales Error Handling (falls strukturiertes fehlschlägt)
    app.add_middleware(ErrorHandlingMiddleware, include_details=config.environment == "development")

    active.append("Structured Error Handling")
    active.append("Legacy Error Compatibility")

    # Request Detective entfernt - Quelle identifiziert: Prometheus Alertmanager

    # WebSocket Support - DISABLED (Router-based auth)
    # app.add_middleware(WebSocketMiddleware)
    # active.append("WebSocket")

    # Unified Authentication - PRODUCTION SECURITY
    if config.auth_enabled:
        app.add_middleware(UnifiedAuthMiddleware)
        active.append("Unified Auth")

        # Scope‑Middleware nach erfolgreicher Auth einreihen
        app.add_middleware(ScopeMiddleware)
        active.append("Scope")

    # Deadline + Budget + Tracing
    app.add_middleware(DeadlineMiddleware)
    active.append("Deadline")
    app.add_middleware(BudgetMiddleware)
    active.append("Budget")
    app.add_middleware(TracingMiddleware)
    active.append("Tracing")



    logger.info(f"Middleware aktiviert: {', '.join(active)}")
    return app


def health_check() -> dict[str, Any]:
    """Health Check für Middleware."""
    return {
        "status": "healthy",
        "available_middleware": {
            "tracing": True,
            "auth": True,
            "error_handling": True,
            "websocket": True
        }
    }


__all__ = [
    "MiddlewareConfig",
    "health_check",
    "setup_middleware",
]
