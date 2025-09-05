"""API Package - Keiko Personal Assistant Backend.

Konsolidiert REST-APIs und gRPC-APIs in einem einheitlichen Package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from api.routes.n8n_integration_routes import router as n8n_router

# Router- und Middleware-Imports ohne Fallbacks
from api.v1 import configurations_router
from kei_logging import get_logger

# Konsolidierte Utilities
from .common.api_constants import APIPaths
from .common.router_factory import RouterFactory

# Logger initialisieren
logger = get_logger(__name__)

# gRPC-Komponenten (migriert aus kei_rpc/)
try:
    from .grpc import (
        AuthInterceptor,
        BaseInterceptor,
        GRPCServerFactory,
        GRPCServerManager,
        KEIRPCService,
        RateLimitInterceptor,
        serve_grpc,
        shutdown_grpc,
    )
    _GRPC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"gRPC-Komponenten nicht verfügbar: {e}")
    _GRPC_AVAILABLE = False

# Webhook Router wird separat importiert (kann fehlen in minimalen Umgebungen)
try:
    from api.routes.webhook_routes import router as webhook_router
except Exception:  # pragma: no cover - optional module
    webhook_router = None  # type: ignore
from api.routes.rpc_routes import router as rpc_router
from services.streaming import router as websocket_router

from .middleware import MiddlewareConfig, setup_middleware

if TYPE_CHECKING:
    from fastapi import APIRouter

logger = get_logger(__name__)

__all__ = [
    # REST API Components
    "MiddlewareConfig",
    "configurations_router",
    "create_api_router",
    "setup_middleware",
    "websocket_router",
]

# Conditional gRPC exports
if _GRPC_AVAILABLE:
    __all__.extend([
        # gRPC Components
        "AuthInterceptor",
        "BaseInterceptor",
        "GRPCServerFactory",
        "GRPCServerManager",
        "KEIRPCService",
        "RateLimitInterceptor",
        "serve_grpc",
        "shutdown_grpc",
    ])


def create_api_router(prefix: str = APIPaths.API_V1_PREFIX) -> APIRouter:
    """Erstellt kombinierten API-Router mit konsolidierten Utilities.

    Args:
        prefix: URL-Präfix für Router

    Returns:
        Konfigurierter API-Router
    """
    # Verwende RouterFactory für einheitliche Router-Erstellung
    main_router = RouterFactory.create_router(
        prefix=prefix,
        tags=["api"],
        include_standard_responses=True
    )

    # Router registrieren mit verbesserter Null-Behandlung
    routers = [
        configurations_router,
        websocket_router,
        n8n_router,
        rpc_router,
        webhook_router
    ]

    registered_count = 0
    for router in routers:
        if router is not None:
            main_router.include_router(router)
            registered_count += 1

    logger.info(f"API Router erstellt - {registered_count} Router registriert")
    return main_router
