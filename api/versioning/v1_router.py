"""V1 Router – Weiterleitung auf bestehende V1‑Routen.

Dieser Router inkludiert die bestehenden V1 Endpunkte unter "/api/v1".
"""

from __future__ import annotations

from fastapi import APIRouter

from api.routes.agents_routes import router as agents_router
from api.routes.health_routes import router as health_router
from api.routes.logs_routes import router as logs_router
from api.routes.webhook_deliveries_routes import router as webhook_deliveries_router
from api.routes.webhook_routes import router as webhook_router

from .constants import URL_PREFIX_V1

v1_router = APIRouter(prefix=URL_PREFIX_V1)

v1_router.include_router(health_router)
v1_router.include_router(agents_router)
v1_router.include_router(webhook_router)
v1_router.include_router(webhook_deliveries_router)
v1_router.include_router(logs_router)

__all__ = ["v1_router"]
