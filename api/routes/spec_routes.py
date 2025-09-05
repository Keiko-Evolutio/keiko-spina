"""API‑Routen zur Spezifikationsgenerierung (OpenAPI/AsyncAPI)."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from kei_logging import get_logger
from specs.spec_cli import generate_all_specs

from .base import create_router

logger = get_logger(__name__)

router = create_router("/api/v1/specs", ["specs"])


@router.post("/generate")
async def generate_specs(request: Request) -> dict[str, Any]:
    """Generiert AsyncAPI/OpenAPI Artefakte und liefert Pfade zurück."""
    app = request.app
    summary = generate_all_specs(app)  # schreibt Dateien in backend/specs/out
    return {"status": "ok", "artifacts": summary}


@router.get("/asyncapi")
async def get_asyncapi() -> dict[str, Any]:
    """Gibt generierte AsyncAPI Spec zurück (JSON)."""
    from specs.asyncapi_generator import build_asyncapi_dict

    return build_asyncapi_dict()


@router.get("/openapi")
async def get_openapi(request: Request) -> dict[str, Any]:
    """Gibt generierte OpenAPI Spec zurück (JSON)."""
    from specs.openapi_generator import generate_openapi_dict

    return generate_openapi_dict(request.app)


__all__ = ["router"]
