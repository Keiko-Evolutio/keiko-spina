"""OpenAPI Generator für Keiko REST‑APIs.

Erzeugt eine vollständige OpenAPI 3.1 Spezifikation aus der laufenden FastAPI‑App.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .constants import SecuritySchemes, SpecConstants

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)


def generate_openapi_dict(app: FastAPI) -> dict[str, Any]:
    """Generiert OpenAPI Dict über FastAPI `openapi()`.

    Args:
        app: Laufende FastAPI Anwendung

    Returns:
        OpenAPI Dokument als Dictionary
    """
    try:
        spec = app.openapi()
        # Ergänze globale Security Hinweise
        spec.setdefault("components", {}).setdefault("securitySchemes", {}).update(
            {
                "bearerAuth": SecuritySchemes.BEARER_AUTH,
                "mtls": SecuritySchemes.MTLS,
            }
        )
        return spec
    except Exception as exc:  # pragma: no cover
        logger.exception(f"OpenAPI Generierung fehlgeschlagen: {exc}")
        return {"openapi": SpecConstants.OPENAPI_VERSION, "info": {"title": "Keiko", "version": "unknown"}, "paths": {}}


__all__ = ["generate_openapi_dict"]

