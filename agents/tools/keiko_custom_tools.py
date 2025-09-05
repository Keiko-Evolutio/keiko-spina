"""Spezifische LangChain Tools.

Stellt benutzerdefinierte Tools für LangChain-Integration bereit.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .tools_constants import (
    DEFAULT_URLS,
    get_error_message,
    get_timeout,
)

logger = get_logger(__name__)

# LangChain ist optional - für Tests und generische Nutzung nicht erforderlich
LANGCHAIN_AVAILABLE = False

# Optionaler Import der Keiko Image Services auf Modulebene, damit Tests diese
# Symbole via Monkeypatch ersetzen können (kein lokaler Funktionsimport).
try:  # pragma: no cover - optional
    from services.clients.image_generation import (
        ImageGenerationRequest as _ImageGenerationRequest,
    )
    from services.clients.image_generation import (
        ImageGenerationService as _ImageGenerationService,
    )
except ImportError:  # pragma: no cover - defensiv
    _ImageGenerationService = None  # type: ignore[assignment]
    _ImageGenerationRequest = None  # type: ignore[assignment]
except Exception as e:  # pragma: no cover - defensiv
    logger.debug(f"Unerwarteter Fehler beim Import der Image-Generation-Services: {e}")
    _ImageGenerationService = None  # type: ignore[assignment]
    _ImageGenerationRequest = None  # type: ignore[assignment]


async def _image_generate_impl(params: dict[str, Any]) -> dict[str, Any]:
    """Implementierung ruft Keiko Image Service auf (vereinfacht)."""
    image_generation_service = globals().get("ImageGenerationService", _ImageGenerationService)
    image_generation_request = globals().get("ImageGenerationRequest", _ImageGenerationRequest)
    if image_generation_service is None or image_generation_request is None:
        return {"ok": False, "error": get_error_message("image_service_unavailable")}

    prompt = str(params.get("prompt", ""))
    size = str(params.get("size", "1024x1024"))
    quality = str(params.get("quality", "standard"))
    style = str(params.get("style", "vivid"))

    svc = image_generation_service()  # type: ignore[operator]
    req = image_generation_request(prompt=prompt, size=size, quality=quality, style=style)  # type: ignore[operator]
    result = await svc.generate(req)  # type: ignore[operator]
    return {
        "ok": True,
        "content_type": result.content_type,
        "image_bytes_len": len(result.image_bytes),
    }


async def _web_research_impl(params: dict[str, Any]) -> dict[str, Any]:
    """Einfaches Web-Research via HTTP GET (nur Demo; produktiv: Bing/Azure Search)."""
    import httpx

    query = str(params.get("query", ""))
    url = f"{DEFAULT_URLS['duckduckgo_search']}{query}"
    try:
        async with httpx.AsyncClient(timeout=get_timeout("web_research")) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return {"ok": True, "len": len(resp.text)}
    except Exception as exc:  # pragma: no cover - defensiv
        logger.debug(f"Web-Research fehlgeschlagen: {exc}")
        return {"ok": False}


def create_tools():
    """Erstellt Tool-Funktionen mit oder ohne langchain_core.tools."""

    # Für Tests und generische Nutzung geben wir einfache async-Funktionen
    # zurück. LC-Annotation ist optional und wird nicht benötigt.
    async def image_generate(**kwargs: Any) -> Any:
        return await _image_generate_impl(kwargs)

    async def web_research(**kwargs: Any) -> Any:
        return await _web_research_impl(kwargs)

    return image_generate, web_research


__all__ = ["create_tools"]
