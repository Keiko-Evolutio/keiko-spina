# backend/api/routes/image_health_routes.py
"""Health Endpoint für Bildgenerierungspipeline."""
from __future__ import annotations

import asyncio
import time
from typing import Any

# Optional import - Registry-System ist nicht immer verfügbar
try:
    from agents.registry.dynamic_registry import dynamic_registry
except ImportError:
    # Fallback wenn Registry nicht verfügbar ist
    dynamic_registry = None
from urllib.parse import urlparse

from config.settings import settings
from kei_logging import get_logger
from services.clients.content_safety import create_content_safety_client
from services.clients.image_generation import create_image_generation_service
from storage.azure_blob_storage.azure_blob_storage import generate_sas_url, upload_image_bytes

from .base import create_router

logger = get_logger(__name__)

router = create_router("/api/health", ["health"])  # konsolidierte Health-Routen


async def _check_openai() -> dict[str, Any]:
    start = time.perf_counter()
    svc = create_image_generation_service()
    try:
        # Minimalprüfung: Client verfügbar (kein Request auslösen)
        ok = svc.is_available
        return {"status": "ok" if ok else "degraded",
                "time_ms": int((time.perf_counter() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e),
                "time_ms": int((time.perf_counter() - start) * 1000)}


async def _check_storage() -> dict[str, Any]:
    start = time.perf_counter()
    try:
        data = b"ok"
        name = "health/ok.txt"
        container = settings.keiko_storage_container_for_img or "keiko-images"
        _ = await upload_image_bytes(container, name, data, content_type="text/plain")
        _ = await generate_sas_url(container, name, expiry_minutes=5)
        return {"status": "ok", "time_ms": int((time.perf_counter() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e),
                "time_ms": int((time.perf_counter() - start) * 1000)}


async def _check_safety() -> dict[str, Any]:
    start = time.perf_counter()
    client = create_content_safety_client()
    try:
        res = await client.analyze_text("Hello, this is a safe text.")
        return {"status": "ok" if res.is_safe else "degraded",
                "time_ms": int((time.perf_counter() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e),
                "time_ms": int((time.perf_counter() - start) * 1000)}


async def _check_agent() -> dict[str, Any]:
    start = time.perf_counter()
    try:
        # Prüfe ob Registry verfügbar ist
        if dynamic_registry is None:
            return {"status": "error", "error": "Dynamic registry not available",
                    "time_ms": int((time.perf_counter() - start) * 1000)}

        if dynamic_registry and hasattr(dynamic_registry, "is_initialized") and not dynamic_registry.is_initialized():
            # Registry auto-initialized - no action needed
            pass

        from config.settings import settings as _settings
        target_id = getattr(_settings, "agent_image_generator_id", "") or "agent_image_generator"

        if dynamic_registry and hasattr(dynamic_registry, "get_agent_by_id"):
            agent = await dynamic_registry.get_agent_by_id(target_id)
        else:
            agent = None
        status = "ok" if agent is not None else "error"
        return {"status": status, "time_ms": int((time.perf_counter() - start) * 1000)}
    except Exception as e:
        return {"status": "error", "error": str(e),
                "time_ms": int((time.perf_counter() - start) * 1000)}


@router.get("/image-generation")
async def image_generation_health():
    """Health Check für Bildpipeline."""
    openai, storage, safety, agent = await asyncio.gather(
        _check_openai(), _check_storage(), _check_safety(), _check_agent()
    )
    overall = "healthy" if all(
        s.get("status") == "ok" for s in [openai, storage, safety, agent]) else "degraded"
    return {
        "service": "image_generation",
        "status": overall,
        "components": {
            "openai": openai,
            "storage": storage,
            "content_safety": safety,
            "agent": agent,
        }
    }


@router.get("/image-proxy")
async def image_proxy(url: str):
    """CORS-Proxy für Azure Blob Storage Bilder."""
    import aiohttp
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    # Sicherheitscheck: Nur Azure Storage Account aus Konfiguration erlauben
    allowed = urlparse(settings.storage_account_url)
    target = urlparse(url)
    if not (allowed.scheme in {"https", "http"} and target.scheme in {"https", "http"}
            and target.netloc == allowed.netloc):
        raise HTTPException(status_code=400, detail="Ungültige URL")

    try:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Bild nicht gefunden")

            content_type = resp.headers.get("content-type", "image/png")

            async def generate():
                async for chunk in resp.content.iter_chunked(8192):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type=content_type,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET",
                    "Access-Control-Allow-Headers": "*",
                }
            )
    except Exception as e:
        logger.exception(f"Image proxy error: {e}")
        raise HTTPException(status_code=500, detail="Fehler beim Laden des Bildes")
