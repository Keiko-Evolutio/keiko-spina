"""Idempotency Middleware für REST-Operationen.

Diese Middleware implementiert eine standardisierte Idempotenz-Verarbeitung
für nicht-idempotente Methoden (POST/PUT/PATCH/DELETE) auf Basis eines
`Idempotency-Key` Headers. Ergebnisse werden für eine konfigurierte TTL
zwischengespeichert. Redis wird bevorzugt, In-Memory dient als Fallback.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from kei_logging import get_logger
from storage.cache.redis_cache import get_cache_client

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request

logger = get_logger(__name__)


class _MemoryStore:
    """Einfacher In-Memory Store als Fallback für Idempotenz.

    Achtung: Dieser Store ist pro-Prozess und nicht verteilt. Nur als Fallback
    verwenden, wenn Redis nicht verfügbar ist.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, dict[str, str], float, int, int]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> tuple[bytes, dict[str, str], int] | None:
        """Liest gespeicherten Eintrag, sofern nicht abgelaufen."""
        async with self._lock:
            if key not in self._data:
                return None
            body, headers, created, status_code, ttl = self._data[key]
            if time.time() - created > ttl:
                self._data.pop(key, None)
                return None
            return body, headers, status_code

    async def set(self, key: str, body: bytes, headers: dict[str, str], ttl: int, status_code: int) -> None:
        """Speichert Response-Daten mit TTL."""
        async with self._lock:
            self._data[key] = (body, headers, time.time(), status_code, ttl)


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Middleware zur Behandlung von Idempotency-Keys.

    Erwartet Header `Idempotency-Key` bei schreibenden Methoden. Der Key wird
    in Kombination aus Pfad und Methode genutzt, um Antworten deterministisch
    zwischenzuspeichern. Bei Cache-Hit wird die ursprüngliche Antwort inkl.
    Statuscode und relevanter Headers zurückgegeben.
    """

    def __init__(self, app, ttl_seconds: int = 600) -> None:
        """Initialisiert Middleware.

        Args:
            app: FastAPI/Starlette App
            ttl_seconds: Cache-TTL in Sekunden
        """
        super().__init__(app)
        self.ttl_seconds = ttl_seconds
        self._memory = _MemoryStore()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Haupt-Dispatch mit Idempotenzbehandlung."""
        method = request.method.upper()
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return await call_next(request)

        key = request.headers.get("Idempotency-Key")
        if not key:
            # Ohne Key normal weiterleiten
            return await call_next(request)

        # Schlüssel deterministisch erweitern (Pfad + Methode)
        cache_key = f"idemp:{method}:{request.url.path}:{key}"

        # Zuerst Redis, dann Memory prüfen
        hit_response = await self._try_get_cached_response(cache_key)
        if hit_response is not None:
            body, headers, status_code = hit_response
            resp = Response(content=body, status_code=status_code, media_type=headers.get("content-type", "application/json"))
            for h, v in headers.items():
                # Sicherheitsfilter für hop-by-hop Header
                if h.lower() not in {"content-length", "transfer-encoding"}:
                    resp.headers[h] = v
            resp.headers["Idempotency-Key"] = key
            resp.headers["Idempotency-Cache"] = "hit"
            return resp

        # Request ausführen und Antwort abfangen
        response = await call_next(request)

        try:
            # Body extrahieren (Streaming-Responses werden ausgelassen)
            if hasattr(response, "body_iterator"):
                # Response in Speicher puffern
                body_bytes = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]
                # Iterator resetten mit gepuffertem Inhalt
                response.body_iterator = iter([body_bytes])  # type: ignore[assignment]
            else:
                body_bytes = await response.body()

            # Relevante Headers kopieren
            headers = {k.lower(): v for k, v in response.headers.items()}
            headers.setdefault("content-type", "application/json")

            # In Cache speichern
            await self._try_set_cached_response(cache_key, body_bytes, headers, response.status_code)

            # Antwort mit Miss-Header markieren
            response.headers["Idempotency-Key"] = key
            response.headers["Idempotency-Cache"] = "miss"
        except Exception as e:
            logger.warning(f"Idempotency Cache konnte nicht gesetzt werden: {e}")

        return response

    async def _try_get_cached_response(self, cache_key: str) -> tuple[bytes, dict[str, str], int] | None:
        """Versucht, einen Treffer aus Redis oder Memory zu lesen."""
        try:
            client = await get_cache_client()
            if client and hasattr(client, "get"):
                raw = await client.get(cache_key)
                if raw:
                    payload = json.loads(raw)
                    body = bytes.fromhex(payload["body_hex"]) if "body_hex" in payload else payload["body"].encode()
                    headers = payload.get("headers", {})
                    status = int(payload.get("status", 200))
                    return body, {str(k): str(v) for k, v in headers.items()}, status
        except Exception as e:  # pragma: no cover - Redis optional
            logger.debug(f"Redis Idempotency get Fehler: {e}")

        # Fallback Memory
        try:
            return await self._memory.get(cache_key)
        except Exception:
            return None

    async def _try_set_cached_response(
        self,
        cache_key: str,
        body: bytes,
        headers: dict[str, str],
        status_code: int,
    ) -> None:
        """Speichert Eintrag in Redis und Memory (Best Effort)."""
        payload = json.dumps({
            "body_hex": body.hex(),
            "headers": headers,
            "status": status_code,
        })

        try:
            client = await get_cache_client()
            if client and hasattr(client, "setex"):
                await client.setex(cache_key, self.ttl_seconds, payload)
        except Exception as e:  # pragma: no cover - Redis optional
            logger.debug(f"Redis Idempotency set Fehler: {e}")

        # Immer auch im Memory-Fallback speichern
        await self._memory.set(cache_key, body, headers, self.ttl_seconds, status_code)


__all__ = ["IdempotencyMiddleware"]
