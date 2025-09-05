"""Redis-basierte HTTP Response Caching Middleware.

Erzeugt Cache-Keys basierend auf Pfad, Query und optionalem Tenant-Kontext.
Unterstützt TTL-Regeln pro Endpoint-Muster und ETag/Cache-Control Header.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import settings
from kei_logging import get_logger
from monitoring.metrics_definitions import (
    record_etag_generation,
    record_http_cache_304,
    record_ttl_rule_application,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request

try:
    from storage.cache.redis_cache import NoOpCache, get_cache_client
except Exception:  # pragma: no cover
    get_cache_client = None  # type: ignore
    NoOpCache = object  # type: ignore


logger = get_logger(__name__)


def _hash_content(data: bytes) -> str:
    """Erzeugt ETag mittels SHA256."""
    h = hashlib.sha256()
    h.update(data)
    return 'W/"' + h.hexdigest() + '"'


def _parse_rules() -> dict[str, int]:
    """Parst Regel-CSV zu Pattern→TTL Mapping."""
    rules: dict[str, int] = {}
    raw = (settings.response_cache_rules or "").split(",")
    for item in raw:
        item = item.strip()
        if not item or ":" not in item:
            continue
        pat, ttl = item.split(":", 1)
        try:
            rules[pat] = int(ttl)
        except (ValueError, TypeError):
            continue
    return rules


def _match_ttl(path: str) -> int:
    """Ermittelt TTL für Pfad basierend auf Regeln und zeichnet Metrik."""
    rules = _parse_rules()
    for pat, ttl in rules.items():
        try:
            if re.search(pat.replace("*", ".*"), path):
                # TTL-Rule Treffer aufzeichnen
                with contextlib.suppress(Exception):
                    record_ttl_rule_application(endpoint=path, pattern=pat, ttl=str(max(0, ttl)))
                return max(0, ttl)
        except (re.error, ValueError, TypeError):
            continue
    default_ttl = max(0, settings.response_cache_default_ttl_seconds)
    with contextlib.suppress(Exception):
        record_ttl_rule_application(endpoint=path, pattern="default", ttl=str(default_ttl))
    return default_ttl


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """Middleware für HTTP Response Caching mit Redis Backend."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.response_cache_enabled:
            return await call_next(request)

        # Nur GET/HEAD cachen
        if request.method.upper() not in {"GET", "HEAD"}:
            return await call_next(request)

        # Cache-Key generieren (Pfad + Query + Tenant)
        tenant = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant-id") or "-"
        key = f"resp:{request.url.path}?{request.url.query}|t={tenant}"
        ttl = _match_ttl(request.url.path)

        client = None
        if get_cache_client is not None:
            try:
                client = await get_cache_client()
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Cache-Client-Verbindung fehlgeschlagen: {e}")
                client = None
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Abrufen des Cache-Clients: {e}")
                client = None

        # Versuch: aus Cache lesen
        if client and not isinstance(client, NoOpCache):
            try:
                from time import perf_counter
                t0 = perf_counter()
                val = await client.get(key)  # type: ignore[attr-defined]
                lookup_ms = (perf_counter() - t0) * 1000.0
                try:
                    request.state.cache_lookup_ms = lookup_ms  # für Tracing/Budget
                except AttributeError:
                    # Request.state ist nicht verfügbar - das ist ungewöhnlich aber nicht kritisch
                    pass
                except Exception as e:
                    logger.debug(f"Fehler beim Setzen der Cache-Lookup-Zeit: {e}")
                if val:
                    obj = json.loads(val)
                    etag = obj.get("etag", "")
                    # Conditional Request Handling
                    inm = request.headers.get("If-None-Match")
                    if inm and etag and inm == etag:
                        # 304 Not Modified aus Cache
                        try:
                            record_http_cache_304(endpoint=request.url.path, cache_type="etag")
                            request.state.cache_outcome = "hit_304"
                        except AttributeError:
                            # Request.state ist nicht verfügbar oder record_http_cache_304 fehlt
                            pass
                        except Exception as e:
                            logger.debug(f"Fehler beim Aufzeichnen des Cache-304-Hits: {e}")
                        return Response(status_code=304)
                    headers = obj.get("headers", {})
                    content = obj.get("content", "")
                    # Problematische Header entfernen, Starlette berechnet diese automatisch
                    filtered_headers = {k: v for k, v in headers.items() if k.lower() not in {"content-length", "transfer-encoding"}}
                    return Response(
                        content=content.encode("utf-8"),
                        media_type=headers.get("content-type", "application/json"),
                        headers={**filtered_headers, "ETag": etag},
                    )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Upstream ausführen
        response = await call_next(request)
        # Body sicher extrahieren (keine leeren Bodies erzeugen)
        try:
            # Bevorzugt direktes Body-Attribut, da Starlette dies stabil befüllt
            body = getattr(response, "body", None)
            if body is None:
                # Fallback: Iterator konsumieren, falls vorhanden
                if hasattr(response, "body_iterator") and response.body_iterator is not None:
                    chunks = []
                    async for chunk in response.body_iterator:
                        chunks.append(chunk)
                    body = b"".join(chunks)
                else:
                    body = b""
        except (AttributeError, TypeError) as e:
            logger.debug(f"Fehler beim Extrahieren des Response-Body: {e}")
            body = b""
        except Exception as e:
            logger.warning(f"Unerwarteter Fehler beim Response-Body-Handling: {e}")
            body = b""

        # ETag erzeugen und Header setzen
        etag = ""
        try:
            etag = _hash_content(body)
            response.headers["ETag"] = etag
            with contextlib.suppress(Exception):
                record_etag_generation(endpoint=request.url.path)
            # Cache-Control Header abhängig vom TTL
            if ttl > 0:
                response.headers["Cache-Control"] = f"public, max-age={ttl}"
        except (ValueError, TypeError, AttributeError):
            pass

        # Im Cache speichern
        if client and not isinstance(client, NoOpCache) and ttl > 0:
            try:
                payload = {
                    "etag": etag,
                    "headers": {k.lower(): v for k, v in response.headers.items()},
                    "content": body.decode(encoding="utf-8", errors="ignore"),
                }
                from time import perf_counter
                t0 = perf_counter()
                await client.setex(key, ttl, json.dumps(payload))  # type: ignore[attr-defined]
                write_ms = (perf_counter() - t0) * 1000.0
                try:
                    request.state.cache_write_ms = write_ms  # für Tracing/Budget
                except AttributeError:
                    # Request.state ist nicht verfügbar - das ist ungewöhnlich aber nicht kritisch
                    pass
                except Exception as e:
                    logger.debug(f"Fehler beim Setzen der Cache-Write-Zeit: {e}")
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Cache-Schreibfehler - Verbindungsproblem: {e}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Cache-Schreibfehler - Serialisierungsproblem: {e}")
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Cache-Schreiben: {e}")

        # Response neu konstruieren mit Body
        try:
            with contextlib.suppress(Exception):
                request.state.cache_outcome = request.state.cache_outcome if hasattr(request.state, "cache_outcome") else "miss_or_uncached"
        except AttributeError:
            # Request.state ist nicht verfügbar - das ist ungewöhnlich aber nicht kritisch
            pass
        except Exception as e:
            logger.debug(f"Fehler beim Setzen des Cache-Outcome: {e}")

        # Headers filtern: Content-Length und Transfer-Encoding ausschließen
        # Diese werden von Starlette automatisch neu berechnet
        filtered_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding"}
        }

        return Response(
            content=body,
            status_code=response.status_code,
            headers=filtered_headers,
            media_type=response.media_type
        )


__all__ = ["ResponseCacheMiddleware"]
