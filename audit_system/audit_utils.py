# backend/audit_system/audit_utils.py
"""Utility-Funktionen für das Audit-System.

Konsolidiert wiederkehrende Patterns und Funktionalitäten
zu wiederverwendbaren Komponenten.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .audit_constants import AuditConstants

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request

logger = get_logger(__name__)


@dataclass
class RequestMetadata:
    """Metadaten für HTTP-Requests."""

    correlation_id: str
    trace_id: str | None
    client_ip: str
    user_agent: str
    actor: str
    start_time: float
    method: str
    path: str


def generate_correlation_id() -> str:
    """Generiert eine eindeutige Korrelations-ID.

    Returns:
        Eindeutige UUID als String
    """
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """Gibt aktuellen Zeitstempel im ISO-Format zurück.

    Returns:
        ISO-formatierter Zeitstempel mit UTC-Timezone
    """
    return datetime.now(UTC).isoformat()


def extract_client_ip(request: Request) -> str:
    """Extrahiert Client-IP aus Request.

    Args:
        request: FastAPI-Request-Objekt

    Returns:
        Client-IP-Adresse als String
    """
    # Prüfe X-Forwarded-For Header (Proxy/Load Balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Nimm erste IP (Client)
        return forwarded_for.split(",")[0].strip()

    # Prüfe X-Real-IP Header (Nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback auf direkte Client-IP
    if hasattr(request, "client") and request.client:
        return request.client.host

    return AuditConstants.DEFAULT_CLIENT_IP


def extract_user_agent(request: Request) -> str:
    """Extrahiert User-Agent aus Request.

    Args:
        request: FastAPI-Request-Objekt

    Returns:
        User-Agent-String
    """
    return request.headers.get("User-Agent", "unknown")


def extract_actor_from_request(request: Request) -> str:
    """Extrahiert Akteur-Information aus Request.

    Args:
        request: FastAPI-Request-Objekt

    Returns:
        Akteur-Identifier
    """
    # Prüfe Authorization Header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # Vereinfachte Extraktion - in Produktion würde hier JWT-Parsing stattfinden
        if auth_header.startswith("Bearer "):
            return "authenticated_user"

    # Prüfe API-Key Header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return "api_client"

    return AuditConstants.DEFAULT_ACTOR


def create_request_metadata(request: Request) -> RequestMetadata:
    """Erstellt vollständige Request-Metadaten.

    Args:
        request: FastAPI-Request-Objekt

    Returns:
        RequestMetadata-Objekt mit allen extrahierten Informationen
    """
    return RequestMetadata(
        correlation_id=generate_correlation_id(),
        trace_id=request.headers.get("X-Trace-ID"),
        client_ip=extract_client_ip(request),
        user_agent=extract_user_agent(request),
        actor=extract_actor_from_request(request),
        start_time=time.time(),
        method=request.method,
        path=request.url.path
    )


def calculate_duration_ms(start_time: float) -> float:
    """Berechnet Dauer in Millisekunden.

    Args:
        start_time: Start-Zeitpunkt (time.time())

    Returns:
        Dauer in Millisekunden
    """
    return (time.time() - start_time) * 1000


def is_path_excluded(path: str, excluded_paths: list[str]) -> bool:
    """Prüft ob Pfad von Audit ausgeschlossen ist.

    Args:
        path: URL-Pfad
        excluded_paths: Liste ausgeschlossener Pfade

    Returns:
        True wenn Pfad ausgeschlossen ist
    """
    return path in excluded_paths


def sanitize_for_logging(data: Any, max_length: int = 1000) -> str:
    """Sanitisiert Daten für sicheres Logging.

    Args:
        data: Zu sanitisierende Daten
        max_length: Maximale Länge des Outputs

    Returns:
        Sanitisierter String
    """
    if data is None:
        return "null"

    # Konvertiere zu String
    str_data = str(data)

    # Kürze wenn zu lang
    if len(str_data) > max_length:
        str_data = str_data[:max_length] + "..."

    # Entferne potentiell gefährliche Zeichen
    return str_data.replace("\n", "\\n").replace("\r", "\\r")



async def safe_async_call(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any
) -> Any | None:
    """Führt asynchronen Aufruf sicher aus.

    Args:
        func: Asynchrone Funktion
        *args: Positionsargumente
        **kwargs: Keyword-Argumente

    Returns:
        Ergebnis der Funktion oder None bei Fehler
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.exception("Fehler bei asynchronem Aufruf %s: %s", func.__name__, e)
        return None


def create_error_context(
    error: Exception,
    operation: str,
    additional_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt Fehler-Kontext für Audit-Events.

    Args:
        error: Exception-Objekt
        operation: Name der Operation
        additional_context: Zusätzlicher Kontext

    Returns:
        Fehler-Kontext-Dictionary
    """
    context = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "operation": operation,
        "timestamp": get_current_timestamp()
    }

    if additional_context:
        context.update(additional_context)

    return context


class AsyncTaskManager:
    """Manager für asynchrone Tasks mit sicherer Cleanup."""

    def __init__(self):
        """Initialisiert Task-Manager."""
        self._tasks: list[asyncio.Task] = []
        self._is_running = False

    def add_task(self, coro: Awaitable[Any]) -> asyncio.Task:
        """Fügt Task hinzu.

        Args:
            coro: Coroutine oder Awaitable

        Returns:
            Erstellter Task
        """
        # Konvertiere Awaitable zu Coroutine falls nötig
        if hasattr(coro, "__await__"):
            task = asyncio.create_task(coro)
        else:
            # Fallback für nicht-Coroutine Awaitables
            task = asyncio.create_task(asyncio.ensure_future(coro))
        self._tasks.append(task)
        return task

    async def start(self) -> None:
        """Startet Task-Manager."""
        self._is_running = True
        logger.info("AsyncTaskManager gestartet")

    async def stop(self) -> None:
        """Stoppt alle Tasks sicher."""
        self._is_running = False

        # Cancelle alle Tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Warte auf Completion
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("AsyncTaskManager gestoppt")

    @property
    def is_running(self) -> bool:
        """Gibt Running-Status zurück."""
        return self._is_running


__all__ = [
    "AsyncTaskManager",
    "RequestMetadata",
    "calculate_duration_ms",
    "create_error_context",
    "create_request_metadata",
    "extract_actor_from_request",
    "extract_client_ip",
    "extract_user_agent",
    "generate_correlation_id",
    "get_current_timestamp",
    "is_path_excluded",
    "safe_async_call",
    "sanitize_for_logging"
]
