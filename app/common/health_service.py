"""Konsolidierter Health Service für das App-Modul.

Zentrale Health Check Logik um Code-Duplikation zwischen application.py
und health_endpoints.py zu eliminieren.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse

from services import is_services_healthy_async

from .constants import (
    APP_VERSION,
    HEALTH_STATUS_DEGRADED,
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_WARNING,
)
from .logger_utils import get_module_logger, safe_log_exception

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_module_logger(__name__)


class HealthStatus:
    """Repräsentiert den Health-Status der Anwendung."""

    def __init__(
        self,
        status: str = HEALTH_STATUS_HEALTHY,
        version: str = APP_VERSION,
        services_healthy: bool = True,
        critical_services_down: int = 0,
        additional_info: dict[str, Any] | None = None
    ) -> None:
        self.status = status
        self.version = version
        self.services_healthy = services_healthy
        self.critical_services_down = critical_services_down
        self.additional_info = additional_info or {}

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert den Health-Status zu einem Dictionary."""
        result = {
            "status": self.status,
            "version": self.version,
        }

        if self.additional_info:
            result.update(self.additional_info)

        return result

    def to_json_response(self) -> JSONResponse:
        """Erstellt eine JSONResponse basierend auf dem Health-Status."""
        status_code = 200 if self.status == HEALTH_STATUS_HEALTHY else 503
        return JSONResponse(content=self.to_dict(), status_code=status_code)

    @property
    def is_healthy(self) -> bool:
        """Prüft ob der Service als gesund gilt."""
        return self.status == HEALTH_STATUS_HEALTHY


async def check_services_health() -> bool:
    """Prüft die Gesundheit aller kritischen Services.

    Returns:
        True wenn alle Services gesund sind, False sonst
    """
    try:
        return await is_services_healthy_async()
    except Exception as exc:
        safe_log_exception(
            logger,
            exc,
            "Fehler bei Service Health Check",
            service="health_check"
        )
        return False


def calculate_health_status(services_healthy: bool, critical_services_down: int = 0) -> str:
    """Berechnet den Gesamt-Health-Status basierend auf Service-Zuständen.

    Args:
        services_healthy: Ob alle Services gesund sind
        critical_services_down: Anzahl kritischer Services die down sind

    Returns:
        Health-Status String (healthy, warning, degraded)
    """
    if not services_healthy:
        critical_services_down = max(1, critical_services_down)

    if critical_services_down > 1:
        return HEALTH_STATUS_DEGRADED
    if critical_services_down > 0:
        return HEALTH_STATUS_WARNING
    return HEALTH_STATUS_HEALTHY


async def get_application_health(
    include_additional_info: bool = False
) -> HealthStatus:
    """Ermittelt den aktuellen Health-Status der Anwendung.

    Args:
        include_additional_info: Ob zusätzliche Informationen eingeschlossen werden sollen

    Returns:
        HealthStatus-Objekt mit aktuellem Status
    """
    services_healthy = await check_services_health()
    critical_services_down = 0 if services_healthy else 1

    status = calculate_health_status(services_healthy, critical_services_down)

    additional_info = {}
    if include_additional_info:
        additional_info.update({
            "services_healthy": services_healthy,
            "critical_services_down": critical_services_down,
        })

    return HealthStatus(
        status=status,
        version=APP_VERSION,
        services_healthy=services_healthy,
        critical_services_down=critical_services_down,
        additional_info=additional_info
    )


async def create_health_response(
    include_additional_info: bool = False
) -> JSONResponse:
    """Erstellt eine standardisierte Health Check Response.

    Args:
        include_additional_info: Ob zusätzliche Debug-Informationen eingeschlossen werden sollen

    Returns:
        JSONResponse mit Health-Status
    """
    try:
        health_status = await get_application_health(include_additional_info)
        return health_status.to_json_response()
    except Exception as exc:
        safe_log_exception(
            logger,
            exc,
            "Kritischer Fehler bei Health Check",
            include_additional_info=include_additional_info
        )

        # Fallback: Degraded Status zurückgeben
        fallback_status = HealthStatus(
            status=HEALTH_STATUS_DEGRADED,
            services_healthy=False,
            critical_services_down=1,
            additional_info={"error": "Health check failed"}
        )
        return fallback_status.to_json_response()


def is_service_healthy(service_name: str, check_function: Callable[[], bool]) -> bool:
    """Prüft die Gesundheit eines spezifischen Services.

    Args:
        service_name: Name des Services
        check_function: Funktion die den Service-Status prüft

    Returns:
        True wenn Service gesund ist, False sonst
    """
    try:
        return bool(check_function())
    except Exception as exc:
        safe_log_exception(
            logger,
            exc,
            f"Health Check für {service_name} fehlgeschlagen",
            service=service_name
        )
        return False


async def is_async_service_healthy(service_name: str, check_function: Callable[[], Any]) -> bool:
    """Prüft die Gesundheit eines spezifischen Services (async).

    Args:
        service_name: Name des Services
        check_function: Async Funktion die den Service-Status prüft

    Returns:
        True wenn Service gesund ist, False sonst
    """
    try:
        result = await check_function()
        return bool(result)
    except Exception as exc:
        safe_log_exception(
            logger,
            exc,
            f"Async Health Check für {service_name} fehlgeschlagen",
            service=service_name
        )
        return False


__all__ = [
    "HealthStatus",
    "calculate_health_status",
    "check_services_health",
    "create_health_response",
    "get_application_health",
    "is_async_service_healthy",
    "is_service_healthy",
]
