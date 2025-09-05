"""Utility-Funktionen für das API-Versionierungsmodul.

Wiederverwendbare Funktionen für Health-Checks, Error-Handling und
Response-Building zur Reduzierung von Code-Duplikation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kei_logging import get_logger, structured_msg

from .constants import (
    COMPONENT_AGENT_SYSTEM,
    COMPONENT_SYSTEM,
    EXECUTION_ID_PREFIX,
    EXECUTION_STATUS_ERROR,
    HEALTH_STATUS_HEALTHY,
    HEALTH_STATUS_UNHEALTHY,
)

logger = get_logger(__name__)


def create_execution_id() -> str:
    """Erstellt eine eindeutige Execution-ID.

    Returns:
        Eindeutige Execution-ID mit Prefix
    """
    return f"{EXECUTION_ID_PREFIX}{uuid4().hex[:8]}"


def calculate_duration_ms(start_time: datetime) -> int:
    """Berechnet die Dauer in Millisekunden seit einem Startzeitpunkt.

    Args:
        start_time: Startzeitpunkt der Messung

    Returns:
        Dauer in Millisekunden als Integer
    """
    return int((datetime.now(UTC) - start_time).total_seconds() * 1000)


def safe_get_sla_metrics() -> dict[str, Any] | None:
    """Holt SLA-Metriken mit Error-Handling.

    Returns:
        SLA-Metriken oder None bei Fehlern
    """
    try:
        from api.routes.health_routes import _HEALTH_MANAGER  # type: ignore
        summary = _HEALTH_MANAGER.check_health()
        if not summary:
            return None

        sla = summary.get("sla", {}) or summary.get("summary", {}).get("sla", {})
        return {
            "availability_percentage": round(float(sla.get("availability", 0.0)) * 100.0, 3),
            "p95_latency_ms": float(sla.get("p95_latency_ms", 0.0)),
            "error_rate_percentage": float(sla.get("error_rate_pct", 0.0)),
            "uptime_seconds": int(summary.get("uptime_seconds", 0)),
        }
    except Exception as exc:
        logger.debug(structured_msg("SLA metrics unavailable", error=str(exc)))
        return None


def safe_get_trend_data() -> dict[str, Any] | None:
    """Holt Trend-Daten mit Error-Handling.

    Returns:
        Trend-Daten oder None bei Fehlern
    """
    try:
        from api.routes.health_routes import _HEALTH_MANAGER  # type: ignore
        summary = _HEALTH_MANAGER.check_health()
        if not summary:
            return None

        sla = summary.get("sla", {}) or summary.get("summary", {}).get("sla", {})
        return {
            "availability_trend": sla.get("availability_trend", []),
            "latency_trend": sla.get("latency_trend", []),
            "error_trend": sla.get("error_trend", []),
        }
    except Exception as exc:
        logger.debug(structured_msg("Trend data unavailable", error=str(exc)))
        return None


def create_health_components() -> list[dict[str, Any]]:
    """Erstellt Health-Komponenten-Liste.

    Returns:
        Liste der Health-Komponenten
    """
    components = []

    try:
        from api.routes.health_routes import check_agent_system, get_system_metrics  # type: ignore

        # Agent System Component
        agent = check_agent_system()
        components.append({
            "name": COMPONENT_AGENT_SYSTEM,
            "status": agent.status,
            "details": agent.details or {}
        })

        # System Component
        metrics = get_system_metrics()
        components.append({
            "name": COMPONENT_SYSTEM,
            "status": HEALTH_STATUS_HEALTHY,
            "details": {
                "cpu": metrics.cpu_percent,
                "mem": metrics.memory_percent
            }
        })

    except Exception as exc:
        logger.warning(structured_msg("Health components creation failed", error=str(exc)))

    return components


def determine_overall_status(components: list[dict[str, Any]]) -> str:
    """Bestimmt den Gesamtstatus basierend auf Komponenten.

    Args:
        components: Liste der Health-Komponenten

    Returns:
        Gesamtstatus als String
    """
    if not components:
        return HEALTH_STATUS_UNHEALTHY

    # Wenn alle Komponenten healthy sind, ist der Gesamtstatus healthy
    all_healthy = all(
        comp.get("status") == HEALTH_STATUS_HEALTHY
        for comp in components
    )

    return HEALTH_STATUS_HEALTHY if all_healthy else "degraded"


def create_error_response(
    execution_id: str,
    function_name: str,
    start_time: datetime,
    error_code: str,
    error_message: str,
    details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt eine standardisierte Error-Response.

    Args:
        execution_id: Eindeutige Execution-ID
        function_name: Name der Funktion
        start_time: Startzeitpunkt der Ausführung
        error_code: Fehlercode
        error_message: Fehlermeldung
        details: Zusätzliche Fehlerdetails

    Returns:
        Standardisierte Error-Response
    """
    return {
        "execution_id": execution_id,
        "function": function_name,
        "status": EXECUTION_STATUS_ERROR,
        "executed_at": start_time,
        "duration_ms": calculate_duration_ms(start_time),
        "error": {
            "code": error_code,
            "message": error_message,
            "details": details
        }
    }


def log_and_handle_exception(
    operation: str,
    exception: Exception,
    context: dict[str, Any] | None = None
) -> None:
    """Loggt und behandelt Exceptions einheitlich.

    Args:
        operation: Name der Operation
        exception: Aufgetretene Exception
        context: Zusätzlicher Kontext für das Logging
    """
    log_context = {"operation": operation, "error": str(exception)}
    if context:
        log_context.update(context)

    logger.warning(structured_msg(f"{operation} error", **log_context))


def safe_import_and_call(
    module_path: str,
    function_name: str,
    *args,
    **kwargs
) -> Any:
    """Sicherer Import und Funktionsaufruf mit Error-Handling.

    Args:
        module_path: Pfad zum Modul
        function_name: Name der Funktion
        *args: Positionsargumente
        **kwargs: Keyword-Argumente

    Returns:
        Funktionsergebnis oder None bei Fehlern
    """
    try:
        module = __import__(module_path, fromlist=[function_name])
        func = getattr(module, function_name)
        return func(*args, **kwargs)
    except Exception as exc:
        log_and_handle_exception(
            f"safe_import_and_call({module_path}.{function_name})",
            exc
        )
        return None


__all__ = [
    "calculate_duration_ms",
    "create_error_response",
    "create_execution_id",
    "create_health_components",
    "determine_overall_status",
    "log_and_handle_exception",
    "safe_get_sla_metrics",
    "safe_get_trend_data",
    "safe_import_and_call",
]
