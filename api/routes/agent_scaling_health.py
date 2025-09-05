# backend/api/routes/agent_scaling_health.py
"""Agent Scaling und Health Monitoring Endpoints für Keiko Personal Assistant

Implementiert Auto-Scaling-Konfiguration, Performance-Hints und detaillierte
Gesundheitschecks für Agenten.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field

from kei_logging import (
    BusinessLogicError,
    LogLinkedError,
    ValidationError,
    get_logger,
    with_log_links,
)

from .enhanced_agents_management import (
    HealthCheckResult,
    HealthStatus,
    ScalingConfiguration,
    ScalingStatusResponse,
    get_agent_or_404,
    validate_agent_access,
)

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/agents", tags=["Agent Scaling & Health"])


class ScalingUpdateRequest(BaseModel):
    """Request-Model für Scaling-Updates."""
    strategy: str | None = Field(None, description="Scaling-Strategie")
    min_instances: int | None = Field(None, ge=1, description="Minimale Instanzen")
    max_instances: int | None = Field(None, ge=1, description="Maximale Instanzen")
    target_cpu_utilization: float | None = Field(None, ge=0.1, le=1.0, description="Ziel-CPU-Auslastung")
    target_memory_utilization: float | None = Field(None, ge=0.1, le=1.0, description="Ziel-Memory-Auslastung")
    scale_up_threshold: float | None = Field(None, ge=0.1, le=1.0, description="Scale-Up-Schwellwert")
    scale_down_threshold: float | None = Field(None, ge=0.1, le=1.0, description="Scale-Down-Schwellwert")
    cooldown_period_seconds: int | None = Field(None, ge=60, description="Cooldown-Periode in Sekunden")


class ScalingEvent(BaseModel):
    """Scaling-Event."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Event-ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Zeitstempel")
    event_type: str = Field(..., description="Event-Typ (scale_up, scale_down)")
    trigger_reason: str = Field(..., description="Auslöser-Grund")
    previous_instances: int = Field(..., description="Vorherige Instanzen-Anzahl")
    new_instances: int = Field(..., description="Neue Instanzen-Anzahl")
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict, description="Metriken-Snapshot")
    success: bool = Field(default=True, description="Scaling erfolgreich")
    error_message: str | None = Field(None, description="Fehlermeldung bei Misserfolg")


class HealthCheck(BaseModel):
    """Gesundheitscheck-Definition."""
    check_name: str = Field(..., description="Name des Checks")
    check_type: str = Field(..., description="Typ des Checks")
    endpoint: str | None = Field(None, description="Endpoint für HTTP-Checks")
    timeout_seconds: int = Field(default=30, description="Timeout in Sekunden")
    interval_seconds: int = Field(default=60, description="Intervall in Sekunden")
    failure_threshold: int = Field(default=3, description="Fehlschlag-Schwellwert")
    success_threshold: int = Field(default=1, description="Erfolg-Schwellwert")
    enabled: bool = Field(default=True, description="Check aktiviert")


class DetailedHealthStatus(BaseModel):
    """Detaillierter Gesundheitsstatus."""
    overall_status: HealthStatus = Field(..., description="Gesamt-Gesundheitsstatus")
    last_check: datetime = Field(..., description="Letzter Check")
    uptime_seconds: float = Field(..., description="Uptime in Sekunden")
    checks: list[HealthCheckResult] = Field(default_factory=list, description="Einzelne Check-Ergebnisse")
    performance_metrics: dict[str, Any] = Field(default_factory=dict, description="Performance-Metriken")
    resource_usage: dict[str, Any] = Field(default_factory=dict, description="Ressourcenverbrauch")
    alerts: list[str] = Field(default_factory=list, description="Aktive Alerts")
    recommendations: list[str] = Field(default_factory=list, description="Empfehlungen")


def _get_scaling_configuration(agent_id: str) -> ScalingConfiguration:
    """Holt Scaling-Konfiguration für Agent."""
    try:
        # Integration mit Policy Engine oder Konfigurationssystem
        from policy_engine import policy_engine

        scaling_config = policy_engine.get_agent_scaling_config(agent_id)

        return ScalingConfiguration(**scaling_config)

    except ImportError:
        logger.warning("Policy Engine nicht verfügbar - verwende Standard-Scaling-Konfiguration")
        return ScalingConfiguration(
            strategy="horizontal",
            min_instances=1,
            max_instances=5,
            target_cpu_utilization=0.7,
            target_memory_utilization=0.8,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period_seconds=300
        )


def _save_scaling_configuration(agent_id: str, config: ScalingConfiguration) -> None:
    """Speichert Scaling-Konfiguration für Agent."""
    try:
        from policy_engine import policy_engine

        policy_engine.set_agent_scaling_config(agent_id, config.dict())

    except ImportError:
        logger.warning("Policy Engine nicht verfügbar - Scaling-Konfiguration nicht gespeichert")


def _get_current_scaling_status(agent_id: str) -> dict[str, Any]:
    """Holt aktuellen Scaling-Status für Agent."""
    try:
        # Integration mit Container-Orchestrierung oder Agent-Runtime
        from agent_runtime import runtime_manager

        status = runtime_manager.get_agent_scaling_status(agent_id)

        return {
            "current_instances": status.get("current_instances", 1),
            "target_instances": status.get("target_instances", 1),
            "pending_instances": status.get("pending_instances", 0),
            "last_scaling_action": status.get("last_scaling_action"),
            "scaling_in_progress": status.get("scaling_in_progress", False)
        }

    except ImportError:
        logger.warning("Agent Runtime nicht verfügbar - verwende Standard-Status")
        return {
            "current_instances": 1,
            "target_instances": 1,
            "pending_instances": 0,
            "last_scaling_action": None,
            "scaling_in_progress": False
        }


def _get_scaling_events(agent_id: str, limit: int = 10) -> list[ScalingEvent]:
    """Holt Scaling-Events für Agent."""
    try:
        from agent_runtime import runtime_manager

        events_data = runtime_manager.get_agent_scaling_events(agent_id, limit=limit)

        events = []
        for event_data in events_data:
            events.append(ScalingEvent(**event_data))

        return events

    except ImportError:
        logger.warning("Agent Runtime nicht verfügbar - keine Scaling-Events verfügbar")
        return []


async def _perform_health_check(agent_id: str, check: HealthCheck) -> HealthCheckResult:
    """Führt einzelnen Gesundheitscheck durch."""
    start_time = datetime.now(UTC)

    try:
        if check.check_type == "http":
            # HTTP-Gesundheitscheck
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=check.timeout_seconds)) as session:
                async with session.get(check.endpoint) as response:
                    response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

                    if response.status == 200:
                        return HealthCheckResult(
                            status=HealthStatus.HEALTHY,
                            response_time_ms=response_time,
                            details={
                                "check_name": check.check_name,
                                "status_code": response.status,
                                "response_size": len(await response.text())
                            }
                        )
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=response_time,
                        details={
                            "check_name": check.check_name,
                            "status_code": response.status
                        },
                        errors=[f"HTTP-Status {response.status}"]
                    )

        elif check.check_type == "agent_ping":
            # Agent-Ping-Check
            try:
                from agents.registry.dynamic_registry import dynamic_registry

                agent = await dynamic_registry.get_agent_by_id(agent_id)
                if agent and hasattr(agent, "ping"):
                    ping_result = await agent.ping()
                    response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY if ping_result else HealthStatus.UNHEALTHY,
                        response_time_ms=response_time,
                        details={
                            "check_name": check.check_name,
                            "ping_result": ping_result
                        }
                    )
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    details={"check_name": check.check_name},
                    errors=["Agent nicht verfügbar oder ping nicht unterstützt"]
                )

            except Exception as e:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    details={"check_name": check.check_name},
                    errors=[f"Agent-Ping fehlgeschlagen: {e!s}"]
                )

        elif check.check_type == "metrics":
            # Metriken-basierter Check
            try:
                from observability import get_agent_metrics_collector

                collector = get_agent_metrics_collector(agent_id)
                metrics = collector.get_comprehensive_metrics()

                # Prüfe Error-Rate
                task_metrics = metrics.get("task_metrics", {})
                error_rate = task_metrics.get("error_rate", 0)

                if error_rate < 0.1:  # Weniger als 10% Fehler
                    status = HealthStatus.HEALTHY
                elif error_rate < 0.3:  # Weniger als 30% Fehler
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.UNHEALTHY

                response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

                return HealthCheckResult(
                    status=status,
                    response_time_ms=response_time,
                    details={
                        "check_name": check.check_name,
                        "error_rate": error_rate,
                        "success_count": task_metrics.get("success_count", 0),
                        "failure_count": task_metrics.get("failure_count", 0)
                    }
                )

            except Exception as e:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    details={"check_name": check.check_name},
                    errors=[f"Metriken-Check fehlgeschlagen: {e!s}"]
                )

        else:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                details={"check_name": check.check_name},
                errors=[f"Unbekannter Check-Typ: {check.check_type}"]
            )

    except TimeoutError:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            details={"check_name": check.check_name},
            errors=[f"Timeout nach {check.timeout_seconds} Sekunden"]
        )

    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            details={"check_name": check.check_name},
            errors=[f"Unerwarteter Fehler: {e!s}"]
        )


def _get_default_health_checks() -> list[HealthCheck]:
    """Holt Standard-Gesundheitschecks für Agent."""
    return [
        HealthCheck(
            check_name="agent_ping",
            check_type="agent_ping",
            timeout_seconds=10,
            interval_seconds=30
        ),
        HealthCheck(
            check_name="metrics_health",
            check_type="metrics",
            timeout_seconds=15,
            interval_seconds=60
        ),
        HealthCheck(
            check_name="error_rate",
            check_type="metrics",
            timeout_seconds=10,
            interval_seconds=120
        )
    ]


def _calculate_overall_health_status(check_results: list[HealthCheckResult]) -> HealthStatus:
    """Berechnet Gesamt-Gesundheitsstatus aus einzelnen Checks."""
    if not check_results:
        return HealthStatus.UNKNOWN

    healthy_count = sum(1 for result in check_results if result.status == HealthStatus.HEALTHY)
    degraded_count = sum(1 for result in check_results if result.status == HealthStatus.DEGRADED)
    unhealthy_count = sum(1 for result in check_results if result.status == HealthStatus.UNHEALTHY)

    total_checks = len(check_results)

    # Wenn mehr als 50% unhealthy -> unhealthy
    if unhealthy_count / total_checks > 0.5:
        return HealthStatus.UNHEALTHY

    # Wenn mehr als 30% degraded oder unhealthy -> degraded
    if (degraded_count + unhealthy_count) / total_checks > 0.3:
        return HealthStatus.DEGRADED

    # Wenn mindestens 70% healthy -> healthy
    if healthy_count / total_checks >= 0.7:
        return HealthStatus.HEALTHY

    return HealthStatus.DEGRADED


def _generate_health_recommendations(check_results: list[HealthCheckResult], performance_metrics: dict[str, Any]) -> list[str]:
    """Generiert Gesundheits-Empfehlungen basierend auf Check-Ergebnissen."""
    recommendations = []

    # Prüfe auf häufige Probleme
    unhealthy_checks = [result for result in check_results if result.status == HealthStatus.UNHEALTHY]

    if unhealthy_checks:
        recommendations.append("Untersuche fehlgeschlagene Gesundheitschecks und behebe zugrundeliegende Probleme")

    # Performance-basierte Empfehlungen
    if performance_metrics.get("average_response_time_ms", 0) > 1000:
        recommendations.append("Hohe Antwortzeiten - prüfe Performance-Optimierungen")

    if performance_metrics.get("error_rate", 0) > 0.1:
        recommendations.append("Hohe Fehlerrate - analysiere Fehlerursachen und implementiere Fixes")

    if performance_metrics.get("memory_usage_percentage", 0) > 80:
        recommendations.append("Hoher Memory-Verbrauch - prüfe Memory-Leaks oder erhöhe Ressourcen")

    if performance_metrics.get("cpu_usage_percentage", 0) > 80:
        recommendations.append("Hohe CPU-Auslastung - optimiere Algorithmen oder skaliere horizontal")

    return recommendations


# Scaling Management Endpoints

@router.get("/{agent_id}/scaling", response_model=ScalingStatusResponse)
@with_log_links(component="scaling_management", operation="get_scaling")
async def get_agent_scaling(
    agent_id: str = Path(..., description="Agent-ID"),
    include_events: bool = Query(default=True, description="Scaling-Events einschließen"),
    events_limit: int = Query(default=10, ge=1, le=100, description="Anzahl Events")
) -> ScalingStatusResponse:
    """Holt Scaling-Konfiguration und aktuellen Status für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        include_events: Scaling-Events in Response einschließen
        events_limit: Maximale Anzahl Events

    Returns:
        Scaling-Konfiguration und Status-Informationen

    Raises:
        ValidationError: Bei ungültiger Agent-ID
        AuthorizationError: Bei fehlenden Berechtigungen
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="read")

    try:
        # Hole Scaling-Konfiguration
        config = _get_scaling_configuration(agent_id)

        # Hole aktuellen Status
        current_status = _get_current_scaling_status(agent_id)

        # Hole Scaling-Events
        scaling_events = []
        if include_events:
            events = _get_scaling_events(agent_id, limit=events_limit)
            scaling_events = [event.dict() for event in events]

        # Erstelle Response
        response_data = {
            "configuration": config,
            "current_instances": current_status["current_instances"],
            "target_instances": current_status["target_instances"],
            "scaling_events": scaling_events,
            "last_scaling_action": current_status["last_scaling_action"]
        }

        logger.info(
            f"Scaling-Informationen für Agent {agent_id} abgerufen",
            extra={
                "agent_id": agent_id,
                "current_instances": current_status["current_instances"],
                "target_instances": current_status["target_instances"],
                "correlation_id": f"scaling_get_{uuid.uuid4().hex[:8]}"
            }
        )

        return ScalingStatusResponse(**response_data)

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Scaling-Abfrage fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="scaling_management",
            operation="get_scaling",
            cause=e
        )


@router.put("/{agent_id}/scaling")
@with_log_links(component="scaling_management", operation="update_scaling")
async def update_agent_scaling(
    agent_id: str = Path(..., description="Agent-ID"),
    request: ScalingUpdateRequest = ...
) -> dict[str, Any]:
    """Aktualisiert Scaling-Konfiguration für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        request: Neue Scaling-Konfiguration

    Returns:
        Bestätigung der Scaling-Aktualisierung

    Raises:
        ValidationError: Bei ungültigen Scaling-Werten
        AuthorizationError: Bei fehlenden Berechtigungen
        BusinessLogicError: Bei Scaling-Update-Fehlern
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="admin")

    try:
        # Hole aktuelle Konfiguration
        current_config = _get_scaling_configuration(agent_id)

        # Update-Felder anwenden
        update_data = request.dict(exclude_unset=True)

        # Validiere Scaling-Konsistenz
        if "min_instances" in update_data and "max_instances" in update_data:
            if update_data["min_instances"] > update_data["max_instances"]:
                raise ValidationError(
                    message="min_instances darf nicht größer als max_instances sein",
                    field="min_instances",
                    value=update_data["min_instances"]
                )

        if "scale_up_threshold" in update_data and "scale_down_threshold" in update_data:
            if update_data["scale_down_threshold"] >= update_data["scale_up_threshold"]:
                raise ValidationError(
                    message="scale_down_threshold muss kleiner als scale_up_threshold sein",
                    field="scale_down_threshold",
                    value=update_data["scale_down_threshold"]
                )

        # Erstelle neue Konfiguration
        new_config_data = current_config.dict()
        new_config_data.update(update_data)
        new_config = ScalingConfiguration(**new_config_data)

        # Speichere neue Konfiguration
        _save_scaling_configuration(agent_id, new_config)

        logger.info(
            f"Scaling-Konfiguration für Agent {agent_id} aktualisiert",
            extra={
                "agent_id": agent_id,
                "updated_fields": list(update_data.keys()),
                "correlation_id": f"scaling_update_{uuid.uuid4().hex[:8]}"
            }
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "updated_at": datetime.now(UTC).isoformat(),
            "updated_fields": list(update_data.keys()),
            "new_configuration": new_config.dict()
        }

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Scaling-Update fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="scaling_management",
            operation="update_scaling",
            cause=e
        )


# Health Monitoring Endpoints

@router.get("/{agent_id}/health", response_model=DetailedHealthStatus)
@with_log_links(component="health_monitoring", operation="get_health")
async def get_agent_health(
    agent_id: str = Path(..., description="Agent-ID"),
    run_checks: bool = Query(default=True, description="Gesundheitschecks ausführen"),
    include_metrics: bool = Query(default=True, description="Performance-Metriken einschließen")
) -> DetailedHealthStatus:
    """Führt detaillierte Gesundheitschecks für Agent durch.

    Args:
        agent_id: Eindeutige Agent-ID
        run_checks: Gesundheitschecks ausführen
        include_metrics: Performance-Metriken einschließen

    Returns:
        Detaillierter Gesundheitsstatus

    Raises:
        ValidationError: Bei ungültiger Agent-ID
        AuthorizationError: Bei fehlenden Berechtigungen
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="read")

    try:
        check_results = []

        if run_checks:
            # Hole Gesundheitschecks
            health_checks = _get_default_health_checks()

            # Führe Checks parallel aus
            check_tasks = [_perform_health_check(agent_id, check) for check in health_checks]
            check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

            # Filtere Exceptions
            check_results = [
                result for result in check_results
                if isinstance(result, HealthCheckResult)
            ]

        # Berechne Gesamt-Status
        overall_status = _calculate_overall_health_status(check_results)

        # Hole Performance-Metriken
        performance_metrics = {}
        resource_usage = {}

        if include_metrics:
            try:
                from observability import get_agent_metrics_collector

                collector = get_agent_metrics_collector(agent_id)
                metrics = collector.get_comprehensive_metrics()

                task_metrics = metrics.get("task_metrics", {})
                performance_metrics = {
                    "total_requests": task_metrics.get("success_count", 0) + task_metrics.get("failure_count", 0),
                    "success_rate": task_metrics.get("success_rate", 0),
                    "error_rate": task_metrics.get("error_rate", 0),
                    "average_response_time_ms": task_metrics.get("latency", {}).get("mean", 0),
                    "p95_response_time_ms": task_metrics.get("latency", {}).get("p95", 0),
                    "requests_per_second": task_metrics.get("rate", {}).get("current_rate", 0)
                }

                # Placeholder für Ressourcenverbrauch
                resource_usage = {
                    "cpu_usage_percentage": 0,  # Placeholder
                    "memory_usage_percentage": 0,  # Placeholder
                    "disk_usage_percentage": 0,  # Placeholder
                    "network_usage_mbps": 0  # Placeholder
                }

            except ImportError:
                get_agent_metrics_collector = None  # type: ignore
                logger.warning("Observability-System nicht verfügbar - keine Metriken verfügbar")

        # Berechne Uptime
        uptime_seconds = 0
        try:
            from agents.registry.dynamic_registry import dynamic_registry
            agent = await dynamic_registry.get_agent_by_id(agent_id)
            if agent and hasattr(agent, "created_at"):
                uptime_seconds = (datetime.now(UTC) - agent.created_at).total_seconds()
        except:
            pass

        # Generiere Empfehlungen
        recommendations = _generate_health_recommendations(check_results, performance_metrics)

        # Generiere Alerts
        alerts = []
        if overall_status == HealthStatus.UNHEALTHY:
            alerts.append("Agent ist ungesund - sofortige Aufmerksamkeit erforderlich")
        elif overall_status == HealthStatus.DEGRADED:
            alerts.append("Agent-Performance ist beeinträchtigt")

        if performance_metrics.get("error_rate", 0) > 0.2:
            alerts.append("Hohe Fehlerrate erkannt")

        if performance_metrics.get("p95_response_time_ms", 0) > 2000:
            alerts.append("Hohe Latenz erkannt")

        # Erstelle Response
        health_status = DetailedHealthStatus(
            overall_status=overall_status,
            last_check=datetime.now(UTC),
            uptime_seconds=uptime_seconds,
            checks=check_results,
            performance_metrics=performance_metrics,
            resource_usage=resource_usage,
            alerts=alerts,
            recommendations=recommendations
        )

        logger.info(
            f"Gesundheitscheck für Agent {agent_id} durchgeführt",
            extra={
                "agent_id": agent_id,
                "overall_status": overall_status.value,
                "checks_performed": len(check_results),
                "alerts_count": len(alerts),
                "correlation_id": f"health_{uuid.uuid4().hex[:8]}"
            }
        )

        return health_status

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Gesundheitscheck fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="health_monitoring",
            operation="get_health",
            cause=e
        )
