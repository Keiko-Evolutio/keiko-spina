"""Monitoring API Routes.
Stellt REST-Endpoints für das Comprehensive Monitoring System bereit.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from core.container import get_container
from kei_logging import get_logger
from monitoring.interfaces import IMonitoringService

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def get_monitoring_service() -> IMonitoringService:
    """Dependency für Monitoring Service."""
    try:
        container = get_container()
        return container.resolve(IMonitoringService)
    except Exception as e:
        logger.error(f"Failed to resolve monitoring service: {e}")
        raise HTTPException(status_code=503, detail="Monitoring service not available")


@router.get("/health", summary="Overall Health Check")
async def health_check(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Gibt den Gesamtstatus aller Services zurück.

    Returns:
        Dict mit Health-Status aller Services
    """
    try:
        await monitoring_service.health_checker.check_all_services()
        overall_health = monitoring_service.health_checker.get_overall_health()

        return {
            "status": overall_health["status"],
            "timestamp": datetime.utcnow().isoformat(),
            "services": overall_health["services"],
            "summary": {
                "total_services": overall_health["total_services"],
                "healthy_services": overall_health["healthy_services"],
                "all_services_healthy": overall_health["all_services_healthy"],
                "critical_services_healthy": overall_health["critical_services_healthy"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/{service_name}", summary="Service-specific Health Check")
async def service_health_check(
    service_name: str,
    monitoring_service: IMonitoringService = Depends(get_monitoring_service)
) -> dict[str, Any]:
    """Gibt Health-Status eines spezifischen Services zurück.

    Args:
        service_name: Name des Services
        monitoring_service: Monitoring Service für Health-Checks

    Returns:
        Dict mit Health-Status des Services
    """
    try:
        health_status = await monitoring_service.health_checker.check_health(service_name)

        return {
            "service_name": health_status.service_name,
            "is_healthy": health_status.is_healthy,
            "status": health_status.status,
            "details": health_status.details,
            "last_check": health_status.last_check.isoformat(),
            "response_time_ms": health_status.response_time_ms
        }
    except Exception as e:
        logger.error(f"Service health check failed for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed for service {service_name}")


@router.get("/readiness", summary="Readiness Probe")
async def readiness_probe(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Readiness Probe für Kubernetes/Container-Orchestrierung.

    Returns:
        Dict mit Readiness-Status
    """
    try:
        overall_health = monitoring_service.health_checker.get_overall_health()

        # Service ist ready wenn kritische Services gesund sind
        is_ready = overall_health.get("critical_services_healthy", False)

        if not is_ready:
            raise HTTPException(status_code=503, detail="Service not ready")

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Readiness probe failed")


@router.get("/liveness", summary="Liveness Probe")
async def liveness_probe(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Liveness Probe für Kubernetes/Container-Orchestrierung.

    Returns:
        Dict mit Liveness-Status
    """
    try:
        # Einfacher Check ob Monitoring Service läuft
        status = monitoring_service.get_monitoring_status()

        if not status.get("running", False):
            raise HTTPException(status_code=503, detail="Service not alive")

        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Liveness probe failed")


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus Metrics")
async def prometheus_metrics(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> str:
    """Exportiert Metriken im Prometheus-Format.

    Returns:
        Prometheus-formatierte Metriken als Plain Text
    """
    try:
        metrics = await monitoring_service.export_metrics("prometheus")
        return metrics
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics export failed")


@router.get("/status", summary="Monitoring System Status")
async def monitoring_status(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Gibt detaillierten Status des Monitoring-Systems zurück.

    Returns:
        Dict mit Monitoring-System-Status
    """
    try:
        return monitoring_service.get_monitoring_status()
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring status")


@router.get("/dashboard", summary="Dashboard Data")
async def dashboard_data(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Gibt Dashboard-Daten für Frontend zurück.

    Returns:
        Dict mit Dashboard-Daten
    """
    try:
        return monitoring_service.get_dashboard_data()
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")


@router.get("/voice/workflows", summary="Voice Workflow Statistics")
async def voice_workflow_stats(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Gibt Voice-Workflow-Statistiken zurück.

    Returns:
        Dict mit Voice-Workflow-Statistiken
    """
    try:
        return monitoring_service.voice_monitor.get_workflow_statistics()
    except Exception as e:
        logger.error(f"Failed to get voice workflow stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get voice workflow statistics")


@router.get("/performance/response-times", summary="Response Time Statistics")
async def response_time_stats(
    endpoint: str | None = Query(None, description="Filter by specific endpoint"),
    monitoring_service: IMonitoringService = Depends(get_monitoring_service)
) -> list[dict[str, Any]]:
    """Gibt Response-Zeit-Statistiken zurück.

    Args:
        endpoint: Optional - Filter für spezifischen Endpoint
        monitoring_service: Monitoring Service für Performance-Statistiken

    Returns:
        Liste mit Response-Zeit-Statistiken
    """
    try:
        stats = monitoring_service.performance_monitor.get_response_time_stats(endpoint)

        return [
            {
                "endpoint": stat.endpoint,
                "count": stat.count,
                "total_time_ms": stat.total_time_ms,
                "min_time_ms": stat.min_time_ms,
                "max_time_ms": stat.max_time_ms,
                "avg_time_ms": stat.avg_time_ms,
                "p95_time_ms": stat.p95_time_ms,
                "p99_time_ms": stat.p99_time_ms
            }
            for stat in stats
        ]
    except Exception as e:
        logger.error(f"Failed to get response time stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response time statistics")


@router.get("/performance/throughput", summary="Throughput Statistics")
async def throughput_stats(
    endpoint: str | None = Query(None, description="Filter by specific endpoint"),
    window_seconds: int = Query(60, description="Time window in seconds"),
    monitoring_service: IMonitoringService = Depends(get_monitoring_service)
) -> list[dict[str, Any]]:
    """Gibt Throughput-Statistiken zurück.

    Args:
        endpoint: Optional - Filter für spezifischen Endpoint
        window_seconds: Zeitfenster in Sekunden
        monitoring_service: Monitoring Service für Throughput-Statistiken

    Returns:
        Liste mit Throughput-Statistiken
    """
    try:
        stats = monitoring_service.performance_monitor.get_throughput_stats(endpoint, window_seconds)

        return [
            {
                "endpoint": stat.endpoint,
                "requests_per_second": stat.requests_per_second,
                "total_requests": stat.total_requests,
                "time_window_seconds": stat.time_window_seconds
            }
            for stat in stats
        ]
    except Exception as e:
        logger.error(f"Failed to get throughput stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get throughput statistics")


@router.get("/performance/resources", summary="Resource Usage Statistics")
async def resource_stats(
    window_seconds: int = Query(300, description="Time window in seconds"),
    monitoring_service: IMonitoringService = Depends(get_monitoring_service)
) -> dict[str, Any]:
    """Gibt Resource-Usage-Statistiken zurück.

    Args:
        window_seconds: Zeitfenster in Sekunden
        monitoring_service: Monitoring Service für Resource-Statistiken

    Returns:
        Dict mit Resource-Usage-Statistiken
    """
    try:
        return monitoring_service.performance_monitor.get_resource_stats(window_seconds)
    except Exception as e:
        logger.error(f"Failed to get resource stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get resource statistics")


@router.get("/alerts", summary="Active Alerts")
async def active_alerts(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> list[dict[str, Any]]:
    """Gibt aktive Alerts zurück.

    Returns:
        Liste mit aktiven Alerts
    """
    try:
        alerts = monitoring_service.alert_manager.get_active_alerts()

        return [
            {
                "name": alert.name,
                "severity": alert.severity.value,
                "message": alert.message,
                "labels": alert.labels,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in alerts
        ]
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active alerts")


@router.get("/circuit-breakers", summary="Circuit Breaker Status")
async def circuit_breaker_status(monitoring_service: IMonitoringService = Depends(get_monitoring_service)) -> dict[str, Any]:
    """Gibt Status aller Circuit Breaker zurück.

    Returns:
        Dict mit Circuit Breaker Status
    """
    try:
        return monitoring_service.circuit_breaker_manager.get_all_statistics()
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get circuit breaker status")
