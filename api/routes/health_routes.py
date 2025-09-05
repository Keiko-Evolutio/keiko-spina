# backend/api/routes/health_routes.py
"""Health Check API Routes für Azure AI Foundry - Refactored mit konsolidierten Utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Legacy-Imports (werden schrittweise ersetzt)
from monitoring.health_checks import HealthCheckConfig, HealthCheckManager
from storage.cache.redis_cache import NoOpCache, get_cache_client

# Konsolidierte Utilities
from ..common.health_utils import (
    ServiceNames,
    create_base_health_response,
    create_detailed_health_response,
)
from ..common.router_factory import RouterFactory

# from ..common.api_constants import APIPaths  # Entfernt - ersetzt durch UnifiedAuthMiddleware

# Router-Konfiguration mit konsolidierter Factory
router = RouterFactory.create_health_router("/health")  # Hardcoded path statt APIPaths


# Datenmodelle
class ComponentHealth(BaseModel):
    """Health Status einer Komponente."""
    name: str = Field(..., description="Komponenten-Name")
    status: str = Field(..., description="Status: healthy, degraded, unhealthy")
    message: str | None = Field(None, description="Status-Nachricht")
    details: dict[str, Any] | None = Field(None, description="Detail-Informationen")
    last_check: datetime = Field(default_factory=lambda: datetime.now(UTC))
    response_time_ms: int | None = Field(None, description="Response-Zeit in ms")


class SystemMetrics(BaseModel):
    """System-Metriken."""
    cpu_percent: float = Field(..., description="CPU-Auslastung in %")
    memory_percent: float = Field(..., description="RAM-Auslastung in %")
    disk_percent: float = Field(..., description="Festplatten-Auslastung in %")
    uptime_seconds: int = Field(..., description="System-Uptime in Sekunden")
    process_count: int = Field(..., description="Anzahl laufender Prozesse")


class SLOStatus(BaseModel):
    """SLO Kennzahlen pro Operation."""
    operation: str
    p95_ms: float
    error_rate_pct: float
    p95_ok: bool
    error_rate_ok: bool


# Health Check Funktionen werden aus health_utils importiert
# Lokale Wrapper für Kompatibilität mit Tests
def get_local_system_metrics() -> SystemMetrics:
    """Sammelt System-Metriken als SystemMetrics-Objekt für lokale Verwendung."""
    return SystemMetrics(
        cpu_percent=round(psutil.cpu_percent(), 1),
        memory_percent=round(psutil.virtual_memory().percent, 1),
        disk_percent=round(psutil.disk_usage("/").percent, 1),
        uptime_seconds=int(datetime.now().timestamp() - psutil.boot_time()),
        process_count=len(psutil.pids())
    )


def check_agents_integration() -> bool:
    """Prüft ob Agent-Integration verfügbar ist."""
    try:
        import agents
        return agents.get_system_status().get("core_functional", False)
    except ImportError:
        return False

def check_agent_system() -> ComponentHealth:
    """Prüft Agent System Health."""
    if not check_agents_integration():
        return ComponentHealth(
            name="agent_system",
            status="unavailable",
            message="Agent System nicht verfügbar"
        )

    try:
        import agents
        status = agents.get_system_status()
        return ComponentHealth(
            name="agent_system",
            status="healthy",
            message="Agent System funktionsfähig",
            details=status
        )
    except Exception as e:
        return ComponentHealth(
            name="agent_system",
            status="degraded",
            message=f"Agent System Fehler: {e!s}"
        )


# API Endpunkte - Refactored mit konsolidierten Utilities
@router.get("/")
async def health_check():
    """Basis Health Check mit konsolidierten Utilities.

    Returns:
        Standardisierte Health-Response
    """
    return create_base_health_response(
        service_name=ServiceNames.KEIKO_API,
        version="1.0.0"
    )


@router.get("/detailed")
async def detailed_health_check():
    """Detaillierter Health Check mit konsolidierten Utilities.

    Returns:
        Detaillierte Health-Response mit System-Metriken
    """
    # Note: Alerting status could be added to health response in the future

    # Verwende konsolidierte Health-Response-Erstellung
    return create_detailed_health_response(
        service_name=ServiceNames.KEIKO_API,
        version="1.0.0",
        include_system_metrics=True,
        include_python_info=True,
        additional_checkers=None  # Können später hinzugefügt werden
    )


# Health Manager (SLO-Monitoring wurde durch kei_agents/slo_sla/ System ersetzt)
_HEALTH_MANAGER = HealthCheckManager(HealthCheckConfig(sla_latency_ms=200.0, sla_availability_target=0.999))


@router.on_event("startup")
async def _start_health_manager() -> None:
    """Startet den HealthCheckManager beim App-Startup und registriert Checks."""
    try:
        async def _redis_check() -> dict[str, Any]:
            """Prüft Redis-Verfügbarkeit und Latenz."""
            try:
                client = await get_cache_client()
            except Exception:
                client = None
            if client is None or isinstance(client, NoOpCache):
                raise RuntimeError("redis_not_configured")
            try:
                pong = await client.ping()  # type: ignore[attr-defined]
                return {"redis": "ok", "pong": bool(pong)}
            except Exception as exc:
                raise RuntimeError(f"redis_error:{exc}")

        _HEALTH_MANAGER.add_check("redis", _redis_check)
        await _HEALTH_MANAGER.start()
    except Exception:
        # Graceful Degradation: Health-Manager Fehler sind nicht fatal
        pass


@router.get("/slo", response_model=list[SLOStatus])
async def slo_status():
    """Gibt SLO‑Status (p95 Latenz, Error Rate) pro Operation zurück."""
    try:
        from agents.slo_sla import SLOSLAMonitor
        from agents.slo_sla.config import SLOSLAConfig
        config = SLOSLAConfig.create_default_config()
        slo_monitor = SLOSLAMonitor(config)
        results = slo_monitor.get_slo_metrics()
    except ImportError:
        # Define fallbacks for static analysis clarity
        SLOSLAMonitor = None  # type: ignore
        SLOSLAConfig = None  # type: ignore
        results = {}
    out: list[SLOStatus] = []
    for op, vals in results.items():
        out.append(SLOStatus(
            operation=op,
            p95_ms=float(vals.get("p95_ms", 0.0)),
            error_rate_pct=float(vals.get("error_rate_pct", 0.0)),
            p95_ok=bool(vals.get("p95_ok", 0.0)),
            error_rate_ok=bool(vals.get("error_rate_ok", 0.0)),
        ))
    return out


@router.get("/ready")
async def readiness_check():
    """Kubernetes Readiness Check."""
    agent_health = check_agent_system()

    if agent_health.status in ["degraded", "unhealthy"]:
        raise HTTPException(status_code=503, detail="Service nicht bereit")

    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Kubernetes Liveness Check."""
    return {
        "status": "alive",
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_seconds": int(datetime.now().timestamp() - psutil.boot_time())
    }
