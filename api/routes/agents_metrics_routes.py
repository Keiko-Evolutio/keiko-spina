"""Agent Metrics Exporter – schreibt Registry-Metriken (queue_length, desired_concurrency)
zyklisch als Gauges in den internen MetricsCollector, abrufbar für Dashboards.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector, MetricsConfig

# Optional import - Registry-System ist nicht immer verfügbar
try:
    from agents.capabilities.dynamic_registry import dynamic_registry
except ImportError:
    # Fallback wenn Registry nicht verfügbar ist
    dynamic_registry = None
import contextlib

# from security.kei_mcp_auth import require_auth  # Entfernt - ersetzt durch UnifiedAuthMiddleware

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agents-metrics", tags=["agents-metrics"])


# Globaler Metrics Collector für Agent‑Metriken
AGENTS_METRICS_COLLECTOR = MetricsCollector(MetricsConfig(enabled=True))

_exporter_task: asyncio.Task | None = None
_exporter_running: bool = False


async def _export_loop(interval_sec: float) -> None:
    """Hintergrundloop: schreibt Gauges aus Registry in den Collector."""
    global _exporter_running
    while _exporter_running:
        try:
            # Registry sicherstellen
            if dynamic_registry is None:
                await asyncio.sleep(5)  # Warte und versuche erneut
                continue

            if not getattr(dynamic_registry, "_initialized", False):
                # Registry auto-initialized - no action needed
                pass

            agents = getattr(dynamic_registry, "agents", {})
            # Gauges pro Agent schreiben
            for agent_id, agent in list(agents.items()):
                try:
                    qlen = getattr(agent, "queue_length", None)
                    dcon = getattr(agent, "desired_concurrency", None)
                    ccon = getattr(agent, "concurrency", None)
                    tags = {"agent_id": str(agent_id)}
                    if isinstance(qlen, int | float):
                        AGENTS_METRICS_COLLECTOR.record_gauge("agent.queue_length", float(qlen), tags)
                    if isinstance(dcon, int | float):
                        AGENTS_METRICS_COLLECTOR.record_gauge("agent.desired_concurrency", float(dcon), tags)
                    if isinstance(ccon, int | float):
                        AGENTS_METRICS_COLLECTOR.record_gauge("agent.current_concurrency", float(ccon), tags)
                except Exception as e:
                    logger.debug(f"Exporter Agent-Metriken Fehler ({agent_id}): {e}")
        except Exception as e:
            logger.debug(f"Exporter Loop Fehler: {e}")
        await asyncio.sleep(max(0.5, interval_sec))


@router.post("/exporter/start")  # Auth durch UnifiedAuthMiddleware
async def start_exporter(interval_sec: float = Query(5.0, ge=0.5, le=60.0)) -> dict[str, Any]:
    """Startet den Hintergrundexporter (idempotent)."""
    global _exporter_task, _exporter_running
    if _exporter_running and _exporter_task and not _exporter_task.done():
        return {"status": "already_running", "interval_sec": interval_sec}
    _exporter_running = True
    _exporter_task = asyncio.create_task(_export_loop(interval_sec))
    logger.info(f"Agent Metrics Exporter gestartet (interval={interval_sec}s)")
    return {"status": "started", "interval_sec": interval_sec, "started_at": datetime.utcnow().isoformat()}


@router.post("/exporter/stop")  # Auth durch UnifiedAuthMiddleware
async def stop_exporter() -> dict[str, Any]:
    """Stoppt den Hintergrundexporter (idempotent)."""
    global _exporter_task, _exporter_running
    _exporter_running = False
    if _exporter_task:
        with contextlib.suppress(Exception):
            _exporter_task.cancel()
    logger.info("Agent Metrics Exporter gestoppt")
    return {"status": "stopped", "stopped_at": datetime.utcnow().isoformat()}


@router.get("/metrics")  # Auth durch UnifiedAuthMiddleware
async def metrics_snapshot() -> dict[str, Any]:
    """Gibt eine Momentaufnahme der gesammelten Agent‑Metriken zurück."""
    return {
        "snapshot": AGENTS_METRICS_COLLECTOR.get_metrics(),
        "timestamp": datetime.utcnow().isoformat(),
    }
