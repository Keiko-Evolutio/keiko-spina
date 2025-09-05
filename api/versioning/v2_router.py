"""V2 Router – Verbesserte v2 Endpunkte mit angepassten Schemas."""

from __future__ import annotations

from fastapi import APIRouter, Header

from kei_logging import get_logger

from .constants import (
    API_VERSION_NUMBER_V2,
    API_VERSION_V2,
    HTTP_STATUS_ACCEPTED,
    HTTP_STATUS_OK,
    SERVICE_NAME,
    URL_PREFIX_V2,
)
from .models import AgentSummary, HealthV2Response, WebhookAck
from .utils import (
    create_health_components,
    determine_overall_status,
    log_and_handle_exception,
    safe_get_sla_metrics,
    safe_get_trend_data,
)

logger = get_logger(__name__)

v2_router = APIRouter(prefix=URL_PREFIX_V2, tags=[API_VERSION_V2])


@v2_router.get("/health", response_model=HealthV2Response, status_code=HTTP_STATUS_OK)
async def health_v2() -> HealthV2Response:
    """V2 Health Endpoint mit konsistentem Schema inkl. SLA Feldern."""
    try:
        components = create_health_components()
        overall_status = determine_overall_status(components)
        sla_metrics = safe_get_sla_metrics()
        trend_data = safe_get_trend_data()

        return HealthV2Response(
            service=SERVICE_NAME,
            version=API_VERSION_NUMBER_V2,
            status=overall_status,
            components=components,
            sla_metrics=sla_metrics,
            trend_data=trend_data,
        )
    except Exception as exc:  # pragma: no cover
        log_and_handle_exception("v2.health", exc)
        return _create_fallback_health_response()


def _create_fallback_health_response() -> HealthV2Response:
    """Erstellt eine Fallback-Health-Response bei Fehlern.

    Returns:
        Minimale Health-Response mit unhealthy Status
    """
    return HealthV2Response(
        service=SERVICE_NAME,
        version=API_VERSION_NUMBER_V2,
        status="unhealthy",
        components=[]
    )


@v2_router.get("/agents", response_model=list[AgentSummary])
async def list_agents_v2() -> list[AgentSummary]:
    """V2 Agents‑Liste (vereinfachtes Schema)."""
    try:
        from api.routes.agents_routes import list_agents  # type: ignore
        agents = await list_agents()  # type: ignore[func-returns-value]
        return _convert_agents_to_summaries(agents)
    except Exception as exc:
        log_and_handle_exception("v2.list_agents", exc)
        return []


def _convert_agents_to_summaries(agents: list) -> list[AgentSummary]:
    """Konvertiert Agent-Daten zu AgentSummary-Objekten.

    Args:
        agents: Liste der Agent-Daten

    Returns:
        Liste der AgentSummary-Objekte
    """
    summaries: list[AgentSummary] = []
    for item in agents:
        try:
            cfg = item.get("configuration", {}) if isinstance(item, dict) else {}
            agent_id = str(cfg.get("id", cfg.get("name", "agent")))
            agent_name = str(cfg.get("name", "agent"))
            category = cfg.get("category")

            summaries.append(AgentSummary(
                id=agent_id,
                name=agent_name,
                category=category
            ))
        except Exception as exc:
            log_and_handle_exception("agent_conversion", exc, {"item": str(item)})
            continue

    return summaries


@v2_router.post("/webhooks/ingest", response_model=WebhookAck, status_code=HTTP_STATUS_ACCEPTED)
async def ingest_webhook_v2(
    _x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id")
) -> WebhookAck:
    """V2 Webhook Ingest Dummy – bestätigt Empfang (Schema stabil)."""
    return WebhookAck()


# Funktionen v2 als Subrouter einbinden
try:
    from .v2_functions import router as v2_functions_router  # type: ignore
    v2_router.include_router(v2_functions_router)
except Exception:
    pass


__all__ = ["v2_router"]
