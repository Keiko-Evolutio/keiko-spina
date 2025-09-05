"""Alertmanager Webhook API-Routen (mit WS-Broadcast).

Empfängt Prometheus Alertmanager Webhooks, verteilt Benachrichtigungen
und sendet optionale Echtzeit-Updates über WebSocket an verbundene Clients.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request, status
from fastapi.responses import JSONResponse

from config.settings import settings
from data_models.websocket import create_status_update
from kei_logging import get_logger
from services.streaming import websocket_manager
from services.webhooks.alerting import get_alert_dispatcher
from services.webhooks.templates.alert_templates import (
    render_compact_summary,
    render_email_body,
    render_email_subject,
    render_sms_text,
)

logger = get_logger(__name__)


router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


def _extract_tenant(labels: dict[str, Any]) -> str | None:
    """Extrahiert optionalen Tenant aus Alert-Labels."""
    for key in ("tenant", "tenant_id", "x_tenant_id", "X-Tenant-Id"):
        value = labels.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@router.post("/alertmanager-ws", status_code=status.HTTP_202_ACCEPTED, operation_id="alerts-receive_alertmanager_webhook_ws")
async def receive_alertmanager_webhook_ws(
    request: Request,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> JSONResponse:
    """Empfängt Alertmanager Webhook-Events und triggert Benachrichtigungen."""
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:
        logger.warning(f"Ungültige Alertmanager-Payload: {exc}")
        return JSONResponse({"status": "invalid"}, status_code=status.HTTP_400_BAD_REQUEST)

    alerts: list[dict[str, Any]] = payload.get("alerts", []) if isinstance(payload, dict) else []
    if not alerts:
        return JSONResponse({"status": "ignored", "reason": "no_alerts"}, status_code=status.HTTP_202_ACCEPTED)

    dispatcher = get_alert_dispatcher()

    processed = 0
    for alert in alerts:
        try:
            labels: dict[str, Any] = alert.get("labels", {})
            alert_annotations: dict[str, Any] = alert.get("annotations", {})
            severity: str = str(labels.get("severity", "warning")).lower()

            tenant_id = x_tenant_id or _extract_tenant(labels)

            title = alert_annotations.get("summary") or labels.get("alertname") or "Keiko Alert"
            compact = render_compact_summary(labels=labels, annotations=alert_annotations, tenant_id=tenant_id)

            await dispatcher.send_alert(title=title, message={**compact, "tenant": tenant_id}, severity=severity)
            if severity in ("warning", "critical"):
                try:
                    subject = render_email_subject(title=title, severity=severity)
                    body = render_email_body(labels=labels, annotations=alert_annotations, tenant_id=tenant_id)
                    await dispatcher.send_email(subject=subject, body=body, severity=severity)
                except Exception:
                    pass
            if severity == "critical":
                try:
                    sms_text = render_sms_text(title=title, labels=labels, tenant_id=tenant_id)
                    await dispatcher.send_sms(text=sms_text, severity=severity)
                except Exception:
                    pass

            # Echtzeit WS-Update (Best Effort)
            try:
                status_event = create_status_update(status="alert", details=f"{severity.upper()}: {title}")
                await websocket_manager.broadcast(status_event)
            except Exception:
                pass

            processed += 1
        except Exception as exc:
            logger.warning(f"Fehler bei Alert-Verarbeitung: {exc}")
            continue

    return JSONResponse({"status": "accepted", "processed": processed}, status_code=status.HTTP_202_ACCEPTED)


@router.get("/health", status_code=status.HTTP_200_OK)
async def alerting_health() -> dict[str, Any]:
    """Einfacher Health-Check für das Alerting-System."""
    disp = get_alert_dispatcher()
    summary = settings.get_config_summary()
    return {
        "status": "healthy",
        "channels": {
            "slack": bool(getattr(disp, "slack", None)),
            "teams": bool(getattr(disp, "teams", None)),
            "email": bool(getattr(disp, "email", None)),
            "sms": bool(getattr(disp, "sms", None)),
        },
        "config": {
            "alerting_enabled": settings.alerting_enabled,
            "rate_limit_per_minute": settings.alert_rate_limit_per_minute,
            "environment": summary.get("environment"),
        },
    }


__all__ = ["router"]
