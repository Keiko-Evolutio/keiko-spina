"""Alertmanager Webhook API-Routen.

Empfängt Prometheus Alertmanager Webhooks und triggert Benachrichtigungen.
Die Endpunkte bieten sowohl einen stabilen Pfad ``/api/v1/alerts/alertmanager``
als auch das Alias ``/api/v1/alerts/webhook`` zur einfachen Konfiguration.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request, status
from fastapi.responses import JSONResponse

from kei_logging import get_logger
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


async def _process_alertmanager_payload(payload: dict[str, Any], tenant_header: str | None) -> dict[str, Any]:
    """Verarbeitet Alertmanager Payload und versendet Benachrichtigungen.

    Args:
        payload: JSON-Payload des Alertmanager Webhooks
        tenant_header: Optionaler Tenant Header

    Returns:
        Ergebnisobjekt mit Anzahl verarbeiteter Alerts
    """
    alerts: list[dict[str, Any]] = payload.get("alerts", []) if isinstance(payload, dict) else []
    if not alerts:
        return {"status": "ignored", "reason": "no_alerts", "processed": 0}

    dispatcher = get_alert_dispatcher()

    processed = 0
    for alert in alerts:
        try:
            labels: dict[str, Any] = alert.get("labels", {})
            alert_annotations: dict[str, Any] = alert.get("annotations", {})
            severity: str = str(labels.get("severity", "warning")).lower()

            tenant_id = tenant_header or _extract_tenant(labels)

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

            processed += 1
        except Exception as exc:
            logger.warning(f"Fehler bei Alert-Verarbeitung: {exc}")
            continue

    return {"status": "accepted", "processed": processed}


@router.post("/alertmanager", status_code=status.HTTP_202_ACCEPTED)
async def receive_alertmanager_webhook(
    request: Request,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> JSONResponse:
    """Empfängt Alertmanager Webhook-Events und triggert Benachrichtigungen."""
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:
        logger.warning(f"Ungültige Alertmanager-Payload: {exc}")
        return JSONResponse({"status": "invalid"}, status_code=status.HTTP_400_BAD_REQUEST)

    result = await _process_alertmanager_payload(payload, x_tenant_id)
    code = status.HTTP_202_ACCEPTED if result.get("status") != "invalid" else status.HTTP_400_BAD_REQUEST
    return JSONResponse(result, status_code=code)


@router.post("/webhook", status_code=status.HTTP_202_ACCEPTED)
async def receive_alertmanager_webhook_alias(
    request: Request,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> JSONResponse:
    """Alias-Endpunkt für Alertmanager Webhook (gleiches Verhalten)."""
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:
        logger.warning(f"Ungültige Alertmanager-Payload: {exc}")
        return JSONResponse({"status": "invalid"}, status_code=status.HTTP_400_BAD_REQUEST)

    result = await _process_alertmanager_payload(payload, x_tenant_id)
    code = status.HTTP_202_ACCEPTED if result.get("status") != "invalid" else status.HTTP_400_BAD_REQUEST
    return JSONResponse(result, status_code=code)


__all__ = ["router"]
