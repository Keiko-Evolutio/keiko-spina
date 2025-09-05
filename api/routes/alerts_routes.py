"""Alerts API Routes - Prometheus Alertmanager Integration
Empfängt und verarbeitet Alerts von Prometheus Alertmanager.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from kei_logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class AlertLabel(BaseModel):
    """Alert Labels von Prometheus."""
    alertname: str
    instance: str | None = None
    job: str | None = None
    severity: str | None = None
    component: str | None = None
    environment: str | None = None


class Alert(BaseModel):
    """Einzelner Alert von Prometheus Alertmanager."""
    status: str = Field(..., description="Alert Status: firing, resolved")
    labels: dict[str, str] = Field(..., description="Alert Labels")
    annotations: dict[str, str] | None = Field(default_factory=dict, description="Alert Annotations")
    startsAt: str | None = Field(None, description="Alert Start Time")
    endsAt: str | None = Field(None, description="Alert End Time")
    generatorURL: str | None = Field(None, description="Generator URL")
    fingerprint: str | None = Field(None, description="Alert Fingerprint")


class AlertmanagerWebhook(BaseModel):
    """Alertmanager Webhook Payload."""
    receiver: str = Field(..., description="Webhook Receiver Name")
    status: str = Field(..., description="Group Status: firing, resolved")
    alerts: list[Alert] = Field(..., description="List of Alerts")
    groupLabels: dict[str, str] | None = Field(default_factory=dict, description="Group Labels")
    commonLabels: dict[str, str] | None = Field(default_factory=dict, description="Common Labels")
    commonAnnotations: dict[str, str] | None = Field(default_factory=dict, description="Common Annotations")
    externalURL: str | None = Field(None, description="Alertmanager External URL")
    version: str | None = Field(None, description="Alertmanager Version")
    groupKey: str | None = Field(None, description="Group Key")
    truncatedAlerts: int | None = Field(None, description="Number of Truncated Alerts")


@router.post("/alertmanager", operation_id="alerts_receive_alertmanager_webhook")
async def receive_alertmanager_webhook(
    webhook: AlertmanagerWebhook,
    request: Request
) -> dict[str, Any]:
    """Empfängt Prometheus Alertmanager Webhooks.

    Dieser Endpoint verarbeitet eingehende Alerts von Prometheus Alertmanager
    und kann für Benachrichtigungen, Logging oder weitere Aktionen verwendet werden.
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info("📊 Alertmanager Webhook erhalten:")
    logger.info(f"  📍 Von: {client_ip} ({user_agent})")
    logger.info(f"  📨 Receiver: {webhook.receiver}")
    logger.info(f"  🚨 Status: {webhook.status}")
    logger.info(f"  📋 Anzahl Alerts: {len(webhook.alerts)}")

    # Verarbeite jeden Alert
    processed_alerts = []
    for alert in webhook.alerts:
        alert_info = {
            "alertname": alert.labels.get("alertname", "unknown"),
            "status": alert.status,
            "instance": alert.labels.get("instance", "unknown"),
            "severity": alert.labels.get("severity", "unknown"),
            "component": alert.labels.get("component", "unknown"),
            "environment": alert.labels.get("environment", "unknown"),
            "timestamp": datetime.now().isoformat()
        }

        # Log Alert Details
        if alert.status == "firing":
            logger.warning(f"🔥 FIRING Alert: {alert_info['alertname']} on {alert_info['instance']}")
        elif alert.status == "resolved":
            logger.info(f"✅ RESOLVED Alert: {alert_info['alertname']} on {alert_info['instance']}")

        # Log Annotations (falls vorhanden)
        if alert.annotations:
            for key, value in alert.annotations.items():
                logger.info(f"  📝 {key}: {value}")

        processed_alerts.append(alert_info)

    # Response für Alertmanager
    response = {
        "status": "received",
        "timestamp": datetime.now().isoformat(),
        "receiver": webhook.receiver,
        "alerts_processed": len(processed_alerts),
        "alerts": processed_alerts
    }

    logger.info(f"✅ Alertmanager Webhook erfolgreich verarbeitet: {len(processed_alerts)} Alerts")

    return response


@router.get("/health", operation_id="alerts_health_check")
async def alerts_health_check() -> dict[str, Any]:
    """Health Check für Alerts API."""
    return {
        "status": "healthy",
        "service": "alerts-api",
        "endpoints": {
            "alertmanager": "/api/v1/alerts/alertmanager",
            "health": "/api/v1/alerts/health"
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/", operation_id="alerts_list_alerts_endpoints")
async def list_alerts_endpoints() -> dict[str, Any]:
    """Listet verfügbare Alerts Endpoints auf."""
    return {
        "service": "Keiko Alerts API",
        "description": "Prometheus Alertmanager Integration",
        "endpoints": {
            "POST /api/v1/alerts/alertmanager": "Empfängt Alertmanager Webhooks",
            "GET /api/v1/alerts/health": "Health Check",
            "GET /api/v1/alerts/": "Diese Übersicht"
        },
        "supported_formats": ["Prometheus Alertmanager v0.27.0+"],
        "timestamp": datetime.now().isoformat()
    }


# Backward compatibility - falls jemand den alten Pfad verwendet
@router.post("/prometheus", operation_id="alerts_receive_prometheus_webhook")
async def receive_prometheus_webhook(webhook: AlertmanagerWebhook, request: Request):
    """Backward compatibility für Prometheus Webhooks."""
    logger.info("📊 Prometheus Webhook über Legacy-Endpoint erhalten - weiterleitung an Alertmanager")
    return await receive_alertmanager_webhook(webhook, request)


__all__ = ["router"]
