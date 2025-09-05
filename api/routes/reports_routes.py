"""Report-Management API.

Enterprise-grade API für Report-Generierung, Scheduler-Management und Service-Status.
Integriert mit dem ReportingService für robuste Funktionalität.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from kei_logging import get_logger
from services.reporting import ReportingService

logger = get_logger(__name__)


router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

_service: ReportingService | None = None


def _get_service() -> ReportingService:
    """Gibt die globale ReportingService-Instanz zurück.

    Returns:
        ReportingService-Instanz
    """
    global _service
    if _service is None:
        _service = ReportingService()
    return _service


@router.post("/generate")
async def generate_report() -> dict[str, Any]:
    """Generiert einen manuellen Report.

    Führt eine einmalige Report-Generierung und -Verteilung durch,
    unabhängig vom automatischen Scheduler.

    Returns:
        Dictionary mit Generierungs-Ergebnis

    Raises:
        HTTPException: Bei Report-Generierungsfehlern
    """
    try:
        service = _get_service()
        return await service.generate_manual_report()
    except Exception as exc:
        logger.exception(f"Manuelle Report-Generierung fehlgeschlagen: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Report-Generierung fehlgeschlagen: {exc!s}"
        ) from exc


@router.post("/scheduler/start")
async def start_scheduler() -> dict[str, Any]:
    """Startet den Reporting-Scheduler manuell.

    Returns:
        Dictionary mit Start-Ergebnis

    Raises:
        HTTPException: Bei Scheduler-Start-Fehlern
    """
    try:
        service = _get_service()
        return await service.start_scheduler()
    except Exception as exc:
        logger.exception(f"Scheduler-Start fehlgeschlagen: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Scheduler-Start fehlgeschlagen: {exc!s}"
        ) from exc


@router.post("/scheduler/stop")
async def stop_scheduler() -> dict[str, Any]:
    """Stoppt den Reporting-Scheduler manuell.

    Returns:
        Dictionary mit Stop-Ergebnis
    """
    try:
        service = _get_service()
        return await service.stop_scheduler()
    except Exception as exc:
        logger.exception(f"Scheduler-Stop fehlgeschlagen: {exc}")
        # Bei Stop-Fehlern nicht mit Exception antworten
        return {
            "success": False,
            "message": f"Scheduler-Stop fehlgeschlagen: {exc!s}"
        }


@router.get("/status")
async def get_service_status() -> dict[str, Any]:
    """Gibt detaillierten Service-Status zurück.

    Returns:
        Dictionary mit Service-Status und Health-Informationen
    """
    try:
        service = _get_service()
        health_data = await service.health_check()
        service_info = service.get_service_info()

        return {
            "health": health_data,
            "info": service_info,
            "timestamp": health_data.get("timestamp")
        }
    except Exception as exc:
        logger.exception(f"Status-Abfrage fehlgeschlagen: {exc}")
        return {
            "health": {
                "status": "unhealthy",
                "error": str(exc)
            },
            "info": {
                "name": "reporting",
                "status": "error"
            }
        }


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Einfacher Health-Check-Endpunkt.

    Returns:
        Dictionary mit Health-Status
    """
    try:
        service = _get_service()
        return await service.health_check()
    except Exception as exc:
        logger.exception(f"Health-Check fehlgeschlagen: {exc}")
        return {
            "status": "unhealthy",
            "service": "ReportingService",
            "error": str(exc)
        }


@router.post("/generate/template/{template_id}")
async def generate_template_report(template_id: str, recipients: list[str] | None = None) -> dict[str, Any]:
    """Generiert Report basierend auf Template.

    Args:
        template_id: ID des zu verwendenden Templates
        recipients: Optionale Empfänger-Liste

    Returns:
        Dictionary mit Generierungs-Ergebnis

    Raises:
        HTTPException: Bei Template- oder Generierungsfehlern
    """
    try:
        service = _get_service()
        return await service.scheduler.generate_template_report(template_id, recipients)
    except Exception as exc:
        logger.exception(f"Template-Report-Generierung fehlgeschlagen: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Template-Report-Generierung fehlgeschlagen: {exc!s}"
        ) from exc


@router.get("/templates")
async def list_templates() -> dict[str, Any]:
    """Gibt verfügbare Report-Templates zurück.

    Returns:
        Dictionary mit Template-Liste
    """
    try:
        service = _get_service()
        templates = service.scheduler.get_available_templates()
        return {
            "templates": templates,
            "count": len(templates)
        }
    except Exception as exc:
        logger.exception(f"Template-Liste-Abfrage fehlgeschlagen: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Template-Liste-Abfrage fehlgeschlagen: {exc!s}"
        ) from exc


@router.get("/templates/{template_id}")
async def get_template(template_id: str) -> dict[str, Any]:
    """Gibt spezifisches Template zurück.

    Args:
        template_id: Template-ID

    Returns:
        Template-Details

    Raises:
        HTTPException: Falls Template nicht gefunden
    """
    try:
        service = _get_service()
        template = service.scheduler.template_manager.get_template(template_id)

        if not template:
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_id}' nicht gefunden"
            )

        return {
            "template": template.to_dict()
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Template-Abfrage fehlgeschlagen: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Template-Abfrage fehlgeschlagen: {exc!s}"
        ) from exc
