"""Base Router Utilities für Azure AI Foundry."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter


def check_agents_integration() -> bool:
    """Prüft Agent System Verfügbarkeit."""
    try:
        from agents import get_system_status
        return True
    except ImportError:
        return False


def create_health_response(additional_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Erstellt standardisierte Health Response."""
    response = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "agents_available": check_agents_integration()
    }

    if additional_data:
        response.update(additional_data)

    return response


def get_agent_system_status() -> dict[str, Any] | None:
    """Holt Agent System Status mit einheitlicher Fehlerbehandlung."""
    if not check_agents_integration():
        return None

    try:
        from agents import get_system_status
        return get_system_status()
    except Exception:
        return None


def create_router(prefix: str, tags: list) -> APIRouter:
    """Erstellt einen neuen Router mit Standard-Konfiguration."""
    return APIRouter(prefix=prefix, tags=tags)
