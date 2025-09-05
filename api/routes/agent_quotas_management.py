# backend/api/routes/agent_quotas_management.py
"""Agent Quotas Management Endpoints für Keiko Personal Assistant

Implementiert Ressourcen-Limits und Nutzungs-Quotas für Agenten mit
Real-time-Monitoring und automatischer Enforcement.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
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
    QuotaConfiguration,
    QuotaUsageResponse,
    _calculate_usage_percentage,
    _get_current_quota_usage,
    get_agent_or_404,
    validate_agent_access,
)

logger = get_logger(__name__)

# Router erstellen
router = APIRouter(prefix="/api/v1/agents", tags=["Agent Quotas Management"])


class QuotaUpdateRequest(BaseModel):
    """Request-Model für Quota-Updates."""
    max_requests_per_minute: int | None = Field(None, ge=1, description="Max. Requests pro Minute")
    max_requests_per_hour: int | None = Field(None, ge=1, description="Max. Requests pro Stunde")
    max_requests_per_day: int | None = Field(None, ge=1, description="Max. Requests pro Tag")
    max_concurrent_requests: int | None = Field(None, ge=1, description="Max. gleichzeitige Requests")
    max_memory_mb: int | None = Field(None, ge=1, description="Max. Memory in MB")
    max_cpu_cores: float | None = Field(None, ge=0.1, description="Max. CPU-Kerne")
    max_storage_gb: int | None = Field(None, ge=1, description="Max. Storage in GB")
    max_network_mbps: int | None = Field(None, ge=1, description="Max. Netzwerk-Bandbreite in Mbps")


class QuotaViolation(BaseModel):
    """Quota-Verletzung."""
    quota_type: str = Field(..., description="Typ der verletzten Quota")
    current_value: float = Field(..., description="Aktueller Wert")
    limit_value: float = Field(..., description="Limit-Wert")
    violation_percentage: float = Field(..., description="Verletzung in Prozent")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Zeitstempel")
    action_taken: str | None = Field(None, description="Ergriffene Maßnahme")


class QuotaAlert(BaseModel):
    """Quota-Alert."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Alert-ID")
    agent_id: str = Field(..., description="Agent-ID")
    quota_type: str = Field(..., description="Quota-Typ")
    threshold_percentage: float = Field(..., description="Schwellwert in Prozent")
    current_percentage: float = Field(..., description="Aktuelle Nutzung in Prozent")
    severity: str = Field(..., description="Schweregrad")
    message: str = Field(..., description="Alert-Message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Zeitstempel")


def _get_quota_configuration(agent_id: str) -> QuotaConfiguration:
    """Holt Quota-Konfiguration für Agent."""
    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        # Hole Agent-spezifische Quotas
        quotas = policy_engine.get_agent_quotas(agent_id)

        return QuotaConfiguration(**quotas)

    except ImportError:
        logger.warning("Policy Engine nicht verfügbar - verwende Standard-Quotas")
        return QuotaConfiguration(
            max_requests_per_minute=100,
            max_requests_per_hour=1000,
            max_requests_per_day=10000,
            max_concurrent_requests=10,
            max_memory_mb=1024,
            max_cpu_cores=2.0,
            max_storage_gb=10,
            max_network_mbps=100
        )


def _save_quota_configuration(agent_id: str, config: QuotaConfiguration) -> None:
    """Speichert Quota-Konfiguration für Agent."""
    try:
        # Integration mit Policy Engine
        from policy_engine import policy_engine

        policy_engine.set_agent_quotas(agent_id, config.dict())

    except ImportError:
        logger.warning("Policy Engine nicht verfügbar - Quota-Speicherung übersprungen")


def _check_quota_violations(current_usage: dict[str, Any], limits: QuotaConfiguration) -> list[QuotaViolation]:
    """Prüft auf Quota-Verletzungen."""
    violations = []

    # Requests pro Minute
    if limits.max_requests_per_minute and current_usage.get("requests_per_minute", 0) > limits.max_requests_per_minute:
        violations.append(QuotaViolation(
            quota_type="requests_per_minute",
            current_value=current_usage["requests_per_minute"],
            limit_value=limits.max_requests_per_minute,
            violation_percentage=(current_usage["requests_per_minute"] / limits.max_requests_per_minute - 1) * 100
        ))

    # Gleichzeitige Requests
    if limits.max_concurrent_requests and current_usage.get("concurrent_requests", 0) > limits.max_concurrent_requests:
        violations.append(QuotaViolation(
            quota_type="concurrent_requests",
            current_value=current_usage["concurrent_requests"],
            limit_value=limits.max_concurrent_requests,
            violation_percentage=(current_usage["concurrent_requests"] / limits.max_concurrent_requests - 1) * 100
        ))

    # Memory
    if limits.max_memory_mb and current_usage.get("memory_usage_mb", 0) > limits.max_memory_mb:
        violations.append(QuotaViolation(
            quota_type="memory_usage",
            current_value=current_usage["memory_usage_mb"],
            limit_value=limits.max_memory_mb,
            violation_percentage=(current_usage["memory_usage_mb"] / limits.max_memory_mb - 1) * 100
        ))

    # CPU
    if limits.max_cpu_cores and current_usage.get("cpu_usage_cores", 0) > limits.max_cpu_cores:
        violations.append(QuotaViolation(
            quota_type="cpu_usage",
            current_value=current_usage["cpu_usage_cores"],
            limit_value=limits.max_cpu_cores,
            violation_percentage=(current_usage["cpu_usage_cores"] / limits.max_cpu_cores - 1) * 100
        ))

    return violations


def _generate_quota_alerts(agent_id: str, usage_percentages: dict[str, float]) -> list[QuotaAlert]:
    """Generiert Quota-Alerts basierend auf Nutzung."""
    alerts = []

    # Alert-Schwellwerte
    warning_threshold = 80.0
    critical_threshold = 95.0

    for quota_type, percentage in usage_percentages.items():
        if percentage >= critical_threshold:
            alerts.append(QuotaAlert(
                agent_id=agent_id,
                quota_type=quota_type,
                threshold_percentage=critical_threshold,
                current_percentage=percentage,
                severity="critical",
                message=f"Kritische Quota-Nutzung für {quota_type}: {percentage:.1f}%"
            ))
        elif percentage >= warning_threshold:
            alerts.append(QuotaAlert(
                agent_id=agent_id,
                quota_type=quota_type,
                threshold_percentage=warning_threshold,
                current_percentage=percentage,
                severity="warning",
                message=f"Hohe Quota-Nutzung für {quota_type}: {percentage:.1f}%"
            ))

    return alerts


def _calculate_reset_times(limits: QuotaConfiguration) -> dict[str, datetime]:
    """Berechnet Reset-Zeiten für verschiedene Quotas."""
    now = datetime.now(UTC)
    reset_times = {}

    if limits.max_requests_per_minute:
        # Nächste volle Minute
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        reset_times["requests_per_minute"] = next_minute

    if limits.max_requests_per_hour:
        # Nächste volle Stunde
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        reset_times["requests_per_hour"] = next_hour

    if limits.max_requests_per_day:
        # Nächster Tag
        next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        reset_times["requests_per_day"] = next_day

    return reset_times


# Quotas Management Endpoints

@router.get("/{agent_id}/quotas", response_model=QuotaUsageResponse)
@with_log_links(component="quota_management", operation="get_quotas")
async def get_agent_quotas(
    agent_id: str = Path(..., description="Agent-ID"),
    include_violations: bool = Query(default=True, description="Quota-Verletzungen einschließen"),
    include_alerts: bool = Query(default=True, description="Quota-Alerts einschließen")
) -> QuotaUsageResponse:
    """Holt Quota-Konfiguration und aktuelle Nutzung für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        include_violations: Quota-Verletzungen in Response einschließen
        include_alerts: Quota-Alerts in Response einschließen

    Returns:
        Quota-Konfiguration und Nutzungsstatistiken

    Raises:
        ValidationError: Bei ungültiger Agent-ID
        AuthorizationError: Bei fehlenden Berechtigungen
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="read")

    try:
        # Hole Quota-Konfiguration
        config = _get_quota_configuration(agent_id)

        # Hole aktuelle Nutzung
        current_usage = _get_current_quota_usage(agent_id)

        # Berechne Nutzung in Prozent
        usage_percentages = _calculate_usage_percentage(current_usage, config)

        # Prüfe Verletzungen
        violations = []
        if include_violations:
            violations = _check_quota_violations(current_usage, config)

        # Generiere Alerts
        alerts = []
        if include_alerts:
            alerts = _generate_quota_alerts(agent_id, usage_percentages)

        # Berechne Reset-Zeiten
        reset_times = _calculate_reset_times(config)

        # Erstelle Response
        response_data = {
            "configuration": config,
            "current_usage": current_usage,
            "usage_percentage": usage_percentages,
            "limits_exceeded": [v.quota_type for v in violations],
            "reset_times": reset_times
        }

        # Füge zusätzliche Informationen hinzu
        if violations:
            response_data["violations"] = [v.dict() for v in violations]

        if alerts:
            response_data["alerts"] = [a.dict() for a in alerts]

        logger.info(
            f"Quota-Informationen für Agent {agent_id} abgerufen",
            extra={
                "agent_id": agent_id,
                "violations_count": len(violations),
                "alerts_count": len(alerts),
                "correlation_id": f"quota_get_{uuid.uuid4().hex[:8]}"
            }
        )

        return QuotaUsageResponse(**response_data)

    except Exception as e:
        if isinstance(e, LogLinkedError):
            raise

        raise BusinessLogicError(
            message=f"Quota-Abfrage fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="quota_management",
            operation="get_quotas",
            cause=e
        )


@router.put("/{agent_id}/quotas")
@with_log_links(component="quota_management", operation="update_quotas")
async def update_agent_quotas(
    agent_id: str = Path(..., description="Agent-ID"),
    request: QuotaUpdateRequest = ...
) -> dict[str, Any]:
    """Aktualisiert Quota-Konfiguration für Agent.

    Args:
        agent_id: Eindeutige Agent-ID
        request: Neue Quota-Konfiguration

    Returns:
        Bestätigung der Quota-Aktualisierung

    Raises:
        ValidationError: Bei ungültigen Quota-Werten
        AuthorizationError: Bei fehlenden Berechtigungen
        BusinessLogicError: Bei Quota-Update-Fehlern
    """
    # Validiere Agent-Existenz
    await get_agent_or_404(agent_id)

    # Validiere Zugriff
    await validate_agent_access(agent_id, required_permission="admin")

    try:
        # Hole aktuelle Konfiguration
        current_config = _get_quota_configuration(agent_id)

        # Update-Felder anwenden
        update_data = request.dict(exclude_unset=True)

        # Validiere Quota-Konsistenz
        if "max_requests_per_hour" in update_data and "max_requests_per_minute" in update_data:
            if update_data["max_requests_per_hour"] < update_data["max_requests_per_minute"] * 60:
                raise ValidationError(
                    message="max_requests_per_hour muss >= max_requests_per_minute * 60 sein",
                    field="max_requests_per_hour",
                    value=update_data["max_requests_per_hour"]
                )

        if "max_requests_per_day" in update_data and "max_requests_per_hour" in update_data:
            if update_data["max_requests_per_day"] < update_data["max_requests_per_hour"] * 24:
                raise ValidationError(
                    message="max_requests_per_day muss >= max_requests_per_hour * 24 sein",
                    field="max_requests_per_day",
                    value=update_data["max_requests_per_day"]
                )

        # Erstelle neue Konfiguration
        new_config_data = current_config.dict()
        new_config_data.update(update_data)
        new_config = QuotaConfiguration(**new_config_data)

        # Speichere neue Konfiguration
        _save_quota_configuration(agent_id, new_config)

        logger.info(
            f"Quota-Konfiguration für Agent {agent_id} aktualisiert",
            extra={
                "agent_id": agent_id,
                "updated_fields": list(update_data.keys()),
                "correlation_id": f"quota_update_{uuid.uuid4().hex[:8]}"
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
            message=f"Quota-Update fehlgeschlagen: {e!s}",
            agent_id=agent_id,
            component="quota_management",
            operation="update_quotas",
            cause=e
        )
