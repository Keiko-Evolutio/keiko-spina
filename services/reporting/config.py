"""Konfiguration für das Reporting-Service-Modul.

Zentrale Konfiguration für alle Reporting-Parameter und -Einstellungen.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GrafanaConfig(BaseModel):
    """Konfiguration für Grafana-Client."""

    # Timeout-Einstellungen
    panel_timeout_seconds: float = Field(default=30.0, description="Timeout für Panel-Exports in Sekunden")
    dashboard_timeout_seconds: float = Field(default=60.0, description="Timeout für Dashboard-Exports in Sekunden")

    # Standard-Export-Parameter
    default_panel_width: int = Field(default=1600, description="Standard-Breite für Panel-Exports")
    default_panel_height: int = Field(default=900, description="Standard-Höhe für Panel-Exports")

    # Retry-Konfiguration
    max_retries: int = Field(default=3, description="Maximale Anzahl von Retry-Versuchen")
    retry_delay_seconds: float = Field(default=1.0, description="Verzögerung zwischen Retry-Versuchen")


class SchedulerConfig(BaseModel):
    """Konfiguration für Reporting-Scheduler."""

    # Timing-Einstellungen
    default_interval_minutes: int = Field(default=60, description="Standard-Intervall in Minuten")
    seconds_per_minute: int = Field(default=60, description="Sekunden pro Minute für Konvertierung")

    # Task-Management
    graceful_shutdown_timeout: float = Field(default=5.0, description="Timeout für graceful Shutdown")


class ReportConfig(BaseModel):
    """Konfiguration für Report-Generierung."""

    # Standard-Dashboard-Einstellungen
    default_dashboard_uid: str = Field(default="keiko-overview", description="Standard-Dashboard-UID")
    default_panel_id: int = Field(default=1, description="Standard-Panel-ID")

    # E-Mail-Einstellungen
    default_subject: str = Field(default="Keiko KPI Report", description="Standard-E-Mail-Betreff")
    default_body: str = Field(
        default="Automatischer KPI-Report. Siehe Grafana-Dashboard: Keiko Overview.",
        description="Standard-E-Mail-Text"
    )
    default_severity: str = Field(default="info", description="Standard-Severity für E-Mails")

    # Export-Parameter
    default_export_params: dict[str, Any] = Field(
        default_factory=lambda: {"width": 1600, "height": 900},
        description="Standard-Parameter für Exports"
    )


class ReportingServiceConfig(BaseModel):
    """Hauptkonfiguration für das Reporting-Service."""

    grafana: GrafanaConfig = Field(default_factory=GrafanaConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # Service-Einstellungen
    service_name: str = Field(default="reporting", description="Name des Services")
    version: str = Field(default="1.0.0", description="Version des Services")

    # Logging-Konfiguration
    log_level: str = Field(default="INFO", description="Log-Level für das Service")
    enable_debug_logging: bool = Field(default=False, description="Debug-Logging aktivieren")

    class Config:
        """Pydantic-Konfiguration."""
        extra = "forbid"  # Keine zusätzlichen Felder erlaubt
        validate_assignment = True  # Validierung bei Zuweisungen


# Globale Konfigurationsinstanz
_config: ReportingServiceConfig | None = None


def get_reporting_config() -> ReportingServiceConfig:
    """Gibt die globale Reporting-Konfiguration zurück.

    Returns:
        ReportingServiceConfig: Konfigurationsinstanz
    """
    global _config
    if _config is None:
        _config = ReportingServiceConfig()
    return _config


def set_reporting_config(config: ReportingServiceConfig) -> None:
    """Setzt eine neue globale Reporting-Konfiguration.

    Args:
        config: Neue Konfigurationsinstanz
    """
    global _config
    _config = config


__all__ = [
    "GrafanaConfig",
    "ReportConfig",
    "ReportingServiceConfig",
    "SchedulerConfig",
    "get_reporting_config",
    "set_reporting_config"
]
