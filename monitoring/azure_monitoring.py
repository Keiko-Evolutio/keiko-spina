# backend/monitoring/azure_monitoring.py
"""Integration mit Azure Monitor (vereinfacht).

Ermöglicht das optionale Erfassen von Metriken, Events und Exceptions
über Application Insights, sofern konfiguriert und aktiviert.
"""

import os
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AzureMonitoringConfig:
    """Azure Monitor Konfiguration."""
    connection_string: str | None = None
    environment: str = "production"
    application_name: str = "keiko-backend"
    enabled: bool = True

    @classmethod
    def from_environment(cls) -> "AzureMonitoringConfig":
        """Erstellt Config aus Umgebungsvariablen."""
        return cls(
            connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
            environment=os.getenv("ENVIRONMENT", "production"),
            application_name=os.getenv("APPLICATION_NAME", "keiko-backend"),
            enabled=os.getenv("AZURE_MONITORING_ENABLED", "true").lower() == "true"
        )


# ============================================================================
# AZURE MONITORING INTEGRATION - VEREINFACHT
# ============================================================================

class AzureMonitoringIntegration:
    """Azure Monitor Integration."""

    def __init__(self, config: AzureMonitoringConfig):
        self.config = config
        self.is_enabled = False
        self._client = None

    async def start(self) -> bool:
        """Startet Azure Monitor Integration."""
        if not self.config.enabled or not self.config.connection_string:
            logger.info("Azure Monitor deaktiviert oder nicht konfiguriert")
            return False

        try:
            # Initialisierung
            self.is_enabled = True
            logger.info("Azure Monitor Integration gestartet")
            return True

        except Exception as e:
            logger.exception(f"Azure Monitor Fehler: {e}")
            return False

    async def stop(self) -> None:
        """Stoppt Azure Monitor Integration."""
        self.is_enabled = False
        self._client = None
        logger.info("Azure Monitor Integration gestoppt")

    def record_metric(self, name: str, value: float, properties: dict[str, Any] | None = None) -> None:
        """Zeichnet Metrik in Azure Monitor auf."""
        if not self.is_enabled:
            return

        # Metrik-Aufzeichnung
        logger.debug(f"Azure Metrik: {name}={value}")

    def record_event(self, name: str, properties: dict[str, Any] | None = None) -> None:
        """Zeichnet Event in Azure Monitor auf."""
        if not self.is_enabled:
            return

        logger.debug(f"Azure Event: {name}")

    def record_exception(self, exception: Exception, properties: dict[str, Any] | None = None) -> None:
        """Zeichnet Exception in Azure Monitor auf."""
        if not self.is_enabled:
            return

        logger.error(f"Azure Exception: {exception}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Globale Instanz
_azure_monitoring: AzureMonitoringIntegration | None = None


async def initialize_azure_monitoring(config: AzureMonitoringConfig | None = None) -> AzureMonitoringIntegration | None:
    """Initialisiert Azure Monitor Integration."""
    global _azure_monitoring

    if config is None:
        config = AzureMonitoringConfig.from_environment()

    _azure_monitoring = AzureMonitoringIntegration(config)

    if await _azure_monitoring.start():
        return _azure_monitoring
    return None


async def shutdown_azure_monitoring() -> None:
    """Beendet Azure Monitor Integration."""
    global _azure_monitoring
    if _azure_monitoring:
        await _azure_monitoring.stop()
        _azure_monitoring = None


def record_azure_metric(name: str, value: float, properties: dict[str, Any] | None = None) -> None:
    """Zeichnet Azure Metrik auf."""
    if _azure_monitoring:
        _azure_monitoring.record_metric(name, value, properties)


def record_azure_business_event(name: str, properties: dict[str, Any] | None = None) -> None:
    """Zeichnet Azure Business Event auf."""
    if _azure_monitoring:
        _azure_monitoring.record_event(name, properties)


def record_azure_error(exception: Exception, properties: dict[str, Any] | None = None) -> None:
    """Zeichnet Azure Fehler auf."""
    if _azure_monitoring:
        _azure_monitoring.record_exception(exception, properties)


def get_azure_metrics() -> dict[str, Any]:
    """Gibt Azure Monitor Status zurück."""
    if _azure_monitoring:
        return {
            "azure_monitor_enabled": _azure_monitoring.is_enabled,
            "status": "active" if _azure_monitoring.is_enabled else "inactive",
            "config": {
                "environment": _azure_monitoring.config.environment,
                "application": _azure_monitoring.config.application_name,
                "connection_configured": bool(_azure_monitoring.config.connection_string)
            }
        }
    return {
        "azure_monitor_enabled": False,
        "status": "not_initialized"
    }


async def get_azure_monitoring_status() -> dict[str, Any]:
    """Gibt detaillierten Azure Monitor Status zurück."""
    return get_azure_metrics()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AzureMonitoringConfig",
    "AzureMonitoringIntegration",
    "get_azure_metrics",
    "get_azure_monitoring_status",
    "initialize_azure_monitoring",
    "record_azure_business_event",
    "record_azure_error",
    "record_azure_metric",
    "shutdown_azure_monitoring",
]
