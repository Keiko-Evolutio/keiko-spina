"""Reporting-Service-Modul für Keiko Personal Assistant.

Enterprise-grade Reporting-System mit:
- Grafana-Integration für Dashboard-Exports
- Automatisierte Report-Generierung und -Verteilung
- Robuste Scheduler-Funktionalität
- Umfassendes Error Handling
- Type-safe APIs

Hauptkomponenten:
- GrafanaClient: API-Client für Grafana-Integrationen
- ReportingScheduler: Automatisierte Report-Generierung
- Konfiguration: Zentrale Konfigurationsverwaltung
- Exceptions: Spezifische Exception-Typen
"""

from __future__ import annotations

# Configuration
from .config import (
    GrafanaConfig,
    ReportConfig,
    ReportingServiceConfig,
    SchedulerConfig,
    get_reporting_config,
    set_reporting_config,
)

# Exceptions
from .exceptions import (
    ConfigurationError,
    GrafanaAuthenticationError,
    GrafanaClientError,
    GrafanaConnectionError,
    GrafanaTimeoutError,
    ReportDistributionError,
    ReportGenerationError,
    ReportingServiceError,
    SchedulerError,
)

# Core Components
from .grafana_client import GrafanaClient
from .scheduler import ReportingScheduler
from .service import ReportingService
from .templates import ReportFormat, ReportTemplate, ReportTemplateManager, ReportType

# Version Information
__version__ = "1.0.0"
__author__ = "Keiko Development Team"
__description__ = "Enterprise-grade Reporting Service für Keiko Personal Assistant"

# Public API
__all__ = [
    "ConfigurationError",
    "GrafanaAuthenticationError",
    # Core Components
    "GrafanaClient",
    "GrafanaClientError",
    # Configuration
    "GrafanaConfig",
    "GrafanaConnectionError",
    "GrafanaTimeoutError",
    "ReportConfig",
    "ReportDistributionError",
    # Templates
    "ReportFormat",
    "ReportGenerationError",
    "ReportTemplate",
    "ReportTemplateManager",
    "ReportType",
    "ReportingScheduler",
    "ReportingService",
    "ReportingServiceConfig",
    # Exceptions
    "ReportingServiceError",
    "SchedulerConfig",
    "SchedulerError",
    "__author__",
    "__description__",
    # Metadata
    "__version__",
    "get_reporting_config",
    "set_reporting_config"
]


def get_service_info() -> dict:
    """Gibt Service-Informationen zurück.

    Returns:
        Dict mit Service-Metadaten
    """
    return {
        "name": "reporting",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components": {
            "grafana_client": "Grafana API Integration",
            "scheduler": "Automatisierte Report-Generierung",
            "config": "Konfigurationsverwaltung",
            "exceptions": "Error Handling"
        }
    }
