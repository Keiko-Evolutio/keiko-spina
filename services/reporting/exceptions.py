"""Exception-Klassen für das Reporting-Service-Modul.

Spezifische Exception-Typen für bessere Fehlerbehandlung und Debugging.
"""

from __future__ import annotations

from typing import Any


class ReportingServiceError(Exception):
    """Basis-Exception für alle Reporting-Service-Fehler."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialisiert die Exception.

        Args:
            message: Fehlermeldung
            details: Zusätzliche Fehlerdetails
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class GrafanaClientError(ReportingServiceError):
    """Exception für Grafana-Client-Fehler."""

    def __init__(self, message: str, status_code: int | None = None,
                 response_body: str | None = None) -> None:
        """Initialisiert die Grafana-Client-Exception.

        Args:
            message: Fehlermeldung
            status_code: HTTP-Status-Code
            response_body: Response-Body der fehlgeschlagenen Anfrage
        """
        details = {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_body is not None:
            details["response_body"] = response_body

        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class GrafanaConnectionError(GrafanaClientError):
    """Exception für Grafana-Verbindungsfehler."""


class GrafanaAuthenticationError(GrafanaClientError):
    """Exception für Grafana-Authentifizierungsfehler."""


class GrafanaTimeoutError(GrafanaClientError):
    """Exception für Grafana-Timeout-Fehler."""


class ReportGenerationError(ReportingServiceError):
    """Exception für Report-Generierungsfehler."""

    def __init__(self, message: str, report_type: str | None = None,
                 dashboard_uid: str | None = None, panel_id: int | None = None) -> None:
        """Initialisiert die Report-Generierungs-Exception.

        Args:
            message: Fehlermeldung
            report_type: Typ des Reports
            dashboard_uid: Dashboard-UID
            panel_id: Panel-ID
        """
        details = {}
        if report_type is not None:
            details["report_type"] = report_type
        if dashboard_uid is not None:
            details["dashboard_uid"] = dashboard_uid
        if panel_id is not None:
            details["panel_id"] = panel_id

        super().__init__(message, details)
        self.report_type = report_type
        self.dashboard_uid = dashboard_uid
        self.panel_id = panel_id


class ReportDistributionError(ReportingServiceError):
    """Exception für Report-Verteilungsfehler."""

    def __init__(self, message: str, recipients: list | None = None,
                 distribution_method: str | None = None) -> None:
        """Initialisiert die Report-Verteilungs-Exception.

        Args:
            message: Fehlermeldung
            recipients: Liste der Empfänger
            distribution_method: Verteilungsmethode (z.B. "email", "webhook")
        """
        details = {}
        if recipients is not None:
            details["recipients"] = recipients
        if distribution_method is not None:
            details["distribution_method"] = distribution_method

        super().__init__(message, details)
        self.recipients = recipients
        self.distribution_method = distribution_method


class SchedulerError(ReportingServiceError):
    """Exception für Scheduler-Fehler."""

    def __init__(self, message: str, scheduler_state: str | None = None) -> None:
        """Initialisiert die Scheduler-Exception.

        Args:
            message: Fehlermeldung
            scheduler_state: Aktueller Zustand des Schedulers
        """
        details = {}
        if scheduler_state is not None:
            details["scheduler_state"] = scheduler_state

        super().__init__(message, details)
        self.scheduler_state = scheduler_state


class ConfigurationError(ReportingServiceError):
    """Exception für Konfigurationsfehler."""

    def __init__(self, message: str, config_key: str | None = None,
                 config_value: Any | None = None) -> None:
        """Initialisiert die Konfigurations-Exception.

        Args:
            message: Fehlermeldung
            config_key: Konfigurationsschlüssel
            config_value: Konfigurationswert
        """
        details = {}
        if config_key is not None:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)

        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


__all__ = [
    "ConfigurationError",
    "GrafanaAuthenticationError",
    "GrafanaClientError",
    "GrafanaConnectionError",
    "GrafanaTimeoutError",
    "ReportDistributionError",
    "ReportGenerationError",
    "ReportingServiceError",
    "SchedulerError"
]
