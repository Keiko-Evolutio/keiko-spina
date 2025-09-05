# backend/agents/slo_sla/exceptions.py
"""SLO/SLA System Exception-Klassen.

Definiert spezifische Exception-Klassen für das SLO/SLA System.
"""

from typing import Any


class SLOSLAError(Exception):
    """Basis-Exception für alle SLO/SLA-Fehler."""

    def __init__(self, message: str, error_code: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


# SLO Monitoring Exceptions
class SLOMonitoringError(SLOSLAError):
    """Exception für SLO Monitoring-Fehler."""


class SLODefinitionError(SLOMonitoringError):
    """Exception für ungültige SLO-Definitionen."""


class ErrorBudgetExhaustedError(SLOMonitoringError):
    """Exception wenn Error Budget erschöpft ist."""


# Breach Management Exceptions
class BreachDetectionError(SLOSLAError):
    """Exception für Breach Detection-Fehler."""


class RemediationFailedError(SLOSLAError):
    """Exception wenn Remediation fehlschlägt."""


class ComplianceViolationError(SLOSLAError):
    """Exception für Compliance-Verletzungen."""


# Alerting Exceptions
class AlertingError(SLOSLAError):
    """Exception für Alerting-Fehler."""


class AlertChannelError(AlertingError):
    """Exception für Alert Channel-Fehler."""


class EscalationError(AlertingError):
    """Exception für Escalation-Fehler."""


# Reporting Exceptions
class ReportGenerationError(SLOSLAError):
    """Exception für Report Generation-Fehler."""


class ReportExportError(ReportGenerationError):
    """Exception für Report Export-Fehler."""


class TemplateError(ReportGenerationError):
    """Exception für Template-Fehler."""


# Utility Functions
def create_slo_sla_error(error_type: str, message: str,
                        error_code: str | None = None,
                        details: dict[str, Any] | None = None) -> SLOSLAError:
    """Factory-Funktion für SLO/SLA-Exceptions.

    Args:
        error_type: Typ der Exception
        message: Fehlermeldung
        error_code: Optional error code
        details: Optional zusätzliche Details

    Returns:
        Entsprechende SLO/SLA-Exception
    """
    error_map = {
        "slo_monitoring": SLOMonitoringError,
        "slo_definition": SLODefinitionError,
        "error_budget_exhausted": ErrorBudgetExhaustedError,
        "breach_detection": BreachDetectionError,
        "remediation_failed": RemediationFailedError,
        "compliance_violation": ComplianceViolationError,
        "alerting": AlertingError,
        "alert_channel": AlertChannelError,
        "escalation": EscalationError,
        "report_generation": ReportGenerationError,
        "report_export": ReportExportError,
        "template": TemplateError
    }

    exception_class = error_map.get(error_type, SLOSLAError)
    return exception_class(message, error_code, details)


def is_slo_sla_error(exception: Exception) -> bool:
    """Prüft ob Exception eine SLO/SLA-Exception ist.

    Args:
        exception: Zu prüfende Exception

    Returns:
        True wenn SLO/SLA-Exception, sonst False
    """
    return isinstance(exception, SLOSLAError)


def get_error_details(exception: Exception) -> dict[str, Any]:
    """Extrahiert Details aus SLO/SLA-Exception.

    Args:
        exception: SLO/SLA-Exception

    Returns:
        Dict mit Exception-Details
    """
    if isinstance(exception, SLOSLAError):
        return {
            "type": type(exception).__name__,
            "message": str(exception),
            "error_code": getattr(exception, "error_code", None),
            "details": getattr(exception, "details", {})
        }
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "error_code": None,
        "details": {}
    }
