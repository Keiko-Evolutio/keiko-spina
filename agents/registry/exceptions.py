# backend/kei_agents/registry/exceptions.py
"""Registry System Exception-Klassen.

Definiert spezifische Exception-Klassen für das KEI-Agent Registry System.
"""

from typing import Any


class RegistryError(Exception):
    """Basis-Exception für alle Registry-Fehler."""

    def __init__(self, message: str, error_code: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


# Dynamic Registry Exceptions
class RegistrationError(RegistryError):
    """Exception für Agent-Registrierungsfehler."""


class AgentNotFoundError(RegistryError):
    """Exception wenn Agent nicht gefunden wird."""


class DuplicateRegistrationError(RegistrationError):
    """Exception für doppelte Registrierungen."""


class RegistryUnavailableError(RegistryError):
    """Exception wenn Registry nicht verfügbar ist."""


# Service Discovery Exceptions
class ServiceNotFoundError(RegistryError):
    """Exception wenn Service nicht gefunden wird."""


class DiscoveryTimeoutError(RegistryError):
    """Exception für Discovery-Timeouts."""


class DiscoveryError(RegistryError):
    """Basis-Exception für Service Discovery-Fehler."""


# Load Balancing Exceptions
class NoHealthyInstancesError(RegistryError):
    """Exception wenn keine gesunden Instances verfügbar sind."""


class LoadBalancingError(RegistryError):
    """Exception für Load Balancing-Fehler."""


class InstanceOverloadedError(LoadBalancingError):
    """Exception wenn Instance überlastet ist."""


# Health Integration Exceptions
class HealthIntegrationError(RegistryError):
    """Exception für Health Integration-Fehler."""


class ServiceUnhealthyError(HealthIntegrationError):
    """Exception für ungesunde Services."""


class RecoveryFailedError(HealthIntegrationError):
    """Exception wenn Recovery fehlschlägt."""


# Utility Functions
def create_registry_error(error_type: str, message: str,
                         error_code: str | None = None,
                         details: dict[str, Any] | None = None) -> RegistryError:
    """Factory-Funktion für Registry-Exceptions.
    
    Args:
        error_type: Typ der Exception
        message: Fehlermeldung
        error_code: Optional error code
        details: Optional zusätzliche Details
        
    Returns:
        Entsprechende Registry-Exception
    """
    error_map = {
        "registration": RegistrationError,
        "agent_not_found": AgentNotFoundError,
        "duplicate_registration": DuplicateRegistrationError,
        "registry_unavailable": RegistryUnavailableError,
        "service_not_found": ServiceNotFoundError,
        "discovery_timeout": DiscoveryTimeoutError,
        "discovery": DiscoveryError,
        "no_healthy_instances": NoHealthyInstancesError,
        "load_balancing": LoadBalancingError,
        "instance_overloaded": InstanceOverloadedError,
        "health_integration": HealthIntegrationError,
        "service_unhealthy": ServiceUnhealthyError,
        "recovery_failed": RecoveryFailedError
    }

    exception_class = error_map.get(error_type, RegistryError)
    return exception_class(message, error_code, details)


def is_registry_error(exception: Exception) -> bool:
    """Prüft ob Exception eine Registry-Exception ist.
    
    Args:
        exception: Zu prüfende Exception
        
    Returns:
        True wenn Registry-Exception, sonst False
    """
    return isinstance(exception, RegistryError)


def get_error_details(exception: Exception) -> dict[str, Any]:
    """Extrahiert Details aus Registry-Exception.
    
    Args:
        exception: Registry-Exception
        
    Returns:
        Dict mit Exception-Details
    """
    if isinstance(exception, RegistryError):
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
