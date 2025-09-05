"""Konsolidierte Health-Check-Utilities für die Keiko-API.

Eliminiert Code-Duplikation durch zentrale Health-Check-Funktionalität
für alle API-Module. Implementiert einheitliche Health-Response-Patterns.
"""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from typing import Any, Protocol, TypedDict

import psutil

from kei_logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "HealthStatus",
    "ServiceNames",
    # Type Definitions
    "ComponentHealthDict",
    "HealthResponseDict",
    "HealthChecker",
    # Core Functions
    "create_base_health_response",
    "create_component_health",
    "get_system_metrics",
    "get_python_info",
    "get_system_status",
    "get_orchestrator_tools",
    # Agent System
    "check_agent_system_health",
    # Health Checkers
    "ChatSystemHealthChecker",
    "FunctionSystemHealthChecker",
    "AgentsSystemHealthChecker",
    # Convenience Functions
    "create_detailed_health_response"
]


# ============================================================================
# CONSTANTS
# ============================================================================

class HealthStatus:
    """Health-Status-Konstanten."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"


class ServiceNames:
    """Service-Namen-Konstanten."""
    KEIKO_API = "keiko-api"
    AGENT_SYSTEM = "agent_system"
    CHAT_SYSTEM = "chat_system"
    FUNCTION_SYSTEM = "function_system"
    CAPABILITIES_SYSTEM = "capabilities_system"


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class ComponentHealthDict(TypedDict):
    """Type-Definition für Component-Health-Daten."""
    name: str
    status: str
    message: str
    details: dict[str, Any] | None


class HealthResponseDict(TypedDict):
    """Type-Definition für Health-Response-Daten."""
    status: str
    timestamp: str
    service: str
    version: str


class HealthChecker(Protocol):
    """Protocol für Health-Check-Implementierungen."""

    def check_health(self) -> ComponentHealthDict:
        """Führt Health-Check durch und gibt Ergebnis zurück."""
        ...


# ============================================================================
# CORE HEALTH UTILITIES
# ============================================================================

def create_base_health_response(
    service_name: str,
    version: str = "1.0.0",
    additional_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt standardisierte Health-Response.

    Konsolidiert Health-Response-Erstellung für alle API-Module.

    Args:
        service_name: Name des Services
        version: Service-Version
        additional_data: Zusätzliche Daten für Response

    Returns:
        Standardisierte Health-Response
    """
    response: dict[str, Any] = {
        "status": HealthStatus.HEALTHY,
        "timestamp": datetime.now(UTC).isoformat(),
        "service": service_name,
        "version": version,
        "agents_available": _check_agents_integration()
    }

    if additional_data:
        response.update(additional_data)

    return response


def create_component_health(
    name: str,
    status: str,
    message: str,
    details: dict[str, Any] | None = None
) -> ComponentHealthDict:
    """Erstellt Component-Health-Objekt.

    Args:
        name: Component-Name
        status: Health-Status
        message: Status-Nachricht
        details: Zusätzliche Details

    Returns:
        Component-Health-Daten
    """
    return ComponentHealthDict(
        name=name,
        status=status,
        message=message,
        details=details
    )


def get_system_metrics() -> dict[str, Any]:
    """Sammelt System-Metriken.

    Konsolidiert System-Metriken-Sammlung für alle Health-Checks.

    Returns:
        System-Metriken-Daten
    """
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
        }
    except (OSError, AttributeError, ValueError) as e:
        logger.warning(f"Fehler beim Sammeln von System-Metriken: {e}")
        return {"error": str(e)}


def get_python_info() -> dict[str, str]:
    """Sammelt Python-Umgebungsinformationen.

    Returns:
        Python-Umgebungsdaten
    """
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.architecture()[0]
    }


# ============================================================================
# AGENT SYSTEM INTEGRATION
# ============================================================================

def _check_agents_integration() -> bool:
    """Prüft Agent-System-Verfügbarkeit.

    Returns:
        True wenn Agent-System verfügbar
    """
    try:
        from agents import get_system_status
        return True
    except ImportError:
        return False


def check_agent_system_health() -> ComponentHealthDict:
    """Prüft Agent-System-Health.

    Returns:
        Agent-System-Health-Status
    """
    if not _check_agents_integration():
        return create_component_health(
            name=ServiceNames.AGENT_SYSTEM,
            status=HealthStatus.UNAVAILABLE,
            message="Agent System nicht verfügbar"
        )

    try:
        import agents
        # Verwende agents Modul direkt statt kei_agents
        if hasattr(agents, "get_system_status"):
            status = agents.get_system_status()
        else:
            status = {"status": "available", "modules": ["agents"]}
        return create_component_health(
            name=ServiceNames.AGENT_SYSTEM,
            status=HealthStatus.HEALTHY,
            message="Agent System funktionsfähig",
            details=status
        )
    except (ImportError, AttributeError, RuntimeError) as e:
        return create_component_health(
            name=ServiceNames.AGENT_SYSTEM,
            status=HealthStatus.DEGRADED,
            message=f"Agent System Fehler: {e!s}"
        )


# ============================================================================
# SPECIALIZED HEALTH CHECKERS
# ============================================================================

class ChatSystemHealthChecker:
    """Health-Checker für Chat-System."""

    def __init__(self, chat_sessions: dict[str, Any], user_sessions: dict[str, Any]):
        """Initialisiert Chat-Health-Checker.

        Args:
            chat_sessions: Chat-Sessions-Dictionary
            user_sessions: User-Sessions-Dictionary
        """
        self.chat_sessions = chat_sessions
        self.user_sessions = user_sessions

    def check_health(self) -> dict[str, Any]:
        """Führt Chat-System-Health-Check durch.

        Returns:
            Chat-System-Health-Daten
        """
        return {
            "chat_sessions": {
                "total_sessions": len(self.chat_sessions),
                "active_users": len(self.user_sessions),
                "total_messages": sum(len(s.messages) for s in self.chat_sessions.values())
            }
        }


class FunctionSystemHealthChecker:
    """Health-Checker für Function-System."""

    def __init__(
        self,
        built_in_functions: dict[str, Any],
        registered_tools: dict[str, Any],
        execution_history: dict[str, Any]
    ):
        """Initialisiert Function-Health-Checker.

        Args:
            built_in_functions: Built-in-Functions-Dictionary
            registered_tools: Registered-Tools-Dictionary
            execution_history: Execution-History-Dictionary
        """
        self.built_in_functions = built_in_functions
        self.registered_tools = registered_tools
        self.execution_history = execution_history

    def check_health(self) -> dict[str, Any]:
        """Führt Function-System-Health-Check durch.

        Returns:
            Function-System-Health-Daten
        """
        return {
            "functions": {
                "built_in_functions": len(self.built_in_functions),
                "registered_tools": len(self.registered_tools),
                "total_executions": len(self.execution_history),
                "orchestrator_tools": FunctionSystemHealthChecker._get_orchestrator_tools_count()
            },
            "recent_executions": {
                "total": len(self.execution_history),
                "success": len([e for e in self.execution_history.values() if e.status == "success"]),
                "errors": len([e for e in self.execution_history.values() if e.status == "error"])
            }
        }

    @staticmethod
    def _get_orchestrator_tools_count() -> int:
        """Ermittelt Anzahl der Orchestrator-Tools.

        Returns:
            Anzahl der verfügbaren Orchestrator-Tools
        """
        try:
            import agents
            # Prüfe verschiedene mögliche Funktionen
            if hasattr(agents, "get_orchestrator_tools"):
                tools_map = agents.get_orchestrator_tools()
                return len(tools_map)
            if hasattr(agents, "get_available_tools"):
                tools_map = agents.get_available_tools()
                return len(tools_map)
            return 0
        except (ImportError, AttributeError):
            return 0


class AgentsSystemHealthChecker:
    """Health-Checker für Agents-System."""

    def __init__(self, agents: list[Any]):
        """Initialisiert Agents-Health-Checker.

        Args:
            agents: Liste der verfügbaren Agents
        """
        self.agents = agents

    def check_health(self) -> dict[str, Any]:
        """Führt Agents-System-Health-Check durch.

        Returns:
            Agents-System-Health-Daten
        """
        return {
            "agents": {
                "total_agents": len(self.agents),
                "active_agents": len([a for a in self.agents if a.status == "active"]),
                "available_capabilities": list({
                    cap for agent in self.agents for cap in agent.capabilities
                })
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_detailed_health_response(
    service_name: str,
    version: str = "1.0.0",
    include_system_metrics: bool = True,
    include_python_info: bool = True,
    additional_checkers: list[HealthChecker] | None = None
) -> dict[str, Any]:
    """Erstellt detaillierte Health-Response mit System-Metriken.

    Args:
        service_name: Name des Services
        version: Service-Version
        include_system_metrics: Ob System-Metriken eingeschlossen werden sollen
        include_python_info: Ob Python-Info eingeschlossen werden soll
        additional_checkers: Zusätzliche Health-Checker

    Returns:
        Detaillierte Health-Response
    """
    response = create_base_health_response(service_name, version)

    if include_system_metrics:
        response["system_metrics"] = get_system_metrics()

    if include_python_info:
        response["python_info"] = get_python_info()

    # Agent-System-Health hinzufügen
    agent_health = check_agent_system_health()
    response["agent_system_status"] = agent_health

    # Zusätzliche Health-Checker ausführen
    if additional_checkers:
        for checker in additional_checkers:
            try:
                checker_result = checker.check_health()
                response.update(checker_result)
            except (AttributeError, RuntimeError, ValueError) as e:
                logger.warning(f"Health-Checker-Fehler: {e}")

    return response


# ============================================================================
# MISSING FUNCTIONS FOR __all__ COMPATIBILITY
# ============================================================================

def get_system_status() -> dict[str, Any]:
    """Gibt System-Status zurück.

    Returns:
        System-Status-Informationen
    """
    try:
        import agents
        if hasattr(agents, "get_system_status"):
            return agents.get_system_status()
        return {
            "status": "available",
            "modules": ["agents"],
            "timestamp": datetime.now(UTC).isoformat()
        }
    except ImportError:
        return {
            "status": "unavailable",
            "error": "agents module not found",
            "timestamp": datetime.now(UTC).isoformat()
        }


def get_orchestrator_tools() -> dict[str, Any]:
    """Gibt verfügbare Orchestrator-Tools zurück.

    Returns:
        Dictionary mit verfügbaren Tools
    """
    try:
        import agents
        if hasattr(agents, "get_orchestrator_tools"):
            return agents.get_orchestrator_tools()
        if hasattr(agents, "get_available_tools"):
            return agents.get_available_tools()
        return {}
    except ImportError:
        return {}
