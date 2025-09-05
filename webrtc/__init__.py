"""WebRTC Integration für Voice-Service-System

Dieses Modul implementiert WebRTC-Unterstützung für das Voice-Service-System
mit Signaling Server, Session Management und Performance Monitoring.

Features:
- WebRTC Signaling Server für Offer/Answer/ICE-Austausch
- Session Management für P2P-Verbindungen
- Performance Monitoring und Metrics
- Integration mit bestehender Voice-Architektur
- Fallback-Mechanismus zu WebSocket

@version 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

# =============================================================================
# Modul-Metadaten
# =============================================================================

__version__ = "1.0.0"
__author__ = "Keiko Development Team"

# =============================================================================
# Komponenten-Importe
# =============================================================================

try:
    from .config import WebRTCConfig, create_webrtc_config, get_webrtc_config
    from .monitoring import WebRTCMonitor, create_webrtc_monitor
    from .session_manager import WebRTCSessionManager, create_session_manager
    from .signaling_server import WebRTCSignalingServer, create_signaling_server
    from .types import (
        SignalingMessage,
        WebRTCConfiguration,
        WebRTCMetrics,
        WebRTCSession,
        WebRTCSessionState,
    )

    _WEBRTC_AVAILABLE = True
    logger.info("WebRTC-Komponenten erfolgreich geladen")

except ImportError as e:
    logger.warning(f"WebRTC-Komponenten nicht verfügbar: {e}")

    # Fallback-Implementierungen
    WebRTCSignalingServer = None  # type: ignore[assignment]
    WebRTCSessionManager = None  # type: ignore[assignment]
    WebRTCMonitor = None  # type: ignore[assignment]

    # Fallback Factory-Funktionen
    create_signaling_server = None  # type: ignore[assignment]
    create_session_manager = None  # type: ignore[assignment]
    create_webrtc_monitor = None  # type: ignore[assignment]

    # Fallback Types
    SignalingMessage = None  # type: ignore[assignment]
    WebRTCSession = None  # type: ignore[assignment]
    WebRTCSessionState = None  # type: ignore[assignment]
    WebRTCConfiguration = None  # type: ignore[assignment]
    WebRTCMetrics = None  # type: ignore[assignment]

    # Fallback Config
    WebRTCConfig = None  # type: ignore[assignment]
    get_webrtc_config = None  # type: ignore[assignment]
    create_webrtc_config = None  # type: ignore[assignment]

    _WEBRTC_AVAILABLE = False

# =============================================================================
# Service Registry
# =============================================================================

_webrtc_services: dict[str, Any] = {}

def register_webrtc_service(name: str, service: Any) -> None:
    """Registriert einen WebRTC-Service."""
    _webrtc_services[name] = service
    logger.debug(f"WebRTC-Service registriert: {name}")

def get_webrtc_service(name: str) -> Any:
    """Gibt einen registrierten WebRTC-Service zurück."""
    return _webrtc_services.get(name)

def unregister_webrtc_service(name: str) -> None:
    """Entfernt einen WebRTC-Service aus der Registry."""
    if name in _webrtc_services:
        del _webrtc_services[name]
        logger.debug(f"WebRTC-Service entfernt: {name}")

# =============================================================================
# Factory Functions
# =============================================================================

def create_webrtc_signaling_server(
    config: WebRTCConfig | None = None
) -> WebRTCSignalingServer | None:
    """Erstellt einen WebRTC Signaling Server."""
    if not _WEBRTC_AVAILABLE or not WebRTCSignalingServer:
        logger.warning("WebRTC Signaling Server nicht verfügbar")
        return None

    try:
        server_config = config or get_webrtc_config()
        server = create_signaling_server(server_config)
        register_webrtc_service("signaling_server", server)
        return server
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des WebRTC Signaling Servers: {e}")
        return None

def create_webrtc_session_manager(
    config: WebRTCConfig | None = None
) -> WebRTCSessionManager | None:
    """Erstellt einen WebRTC Session Manager."""
    if not _WEBRTC_AVAILABLE or not WebRTCSessionManager:
        logger.warning("WebRTC Session Manager nicht verfügbar")
        return None

    try:
        manager_config = config or get_webrtc_config()
        manager = create_session_manager(manager_config)
        register_webrtc_service("session_manager", manager)
        return manager
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des WebRTC Session Managers: {e}")
        return None

def create_webrtc_monitor_instance(
    config: WebRTCConfig | None = None
) -> WebRTCMonitor | None:
    """Erstellt einen WebRTC Monitor (interne Implementierung)."""
    if not _WEBRTC_AVAILABLE or not WebRTCMonitor:
        logger.warning("WebRTC Monitor nicht verfügbar")
        return None

    try:
        from .monitoring import create_webrtc_monitor as create_monitor_impl

        monitor_config = config or get_webrtc_config()
        if not monitor_config:
            logger.error("Keine gültige WebRTC-Konfiguration verfügbar")
            return None

        monitor = create_monitor_impl(monitor_config)
        if monitor:
            register_webrtc_service("monitor", monitor)
            logger.debug("WebRTC Monitor erfolgreich erstellt")
        return monitor
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des WebRTC Monitors: {e}")
        return None


def create_webrtc_monitor_with_service_registration(
    config: WebRTCConfig | None = None
) -> WebRTCMonitor | None:
    """Erstellt einen WebRTC Monitor mit Service-Registrierung (öffentliche API)."""
    return create_webrtc_monitor_instance(config)

# =============================================================================
# System Status und Health Checks
# =============================================================================

def get_webrtc_system_status() -> dict[str, Any]:
    """Gibt den WebRTC-System-Status zurück."""
    return {
        "webrtc_available": _WEBRTC_AVAILABLE,
        "version": __version__,
        "services": {
            "signaling_server": get_webrtc_service("signaling_server") is not None,
            "session_manager": get_webrtc_service("session_manager") is not None,
            "monitor": get_webrtc_service("monitor") is not None
        },
        "registered_services": list(_webrtc_services.keys()),
        "components": {
            "signaling": _WEBRTC_AVAILABLE,
            "session_management": _WEBRTC_AVAILABLE,
            "monitoring": _WEBRTC_AVAILABLE,
            "configuration": _WEBRTC_AVAILABLE
        }
    }

def is_webrtc_system_healthy() -> bool:
    """Prüft, ob das WebRTC-System funktionsfähig ist."""
    if not _WEBRTC_AVAILABLE:
        return False

    # Prüfe kritische Services
    signaling_server = get_webrtc_service("signaling_server")
    session_manager = get_webrtc_service("session_manager")

    return signaling_server is not None and session_manager is not None

async def perform_webrtc_health_check() -> dict[str, Any]:
    """Führt einen umfassenden WebRTC-Health-Check durch."""
    health_status = {
        "healthy": False,
        "timestamp": None,
        "checks": {}
    }

    try:
        from datetime import UTC, datetime
        health_status["timestamp"] = datetime.now(UTC).isoformat()

        # System Availability Check
        health_status["checks"]["system_available"] = _WEBRTC_AVAILABLE

        # Service Checks
        signaling_server = get_webrtc_service("signaling_server")
        session_manager = get_webrtc_service("session_manager")
        monitor = get_webrtc_service("monitor")

        health_status["checks"]["signaling_server"] = signaling_server is not None
        health_status["checks"]["session_manager"] = session_manager is not None
        health_status["checks"]["monitor"] = monitor is not None

        # Service Health Checks
        if signaling_server and hasattr(signaling_server, "is_healthy"):
            health_status["checks"]["signaling_server_healthy"] = await signaling_server.is_healthy()

        if session_manager and hasattr(session_manager, "is_healthy"):
            health_status["checks"]["session_manager_healthy"] = await session_manager.is_healthy()

        if monitor and hasattr(monitor, "is_healthy"):
            health_status["checks"]["monitor_healthy"] = await monitor.is_healthy()

        # Overall Health
        health_status["healthy"] = all([
            _WEBRTC_AVAILABLE,
            health_status["checks"].get("signaling_server", False),
            health_status["checks"].get("session_manager", False)
        ])

    except Exception as e:
        logger.error(f"Fehler beim WebRTC Health Check: {e}")
        health_status["checks"]["error"] = str(e)

    return health_status

# =============================================================================
# Cleanup und Shutdown
# =============================================================================

async def shutdown_webrtc_system() -> None:
    """Fährt das WebRTC-System ordnungsgemäß herunter."""
    logger.info("Fahre WebRTC-System herunter")

    # Services herunterfahren
    for name, service in _webrtc_services.items():
        try:
            if hasattr(service, "shutdown"):
                await service.shutdown()
            logger.debug(f"WebRTC-Service heruntergefahren: {name}")
        except Exception as e:
            logger.error(f"Fehler beim Herunterfahren von {name}: {e}")

    # Registry leeren
    _webrtc_services.clear()
    logger.info("WebRTC-System heruntergefahren")

# =============================================================================
# Öffentliche API
# =============================================================================

__all__ = [
    # Core Components
    "WebRTCSignalingServer",
    "WebRTCSessionManager",
    "WebRTCMonitor",
    # Types
    "SignalingMessage",
    "WebRTCSession",
    "WebRTCSessionState",
    "WebRTCConfiguration",
    "WebRTCMetrics",
    # Configuration
    "WebRTCConfig",
    "get_webrtc_config",
    "create_webrtc_config",
    # Factory Functions
    "create_webrtc_signaling_server",
    "create_webrtc_session_manager",
    "create_webrtc_monitor",  # Importiert aus monitoring.py
    "create_webrtc_monitor_with_service_registration",
    # Service Management
    "register_webrtc_service",
    "get_webrtc_service",
    "unregister_webrtc_service",
    # System Status
    "get_webrtc_system_status",
    "is_webrtc_system_healthy",
    "perform_webrtc_health_check",
    # Lifecycle
    "shutdown_webrtc_system",
    # Constants
    "__version__",
]

# =============================================================================
# Initialisierung
# =============================================================================

if _WEBRTC_AVAILABLE:
    logger.info(f"WebRTC-System initialisiert (Version {__version__})")
else:
    logger.warning("WebRTC-System nicht verfügbar - läuft im Fallback-Modus")
