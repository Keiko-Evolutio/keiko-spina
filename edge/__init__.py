"""Edge Computing Backend für Voice-Service-System

Dieses Modul implementiert die Backend-Komponenten für Edge Computing,
einschließlich Edge-Node-Management, Distributed Processing und Performance-Monitoring.

Features:
- Edge-Node-Registry und Health Monitoring
- Distributed Task Processing
- Adaptive Load Balancing
- Performance-Metriken und Auto-Scaling
- Integration mit bestehender Voice-Architektur

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
    from .auto_scaler import EdgeAutoScaler, create_auto_scaler
    from .config import EdgeConfig, create_edge_config, get_edge_config
    from .edge_types import (
        EdgeConfiguration,
        EdgeMetrics,
        EdgeNodeInfo,
        EdgeNodeStatus,
        EdgeTask,
        EdgeTaskResult,
    )
    from .load_balancer import EdgeLoadBalancer, create_load_balancer
    from .node_registry import EdgeNodeRegistry, create_node_registry
    from .performance_monitor import EdgePerformanceMonitor, create_performance_monitor
    from .task_processor import DistributedTaskProcessor, create_task_processor

    _EDGE_AVAILABLE = True
    logger.info("Edge Computing-Komponenten erfolgreich geladen")

except ImportError as import_error:
    logger.warning(f"Edge Computing-Komponenten nicht verfügbar: {import_error}")

    # Fallback-Implementierungen
    EdgeNodeRegistry = None  # type: ignore[assignment]
    DistributedTaskProcessor = None  # type: ignore[assignment]
    EdgeLoadBalancer = None  # type: ignore[assignment]
    EdgePerformanceMonitor = None  # type: ignore[assignment]
    EdgeAutoScaler = None  # type: ignore[assignment]

    # Fallback Factory-Funktionen
    create_node_registry = None  # type: ignore[assignment]
    create_task_processor = None  # type: ignore[assignment]
    create_load_balancer = None  # type: ignore[assignment]
    create_performance_monitor = None  # type: ignore[assignment]
    create_auto_scaler = None  # type: ignore[assignment]

    # Fallback Types
    EdgeNodeInfo = None  # type: ignore[assignment]
    EdgeNodeStatus = None  # type: ignore[assignment]
    EdgeTask = None  # type: ignore[assignment]
    EdgeTaskResult = None  # type: ignore[assignment]
    EdgeMetrics = None  # type: ignore[assignment]
    EdgeConfiguration = None  # type: ignore[assignment]

    # Fallback Config
    EdgeConfig = None  # type: ignore[assignment]
    get_edge_config = None  # type: ignore[assignment]
    create_edge_config = None  # type: ignore[assignment]

    _EDGE_AVAILABLE = False

# =============================================================================
# Service Registry
# =============================================================================

_edge_services: dict[str, Any] = {}

def register_edge_service(name: str, service: Any) -> None:
    """Registriert einen Edge-Service."""
    _edge_services[name] = service
    logger.debug(f"Edge-Service registriert: {name}")

def get_edge_service(name: str) -> Any:
    """Gibt einen registrierten Edge-Service zurück."""
    return _edge_services.get(name)

def unregister_edge_service(name: str) -> None:
    """Entfernt einen Edge-Service aus der Registry."""
    if name in _edge_services:
        del _edge_services[name]
        logger.debug(f"Edge-Service entfernt: {name}")

# =============================================================================
# Factory Functions
# =============================================================================

def create_edge_node_registry(
    config: EdgeConfig | None = None
) -> EdgeNodeRegistry | None:
    """Erstellt eine Edge-Node-Registry."""
    if not _EDGE_AVAILABLE or not EdgeNodeRegistry or not create_node_registry:
        logger.warning("Edge-Node-Registry nicht verfügbar")
        return None

    try:
        registry_config = config or get_edge_config()
        if not registry_config:
            logger.error("Keine gültige Edge-Konfiguration verfügbar")
            return None

        registry = create_node_registry(registry_config)
        if registry:
            register_edge_service("node_registry", registry)
            logger.debug("Edge-Node-Registry erfolgreich erstellt")
        return registry
    except Exception as registry_error:
        logger.error(f"Fehler beim Erstellen der Edge-Node-Registry: {registry_error}")
        return None

def create_edge_task_processor(
    config: EdgeConfig | None = None
) -> DistributedTaskProcessor | None:
    """Erstellt einen Distributed Task Processor."""
    if not _EDGE_AVAILABLE or not DistributedTaskProcessor or not create_task_processor:
        logger.warning("Distributed Task Processor nicht verfügbar")
        return None

    try:
        processor_config = config or get_edge_config()
        if not processor_config:
            logger.error("Keine gültige Edge-Konfiguration verfügbar")
            return None

        processor = create_task_processor(processor_config)
        if processor:
            register_edge_service("task_processor", processor)
            logger.debug("Distributed Task Processor erfolgreich erstellt")
        return processor
    except Exception as processor_error:
        logger.error(f"Fehler beim Erstellen des Task Processors: {processor_error}")
        return None

def create_edge_load_balancer(
    config: EdgeConfig | None = None
) -> EdgeLoadBalancer | None:
    """Erstellt einen Edge Load Balancer."""
    if not _EDGE_AVAILABLE or not EdgeLoadBalancer:
        logger.warning("Edge Load Balancer nicht verfügbar")
        return None

    try:
        balancer_config = config or get_edge_config()
        balancer = create_load_balancer(balancer_config)
        register_edge_service("load_balancer", balancer)
        return balancer
    except Exception as balancer_error:
        logger.error(f"Fehler beim Erstellen des Load Balancers: {balancer_error}")
        return None

def create_edge_performance_monitor(
    config: EdgeConfig | None = None
) -> EdgePerformanceMonitor | None:
    """Erstellt einen Edge Performance Monitor."""
    if not _EDGE_AVAILABLE or not EdgePerformanceMonitor:
        logger.warning("Edge Performance Monitor nicht verfügbar")
        return None

    try:
        monitor_config = config or get_edge_config()
        monitor = create_performance_monitor(monitor_config)
        register_edge_service("performance_monitor", monitor)
        return monitor
    except Exception as monitor_error:
        logger.error(f"Fehler beim Erstellen des Performance Monitors: {monitor_error}")
        return None

def create_edge_auto_scaler(
    config: EdgeConfig | None = None
) -> EdgeAutoScaler | None:
    """Erstellt einen Edge Auto Scaler."""
    if not _EDGE_AVAILABLE or not EdgeAutoScaler or not create_auto_scaler:
        logger.warning("Edge Auto Scaler nicht verfügbar")
        return None

    try:
        scaler_config = config or get_edge_config()
        if not scaler_config:
            logger.error("Keine gültige Edge-Konfiguration verfügbar")
            return None

        scaler = create_auto_scaler(scaler_config)
        if scaler:
            register_edge_service("auto_scaler", scaler)
            logger.debug("Edge Auto Scaler erfolgreich erstellt")
        return scaler
    except Exception as scaler_error:
        logger.error(f"Fehler beim Erstellen des Auto Scalers: {scaler_error}")
        return None

# =============================================================================
# Edge System Initialization
# =============================================================================

async def initialize_edge_system(config: EdgeConfig | None = None) -> dict[str, Any]:
    """Initialisiert das komplette Edge Computing-System."""
    if not _EDGE_AVAILABLE:
        logger.warning("Edge Computing-System nicht verfügbar")
        return {}

    edge_config = config or get_edge_config()
    initialized_services = {}

    try:
        # Node Registry initialisieren
        node_registry = create_edge_node_registry(edge_config)
        if node_registry:
            await node_registry.start()
            initialized_services["node_registry"] = node_registry
            logger.info("Edge-Node-Registry initialisiert")

        # Task Processor initialisieren
        task_processor = create_edge_task_processor(edge_config)
        if task_processor:
            await task_processor.start()
            initialized_services["task_processor"] = task_processor
            logger.info("Distributed Task Processor initialisiert")

        # Load Balancer initialisieren
        load_balancer = create_edge_load_balancer(edge_config)
        if load_balancer:
            await load_balancer.start()
            initialized_services["load_balancer"] = load_balancer
            logger.info("Edge Load Balancer initialisiert")

        # Performance Monitor initialisieren
        performance_monitor = create_edge_performance_monitor(edge_config)
        if performance_monitor:
            await performance_monitor.start()
            initialized_services["performance_monitor"] = performance_monitor
            logger.info("Edge Performance Monitor initialisiert")

        # Auto Scaler initialisieren
        auto_scaler = create_edge_auto_scaler(edge_config)
        if auto_scaler:
            await auto_scaler.start()
            initialized_services["auto_scaler"] = auto_scaler
            logger.info("Edge Auto Scaler initialisiert")

        logger.info(f"Edge Computing-System initialisiert mit {len(initialized_services)} Services")
        return initialized_services

    except Exception as init_error:
        logger.error(f"Fehler bei der Edge-System-Initialisierung: {init_error}")
        # Cleanup bei Fehler
        await shutdown_edge_system()
        raise

# =============================================================================
# System Status und Health Checks
# =============================================================================

def get_edge_system_status() -> dict[str, Any]:
    """Gibt den Edge-System-Status zurück."""
    return {
        "edge_available": _EDGE_AVAILABLE,
        "version": __version__,
        "services": {
            "node_registry": get_edge_service("node_registry") is not None,
            "task_processor": get_edge_service("task_processor") is not None,
            "load_balancer": get_edge_service("load_balancer") is not None,
            "performance_monitor": get_edge_service("performance_monitor") is not None,
            "auto_scaler": get_edge_service("auto_scaler") is not None
        },
        "registered_services": list(_edge_services.keys()),
        "components": {
            "node_management": _EDGE_AVAILABLE,
            "task_processing": _EDGE_AVAILABLE,
            "load_balancing": _EDGE_AVAILABLE,
            "performance_monitoring": _EDGE_AVAILABLE,
            "auto_scaling": _EDGE_AVAILABLE
        }
    }

def is_edge_system_healthy() -> bool:
    """Prüft, ob das Edge-System funktionsfähig ist."""
    if not _EDGE_AVAILABLE:
        return False

    # Prüfe kritische Services
    node_registry = get_edge_service("node_registry")
    task_processor = get_edge_service("task_processor")

    return node_registry is not None and task_processor is not None

async def perform_edge_health_check() -> dict[str, Any]:
    """Führt einen umfassenden Edge-Health-Check durch."""
    health_status = {
        "healthy": False,
        "timestamp": None,
        "checks": {}
    }

    try:
        from datetime import UTC, datetime
        health_status["timestamp"] = datetime.now(UTC).isoformat()

        # System Availability Check
        health_status["checks"]["system_available"] = _EDGE_AVAILABLE

        # Service Checks
        services = ["node_registry", "task_processor", "load_balancer",
                   "performance_monitor", "auto_scaler"]

        for service_name in services:
            service = get_edge_service(service_name)
            health_status["checks"][f"{service_name}_available"] = service is not None

            # Service Health Checks
            if service and hasattr(service, "is_healthy"):
                health_status["checks"][f"{service_name}_healthy"] = await service.is_healthy()

        # Overall Health
        health_status["healthy"] = all([
            _EDGE_AVAILABLE,
            health_status["checks"].get("node_registry_available", False),
            health_status["checks"].get("task_processor_available", False)
        ])

    except Exception as health_error:
        logger.error(f"Fehler beim Edge Health Check: {health_error}")
        health_status["checks"]["error"] = str(health_error)

    return health_status

# =============================================================================
# Performance Metrics
# =============================================================================

async def get_edge_performance_metrics() -> dict[str, Any]:
    """Sammelt Performance-Metriken von allen Edge-Services."""
    metrics = {
        "timestamp": None,
        "overall": {},
        "services": {}
    }

    try:
        from datetime import UTC, datetime
        metrics["timestamp"] = datetime.now(UTC).isoformat()

        # Performance Monitor Metriken
        performance_monitor = get_edge_service("performance_monitor")
        if performance_monitor and hasattr(performance_monitor, "get_metrics"):
            metrics["overall"] = await performance_monitor.get_metrics()

        # Service-spezifische Metriken
        services = ["node_registry", "task_processor", "load_balancer", "auto_scaler"]

        for service_name in services:
            service = get_edge_service(service_name)
            if service and hasattr(service, "get_metrics"):
                metrics["services"][service_name] = await service.get_metrics()

    except Exception as metrics_error:
        logger.error(f"Fehler beim Sammeln der Edge-Performance-Metriken: {metrics_error}")
        metrics["error"] = str(metrics_error)

    return metrics

# =============================================================================
# Cleanup und Shutdown
# =============================================================================

async def shutdown_edge_system() -> None:
    """Fährt das Edge-System ordnungsgemäß herunter."""
    logger.info("Fahre Edge-System herunter")

    # Services in umgekehrter Reihenfolge herunterfahren
    service_order = ["auto_scaler", "performance_monitor", "load_balancer",
                    "task_processor", "node_registry"]

    for service_name in service_order:
        service = get_edge_service(service_name)
        if service:
            try:
                if hasattr(service, "shutdown"):
                    await service.shutdown()
                logger.debug(f"Edge-Service heruntergefahren: {service_name}")
            except Exception as shutdown_error:
                logger.error(f"Fehler beim Herunterfahren von {service_name}: {shutdown_error}")

    # Registry leeren
    _edge_services.clear()
    logger.info("Edge-System heruntergefahren")

# =============================================================================
# Öffentliche API
# =============================================================================

__all__ = [
    # Core Components
    "EdgeNodeRegistry",
    "DistributedTaskProcessor",
    "EdgeLoadBalancer",
    "EdgePerformanceMonitor",
    "EdgeAutoScaler",
    # Types
    "EdgeNodeInfo",
    "EdgeNodeStatus",
    "EdgeTask",
    "EdgeTaskResult",
    "EdgeMetrics",
    "EdgeConfiguration",
    # Configuration
    "EdgeConfig",
    "get_edge_config",
    "create_edge_config",
    # Factory Functions
    "create_edge_node_registry",
    "create_edge_task_processor",
    "create_edge_load_balancer",
    "create_edge_performance_monitor",
    "create_edge_auto_scaler",
    # System Management
    "initialize_edge_system",
    "shutdown_edge_system",
    # Service Management
    "register_edge_service",
    "get_edge_service",
    "unregister_edge_service",
    # System Status
    "get_edge_system_status",
    "is_edge_system_healthy",
    "perform_edge_health_check",
    "get_edge_performance_metrics",
    # Constants
    "__version__",
]

# =============================================================================
# Initialisierung
# =============================================================================

if _EDGE_AVAILABLE:
    logger.info(f"Edge Computing-System initialisiert (Version {__version__})")
else:
    logger.warning("Edge Computing-System nicht verfügbar - läuft im Fallback-Modus")
