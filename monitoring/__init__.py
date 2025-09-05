# backend/monitoring/__init__.py
"""Zentrales Monitoring-System für Keiko.

Stellt einen einfachen Manager bereit, der Metriken, Health-Checks und
Tracing initialisiert und verwaltet. Öffentliche Convenience-Funktionen
erlauben eine schlanke Nutzung ohne direkte Instanziierung.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

# Runtime imports für Type Annotations
try:
    from .metrics_collector import MetricsCollector
except ImportError:
    MetricsCollector = None  # Fallback für fehlende MetricsCollector

try:
    from .health_checker import HealthChecker
except ImportError:
    HealthChecker = None  # Fallback für fehlende HealthChecker

try:
    from opentelemetry.sdk.trace import TracerProvider
except ImportError:
    TracerProvider = None  # Fallback für fehlende OpenTelemetry

logger = get_logger(__name__)

# OpenTelemetry Konstanten
_DEFAULT_SERVICE_NAME: Final = "keiko-backend"
_AZURE_LOG_LEVEL: Final = logging.WARNING


# ============================================================================
# MONITORING MANAGER - ZENTRALE KLASSE
# ============================================================================

class MonitoringManager:
    """Zentraler Manager für alle Monitoring-Komponenten.

    Verwaltet die Initialisierung von benutzerdefinierten Metriken,
    Health-Checks sowie optionalem OpenTelemetry-Tracing.
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._components: dict[str, bool] = {}
        self._metrics_collector: Any | None = None
        self._health_checker: Any | None = None

    async def initialize(
        self,
        service_name: str = "keiko-backend",
        enable_metrics: bool = True,
        enable_health: bool = True,
        enable_tracing: bool = True,
    ) -> dict[str, Any]:
        """Initialisiert alle Monitoring-Komponenten.

        Args:
            service_name: Name des Dienstes für Tracing.
            enable_metrics: Aktiviert das Sammeln von Metriken.
            enable_health: Aktiviert Health-Checks.
            enable_tracing: Aktiviert OpenTelemetry-Tracing/Export.

        Returns:
            Zusammenfassung der initialisierten Komponenten und Fehler.
        """
        if self._initialized:
            return {"success": True, "already_initialized": True}

        result = {"success": True, "components": [], "errors": []}

        # Custom Metrics
        if enable_metrics:
            self._metrics_collector = self._init_metrics()
            if self._metrics_collector:
                # Asynchronen Start ausführen, wenn vorhanden
                start_fn: Callable[..., Any] | None = getattr(self._metrics_collector, "start", None)
                if callable(start_fn):
                    try:
                        await start_fn()  # type: ignore[func-returns-value]
                    except Exception as exc:
                        logger.warning(f"Metrics Start fehlgeschlagen: {exc}")
                result["components"].append("metrics")
                self._components["metrics"] = True
            else:
                result["errors"].append("metrics_failed")

        # Health Checks
        if enable_health:
            self._health_checker = self._init_health()
            if self._health_checker:
                # Asynchronen Start ausführen, damit Checks laufen
                start_fn = getattr(self._health_checker, "start", None)
                if callable(start_fn):
                    try:
                        await start_fn()  # type: ignore[func-returns-value]
                    except Exception as exc:
                        logger.warning(f"Health Manager Start fehlgeschlagen: {exc}")
                result["components"].append("health")
                self._components["health"] = True
            else:
                result["errors"].append("health_failed")

        # OpenTelemetry Tracing
        if enable_tracing:
            if self._init_tracing(service_name):
                result["components"].append("tracing")
                self._components["tracing"] = True
            else:
                result["errors"].append("tracing_failed")

        self._initialized = True
        logger.info(f"Monitoring initialisiert: {len(result['components'])} Komponenten aktiv")
        return result

    def _init_metrics(self) -> MetricsCollector | None:
        """Initialisiert Metrics Collection."""
        try:
            from monitoring.custom_metrics import MetricsCollector
            return MetricsCollector()
        except ImportError:
            logger.warning("Metrics nicht verfügbar")
            return None

    def _init_health(self) -> HealthChecker | None:
        """Initialisiert Health Checks."""
        try:
            from monitoring.health_checks import HealthCheckManager
            return HealthCheckManager()
        except ImportError:
            logger.warning("Health Checks nicht verfügbar")
            return None

    def _init_tracing(self, service_name: str) -> bool:
        """Initialisiert OpenTelemetry Tracing."""
        try:
            init_tracing(service_name=service_name)
            # LangSmith-Bridge lazy-initialisieren, wenn verfügbar
            try:
                from monitoring.langsmith_integration import get_langsmith_integration
                get_langsmith_integration()
            except Exception as exc:
                logger.debug(f"LangSmith Bridge nicht initialisiert: {exc}")
            return True
        except ImportError:
            logger.warning("Tracing nicht verfügbar")
            return False

    async def shutdown(self) -> None:
        """Beendet alle Monitoring-Komponenten."""
        if self._metrics_collector and hasattr(self._metrics_collector, "stop"):
            try:
                await self._metrics_collector.stop()
            except Exception as exc:
                logger.debug(f"Metrics Stop Fehler: {exc}")
        if self._health_checker and hasattr(self._health_checker, "stop"):
            try:
                await self._health_checker.stop()
            except Exception as exc:
                logger.debug(f"Health Manager Stop Fehler: {exc}")
        self._initialized = False
        logger.info("Monitoring heruntergefahren")

    def get_status(self) -> dict[str, Any]:
        """Status des Monitoring-Systems."""
        return {
            "initialized": self._initialized,
            "components": self._components.copy(),
            "timestamp": datetime.now().isoformat()
        }

    def record_metric(self, name: str, value: int | float, tags: dict[str, Any] | None = None) -> None:
        """Zeichnet Metrik auf."""
        if self._metrics_collector:
            self._metrics_collector.record_metric(name, value, tags)


# ============================================================================
# GLOBALE INSTANZ UND CONVENIENCE FUNCTIONS
# ============================================================================

# Globale Manager-Instanz
_monitoring_manager: MonitoringManager | None = None


def get_monitoring_manager() -> MonitoringManager:
    """Gibt globale MonitoringManager-Instanz zurück."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager


# Legacy-Kompatibilitätsfunktionen - konsolidiert
async def initialize_monitoring(**kwargs) -> dict[str, Any]:
    """Initialisiert Monitoring-System."""
    return await get_monitoring_manager().initialize(**kwargs)

async def shutdown_monitoring():
    """Beendet Monitoring-System."""
    await get_monitoring_manager().shutdown()

def get_monitoring_status() -> dict[str, Any]:
    """Status des Monitoring-Systems."""
    return get_monitoring_manager().get_status()

def record_custom_metric(name: str, value: int | float, tags: dict | None = None):
    """Zeichnet Metrik auf."""
    get_monitoring_manager().record_metric(name, value, tags)


# ============================================================================
# TRACER UND METRICS - KRITISCH FÜR MIDDLEWARE
# ============================================================================

def get_tracer(name: str = "default"):
    """Gibt OpenTelemetry Tracer zurück - essentiell für TracingMiddleware."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        # Einfacher Fallback-Tracer
        return _DummyTracer()


class _DummyTracer:
    """Fallback-Tracer für fehlende OpenTelemetry."""
    def start_span(self, name: str, **kwargs):
        return _DummySpan()

class _DummySpan:
    """Fallback-Span für fehlende OpenTelemetry."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_attribute(self, key: str, value: Any):
        pass


# Metrics Collector Zugriff für Middleware
def metrics_collector() -> Any:
    """Gibt Metrics Collector zurück - essentiell für Tracing.

    Returns:
        Aktueller Metrics Collector oder Dummy-Implementation.
    """
    manager = get_monitoring_manager()
    if manager._metrics_collector:
        return manager._metrics_collector
    return _DummyMetrics()


class _DummyMetrics:
    """Fallback für fehlende Metrics."""
    def record_metric(self, name: str, value: int | float, tags: dict[str, Any] | None = None) -> None:
        pass

    def increment_counter(self, name: str, value: int = 1, tags: dict[str, Any] | None = None) -> None:
        pass


# ============================================================================
# OPENTELEMETRY INTEGRATION
# ============================================================================

def init_tracing(
    *,
    enable_local: bool = True,
    service_name: str = _DEFAULT_SERVICE_NAME,
) -> TracerProvider | None:
    """Initialisiert OpenTelemetry-Tracing.

    Args:
        enable_local: Lokales Tracing ohne Export (Standard: True)
        service_name: Service-Name für Tracing

    Returns:
        TracerProvider bei Azure-Tracing, sonst None
    """
    if enable_local:
        logger.info("Lokales Tracing aktiv - keine Azure-Exportierung")
        return None

    try:
        from config.settings import settings
        conn_str = settings.app_insights_connection_string
    except Exception:
        conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if not conn_str:
        logger.warning("Azure Connection String fehlt - Tracing deaktiviert")
        return None

    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        from opentelemetry import trace as otel
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Provider und Exporter konfigurieren
        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = AzureMonitorTraceExporter(connection_string=conn_str)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        otel.set_tracer_provider(provider)
        logger.info("Azure Monitor Tracing für Service '%s' initialisiert", service_name)

        # Azure SDK-Logging reduzieren
        logging.getLogger("azure").setLevel(_AZURE_LOG_LEVEL)
        return provider

    except ImportError:
        logger.warning("Azure Monitor OpenTelemetry nicht verfügbar")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def start_performance_tracking(operation: str) -> str:
    """Startet Performance-Tracking."""
    tracking_id = str(uuid.uuid4())
    record_custom_metric(f"performance.{operation}.start", 1, {"tracking_id": tracking_id})
    return tracking_id

def stop_performance_tracking(tracking_id: str) -> dict[str, Any] | None:
    """Stoppt Performance-Tracking."""
    record_custom_metric("performance.stop", 1, {"tracking_id": tracking_id})
    return {"tracking_id": tracking_id, "timestamp": datetime.now().isoformat()}


async def get_health_status() -> dict[str, Any]:
    """Führt Health Check durch."""
    manager = get_monitoring_manager()
    if manager._health_checker and hasattr(manager._health_checker, "check_health"):
        try:
            return await manager._health_checker.check_health()
        except Exception as e:
            logger.exception(f"Health check fehlgeschlagen: {e}")

    return {
        "healthy": manager._initialized,
        "components": manager._components,
        "timestamp": datetime.now().isoformat()
    }


def is_monitoring_healthy() -> bool:
    """Prüft ob Monitoring gesund ist."""
    return get_monitoring_manager()._initialized


# ============================================================================
# PUBLIC API - NUR ESSENTIELLE EXPORTS
# ============================================================================

__all__ = [
    # Hauptklasse
    "MonitoringManager",
    "get_health_status",
    # Status und Health
    "get_monitoring_status",
    "get_tracer",  # KRITISCH für TracingMiddleware
    # OpenTelemetry Integration
    "init_tracing",  # Für main.py Kompatibilität
    # Core Functions - essentiell für Middleware
    "initialize_monitoring",
    "is_monitoring_healthy",
    "metrics_collector",  # KRITISCH für TracingMiddleware
    # Performance und Metriken
    "record_custom_metric",
    "shutdown_monitoring",
    "start_performance_tracking",
    "stop_performance_tracking",
]

# Package Metadaten
__version__ = "0.0.1"
__author__ = "Keiko Development Team"
