"""Zentraler Dependency Injection Container für Keiko.

Stellt Registrierung, Auflösung und Lifecycle-Management für Services bereit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from kei_logging import get_logger

# Service-Imports
from services.interfaces.rate_limiter import RateLimiterBackend, RateLimiterService
from services.interfaces.service_manager import ServiceManagerInterface, ServiceManagerService
from services.interfaces.webhook_manager import WebhookManagerService

# Conditional imports mit try/except für optionale Services
try:
    from services.interfaces.bus_service import BusService as IMessagingService
    from services.messaging.service import get_bus_service as legacy_bus_factory
    BUS_SERVICE_AVAILABLE = True
except ImportError:
    IMessagingService = None
    legacy_bus_factory = None
    BUS_SERVICE_AVAILABLE = False

try:
    from config.monitoring_config import create_monitoring_config_from_settings
    from monitoring.interfaces import IMonitoringService
    from monitoring.monitoring_service import MonitoringService
    MONITORING_SERVICE_AVAILABLE = True
except ImportError:
    IMonitoringService = None
    MonitoringService = None
    create_monitoring_config_from_settings = None
    MONITORING_SERVICE_AVAILABLE = False

try:
    from config.voice_rate_limiting_config import get_voice_rate_limiting_settings
    from voice_rate_limiting.interfaces import IVoiceRateLimitService
    from voice_rate_limiting.service import create_voice_rate_limit_service
    VOICE_RATE_LIMITING_AVAILABLE = True
except ImportError:
    IVoiceRateLimitService = None
    create_voice_rate_limit_service = None
    get_voice_rate_limiting_settings = None
    VOICE_RATE_LIMITING_AVAILABLE = False

try:
    from agents.circuit_breaker import create_agent_circuit_breaker_service
    from agents.circuit_breaker.interfaces import IAgentCircuitBreakerService
    from config.agent_circuit_breaker_config import get_agent_circuit_breaker_settings
    AGENT_CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    IAgentCircuitBreakerService = None
    create_agent_circuit_breaker_service = None
    get_agent_circuit_breaker_settings = None
    AGENT_CIRCUIT_BREAKER_AVAILABLE = False

try:
    from config.voice_performance_config import get_voice_performance_settings
    from voice_performance.service import VoicePerformanceService, create_voice_performance_service
    VOICE_PERFORMANCE_AVAILABLE = True
except ImportError:
    VoicePerformanceService = None
    create_voice_performance_service = None
    get_voice_performance_settings = None
    VOICE_PERFORMANCE_AVAILABLE = False

try:
    from config.unified_rate_limiting import get_unified_rate_limit_config
    from services.redis_rate_limiter import MemoryRateLimiter, RedisRateLimiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    get_unified_rate_limit_config = None
    RedisRateLimiter = None
    MemoryRateLimiter = None
    RATE_LIMITER_AVAILABLE = False

try:
    from services.webhooks.manager import get_webhook_manager
    WEBHOOK_MANAGER_AVAILABLE = True
except ImportError:
    get_webhook_manager = None
    WEBHOOK_MANAGER_AVAILABLE = False

try:
    from services.core.manager import ServiceManager
    SERVICE_MANAGER_AVAILABLE = True
except ImportError:
    ServiceManager = None
    SERVICE_MANAGER_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class ServiceDescriptor:
    """Beschreibt eine Service-Registrierung im Container."""

    factory: Callable[[Container], Any]
    singleton: bool = True
    instance: Any | None = None


class Container:
    """Einfacher DI-Container mit Singleton- und Factory-Support.

    - Registrierung per Interface-Typ -> Factory
    - Auflösung mit automatischer Konstruktor-Injektion (best-effort)
    - Singleton-Caching optional aktivierbar
    """

    def __init__(self) -> None:
        self._registry: dict[Any, ServiceDescriptor] = {}

    def register(self, interface: Any, factory: Callable[[Container], Any], *, singleton: bool = True) -> None:
        """Registriert eine Factory für ein Interface.

        Args:
            interface: Interface- oder Basisklasse.
            factory: Factory-Funktion, die die Implementierung erstellt.
            singleton: Ob als Singleton gecached wird.
        """
        self._registry[interface] = ServiceDescriptor(factory=factory, singleton=singleton)

    def resolve(self, interface: type[T]) -> T:
        """Löst eine Implementierung für ein Interface auf.

        Führt Lazy-Instanziierung durch und cached bei Singleton.
        """
        if interface not in self._registry:
            raise KeyError(f"Kein Service für Interface {interface} registriert")

        desc = self._registry[interface]
        if desc.singleton and desc.instance is not None:
            return desc.instance  # type: ignore[return-value]

        # Instanz erzeugen
        instance = desc.factory(self)
        if desc.singleton:
            desc.instance = instance
        return instance  # type: ignore[return-value]

    # Komfort-API
    def try_resolve(self, interface: type[T]) -> T | None:
        """Versucht, ein Interface aufzulösen; gibt None bei Fehlern zurück."""
        try:
            return self.resolve(interface)
        except Exception:
            return None


# Globaler Container für App-Kontext
_global_container: Container | None = None


def get_container() -> Container:
    """Gibt den globalen Container zurück (lazy erstellt)."""
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def bootstrap_defaults() -> None:
    """Registriert Standard-Implementierungen für bekannte Interfaces."""
    container = get_container()

    # Enhanced services removed - focusing on core DI container only

    # MessagingService -> Adapter auf bestehende Implementierung (falls verfügbar)
    if BUS_SERVICE_AVAILABLE and IMessagingService is not None:
        def bus_factory(_: Container) -> Any:
            # Nutzt bestehende globale Factory; bleibt rückwärtskompatibel
            return legacy_bus_factory()

        container.register(IMessagingService, bus_factory, singleton=True)
    else:
        logger.debug("MessagingService nicht verfügbar, überspringen")

    # RateLimiterService -> Redis oder Memory Fallback
    def rate_limiter_factory(_: Container) -> Any:
        if not RATE_LIMITER_AVAILABLE:
            raise RuntimeError("Rate limiter services not available")

        try:
            config = get_unified_rate_limit_config()
            # Versuche Redis Rate Limiter
            if config.backend.value == "redis":
                return RedisRateLimiter(config)
            # Fallback zu Memory Rate Limiter
            return MemoryRateLimiter()
        except Exception:
            # Fallback bei Konfigurationsfehlern
            return MemoryRateLimiter()

    container.register(RateLimiterService, rate_limiter_factory, singleton=True)
    container.register(RateLimiterBackend, rate_limiter_factory, singleton=True)  # Backward compatibility

    # WebhookManagerService -> Adapter auf bestehende WebhookManager Implementierung
    def webhook_manager_factory(_: Container) -> Any:
        if not WEBHOOK_MANAGER_AVAILABLE:
            raise RuntimeError("Webhook manager service not available")

        # Erstelle Adapter-Klasse die WebhookManagerService Interface implementiert
        class WebhookManagerAdapter(WebhookManagerService):
            def __init__(self):
                self._manager = get_webhook_manager()
                self._initialized = False

            async def initialize(self) -> None:
                """Initialisiert den Webhook Manager Service."""
                if not self._initialized:
                    # WebhookManager ist bereits initialisiert durch get_webhook_manager()
                    self._initialized = True

            async def shutdown(self) -> None:
                """Fährt den Webhook Manager Service herunter."""
                if self._initialized:
                    await self._manager.shutdown_with_timeout()
                    self._initialized = False

            async def start_worker_pool(self) -> None:
                await self._manager.start_worker_pool()

            async def stop_worker_pool(self) -> None:
                await self._manager.shutdown_with_timeout()

            async def send_webhook(self, url: str, payload: Any, config: Any) -> bool:
                try:
                    await self._manager.enqueue_outbound(url, payload, config)
                    return True
                except Exception:
                    return False

            async def register_webhook(self, event_type: str, url: str, config: Any) -> str:
                # Vereinfachte Implementierung - nutzt Target Registry
                import uuid
                webhook_id = str(uuid.uuid4())
                # Hier könnte die Target Registry verwendet werden
                return webhook_id

            async def unregister_webhook(self, webhook_id: str) -> bool:
                # Vereinfachte Implementierung
                return True

            async def get_registered_webhooks(self) -> list[str]:
                # Vereinfachte Implementierung
                return []

        return WebhookManagerAdapter()

    container.register(WebhookManagerService, webhook_manager_factory, singleton=True)

    # ServiceManagerService -> Adapter auf bestehende ServiceManager Implementierung
    def service_manager_factory(_: Container) -> Any:
        if not SERVICE_MANAGER_AVAILABLE:
            raise RuntimeError("Service manager not available")

        # Erstelle Adapter-Klasse die ServiceManagerService Interface implementiert
        class ServiceManagerAdapter(ServiceManagerService):
            def __init__(self):
                self._manager = ServiceManager()
                self._initialized = False

            async def initialize(self) -> None:
                """Initialisiert den Service Manager."""
                if not self._initialized:
                    await self._manager.initialize()
                    self._initialized = True

            async def shutdown(self) -> None:
                """Fährt den Service Manager herunter."""
                if self._initialized:
                    await self._manager.cleanup()
                    self._initialized = False

            async def cleanup(self) -> None:
                """Bereinigt den Service Manager (Alias für shutdown)."""
                await self.shutdown()

            async def register_service(self, service_id: str, service: Any) -> None:
                await self._manager.register_service(service_id, service)

            async def unregister_service(self, service_id: str) -> None:
                await self._manager.unregister_service(service_id)

            async def get_health_status(self) -> dict[str, Any]:
                return await self._manager.get_health_status()

            def is_healthy(self) -> bool:
                return self._manager.is_healthy()

            async def get_service_list(self) -> list[str]:
                return await self._manager.get_service_list()

            async def get_service_status(self, service_id: str) -> dict[str, Any]:
                # Vereinfachte Implementierung - prüft ob Service existiert
                service = self._manager.get_service(service_id)
                if service is None:
                    raise RuntimeError(f"Service nicht gefunden: {service_id}")

                return {
                    "service_id": service_id,
                    "status": "available" if service else "unavailable",
                    "healthy": True
                }

        return ServiceManagerAdapter()

    container.register(ServiceManagerService, service_manager_factory, singleton=True)
    container.register(ServiceManagerInterface, service_manager_factory, singleton=True)  # Backward compatibility

    # MonitoringService -> Comprehensive Monitoring System
    if MONITORING_SERVICE_AVAILABLE:
        def monitoring_service_factory(_: Container) -> Any:
            config = create_monitoring_config_from_settings()
            return MonitoringService(config)

        container.register(IMonitoringService, monitoring_service_factory, singleton=True)
        logger.info("Monitoring service registered in DI container")
    else:
        logger.warning("Monitoring service not available, skipping registration")

    # VoiceRateLimitService -> Voice-specific Rate Limiting
    if VOICE_RATE_LIMITING_AVAILABLE:
        # Prüfe Konfiguration - nur registrieren wenn aktiviert
        settings = get_voice_rate_limiting_settings()
        if settings.enabled:
            def voice_rate_limit_service_factory(_: Container) -> Any:
                return create_voice_rate_limit_service(settings)

            container.register(IVoiceRateLimitService, voice_rate_limit_service_factory, singleton=True)
            logger.info("Voice rate limiting service registered in DI container")
        else:
            logger.info("Voice rate limiting service disabled in configuration, skipping registration")
    else:
        logger.warning("Voice rate limiting service not available, skipping registration")

    # AgentCircuitBreakerService -> Agent-specific Circuit Breaker
    if AGENT_CIRCUIT_BREAKER_AVAILABLE:
        def agent_circuit_breaker_service_factory(_: Container) -> Any:
            circuit_breaker_settings = get_agent_circuit_breaker_settings()
            return create_agent_circuit_breaker_service(circuit_breaker_settings)

        container.register(IAgentCircuitBreakerService, agent_circuit_breaker_service_factory, singleton=True)
        logger.info("Agent circuit breaker service registered in DI container")
    else:
        logger.warning("Agent circuit breaker service not available, skipping registration")

    # VoicePerformanceService -> Voice Performance Optimization
    if VOICE_PERFORMANCE_AVAILABLE:
        def voice_performance_service_factory(_: Container) -> Any:
            performance_settings = get_voice_performance_settings()
            return create_voice_performance_service(performance_settings)

        container.register(VoicePerformanceService, voice_performance_service_factory, singleton=True)
        logger.info("Voice performance service registered in DI container")
    else:
        logger.warning("Voice performance service not available, skipping registration")


# Enhanced services function removed - files no longer exist


__all__ = [
    "Container",
    "bootstrap_defaults",
    "get_container",
    "reset_container",
]


def reset_container() -> None:
    """Setzt den globalen Container zurück (nur für Tests)."""
    global _global_container
    _global_container = None
