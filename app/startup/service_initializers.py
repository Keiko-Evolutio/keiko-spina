"""Service-spezifische Initializer für das Startup-Management.

Dieses Modul extrahiert die Service-Initialisierungslogik aus StartupManager
um die Komplexität zu reduzieren und Single Responsibility Principle zu befolgen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.constants import SERVICE_AGENTS, SERVICE_KEI_BUS, SERVICE_VOICE
from ..common.error_handlers import handle_async_service_errors
from ..common.logger_utils import get_module_logger, log_service_operation

# Service-Imports
try:
    from core.container import get_container
    from services.interfaces.service_manager import ServiceManagerInterface
    from services.interfaces.webhook_manager import WebhookManagerInterface
    CONTAINER_SERVICES_AVAILABLE = True
except ImportError:
    get_container = None
    ServiceManagerInterface = None
    WebhookManagerInterface = None
    CONTAINER_SERVICES_AVAILABLE = False

try:
    from agents.tools.mcp import mcp_registry
    from agents.tools.mcp.kei_mcp_client import ExternalMCPConfig
    from config.kei_mcp_config import KEI_MCP_SETTINGS
    MCP_AVAILABLE = True
except ImportError:
    KEI_MCP_SETTINGS = None
    ExternalMCPConfig = None
    mcp_registry = None
    MCP_AVAILABLE = False

try:
    from services.unified_domain_revalidation_service import UnifiedDomainRevalidationService
    DOMAIN_REVALIDATION_AVAILABLE = True
except ImportError:
    UnifiedDomainRevalidationService = None
    DOMAIN_REVALIDATION_AVAILABLE = False

try:
    from monitoring.interfaces import IMonitoringService
    MONITORING_SERVICE_AVAILABLE = True
except ImportError:
    IMonitoringService = None
    MONITORING_SERVICE_AVAILABLE = False

try:
    from voice_rate_limiting.interfaces import IVoiceRateLimitService
    VOICE_RATE_LIMITING_AVAILABLE = True
except ImportError:
    IVoiceRateLimitService = None
    VOICE_RATE_LIMITING_AVAILABLE = False

try:
    from agents.circuit_breaker.interfaces import IAgentCircuitBreakerService
    AGENT_CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    IAgentCircuitBreakerService = None
    AGENT_CIRCUIT_BREAKER_AVAILABLE = False

try:
    from voice_performance.service import VoicePerformanceService
    VOICE_PERFORMANCE_AVAILABLE = True
except ImportError:
    VoicePerformanceService = None
    VOICE_PERFORMANCE_AVAILABLE = False

if TYPE_CHECKING:
    from ..service_container import ServiceContainer

logger = get_module_logger(__name__)


class ServiceInitializer:
    """Basis-Klasse für Service-Initializer."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def initialize(self, app=None) -> bool:
        """Initialisiert den Service.

        Args:
            app: Optional FastAPI-Anwendung für Initializer die diese benötigen

        Returns:
            True wenn erfolgreich, False bei Fehlern
        """
        raise NotImplementedError("Subclasses must implement initialize()")


# Enhanced DI Container Initializer removed - files no longer exist


class MonitoringServiceInitializer(ServiceInitializer):
    """Initializer für den Monitoring-Service."""

    @handle_async_service_errors("monitoring_service", fallback_value=False)
    @log_service_operation(logger, "monitoring_service", "initialization")
    async def initialize(self) -> bool:
        """Initialisiert den Monitoring-Service."""
        try:
            if not MONITORING_SERVICE_AVAILABLE:
                logger.warning("Monitoring service not available, skipping initialization")
                return True  # Nicht kritisch für Startup

            # DI-basierte Initialisierung
            from core.container import get_container
            container = get_container()
            monitoring_service = container.resolve(IMonitoringService)

            # Monitoring-Service initialisieren
            await monitoring_service.initialize()

            logger.info("Monitoring service successfully initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            # Monitoring ist nicht kritisch für Startup
            return True


class VoiceRateLimitingInitializer(ServiceInitializer):
    """Initializer für den Voice Rate Limiting Service."""

    @handle_async_service_errors("voice_rate_limiting", fallback_value=True)  # Fallback auf True
    @log_service_operation(logger, "voice_rate_limiting", "initialization")
    async def initialize(self) -> bool:
        """Initialisiert den Voice Rate Limiting Service."""
        try:
            if not VOICE_RATE_LIMITING_AVAILABLE:
                logger.warning("Voice rate limiting service not available, skipping initialization")
                return True  # Nicht kritisch für Startup

            # Prüfe Konfiguration - nur initialisieren wenn aktiviert
            from config.voice_rate_limiting_config import get_voice_rate_limiting_settings
            settings = get_voice_rate_limiting_settings()

            if not settings.enabled:
                logger.info("Voice rate limiting service disabled in configuration, skipping initialization")
                return True  # Service ist deaktiviert, das ist OK

            # DI-basierte Initialisierung nur wenn Service aktiviert ist
            from core.container import get_container
            container = get_container()

            try:
                voice_rate_limit_service = container.resolve(IVoiceRateLimitService)
            except Exception as e:
                logger.warning(f"Failed to resolve voice rate limiting service from DI container: {e}, continuing without rate limiting")
                return True  # Service nicht registriert, das ist OK wenn deaktiviert

            # Voice Rate Limiting Service initialisieren mit robustem Error-Handling
            try:
                await voice_rate_limit_service.initialize()
                logger.info("Voice rate limiting service successfully initialized")
                return True
            except ImportError as e:
                logger.warning(f"Voice rate limiting service initialization failed due to missing dependency ({e}), continuing without rate limiting")
                return True  # Nicht kritisch für Startup
            except Exception as e:
                logger.warning(f"Voice rate limiting service initialization failed ({e}), continuing without rate limiting")
                return True  # Nicht kritisch für Startup

        except Exception as e:
            logger.warning(f"Failed to initialize voice rate limiting service: {e}, continuing without rate limiting")
            # Rate Limiting ist nicht kritisch für Startup
            return True


class AgentCircuitBreakerInitializer(ServiceInitializer):
    """Initializer für den Agent Circuit Breaker Service."""

    @handle_async_service_errors("circuit_breaker", fallback_value=True)  # Fallback auf True
    @log_service_operation(logger, "circuit_breaker", "initialization")
    async def initialize(self) -> bool:
        """Initialisiert den Agent Circuit Breaker Service."""
        try:
            if not AGENT_CIRCUIT_BREAKER_AVAILABLE:
                logger.warning("Agent circuit breaker service not available, skipping initialization")
                return True  # Nicht kritisch für Startup

            # DI-basierte Initialisierung
            from core.container import get_container
            container = get_container()
            agent_circuit_breaker_service = container.resolve(IAgentCircuitBreakerService)

            # Agent Circuit Breaker Service initialisieren mit robustem Error-Handling
            try:
                await agent_circuit_breaker_service.initialize()
                logger.info("Agent circuit breaker service successfully initialized")
                return True
            except ImportError as e:
                logger.warning(f"Agent circuit breaker service initialization failed due to missing dependency ({e}), continuing without circuit breaker")
                return True  # Nicht kritisch für Startup
            except Exception as e:
                logger.warning(f"Agent circuit breaker service initialization failed ({e}), continuing without circuit breaker")
                return True  # Nicht kritisch für Startup

        except Exception as e:
            logger.warning(f"Failed to initialize agent circuit breaker service: {e}, continuing without circuit breaker")
            # Circuit Breaker ist nicht kritisch für Startup
            return True


class VoicePerformanceInitializer(ServiceInitializer):
    """Initializer für den Voice Performance Service."""

    @handle_async_service_errors("voice_performance", fallback_value=True)  # Fallback auf True
    @log_service_operation(logger, "voice_performance", "initialization")
    async def initialize(self) -> bool:
        """Initialisiert den Voice Performance Service."""
        try:
            if not VOICE_PERFORMANCE_AVAILABLE:
                logger.warning("Voice performance service not available, skipping initialization")
                return True  # Nicht kritisch für Startup

            # DI-basierte Initialisierung
            from core.container import get_container
            container = get_container()
            voice_performance_service = container.resolve(VoicePerformanceService)

            # Voice Performance Service initialisieren mit robustem Error-Handling
            try:
                await voice_performance_service.initialize()
                logger.info("Voice performance service successfully initialized")
                return True
            except ImportError as e:
                logger.warning(f"Voice performance service initialization failed due to missing dependency ({e}), continuing without performance optimization")
                return True  # Nicht kritisch für Startup
            except Exception as e:
                logger.warning(f"Voice performance service initialization failed ({e}), continuing without performance optimization")
                return True  # Nicht kritisch für Startup

        except Exception as e:
            logger.warning(f"Failed to initialize voice performance service: {e}, continuing without performance optimization")
            # Performance Optimization ist nicht kritisch für Startup
            return True


class BusServiceInitializer(ServiceInitializer):
    """Initializer für den KEI-Bus Service."""

    @handle_async_service_errors(SERVICE_KEI_BUS, fallback_value=True)  # Fallback auf True - nicht kritisch
    @log_service_operation(logger, SERVICE_KEI_BUS, "initialization")
    async def initialize(self) -> bool:
        """Initialisiert den KEI-Bus Service."""
        try:
            # Prüfe ob KEI Bus aktiviert ist
            from services.messaging.config import bus_settings

            if not bus_settings.enabled:
                logger.info("KEI Bus is disabled, skipping initialization")
                return True  # Nicht kritisch für Startup

            # Versuche KEI Bus zu initialisieren
            bus = self.container.get_bus_service()
            await bus.initialize()
            logger.info("KEI Bus service successfully initialized")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize KEI Bus service: {e}, continuing without message bus")
            # KEI Bus ist nicht kritisch für Startup
            return True


class ServiceManagerInitializer(ServiceInitializer):
    """Initializer für den Service Manager."""

    @handle_async_service_errors("service_manager", fallback_value=False)
    @log_service_operation(logger, "service_manager", "initialization")
    async def initialize(self) -> bool:
        """Initialisiert den Service Manager mit DI-Fallback."""
        try:
            # DI-basierte Initialisierung bevorzugen
            if not CONTAINER_SERVICES_AVAILABLE:
                raise ImportError("Container services not available")
            svc = get_container().resolve(ServiceManagerInterface)
            self.container._service_manager = svc  # Bridge für Legacy-Aufrufe
            await svc.initialize()
            logger.info("Service Manager initialisiert (DI)")
            return True
        except Exception as exc:
            logger.warning(f"Service Manager (DI) konnte nicht initialisiert werden: {exc}")
            # Fallback: Legacy Service Manager
            await self.container.service_manager.initialize()
            logger.info("Service Manager initialisiert (Legacy)")
            return True


class AgentsInitializer(ServiceInitializer):
    """Initializer für das Agent System."""

    @handle_async_service_errors(SERVICE_AGENTS, fallback_value=False)
    @log_service_operation(logger, SERVICE_AGENTS, "startup")
    async def initialize(self) -> bool:
        """Startet das Agent System."""
        from agents.common import startup_agents
        result = await startup_agents()  # Kein Parameter mehr benötigt
        logger.info(f"Agent System gestartet: {result}")
        return True


class VoiceSystemInitializer(ServiceInitializer):
    """Initializer für das Voice System."""

    @handle_async_service_errors(SERVICE_VOICE, fallback_value=False)
    @log_service_operation(logger, SERVICE_VOICE, "validation_and_monitoring")
    async def initialize(self) -> bool:
        """Validiert und startet Voice System Monitoring."""
        try:
            from api.routes.voice_routes import voice_health_monitor, voice_monitoring_manager

            # Voice System Validation
            if voice_health_monitor:
                logger.info("Performing voice system startup validation...")
                voice_health = await voice_health_monitor.perform_startup_validation()

                if voice_health.startup_validation_passed:
                    logger.info("Voice system startup validation PASSED")
                    if voice_health.warning_count > 0:
                        logger.warning(f"Voice system has {voice_health.warning_count} warnings")
                else:
                    logger.error("Voice system startup validation FAILED")
                    self._log_voice_health_issues(voice_health)

            # Voice Monitoring starten
            if voice_monitoring_manager:
                logger.info("Starting voice system real-time monitoring...")
                import asyncio
                asyncio.create_task(voice_monitoring_manager.start_monitoring())

            return True
        except Exception as exc:
            logger.warning(f"Voice System Validierung/Monitoring Problem: {exc}")
            return False

    def _log_voice_health_issues(self, voice_health: Any) -> None:
        """Loggt Voice Health Issues detailliert."""
        logger.error(
            f"Critical issues detected: {voice_health.error_count} errors, "
            f"{voice_health.warning_count} warnings"
        )
        for component in voice_health.components:
            if component.status.value in ["failed", "critical"]:
                logger.error(f"{component.component.value}: {component.message}")
                if component.error:
                    logger.error(f"   Error: {component.error}")
            elif component.status.value == "warning":
                logger.warning(f"{component.component.value}: {component.message}")


class MCPServerInitializer(ServiceInitializer):
    """Initializer für MCP Server."""

    @handle_async_service_errors("mcp_servers", fallback_value=True)
    @log_service_operation(logger, "mcp_servers", "registration")
    async def initialize(self) -> bool:
        """Registriert MCP Server."""
        try:
            if not MCP_AVAILABLE:
                raise ImportError("MCP services not available")

            logger.info("Registriere KEI-MCP Server...")
            success_count = 0

            # Prüfe ob Server konfiguriert sind
            if not KEI_MCP_SETTINGS.auto_register_servers:
                logger.info("Keine MCP Server zur Registrierung konfiguriert")
                return True

            for server_config in KEI_MCP_SETTINGS.auto_register_servers:
                try:
                    # Konvertiere KEIMCPConfig zu ExternalMCPConfig
                    external_config = ExternalMCPConfig(
                        server_name=server_config.server_name,
                        base_url=server_config.base_url,
                        api_key=server_config.api_key,
                        timeout_seconds=server_config.timeout_seconds,
                        max_retries=server_config.max_retries,
                        custom_headers=server_config.custom_headers or {}
                    )

                    success = await mcp_registry.register_server(external_config, domain_validated=True)
                    if success:
                        logger.info(f"KEI-MCP Server {server_config.server_name} registriert")
                        success_count += 1
                    else:
                        logger.debug(f"Registrierung von {server_config.server_name} übersprungen")
                except Exception as e:
                    logger.debug(f"Fehler bei Server-Registrierung {server_config.server_name}: {e}")

            logger.info(f"MCP Server Registrierung abgeschlossen: {success_count} erfolgreich")
            return True  # Immer erfolgreich, auch wenn keine Server registriert
        except ImportError as e:
            logger.debug(f"MCP Module nicht verfügbar: {e}")
            return True  # Nicht kritisch
        except Exception as e:
            logger.debug(f"MCP Server Initialisierung fehlgeschlagen: {e}")
            return True  # Nicht kritisch für Startup


class DomainRevalidationInitializer(ServiceInitializer):
    """Initializer für Domain-Revalidierung Service."""

    @handle_async_service_errors("domain_revalidation", fallback_value=False)
    @log_service_operation(logger, "domain_revalidation", "startup")
    async def initialize(self) -> bool:
        """Startet den Domain-Revalidierung Service."""
        if not DOMAIN_REVALIDATION_AVAILABLE:
            return False
        service = UnifiedDomainRevalidationService()
        await service.start()
        return True


class WebhookAndRateLimitingInitializer(ServiceInitializer):
    """Initializer für Webhook Worker-Pool und Rate Limiting."""

    @handle_async_service_errors("webhook_rate_limiting", fallback_value=False)
    async def initialize(self) -> bool:
        """Startet Webhook Worker-Pool und Rate Limiting."""
        # Webhook Worker-Pool starten
        webhook_success = await self._start_webhook_pool()

        # Rate Limiting initialisieren
        rate_limiting_success = await self._start_rate_limiting()

        return webhook_success or rate_limiting_success

    @log_service_operation(logger, "webhook_pool", "startup")
    async def _start_webhook_pool(self) -> bool:
        """Startet den Webhook Worker-Pool."""
        try:
            # DI-basierter Webhook Manager
            if not CONTAINER_SERVICES_AVAILABLE:
                raise ImportError("Container services not available")
            webhook_mgr = get_container().resolve(WebhookManagerInterface)
            await webhook_mgr.start_worker_pool()
            return True
        except Exception as exc:
            logger.warning(f"KEI-Webhook Worker-Pool konnte nicht gestartet werden: {exc}")
            return False

    @log_service_operation(logger, "rate_limiting", "initialization")
    async def _start_rate_limiting(self) -> bool:
        """Initialisiert das Rate Limiting System."""
        try:
            from middleware.rate_limiting import initialize_rate_limiting
            await initialize_rate_limiting()
            return True
        except Exception as exc:
            logger.warning(f"Rate Limiting System konnte nicht initialisiert werden: {exc}")
            return False


__all__ = [
    "AgentsInitializer",
    "BusServiceInitializer",
    "DomainRevalidationInitializer",
    "MCPServerInitializer",
    "ServiceInitializer",
    "ServiceManagerInitializer",
    "VoiceSystemInitializer",
    "WebhookAndRateLimitingInitializer",
]
