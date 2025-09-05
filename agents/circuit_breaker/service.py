"""Agent Circuit Breaker Service Implementation.
Hauptservice für Agent-spezifische Circuit Breaker.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from kei_logging import get_logger

from .interfaces import (
    AgentCallContext,
    AgentCircuitBreakerSettings,
    AgentExecutionResult,
    IAgentCircuitBreakerManager,
    IAgentCircuitBreakerService,
)
from .manager import AgentCircuitBreakerManager
from .strategies import MemoryCacheStrategy

logger = get_logger(__name__)


class AgentCircuitBreakerService(IAgentCircuitBreakerService):
    """Agent Circuit Breaker Service Implementation.
    Orchestriert alle Agent Circuit Breaker Komponenten.
    """

    def __init__(self, settings: AgentCircuitBreakerSettings):
        self.settings = settings

        # Core Components
        self._manager: IAgentCircuitBreakerManager | None = None
        self._cache_strategy: MemoryCacheStrategy | None = None

        # Service Status
        self._initialized = False
        self._running = False

        # Monitoring
        self._start_time: float | None = None

        logger.info(f"Agent circuit breaker service created with settings: {settings}")

    @property
    def manager(self) -> IAgentCircuitBreakerManager:
        """Circuit Breaker Manager."""
        if not self._manager:
            raise RuntimeError("Agent circuit breaker service not initialized")
        return self._manager

    async def initialize(self) -> None:
        """Initialisiert Circuit Breaker Service."""
        if self._initialized:
            return

        try:
            logger.info("Initializing agent circuit breaker service...")

            # Manager initialisieren
            self._manager = AgentCircuitBreakerManager(self.settings)

            # Cache Strategy initialisieren (falls aktiviert)
            if self.settings.caching_enabled:
                self._cache_strategy = MemoryCacheStrategy(
                    max_size=self.settings.cache_max_size,
                    cleanup_interval_seconds=self.settings.cache_cleanup_interval_seconds
                )
                await self._cache_strategy.start_cleanup_task()

            self._initialized = True
            self._running = True
            self._start_time = asyncio.get_event_loop().time()

            logger.info("Agent circuit breaker service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent circuit breaker service: {e}")
            raise

    async def shutdown(self) -> None:
        """Fährt Circuit Breaker Service herunter."""
        if not self._running:
            return

        try:
            logger.info("Shutting down agent circuit breaker service...")

            # Cache Cleanup stoppen
            if self._cache_strategy:
                await self._cache_strategy.stop_cleanup_task()

            self._running = False

            logger.info("Agent circuit breaker service shut down successfully")

        except Exception as e:
            logger.error(f"Error during agent circuit breaker service shutdown: {e}")

    async def execute_agent_with_protection(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Führt Agent-Execution mit vollständiger Protection aus."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        if not self.settings.enabled:
            # Circuit Breaker deaktiviert - führe direkt aus
            try:
                result = await func(*args, **kwargs)
                return AgentExecutionResult(
                    success=True,
                    result=result,
                    execution_time_ms=0.0,
                    fallback_used=False
                )
            except Exception as e:
                return AgentExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=0.0,
                    fallback_used=False
                )

        # Führe mit Circuit Breaker Protection aus
        return await self._manager.execute_agent_call(context, func, *args, **kwargs)

    async def health_check(self) -> dict[str, Any]:
        """Führt Health Check für Circuit Breaker Service durch."""
        health = {
            "healthy": True,
            "details": {}
        }

        # Service-Status prüfen
        if not self._initialized or not self._running:
            health["healthy"] = False
            health["details"]["service"] = "Service not running"
        else:
            health["details"]["service"] = "OK"

        # Manager-Status prüfen
        try:
            if self._manager:
                manager_stats = await self._manager.get_statistics()
                health["details"]["manager"] = "OK"
                health["details"]["total_circuit_breakers"] = manager_stats["total_circuit_breakers"]
                health["details"]["success_rate"] = manager_stats["success_rate"]
            else:
                health["healthy"] = False
                health["details"]["manager"] = "Manager not initialized"
        except Exception as e:
            health["healthy"] = False
            health["details"]["manager"] = f"Manager error: {e!s}"

        # Cache-Status prüfen (falls aktiviert)
        if self.settings.caching_enabled and self._cache_strategy:
            try:
                cache_stats = await self._cache_strategy.get_cache_statistics()
                health["details"]["cache"] = "OK"
                health["details"]["cache_usage"] = f"{cache_stats['usage_percentage']:.1f}%"
            except Exception as e:
                health["details"]["cache"] = f"Cache error: {e!s}"

        return health

    async def get_service_statistics(self) -> dict[str, Any]:
        """Gibt Service-Statistiken zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        stats: dict[str, Any] = {
            "service": {
                "initialized": self._initialized,
                "running": self._running,
                "uptime_seconds": 0,
                "settings": {
                    "enabled": self.settings.enabled,
                    "monitoring_enabled": self.settings.monitoring_enabled,
                    "fallback_enabled": self.settings.fallback_enabled,
                    "caching_enabled": self.settings.caching_enabled
                }
            }
        }

        # Uptime berechnen
        if self._start_time:
            current_time = asyncio.get_event_loop().time()
            stats["service"]["uptime_seconds"] = int(current_time - self._start_time)

        # Manager-Statistiken
        if self._manager:
            try:
                stats["manager"] = await self._manager.get_statistics()
            except Exception as e:
                stats["manager"] = {"error": str(e), "status": "failed"}

        # Cache-Statistiken (falls aktiviert)
        if self.settings.caching_enabled and self._cache_strategy:
            try:
                stats["cache"] = await self._cache_strategy.get_cache_statistics()
            except Exception as e:
                stats["cache"] = {"error": str(e), "status": "failed"}

        return stats

    async def get_circuit_breaker_status(self, agent_id: str, agent_type: str = "custom_agent") -> dict[str, Any]:
        """Gibt Status eines spezifischen Circuit Breakers zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        try:
            from .interfaces import AgentType

            # Parse Agent Type
            try:
                parsed_agent_type = AgentType(agent_type)
            except ValueError:
                parsed_agent_type = AgentType.CUSTOM_AGENT

            # Circuit Breaker holen
            circuit_breaker = await self._manager.get_circuit_breaker(agent_id, parsed_agent_type)

            # Statistiken zurückgeben
            return await circuit_breaker.get_statistics()

        except Exception as e:
            return {"error": str(e)}

    async def force_circuit_breaker_state(
        self,
        agent_id: str,
        agent_type: str,
        state: str,
        reason: str = ""
    ) -> dict[str, Any]:
        """Erzwingt Circuit Breaker State (Admin-Funktion)."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        try:
            from .interfaces import AgentType

            # Parse Agent Type
            try:
                parsed_agent_type = AgentType(agent_type)
            except ValueError:
                parsed_agent_type = AgentType.CUSTOM_AGENT

            # Circuit Breaker holen
            circuit_breaker = await self._manager.get_circuit_breaker(agent_id, parsed_agent_type)

            # State setzen
            if state.lower() == "open":
                await circuit_breaker.force_open(reason)
            elif state.lower() == "closed":
                await circuit_breaker.force_close(reason)
            elif state.lower() == "reset":
                await circuit_breaker.reset()
            else:
                return {"error": f"Invalid state: {state}. Use 'open', 'closed', or 'reset'"}

            return {
                "message": f"Circuit breaker {agent_id} state changed to {state}",
                "agent_id": agent_id,
                "agent_type": agent_type,
                "new_state": state,
                "reason": reason
            }

        except Exception as e:
            return {"error": str(e)}

    async def get_all_circuit_breakers_status(self) -> dict[str, Any]:
        """Gibt Status aller Circuit Breaker zurück."""
        if not self._initialized:
            return {"error": "Service not initialized"}

        try:
            all_circuit_breakers = await self._manager.get_all_circuit_breakers()

            status = {
                "total_count": len(all_circuit_breakers),
                "circuit_breakers": {}
            }

            for name, circuit_breaker in all_circuit_breakers.items():
                status["circuit_breakers"][name] = await circuit_breaker.get_statistics()

            return status

        except Exception as e:
            return {"error": str(e)}


def create_agent_circuit_breaker_service(
    settings: AgentCircuitBreakerSettings | None = None
) -> AgentCircuitBreakerService:
    """Factory-Funktion für Agent Circuit Breaker Service.

    Args:
        settings: Circuit Breaker Settings, falls None werden Defaults verwendet

    Returns:
        Agent Circuit Breaker Service Instance
    """
    if settings is None:
        settings = AgentCircuitBreakerSettings()

    return AgentCircuitBreakerService(settings)
