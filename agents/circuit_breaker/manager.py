"""Agent Circuit Breaker Manager Implementation.
Verwaltet Circuit Breaker für verschiedene Agents und Agent-Typen.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from kei_logging import get_logger

from .circuit_breaker import AgentCircuitBreaker, CircuitBreakerOpenError
from .interfaces import (
    AgentCallContext,
    AgentCircuitBreakerSettings,
    AgentExecutionResult,
    AgentType,
    CircuitBreakerConfig,
    FailureType,
    IAgentCircuitBreakerManager,
    ICacheStrategy,
    ICircuitBreaker,
    IFallbackStrategy,
)
from .strategies import MemoryCacheStrategy, SimpleFallbackStrategy

logger = get_logger(__name__)


class AgentCircuitBreakerManager(IAgentCircuitBreakerManager):
    """Agent Circuit Breaker Manager Implementation.
    Verwaltet Circuit Breaker für verschiedene Agents mit Fallback und Caching.
    """

    def __init__(self, settings: AgentCircuitBreakerSettings):
        self.settings = settings

        # Circuit Breaker Storage
        self._circuit_breakers: dict[str, ICircuitBreaker] = {}
        self._agent_type_configs: dict[AgentType, CircuitBreakerConfig] = {}

        # Strategies
        self._fallback_strategy: IFallbackStrategy = SimpleFallbackStrategy()
        self._cache_strategy: ICacheStrategy = MemoryCacheStrategy()

        # Thread Safety
        self._lock = asyncio.Lock()

        # Monitoring
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._fallback_calls = 0
        self._cached_calls = 0

        # Initialisiere Agent-Type-spezifische Konfigurationen
        self._initialize_agent_type_configs()

        logger.info(f"Agent Circuit Breaker Manager initialisiert - "
                    f"Fallback: {settings.fallback_enabled}, "
                    f"Caching: {settings.caching_enabled}, "
                    f"Monitoring: {settings.monitoring_enabled}")

    def _initialize_agent_type_configs(self) -> None:
        """Initialisiert Agent-Type-spezifische Konfigurationen."""
        # Voice Agent Config
        self._agent_type_configs[AgentType.VOICE_AGENT] = CircuitBreakerConfig(
            failure_threshold=self.settings.voice_agent_failure_threshold,
            timeout_seconds=self.settings.voice_agent_timeout_seconds,
            recovery_timeout_seconds=30,  # Schnelle Recovery für Voice
            success_threshold=2,
            agent_type=AgentType.VOICE_AGENT,
            recovery_strategy=self.settings.recovery_strategy,
            fallback_enabled=True,
            cached_response_enabled=True
        )

        # Tool Agent Config
        self._agent_type_configs[AgentType.TOOL_AGENT] = CircuitBreakerConfig(
            failure_threshold=self.settings.tool_agent_failure_threshold,
            timeout_seconds=self.settings.tool_agent_timeout_seconds,
            recovery_timeout_seconds=60,
            success_threshold=3,
            agent_type=AgentType.TOOL_AGENT,
            recovery_strategy=self.settings.recovery_strategy,
            fallback_enabled=True,
            cached_response_enabled=True
        )

        # Workflow Agent Config
        self._agent_type_configs[AgentType.WORKFLOW_AGENT] = CircuitBreakerConfig(
            failure_threshold=self.settings.workflow_agent_failure_threshold,
            timeout_seconds=self.settings.workflow_agent_timeout_seconds,
            recovery_timeout_seconds=120,  # Längere Recovery für Workflows
            success_threshold=2,
            agent_type=AgentType.WORKFLOW_AGENT,
            recovery_strategy=self.settings.recovery_strategy,
            fallback_enabled=True,
            cached_response_enabled=False  # Workflows nicht cachen
        )

        # Orchestrator Agent Config
        self._agent_type_configs[AgentType.ORCHESTRATOR_AGENT] = CircuitBreakerConfig(
            failure_threshold=self.settings.orchestrator_agent_failure_threshold,
            timeout_seconds=self.settings.orchestrator_agent_timeout_seconds,
            recovery_timeout_seconds=30,  # Schnelle Recovery für Orchestrator
            success_threshold=2,
            agent_type=AgentType.ORCHESTRATOR_AGENT,
            recovery_strategy=self.settings.recovery_strategy,
            fallback_enabled=False,  # Orchestrator hat keine Fallbacks
            cached_response_enabled=False
        )

        # Custom Agent Config (Default)
        self._agent_type_configs[AgentType.CUSTOM_AGENT] = CircuitBreakerConfig(
            failure_threshold=self.settings.default_failure_threshold,
            timeout_seconds=self.settings.default_timeout_seconds,
            recovery_timeout_seconds=self.settings.default_recovery_timeout_seconds,
            success_threshold=self.settings.default_success_threshold,
            agent_type=AgentType.CUSTOM_AGENT,
            recovery_strategy=self.settings.recovery_strategy,
            fallback_enabled=self.settings.fallback_enabled,
            cached_response_enabled=self.settings.caching_enabled
        )

    async def get_circuit_breaker(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.CUSTOM_AGENT
    ) -> ICircuitBreaker:
        """Holt oder erstellt Circuit Breaker für Agent."""
        circuit_breaker_key = f"{agent_type.value}:{agent_id}"

        async with self._lock:
            if circuit_breaker_key not in self._circuit_breakers:
                # Erstelle neuen Circuit Breaker
                config = self._agent_type_configs.get(agent_type, self._agent_type_configs[AgentType.CUSTOM_AGENT])
                config.agent_id = agent_id

                circuit_breaker = AgentCircuitBreaker(
                    name=circuit_breaker_key,
                    config=config
                )

                self._circuit_breakers[circuit_breaker_key] = circuit_breaker
                logger.debug(f"Circuit Breaker erstellt für Agent '{circuit_breaker_key}' "
                            f"(Typ: {agent_type.value}, "
                            f"Failure Threshold: {config.failure_threshold})")

            return self._circuit_breakers[circuit_breaker_key]

    async def execute_agent_call(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Führt Agent-Call mit Circuit Breaker Protection aus."""
        start_time = time.time()
        self._total_calls += 1

        circuit_breaker = await self.get_circuit_breaker(context.agent_id, context.agent_type)

        try:
            # Prüfe Cache zuerst
            cached_result = await self._try_get_cached_response(context, circuit_breaker, start_time)
            if cached_result:
                return cached_result

            # Führe Call mit Circuit Breaker aus
            result = await circuit_breaker.call(func, *args, **kwargs)
            return await self._handle_successful_call(context, circuit_breaker, result, start_time)

        except CircuitBreakerOpenError:
            return await self._handle_circuit_breaker_open(context, func, circuit_breaker, start_time, *args, **kwargs)

        except Exception as e:
            return await self._handle_call_exception(context, func, circuit_breaker, e, start_time, *args, **kwargs)

    async def _try_get_cached_response(
        self,
        context: AgentCallContext,
        circuit_breaker: ICircuitBreaker,
        start_time: float
    ) -> AgentExecutionResult | None:
        """Versucht gecachte Response zu holen."""
        if not (circuit_breaker.config.cached_response_enabled and self.settings.caching_enabled):
            return None

        cached_response = await self.get_cached_response(context)
        if cached_response is None:
            return None

        self._cached_calls += 1
        execution_time_ms = (time.time() - start_time) * 1000

        return AgentExecutionResult(
            success=True,
            result=cached_response,
            execution_time_ms=execution_time_ms,
            fallback_used=False
        )

    async def _handle_successful_call(
        self,
        context: AgentCallContext,
        circuit_breaker: ICircuitBreaker,
        result: Any,
        start_time: float
    ) -> AgentExecutionResult:
        """Behandelt erfolgreichen Call."""
        execution_time_ms = (time.time() - start_time) * 1000
        self._successful_calls += 1

        # Cache Response falls aktiviert
        if circuit_breaker.config.cached_response_enabled and self.settings.caching_enabled:
            try:
                await self.cache_response(context, result, self.settings.cache_ttl_seconds)
            except Exception as cache_error:
                logger.warning(f"Fehler beim Cachen der Response für {context.agent_id}: {cache_error}")

        return AgentExecutionResult(
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            fallback_used=False
        )

    async def _handle_circuit_breaker_open(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        circuit_breaker: ICircuitBreaker,
        start_time: float,
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Behandelt Circuit Breaker Open Error."""
        self._failed_calls += 1

        if circuit_breaker.config.fallback_enabled and self.settings.fallback_enabled:
            return await self._execute_fallback(context, func, *args, **kwargs)

        execution_time_ms = (time.time() - start_time) * 1000
        return AgentExecutionResult(
            success=False,
            error="Circuit breaker open and no fallback available",
            failure_type=FailureType.RESOURCE_UNAVAILABLE,
            execution_time_ms=execution_time_ms,
            fallback_used=False
        )

    async def _handle_call_exception(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        circuit_breaker: ICircuitBreaker,
        exception: Exception,
        start_time: float,
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Behandelt allgemeine Call-Exceptions."""
        self._failed_calls += 1
        execution_time_ms = (time.time() - start_time) * 1000

        if circuit_breaker.config.fallback_enabled and self.settings.fallback_enabled:
            return await self._execute_fallback(context, func, *args, **kwargs)

        return AgentExecutionResult(
            success=False,
            error=str(exception),
            failure_type=self._categorize_error(exception),
            execution_time_ms=execution_time_ms,
            fallback_used=False
        )

    async def _execute_fallback(
        self,
        context: AgentCallContext,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> AgentExecutionResult:
        """Führt Fallback-Execution aus."""
        start_time = time.time()
        self._fallback_calls += 1

        try:
            # Fallback-Agent bestimmen
            fallback_agent_id = await self.get_fallback_agent(context.agent_id, context.agent_type)

            if fallback_agent_id:
                # Erstelle neuen Kontext für Fallback-Agent
                _fallback_context = AgentCallContext(
                    agent_id=fallback_agent_id,
                    agent_type=context.agent_type,
                    framework=context.framework,
                    task=context.task,
                    voice_workflow_id=context.voice_workflow_id,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    request_id=context.request_id,
                    timeout_seconds=self.settings.fallback_timeout_seconds,
                    priority=context.priority
                )

                # Fallback-Agent Circuit Breaker holen
                fallback_circuit_breaker = await self.get_circuit_breaker(
                    fallback_agent_id, context.agent_type
                )

                # Fallback-Call ausführen
                result = await fallback_circuit_breaker.call(func, *args, **kwargs)

                execution_time_ms = (time.time() - start_time) * 1000

                return AgentExecutionResult(
                    success=True,
                    result=result,
                    execution_time_ms=execution_time_ms,
                    fallback_used=True,
                    fallback_agent_id=fallback_agent_id
                )

            # Kein Fallback-Agent verfügbar
            execution_time_ms = (time.time() - start_time) * 1000
            return AgentExecutionResult(
                success=False,
                error="No fallback agent available",
                failure_type=FailureType.RESOURCE_UNAVAILABLE,
                execution_time_ms=execution_time_ms,
                fallback_used=False
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return AgentExecutionResult(
                success=False,
                error=f"Fallback execution failed: {e!s}",
                failure_type=self._categorize_error(e),
                execution_time_ms=execution_time_ms,
                fallback_used=True
            )

    @staticmethod
    def _categorize_error(error: Exception) -> FailureType:
        """Kategorisiert Fehler-Typ.

        Args:
            error: Zu kategorisierender Fehler

        Returns:
            Kategorisierter Fehler-Typ
        """
        error_str = str(error).lower()

        if isinstance(error, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        if "rate limit" in error_str or "too many requests" in error_str:
            return FailureType.RATE_LIMIT_EXCEEDED
        if "authentication" in error_str or "unauthorized" in error_str:
            return FailureType.AUTHENTICATION_FAILURE
        if "validation" in error_str or "invalid" in error_str:
            return FailureType.VALIDATION_ERROR
        if "resource" in error_str or "unavailable" in error_str:
            return FailureType.RESOURCE_UNAVAILABLE
        return FailureType.EXCEPTION

    async def get_fallback_agent(
        self,
        original_agent_id: str,
        agent_type: AgentType
    ) -> str | None:
        """Bestimmt Fallback-Agent für fehlgeschlagenen Agent."""
        # Erstelle einen minimalen Context für die Fallback-Strategie
        context = AgentCallContext(
            agent_id=original_agent_id,
            agent_type=agent_type,
            framework="unknown",
            task="fallback_determination",
            voice_workflow_id=None,
            user_id=None,
            session_id=None,
            request_id=None,
            timeout_seconds=None,
            priority=0,
            tool_name=None,
            tool_parameters=None,
            workflow_step=None,
            step_index=None
        )
        return await self._fallback_strategy.get_fallback_agent(
            original_agent_id, agent_type, context
        )

    async def get_cached_response(
        self,
        context: AgentCallContext
    ) -> Any | None:
        """Holt gecachte Response für Agent-Call."""
        try:
            cache_key = await self._cache_strategy.get_cache_key(context)
            return await self._cache_strategy.get_cached_response(cache_key)
        except Exception as e:
            logger.warning(f"Fehler beim Abrufen der gecachten Response für {context.agent_id}: {e}")
            return None

    async def cache_response(
        self,
        context: AgentCallContext,
        response: Any,
        ttl_seconds: int = 300
    ) -> None:
        """Cached Response für Agent-Call."""
        try:
            cache_key = await self._cache_strategy.get_cache_key(context)
            await self._cache_strategy.cache_response(cache_key, response, ttl_seconds)
        except Exception as e:
            logger.warning(f"Fehler beim Cachen der Response für {context.agent_id}: {e}")
            # Fehler beim Cachen sollte nicht den gesamten Call zum Scheitern bringen

    async def get_all_circuit_breakers(self) -> dict[str, ICircuitBreaker]:
        """Gibt alle Circuit Breaker zurück."""
        async with self._lock:
            return self._circuit_breakers.copy()

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Manager-Statistiken zurück."""
        circuit_breaker_stats = {}

        async with self._lock:
            for name, cb in self._circuit_breakers.items():
                try:
                    circuit_breaker_stats[name] = await cb.get_statistics()
                except Exception as e:
                    logger.warning(f"Fehler beim Abrufen der Statistiken für Circuit Breaker '{name}': {e}")
                    circuit_breaker_stats[name] = {"error": str(e)}

        success_rate = 0.0
        if self._total_calls > 0:
            success_rate = self._successful_calls / self._total_calls

        fallback_rate = 0.0
        if self._total_calls > 0:
            fallback_rate = self._fallback_calls / self._total_calls

        cache_hit_rate = 0.0
        if self._total_calls > 0:
            cache_hit_rate = self._cached_calls / self._total_calls

        return {
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "fallback_calls": self._fallback_calls,
            "cached_calls": self._cached_calls,
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "cache_hit_rate": cache_hit_rate,
            "total_circuit_breakers": len(self._circuit_breakers),
            "circuit_breakers": circuit_breaker_stats,
            "settings": {
                "enabled": self.settings.enabled,
                "fallback_enabled": self.settings.fallback_enabled,
                "caching_enabled": self.settings.caching_enabled,
                "monitoring_enabled": self.settings.monitoring_enabled
            }
        }

    @staticmethod
    def _categorize_error(exception: Exception) -> FailureType:
        """Kategorisiert Exception in FailureType.

        Args:
            exception: Die zu kategorisierende Exception

        Returns:
            Entsprechender FailureType
        """
        if isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        if isinstance(exception, (ConnectionError, OSError)):
            return FailureType.RESOURCE_UNAVAILABLE
        if isinstance(exception, PermissionError):
            return FailureType.AUTHENTICATION_FAILURE
        if (hasattr(exception, "__name__") and "validation" in exception.__class__.__name__.lower()) or isinstance(exception, ValueError):
            return FailureType.VALIDATION_ERROR
        return FailureType.EXCEPTION
