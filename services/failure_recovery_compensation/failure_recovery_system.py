# backend/services/failure_recovery_compensation/failure_recovery_system.py
"""Failure Recovery System.

Implementiert Enterprise-Grade Failure Recovery mit automatischer Fehlerbehandlung,
Retry-Mechanismen und Circuit Breaker Patterns für alle kritischen Services.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import SecurityContext

from .data_models import (
    DistributedSystemHealth,
    FailureContext,
    FailureRecoveryMetrics,
    FailureType,
    RecoveryAttempt,
    RecoveryConfiguration,
    RecoveryState,
    RecoveryStrategy,
)

logger = get_logger(__name__)


class CircuitBreakerState:
    """Circuit Breaker State Management."""

    def __init__(self, config: RecoveryConfiguration):
        self.config = config
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.next_attempt_time: datetime | None = None


class FailureRecoverySystem:
    """Enterprise-Grade Failure Recovery System."""

    def __init__(self):
        """Initialisiert Failure Recovery System."""
        # Recovery-Konfigurationen
        self._recovery_configs: dict[str, RecoveryConfiguration] = {}

        # Circuit Breaker States
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}

        # Active Recovery Attempts
        self._active_recoveries: dict[str, RecoveryAttempt] = {}
        self._recovery_history: dict[str, list[RecoveryAttempt]] = defaultdict(list)

        # Failure Tracking
        self._failure_contexts: dict[str, FailureContext] = {}
        self._failure_history: deque = deque(maxlen=10000)

        # Recovery Strategies
        self._recovery_strategies = {
            RecoveryStrategy.IMMEDIATE_RETRY: self._immediate_retry,
            RecoveryStrategy.EXPONENTIAL_BACKOFF: self._exponential_backoff_retry,
            RecoveryStrategy.LINEAR_BACKOFF: self._linear_backoff_retry,
            RecoveryStrategy.FIXED_INTERVAL: self._fixed_interval_retry,
            RecoveryStrategy.FALLBACK_SERVICE: self._fallback_service,
            RecoveryStrategy.CACHED_RESPONSE: self._cached_response,
            RecoveryStrategy.DEFAULT_RESPONSE: self._default_response,
            RecoveryStrategy.DEGRADED_SERVICE: self._degraded_service,
            RecoveryStrategy.CIRCUIT_BREAKER: self._circuit_breaker_recovery,
            RecoveryStrategy.ADAPTIVE_RECOVERY: self._adaptive_recovery
        }

        # System Health
        self._system_health = DistributedSystemHealth(
            system_id="keiko_personal_assistant",
            system_name="Keiko Personal Assistant"
        )

        # Metrics
        self._metrics = FailureRecoveryMetrics(
            system_id="keiko_personal_assistant",
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow()
        )

        # Background Tasks
        self._background_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Event Callbacks
        self._failure_callbacks: list[Callable] = []
        self._recovery_callbacks: list[Callable] = []

        logger.info("Failure Recovery System initialisiert")

    async def start(self) -> None:
        """Startet Failure Recovery System."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Background-Tasks
        self._background_tasks = [
            asyncio.create_task(self._recovery_monitoring_loop()),
            asyncio.create_task(self._circuit_breaker_monitoring_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]

        logger.info("Failure Recovery System gestartet")

    async def stop(self) -> None:
        """Stoppt Failure Recovery System."""
        self._is_running = False

        # Stoppe Background-Tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Failure Recovery System gestoppt")

    async def register_recovery_configuration(
        self,
        config: RecoveryConfiguration
    ) -> None:
        """Registriert Recovery-Konfiguration für Service/Operation.

        Args:
            config: Recovery-Konfiguration
        """
        try:
            config_key = f"{config.service_name}:{config.operation_name}"
            self._recovery_configs[config_key] = config

            # Initialisiere Circuit Breaker
            if RecoveryStrategy.CIRCUIT_BREAKER in [config.primary_strategy] + config.fallback_strategies:
                self._circuit_breakers[config_key] = CircuitBreakerState(config)

            logger.info({
                "event": "recovery_configuration_registered",
                "service_name": config.service_name,
                "operation_name": config.operation_name,
                "primary_strategy": config.primary_strategy.value,
                "fallback_strategies": [s.value for s in config.fallback_strategies]
            })

        except Exception as e:
            logger.error(f"Recovery configuration registration fehlgeschlagen: {e}")
            raise

    async def handle_failure(
        self,
        failure_context: FailureContext,
        security_context: SecurityContext | None = None
    ) -> RecoveryAttempt:
        """Behandelt Failure und startet Recovery-Process.

        Args:
            failure_context: Failure-Context
            security_context: Security-Context

        Returns:
            Recovery-Attempt
        """
        start_time = time.time()

        try:
            # Speichere Failure-Context
            self._failure_contexts[failure_context.failure_id] = failure_context
            self._failure_history.append(failure_context)

            # Update Metrics
            self._metrics.total_failures += 1
            failure_type_key = failure_context.failure_type.value
            self._metrics.failures_by_type[failure_type_key] = (
                self._metrics.failures_by_type.get(failure_type_key, 0) + 1
            )
            self._metrics.failures_by_service[failure_context.service_name] = (
                self._metrics.failures_by_service.get(failure_context.service_name, 0) + 1
            )

            # Hole Recovery-Konfiguration
            config_key = f"{failure_context.service_name}:{failure_context.operation_name}"
            recovery_config = self._recovery_configs.get(config_key)

            if not recovery_config:
                # Erstelle Default-Konfiguration
                recovery_config = self._create_default_recovery_config(
                    failure_context.service_name,
                    failure_context.operation_name
                )
                self._recovery_configs[config_key] = recovery_config

            # Erstelle Recovery-Attempt
            import uuid

            recovery_attempt = RecoveryAttempt(
                attempt_id=str(uuid.uuid4()),
                failure_id=failure_context.failure_id,
                strategy=recovery_config.primary_strategy,
                strategy_config={
                    "max_attempts": recovery_config.max_retry_attempts,
                    "initial_delay_ms": recovery_config.initial_retry_delay_ms,
                    "max_delay_ms": recovery_config.max_retry_delay_ms,
                    "multiplier": recovery_config.retry_multiplier,
                    "jitter": recovery_config.retry_jitter
                }
            )

            # Speichere Recovery-Attempt
            self._active_recoveries[recovery_attempt.attempt_id] = recovery_attempt
            self._recovery_history[failure_context.failure_id].append(recovery_attempt)

            # Starte Recovery-Process
            await self._execute_recovery_strategy(
                recovery_attempt,
                failure_context,
                recovery_config,
                security_context
            )

            # Trigger Failure-Callbacks
            await self._trigger_failure_callbacks(failure_context, recovery_attempt)

            # Update Performance-Metriken
            processing_time_ms = (time.time() - start_time) * 1000

            logger.info({
                "event": "failure_handled",
                "failure_id": failure_context.failure_id,
                "failure_type": failure_context.failure_type.value,
                "service_name": failure_context.service_name,
                "operation_name": failure_context.operation_name,
                "recovery_strategy": recovery_attempt.strategy.value,
                "processing_time_ms": processing_time_ms
            })

            return recovery_attempt

        except Exception as e:
            logger.error(f"Failure handling fehlgeschlagen: {e}")
            raise

    async def execute_with_recovery(
        self,
        operation: Callable,
        service_name: str,
        operation_name: str,
        *args,
        security_context: SecurityContext | None = None,
        **kwargs
    ) -> Any:
        """Führt Operation mit automatischem Recovery aus.

        Args:
            operation: Auszuführende Operation
            service_name: Service-Name
            operation_name: Operation-Name
            *args: Operation-Argumente
            security_context: Security-Context
            **kwargs: Operation-Keyword-Argumente

        Returns:
            Operation-Result
        """
        config_key = f"{service_name}:{operation_name}"
        recovery_config = self._recovery_configs.get(config_key)

        if not recovery_config:
            # Führe Operation ohne Recovery aus
            return await operation(*args, **kwargs)

        # Prüfe Circuit Breaker
        circuit_breaker = self._circuit_breakers.get(config_key)
        if circuit_breaker and circuit_breaker.state == "open":
            if not await self._should_attempt_circuit_breaker_recovery(circuit_breaker):
                raise Exception(f"Circuit breaker open für {service_name}:{operation_name}")

        # Führe Operation mit Recovery aus
        for attempt in range(recovery_config.max_retry_attempts + 1):
            try:
                # Führe Operation aus
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=recovery_config.operation_timeout_ms / 1000.0
                )

                # Update Circuit Breaker bei Erfolg
                if circuit_breaker:
                    await self._record_circuit_breaker_success(circuit_breaker)

                return result

            except Exception as e:
                # Erstelle Failure-Context
                import uuid

                failure_context = FailureContext(
                    failure_id=str(uuid.uuid4()),
                    failure_type=self._classify_failure_type(e),
                    service_name=service_name,
                    operation_name=operation_name,
                    error_message=str(e),
                    occurred_at=datetime.utcnow(),
                    request_id=security_context.request_id if security_context else None,
                    user_id=security_context.user_id if security_context else None,
                    tenant_id=security_context.tenant_id if security_context else None
                )

                # Update Circuit Breaker bei Failure
                if circuit_breaker:
                    await self._record_circuit_breaker_failure(circuit_breaker)

                # Letzter Versuch - werfe Exception
                if attempt >= recovery_config.max_retry_attempts:
                    await self.handle_failure(failure_context, security_context)
                    raise

                # Berechne Retry-Delay
                delay_ms = await self._calculate_retry_delay(
                    recovery_config,
                    attempt
                )

                logger.warning({
                    "event": "operation_retry",
                    "service_name": service_name,
                    "operation_name": operation_name,
                    "attempt": attempt + 1,
                    "max_attempts": recovery_config.max_retry_attempts + 1,
                    "delay_ms": delay_ms,
                    "error": str(e)
                })

                # Warte vor nächstem Versuch
                await asyncio.sleep(delay_ms / 1000.0)

    async def get_system_health(self) -> DistributedSystemHealth:
        """Gibt aktuellen System-Health-Status zurück.

        Returns:
            System-Health-Status
        """
        try:
            # Update System-Health
            await self._update_system_health()

            return self._system_health

        except Exception as e:
            logger.error(f"System health retrieval fehlgeschlagen: {e}")
            raise

    async def get_recovery_metrics(self) -> FailureRecoveryMetrics:
        """Gibt Recovery-Metriken zurück.

        Returns:
            Recovery-Metriken
        """
        try:
            # Update Metrics
            await self._update_recovery_metrics()

            return self._metrics

        except Exception as e:
            logger.error(f"Recovery metrics retrieval fehlgeschlagen: {e}")
            raise

    async def register_failure_callback(self, callback: Callable) -> None:
        """Registriert Failure-Callback.

        Args:
            callback: Callback-Funktion
        """
        self._failure_callbacks.append(callback)

    async def register_recovery_callback(self, callback: Callable) -> None:
        """Registriert Recovery-Callback.

        Args:
            callback: Callback-Funktion
        """
        self._recovery_callbacks.append(callback)

    async def _execute_recovery_strategy(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        security_context: SecurityContext | None
    ) -> None:
        """Führt Recovery-Strategie aus."""
        try:
            recovery_attempt.state = RecoveryState.IN_PROGRESS

            # Hole Recovery-Strategie-Handler
            strategy_handler = self._recovery_strategies.get(recovery_attempt.strategy)

            if not strategy_handler:
                logger.error(f"Unbekannte Recovery-Strategie: {recovery_attempt.strategy}")
                recovery_attempt.state = RecoveryState.FAILED
                recovery_attempt.error_message = f"Unbekannte Recovery-Strategie: {recovery_attempt.strategy}"
                return

            # Führe Recovery-Strategie aus
            success = await strategy_handler(
                recovery_attempt,
                failure_context,
                recovery_config,
                security_context
            )

            # Update Recovery-Attempt
            recovery_attempt.completed_at = datetime.utcnow()
            recovery_attempt.success = success
            recovery_attempt.state = RecoveryState.RECOVERED if success else RecoveryState.FAILED

            if recovery_attempt.started_at:
                recovery_attempt.recovery_time_ms = (
                    recovery_attempt.completed_at - recovery_attempt.started_at
                ).total_seconds() * 1000

            # Update Metrics
            self._metrics.total_recovery_attempts += 1
            if success:
                self._metrics.successful_recoveries += 1
            else:
                self._metrics.failed_recoveries += 1

            # Trigger Recovery-Callbacks
            await self._trigger_recovery_callbacks(recovery_attempt, failure_context)

        except Exception as e:
            logger.error(f"Recovery strategy execution fehlgeschlagen: {e}")
            recovery_attempt.state = RecoveryState.FAILED
            recovery_attempt.error_message = str(e)

    async def _immediate_retry(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        _recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Immediate Retry Recovery-Strategie."""
        try:
            # Simuliere Immediate Retry
            await asyncio.sleep(0.1)  # Minimale Delay

            # Simuliere Recovery-Success basierend auf Failure-Type
            success_probability = self._get_recovery_success_probability(
                failure_context.failure_type,
                RecoveryStrategy.IMMEDIATE_RETRY
            )

            success = random.random() < success_probability

            recovery_attempt.recovery_data = {
                "strategy": "immediate_retry",
                "success_probability": success_probability,
                "simulated": True
            }

            return success

        except Exception as e:
            logger.error(f"Immediate retry fehlgeschlagen: {e}")
            return False

    async def _exponential_backoff_retry(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Exponential Backoff Retry Recovery-Strategie."""
        try:
            # Berechne Exponential Backoff Delay
            delay_ms = min(
                recovery_config.initial_retry_delay_ms * (
                    recovery_config.retry_multiplier ** (recovery_attempt.attempt_number - 1)
                ),
                recovery_config.max_retry_delay_ms
            )

            # Füge Jitter hinzu
            if recovery_config.retry_jitter:
                jitter = random.uniform(0.8, 1.2)
                delay_ms *= jitter

            await asyncio.sleep(delay_ms / 1000.0)

            # Simuliere Recovery-Success
            success_probability = self._get_recovery_success_probability(
                failure_context.failure_type,
                RecoveryStrategy.EXPONENTIAL_BACKOFF
            )

            success = random.random() < success_probability

            recovery_attempt.recovery_data = {
                "strategy": "exponential_backoff",
                "delay_ms": delay_ms,
                "success_probability": success_probability,
                "simulated": True
            }

            return success

        except Exception as e:
            logger.error(f"Exponential backoff retry fehlgeschlagen: {e}")
            return False

    async def _linear_backoff_retry(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Linear Backoff Retry Recovery-Strategie."""
        try:
            # Berechne Linear Backoff Delay
            delay_ms = min(
                recovery_config.initial_retry_delay_ms * recovery_attempt.attempt_number,
                recovery_config.max_retry_delay_ms
            )

            await asyncio.sleep(delay_ms / 1000.0)

            # Simuliere Recovery-Success
            success_probability = self._get_recovery_success_probability(
                failure_context.failure_type,
                RecoveryStrategy.LINEAR_BACKOFF
            )

            success = random.random() < success_probability

            recovery_attempt.recovery_data = {
                "strategy": "linear_backoff",
                "delay_ms": delay_ms,
                "success_probability": success_probability,
                "simulated": True
            }

            return success

        except Exception as e:
            logger.error(f"Linear backoff retry fehlgeschlagen: {e}")
            return False

    async def _fixed_interval_retry(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Fixed Interval Retry Recovery-Strategie."""
        try:
            # Verwende feste Delay
            delay_ms = recovery_config.initial_retry_delay_ms

            await asyncio.sleep(delay_ms / 1000.0)

            # Simuliere Recovery-Success
            success_probability = self._get_recovery_success_probability(
                failure_context.failure_type,
                RecoveryStrategy.FIXED_INTERVAL
            )

            success = random.random() < success_probability

            recovery_attempt.recovery_data = {
                "strategy": "fixed_interval",
                "delay_ms": delay_ms,
                "success_probability": success_probability,
                "simulated": True
            }

            return success

        except Exception as e:
            logger.error(f"Fixed interval retry fehlgeschlagen: {e}")
            return False

    async def _fallback_service(
        self,
        recovery_attempt: RecoveryAttempt,
        _failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Fallback Service Recovery-Strategie."""
        try:
            # Simuliere Fallback Service Call
            if recovery_config.fallback_service_url:
                await asyncio.sleep(0.2)  # Simuliere Fallback-Call

                # Fallback Services haben höhere Success-Rate
                success_probability = 0.9
                success = random.random() < success_probability

                recovery_attempt.recovery_data = {
                    "strategy": "fallback_service",
                    "fallback_url": recovery_config.fallback_service_url,
                    "success_probability": success_probability,
                    "simulated": True
                }

                return success
            return False

        except Exception as e:
            logger.error(f"Fallback service fehlgeschlagen: {e}")
            return False

    async def _cached_response(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Cached Response Recovery-Strategie."""
        try:
            # Simuliere Cache-Lookup
            await asyncio.sleep(0.05)  # Schneller Cache-Access

            # Cache-Success hängt von Operation ab
            cache_hit_probability = 0.7 if "read" in failure_context.operation_name.lower() else 0.3
            success = random.random() < cache_hit_probability

            recovery_attempt.recovery_data = {
                "strategy": "cached_response",
                "cache_hit_probability": cache_hit_probability,
                "cache_ttl_seconds": recovery_config.fallback_cache_ttl_seconds,
                "simulated": True
            }

            return success

        except Exception as e:
            logger.error(f"Cached response fehlgeschlagen: {e}")
            return False

    async def _default_response(
        self,
        recovery_attempt: RecoveryAttempt,
        _failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        _security_context: SecurityContext | None
    ) -> bool:
        """Default Response Recovery-Strategie."""
        try:
            # Default Response ist immer verfügbar
            if recovery_config.default_response:
                recovery_attempt.recovery_data = {
                    "strategy": "default_response",
                    "default_response": recovery_config.default_response,
                    "simulated": True
                }
                return True
            return False

        except Exception as e:
            logger.error(f"Default response fehlgeschlagen: {e}")
            return False

    async def _degraded_service(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        security_context: SecurityContext | None
    ) -> bool:
        """Degraded Service Recovery-Strategie."""
        try:
            # Simuliere Degraded Service mit reduzierter Funktionalität
            await asyncio.sleep(0.1)

            # Degraded Service hat moderate Success-Rate
            success_probability = 0.8
            success = random.random() < success_probability

            recovery_attempt.recovery_data = {
                "strategy": "degraded_service",
                "degraded_mode": True,
                "success_probability": success_probability,
                "simulated": True
            }

            return success

        except Exception as e:
            logger.error(f"Degraded service fehlgeschlagen: {e}")
            return False

    async def _circuit_breaker_recovery(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        security_context: SecurityContext | None
    ) -> bool:
        """Circuit Breaker Recovery-Strategie."""
        try:
            config_key = f"{failure_context.service_name}:{failure_context.operation_name}"
            circuit_breaker = self._circuit_breakers.get(config_key)

            if not circuit_breaker:
                return False

            # Prüfe Circuit Breaker State
            if circuit_breaker.state == "open":
                # Circuit Breaker ist offen - kein Recovery möglich
                recovery_attempt.recovery_data = {
                    "strategy": "circuit_breaker",
                    "circuit_state": "open",
                    "recovery_possible": False,
                    "simulated": True
                }
                return False

            if circuit_breaker.state == "half_open":
                # Half-Open - versuche Recovery
                success_probability = 0.5
                success = random.random() < success_probability

                if success:
                    circuit_breaker.state = "closed"
                    circuit_breaker.failure_count = 0
                else:
                    circuit_breaker.state = "open"
                    circuit_breaker.next_attempt_time = datetime.utcnow() + timedelta(
                        milliseconds=recovery_config.circuit_timeout_ms
                    )

                recovery_attempt.recovery_data = {
                    "strategy": "circuit_breaker",
                    "circuit_state": "half_open",
                    "recovery_success": success,
                    "simulated": True
                }

                return success

            # Closed - normaler Betrieb
            return True

        except Exception as e:
            logger.error(f"Circuit breaker recovery fehlgeschlagen: {e}")
            return False

    async def _adaptive_recovery(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext,
        recovery_config: RecoveryConfiguration,
        security_context: SecurityContext | None
    ) -> bool:
        """Adaptive Recovery-Strategie basierend auf historischen Daten."""
        try:
            # Analysiere historische Recovery-Success-Rates
            service_history = self._recovery_history.get(failure_context.failure_id, [])

            if service_history:
                # Berechne Success-Rate für verschiedene Strategien
                strategy_success_rates = defaultdict(list)

                for attempt in service_history:
                    strategy_success_rates[attempt.strategy].append(attempt.success)

                # Wähle beste Strategie
                best_strategy = None
                best_success_rate = 0.0

                for strategy, successes in strategy_success_rates.items():
                    success_rate = sum(successes) / len(successes)
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_strategy = strategy

                if best_strategy and best_strategy != recovery_attempt.strategy:
                    # Verwende beste Strategie
                    recovery_attempt.strategy = best_strategy

                    # Führe beste Strategie aus
                    strategy_handler = self._recovery_strategies.get(best_strategy)
                    if strategy_handler:
                        return await strategy_handler(
                            recovery_attempt,
                            failure_context,
                            recovery_config,
                            security_context
                        )

            # Fallback zu Exponential Backoff
            return await self._exponential_backoff_retry(
                recovery_attempt,
                failure_context,
                recovery_config,
                security_context
            )

        except Exception as e:
            logger.error(f"Adaptive recovery fehlgeschlagen: {e}")
            return False

    def _create_default_recovery_config(
        self,
        service_name: str,
        operation_name: str
    ) -> RecoveryConfiguration:
        """Erstellt Default-Recovery-Konfiguration."""
        import uuid

        return RecoveryConfiguration(
            config_id=str(uuid.uuid4()),
            service_name=service_name,
            operation_name=operation_name,
            primary_strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            fallback_strategies=[
                RecoveryStrategy.FALLBACK_SERVICE,
                RecoveryStrategy.CACHED_RESPONSE,
                RecoveryStrategy.DEFAULT_RESPONSE
            ],
            max_retry_attempts=3,
            initial_retry_delay_ms=1000,
            max_retry_delay_ms=30000,
            retry_multiplier=2.0,
            retry_jitter=True
        )

    def _classify_failure_type(self, exception: Exception) -> FailureType:
        """Klassifiziert Exception zu Failure-Type."""
        error_message = str(exception).lower()

        if "timeout" in error_message:
            return FailureType.SERVICE_TIMEOUT
        if "connection" in error_message:
            if "refused" in error_message:
                return FailureType.CONNECTION_REFUSED
            return FailureType.NETWORK_UNREACHABLE
        if "authentication" in error_message or "auth" in error_message:
            return FailureType.AUTHENTICATION_FAILED
        if "authorization" in error_message or "permission" in error_message:
            return FailureType.AUTHORIZATION_FAILED
        if "memory" in error_message:
            return FailureType.MEMORY_EXHAUSTED
        if "cpu" in error_message:
            return FailureType.CPU_EXHAUSTED
        if "disk" in error_message:
            return FailureType.DISK_FULL
        if "validation" in error_message:
            return FailureType.DATA_VALIDATION_FAILED
        if "unavailable" in error_message:
            return FailureType.SERVICE_UNAVAILABLE
        return FailureType.UNKNOWN_ERROR

    def _get_recovery_success_probability(
        self,
        failure_type: FailureType,
        recovery_strategy: RecoveryStrategy
    ) -> float:
        """Berechnet Recovery-Success-Wahrscheinlichkeit."""
        # Base Success-Rates per Failure-Type
        base_rates = {
            FailureType.SERVICE_TIMEOUT: 0.7,
            FailureType.NETWORK_TIMEOUT: 0.6,
            FailureType.SERVICE_UNAVAILABLE: 0.4,
            FailureType.CONNECTION_REFUSED: 0.3,
            FailureType.AUTHENTICATION_FAILED: 0.1,
            FailureType.AUTHORIZATION_FAILED: 0.1,
            FailureType.MEMORY_EXHAUSTED: 0.2,
            FailureType.CPU_EXHAUSTED: 0.3,
            FailureType.DATA_VALIDATION_FAILED: 0.8,
            FailureType.UNKNOWN_ERROR: 0.5
        }

        # Strategy-Multipliers
        strategy_multipliers = {
            RecoveryStrategy.IMMEDIATE_RETRY: 0.8,
            RecoveryStrategy.EXPONENTIAL_BACKOFF: 1.0,
            RecoveryStrategy.LINEAR_BACKOFF: 0.9,
            RecoveryStrategy.FIXED_INTERVAL: 0.85,
            RecoveryStrategy.FALLBACK_SERVICE: 1.2,
            RecoveryStrategy.CACHED_RESPONSE: 1.1,
            RecoveryStrategy.DEFAULT_RESPONSE: 1.0,
            RecoveryStrategy.DEGRADED_SERVICE: 1.0,
            RecoveryStrategy.CIRCUIT_BREAKER: 0.9,
            RecoveryStrategy.ADAPTIVE_RECOVERY: 1.1
        }

        base_rate = base_rates.get(failure_type, 0.5)
        multiplier = strategy_multipliers.get(recovery_strategy, 1.0)

        return min(1.0, base_rate * multiplier)

    async def _calculate_retry_delay(
        self,
        config: RecoveryConfiguration,
        attempt: int
    ) -> float:
        """Berechnet Retry-Delay basierend auf Konfiguration."""
        if config.primary_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            delay_ms = min(
                config.initial_retry_delay_ms * (config.retry_multiplier ** attempt),
                config.max_retry_delay_ms
            )
        elif config.primary_strategy == RecoveryStrategy.LINEAR_BACKOFF:
            delay_ms = min(
                config.initial_retry_delay_ms * (attempt + 1),
                config.max_retry_delay_ms
            )
        else:
            delay_ms = config.initial_retry_delay_ms

        # Füge Jitter hinzu
        if config.retry_jitter:
            jitter = random.uniform(0.8, 1.2)
            delay_ms *= jitter

        return delay_ms

    async def _should_attempt_circuit_breaker_recovery(
        self,
        circuit_breaker: CircuitBreakerState
    ) -> bool:
        """Prüft ob Circuit Breaker Recovery versucht werden soll."""
        if circuit_breaker.state != "open":
            return True

        if circuit_breaker.next_attempt_time and datetime.utcnow() >= circuit_breaker.next_attempt_time:
            circuit_breaker.state = "half_open"
            return True

        return False

    async def _record_circuit_breaker_success(
        self,
        circuit_breaker: CircuitBreakerState
    ) -> None:
        """Registriert Circuit Breaker Success."""
        circuit_breaker.success_count += 1

        if circuit_breaker.state == "half_open":
            if circuit_breaker.success_count >= circuit_breaker.config.success_threshold:
                circuit_breaker.state = "closed"
                circuit_breaker.failure_count = 0
                circuit_breaker.success_count = 0

    async def _record_circuit_breaker_failure(
        self,
        circuit_breaker: CircuitBreakerState
    ) -> None:
        """Registriert Circuit Breaker Failure."""
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = datetime.utcnow()
        circuit_breaker.success_count = 0

        if circuit_breaker.failure_count >= circuit_breaker.config.failure_threshold:
            circuit_breaker.state = "open"
            circuit_breaker.next_attempt_time = datetime.utcnow() + timedelta(
                milliseconds=circuit_breaker.config.circuit_timeout_ms
            )

    async def _trigger_failure_callbacks(
        self,
        failure_context: FailureContext,
        recovery_attempt: RecoveryAttempt
    ) -> None:
        """Triggert Failure-Callbacks."""
        for callback in self._failure_callbacks:
            try:
                await callback(failure_context, recovery_attempt)
            except Exception as e:
                logger.error(f"Failure callback fehlgeschlagen: {e}")

    async def _trigger_recovery_callbacks(
        self,
        recovery_attempt: RecoveryAttempt,
        failure_context: FailureContext
    ) -> None:
        """Triggert Recovery-Callbacks."""
        for callback in self._recovery_callbacks:
            try:
                await callback(recovery_attempt, failure_context)
            except Exception as e:
                logger.error(f"Recovery callback fehlgeschlagen: {e}")

    async def _recovery_monitoring_loop(self) -> None:
        """Background-Loop für Recovery-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Alle 30 Sekunden

                if self._is_running:
                    await self._monitor_active_recoveries()

            except Exception as e:
                logger.error(f"Recovery monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(30)

    async def _circuit_breaker_monitoring_loop(self) -> None:
        """Background-Loop für Circuit Breaker-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Jede Minute

                if self._is_running:
                    await self._monitor_circuit_breakers()

            except Exception as e:
                logger.error(f"Circuit breaker monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(60)

    async def _health_monitoring_loop(self) -> None:
        """Background-Loop für Health-Monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(120)  # Alle 2 Minuten

                if self._is_running:
                    await self._update_system_health()

            except Exception as e:
                logger.error(f"Health monitoring loop fehlgeschlagen: {e}")
                await asyncio.sleep(120)

    async def _metrics_collection_loop(self) -> None:
        """Background-Loop für Metrics-Collection."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Alle 5 Minuten

                if self._is_running:
                    await self._update_recovery_metrics()

            except Exception as e:
                logger.error(f"Metrics collection loop fehlgeschlagen: {e}")
                await asyncio.sleep(300)

    async def _monitor_active_recoveries(self) -> None:
        """Monitort aktive Recovery-Attempts."""
        try:
            current_time = datetime.utcnow()

            for attempt_id, recovery_attempt in list(self._active_recoveries.items()):
                # Prüfe Timeout
                if recovery_attempt.state == RecoveryState.IN_PROGRESS:
                    elapsed_time = (current_time - recovery_attempt.started_at).total_seconds() * 1000

                    if elapsed_time > 300000:  # 5 Minuten Timeout
                        recovery_attempt.state = RecoveryState.FAILED
                        recovery_attempt.error_message = "Recovery timeout"
                        recovery_attempt.completed_at = current_time

                # Entferne abgeschlossene Recoveries
                if recovery_attempt.state in [RecoveryState.RECOVERED, RecoveryState.FAILED, RecoveryState.ABANDONED]:
                    del self._active_recoveries[attempt_id]

        except Exception as e:
            logger.error(f"Active recoveries monitoring fehlgeschlagen: {e}")

    async def _monitor_circuit_breakers(self) -> None:
        """Monitort Circuit Breaker States."""
        try:
            current_time = datetime.utcnow()

            for config_key, circuit_breaker in self._circuit_breakers.items():
                # Prüfe Half-Open Timeout
                if (circuit_breaker.state == "open" and
                    circuit_breaker.next_attempt_time and
                    current_time >= circuit_breaker.next_attempt_time):

                    circuit_breaker.state = "half_open"
                    circuit_breaker.success_count = 0

                    logger.info({
                        "event": "circuit_breaker_half_open",
                        "config_key": config_key,
                        "failure_count": circuit_breaker.failure_count
                    })

        except Exception as e:
            logger.error(f"Circuit breakers monitoring fehlgeschlagen: {e}")

    async def _update_system_health(self) -> None:
        """Aktualisiert System-Health-Status."""
        try:
            # Berechne Overall Health basierend auf Failures und Recoveries
            total_failures = len(self._failure_history)
            active_failures = len([f for f in self._failure_contexts.values()
                                 if f.occurred_at > datetime.utcnow() - timedelta(minutes=5)])

            active_recoveries = len([r for r in self._active_recoveries.values()
                                   if r.state == RecoveryState.IN_PROGRESS])

            # Berechne Health-Score
            if total_failures == 0:
                health_score = 1.0
            else:
                recent_failures = len([f for f in self._failure_history
                                     if f.occurred_at > datetime.utcnow() - timedelta(hours=1)])
                health_score = max(0.0, 1.0 - (recent_failures / 100.0))

            # Bestimme Overall Health
            if health_score >= 0.9:
                overall_health = "healthy"
            elif health_score >= 0.7:
                overall_health = "degraded"
            elif health_score >= 0.5:
                overall_health = "unhealthy"
            else:
                overall_health = "critical"

            # Update System-Health
            self._system_health.overall_health = overall_health
            self._system_health.health_score = health_score
            self._system_health.active_failures = [f.failure_id for f in self._failure_contexts.values()
                                                  if f.occurred_at > datetime.utcnow() - timedelta(minutes=5)]
            self._system_health.active_recoveries = list(self._active_recoveries.keys())
            self._system_health.last_updated = datetime.utcnow()

            # Berechne Performance-Metriken
            if self._recovery_history:
                all_attempts = [attempt for attempts in self._recovery_history.values() for attempt in attempts]
                successful_attempts = [a for a in all_attempts if a.success]

                if all_attempts:
                    recovery_times = [a.recovery_time_ms for a in successful_attempts if a.recovery_time_ms]
                    if recovery_times:
                        self._system_health.response_time_p95_ms = sorted(recovery_times)[int(len(recovery_times) * 0.95)]

                if total_failures > 0:
                    self._system_health.error_rate_percent = (len(all_attempts) - len(successful_attempts)) / total_failures * 100

        except Exception as e:
            logger.error(f"System health update fehlgeschlagen: {e}")

    async def _update_recovery_metrics(self) -> None:
        """Aktualisiert Recovery-Metriken."""
        try:
            # Update Period
            self._metrics.period_end = datetime.utcnow()

            # Berechne Recovery-Success-Rate
            if self._metrics.total_recovery_attempts > 0:
                self._metrics.recovery_success_rate = (
                    self._metrics.successful_recoveries / self._metrics.total_recovery_attempts
                )

            # Berechne Average Recovery Time
            if self._recovery_history:
                all_attempts = [attempt for attempts in self._recovery_history.values() for attempt in attempts]
                recovery_times = [a.recovery_time_ms for a in all_attempts if a.recovery_time_ms and a.success]

                if recovery_times:
                    self._metrics.avg_recovery_time_ms = sum(recovery_times) / len(recovery_times)

            # Berechne System-Availability
            total_time_ms = (self._metrics.period_end - self._metrics.period_start).total_seconds() * 1000
            if total_time_ms > 0:
                downtime_ms = sum([
                    (f.occurred_at - self._metrics.period_start).total_seconds() * 1000
                    for f in self._failure_history
                    if f.occurred_at >= self._metrics.period_start
                ])

                self._metrics.system_availability = max(0.0, (total_time_ms - downtime_ms) / total_time_ms * 100)

        except Exception as e:
            logger.error(f"Recovery metrics update fehlgeschlagen: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        try:
            return {
                "failure_recovery_system": {
                    "is_running": self._is_running,
                    "total_failures": self._metrics.total_failures,
                    "total_recovery_attempts": self._metrics.total_recovery_attempts,
                    "recovery_success_rate": self._metrics.recovery_success_rate,
                    "avg_recovery_time_ms": self._metrics.avg_recovery_time_ms,
                    "system_availability": self._metrics.system_availability,
                    "active_recoveries": len(self._active_recoveries),
                    "circuit_breakers": {
                        config_key: {
                            "state": cb.state,
                            "failure_count": cb.failure_count,
                            "success_count": cb.success_count
                        }
                        for config_key, cb in self._circuit_breakers.items()
                    },
                    "system_health": {
                        "overall_health": self._system_health.overall_health,
                        "health_score": self._system_health.health_score,
                        "active_failures": len(self._system_health.active_failures),
                        "active_recoveries": len(self._system_health.active_recoveries)
                    }
                }
            }

        except Exception as e:
            logger.error(f"Performance stats retrieval fehlgeschlagen: {e}")
            return {}
