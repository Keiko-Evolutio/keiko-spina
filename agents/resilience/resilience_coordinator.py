# backend/agents/resilience/resilience_coordinator.py
"""Resilience-Coordinator für Personal Assistant

Koordiniert alle Resilience-Komponenten und stellt einheitliche APIs
für Capability-spezifische Resilience-Policies bereit.
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

from .circuit_breaker import CircuitBreakerConfig, CircuitBreakerManager
from .performance_monitor import AlertManager, AlertSeverity, PerformanceMonitor
from .request_budgets import (
    BudgetConfig,
    BudgetManager,
)
from .retry_manager import RetryConfig, RetryManager, RetryStrategy

logger = get_logger(__name__)


@dataclass
class ResiliencePolicy:
    """Resilience-Policy für spezifische Capability."""

    capability: str
    agent_id: str

    # Circuit Breaker-Konfiguration
    circuit_breaker_config: CircuitBreakerConfig | None = None

    # Retry-Konfiguration
    retry_config: RetryConfig | None = None
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    # Budget-Konfiguration
    budget_config: BudgetConfig | None = None

    # Policy-Flags
    circuit_breaker_enabled: bool = True
    retry_enabled: bool = True
    budget_tracking_enabled: bool = True
    performance_monitoring_enabled: bool = True

    # Custom Callbacks
    on_circuit_breaker_trip: Callable[[str, str], Awaitable[None]] | None = None
    on_retry_exhausted: Callable[[str, str, Exception], Awaitable[None]] | None = None
    on_budget_exhausted: Callable[[str, str, str], Awaitable[None]] | None = None
    on_deadline_exceeded: Callable[[str, str], Awaitable[None]] | None = None


@dataclass
class ResilienceConfig:
    """Globale Resilience-Konfiguration."""

    # Default-Konfigurationen
    default_circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )
    default_retry_config: RetryConfig = field(default_factory=RetryConfig)
    default_budget_config: BudgetConfig = field(default_factory=BudgetConfig)

    # Capability-spezifische Policies
    capability_policies: dict[str, ResiliencePolicy] = field(default_factory=dict)

    # Globale Einstellungen
    enable_performance_monitoring: bool = True
    enable_alerting: bool = True
    monitoring_interval: float = 10.0

    # Alert-Konfiguration
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "error_rate": 0.05,
            "response_time": 5.0,
            "circuit_breaker_trips": 3,
            "budget_exhaustions": 5,
        }
    )


class ResilienceCoordinator:
    """Haupt-Coordinator für alle Resilience-Features."""

    def __init__(self, config: ResilienceConfig | None = None):
        """Initialisiert Resilience-Coordinator.

        Args:
            config: Resilience-Konfiguration
        """
        self.config = config or ResilienceConfig()

        # Resilience-Komponenten
        self._circuit_breaker_manager = CircuitBreakerManager(
            self.config.default_circuit_breaker_config
        )
        self._retry_manager = RetryManager(self.config.default_retry_config)
        self._budget_manager = BudgetManager(self.config.default_budget_config)
        self._performance_monitor = PerformanceMonitor()

        # Thread-Safety
        self._lock = threading.RLock()

        # Metrics
        self._metrics_collector = MetricsCollector()

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None

        # Setup-Callbacks
        self._setup_callbacks()

        # Starte Monitoring
        if self.config.enable_performance_monitoring:
            self.start_monitoring()

    async def initialize(self) -> None:
        """Initialisiert den Resilience-Coordinator asynchron."""
        # Bereits im Konstruktor initialisiert, aber für Kompatibilität

    def _setup_callbacks(self):
        """Richtet Callbacks zwischen Komponenten ein."""

        # Circuit Breaker-Callbacks
        async def on_circuit_breaker_state_change(cb_name: str, _old_state, new_state):
            parts = cb_name.split(".")
            if len(parts) >= 2:
                agent_id, capability = parts[0], parts[1]

                await self._performance_monitor.record_circuit_breaker_event(
                    agent_id=agent_id,
                    capability=capability,
                    event_type="trip" if new_state.value == "open" else "recovery",
                )

        self.config.default_circuit_breaker_config.on_state_change = on_circuit_breaker_state_change

        # Budget-Callbacks
        async def on_budget_warning(request_id: str, resource_type: str, _utilization: float):
            # Extrahiere Agent-ID und Capability aus Request-ID (falls möglich)
            budget = self._budget_manager.get_budget(request_id)
            if budget:
                await self._performance_monitor.record_budget_event(
                    agent_id=budget.agent_id,
                    capability=budget.capability,
                    event_type="warning",
                    resource_type=resource_type,
                )

        async def on_budget_exhausted(request_id: str, resource_type: str, _utilization: float):
            budget = self._budget_manager.get_budget(request_id)
            if budget:
                await self._performance_monitor.record_budget_event(
                    agent_id=budget.agent_id,
                    capability=budget.capability,
                    event_type="exhausted",
                    resource_type=resource_type,
                )

        async def on_deadline_exceeded(request_id: str, _exceeded_by: float):
            budget = self._budget_manager.get_budget(request_id)
            if budget:
                await self._performance_monitor.record_budget_event(
                    agent_id=budget.agent_id,
                    capability=budget.capability,
                    event_type="deadline_exceeded",
                )

        self.config.default_budget_config.on_budget_warning = on_budget_warning
        self.config.default_budget_config.on_budget_exhausted = on_budget_exhausted
        self.config.default_budget_config.on_deadline_exceeded = on_deadline_exceeded

    def configure_capability(
        self, agent_id: str, capability: str, policy: ResiliencePolicy | None = None
    ) -> ResiliencePolicy:
        """Konfiguriert Resilience-Policy für Capability.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            policy: Optional spezifische Policy

        Returns:
            Konfigurierte Resilience-Policy
        """
        if policy is None:
            policy = ResiliencePolicy(capability=capability, agent_id=agent_id)

        key = f"{agent_id}.{capability}"

        with self._lock:
            self.config.capability_policies[key] = policy

        # Konfiguriere Komponenten
        if policy.circuit_breaker_enabled and policy.circuit_breaker_config:
            self._circuit_breaker_manager.get_circuit_breaker(
                agent_id, capability, policy.circuit_breaker_config
            )

        if policy.retry_enabled and policy.retry_config:
            upstream_id = f"{agent_id}.{capability}"
            self._retry_manager.configure_upstream(
                upstream_id, policy.retry_strategy, policy.retry_config
            )

        return policy

    def get_policy(self, agent_id: str, capability: str) -> ResiliencePolicy:
        """Holt Resilience-Policy für Capability.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name

        Returns:
            Resilience-Policy
        """
        key = f"{agent_id}.{capability}"

        with self._lock:
            if key not in self.config.capability_policies:
                # Erstelle Standard-Policy
                self.config.capability_policies[key] = ResiliencePolicy(
                    capability=capability, agent_id=agent_id
                )

            return self.config.capability_policies[key]

    @asynccontextmanager
    async def execute_with_resilience(
        self,
        agent_id: str,
        capability: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        request_id: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        """Context Manager für resiliente Ausführung mit allen Features.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            func: Auszuführende Funktion
            *args: Funktions-Argumente
            request_id: Optional Request-ID
            timeout: Optional Timeout
            **kwargs: Funktions-Keyword-Argumente

        Yields:
            Ausführungs-Kontext mit Resilience-Features
        """
        policy = self.get_policy(agent_id, capability)
        start_time = time.time()
        success = False
        _exception = None

        # Request-ID generieren falls nicht vorhanden
        if request_id is None:
            request_id = f"{agent_id}.{capability}.{int(time.time() * 1000)}"

        # Budget-Tracking starten
        budget_context = None
        if policy.budget_tracking_enabled:
            budget_context = self._budget_manager.track_request(
                request_id, capability, agent_id, timeout
            )

        try:
            if budget_context:
                async with budget_context as budget:
                    # Circuit Breaker + Retry + Budget-Tracking
                    result = await self._execute_with_all_features(
                        agent_id, capability, func, budget, *args, **kwargs
                    )
                    success = True
                    yield result
            else:
                # Nur Circuit Breaker + Retry
                result = await self._execute_with_circuit_breaker_and_retry(
                    agent_id, capability, func, *args, **kwargs
                )
                success = True
                yield result

        except Exception as e:
            _exception = e
            success = False
            raise

        finally:
            # Performance-Metriken aufzeichnen
            if policy.performance_monitoring_enabled:
                response_time = time.time() - start_time

                await self._performance_monitor.record_capability_request(
                    agent_id=agent_id,
                    capability=capability,
                    success=success,
                    response_time=response_time,
                )

            # Metrics
            self._metrics_collector.increment_counter(
                "resilience.executions",
                tags={"agent_id": agent_id, "capability": capability, "success": str(success)},
            )

    async def _execute_with_all_features(
        self,
        agent_id: str,
        capability: str,
        func: Callable[..., Awaitable[Any]],
        _budget,
        *args,
        **kwargs,
    ) -> Any:
        """Führt Funktion mit allen Resilience-Features aus."""
        policy = self.get_policy(agent_id, capability)

        # Circuit Breaker
        if policy.circuit_breaker_enabled:
            circuit_breaker = self._circuit_breaker_manager.get_circuit_breaker(
                agent_id, capability
            )

            # Retry mit Circuit Breaker
            if policy.retry_enabled:
                upstream_id = f"{agent_id}.{capability}"

                return await self._retry_manager.execute_with_retry(
                    upstream_id, lambda: circuit_breaker.call(func, *args, **kwargs)
                )
            return await circuit_breaker.call(func, *args, **kwargs)

        # Nur Retry
        if policy.retry_enabled:
            upstream_id = f"{agent_id}.{capability}"
            return await self._retry_manager.execute_with_retry(upstream_id, func, *args, **kwargs)

        # Direkte Ausführung
        return await func(*args, **kwargs)

    async def _execute_with_circuit_breaker_and_retry(
        self, agent_id: str, capability: str, func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """Führt Funktion mit Circuit Breaker und Retry aus."""
        return await self._execute_with_all_features(
            agent_id, capability, func, None, *args, **kwargs
        )

    async def execute_capability(
        self,
        agent_id: str,
        capability: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        request_id: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:
        """Führt Capability mit vollständigen Resilience-Features aus.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            func: Auszuführende Funktion
            *args: Funktions-Argumente
            request_id: Optional Request-ID
            timeout: Optional Timeout
            **kwargs: Funktions-Keyword-Argumente

        Returns:
            Funktions-Ergebnis
        """
        async with self.execute_with_resilience(
            agent_id, capability, func, *args, request_id=request_id, timeout=timeout, **kwargs
        ) as result:
            return result

    def start_monitoring(self):
        """Startet Resilience-Monitoring."""
        if self.config.enable_performance_monitoring:
            self._performance_monitor.start_monitoring()

        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """Stoppt Resilience-Monitoring."""
        self._performance_monitor.stop_monitoring()

        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()

    async def _monitoring_loop(self):
        """Monitoring-Loop für Resilience-Koordination."""
        while True:
            try:
                await asyncio.sleep(self.config.monitoring_interval)
                await self._check_system_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Resilience-Monitoring: {e}")

    async def _check_system_health(self):
        """Prüft System-Health und erstellt Alerts."""
        # Circuit Breaker-Health
        cb_summary = self._circuit_breaker_manager.get_metrics_summary()

        if cb_summary["states"]["open"] > 0:
            await self._performance_monitor.get_alert_manager().create_alert(
                alert_id=f"system_circuit_breakers_open_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                title="Circuit Breakers Open",
                description=f"{cb_summary['states']['open']} Circuit Breaker sind offen",
                metric_name="open_circuit_breakers",
                metric_value=cb_summary["states"]["open"],
            )

        # Budget-Health
        budget_summary = self._budget_manager.get_metrics_summary()

        if budget_summary["active_budgets"] > 100:  # Threshold für zu viele aktive Budgets
            await self._performance_monitor.get_alert_manager().create_alert(
                alert_id=f"high_active_budgets_{int(time.time())}",
                severity=AlertSeverity.INFO,
                title="High Active Budgets",
                description=f"{budget_summary['active_budgets']} aktive Request-Budgets",
                metric_name="active_budgets",
                metric_value=budget_summary["active_budgets"],
            )

    def get_circuit_breaker_manager(self) -> CircuitBreakerManager:
        """Holt Circuit Breaker-Manager."""
        return self._circuit_breaker_manager

    def get_retry_manager(self) -> RetryManager:
        """Holt Retry-Manager."""
        return self._retry_manager

    def get_budget_manager(self) -> BudgetManager:
        """Holt Budget-Manager."""
        return self._budget_manager

    def get_performance_monitor(self) -> PerformanceMonitor:
        """Holt Performance-Monitor."""
        return self._performance_monitor

    def get_alert_manager(self) -> AlertManager:
        """Holt Alert-Manager."""
        return self._performance_monitor.get_alert_manager()

    async def health_check(self) -> dict[str, Any]:
        """Führt vollständigen Health-Check durch.

        Returns:
            Health-Check-Ergebnis
        """
        health_status = {"healthy": True, "timestamp": time.time(), "components": {}}

        # Circuit Breaker-Health
        cb_health = await self._circuit_breaker_manager.health_check()
        health_status["components"]["circuit_breakers"] = cb_health

        if not cb_health["healthy"]:
            health_status["healthy"] = False

        # Budget-Health
        budget_summary = self._budget_manager.get_metrics_summary()
        budget_health = {
            "healthy": budget_summary["active_budgets"] < 1000,  # Threshold
            "active_budgets": budget_summary["active_budgets"],
        }
        health_status["components"]["budgets"] = budget_health

        if not budget_health["healthy"]:
            health_status["healthy"] = False

        # Performance-Monitor-Health
        perf_summary = self._performance_monitor.get_metrics_summary()
        perf_health = {
            "healthy": perf_summary["alerts"]["active_alerts"] < 10,  # Threshold
            "active_alerts": perf_summary["alerts"]["active_alerts"],
        }
        health_status["components"]["performance_monitor"] = perf_health

        if not perf_health["healthy"]:
            health_status["healthy"] = False

        return health_status

    def get_metrics_summary(self) -> dict[str, Any]:
        """Holt vollständige Resilience-Metriken.

        Returns:
            Vollständige Resilience-Metriken
        """
        return {
            "circuit_breakers": self._circuit_breaker_manager.get_metrics_summary(),
            "retries": self._retry_manager.get_metrics_summary(),
            "budgets": self._budget_manager.get_metrics_summary(),
            "performance": self._performance_monitor.get_metrics_summary(),
            "policies": {
                key: {
                    "capability": policy.capability,
                    "agent_id": policy.agent_id,
                    "circuit_breaker_enabled": policy.circuit_breaker_enabled,
                    "retry_enabled": policy.retry_enabled,
                    "budget_tracking_enabled": policy.budget_tracking_enabled,
                    "performance_monitoring_enabled": policy.performance_monitoring_enabled,
                }
                for key, policy in self.config.capability_policies.items()
            },
        }

    async def get_metrics(self) -> dict[str, Any]:
        """Holt Resilience-Metriken (async Wrapper für get_metrics_summary).

        Returns:
            Resilience-Metriken
        """
        return self.get_metrics_summary()

    async def close(self):
        """Schließt Resilience-Coordinator."""
        self.stop_monitoring()

        # Warte auf Monitoring-Task
        if self._monitoring_task and not self._monitoring_task.done():
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Monitoring-Task konnte nicht sauber beendet werden")

        logger.info("Resilience-Coordinator geschlossen")
