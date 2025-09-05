# backend/kei_agents/slo_sla/coordinator.py
"""SLO/SLA Coordinator System

Zentrale Orchestrierung aller SLO/SLA-Komponenten mit Integration
in die Resilience-Infrastruktur für Enterprise-Grade
Service Level Management.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from kei_logging import get_logger
from monitoring.custom_metrics import MetricsCollector

from ..resilience.circuit_breaker import CircuitBreakerManager
from ..resilience.performance_monitor import AlertManager, AlertSeverity, PerformanceMonitor
from ..resilience.request_budgets import BudgetManager
from ..resilience.retry_manager import RetryManager
from .breach_manager import EscalationLevel, EscalationWorkflow, SLABreachManager
from .capacity_planner import CapacityPlanner, ScalingRecommendation
from .config import SLOSLAConfig
from .models import SLADefinition, SLODefinition
from .monitor import SLOSLAMonitor

logger = get_logger(__name__)


@dataclass
class SLOSLAPolicy:
    """SLO/SLA-Policy für spezifische Capability."""

    capability: str
    agent_id: str

    # SLO/SLA-Definitionen
    slo_definitions: list[SLODefinition] = field(default_factory=list)
    sla_definitions: list[SLADefinition] = field(default_factory=list)

    # Integration-Flags
    integrate_with_circuit_breaker: bool = True
    integrate_with_retry_manager: bool = True
    integrate_with_budget_manager: bool = True

    # Performance-Budget-Einstellungen
    enable_performance_budgets: bool = True
    enable_capacity_planning: bool = True
    enable_load_shedding: bool = True

    # Alert-Einstellungen
    alert_on_slo_violation: bool = True
    alert_on_sla_breach: bool = True
    alert_on_budget_exhaustion: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "capability": self.capability,
            "agent_id": self.agent_id,
            "slo_definitions": [slo.to_dict() for slo in self.slo_definitions],
            "sla_definitions": [sla.to_dict() for sla in self.sla_definitions],
            "integrate_with_circuit_breaker": self.integrate_with_circuit_breaker,
            "integrate_with_retry_manager": self.integrate_with_retry_manager,
            "integrate_with_budget_manager": self.integrate_with_budget_manager,
            "enable_performance_budgets": self.enable_performance_budgets,
            "enable_capacity_planning": self.enable_capacity_planning,
            "enable_load_shedding": self.enable_load_shedding,
            "alert_on_slo_violation": self.alert_on_slo_violation,
            "alert_on_sla_breach": self.alert_on_sla_breach,
            "alert_on_budget_exhaustion": self.alert_on_budget_exhaustion,
        }


@dataclass
class SLOSLAReport:
    """Umfassender SLO/SLA-Report."""

    report_id: str
    generated_at: float = field(default_factory=time.time)

    # Report-Zeitraum
    start_time: float = 0.0
    end_time: float = 0.0

    # SLO-Summary
    total_slos: int = 0
    slos_compliant: int = 0
    slos_violated: int = 0
    avg_slo_compliance: float = 100.0

    # SLA-Summary
    total_slas: int = 0
    slas_compliant: int = 0
    slas_breached: int = 0
    avg_sla_compliance: float = 100.0

    # Performance-Summary
    avg_response_time_p95: float = 0.0
    avg_error_rate: float = 0.0
    avg_availability: float = 100.0

    # Capacity-Summary
    performance_budgets_exhausted: int = 0
    scaling_recommendations: int = 0
    load_shedding_events: int = 0

    # Detailed-Data
    slo_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    sla_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    breaches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "report_period": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_hours": (self.end_time - self.start_time) / 3600.0,
            },
            "slo_summary": {
                "total_slos": self.total_slos,
                "slos_compliant": self.slos_compliant,
                "slos_violated": self.slos_violated,
                "compliance_rate": (
                    self.slos_compliant / self.total_slos if self.total_slos > 0 else 1.0
                ),
                "avg_compliance": self.avg_slo_compliance,
            },
            "sla_summary": {
                "total_slas": self.total_slas,
                "slas_compliant": self.slas_compliant,
                "slas_breached": self.slas_breached,
                "compliance_rate": (
                    self.slas_compliant / self.total_slas if self.total_slas > 0 else 1.0
                ),
                "avg_compliance": self.avg_sla_compliance,
            },
            "performance_summary": {
                "avg_response_time_p95": self.avg_response_time_p95,
                "avg_error_rate": self.avg_error_rate,
                "avg_availability": self.avg_availability,
            },
            "capacity_summary": {
                "performance_budgets_exhausted": self.performance_budgets_exhausted,
                "scaling_recommendations": self.scaling_recommendations,
                "load_shedding_events": self.load_shedding_events,
            },
            "detailed_data": {
                "slo_metrics": self.slo_metrics,
                "sla_metrics": self.sla_metrics,
                "violations": self.violations,
                "breaches": self.breaches,
            },
        }


class SLOSLACoordinator:
    """Haupt-Coordinator für SLO/SLA Management-System."""

    def __init__(
        self,
        config: SLOSLAConfig,
        performance_monitor: PerformanceMonitor | None = None,
        circuit_breaker_manager: CircuitBreakerManager | None = None,
        retry_manager: RetryManager | None = None,
        budget_manager: BudgetManager | None = None,
    ):
        """Initialisiert SLO/SLA-Coordinator.

        Args:
            config: SLO/SLA-Konfiguration
            performance_monitor: Optional Performance-Monitor
            circuit_breaker_manager: Optional Circuit-Breaker-Manager
            retry_manager: Optional Retry-Manager
            budget_manager: Optional Budget-Manager
        """
        self.config = config

        # Resilience-Integration
        self.performance_monitor = performance_monitor
        self.circuit_breaker_manager = circuit_breaker_manager
        self.retry_manager = retry_manager
        self.budget_manager = budget_manager

        # SLO/SLA-Komponenten
        self.slo_sla_monitor = SLOSLAMonitor(config)
        self.alert_manager = (
            performance_monitor.get_alert_manager() if performance_monitor else AlertManager()
        )
        self.breach_manager = SLABreachManager(self.alert_manager)
        self.capacity_planner = CapacityPlanner(config)

        # Policies
        self.slo_sla_policies: dict[str, SLOSLAPolicy] = {}

        # Thread-Safety
        self._lock = threading.RLock()

        # Coordination-Task
        self._coordination_task: asyncio.Task | None = None

        # Metrics
        self._metrics_collector = MetricsCollector()

        # Setup
        self._setup_integrations()
        self._setup_default_policies()

    async def initialize(self) -> None:
        """Initialisiert den SLO/SLA-Coordinator asynchron."""
        # Bereits im Konstruktor initialisiert, aber für Kompatibilität

    def _setup_integrations(self):
        """Richtet Integrationen mit Resilience-Komponenten ein."""
        # Capacity-Planner-Callbacks
        self.capacity_planner.register_scaling_callback(self._handle_scaling_recommendation)
        self.capacity_planner.register_load_shedding_callback(self._handle_load_shedding)

        # Breach-Manager-Workflows
        self._setup_default_escalation_workflows()

    def _setup_default_escalation_workflows(self):
        """Richtet Standard-Escalation-Workflows ein."""
        # Chat-Capability: Schnelle Escalation
        chat_workflow = EscalationWorkflow(
            sla_name="chat_sla",
            escalation_delays={
                EscalationLevel.LEVEL_0: 0.0,
                EscalationLevel.LEVEL_1: 180.0,  # 3 Minuten
                EscalationLevel.LEVEL_2: 600.0,  # 10 Minuten
                EscalationLevel.LEVEL_3: 1200.0,  # 20 Minuten
            },
        )

        self.breach_manager.register_escalation_workflow("chat_sla", chat_workflow)

        # Search-Capability: Standard-Escalation
        search_workflow = EscalationWorkflow(
            sla_name="search_sla",
            escalation_delays={
                EscalationLevel.LEVEL_0: 0.0,
                EscalationLevel.LEVEL_1: 300.0,  # 5 Minuten
                EscalationLevel.LEVEL_2: 900.0,  # 15 Minuten
                EscalationLevel.LEVEL_3: 1800.0,  # 30 Minuten
            },
        )

        self.breach_manager.register_escalation_workflow("search_sla", search_workflow)

    def _setup_default_policies(self):
        """Richtet Standard-SLO/SLA-Policies ein."""
        # Erstelle Policies für alle konfigurierten Capabilities
        for capability in self.config.slo_config.capability_slo_configs.keys():
            policy = SLOSLAPolicy(
                capability=capability,
                agent_id="*",  # Wildcard für alle Agents
                integrate_with_circuit_breaker=self.config.integrate_with_circuit_breaker,
                integrate_with_retry_manager=self.config.integrate_with_retry_manager,
                integrate_with_budget_manager=self.config.integrate_with_budget_manager,
                enable_performance_budgets=self.config.enable_performance_budgets,
                enable_capacity_planning=self.config.enable_capacity_planning,
                alert_on_slo_violation=self.config.alert_on_slo_violation,
                alert_on_sla_breach=self.config.alert_on_sla_breach,
                alert_on_budget_exhaustion=self.config.alert_on_error_budget_exhaustion,
            )

            # Erstelle SLO/SLA-Definitionen
            slo_config = self.config.slo_config.get_capability_slo_config(capability)
            policy.slo_definitions = slo_config.create_slo_definitions()

            if capability in self.config.sla_config.capability_sla_configs:
                sla_config = self.config.sla_config.get_capability_sla_config(
                    capability, slo_config
                )
                policy.sla_definitions = [sla_config.create_sla_definition()]

            self.slo_sla_policies[capability] = policy

    def configure_capability_policy(
        self, agent_id: str, capability: str, policy: SLOSLAPolicy | None = None
    ) -> SLOSLAPolicy:
        """Konfiguriert SLO/SLA-Policy für Capability.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            policy: Optional spezifische Policy

        Returns:
            Konfigurierte SLO/SLA-Policy
        """
        if policy is None:
            # Erstelle Policy basierend auf Template
            template_policy = self.slo_sla_policies.get(capability)
            if template_policy:
                policy = SLOSLAPolicy(
                    capability=capability,
                    agent_id=agent_id,
                    slo_definitions=[
                        SLODefinition(
                            name=slo.name,
                            slo_type=slo.slo_type,
                            threshold=slo.threshold,
                            time_window=slo.time_window,
                            target_percentage=slo.target_percentage,
                            capability=capability,
                            agent_id=agent_id,
                            description=slo.description,
                            tags=slo.tags,
                            grace_period_seconds=slo.grace_period_seconds,
                            alert_on_violation=slo.alert_on_violation,
                            error_budget_enabled=slo.error_budget_enabled,
                            error_budget_burn_rate_threshold=slo.error_budget_burn_rate_threshold,
                        )
                        for slo in template_policy.slo_definitions
                    ],
                    sla_definitions=[
                        SLADefinition(
                            name=sla.name.replace("*", agent_id),
                            sla_type=sla.sla_type,
                            slo_definitions=sla.slo_definitions,
                            customer=sla.customer,
                            service=capability,
                            priority=sla.priority,
                            description=sla.description,
                            penalty_enabled=sla.penalty_enabled,
                            penalty_threshold=sla.penalty_threshold,
                            escalation_enabled=sla.escalation_enabled,
                            escalation_delay_minutes=sla.escalation_delay_minutes,
                            tags=sla.tags,
                        )
                        for sla in template_policy.sla_definitions
                    ],
                    integrate_with_circuit_breaker=template_policy.integrate_with_circuit_breaker,
                    integrate_with_retry_manager=template_policy.integrate_with_retry_manager,
                    integrate_with_budget_manager=template_policy.integrate_with_budget_manager,
                    enable_performance_budgets=template_policy.enable_performance_budgets,
                    enable_capacity_planning=template_policy.enable_capacity_planning,
                    enable_load_shedding=template_policy.enable_load_shedding,
                    alert_on_slo_violation=template_policy.alert_on_slo_violation,
                    alert_on_sla_breach=template_policy.alert_on_sla_breach,
                    alert_on_budget_exhaustion=template_policy.alert_on_budget_exhaustion,
                )
            else:
                # Fallback: Erstelle minimale Policy
                policy = SLOSLAPolicy(capability=capability, agent_id=agent_id)

        key = f"{agent_id}.{capability}"

        with self._lock:
            self.slo_sla_policies[key] = policy

        # Erstelle Performance-Budget
        if policy.enable_performance_budgets:
            self.capacity_planner.create_performance_budget(capability, agent_id)

        return policy

    async def record_capability_execution(
        self,
        agent_id: str,
        capability: str,
        response_time: float,
        success: bool,
        parallel_requests: int = 1,
        **additional_metrics,
    ):
        """Zeichnet Capability-Ausführung für SLO/SLA-Tracking auf.

        Args:
            agent_id: Agent-ID
            capability: Capability-Name
            response_time: Response-Zeit in Sekunden
            success: Ob Ausführung erfolgreich war
            parallel_requests: Anzahl paralleler Requests
            **additional_metrics: Zusätzliche Metriken
        """
        # SLO/SLA-Monitor aktualisieren
        await self.slo_sla_monitor.record_capability_request(
            agent_id, capability, response_time, success, parallel_requests
        )

        # Performance-Budget aktualisieren
        if self.config.enable_performance_budgets:
            budget_metrics = {
                "response_time_p95": response_time,  # Vereinfacht - normalerweise P95 berechnet
                "error_rate": 0.0 if success else 1.0,
                "availability": 1.0 if success else 0.0,
                **additional_metrics,
            }

            self.capacity_planner.update_performance_budget(capability, agent_id, **budget_metrics)

        # Integration mit Resilience-Komponenten
        policy = self.slo_sla_policies.get(f"{agent_id}.{capability}")
        if policy:
            await self._integrate_with_resilience_components(
                agent_id, capability, response_time, success, policy
            )

    async def _integrate_with_resilience_components(
        self,
        agent_id: str,
        capability: str,
        response_time: float,
        success: bool,
        policy: SLOSLAPolicy,
    ):
        """Integriert mit Resilience-Komponenten."""
        # Performance-Monitor-Integration
        if self.performance_monitor and policy.alert_on_slo_violation:
            await self.performance_monitor.record_capability_request(
                agent_id=agent_id,
                capability=capability,
                success=success,
                response_time=response_time,
            )

        # Circuit-Breaker-Integration
        if self.circuit_breaker_manager and policy.integrate_with_circuit_breaker and not success:

            # Circuit-Breaker-Event für Fehler
            _circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(agent_id, capability)
            # Circuit-Breaker wird automatisch durch Resilience-Coordinator verwaltet

        # Budget-Manager-Integration
        if (
            self.budget_manager
            and policy.integrate_with_budget_manager
            and policy.alert_on_budget_exhaustion
        ):

            # Budget-Events werden automatisch durch Enhanced-Operations verwaltet
            pass

    async def _handle_scaling_recommendation(self, recommendation: ScalingRecommendation):
        """Behandelt Scaling-Empfehlung."""
        logger.info(
            f"Scaling-Empfehlung erhalten: {recommendation.capability} - "
            f"{recommendation.direction.value} (Confidence: {recommendation.confidence:.1%})"
        )

        # Metrics
        self._metrics_collector.increment_counter(
            "slo_sla.scaling_recommendations",
            tags={
                "capability": recommendation.capability,
                "agent_id": recommendation.agent_id or "global",
                "direction": recommendation.direction.value,
                "urgency": recommendation.urgency,
            },
        )

        # Alert für kritische Scaling-Empfehlungen
        if recommendation.urgency in ["high", "critical"]:
            await self.alert_manager.create_alert(
                alert_id=f"scaling_recommendation_{recommendation.capability}_{int(time.time())}",
                severity=(
                    AlertSeverity.WARNING
                    if recommendation.urgency == "high"
                    else AlertSeverity.ERROR
                ),
                title=f"Scaling Recommendation: {recommendation.capability}",
                description=f"{recommendation.primary_reason} - {recommendation.direction.value}",
                metric_name="scaling_recommendation",
                metric_value=recommendation.confidence,
                tags={
                    "capability": recommendation.capability,
                    "direction": recommendation.direction.value,
                    "urgency": recommendation.urgency,
                },
            )

    async def _handle_load_shedding(self, budget_key: str, active: bool):
        """Behandelt Load-Shedding-Event."""
        parts = budget_key.split(".", 1)
        agent_id = parts[0] if parts[0] != "global" else None
        capability = parts[1] if len(parts) > 1 else "unknown"

        logger.warning(
            f"Load-Shedding {'aktiviert' if active else 'deaktiviert'} für "
            f"{agent_id}.{capability}"
        )

        # Metrics
        self._metrics_collector.increment_counter(
            "slo_sla.load_shedding_events",
            tags={
                "capability": capability,
                "agent_id": agent_id or "global",
                "action": "activated" if active else "deactivated",
            },
        )

        # Integration mit Circuit-Breaker für proaktives Load-Shedding
        if active and self.circuit_breaker_manager and self.config.integrate_with_circuit_breaker:

            # Temporär Circuit-Breaker-Threshold reduzieren
            _circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(
                agent_id or "global", capability
            )
            # Implementation würde Circuit-Breaker-Konfiguration anpassen

    def start_coordination(self):
        """Startet SLO/SLA-Coordination."""
        # Starte alle Komponenten
        self.slo_sla_monitor.start_monitoring()

        if self.config.enable_capacity_planning:
            self.capacity_planner.start_capacity_planning()

        # Starte Coordination-Task
        if self._coordination_task is None or self._coordination_task.done():
            self._coordination_task = asyncio.create_task(self._coordination_loop())

    def stop_coordination(self):
        """Stoppt SLO/SLA-Coordination."""
        # Stoppe alle Komponenten
        self.slo_sla_monitor.stop_monitoring()
        self.capacity_planner.stop_capacity_planning()

        # Stoppe Coordination-Task
        if self._coordination_task and not self._coordination_task.done():
            self._coordination_task.cancel()

    async def _coordination_loop(self):
        """Coordination-Loop für SLO/SLA-Management."""
        while True:
            try:
                await asyncio.sleep(self.config.monitoring_interval_seconds)

                # SLA-Breach-Checks
                await self._check_sla_breaches()

                # Performance-Budget-Checks
                await self._check_performance_budgets()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler in SLO/SLA-Coordination: {e}")

    async def _check_sla_breaches(self):
        """Prüft SLA-Breaches."""
        sla_metrics = self.slo_sla_monitor.get_sla_metrics_summary()

        for sla_key, metrics_data in sla_metrics.get("sla_metrics", {}).items():
            # Rekonstruiere SLA-Metriken-Objekt (vereinfacht)
            # In echter Implementation würde hier das vollständige Objekt verwendet
            if metrics_data.get("is_breached", False):
                # SLA-Breach-Handling würde hier implementiert
                pass

    async def _check_performance_budgets(self):
        """Prüft Performance-Budgets."""
        budgets = self.capacity_planner.get_performance_budgets()

        for budget_key, budget in budgets.items():
            if budget.budget_exhausted:
                logger.warning(
                    f"Performance-Budget erschöpft für {budget_key}: {budget.exhausted_metrics}"
                )

                # Alert erstellen
                await self.alert_manager.create_alert(
                    alert_id=f"budget_exhausted_{budget_key}_{int(time.time())}",
                    severity=AlertSeverity.WARNING,
                    title=f"Performance Budget Exhausted: {budget.capability}",
                    description=f"Budget exhausted for metrics: {', '.join(budget.exhausted_metrics)}",
                    metric_name="budget_exhaustion",
                    metric_value=len(budget.exhausted_metrics),
                    tags={"capability": budget.capability, "agent_id": budget.agent_id or "global"},
                )

    async def generate_slo_sla_report(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> SLOSLAReport:
        """Generiert umfassenden SLO/SLA-Report.

        Args:
            start_time: Start-Zeit für Report (Unix-Timestamp)
            end_time: End-Zeit für Report (Unix-Timestamp)

        Returns:
            SLO/SLA-Report
        """
        if end_time is None:
            end_time = time.time()
        if start_time is None:
            start_time = end_time - 86400.0  # Letzte 24 Stunden

        report = SLOSLAReport(
            report_id=f"slo_sla_report_{int(end_time)}", start_time=start_time, end_time=end_time
        )

        # SLO-Metriken sammeln
        slo_summary = self.slo_sla_monitor.get_slo_metrics_summary()
        report.slo_metrics = slo_summary.get("slo_metrics", {})
        report.total_slos = len(report.slo_metrics)

        # SLA-Metriken sammeln
        sla_summary = self.slo_sla_monitor.get_sla_metrics_summary()
        report.sla_metrics = sla_summary.get("sla_metrics", {})
        report.total_slas = len(report.sla_metrics)

        # Capacity-Metriken sammeln
        capacity_summary = self.capacity_planner.get_metrics_summary()
        report.performance_budgets_exhausted = capacity_summary.get("budgets_exhausted", 0)
        report.scaling_recommendations = capacity_summary.get("scaling_recommendations", 0)
        report.load_shedding_events = capacity_summary.get("active_load_shedding", 0)

        # Berechne Durchschnittswerte
        if report.slo_metrics:
            compliances = [
                metrics.get("compliance_percentage", 100.0)
                for metrics in report.slo_metrics.values()
            ]
            report.avg_slo_compliance = sum(compliances) / len(compliances)
            report.slos_compliant = sum(1 for c in compliances if c >= 99.0)
            report.slos_violated = report.total_slos - report.slos_compliant

        if report.sla_metrics:
            compliances = [
                metrics.get("current_compliance", 100.0) for metrics in report.sla_metrics.values()
            ]
            report.avg_sla_compliance = sum(compliances) / len(compliances)
            report.slas_compliant = sum(1 for c in compliances if c >= 95.0)
            report.slas_breached = report.total_slas - report.slas_compliant

        return report

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Holt umfassenden SLO/SLA-Status.

        Returns:
            Umfassender Status
        """
        return {
            "config": self.config.to_dict(),
            "policies": {key: policy.to_dict() for key, policy in self.slo_sla_policies.items()},
            "slo_sla_monitor": self.slo_sla_monitor.get_comprehensive_summary(),
            "breach_manager": self.breach_manager.get_metrics_summary(),
            "capacity_planner": self.capacity_planner.get_metrics_summary(),
            "coordination_active": (
                self._coordination_task is not None and not self._coordination_task.done()
            ),
        }

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check für SLO/SLA-System durch.

        Returns:
            Health-Check-Ergebnis
        """
        health_status = {"healthy": True, "timestamp": time.time(), "components": {}}

        # SLO/SLA-Monitor-Health
        monitor_summary = self.slo_sla_monitor.get_comprehensive_summary()
        monitor_health = {
            "healthy": monitor_summary.get("monitoring_active", False),
            "total_slos": monitor_summary.get("slo_summary", {}).get("total_slos", 0),
            "total_slas": monitor_summary.get("sla_summary", {}).get("total_slas", 0),
        }
        health_status["components"]["slo_sla_monitor"] = monitor_health

        # Breach-Manager-Health
        breach_summary = self.breach_manager.get_metrics_summary()
        breach_health = {
            "healthy": breach_summary.get("active_breaches", 0) == 0,
            "active_breaches": breach_summary.get("active_breaches", 0),
        }
        health_status["components"]["breach_manager"] = breach_health

        if breach_health["active_breaches"] > 0:
            health_status["healthy"] = False

        # Capacity-Planner-Health
        capacity_summary = self.capacity_planner.get_metrics_summary()
        capacity_health = {
            "healthy": capacity_summary.get("budgets_exhausted", 0) == 0,
            "budgets_exhausted": capacity_summary.get("budgets_exhausted", 0),
            "active_load_shedding": capacity_summary.get("active_load_shedding", 0),
        }
        health_status["components"]["capacity_planner"] = capacity_health

        if capacity_health["budgets_exhausted"] > 0:
            health_status["healthy"] = False

        return health_status

    async def get_metrics(self) -> dict[str, Any]:
        """Holt SLO/SLA-Metriken.

        Returns:
            SLO/SLA-Metriken
        """
        metrics = {
            "slo_metrics": {},
            "sla_metrics": {},
            "breach_metrics": {},
            "capacity_metrics": {}
        }

        # SLO-Metriken sammeln
        if self.slo_sla_monitor:
            slo_summary = self.slo_sla_monitor.get_slo_metrics_summary()
            metrics["slo_metrics"] = slo_summary

        # Breach-Metriken sammeln
        if self.breach_manager:
            breach_history = self.breach_manager.get_breach_history()
            metrics["breach_metrics"] = {"breach_history": breach_history}

        # Capacity-Metriken sammeln
        if self.capacity_planner:
            capacity_summary = self.capacity_planner.get_metrics_summary()
            metrics["capacity_metrics"] = capacity_summary

        return metrics

    async def close(self):
        """Schließt SLO/SLA-Coordinator."""
        self.stop_coordination()

        # Warte auf Coordination-Task
        if self._coordination_task and not self._coordination_task.done():
            try:
                await asyncio.wait_for(self._coordination_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Coordination-Task konnte nicht sauber beendet werden")

        logger.info("SLO/SLA-Coordinator geschlossen")
