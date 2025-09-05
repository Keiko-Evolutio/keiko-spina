# backend/kei_agents/slo_sla/config.py
"""Konfiguration für SLO/SLA Management-System.

Definiert SLO/SLA-Konfigurationen mit Integration in die bestehende
ResilienceConfig-Struktur für Enterprise-Grade Service Level Management.
"""

from dataclasses import dataclass, field
from typing import Any

from kei_logging import get_logger

from .models import SLADefinition, SLAType, SLODefinition, SLOType, TimeWindow

logger = get_logger(__name__)


@dataclass
class CapabilitySLOConfig:
    """SLO-Konfiguration für spezifische Capability."""

    capability: str

    # Standard-SLOs für Capability
    latency_p95_ms: float | None = 500.0  # P95 < 500ms
    error_rate_percent: float | None = 1.0  # Error-Rate < 1%
    availability_percent: float | None = 99.9  # 99.9% Availability
    throughput_rps: float | None = None  # Min Throughput (optional)

    # Time-Windows
    latency_time_window: TimeWindow = TimeWindow.FIVE_MINUTES
    error_rate_time_window: TimeWindow = TimeWindow.FIFTEEN_MINUTES
    availability_time_window: TimeWindow = TimeWindow.ONE_HOUR

    # SLO-Spezifische Konfiguration
    target_percentage: float = 99.0  # 99% der Zeit soll SLO erfüllt sein
    grace_period_seconds: float = 60.0
    error_budget_enabled: bool = True
    error_budget_burn_rate_threshold: float = 2.0

    # Alert-Konfiguration
    alert_on_violation: bool = True
    alert_on_burn_rate: bool = True

    def create_slo_definitions(self, agent_id: str | None = None) -> list[SLODefinition]:
        """Erstellt SLO-Definitionen für Capability."""
        slos = []

        # Latency P95 SLO
        if self.latency_p95_ms is not None:
            slos.append(
                SLODefinition(
                    name=f"{self.capability}_latency_p95",
                    slo_type=SLOType.LATENCY_P95,
                    threshold=self.latency_p95_ms / 1000.0,  # Convert to seconds
                    time_window=self.latency_time_window,
                    target_percentage=self.target_percentage,
                    capability=self.capability,
                    agent_id=agent_id,
                    description=f"P95 latency for {self.capability} should be < {self.latency_p95_ms}ms",
                    grace_period_seconds=self.grace_period_seconds,
                    alert_on_violation=self.alert_on_violation,
                    error_budget_enabled=self.error_budget_enabled,
                    error_budget_burn_rate_threshold=self.error_budget_burn_rate_threshold,
                )
            )

        # Error-Rate SLO
        if self.error_rate_percent is not None:
            slos.append(
                SLODefinition(
                    name=f"{self.capability}_error_rate",
                    slo_type=SLOType.ERROR_RATE,
                    threshold=self.error_rate_percent / 100.0,  # Convert to decimal
                    time_window=self.error_rate_time_window,
                    target_percentage=self.target_percentage,
                    capability=self.capability,
                    agent_id=agent_id,
                    description=f"Error rate for {self.capability} should be < {self.error_rate_percent}%",
                    grace_period_seconds=self.grace_period_seconds,
                    alert_on_violation=self.alert_on_violation,
                    error_budget_enabled=self.error_budget_enabled,
                    error_budget_burn_rate_threshold=self.error_budget_burn_rate_threshold,
                )
            )

        # Availability SLO
        if self.availability_percent is not None:
            slos.append(
                SLODefinition(
                    name=f"{self.capability}_availability",
                    slo_type=SLOType.AVAILABILITY,
                    threshold=self.availability_percent / 100.0,  # Convert to decimal
                    time_window=self.availability_time_window,
                    target_percentage=self.target_percentage,
                    capability=self.capability,
                    agent_id=agent_id,
                    description=f"Availability for {self.capability} should be > {self.availability_percent}%",
                    grace_period_seconds=self.grace_period_seconds,
                    alert_on_violation=self.alert_on_violation,
                    error_budget_enabled=self.error_budget_enabled,
                    error_budget_burn_rate_threshold=self.error_budget_burn_rate_threshold,
                )
            )

        # Throughput SLO (optional)
        if self.throughput_rps is not None:
            slos.append(
                SLODefinition(
                    name=f"{self.capability}_throughput",
                    slo_type=SLOType.THROUGHPUT,
                    threshold=self.throughput_rps,
                    time_window=TimeWindow.FIVE_MINUTES,
                    target_percentage=self.target_percentage,
                    capability=self.capability,
                    agent_id=agent_id,
                    description=f"Throughput for {self.capability} should be > {self.throughput_rps} RPS",
                    grace_period_seconds=self.grace_period_seconds,
                    alert_on_violation=self.alert_on_violation,
                    error_budget_enabled=self.error_budget_enabled,
                    error_budget_burn_rate_threshold=self.error_budget_burn_rate_threshold,
                )
            )

        return slos


@dataclass
class CapabilitySLAConfig:
    """SLA-Konfiguration für spezifische Capability."""

    capability: str
    slo_config: CapabilitySLOConfig

    # SLA-Spezifikation
    sla_name: str | None = None
    customer: str | None = None
    priority: str = "medium"
    description: str = ""

    # Penalty-Konfiguration
    penalty_enabled: bool = False
    penalty_threshold: float = 95.0  # SLA-Breach bei < 95% SLO-Erfüllung
    penalty_amount: float = 0.0
    penalty_currency: str = "USD"

    # Escalation-Konfiguration
    escalation_enabled: bool = True
    escalation_delay_minutes: float = 15.0
    escalation_contacts: list[str] = field(default_factory=list)

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)

    def create_sla_definition(self, agent_id: str | None = None) -> SLADefinition:
        """Erstellt SLA-Definition für Capability."""
        slo_definitions = self.slo_config.create_slo_definitions(agent_id)

        return SLADefinition(
            name=self.sla_name or f"{self.capability}_sla",
            sla_type=SLAType.CAPABILITY_SLA,
            slo_definitions=slo_definitions,
            customer=self.customer,
            service=self.capability,
            priority=self.priority,
            description=self.description or f"SLA for {self.capability} capability",
            penalty_enabled=self.penalty_enabled,
            penalty_threshold=self.penalty_threshold,
            penalty_amount=self.penalty_amount,
            penalty_currency=self.penalty_currency,
            escalation_enabled=self.escalation_enabled,
            escalation_delay_minutes=self.escalation_delay_minutes,
            escalation_contacts=self.escalation_contacts,
            tags=self.tags,
        )


@dataclass
class SLOConfig:
    """Globale SLO-Konfiguration."""

    # Default-SLO-Werte
    default_latency_p95_ms: float = 500.0
    default_error_rate_percent: float = 1.0
    default_availability_percent: float = 99.9

    # Default-Time-Windows
    default_latency_time_window: TimeWindow = TimeWindow.FIVE_MINUTES
    default_error_rate_time_window: TimeWindow = TimeWindow.FIFTEEN_MINUTES
    default_availability_time_window: TimeWindow = TimeWindow.ONE_HOUR

    # Default-SLO-Parameter
    default_target_percentage: float = 99.0
    default_grace_period_seconds: float = 60.0
    default_error_budget_enabled: bool = True
    default_error_budget_burn_rate_threshold: float = 2.0

    # Capability-spezifische SLO-Konfigurationen
    capability_slo_configs: dict[str, CapabilitySLOConfig] = field(default_factory=dict)

    # Global-SLO-Einstellungen
    enable_parallel_execution_tracking: bool = True
    parallel_execution_threshold: int = 10  # Min parallel requests für P95-Tracking

    def get_capability_slo_config(self, capability: str) -> CapabilitySLOConfig:
        """Holt oder erstellt SLO-Konfiguration für Capability."""
        if capability not in self.capability_slo_configs:
            # Erstelle Default-Konfiguration
            self.capability_slo_configs[capability] = CapabilitySLOConfig(
                capability=capability,
                latency_p95_ms=self.default_latency_p95_ms,
                error_rate_percent=self.default_error_rate_percent,
                availability_percent=self.default_availability_percent,
                latency_time_window=self.default_latency_time_window,
                error_rate_time_window=self.default_error_rate_time_window,
                availability_time_window=self.default_availability_time_window,
                target_percentage=self.default_target_percentage,
                grace_period_seconds=self.default_grace_period_seconds,
                error_budget_enabled=self.default_error_budget_enabled,
                error_budget_burn_rate_threshold=self.default_error_budget_burn_rate_threshold,
            )

        return self.capability_slo_configs[capability]

    def create_all_slo_definitions(self, agent_id: str | None = None) -> list[SLODefinition]:
        """Erstellt alle SLO-Definitionen für alle konfigurierten Capabilities."""
        all_slos = []

        for capability, config in self.capability_slo_configs.items():
            slos = config.create_slo_definitions(agent_id)
            all_slos.extend(slos)

        return all_slos


@dataclass
class SLAConfig:
    """Globale SLA-Konfiguration."""

    # Default-SLA-Parameter
    default_penalty_enabled: bool = False
    default_penalty_threshold: float = 95.0
    default_penalty_amount: float = 0.0
    default_penalty_currency: str = "USD"

    # Default-Escalation-Parameter
    default_escalation_enabled: bool = True
    default_escalation_delay_minutes: float = 15.0
    default_escalation_contacts: list[str] = field(default_factory=list)

    # Capability-spezifische SLA-Konfigurationen
    capability_sla_configs: dict[str, CapabilitySLAConfig] = field(default_factory=dict)

    # Customer-spezifische SLAs
    customer_slas: dict[str, list[SLADefinition]] = field(default_factory=dict)

    def get_capability_sla_config(
        self, capability: str, slo_config: CapabilitySLOConfig
    ) -> CapabilitySLAConfig:
        """Holt oder erstellt SLA-Konfiguration für Capability."""
        if capability not in self.capability_sla_configs:
            # Erstelle Default-Konfiguration
            self.capability_sla_configs[capability] = CapabilitySLAConfig(
                capability=capability,
                slo_config=slo_config,
                penalty_enabled=self.default_penalty_enabled,
                penalty_threshold=self.default_penalty_threshold,
                penalty_amount=self.default_penalty_amount,
                penalty_currency=self.default_penalty_currency,
                escalation_enabled=self.default_escalation_enabled,
                escalation_delay_minutes=self.default_escalation_delay_minutes,
                escalation_contacts=self.default_escalation_contacts.copy(),
            )

        return self.capability_sla_configs[capability]

    def create_all_sla_definitions(self, agent_id: str | None = None) -> list[SLADefinition]:
        """Erstellt alle SLA-Definitionen für alle konfigurierten Capabilities."""
        all_slas = []

        for capability, config in self.capability_sla_configs.items():
            sla = config.create_sla_definition(agent_id)
            all_slas.append(sla)

        return all_slas


@dataclass
class SLOSLAConfig:
    """Kombinierte SLO/SLA-Konfiguration."""

    # Sub-Konfigurationen
    slo_config: SLOConfig = field(default_factory=SLOConfig)
    sla_config: SLAConfig = field(default_factory=SLAConfig)

    # Global-Einstellungen
    enabled: bool = True
    monitoring_interval_seconds: float = 30.0

    # Alert-Konfiguration
    alert_on_slo_violation: bool = True
    alert_on_sla_breach: bool = True
    alert_on_error_budget_exhaustion: bool = True
    alert_on_burn_rate_critical: bool = True

    # Capacity-Planning-Einstellungen
    enable_capacity_planning: bool = True
    capacity_planning_interval_minutes: float = 60.0
    scaling_recommendation_enabled: bool = True

    # Performance-Budget-Einstellungen
    enable_performance_budgets: bool = True
    performance_budget_window_hours: float = 24.0

    # Integration-Einstellungen
    integrate_with_circuit_breaker: bool = True
    integrate_with_retry_manager: bool = True
    integrate_with_budget_manager: bool = True

    def setup_default_capability_configs(self):
        """Richtet Standard-Capability-Konfigurationen ein."""
        # Chat-Capability: Niedrige Latenz, hohe Availability
        chat_slo_config = CapabilitySLOConfig(
            capability="chat",
            latency_p95_ms=300.0,  # 300ms für Chat
            error_rate_percent=0.5,  # 0.5% für User-facing
            availability_percent=99.95,  # 99.95% für kritische User-Interaktion
            target_percentage=99.5,
            grace_period_seconds=30.0,
        )

        chat_sla_config = CapabilitySLAConfig(
            capability="chat",
            slo_config=chat_slo_config,
            priority="high",
            escalation_delay_minutes=5.0,  # Schnelle Escalation für Chat
        )

        self.slo_config.capability_slo_configs["chat"] = chat_slo_config
        self.sla_config.capability_sla_configs["chat"] = chat_sla_config

        # Search-Capability: Moderate Latenz, hoher Throughput
        search_slo_config = CapabilitySLOConfig(
            capability="search",
            latency_p95_ms=800.0,  # 800ms für Search
            error_rate_percent=1.0,  # 1% für Backend-Service
            availability_percent=99.9,
            throughput_rps=50.0,  # Min 50 RPS
            target_percentage=99.0,
        )

        search_sla_config = CapabilitySLAConfig(
            capability="search",
            slo_config=search_slo_config,
            priority="medium",
            escalation_delay_minutes=15.0,
        )

        self.slo_config.capability_slo_configs["search"] = search_slo_config
        self.sla_config.capability_sla_configs["search"] = search_sla_config

        # Analysis-Capability: Höhere Latenz-Toleranz, CPU-intensiv
        analysis_slo_config = CapabilitySLOConfig(
            capability="analysis",
            latency_p95_ms=2000.0,  # 2s für Analysis
            error_rate_percent=2.0,  # 2% für komplexe Operationen
            availability_percent=99.5,
            target_percentage=98.0,  # Niedrigere Target für CPU-intensive Tasks
            grace_period_seconds=120.0,
        )

        analysis_sla_config = CapabilitySLAConfig(
            capability="analysis",
            slo_config=analysis_slo_config,
            priority="medium",
            escalation_delay_minutes=30.0,
        )

        self.slo_config.capability_slo_configs["analysis"] = analysis_slo_config
        self.sla_config.capability_sla_configs["analysis"] = analysis_sla_config

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "enabled": self.enabled,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "alert_on_slo_violation": self.alert_on_slo_violation,
            "alert_on_sla_breach": self.alert_on_sla_breach,
            "alert_on_error_budget_exhaustion": self.alert_on_error_budget_exhaustion,
            "alert_on_burn_rate_critical": self.alert_on_burn_rate_critical,
            "enable_capacity_planning": self.enable_capacity_planning,
            "capacity_planning_interval_minutes": self.capacity_planning_interval_minutes,
            "scaling_recommendation_enabled": self.scaling_recommendation_enabled,
            "enable_performance_budgets": self.enable_performance_budgets,
            "performance_budget_window_hours": self.performance_budget_window_hours,
            "integrate_with_circuit_breaker": self.integrate_with_circuit_breaker,
            "integrate_with_retry_manager": self.integrate_with_retry_manager,
            "integrate_with_budget_manager": self.integrate_with_budget_manager,
            "slo_config": {
                "default_latency_p95_ms": self.slo_config.default_latency_p95_ms,
                "default_error_rate_percent": self.slo_config.default_error_rate_percent,
                "default_availability_percent": self.slo_config.default_availability_percent,
                "capability_count": len(self.slo_config.capability_slo_configs),
            },
            "sla_config": {
                "default_penalty_enabled": self.sla_config.default_penalty_enabled,
                "default_escalation_enabled": self.sla_config.default_escalation_enabled,
                "capability_count": len(self.sla_config.capability_sla_configs),
            },
        }

    @classmethod
    def create_default_config(cls) -> "SLOSLAConfig":
        """Erstellt Standard-SLO/SLA-Konfiguration."""
        config = cls()
        config.setup_default_capability_configs()
        return config
