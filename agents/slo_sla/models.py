# backend/agents/slo_sla/models.py
"""Core Models für SLO/SLA Management-System.

Definiert SLO/SLA-Definitionen, Metriken, Violations und Breach-Tracking
für Enterprise-Grade Service Level Management.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger

from .constants import (
    DEFAULT_ERROR_BUDGET_BURN_RATE_THRESHOLD,
    DEFAULT_GRACE_PERIOD_SECONDS,
    DEFAULT_SLA_TARGET_PERCENTAGE,
    DEFAULT_SLO_TARGET_PERCENTAGE,
    MEASUREMENTS_HISTORY_LIMIT,
)
from .utils import create_limited_history, create_standardized_to_dict, get_current_timestamp

logger = get_logger(__name__)


class SLOType(str, Enum):
    """Typen von Service Level Objectives."""

    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    LATENCY_P50 = "latency_p50"
    AVAILABILITY = "availability"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    CUSTOM = "custom"


class SLAType(str, Enum):
    """Typen von Service Level Agreements."""

    CAPABILITY_SLA = "capability_sla"
    SERVICE_SLA = "service_sla"
    CUSTOMER_SLA = "customer_sla"
    INTERNAL_SLA = "internal_sla"
    EXTERNAL_SLA = "external_sla"


class TimeWindow(str, Enum):
    """Zeitfenster für SLO/SLA-Bewertung."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    SIX_HOURS = "6h"
    TWENTY_FOUR_HOURS = "24h"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class ViolationSeverity(str, Enum):
    """Schweregrade für SLO/SLA-Violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SLODefinition:
    """Definition eines Service Level Objective."""

    name: str
    slo_type: SLOType
    threshold: float
    time_window: TimeWindow

    # Target-Spezifikation
    target_percentage: float = DEFAULT_SLO_TARGET_PERCENTAGE

    # Capability-Kontext
    capability: str | None = None
    agent_id: str | None = None

    # Erweiterte Konfiguration
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    # Violation-Handling
    grace_period_seconds: float = DEFAULT_GRACE_PERIOD_SECONDS
    alert_on_violation: bool = True

    # Error-Budget-Konfiguration
    error_budget_enabled: bool = True
    error_budget_burn_rate_threshold: float = DEFAULT_ERROR_BUDGET_BURN_RATE_THRESHOLD

    def get_time_window_seconds(self) -> float:
        """Konvertiert TimeWindow zu Sekunden."""
        window_map = {
            TimeWindow.ONE_MINUTE: 60,
            TimeWindow.FIVE_MINUTES: 300,
            TimeWindow.FIFTEEN_MINUTES: 900,
            TimeWindow.ONE_HOUR: 3600,
            TimeWindow.SIX_HOURS: 21600,
            TimeWindow.TWENTY_FOUR_HOURS: 86400,
            TimeWindow.ONE_WEEK: 604800,
            TimeWindow.ONE_MONTH: 2592000,
        }
        return window_map.get(self.time_window, 3600)

    def get_error_budget(self) -> float:
        """Berechnet verfügbares Error-Budget (1 - target_percentage)."""
        return (100.0 - self.target_percentage) / 100.0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return create_standardized_to_dict(self, {
            "error_budget": self.get_error_budget(),
            "time_window_seconds": self.get_time_window_seconds(),
        })


@dataclass
class SLADefinition:
    """Definition eines Service Level Agreement."""

    name: str
    sla_type: SLAType
    slo_definitions: list[SLODefinition]

    # Business-Kontext
    customer: str | None = None
    service: str | None = None
    priority: str = "medium"

    # SLA-Spezifikation
    description: str = ""
    effective_date: float | None = None
    expiry_date: float | None = None

    # Penalty-Konfiguration
    penalty_enabled: bool = False
    penalty_threshold: float = DEFAULT_SLA_TARGET_PERCENTAGE
    penalty_amount: float = 0.0
    penalty_currency: str = "USD"

    # Escalation-Konfiguration
    escalation_enabled: bool = True
    escalation_delay_minutes: float = 15.0
    escalation_contacts: list[str] = field(default_factory=list)

    # Metadata
    tags: dict[str, str] = field(default_factory=dict)

    def get_combined_slo_score(self, slo_metrics: dict[str, "SLOMetrics"]) -> float:
        """Berechnet kombinierte SLO-Score für SLA-Bewertung."""
        if not self.slo_definitions:
            return 1.0

        total_score = 0.0
        valid_slos = 0

        for slo_def in self.slo_definitions:
            slo_key = (
                f"{slo_def.agent_id or 'global'}.{slo_def.capability or 'global'}.{slo_def.name}"
            )

            if slo_key in slo_metrics:
                metrics = slo_metrics[slo_key]
                total_score += metrics.compliance_percentage
                valid_slos += 1

        return total_score / valid_slos if valid_slos > 0 else 1.0

    def is_sla_breached(self, slo_metrics: dict[str, "SLOMetrics"]) -> bool:
        """Prüft ob SLA gebrochen ist."""
        combined_score = self.get_combined_slo_score(slo_metrics)
        return combined_score < self.penalty_threshold

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return create_standardized_to_dict(self, {
            "slo_definitions": [slo.to_dict() for slo in self.slo_definitions],
        })


@dataclass
class SLOMetrics:
    """Metriken für SLO-Tracking."""

    slo_definition: SLODefinition

    # Aktuelle Metriken
    current_value: float = 0.0
    compliance_percentage: float = 100.0

    # Historical-Data
    measurements: deque = field(default_factory=lambda: create_limited_history(MEASUREMENTS_HISTORY_LIMIT))
    violations: list["SLOViolation"] = field(default_factory=list)

    # Error-Budget-Tracking
    error_budget_consumed: float = 0.0
    error_budget_burn_rate: float = 0.0

    # Timing
    last_measurement_time: float = field(default_factory=get_current_timestamp)
    last_violation_time: float | None = None

    # Statistics
    total_measurements: int = 0
    violation_count: int = 0

    def add_measurement(self, value: float, timestamp: float | None = None):
        """Fügt neue Messung hinzu."""
        if timestamp is None:
            timestamp = time.time()

        measurement = {
            "value": value,
            "timestamp": timestamp,
            "compliant": self._is_compliant(value),
        }

        self.measurements.append(measurement)
        self.current_value = value
        self.last_measurement_time = timestamp
        self.total_measurements += 1

        # Aktualisiere Compliance-Percentage
        self._update_compliance_percentage()

        # Aktualisiere Error-Budget
        self._update_error_budget()

        # Prüfe Violation
        if not measurement["compliant"]:
            self._record_violation(value, timestamp)

    def is_compliant(self, value: float) -> bool:
        """Public API für Compliance-Prüfung.

        Args:
            value: Zu prüfender Wert

        Returns:
            True wenn Wert SLO-compliant ist
        """
        return self._is_compliant(value)

    def _is_compliant(self, value: float) -> bool:
        """Prüft ob Wert SLO-compliant ist."""
        if self.slo_definition.slo_type in [
            SLOType.LATENCY_P95,
            SLOType.LATENCY_P99,
            SLOType.LATENCY_P50,
        ] or self.slo_definition.slo_type == SLOType.ERROR_RATE:
            return value <= self.slo_definition.threshold
        if self.slo_definition.slo_type in [SLOType.AVAILABILITY, SLOType.SUCCESS_RATE] or self.slo_definition.slo_type == SLOType.THROUGHPUT:
            return value >= self.slo_definition.threshold
        return True  # Custom SLOs - externe Bewertung

    def _update_compliance_percentage(self):
        """Aktualisiert Compliance-Percentage basierend auf Time-Window."""
        if not self.measurements:
            self.compliance_percentage = 100.0
            return

        window_seconds = self.slo_definition.get_time_window_seconds()
        cutoff_time = time.time() - window_seconds

        # Filtere Measurements im Time-Window
        window_measurements = [m for m in self.measurements if m["timestamp"] >= cutoff_time]

        if not window_measurements:
            self.compliance_percentage = 100.0
            return

        compliant_count = sum(1 for m in window_measurements if m["compliant"])
        self.compliance_percentage = (compliant_count / len(window_measurements)) * 100.0

    def _update_error_budget(self):
        """Aktualisiert Error-Budget-Consumption."""
        if not self.slo_definition.error_budget_enabled:
            return

        error_budget = self.slo_definition.get_error_budget()
        current_error_rate = (100.0 - self.compliance_percentage) / 100.0

        self.error_budget_consumed = current_error_rate / error_budget if error_budget > 0 else 0.0

        # Berechne Burn-Rate (vereinfacht)
        window_seconds = self.slo_definition.get_time_window_seconds()
        expected_burn_rate = 1.0 / (window_seconds / 3600.0)  # Pro Stunde
        self.error_budget_burn_rate = (
            self.error_budget_consumed / expected_burn_rate if expected_burn_rate > 0 else 0.0
        )

    def _record_violation(self, value: float, timestamp: float):
        """Zeichnet SLO-Violation auf."""
        violation = SLOViolation(
            slo_name=self.slo_definition.name,
            violation_value=value,
            threshold=self.slo_definition.threshold,
            timestamp=timestamp,
            severity=self._calculate_violation_severity(value),
        )

        self.violations.append(violation)
        self.violation_count += 1
        self.last_violation_time = timestamp

        # Behalte nur letzte 1000 Violations
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]

    def calculate_violation_severity(self, value: float) -> ViolationSeverity:
        """Public API für Violation-Severity-Berechnung.

        Args:
            value: Wert für den die Severity berechnet werden soll

        Returns:
            ViolationSeverity basierend auf Abweichung
        """
        return self._calculate_violation_severity(value)

    def _calculate_violation_severity(self, value: float) -> ViolationSeverity:
        """Berechnet Violation-Severity basierend auf Abweichung."""
        threshold = self.slo_definition.threshold

        if self.slo_definition.slo_type in [
            SLOType.LATENCY_P95,
            SLOType.LATENCY_P99,
            SLOType.LATENCY_P50,
        ]:
            # Latency: Je höher über Threshold, desto schwerer
            deviation = (value - threshold) / threshold
        elif self.slo_definition.slo_type == SLOType.ERROR_RATE:
            # Error-Rate: Je höher über Threshold, desto schwerer
            deviation = (value - threshold) / threshold if threshold > 0 else value
        else:
            # Availability/Success-Rate/Throughput: Je niedriger unter Threshold, desto schwerer
            deviation = (threshold - value) / threshold if threshold > 0 else 0.0

        if deviation >= 0.5:  # 50%+ Abweichung
            return ViolationSeverity.CRITICAL
        if deviation >= 0.25:  # 25%+ Abweichung
            return ViolationSeverity.HIGH
        if deviation >= 0.1:  # 10%+ Abweichung
            return ViolationSeverity.MEDIUM
        return ViolationSeverity.LOW

    def get_recent_violations(self, hours: float = 24.0) -> list["SLOViolation"]:
        """Holt Violations der letzten N Stunden."""
        cutoff_time = time.time() - (hours * 3600)
        return [v for v in self.violations if v.timestamp >= cutoff_time]

    def is_error_budget_exhausted(self) -> bool:
        """Prüft ob Error-Budget erschöpft ist."""
        return self.error_budget_consumed >= 1.0

    def is_burn_rate_critical(self) -> bool:
        """Prüft ob Burn-Rate kritisch ist."""
        return self.error_budget_burn_rate >= self.slo_definition.error_budget_burn_rate_threshold

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "slo_definition": self.slo_definition.to_dict(),
            "current_value": self.current_value,
            "compliance_percentage": self.compliance_percentage,
            "error_budget_consumed": self.error_budget_consumed,
            "error_budget_burn_rate": self.error_budget_burn_rate,
            "last_measurement_time": self.last_measurement_time,
            "last_violation_time": self.last_violation_time,
            "total_measurements": self.total_measurements,
            "violation_count": self.violation_count,
            "recent_violations": len(self.get_recent_violations()),
            "error_budget_exhausted": self.is_error_budget_exhausted(),
            "burn_rate_critical": self.is_burn_rate_critical(),
        }


@dataclass
class SLAMetrics:
    """Metriken für SLA-Tracking."""

    sla_definition: SLADefinition
    slo_metrics: dict[str, SLOMetrics] = field(default_factory=dict)

    # SLA-Status
    current_compliance: float = 100.0
    is_breached: bool = False
    breach_start_time: float | None = None
    last_breach_time: float | None = None

    # Breach-History
    breaches: list["SLABreach"] = field(default_factory=list)
    total_breaches: int = 0

    # Recovery-Tracking
    recovery_time: float | None = None
    mttr: float = 0.0  # Mean Time To Recovery

    def update_compliance(self):
        """Aktualisiert SLA-Compliance basierend auf SLO-Metriken."""
        self.current_compliance = (
            self.sla_definition.get_combined_slo_score(self.slo_metrics) * 100.0
        )

        # Prüfe SLA-Breach
        was_breached = self.is_breached
        self.is_breached = self.sla_definition.is_sla_breached(self.slo_metrics)

        current_time = time.time()

        if self.is_breached and not was_breached:
            # Neue SLA-Breach
            self.breach_start_time = current_time
            self.last_breach_time = current_time
            self._record_breach(current_time)

        elif not self.is_breached and was_breached:
            # SLA-Recovery
            if self.breach_start_time:
                self.recovery_time = current_time - self.breach_start_time
                self._update_mttr()
            self.breach_start_time = None

    def _record_breach(self, timestamp: float):
        """Zeichnet SLA-Breach auf."""
        breach = SLABreach(
            sla_name=self.sla_definition.name,
            breach_timestamp=timestamp,
            compliance_at_breach=self.current_compliance,
            affected_slos=[slo.name for slo in self.sla_definition.slo_definitions],
        )

        self.breaches.append(breach)
        self.total_breaches += 1

        # Behalte nur letzte 100 Breaches
        if len(self.breaches) > 100:
            self.breaches = self.breaches[-100:]

    def _update_mttr(self):
        """Aktualisiert Mean Time To Recovery."""
        if not self.breaches:
            return

        recovery_times = []
        for breach in self.breaches:
            if breach.recovery_time:
                recovery_times.append(breach.recovery_time)

        if recovery_times:
            self.mttr = statistics.mean(recovery_times)

    def get_recent_breaches(self, hours: float = 24.0) -> list["SLABreach"]:
        """Holt Breaches der letzten N Stunden."""
        cutoff_time = time.time() - (hours * 3600)
        return [b for b in self.breaches if b.breach_timestamp >= cutoff_time]

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "sla_definition": self.sla_definition.to_dict(),
            "current_compliance": self.current_compliance,
            "is_breached": self.is_breached,
            "breach_start_time": self.breach_start_time,
            "last_breach_time": self.last_breach_time,
            "total_breaches": self.total_breaches,
            "recovery_time": self.recovery_time,
            "mttr": self.mttr,
            "recent_breaches": len(self.get_recent_breaches()),
            "slo_metrics": {key: metrics.to_dict() for key, metrics in self.slo_metrics.items()},
        }


@dataclass
class SLOViolation:
    """Einzelne SLO-Violation."""

    slo_name: str
    violation_value: float
    threshold: float
    timestamp: float
    severity: ViolationSeverity

    # Optional Context
    agent_id: str | None = None
    capability: str | None = None
    request_id: str | None = None

    # Resolution
    resolved: bool = False
    resolution_time: float | None = None
    resolution_notes: str = ""

    def get_deviation_percentage(self) -> float:
        """Berechnet Abweichung in Prozent."""
        if self.threshold == 0:
            return 0.0
        return abs(self.violation_value - self.threshold) / self.threshold * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "slo_name": self.slo_name,
            "violation_value": self.violation_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "capability": self.capability,
            "request_id": self.request_id,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time,
            "resolution_notes": self.resolution_notes,
            "deviation_percentage": self.get_deviation_percentage(),
        }


@dataclass
class SLABreach:
    """SLA-Breach-Ereignis."""

    sla_name: str
    breach_timestamp: float
    compliance_at_breach: float
    affected_slos: list[str]

    # Breach-Details
    severity: ViolationSeverity = ViolationSeverity.HIGH
    customer_impact: str = ""
    business_impact: str = ""

    # Resolution
    resolved: bool = False
    resolution_timestamp: float | None = None
    recovery_time: float | None = None
    resolution_notes: str = ""

    # Escalation
    escalated: bool = False
    escalation_timestamp: float | None = None
    escalation_level: int = 0

    def calculate_recovery_time(self):
        """Berechnet Recovery-Zeit."""
        if self.resolved and self.resolution_timestamp:
            self.recovery_time = self.resolution_timestamp - self.breach_timestamp

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "sla_name": self.sla_name,
            "breach_timestamp": self.breach_timestamp,
            "compliance_at_breach": self.compliance_at_breach,
            "affected_slos": self.affected_slos,
            "severity": self.severity.value,
            "customer_impact": self.customer_impact,
            "business_impact": self.business_impact,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp,
            "recovery_time": self.recovery_time,
            "resolution_notes": self.resolution_notes,
            "escalated": self.escalated,
            "escalation_timestamp": self.escalation_timestamp,
            "escalation_level": self.escalation_level,
        }
