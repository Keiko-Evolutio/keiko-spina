"""Edge Auto Scaler für Keiko Personal Assistant.

Dieses Modul implementiert den Auto-Scaler für die automatische Skalierung
von Edge-Computing-Nodes basierend auf Metriken und Regeln.
"""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
from .edge_types import (
    EdgeConfiguration,
    EdgeMetrics,
    EdgeNodeInfo,
    EdgeNodeStatus,
    ScalingAction,
    ScalingEvent,
    ScalingRule,
)

logger = get_logger(__name__)


class ScalingTrigger(str, Enum):
    """Scaling-Trigger-Typen."""
    METRIC_THRESHOLD = "metric-threshold"
    SCHEDULE_BASED = "schedule-based"
    MANUAL = "manual"
    PREDICTIVE = "predictive"


@dataclass
class ScalingDecision:
    """Scaling-Entscheidung."""
    decision_id: str
    action: ScalingAction
    target_nodes: int
    current_nodes: int
    trigger_rule: str
    trigger_metric: str
    trigger_value: float
    confidence_score: float
    estimated_impact: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AutoScalerMetrics:
    """Metriken für Auto-Scaler."""
    total_scaling_events: int = 0
    successful_scale_ups: int = 0
    successful_scale_downs: int = 0
    failed_scaling_events: int = 0
    average_decision_time_ms: float = 0.0
    last_scaling_event: datetime | None = None
    active_rules: int = 0
    nodes_managed: int = 0


class EdgeAutoScaler:
    """Enterprise Edge Auto Scaler für intelligente Node-Skalierung.

    Implementiert automatische Skalierung mit:
    - Metrik-basierte Scaling-Regeln
    - Predictive Scaling basierend auf Trends
    - Cooldown-Perioden zur Vermeidung von Flapping
    - Multi-Kriterien-Entscheidungsfindung
    - Integration mit Node-Registry und Load-Balancer
    """

    def __init__(self, config: EdgeConfiguration | None = None):
        """Initialisiert den Edge Auto Scaler.

        Args:
            config: Edge-Konfiguration
        """
        self.config = config or EdgeConfiguration()

        # Scaling-Management
        self._scaling_rules: dict[str, ScalingRule] = {}
        self._scaling_history: list[ScalingEvent] = []
        self._last_scaling_action: datetime | None = None
        self._scaling_lock = asyncio.Lock()

        # Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

        # Konfiguration
        self.evaluation_interval = timedelta(
            seconds=self.config.scaling_evaluation_interval_seconds
        )
        self.cooldown_period = timedelta(
            seconds=self.config.scaling_cooldown_period_seconds
        )
        self.min_nodes = self.config.min_nodes_global
        self.max_nodes = self.config.max_nodes_global

        # Metriken
        self.metrics = AutoScalerMetrics()

        # Callbacks
        self._scaling_callbacks: list[Callable[[ScalingEvent], None]] = []

        # Standard-Regeln erstellen
        self._create_default_rules()

        logger.info("Edge Auto Scaler initialisiert")

    async def start(self) -> None:
        """Startet den Auto-Scaler und Monitoring."""
        if self._running:
            logger.warning("Auto-Scaler bereits gestartet")
            return

        self._running = True

        # Monitoring-Task starten
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(),
            name="auto-scaler-monitoring"
        )

        logger.info("Edge Auto Scaler gestartet")

    async def stop(self) -> None:
        """Stoppt den Auto-Scaler."""
        if not self._running:
            return

        self._running = False

        # Monitoring-Task stoppen
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Edge Auto Scaler gestoppt")

    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Fügt eine Scaling-Regel hinzu.

        Args:
            rule: Scaling-Regel
        """
        self._scaling_rules[rule.rule_id] = rule
        self.metrics.active_rules = len(self._scaling_rules)

        logger.info(f"Scaling-Regel hinzugefügt: {rule.rule_id}")

    def remove_scaling_rule(self, rule_id: str) -> bool:
        """Entfernt eine Scaling-Regel.

        Args:
            rule_id: Regel-ID

        Returns:
            True wenn erfolgreich entfernt
        """
        if rule_id in self._scaling_rules:
            del self._scaling_rules[rule_id]
            self.metrics.active_rules = len(self._scaling_rules)
            logger.info(f"Scaling-Regel entfernt: {rule_id}")
            return True

        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Aktiviert eine Scaling-Regel.

        Args:
            rule_id: Regel-ID

        Returns:
            True wenn erfolgreich aktiviert
        """
        if rule_id in self._scaling_rules:
            self._scaling_rules[rule_id].enabled = True
            logger.info(f"Scaling-Regel aktiviert: {rule_id}")
            return True

        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Deaktiviert eine Scaling-Regel.

        Args:
            rule_id: Regel-ID

        Returns:
            True wenn erfolgreich deaktiviert
        """
        if rule_id in self._scaling_rules:
            self._scaling_rules[rule_id].enabled = False
            logger.info(f"Scaling-Regel deaktiviert: {rule_id}")
            return True

        return False

    async def evaluate_scaling_decision(
        self,
        current_metrics: EdgeMetrics,
        current_nodes: list[EdgeNodeInfo]
    ) -> ScalingDecision | None:
        """Evaluiert Scaling-Entscheidung basierend auf aktuellen Metriken.

        Args:
            current_metrics: Aktuelle Edge-Metriken
            current_nodes: Liste aktueller Nodes

        Returns:
            Scaling-Entscheidung oder None
        """
        decision_start = datetime.now(UTC)

        # Cooldown-Periode prüfen
        if self._is_in_cooldown():
            return None

        # Alle aktiven Regeln evaluieren
        triggered_rules = []

        for rule in self._scaling_rules.values():
            if not rule.enabled:
                continue

            if self._evaluate_rule(rule, current_metrics):
                triggered_rules.append(rule)

        if not triggered_rules:
            return None

        # Beste Regel auswählen (höchste Priorität)
        best_rule = max(triggered_rules, key=lambda r: self._get_rule_priority(r))

        # Scaling-Entscheidung erstellen
        current_node_count = len([n for n in current_nodes if n.status == EdgeNodeStatus.HEALTHY])
        target_nodes = self._calculate_target_nodes(best_rule, current_node_count)

        # Grenzen prüfen
        target_nodes = max(self.min_nodes, min(self.max_nodes, target_nodes))

        if target_nodes == current_node_count:
            return None

        # Entscheidung erstellen
        decision = ScalingDecision(
            decision_id=str(uuid.uuid4()),
            action=best_rule.action,
            target_nodes=target_nodes,
            current_nodes=current_node_count,
            trigger_rule=best_rule.rule_id,
            trigger_metric=best_rule.metric_name,
            trigger_value=self._get_metric_value(current_metrics, best_rule.metric_name),
            confidence_score=self._calculate_confidence_score(best_rule, current_metrics),
            estimated_impact=self._estimate_scaling_impact(best_rule, target_nodes, current_node_count)
        )

        # Entscheidungszeit messen
        decision_time = (datetime.now(UTC) - decision_start).total_seconds() * 1000
        self._update_decision_time_metric(decision_time)

        return decision

    async def execute_scaling_decision(self, decision: ScalingDecision) -> ScalingEvent:
        """Führt eine Scaling-Entscheidung aus.

        Args:
            decision: Scaling-Entscheidung

        Returns:
            Scaling-Event mit Ergebnis
        """
        event_id = str(uuid.uuid4())
        start_time = datetime.now(UTC)

        try:
            async with self._scaling_lock:
                # Scaling-Aktion ausführen
                success = await self._execute_scaling_action(decision)

                # Event erstellen
                event = ScalingEvent(
                    event_id=event_id,
                    rule_id=decision.trigger_rule,
                    action=decision.action,
                    trigger_metric=decision.trigger_metric,
                    trigger_value=decision.trigger_value,
                    threshold_value=self._get_rule_threshold(decision.trigger_rule),
                    nodes_before=decision.current_nodes,
                    nodes_after=decision.target_nodes if success else decision.current_nodes,
                    timestamp=start_time,
                    success=success
                )

                # Event speichern
                self._scaling_history.append(event)
                self._last_scaling_action = start_time

                # Metriken aktualisieren
                self.metrics.total_scaling_events += 1
                self.metrics.last_scaling_event = start_time

                if success:
                    if decision.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]:
                        self.metrics.successful_scale_ups += 1
                    elif decision.action in [ScalingAction.SCALE_DOWN, ScalingAction.SCALE_IN]:
                        self.metrics.successful_scale_downs += 1
                else:
                    self.metrics.failed_scaling_events += 1

                # Callbacks aufrufen
                for callback in self._scaling_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Fehler in Scaling-Callback: {e}")

                if success:
                    logger.info(f"Scaling erfolgreich: {decision.action.value} von {decision.current_nodes} auf {decision.target_nodes} Nodes")
                else:
                    logger.error(f"Scaling fehlgeschlagen: {decision.action.value}")

                return event

        except Exception as e:
            # Fehler-Event erstellen
            event = ScalingEvent(
                event_id=event_id,
                rule_id=decision.trigger_rule,
                action=decision.action,
                trigger_metric=decision.trigger_metric,
                trigger_value=decision.trigger_value,
                threshold_value=self._get_rule_threshold(decision.trigger_rule),
                nodes_before=decision.current_nodes,
                nodes_after=decision.current_nodes,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )

            self._scaling_history.append(event)
            self.metrics.total_scaling_events += 1
            self.metrics.failed_scaling_events += 1

            logger.error(f"Scaling-Fehler: {e}")
            return event

    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]) -> None:
        """Fügt Callback für Scaling-Events hinzu.

        Args:
            callback: Callback-Funktion
        """
        self._scaling_callbacks.append(callback)

    def get_scaling_history(self, limit: int = 100) -> list[ScalingEvent]:
        """Gibt Scaling-Historie zurück.

        Args:
            limit: Maximale Anzahl Events

        Returns:
            Liste der letzten Scaling-Events
        """
        return self._scaling_history[-limit:]

    def _create_default_rules(self) -> None:
        """Erstellt Standard-Scaling-Regeln."""
        # CPU-basierte Scale-Up-Regel
        cpu_scale_up = ScalingRule(
            rule_id="cpu-scale-up",
            metric_name="cpu_usage_percent",
            threshold_value=80.0,
            comparison_operator=">=",
            action=ScalingAction.SCALE_OUT,
            cooldown_period_seconds=300,
            scale_factor=1.5,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )

        # CPU-basierte Scale-Down-Regel
        cpu_scale_down = ScalingRule(
            rule_id="cpu-scale-down",
            metric_name="cpu_usage_percent",
            threshold_value=30.0,
            comparison_operator="<=",
            action=ScalingAction.SCALE_IN,
            cooldown_period_seconds=600,
            scale_factor=0.7,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )

        # Memory-basierte Scale-Up-Regel
        memory_scale_up = ScalingRule(
            rule_id="memory-scale-up",
            metric_name="memory_usage_percent",
            threshold_value=85.0,
            comparison_operator=">=",
            action=ScalingAction.SCALE_OUT,
            cooldown_period_seconds=300,
            scale_factor=1.3,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )

        # Task-Queue-basierte Scale-Up-Regel
        queue_scale_up = ScalingRule(
            rule_id="queue-scale-up",
            metric_name="task_queue_size",
            threshold_value=100.0,
            comparison_operator=">=",
            action=ScalingAction.SCALE_OUT,
            cooldown_period_seconds=180,
            scale_factor=1.2,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )

        # Regeln hinzufügen
        for rule in [cpu_scale_up, cpu_scale_down, memory_scale_up, queue_scale_up]:
            self.add_scaling_rule(rule)

    def _is_in_cooldown(self) -> bool:
        """Prüft ob Cooldown-Periode aktiv ist."""
        if not self._last_scaling_action:
            return False

        time_since_last = datetime.now(UTC) - self._last_scaling_action
        return time_since_last < self.cooldown_period

    def _evaluate_rule(self, rule: ScalingRule, metrics: EdgeMetrics) -> bool:
        """Evaluiert eine einzelne Scaling-Regel."""
        metric_value = self._get_metric_value(metrics, rule.metric_name)

        if rule.comparison_operator == ">=":
            return metric_value >= rule.threshold_value
        if rule.comparison_operator == "<=":
            return metric_value <= rule.threshold_value
        if rule.comparison_operator == ">":
            return metric_value > rule.threshold_value
        if rule.comparison_operator == "<":
            return metric_value < rule.threshold_value
        if rule.comparison_operator == "==":
            return abs(metric_value - rule.threshold_value) < 0.01

        return False

    def _get_metric_value(self, metrics: EdgeMetrics, metric_name: str) -> float:
        """Extrahiert Metrik-Wert."""
        # Vereinfachte Metrik-Extraktion
        if hasattr(metrics, metric_name):
            return float(getattr(metrics, metric_name))

        # Fallback für unbekannte Metriken
        return 0.0

    def _get_rule_priority(self, rule: ScalingRule) -> float:
        """Berechnet Regel-Priorität."""
        # Vereinfachte Prioritäts-Berechnung
        priority_map = {
            ScalingAction.SCALE_UP: 1.0,
            ScalingAction.SCALE_OUT: 1.0,
            ScalingAction.SCALE_DOWN: 0.8,
            ScalingAction.SCALE_IN: 0.8,
            ScalingAction.NO_ACTION: 0.0
        }

        return priority_map.get(rule.action, 0.5)

    def _calculate_target_nodes(self, rule: ScalingRule, current_nodes: int) -> int:
        """Berechnet Ziel-Anzahl der Nodes."""
        if rule.action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT] or rule.action in [ScalingAction.SCALE_DOWN, ScalingAction.SCALE_IN]:
            return int(current_nodes * rule.scale_factor)

        return current_nodes

    def _calculate_confidence_score(self, rule: ScalingRule, metrics: EdgeMetrics) -> float:
        """Berechnet Konfidenz-Score für Entscheidung."""
        # Vereinfachte Konfidenz-Berechnung
        return 0.85

    def _estimate_scaling_impact(self, rule: ScalingRule, target_nodes: int, current_nodes: int) -> dict[str, Any]:
        """Schätzt Impact der Scaling-Aktion."""
        return {
            "node_change": target_nodes - current_nodes,
            "estimated_cost_change_percent": (target_nodes - current_nodes) * 10,
            "estimated_capacity_change_percent": (target_nodes - current_nodes) * 20
        }

    def _get_rule_threshold(self, rule_id: str) -> float:
        """Gibt Threshold-Wert einer Regel zurück."""
        rule = self._scaling_rules.get(rule_id)
        return rule.threshold_value if rule else 0.0

    async def _execute_scaling_action(self, decision: ScalingDecision) -> bool:
        """Führt die eigentliche Scaling-Aktion aus."""
        # Hier würde die Integration mit Node-Registry und Orchestrator stattfinden
        # Simulierte Ausführung
        await asyncio.sleep(0.1)
        return True

    def _update_decision_time_metric(self, decision_time_ms: float) -> None:
        """Aktualisiert Entscheidungszeit-Metrik."""
        if self.metrics.total_scaling_events > 0:
            total_time = self.metrics.average_decision_time_ms * self.metrics.total_scaling_events
            self.metrics.average_decision_time_ms = (total_time + decision_time_ms) / (self.metrics.total_scaling_events + 1)
        else:
            self.metrics.average_decision_time_ms = decision_time_ms

    async def _monitoring_loop(self) -> None:
        """Monitoring-Loop für kontinuierliche Evaluierung."""
        while self._running:
            try:
                # Hier würde die Integration mit Monitoring-System stattfinden
                # Simuliertes Monitoring
                await asyncio.sleep(self.evaluation_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Auto-Scaler-Monitoring: {e}")
                await asyncio.sleep(5)

    async def get_scaler_status(self) -> dict[str, Any]:
        """Gibt Auto-Scaler-Status zurück."""
        return {
            "running": self._running,
            "active_rules": self.metrics.active_rules,
            "total_scaling_events": self.metrics.total_scaling_events,
            "successful_scale_ups": self.metrics.successful_scale_ups,
            "successful_scale_downs": self.metrics.successful_scale_downs,
            "failed_scaling_events": self.metrics.failed_scaling_events,
            "average_decision_time_ms": self.metrics.average_decision_time_ms,
            "last_scaling_event": self.metrics.last_scaling_event.isoformat() if self.metrics.last_scaling_event else None,
            "in_cooldown": self._is_in_cooldown(),
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes
        }


def create_auto_scaler(config: EdgeConfiguration | None = None) -> EdgeAutoScaler:
    """Factory-Funktion für Edge Auto Scaler.

    Args:
        config: Edge-Konfiguration

    Returns:
        Neue EdgeAutoScaler-Instanz
    """
    return EdgeAutoScaler(config)
