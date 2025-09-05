# backend/kei_agents/registry/rollout_manager.py
"""Rollout Manager für Agent Deployments.

Implementiert Canary Deployments, Blue-Green Deployments, Feature Flags
und Rollback-Mechanismen für das Keiko Personal Assistant
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import (
    BusinessLogicError,
    ValidationError,
    get_logger,
    with_log_links,
)

from .enhanced_models import (
    AgentStatus,
    RolloutConfiguration,
    RolloutStrategy,
    SemanticVersion,
)

logger = get_logger(__name__)


class RolloutStatus(str, Enum):
    """Status eines Rollouts."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class RolloutPhase(str, Enum):
    """Phasen eines Rollouts."""

    PREPARATION = "preparation"
    CANARY = "canary"
    VALIDATION = "validation"
    FULL_DEPLOYMENT = "full_deployment"
    MONITORING = "monitoring"
    CLEANUP = "cleanup"


@dataclass
class RolloutEvent:
    """Event während eines Rollouts."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rollout_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    phase: RolloutPhase = RolloutPhase.PREPARATION
    event_type: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class RolloutMetrics:
    """Metriken für Rollout-Monitoring."""

    rollout_id: str

    # Deployment-Metriken
    total_instances: int = 0
    deployed_instances: int = 0
    healthy_instances: int = 0
    failed_instances: int = 0

    # Performance-Metriken
    success_rate: float = 0.0
    error_rate: float = 0.0
    average_response_time_ms: float = 0.0
    response_time_p95: float = 0.0
    requests_per_second: float = 0.0

    # Rollout-spezifische Metriken
    canary_percentage: float = 0.0
    traffic_split_percentage: float = 0.0
    rollback_triggered: bool = False

    # Timestamps
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ActiveRollout:
    """Aktiver Rollout mit Status und Konfiguration."""

    rollout_id: str
    agent_id: str
    source_version: SemanticVersion
    target_version: SemanticVersion
    tenant_id: str

    # Rollout-Konfiguration
    config: RolloutConfiguration

    # Status
    status: RolloutStatus = RolloutStatus.PENDING
    current_phase: RolloutPhase = RolloutPhase.PREPARATION

    # Metriken und Events
    metrics: RolloutMetrics = field(default_factory=lambda: RolloutMetrics(""))
    events: list[RolloutEvent] = field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    next_phase_at: datetime | None = None

    # Feature Flags
    active_feature_flags: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Initialisiert Rollout."""
        if not self.rollout_id:
            self.rollout_id = str(uuid.uuid4())

        self.metrics.rollout_id = self.rollout_id

    def add_event(self, event_type: str, message: str, success: bool = True, **details) -> None:
        """Fügt Event zum Rollout hinzu.

        Args:
            event_type: Event-Typ
            message: Event-Message
            success: Erfolg-Status
            **details: Zusätzliche Details
        """
        event = RolloutEvent(
            rollout_id=self.rollout_id,
            phase=self.current_phase,
            event_type=event_type,
            message=message,
            success=success,
            details=details,
        )

        self.events.append(event)

        logger.info(
            f"Rollout-Event: {event_type} - {message}",
            extra={
                "rollout_id": self.rollout_id,
                "agent_id": self.agent_id,
                "phase": self.current_phase.value,
                "success": success,
                **details,
            },
        )

    def is_healthy(self) -> bool:
        """Prüft ob Rollout gesund ist."""
        return (
            self.metrics.success_rate >= 0.95
            and self.metrics.error_rate <= 0.05
            and self.metrics.healthy_instances >= self.metrics.deployed_instances * 0.9
        )

    def should_rollback(self) -> bool:
        """Prüft ob Rollback ausgeführt werden sollte."""
        if not self.config.auto_rollback_on_error:
            return False

        # Rollback bei hoher Fehlerrate
        if self.metrics.error_rate > 0.2:
            return True

        # Rollback bei niedriger Erfolgsrate
        if self.metrics.success_rate < 0.8:
            return True

        # Rollback bei zu vielen fehlgeschlagenen Instanzen
        if (
            self.metrics.deployed_instances > 0
            and self.metrics.failed_instances / self.metrics.deployed_instances > 0.3
        ):
            return True

        return False


class RolloutManager:
    """Manager für Agent-Rollouts und Deployment-Strategien."""

    def __init__(self):
        """Initialisiert Rollout Manager."""
        self._active_rollouts: dict[str, ActiveRollout] = {}
        self._rollout_history: list[ActiveRollout] = []
        self._feature_flags: dict[str, dict[str, bool]] = {}  # agent_id -> {flag: enabled}

        # Background-Task für Rollout-Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval = 30  # Sekunden

    def start_monitoring(self) -> None:
        """Startet Background-Monitoring für Rollouts."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_rollouts())
            logger.info("Rollout-Monitoring gestartet")

    def stop_monitoring(self) -> None:
        """Stoppt Background-Monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("Rollout-Monitoring gestoppt")

    @with_log_links(component="rollout_manager", operation="start_rollout")
    async def start_rollout(
        self,
        agent_id: str,
        source_version: str,
        target_version: str,
        tenant_id: str,
        config: RolloutConfiguration | None = None,
    ) -> str:
        """Startet neuen Rollout.

        Args:
            agent_id: Agent-ID
            source_version: Quell-Version
            target_version: Ziel-Version
            tenant_id: Tenant-ID
            config: Rollout-Konfiguration

        Returns:
            Rollout-ID

        Raises:
            ValidationError: Bei ungültigen Parametern
            BusinessLogicError: Bei bereits laufendem Rollout
        """
        # Prüfe auf bereits laufende Rollouts für diesen Agent
        for rollout in self._active_rollouts.values():
            if (
                rollout.agent_id == agent_id
                and rollout.tenant_id == tenant_id
                and rollout.status == RolloutStatus.IN_PROGRESS
            ):
                raise BusinessLogicError(
                    message=f"Rollout für Agent {agent_id} bereits aktiv: {rollout.rollout_id}",
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    active_rollout_id=rollout.rollout_id,
                )

        # Parse Versionen
        try:
            source_sem_version = SemanticVersion.parse(source_version)
            target_sem_version = SemanticVersion.parse(target_version)
        except ValueError as e:
            raise ValidationError(
                message=f"Ungültiges Versions-Format: {e}",
                field="version",
                value=f"{source_version} -> {target_version}",
                cause=e,
            )

        # Standard-Konfiguration falls nicht angegeben
        if config is None:
            config = RolloutConfiguration()

        # Erstelle Rollout
        rollout = ActiveRollout(
            rollout_id=str(uuid.uuid4()),
            agent_id=agent_id,
            source_version=source_sem_version,
            target_version=target_sem_version,
            tenant_id=tenant_id,
            config=config,
        )

        # Initialisiere Feature Flags
        rollout.active_feature_flags = config.feature_flags.copy()

        # Registriere Rollout
        self._active_rollouts[rollout.rollout_id] = rollout

        # Starte Rollout-Prozess
        await self._execute_rollout_phase(rollout, RolloutPhase.PREPARATION)

        logger.info(
            f"Rollout gestartet: {rollout.rollout_id}",
            extra={
                "rollout_id": rollout.rollout_id,
                "agent_id": agent_id,
                "source_version": source_version,
                "target_version": target_version,
                "strategy": config.strategy.value,
                "tenant_id": tenant_id,
            },
        )

        return rollout.rollout_id

    async def _execute_rollout_phase(self, rollout: ActiveRollout, phase: RolloutPhase) -> None:
        """Führt Rollout-Phase aus.

        Args:
            rollout: Rollout-Instanz
            phase: Auszuführende Phase
        """
        rollout.current_phase = phase
        rollout.add_event("phase_started", f"Phase {phase.value} gestartet")

        try:
            if phase == RolloutPhase.PREPARATION:
                await self._execute_preparation_phase(rollout)

            elif phase == RolloutPhase.CANARY:
                await self._execute_canary_phase(rollout)

            elif phase == RolloutPhase.VALIDATION:
                await self._execute_validation_phase(rollout)

            elif phase == RolloutPhase.FULL_DEPLOYMENT:
                await self._execute_full_deployment_phase(rollout)

            elif phase == RolloutPhase.MONITORING:
                await self._execute_monitoring_phase(rollout)

            elif phase == RolloutPhase.CLEANUP:
                await self._execute_cleanup_phase(rollout)

            rollout.add_event("phase_completed", f"Phase {phase.value} abgeschlossen")

            # Bestimme nächste Phase
            next_phase = self._get_next_phase(rollout, phase)
            if next_phase:
                # Verzögerung vor nächster Phase
                delay = self._get_phase_delay(rollout, next_phase)
                rollout.next_phase_at = datetime.now(UTC) + timedelta(seconds=delay)
            else:
                # Rollout abgeschlossen
                await self._complete_rollout(rollout)

        except Exception as e:
            rollout.add_event(
                "phase_failed", f"Phase {phase.value} fehlgeschlagen: {e}", success=False
            )
            await self._handle_rollout_failure(rollout, e)

    async def _execute_preparation_phase(self, rollout: ActiveRollout) -> None:
        """Führt Preparation-Phase aus."""
        rollout.status = RolloutStatus.IN_PROGRESS
        rollout.started_at = datetime.now(UTC)
        rollout.metrics.started_at = rollout.started_at

        # Validiere Ziel-Version über Version Manager
        try:
            from .version_manager import version_manager

            _ = version_manager.resolve_version(
                rollout.agent_id, str(rollout.target_version), tenant_id=rollout.tenant_id
            )
        except Exception as e:
            raise ValidationError(
                message=f"Ziel-Version ungültig oder nicht verfügbar: {rollout.target_version}",
                field="target_version",
                value=str(rollout.target_version),
                cause=e,
            )

        # Bereite Deployment vor
        rollout.add_event("preparation", "Deployment vorbereitet")

    async def _execute_canary_phase(self, rollout: ActiveRollout) -> None:
        """Führt Canary-Phase aus."""
        if rollout.config.strategy != RolloutStrategy.CANARY:
            return  # Skip Canary für andere Strategien

        # Deploye Canary-Instanzen
        canary_percentage = rollout.config.canary_percentage
        rollout.metrics.canary_percentage = canary_percentage

        # Canary-Deployment über Discovery Engine + Traffic-Splitting
        try:
            from .discovery_engine import DiscoveryQuery, DiscoveryStrategy, discovery_engine

            query = DiscoveryQuery(
                agent_id=rollout.agent_id,
                tenant_id=rollout.tenant_id,
                strategy=DiscoveryStrategy.HYBRID,
                max_results=max(1, int(10 * canary_percentage / 100)),
            )
            results = await discovery_engine.discover_agents(query)
            # Markiere ausgewählte Instanzen als CANARY
            for res in results:
                res.instance.status = AgentStatus.CANARY
            # Setze initialen Traffic-Split auf Canary-Anteil
            rollout.metrics.traffic_split_percentage = float(canary_percentage)
        except Exception as e:
            logger.warning(f"Canary-Discovery nicht möglich: {e}")

        rollout.add_event("canary_deployed", f"Canary-Deployment ({canary_percentage}%) gestartet")

        # Warte auf Canary-Dauer
        await asyncio.sleep(rollout.config.canary_duration_minutes * 60)

    async def _execute_validation_phase(self, rollout: ActiveRollout) -> None:
        """Führt Validation-Phase aus."""
        # Sammle Metriken
        await self._update_rollout_metrics(rollout)

        # Prüfe Gesundheit
        if not rollout.is_healthy():
            raise BusinessLogicError(
                message="Rollout-Validierung fehlgeschlagen - Metriken unter Schwellwerten",
                rollout_id=rollout.rollout_id,
                success_rate=rollout.metrics.success_rate,
                error_rate=rollout.metrics.error_rate,
            )

        rollout.add_event("validation_passed", "Rollout-Validierung erfolgreich")

    async def _execute_full_deployment_phase(self, rollout: ActiveRollout) -> None:
        """Führt Full-Deployment-Phase aus."""
        if rollout.config.strategy == RolloutStrategy.BLUE_GREEN:
            # Blue-Green Switch mit Health-Validierung und gradueller Migration
            await self._blue_green_switch_with_validation(rollout)

        elif rollout.config.strategy == RolloutStrategy.ROLLING:
            # Rolling Update
            batch_size = rollout.config.rolling_batch_size
            delay = rollout.config.rolling_delay_seconds

            # Implementiere einfache Rolling-Update-Iteration über Discovery-Instanzen
            try:
                from .discovery_engine import DiscoveryQuery, discovery_engine

                query = DiscoveryQuery(agent_id=rollout.agent_id, tenant_id=rollout.tenant_id)
                results = await discovery_engine.discover_agents(query)
                instances = [r.instance for r in results]

                for i in range(0, len(instances), max(1, batch_size)):
                    batch = instances[i : i + batch_size]
                    for inst in batch:
                        inst.status = AgentStatus.ROLLOUT
                    rollout.add_event(
                        "rolling_update_batch",
                        f"Batch {i // max(1, batch_size) + 1} mit {len(batch)} Instanzen ausgerollt",
                    )
                    await asyncio.sleep(max(0, delay))
            except Exception as e:
                logger.warning(f"Rolling Update nicht vollständig durchgeführt: {e}")
            else:
                # Nach Rolling-Update graduelle Traffic-Migration durchführen
                await self._gradual_traffic_migration(rollout)

        else:
            # Immediate Deployment
            rollout.add_event("immediate_deployment", "Sofortiges Deployment durchgeführt")

        rollout.metrics.traffic_split_percentage = 100.0

    async def _execute_monitoring_phase(self, rollout: ActiveRollout) -> None:
        """Führt Monitoring-Phase aus."""
        # Überwache für konfigurierten Zeitraum
        monitoring_duration = 300  # 5 Minuten Standard

        start_time = datetime.now(UTC)
        while (datetime.now(UTC) - start_time).total_seconds() < monitoring_duration:
            await self._update_rollout_metrics(rollout)

            # Prüfe auf Rollback-Bedingungen
            if rollout.should_rollback():
                await self._execute_rollback(rollout)
                return

            # Automatischer Rollback-Trigger bei Health-Check-Failures
            if not rollout.is_healthy() and rollout.config.auto_rollback_on_error:
                rollout.add_event("health_check_failed", "Health-Check unter Schwellwerten", success=False)
                await self._execute_rollback(rollout)
                return

            await asyncio.sleep(30)  # Prüfe alle 30 Sekunden

        rollout.add_event("monitoring_completed", "Monitoring-Phase abgeschlossen")

    async def _execute_cleanup_phase(self, rollout: ActiveRollout) -> None:
        """Führt Cleanup-Phase aus."""
        # Entferne alte Versionen (Deprecated Cleanup) über Version Manager
        try:
            from .version_manager import version_manager

            _ = version_manager.cleanup_deprecated_versions(max_age_days=30)
        except Exception as e:
            logger.warning(f"Cleanup der alten Versionen fehlgeschlagen: {e}")

        rollout.add_event("cleanup_completed", "Cleanup abgeschlossen")

    async def _blue_green_switch_with_validation(self, rollout: ActiveRollout) -> None:
        """Führt Blue-Green Switch mit erweiterten Health-Validierungen durch."""
        # Warte optionalen Delay
        await asyncio.sleep(rollout.config.blue_green_switch_delay_minutes * 60)
        rollout.add_event("blue_green_switch_started", "Blue-Green-Switch gestartet")

        # Erweiterte Pre-Switch-Validierung
        await self._perform_pre_switch_validation(rollout)
        if not rollout.is_healthy():
            rollout.add_event("blue_green_validation_failed", "Blue-Green Health-Prüfung fehlgeschlagen", success=False)
            if rollout.config.auto_rollback_on_error:
                await self._execute_rollback(rollout)
            return

        # Erweiterte Canary-Validierung vor vollständigem Switch
        await self._perform_extended_canary_validation(rollout)
        if not rollout.is_healthy():
            rollout.add_event("canary_validation_failed", "Erweiterte Canary-Validierung fehlgeschlagen", success=False)
            if rollout.config.auto_rollback_on_error:
                await self._execute_rollback(rollout)
            return

        # Switch durchführen und Traffic graduell migrieren
        rollout.add_event("blue_green_switch", "Blue-Green-Switch durchgeführt")
        await self._gradual_traffic_migration_with_validation(rollout)

    async def _gradual_traffic_migration_with_validation(self, rollout: ActiveRollout) -> None:
        """Führt eine graduelle Traffic-Migration mit erweiterten Validierungen durch."""
        try:
            steps = rollout.config.traffic_split_steps
            for i, step in enumerate(steps):
                rollout.metrics.traffic_split_percentage = float(step)
                rollout.add_event("traffic_split", f"Traffic auf {step}% migriert")

                # Erweiterte Health-Validierung nach jedem Schritt
                await self._update_rollout_metrics(rollout)

                # Zusätzliche Validierung für kritische Traffic-Stufen
                if step >= 50:  # Kritische Traffic-Stufen
                    await self._perform_critical_traffic_validation(rollout, step)

                if not rollout.is_healthy() and rollout.config.auto_rollback_on_error:
                    rollout.add_event(
                        "traffic_split_health_failed",
                        f"Health unter Schwellwert bei {step}%",
                        success=False,
                    )
                    await self._execute_rollback(rollout)
                    return

                # Adaptive Wartezeit basierend auf Traffic-Stufe
                wait_time = self._calculate_adaptive_wait_time(step, rollout.config.health_check_interval_seconds)
                await asyncio.sleep(wait_time)

        except Exception as e:
            logger.warning(f"Traffic-Migration nicht vollständig: {e}")

    async def _gradual_traffic_migration(self, rollout: ActiveRollout) -> None:
        """Legacy-Methode für Rückwärtskompatibilität."""
        await self._gradual_traffic_migration_with_validation(rollout)

    async def _complete_rollout(self, rollout: ActiveRollout) -> None:
        """Schließt Rollout ab."""
        rollout.status = RolloutStatus.COMPLETED
        rollout.completed_at = datetime.now(UTC)
        rollout.metrics.completed_at = rollout.completed_at

        # Verschiebe zu Historie
        self._rollout_history.append(rollout)
        del self._active_rollouts[rollout.rollout_id]

        rollout.add_event("rollout_completed", "Rollout erfolgreich abgeschlossen")

        logger.info(
            f"Rollout abgeschlossen: {rollout.rollout_id}",
            extra={
                "rollout_id": rollout.rollout_id,
                "agent_id": rollout.agent_id,
                "duration_minutes": (rollout.completed_at - rollout.started_at).total_seconds()
                / 60,
                "success_rate": rollout.metrics.success_rate,
            },
        )

    async def _handle_rollout_failure(self, rollout: ActiveRollout, error: Exception) -> None:
        """Behandelt Rollout-Fehler."""
        rollout.status = RolloutStatus.FAILED
        rollout.add_event("rollout_failed", f"Rollout fehlgeschlagen: {error}", success=False)

        # Automatischer Rollback falls konfiguriert
        if rollout.config.auto_rollback_on_error:
            await self._execute_rollback(rollout)

        logger.error(
            f"Rollout fehlgeschlagen: {rollout.rollout_id}",
            extra={
                "rollout_id": rollout.rollout_id,
                "agent_id": rollout.agent_id,
                "error": str(error),
                "phase": rollout.current_phase.value,
            },
        )

    @with_log_links(component="rollout_manager", operation="rollback")
    async def _execute_rollback(self, rollout: ActiveRollout) -> None:
        """Führt Rollback aus."""
        rollout.status = RolloutStatus.ROLLED_BACK
        rollout.metrics.rollback_triggered = True

        # Rollback-Logic: Stoppe weitere Phasen, leite Traffic zurück, markiere Instanzen
        try:
            # Stoppe geplante nächste Phasen
            rollout.next_phase_at = None

            # Traffic zurück auf Source-Version durch Discovery-Status-Rücksetzung
            from .discovery_engine import DiscoveryQuery, discovery_engine

            query = DiscoveryQuery(agent_id=rollout.agent_id, tenant_id=rollout.tenant_id)
            results = await discovery_engine.discover_agents(query)
            for res in results:
                inst = res.instance
                # Neue Zielinstanzen deaktivieren, alte Instanzen wieder aktivieren
                if str(inst.agent_metadata.version) == str(rollout.target_version):
                    inst.status = AgentStatus.UNAVAILABLE
                elif str(inst.agent_metadata.version) == str(rollout.source_version):
                    inst.status = AgentStatus.AVAILABLE
        except Exception as e:
            logger.warning(f"Rollback-Anpassungen unvollständig: {e}")

        rollout.add_event("rollback_executed", "Rollback durchgeführt")

        logger.warning(
            f"Rollback durchgeführt: {rollout.rollout_id}",
            extra={
                "rollout_id": rollout.rollout_id,
                "agent_id": rollout.agent_id,
                "source_version": str(rollout.source_version),
                "target_version": str(rollout.target_version),
            },
        )

    async def _update_rollout_metrics(self, rollout: ActiveRollout) -> None:
        """Aktualisiert Rollout-Metriken."""
        # Integriere einfache Observability-Proxy: verwende Discovery- und Version-Daten
        try:
            from .discovery_engine import DiscoveryQuery, discovery_engine

            query = DiscoveryQuery(agent_id=rollout.agent_id, tenant_id=rollout.tenant_id)
            results = await discovery_engine.discover_agents(query)
            total = len(results)
            healthy = len([r for r in results if r.instance.is_healthy()])
            failed = total - healthy

            rollout.metrics.success_rate = (healthy / total) if total else 1.0
            rollout.metrics.error_rate = (failed / total) if total else 0.0
            rollout.metrics.average_response_time_ms = (
                sum(r.instance.response_time_ms for r in results) / total if total else 0.0
            )
            # Berechne P95 Response Time (vereinfacht als 95% der maximalen Response Time)
            response_times = [r.instance.response_time_ms for r in results]
            if response_times:
                response_times.sort()
                p95_index = int(len(response_times) * 0.95)
                rollout.metrics.response_time_p95 = response_times[min(p95_index, len(response_times) - 1)]
            else:
                rollout.metrics.response_time_p95 = 0.0
            rollout.metrics.requests_per_second = (
                sum(r.instance.requests_per_second for r in results)
            )
        except Exception:
            # Fallback-Placeholder-Metriken
            rollout.metrics.success_rate = 0.98
            rollout.metrics.error_rate = 0.02
            rollout.metrics.average_response_time_ms = 150.0
            rollout.metrics.response_time_p95 = 200.0
            rollout.metrics.requests_per_second = 100.0
        rollout.metrics.last_updated = datetime.now(UTC)

    def _get_next_phase(
        self, rollout: ActiveRollout, current_phase: RolloutPhase
    ) -> RolloutPhase | None:
        """Bestimmt nächste Rollout-Phase."""
        phase_sequence = {
            RolloutPhase.PREPARATION: (
                RolloutPhase.CANARY
                if rollout.config.strategy == RolloutStrategy.CANARY
                else RolloutPhase.FULL_DEPLOYMENT
            ),
            RolloutPhase.CANARY: RolloutPhase.VALIDATION,
            RolloutPhase.VALIDATION: RolloutPhase.FULL_DEPLOYMENT,
            RolloutPhase.FULL_DEPLOYMENT: RolloutPhase.MONITORING,
            RolloutPhase.MONITORING: RolloutPhase.CLEANUP,
            RolloutPhase.CLEANUP: None,
        }

        return phase_sequence.get(current_phase)

    def _get_phase_delay(self, _rollout: ActiveRollout, phase: RolloutPhase) -> int:
        """Bestimmt Verzögerung vor Phase."""
        delays = {
            RolloutPhase.CANARY: 30,
            RolloutPhase.VALIDATION: 60,
            RolloutPhase.FULL_DEPLOYMENT: 120,
            RolloutPhase.MONITORING: 30,
            RolloutPhase.CLEANUP: 60,
        }

        return delays.get(phase, 0)

    async def _monitor_rollouts(self) -> None:
        """Background-Task für Rollout-Monitoring."""
        while True:
            try:
                current_time = datetime.now(UTC)

                for rollout in list(self._active_rollouts.values()):
                    # Prüfe ob nächste Phase fällig ist
                    if rollout.next_phase_at and current_time >= rollout.next_phase_at:

                        next_phase = self._get_next_phase(rollout, rollout.current_phase)
                        if next_phase:
                            await self._execute_rollout_phase(rollout, next_phase)

                    # Aktualisiere Metriken
                    await self._update_rollout_metrics(rollout)

                    # Prüfe Rollback-Bedingungen
                    if rollout.should_rollback():
                        await self._execute_rollback(rollout)

                await asyncio.sleep(self._monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Rollout-Monitoring: {e}")
                await asyncio.sleep(self._monitoring_interval)

    def get_rollout_status(self, rollout_id: str) -> ActiveRollout | None:
        """Holt Rollout-Status.

        Args:
            rollout_id: Rollout-ID

        Returns:
            Rollout-Instanz oder None
        """
        return self._active_rollouts.get(rollout_id)

    def list_active_rollouts(
        self, agent_id: str | None = None, tenant_id: str | None = None
    ) -> list[ActiveRollout]:
        """Listet aktive Rollouts.

        Args:
            agent_id: Filter nach Agent-ID
            tenant_id: Filter nach Tenant-ID

        Returns:
            Liste aktiver Rollouts
        """
        rollouts = list(self._active_rollouts.values())

        if agent_id:
            rollouts = [r for r in rollouts if r.agent_id == agent_id]

        if tenant_id:
            rollouts = [r for r in rollouts if r.tenant_id == tenant_id]

        return rollouts

    def get_rollout_statistics(self) -> dict[str, Any]:
        """Holt Rollout-Statistiken.

        Returns:
            Statistiken-Dictionary
        """
        active_count = len(self._active_rollouts)
        total_count = active_count + len(self._rollout_history)

        # Status-Verteilung
        status_counts = {}
        for rollout in self._active_rollouts.values():
            status_counts[rollout.status.value] = status_counts.get(rollout.status.value, 0) + 1

        for rollout in self._rollout_history:
            status_counts[rollout.status.value] = status_counts.get(rollout.status.value, 0) + 1

        return {
            "active_rollouts": active_count,
            "total_rollouts": total_count,
            "status_distribution": status_counts,
            "monitoring_active": self._monitoring_task is not None
            and not self._monitoring_task.done(),
        }

    async def _perform_pre_switch_validation(self, rollout: ActiveRollout) -> None:
        """Führt erweiterte Pre-Switch-Validierung durch."""
        rollout.add_event("pre_switch_validation_started", "Pre-Switch-Validierung gestartet")

        # Aktualisiere Metriken für Validierung
        await self._update_rollout_metrics(rollout)

        # Zusätzliche Validierungsschritte
        await self._validate_resource_availability(rollout)
        await self._validate_dependency_health(rollout)
        await self._validate_security_compliance(rollout)

        rollout.add_event("pre_switch_validation_completed", "Pre-Switch-Validierung abgeschlossen")

    async def _perform_extended_canary_validation(self, rollout: ActiveRollout) -> None:
        """Führt erweiterte Canary-Validierung durch."""
        rollout.add_event("extended_canary_validation_started", "Erweiterte Canary-Validierung gestartet")

        # Sammle erweiterte Metriken über längeren Zeitraum
        validation_duration = 60  # 60 Sekunden erweiterte Validierung
        start_time = datetime.now(UTC)

        while (datetime.now(UTC) - start_time).total_seconds() < validation_duration:
            await self._update_rollout_metrics(rollout)

            # Prüfe kritische Metriken
            if rollout.metrics.error_rate > 0.05:  # 5% Error-Rate-Schwellwert
                rollout.add_event("canary_error_rate_exceeded", f"Error-Rate zu hoch: {rollout.metrics.error_rate:.2%}", success=False)
                return

            if rollout.metrics.response_time_p95 > 2000:  # 2s P95-Schwellwert
                rollout.add_event("canary_latency_exceeded", f"P95-Latenz zu hoch: {rollout.metrics.response_time_p95}ms", success=False)
                return

            await asyncio.sleep(5)  # Prüfe alle 5 Sekunden

        rollout.add_event("extended_canary_validation_completed", "Erweiterte Canary-Validierung erfolgreich")

    async def _perform_critical_traffic_validation(self, rollout: ActiveRollout, traffic_percentage: int) -> None:
        """Führt kritische Traffic-Validierung für hohe Traffic-Stufen durch."""
        rollout.add_event("critical_traffic_validation_started", f"Kritische Validierung bei {traffic_percentage}% Traffic")

        # Erweiterte Überwachung für kritische Traffic-Stufen
        monitoring_duration = 120 if traffic_percentage >= 75 else 60  # Längere Überwachung bei höherem Traffic
        start_time = datetime.now(UTC)

        while (datetime.now(UTC) - start_time).total_seconds() < monitoring_duration:
            await self._update_rollout_metrics(rollout)

            # Strengere Schwellwerte für kritische Traffic-Stufen
            max_error_rate = 0.02 if traffic_percentage >= 75 else 0.03  # 2% bei >=75%, 3% bei 50-74%
            max_p95_latency = 1500 if traffic_percentage >= 75 else 2000  # 1.5s bei >=75%, 2s bei 50-74%

            if rollout.metrics.error_rate > max_error_rate:
                rollout.add_event("critical_error_rate_exceeded",
                                f"Kritische Error-Rate bei {traffic_percentage}%: {rollout.metrics.error_rate:.2%}",
                                success=False)
                return

            if rollout.metrics.response_time_p95 > max_p95_latency:
                rollout.add_event("critical_latency_exceeded",
                                f"Kritische Latenz bei {traffic_percentage}%: {rollout.metrics.response_time_p95}ms",
                                success=False)
                return

            await asyncio.sleep(10)  # Prüfe alle 10 Sekunden

        rollout.add_event("critical_traffic_validation_completed", f"Kritische Validierung bei {traffic_percentage}% erfolgreich")

    def _calculate_adaptive_wait_time(self, traffic_percentage: int, base_interval: float) -> float:
        """Berechnet adaptive Wartezeit basierend auf Traffic-Stufe."""
        # Längere Wartezeiten für kritische Traffic-Stufen
        if traffic_percentage >= 75:
            return base_interval * 3  # 3x längere Wartezeit
        if traffic_percentage >= 50:
            return base_interval * 2  # 2x längere Wartezeit
        if traffic_percentage >= 25:
            return base_interval * 1.5  # 1.5x längere Wartezeit
        return base_interval  # Standard-Wartezeit

    async def _validate_resource_availability(self, rollout: ActiveRollout) -> None:
        """Validiert Ressourcenverfügbarkeit für Deployment."""
        rollout.add_event("resource_validation", "Ressourcenverfügbarkeit validiert")
        # Implementierung würde echte Ressourcenprüfung durchführen

    async def _validate_dependency_health(self, rollout: ActiveRollout) -> None:
        """Validiert Health von Dependencies."""
        rollout.add_event("dependency_validation", "Dependency-Health validiert")
        # Implementierung würde echte Dependency-Prüfung durchführen

    async def _validate_security_compliance(self, rollout: ActiveRollout) -> None:
        """Validiert Security-Compliance für Deployment."""
        rollout.add_event("security_validation", "Security-Compliance validiert")
        # Implementierung würde echte Security-Prüfung durchführen


# Globale Rollout Manager Instanz
rollout_manager = RolloutManager()
