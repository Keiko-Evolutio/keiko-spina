"""Work‑Stealing für Task‑Queues.

Dieser Dienst beobachtet die Dynamic Registry (queue_length‑Hinweise) und
stiehlt Tasks aus überlasteten Queues, indem er sie in die lokale Queue
umleitet. Er nutzt NATS JetStream via `MessagingService.subscribe`.
"""

from __future__ import annotations

# Optional import - dynamic_registry ist nicht immer verfügbar
try:
    from agents.capabilities.dynamic_registry import dynamic_registry
    DYNAMIC_REGISTRY_AVAILABLE = True
except ImportError:
    # Fallback-Typ für bessere IDE-Unterstützung
    class _DynamicRegistryFallback:
        """Fallback für fehlende Dynamic Registry."""
        _initialized: bool = False
        async def initialize(self) -> None: ...

    dynamic_registry: _DynamicRegistryFallback | None = None
    DYNAMIC_REGISTRY_AVAILABLE = False

# Messaging Service Import
try:
    from services.messaging import get_messaging_service
except ImportError:
    # Fallback wenn Messaging Service nicht verfügbar ist
    def get_messaging_service():
        """Fallback für fehlenden Messaging Service."""
        return
from kei_logging import get_logger
from services.messaging.envelope import BusEnvelope
from services.messaging.naming import subject_for_tasks

from .base import PeriodicServiceBase
from .config import DEFAULT_SCHEDULER_CONFIG, WorkStealingConfig

logger = get_logger(__name__)


# Backward compatibility alias
StealPolicy = WorkStealingConfig


class WorkStealer(PeriodicServiceBase):
    """Work‑Stealer für Task‑Queues mit automatischem Load-Balancing.

    Überwacht die Dynamic Registry und stiehlt Tasks aus überlasteten Queues,
    indem sie in die lokale Queue umgeleitet werden.
    """

    def __init__(
        self,
        *,
        local_queue: str,
        tenant: str | None = None,
        config: WorkStealingConfig | None = None
    ) -> None:
        """Initialisiert WorkStealer.

        Args:
            local_queue: Name der lokalen Queue
            tenant: Tenant-ID für Multi-Tenancy
            config: Work-Stealing-Konfiguration
        """
        self.config = config or DEFAULT_SCHEDULER_CONFIG.work_stealing

        super().__init__(
            interval_seconds=self.config.check_interval_seconds,
            service_name=f"WorkStealer({local_queue})"
        )

        self.local_queue = local_queue
        self.tenant = tenant
        self._bus = get_messaging_service()
        self._active_steals: set[str] = set()

    async def _pre_start(self) -> None:
        """Initialisiert Bus-Service vor dem Start."""
        if self._bus:
            await self._bus.initialize()

    async def _post_stop(self) -> None:
        """Cleanup nach dem Stop."""
        self._active_steals.clear()

    async def _execute_cycle(self) -> None:
        """Führt einen Work-Stealing-Zyklus durch."""
        await self._evaluate_and_schedule_steals()

    async def _evaluate_and_schedule_steals(self) -> None:
        """Evaluiert überlastete Queues und startet Work-Stealing."""
        # Initialisiere Registry falls nötig
        if not await self._ensure_registry_initialized():
            return

        # Finde überlastete Agents
        overutilized_agents = self._find_overutilized_agents()
        if not overutilized_agents:
            return

        # Berechne verfügbare Steal-Slots
        available_slots = self._calculate_available_steal_slots()
        if available_slots <= 0:
            return

        # Starte Stealing für top N überlastete Queues
        await self._start_stealing_from_top_candidates(overutilized_agents, available_slots)

    async def _ensure_registry_initialized(self) -> bool:
        """Stellt sicher dass die Dynamic Registry initialisiert ist.

        Returns:
            True wenn Registry verfügbar ist
        """
        try:
            if dynamic_registry and not getattr(dynamic_registry, "_initialized", False):
                await dynamic_registry.initialize()  # Registry auto-initialized
            return True
        except Exception as e:
            logger.debug(f"Registry-Initialisierung fehlgeschlagen: {e}")
            return False

    def _find_overutilized_agents(self) -> dict[str, int]:
        """Findet überlastete Agents basierend auf Queue-Länge.

        Returns:
            Dictionary mit agent_id -> queue_length für überlastete Agents
        """
        overutilized: dict[str, int] = {}
        agents = getattr(dynamic_registry, "agents", {})

        for agent_id, agent in agents.items():
            if agent_id == self.local_queue:
                continue  # Nicht von sich selbst stehlen

            queue_length = getattr(agent, "queue_length", None)
            if (isinstance(queue_length, int) and
                queue_length >= self.config.min_remote_queue_length):
                overutilized[agent_id] = queue_length

        return overutilized

    def _calculate_available_steal_slots(self) -> int:
        """Berechnet verfügbare Slots für neue Steal-Operationen.

        Returns:
            Anzahl verfügbarer Slots
        """
        return max(0, self.config.max_concurrent_steals - len(self._active_steals))

    async def _start_stealing_from_top_candidates(
        self,
        overutilized_agents: dict[str, int],
        available_slots: int
    ) -> None:
        """Startet Stealing von den am meisten überlasteten Agents.

        Args:
            overutilized_agents: Dictionary mit überlasteten Agents
            available_slots: Anzahl verfügbarer Steal-Slots
        """
        # Sortiere nach Queue-Länge (höchste zuerst)
        candidates = sorted(
            overutilized_agents.items(),
            key=lambda item: item[1],
            reverse=True
        )[:available_slots]

        for victim_agent_id, _ in candidates:
            if victim_agent_id not in self._active_steals:
                await self._attach_steal_subscription(victim_agent_id)

    async def _attach_steal_subscription(self, victim_agent_id: str) -> None:
        """Startet Work-Stealing-Subscription für einen überlasteten Agent.

        Args:
            victim_agent_id: ID des Agents von dem gestohlen werden soll
        """
        subject = subject_for_tasks(
            queue=victim_agent_id,
            version=1,
            tenant=self.tenant
        )

        self._active_steals.add(victim_agent_id)
        logger.info(f"Starte Work-Stealing von {victim_agent_id} via {subject}")

        # Erstelle Handler für gestohlene Tasks
        handler = self._create_steal_handler(victim_agent_id)

        try:
            # Abonniere mit eigener Queue-Gruppe für Stealer
            await self._bus.subscribe(subject, queue="stealers", handler=handler)
        except Exception as e:
            logger.warning(f"Steal-Subscribe fehlgeschlagen für {victim_agent_id}: {e}")
            self._active_steals.discard(victim_agent_id)

    def _create_steal_handler(self, victim_agent_id: str):
        """Erstellt Handler für Work-Stealing von einem spezifischen Agent.

        Args:
            victim_agent_id: ID des Agents von dem gestohlen wird

        Returns:
            Async Handler-Funktion
        """
        async def _steal_handler(envelope: BusEnvelope) -> None:
            try:
                await self._redirect_stolen_task(envelope, victim_agent_id)
            except Exception as e:
                logger.exception(f"Work-Stealing Handler-Fehler für {victim_agent_id}: {e}")

        return _steal_handler

    async def _redirect_stolen_task(
        self,
        original_envelope: BusEnvelope,
        victim_agent_id: str
    ) -> None:
        """Leitet gestohlene Task in lokale Queue um.

        Args:
            original_envelope: Ursprüngliches Task-Envelope
            victim_agent_id: ID des Agents von dem gestohlen wurde
        """
        local_subject = subject_for_tasks(
            queue=self.local_queue,
            version=1,
            tenant=self.tenant
        )

        # Erstelle neues Envelope für lokale Queue
        stolen_envelope = BusEnvelope(
            type="task.stolen",
            subject=local_subject,
            tenant=self.tenant,
            key=original_envelope.key or self.local_queue,
            payload={
                **original_envelope.payload,
                "stolen_from": victim_agent_id
            },
            headers={
                **(original_envelope.headers or {}),
                "X-Stolen": "true"
            },
            corr_id=original_envelope.corr_id,
            causation_id=original_envelope.id,
            traceparent=original_envelope.traceparent,
        )

        await self._bus.publish(stolen_envelope)


__all__ = ["StealPolicy", "WorkStealer", "WorkStealingConfig"]
