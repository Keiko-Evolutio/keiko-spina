# backend/agents/lifecycle/agent_lifecycle_manager.py
"""Agent-Lifecycle-Manager für vollständiges Lifecycle-Management.

Implementiert State-Machine, Task-Handling, Suspend/Resume-Mechanismen
und Integration mit Agent-Registry und Monitoring-Systemen.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from kei_logging import get_logger
from observability import trace_function

from ..protocols.core import AgentExecutionContext
from .models import (
    AgentEvent,
    AgentLifecycleState,
    AgentState,
    AgentTask,
    EventHandler,
    EventType,
    LifecycleCallback,
    TaskExecutionResult,
    TaskHandler,
    TaskQueueConfig,
    TaskStatus,
)

logger = get_logger(__name__)





class AgentLifecycleManager:
    """Manager für vollständiges Agent-Lifecycle-Management."""

    def __init__(self) -> None:
        """Initialisiert Lifecycle-Manager."""
        self._agents: dict[str, AgentState] = {}
        self._lifecycle_callbacks: list[LifecycleCallback] = []
        self._task_handlers: dict[str, TaskHandler] = {}
        self._event_handlers: list[EventHandler] = []
        self._task_queue_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Warmup and shutdown callbacks
        self._warmup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._shutdown_callbacks: list[Callable[[], Awaitable[None]]] = []

        # Task- und Event-Handling (Lazy Import um zirkuläre Imports zu vermeiden)
        self._task_queue = None
        self._event_bus = None

        # Konfiguration
        self._max_concurrent_tasks = 10
        self._task_cleanup_interval = 300  # 5 Minuten
        self._heartbeat_timeout = 180  # 3 Minuten

        # Gültige State-Transitions
        self._valid_transitions = {
            AgentLifecycleState.UNREGISTERED: {AgentLifecycleState.REGISTERED},
            AgentLifecycleState.REGISTERED: {
                AgentLifecycleState.INITIALIZING,
                AgentLifecycleState.RUNNING,  # Direkter Übergang für Tests
                AgentLifecycleState.ERROR,
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.INITIALIZING: {
                AgentLifecycleState.RUNNING,
                AgentLifecycleState.ERROR,
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.RUNNING: {
                AgentLifecycleState.SUSPENDED,
                AgentLifecycleState.STOPPING,
                AgentLifecycleState.ERROR,
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.SUSPENDED: {
                AgentLifecycleState.RUNNING,
                AgentLifecycleState.STOPPING,
                AgentLifecycleState.ERROR,
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.STOPPING: {
                AgentLifecycleState.STOPPED,
                AgentLifecycleState.ERROR,
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.STOPPED: {
                AgentLifecycleState.RUNNING,
                AgentLifecycleState.ERROR,
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.ERROR: {
                AgentLifecycleState.RUNNING,  # Recovery möglich
                AgentLifecycleState.TERMINATED,
            },
            AgentLifecycleState.TERMINATED: set(),  # Endstatus
        }

    def _ensure_components_initialized(self) -> None:
        """Stellt sicher, dass Task-Queue und Event-Bus initialisiert sind."""
        if self._task_queue is None:
            from .task_event_handler import TaskQueue
            self._task_queue = TaskQueue(TaskQueueConfig())

        if self._event_bus is None:
            from .task_event_handler import EventBus
            self._event_bus = EventBus()

    async def start(self) -> None:
        """Startet Lifecycle-Manager."""
        logger.info("Starte Agent-Lifecycle-Manager")

        # Initialisiere Komponenten
        self._ensure_components_initialized()

        # Background-Tasks starten
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor_loop())

        self._background_tasks.add(cleanup_task)
        self._background_tasks.add(heartbeat_task)

        # Task-Completion-Callbacks
        cleanup_task.add_done_callback(self._background_tasks.discard)
        heartbeat_task.add_done_callback(self._background_tasks.discard)

    async def stop(self) -> None:
        """Stoppt Lifecycle-Manager."""
        logger.info("Stoppe Agent-Lifecycle-Manager")

        self._shutdown_event.set()

        # Alle Background-Tasks beenden
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Alle aktiven Agents terminieren
        for agent_id in list(self._agents.keys()):
            try:
                await self.terminate_agent(agent_id, reason="Lifecycle-Manager shutdown")
            except Exception as e:
                logger.error(f"Fehler beim Terminieren von Agent {agent_id}: {e}")

    @trace_function("agent.lifecycle.register")
    async def register_agent(self, agent_id: str, capabilities: set[str] | None = None) -> bool:
        """Registriert neuen Agent.

        Args:
            agent_id: Eindeutige Agent-ID
            capabilities: Initiale Capabilities

        Returns:
            True wenn erfolgreich registriert
        """
        self._ensure_components_initialized()

        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} bereits registriert")
            return False

        try:
            agent_state = AgentState(agent_id=agent_id)
            if capabilities:
                agent_state.advertised_capabilities = capabilities

            self._agents[agent_id] = agent_state

            await self._transition_state(
                agent_id, AgentLifecycleState.REGISTERED, reason="Agent registration"
            )

            logger.info(f"Agent {agent_id} erfolgreich registriert")
            return True

        except Exception as e:
            logger.error(f"Agent-Registrierung fehlgeschlagen für {agent_id}: {e}")
            return False

    @trace_function("agent.lifecycle.initialize")
    async def initialize_agent(
        self, agent_id: str, context: AgentExecutionContext | None = None
    ) -> bool:
        """Initialisiert registrierten Agent.

        Args:
            agent_id: Agent-ID
            context: Execution-Context

        Returns:
            True wenn erfolgreich initialisiert
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.error(f"Agent {agent_id} nicht registriert")
            return False

        try:
            await self._transition_state(
                agent_id, AgentLifecycleState.INITIALIZING, reason="Agent initialization"
            )

            # Context speichern falls vorhanden
            if context:
                agent_state.persisted_state["execution_context"] = {
                    "session_id": context.session_id,
                    "thread_id": context.thread_id,
                    "user_id": context.user_id,
                }

            logger.info(f"Agent {agent_id} erfolgreich initialisiert")
            return True

        except Exception as e:
            logger.error(f"Agent-Initialisierung fehlgeschlagen für {agent_id}: {e}")
            await self._transition_state(agent_id, AgentLifecycleState.ERROR, reason=str(e))
            return False

    @trace_function("agent.lifecycle.activate")
    async def activate_agent(self, agent_id: str) -> bool:
        """Aktiviert initialisierten Agent.

        Args:
            agent_id: Agent-ID

        Returns:
            True wenn erfolgreich aktiviert
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.error(f"Agent {agent_id} nicht gefunden")
            return False

        try:
            await self._transition_state(
                agent_id, AgentLifecycleState.RUNNING, reason="Agent activation"
            )

            # Task-Processing starten (aber Fehler nicht propagieren)
            try:
                task_processor = asyncio.create_task(self._process_agent_tasks(agent_id))
                self._background_tasks.add(task_processor)
                task_processor.add_done_callback(self._background_tasks.discard)
            except Exception as task_error:
                logger.warning(f"Task-Processing für Agent {agent_id} konnte nicht gestartet werden: {task_error}")
                # Agent bleibt trotzdem RUNNING

            logger.info(f"Agent {agent_id} erfolgreich aktiviert")
            return True

        except Exception as e:
            logger.error(f"Agent-Aktivierung fehlgeschlagen für {agent_id}: {e}")
            await self._transition_state(agent_id, AgentLifecycleState.TERMINATED, reason=str(e))
            return False

    async def _transition_state(
        self, agent_id: str, new_state: AgentLifecycleState, reason: str | None = None
    ) -> None:
        """Führt State-Transition durch."""
        agent_state = self._get_agent_state(agent_id)
        current_state = agent_state.current_state

        self._validate_state_transition(agent_id, current_state, new_state)
        agent_state.add_transition(new_state, reason)

        await self._notify_state_change(agent_id, current_state, new_state, reason)
        logger.debug(f"Agent {agent_id} Transition: {current_state.value} -> {new_state.value}")

    def _get_agent_state(self, agent_id: str) -> AgentState:
        """Holt Agent-State oder wirft Fehler."""
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            raise ValueError(f"Agent {agent_id} nicht gefunden")
        return agent_state

    def _validate_state_transition(
        self, agent_id: str, current_state: AgentLifecycleState, new_state: AgentLifecycleState
    ) -> None:
        """Validiert State-Transition."""
        valid_next_states = self._valid_transitions.get(current_state, set())
        if new_state not in valid_next_states:
            raise ValueError(
                f"Ungültige Transition für Agent {agent_id}: "
                f"{current_state.value} -> {new_state.value}"
            )

    async def _notify_state_change(
        self,
        agent_id: str,
        current_state: AgentLifecycleState,
        new_state: AgentLifecycleState,
        reason: str | None
    ) -> None:
        """Benachrichtigt über State-Änderung."""
        # Event publizieren
        event = AgentLifecycleManager._create_lifecycle_event(agent_id, current_state, new_state, reason)
        await self._publish_event(event)

        # Lifecycle-Callbacks aufrufen
        await self._execute_lifecycle_callbacks(agent_id, current_state, new_state)

    @staticmethod
    def _create_lifecycle_event(
        agent_id: str,
        current_state: AgentLifecycleState,
        new_state: AgentLifecycleState,
        reason: str | None
    ) -> AgentEvent:
        """Erstellt Lifecycle-Event.

        Args:
            agent_id: Agent-ID für das Event
            current_state: Aktueller Lifecycle-State
            new_state: Neuer Lifecycle-State
            reason: Grund für State-Änderung

        Returns:
            AgentEvent für Lifecycle-Änderung
        """
        return AgentEvent(
            event_id=f"{agent_id}_{int(time.time())}",
            event_type=EventType.LIFECYCLE_CHANGED,
            agent_id=agent_id,
            data={"from_state": current_state.value, "to_state": new_state.value, "reason": reason},
        )

    async def _execute_lifecycle_callbacks(
        self, agent_id: str, current_state: AgentLifecycleState, new_state: AgentLifecycleState
    ) -> None:
        """Führt alle Lifecycle-Callbacks aus."""
        for callback in self._lifecycle_callbacks:
            try:
                await callback(agent_id, current_state, new_state)
            except Exception as e:
                logger.error(f"Lifecycle-Callback-Fehler: {e}")

    async def _publish_event(self, event: AgentEvent) -> None:
        """Publiziert Agent-Event."""
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event-Handler-Fehler: {e}")

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Task-Cleanup."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_completed_tasks()
                await asyncio.sleep(self._task_cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup-Loop-Fehler: {e}")
                await asyncio.sleep(60)

    async def _heartbeat_monitor_loop(self) -> None:
        """Background-Loop für Heartbeat-Monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_agent_heartbeats()
                await asyncio.sleep(60)  # Prüfe jede Minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat-Monitor-Fehler: {e}")
                await asyncio.sleep(60)

    async def _cleanup_completed_tasks(self) -> None:
        """Bereinigt abgeschlossene Tasks."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=1)

        for agent_state in self._agents.values():
            # Entferne alte completed tasks
            agent_state.completed_tasks = [
                task
                for task in agent_state.completed_tasks
                if task.completed_at and task.completed_at > cutoff_time
            ]

    async def _check_agent_heartbeats(self) -> None:
        """Prüft Agent-Heartbeats und markiert unresponsive Agents."""
        timeout_threshold = datetime.now(UTC) - timedelta(seconds=self._heartbeat_timeout)

        for agent_id, agent_state in self._agents.items():
            if (
                agent_state.current_state == AgentLifecycleState.RUNNING
                and agent_state.last_heartbeat
                and agent_state.last_heartbeat < timeout_threshold
            ):

                logger.warning(f"Agent {agent_id} Heartbeat-Timeout")
                try:
                    await self._transition_state(
                        agent_id, AgentLifecycleState.ERROR, reason="Heartbeat timeout"
                    )
                except Exception as e:
                    logger.error(f"Fehler beim Markieren von Agent {agent_id} als ERROR: {e}")

    # Public Interface Methods

    def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Gibt Agent-State zurück."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> dict[str, AgentState]:
        """Gibt alle Agent-States zurück."""
        return self._agents.copy()

    @trace_function("agent.lifecycle.start")
    async def start_agent(self, agent_id: str) -> bool:
        """Startet Agent (Alias für activate_agent)."""
        return await self.activate_agent(agent_id)

    @trace_function("agent.lifecycle.stop")
    async def stop_agent(self, agent_id: str) -> bool:
        """Stoppt Agent."""
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.warning(f"Agent {agent_id} nicht gefunden")
            return False

        if agent_state.current_state not in [AgentLifecycleState.RUNNING, AgentLifecycleState.SUSPENDED]:
            logger.warning(f"Agent {agent_id} kann nicht gestoppt werden (State: {agent_state.current_state})")
            return False

        try:
            # Transition zu STOPPING
            await self._transition_state(agent_id, AgentLifecycleState.STOPPING, "Agent stop requested")

            # Cleanup
            await self._cleanup_agent_resources(agent_id)

            # Transition zu STOPPED
            await self._transition_state(agent_id, AgentLifecycleState.STOPPED, "Agent stopped")

            logger.info(f"Agent {agent_id} erfolgreich gestoppt")
            return True

        except Exception as e:
            logger.error(f"Agent-Stop fehlgeschlagen für {agent_id}: {e}")
            return False

    @trace_function("agent.lifecycle.deregister")
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregistriert Agent."""
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.warning(f"Agent {agent_id} nicht gefunden")
            return False

        # Prüfe, ob Agent bereits terminiert ist
        if agent_state.current_state == AgentLifecycleState.TERMINATED:
            logger.info(f"Agent {agent_id} ist bereits terminiert")
            return True

        try:
            # Stoppe Agent falls noch aktiv
            if agent_state.current_state in [AgentLifecycleState.RUNNING, AgentLifecycleState.SUSPENDED]:
                await self.stop_agent(agent_id)

            # Entferne aus Registry
            del self._agents[agent_id]

            logger.info(f"Agent {agent_id} erfolgreich deregistriert")
            return True

        except Exception as e:
            logger.error(f"Agent-Deregistrierung fehlgeschlagen für {agent_id}: {e}")
            return False

    async def _cleanup_agent_resources(self, agent_id: str) -> None:
        """Bereinigt Agent-Ressourcen."""
        try:
            # Stoppe alle aktiven Tasks für den Agent
            if hasattr(self, "_task_queue") and self._task_queue:
                # Entferne pending Tasks für diesen Agent
                queue_lock = self._task_queue.get_queue_lock()
                async with queue_lock:
                    removed_count = self._task_queue.remove_tasks_by_agent(agent_id)
                    logger.debug(f"Entfernte {removed_count} pending Tasks für Agent {agent_id}")

                # Cancelle laufende Tasks für diesen Agent
                running_tasks_to_cancel = self._task_queue.get_running_tasks_by_agent(agent_id)

                for task in running_tasks_to_cancel:
                    if hasattr(task, "cancel"):
                        task.cancel()

            logger.debug(f"Agent-Ressourcen für {agent_id} bereinigt")

        except Exception as e:
            logger.error(f"Fehler beim Bereinigen der Agent-Ressourcen für {agent_id}: {e}")

    def register_lifecycle_callback(self, callback: LifecycleCallback) -> None:
        """Registriert Lifecycle-Callback."""
        self._lifecycle_callbacks.append(callback)

    def register_task_handler(self, task_type: str, handler: TaskHandler) -> None:
        """Registriert Task-Handler."""
        self._task_handlers[task_type] = handler

    def register_event_handler(self, handler: EventHandler) -> None:
        """Registriert Event-Handler."""
        self._ensure_components_initialized()
        self._event_handlers.append(handler)
        # Registriere auch im EventBus mit allen Event-Types
        subscription_id = f"handler_{len(self._event_handlers)}"
        all_event_types = {EventType.HEARTBEAT, EventType.TASK_COMPLETED, EventType.TASK_FAILED,
                          EventType.LIFECYCLE_CHANGED, EventType.ERROR_OCCURRED}
        self._event_bus.subscribe(subscription_id, all_event_types, handler)

    def register_warmup_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Registriert Warmup-Callback."""
        self._warmup_callbacks.append(callback)
        logger.debug("Warmup-Callback registriert")

    def register_shutdown_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Registriert Shutdown-Callback."""
        self._shutdown_callbacks.append(callback)
        logger.debug("Shutdown-Callback registriert")

    async def run_global_warmup(self) -> None:
        """Führt alle registrierten Warmup-Callbacks aus."""
        logger.info(f"Führe {len(self._warmup_callbacks)} Warmup-Callbacks aus")

        for i, callback in enumerate(self._warmup_callbacks):
            try:
                logger.debug(f"Führe Warmup-Callback {i+1}/{len(self._warmup_callbacks)} aus")
                await callback()
                logger.debug(f"Warmup-Callback {i+1} erfolgreich abgeschlossen")
            except Exception as e:
                logger.error(f"Warmup-Callback {i+1} fehlgeschlagen: {e}")
                # Continue with other callbacks even if one fails

        logger.info("Globaler Warmup abgeschlossen")

    async def run_global_shutdown(self) -> None:
        """Führt alle registrierten Shutdown-Callbacks aus."""
        logger.info(f"Führe {len(self._shutdown_callbacks)} Shutdown-Callbacks aus")

        for i, callback in enumerate(self._shutdown_callbacks):
            try:
                logger.debug(f"Führe Shutdown-Callback {i+1}/{len(self._shutdown_callbacks)} aus")
                await callback()
                logger.debug(f"Shutdown-Callback {i+1} erfolgreich abgeschlossen")
            except Exception as e:
                logger.error(f"Shutdown-Callback {i+1} fehlgeschlagen: {e}")
                # Continue with other callbacks even if one fails

        logger.info("Globaler Shutdown abgeschlossen")

    @trace_function("agent.lifecycle.submit_task")
    async def submit_task(self, task: AgentTask) -> bool:
        """Submittiert Task für Agent."""
        self._ensure_components_initialized()
        return await self.handle_task(task.agent_id, task)

    @trace_function("agent.lifecycle.suspend")
    async def suspend_agent(
        self,
        agent_id: str,
        reason: str | None = None,
        wait_for_tasks: bool = True,
        timeout_seconds: int = 60,
    ) -> bool:
        """Suspendiert aktiven Agent.

        Args:
            agent_id: Agent-ID
            reason: Grund für Suspend
            wait_for_tasks: Warten auf Task-Completion
            timeout_seconds: Timeout für Task-Completion

        Returns:
            True wenn erfolgreich suspendiert
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state or agent_state.current_state != AgentLifecycleState.RUNNING:
            logger.error(
                f"Agent {agent_id} kann nicht suspendiert werden (State: {agent_state.current_state if agent_state else 'not found'})"
            )
            return False

        try:
            # Warte auf Task-Completion falls gewünscht
            if wait_for_tasks and agent_state.active_tasks:
                logger.info(
                    f"Warte auf Completion von {len(agent_state.active_tasks)} Tasks für Agent {agent_id}"
                )

                start_time = time.time()
                while agent_state.active_tasks and (time.time() - start_time) < timeout_seconds:
                    await asyncio.sleep(1)

                if agent_state.active_tasks:
                    logger.warning(f"Timeout beim Warten auf Task-Completion für Agent {agent_id}")

            # State persistieren
            agent_state.persisted_state.update(
                {
                    "suspended_at": datetime.now(UTC).isoformat(),
                    "pending_tasks_count": len(agent_state.pending_tasks),
                    "active_tasks_count": len(agent_state.active_tasks),
                }
            )

            agent_state.suspended_at = datetime.now(UTC)
            agent_state.suspend_reason = reason

            await self._transition_state(
                agent_id, AgentLifecycleState.SUSPENDED, reason=reason or "Manual suspend"
            )

            logger.info(f"Agent {agent_id} erfolgreich suspendiert")
            return True

        except Exception as e:
            logger.error(f"Agent-Suspend fehlgeschlagen für {agent_id}: {e}")
            return False

    @trace_function("agent.lifecycle.resume")
    async def resume_agent(self, agent_id: str) -> bool:
        """Nimmt suspendierten Agent wieder in Betrieb.

        Args:
            agent_id: Agent-ID

        Returns:
            True wenn erfolgreich resumed
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state or agent_state.current_state != AgentLifecycleState.SUSPENDED:
            logger.error(
                f"Agent {agent_id} kann nicht resumed werden (State: {agent_state.current_state if agent_state else 'not found'})"
            )
            return False

        try:
            # State wiederherstellen
            if "suspended_at" in agent_state.persisted_state:
                suspended_duration = datetime.now(UTC) - agent_state.suspended_at
                logger.info(
                    f"Agent {agent_id} war {suspended_duration.total_seconds():.1f}s suspendiert"
                )

            agent_state.suspended_at = None
            agent_state.suspend_reason = None

            await self._transition_state(
                agent_id, AgentLifecycleState.RUNNING, reason="Agent resume"
            )

            # Task-Processing wieder starten
            task_processor = asyncio.create_task(self._process_agent_tasks(agent_id))
            self._background_tasks.add(task_processor)
            task_processor.add_done_callback(self._background_tasks.discard)

            logger.info(f"Agent {agent_id} erfolgreich resumed")
            return True

        except Exception as e:
            logger.error(f"Agent-Resume fehlgeschlagen für {agent_id}: {e}")
            return False

    @trace_function("agent.lifecycle.terminate")
    async def terminate_agent(
        self, agent_id: str, reason: str | None = None, force: bool = False
    ) -> bool:
        """Terminiert Agent gracefully.

        Args:
            agent_id: Agent-ID
            reason: Grund für Termination
            force: Erzwinge Termination ohne Cleanup

        Returns:
            True wenn erfolgreich terminiert
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.warning(f"Agent {agent_id} nicht gefunden für Termination")
            return False

        if agent_state.current_state.value == AgentLifecycleState.TERMINATED.value:
            logger.info(f"Agent {agent_id} bereits terminiert")
            return True

        try:
            # Transition zu TERMINATED
            await self._transition_state(
                agent_id, AgentLifecycleState.TERMINATED, reason=reason or "Manual termination"
            )

            if not force:
                # Graceful Cleanup
                await self._cleanup_agent_resources(agent_id)

            # Finale Transition zu TERMINATED
            await self._transition_state(
                agent_id, AgentLifecycleState.TERMINATED, reason="Termination completed"
            )

            logger.info(f"Agent {agent_id} erfolgreich terminiert")
            return True

        except Exception as e:
            logger.error(f"Agent-Termination fehlgeschlagen für {agent_id}: {e}")

            # Prüfe erneut, ob Agent bereits terminiert ist (Race-Condition)
            agent_state = self._agents.get(agent_id)
            if agent_state and agent_state.current_state.value == AgentLifecycleState.TERMINATED.value:
                logger.info(f"Agent {agent_id} bereits terminiert (nach Exception)")
                return True
            return False



    @trace_function("agent.lifecycle.advertise_capabilities")
    async def advertise_capabilities(
        self, agent_id: str, capabilities: set[str], replace_existing: bool = False
    ) -> bool:
        """Bewirbt Agent-Capabilities dynamisch.

        Args:
            agent_id: Agent-ID
            capabilities: Set von Capabilities
            replace_existing: Ersetze bestehende Capabilities

        Returns:
            True wenn erfolgreich beworben
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.error(f"Agent {agent_id} nicht gefunden für Capability-Advertisement")
            return False

        try:
            if replace_existing:
                old_capabilities = agent_state.advertised_capabilities.copy()
                agent_state.advertised_capabilities = capabilities
            else:
                old_capabilities = agent_state.advertised_capabilities.copy()
                agent_state.advertised_capabilities.update(capabilities)

            # Event publizieren
            event = AgentEvent(
                event_id=f"{agent_id}_capabilities_{int(time.time())}",
                event_type=EventType.CAPABILITY_ADVERTISED,
                agent_id=agent_id,
                data={
                    "old_capabilities": list(old_capabilities),
                    "new_capabilities": list(agent_state.advertised_capabilities),
                    "added_capabilities": list(capabilities - old_capabilities),
                    "replace_existing": replace_existing,
                },
            )
            await self._event_bus.publish(event)

            logger.info(f"Agent {agent_id} Capabilities beworben: {capabilities}")
            return True

        except Exception as e:
            logger.error(f"Capability-Advertisement fehlgeschlagen für Agent {agent_id}: {e}")
            return False

    @trace_function("agent.lifecycle.handle_task")
    async def handle_task(self, agent_id: str, task: AgentTask) -> bool:
        """Behandelt Task für Agent.

        Args:
            agent_id: Agent-ID
            task: Zu behandelnde Task

        Returns:
            True wenn Task erfolgreich zur Queue hinzugefügt
        """
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            logger.error(f"Agent {agent_id} nicht gefunden für Task-Handling")
            return False

        if agent_state.current_state not in [
            AgentLifecycleState.RUNNING,
            AgentLifecycleState.SUSPENDED,
        ]:
            logger.warning(
                f"Agent {agent_id} nicht bereit für Tasks (State: {agent_state.current_state.value})"
            )
            return False

        try:
            # Task zur Agent-Queue hinzufügen
            agent_state.pending_tasks.append(task)

            # Event publizieren
            event = AgentEvent(
                event_id=f"{agent_id}_task_{task.task_id}",
                event_type=EventType.TASK_RECEIVED,
                agent_id=agent_id,
                data={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "priority": task.priority.value,
                    "timeout_seconds": task.timeout_seconds,
                },
            )
            await self._event_bus.publish(event)

            logger.debug(f"Task {task.task_id} für Agent {agent_id} zur Queue hinzugefügt")
            return True

        except Exception as e:
            logger.error(f"Task-Handling fehlgeschlagen für Agent {agent_id}: {e}")
            return False

    @trace_function("agent.lifecycle.handle_event")
    async def handle_event(self, event: AgentEvent) -> None:
        """Behandelt Agent-Event.

        Args:
            event: Zu behandelndes Event
        """
        try:
            # Event über Event-Bus publizieren
            await self._event_bus.publish(event)

            # Rufe auch direkt registrierte Event-Handler auf
            for handler in self._event_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event-Handler-Fehler: {e}")

            # Spezielle Event-Behandlung
            if event.event_type.value == EventType.HEARTBEAT.value:
                await self._handle_heartbeat_event(event)
            elif event.event_type.value == EventType.ERROR_OCCURRED.value:
                await self._handle_error_event(event)

            logger.debug(f"Event {event.event_id} behandelt: {event.event_type.value}")

        except Exception as e:
            logger.error(f"Event-Handling fehlgeschlagen für Event {event.event_id}: {e}")

    async def _handle_heartbeat_event(self, event: AgentEvent) -> None:
        """Behandelt Heartbeat-Event."""
        agent_state = self._agents.get(event.agent_id)
        if agent_state:
            agent_state.last_heartbeat = event.timestamp
            logger.debug(f"Heartbeat für Agent {event.agent_id} aktualisiert")

    async def _handle_error_event(self, event: AgentEvent) -> None:
        """Behandelt Error-Event."""
        agent_state = self._agents.get(event.agent_id)
        if agent_state:
            agent_state.total_errors += 1

            # Bei kritischen Fehlern Agent in ERROR-State versetzen
            error_severity = event.data.get("severity", "medium")
            if error_severity == "critical":
                try:
                    await self._transition_state(
                        event.agent_id,
                        AgentLifecycleState.ERROR,
                        reason=f"Critical error: {event.data.get('error_message', 'Unknown error')}",
                    )
                except Exception as e:
                    logger.error(
                        f"Fehler beim Versetzen von Agent {event.agent_id} in ERROR-State: {e}"
                    )

    async def _process_agent_tasks(self, agent_id: str) -> None:
        """Verarbeitet Tasks für spezifischen Agent."""
        agent_state = self._agents.get(agent_id)
        if not agent_state:
            return

        logger.info(f"Starte Task-Processing für Agent {agent_id}")

        while (
            agent_state.current_state == AgentLifecycleState.RUNNING
            and not self._shutdown_event.is_set()
        ):

            try:
                # Hole nächste Task
                if not agent_state.pending_tasks:
                    await asyncio.sleep(1)
                    continue

                task = agent_state.pending_tasks.pop(0)

                # Task zu aktiven Tasks hinzufügen
                agent_state.active_tasks[task.task_id] = task

                # Task ausführen
                result = await self._execute_agent_task(agent_id, task)

                # Task von aktiven Tasks entfernen
                if task.task_id in agent_state.active_tasks:
                    del agent_state.active_tasks[task.task_id]

                # Task zu completed Tasks hinzufügen
                agent_state.completed_tasks.append(task)
                agent_state.total_tasks_processed += 1

                # Event publizieren
                event_type = (
                    EventType.TASK_COMPLETED
                    if result.status == TaskStatus.COMPLETED
                    else EventType.TASK_FAILED
                )
                event = AgentEvent(
                    event_id=f"{agent_id}_task_result_{task.task_id}",
                    event_type=event_type,
                    agent_id=agent_id,
                    data={
                        "task_id": task.task_id,
                        "status": result.status.value,
                        "execution_time": result.execution_time,
                        "error": result.error,
                    },
                )
                await self._event_bus.publish(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task-Processing-Fehler für Agent {agent_id}: {e}")
                await asyncio.sleep(5)

        logger.info(f"Task-Processing für Agent {agent_id} beendet")

    async def _execute_agent_task(self, _agent_id: str, task: AgentTask) -> TaskExecutionResult:
        """Führt Agent-Task aus."""
        start_time = time.time()

        try:
            # Task-Handler finden
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"Kein Handler für Task-Typ {task.task_type} gefunden")

            # Task ausführen
            result = await handler(task)

            execution_time = (time.time() - start_time) * 1000

            return TaskExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result.__dict__ if hasattr(result, "__dict__") else {"result": str(result)},
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return TaskExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )


# Globaler Lifecycle-Manager
agent_lifecycle_manager = AgentLifecycleManager()
