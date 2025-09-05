"""Scheduler – Push/Direct vs Pull/Queue mit Backpressure Awareness.

Stellt eine einheitliche `schedule_task` API bereit, die anhand von Agent‑Hints
und In‑Flight‑Metriken entscheidet, ob eine Aufgabe direkt (push) delegiert oder
über eine Queue (pull) eingereiht wird.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from services.messaging import get_messaging_service
from services.messaging.envelope import BusEnvelope
from services.messaging.naming import subject_for_tasks

from .backpressure import decide_mode
from .config import DEFAULT_SCHEDULER_CONFIG, SchedulerConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class SchedulingResult:
    """Ergebnis der Scheduling‑Entscheidung."""

    mode: str
    """Verwendeter Scheduling-Modus ('push', 'pull', 'suspended')"""

    accepted: bool
    """Ob die Task erfolgreich geplant wurde"""

    reason: str | None = None
    """Grund bei Ablehnung oder Fehler"""

    task_id: str | None = None
    """ID der geplanten Task"""

    queue_subject: str | None = None
    """Subject der Queue bei Pull-Modus"""

    direct_result: dict[str, Any] | None = None
    """Ergebnis bei direkter Ausführung (Push-Modus)"""


class Scheduler:
    """Einheitlicher Scheduler für Push/Direct und Pull/Queue.

    Bietet intelligente Scheduling-Entscheidungen basierend auf:
    - Agent-Hints (queue_length, desired_concurrency)
    - Backpressure-Metriken
    - Konfigurierbare Schwellenwerte
    """

    def __init__(
        self,
        *,
        tenant: str | None = None,
        config: SchedulerConfig | None = None
    ) -> None:
        """Initialisiert Scheduler.

        Args:
            tenant: Standard-Tenant für Scheduling-Operationen
            config: Scheduler-Konfiguration (Default: globale Konfiguration)
        """
        self.tenant = tenant or config.default_tenant if config else None
        self.config = config or DEFAULT_SCHEDULER_CONFIG
        self._bus = get_messaging_service()
        self._inflight_by_agent: dict[str, int] = {}

    def _get_inflight_count(self, agent_id: str) -> int:
        """Ermittelt aktuelle In-Flight-Task-Anzahl für Agent.

        Args:
            agent_id: Agent-ID

        Returns:
            Anzahl aktuell laufender Tasks (mindestens 0)
        """
        return max(0, self._inflight_by_agent.get(agent_id, 0))

    def _increment_inflight(self, agent_id: str) -> None:
        """Erhöht In-Flight-Counter für Agent.

        Args:
            agent_id: Agent-ID
        """
        current_count = self._get_inflight_count(agent_id)
        self._inflight_by_agent[agent_id] = current_count + 1

    def _decrement_inflight(self, agent_id: str) -> None:
        """Verringert In-Flight-Counter für Agent.

        Args:
            agent_id: Agent-ID
        """
        current_count = self._get_inflight_count(agent_id)
        self._inflight_by_agent[agent_id] = max(0, current_count - 1)

    async def schedule_task(
        self,
        *,
        agent_id: str,
        task_id: str,
        task_payload: dict[str, Any],
        agent_hints: dict[str, Any] | None = None,
        direct_delegate: Callable | None = None,
        queue_name: str | None = None,
    ) -> SchedulingResult:
        """Entscheidet und führt Scheduling aus.

        Args:
            agent_id: Ziel‑Agent
            task_id: Eindeutige Task‑ID
            task_payload: Nutzlast für Ausführung
            agent_hints: Heartbeat‑/Scaling‑Hinweise (readiness, queue_length, desired_concurrency ...)
            direct_delegate: Callable für direkte Ausführung (push)
            queue_name: Name der Queue (pull); Default: agent_id

        Returns:
            SchedulingResult mit Ausführungsdetails
        """
        await self._bus.initialize()

        # Prüfe Agent-Status
        if self._is_agent_suspended(agent_hints):
            return SchedulingResult(
                mode="suspended",
                accepted=False,
                reason="agent_suspended"
            )

        # Entscheide Scheduling-Modus
        inflight_count = self._get_inflight_count(agent_id)
        mode = decide_mode(
            agent_hints=agent_hints,
            inflight_for_agent=inflight_count,
            config=self.config.backpressure
        )

        # Führe entsprechenden Scheduling-Modus aus
        if mode == "pull":
            return await self._schedule_via_queue(
                agent_id=agent_id,
                task_id=task_id,
                task_payload=task_payload,
                queue_name=queue_name
            )
        return await self._schedule_direct(
            agent_id=agent_id,
            task_id=task_id,
            direct_delegate=direct_delegate
        )

    def _is_agent_suspended(self, agent_hints: dict[str, Any] | None) -> bool:
        """Prüft ob Agent suspendiert ist.

        Args:
            agent_hints: Agent-Hints Dictionary

        Returns:
            True wenn Agent suspendiert ist
        """
        if not isinstance(agent_hints, dict):
            return False
        return bool(agent_hints.get("suspended", False))

    async def _schedule_via_queue(
        self,
        *,
        agent_id: str,
        task_id: str,
        task_payload: dict[str, Any],
        queue_name: str | None
    ) -> SchedulingResult:
        """Führt Queue-basiertes Scheduling durch.

        Args:
            agent_id: Ziel-Agent
            task_id: Task-ID
            task_payload: Task-Daten
            queue_name: Queue-Name (Default: agent_id)

        Returns:
            SchedulingResult für Queue-Modus
        """
        effective_queue_name = queue_name or agent_id
        subject = subject_for_tasks(
            queue=effective_queue_name,
            version=1,
            tenant=self.tenant
        )

        envelope = self._create_task_envelope(
            task_id=task_id,
            agent_id=agent_id,
            task_payload=task_payload,
            subject=subject
        )

        try:
            await self._bus.publish(envelope)
            return SchedulingResult(
                mode="pull",
                accepted=True,
                task_id=task_id,
                queue_subject=subject
            )
        except Exception as e:
            logger.warning(f"Queue-Scheduling fehlgeschlagen für {task_id}: {e}")
            return SchedulingResult(
                mode="pull",
                accepted=False,
                reason=str(e)
            )

    async def _schedule_direct(
        self,
        *,
        agent_id: str,
        task_id: str,
        direct_delegate: Callable | None
    ) -> SchedulingResult:
        """Führt direktes Scheduling durch.

        Args:
            agent_id: Ziel-Agent
            task_id: Task-ID
            direct_delegate: Callable für direkte Ausführung

        Returns:
            SchedulingResult für Push-Modus
        """
        if not callable(direct_delegate):
            return SchedulingResult(
                mode="push",
                accepted=False,
                reason="direct_delegate not callable"
            )

        self._increment_inflight(agent_id)
        try:
            result = await direct_delegate()
            return SchedulingResult(
                mode="push",
                accepted=True,
                task_id=task_id,
                direct_result=result
            )
        except Exception as e:
            logger.warning(f"Direct-Scheduling fehlgeschlagen für {task_id}: {e}")
            return SchedulingResult(
                mode="push",
                accepted=False,
                reason=str(e)
            )
        finally:
            self._decrement_inflight(agent_id)

    def _create_task_envelope(
        self,
        *,
        task_id: str,
        agent_id: str,
        task_payload: dict[str, Any],
        subject: str
    ) -> BusEnvelope:
        """Erstellt BusEnvelope für Task-Scheduling.

        Args:
            task_id: Task-ID
            agent_id: Agent-ID
            task_payload: Task-Daten
            subject: Bus-Subject

        Returns:
            Konfiguriertes BusEnvelope
        """
        return BusEnvelope(
            type="task.enqueued",
            subject=subject,
            tenant=self.tenant,
            key=task_payload.get("ordering_key") or agent_id,
            payload={
                "task_id": task_id,
                "agent_id": agent_id,
                "data": task_payload,
            },
            headers={"Idempotency-Key": task_id},
            corr_id=task_payload.get("corr_id"),
            causation_id=task_payload.get("causation_id"),
            traceparent=task_payload.get("traceparent"),
        )


__all__ = ["Scheduler", "SchedulingResult"]
