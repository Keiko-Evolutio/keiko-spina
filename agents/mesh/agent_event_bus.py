"""Agent Event Bus mit erweiterten Features.

Features:
- Idempotenz und Event-Sourcing
- Asynchrone Subscriptions und Callbacks
- Thread-Safety und Performance-Optimierung
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .mesh_constants import MAX_EVENT_HISTORY_SIZE
from .utils import HashGenerator, IdempotencyManager, create_thread_safe_counter, increment_counter

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """Agent-Event mit erweiterten Features."""
    event_type: str
    payload: dict[str, Any]
    agent_id: str | None = None
    idempotency_key: str | None = None
    correlation_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class EventSubscription:
    """Event-Subscription für Callback-System."""
    subscription_id: str
    event_types: set[str]
    callback: Callable[[AgentEvent], Awaitable[None]]
    agent_id_filter: str | None = None
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AgentEventBus:
    """Agent Event Bus.

    Features:
    - Idempotenz mit Hash-basierter Duplikatserkennung
    - Asynchrone Subscriptions mit Callback-System
    - Event-Sourcing mit Replay-Funktionalität
    - Thread-Safety für synchrone und asynchrone Operationen
    """

    def __init__(self, max_history_size: int = MAX_EVENT_HISTORY_SIZE) -> None:
        """Initialisiert den Event-Bus."""
        # Event-Storage
        self._events: deque[AgentEvent] = deque(maxlen=max_history_size)
        self._lock = threading.RLock()

        # Idempotenz-Management
        self._idempotency_manager = IdempotencyManager()

        # Subscription-System
        self._subscriptions: dict[str, EventSubscription] = {}
        self._async_lock = asyncio.Lock()

        # Thread-sichere Statistiken
        self._published_counter = create_thread_safe_counter()
        self._callback_error_counter = create_thread_safe_counter()

    @staticmethod
    def _generate_hash(event: AgentEvent) -> str:
        """Erzeugt eindeutigen Hash für Idempotenz-Prüfung.

        Args:
            event: Agent-Event für das der Hash generiert werden soll

        Returns:
            Eindeutiger Hash-String für Idempotenz-Prüfung
        """
        return HashGenerator.generate_idempotency_hash(
            event_type=event.event_type,
            payload=event.payload,
            idempotency_key=event.idempotency_key,
            agent_id=event.agent_id
        )

    def publish(self, event: AgentEvent) -> bool:
        """Publiziert ein Event synchron mit Idempotenz-Prüfung."""
        with self._lock:
            event_hash = self._generate_hash(event)

            # Duplikatsprüfung mit Idempotency-Manager
            if self._idempotency_manager.is_duplicate(event_hash):
                logger.debug(f"Duplikat-Event ignoriert: {event.event_type}")
                return False

            # Event speichern
            self._events.append(event)
            increment_counter(self._published_counter)

            logger.debug(f"Event publiziert: {event.event_type} (Agent: {event.agent_id})")
            return True

    async def publish_async(self, event: AgentEvent) -> bool:
        """Publiziert ein Event asynchron mit Subscription-Benachrichtigung."""
        # Synchrone Publikation
        published = self.publish(event)
        if not published:
            return False

        # Asynchrone Subscription-Benachrichtigung
        await self._notify_subscribers(event)
        return True


    async def _notify_subscribers(self, event: AgentEvent) -> None:
        """Benachrichtigt alle passenden Subscribers über das Event."""
        if not self._subscriptions:
            return

        async with self._async_lock:
            for subscription in self._subscriptions.values():
                if not subscription.active:
                    continue

                # Event-Type-Filter prüfen
                if event.event_type not in subscription.event_types:
                    continue

                # Agent-ID-Filter prüfen
                if (subscription.agent_id_filter and
                    subscription.agent_id_filter != event.agent_id):
                    continue

                # Callback ausführen
                try:
                    await subscription.callback(event)
                except Exception as e:
                    increment_counter(self._callback_error_counter)
                    logger.error(
                        f"Callback-Fehler für Subscription {subscription.subscription_id}: {e}"
                    )

    def subscribe(
        self,
        subscription_id: str,
        event_types: set[str],
        callback: Callable[[AgentEvent], Awaitable[None]],
        agent_id_filter: str | None = None,
    ) -> bool:
        """Abonniert Events mit Callback-Funktion."""
        if subscription_id in self._subscriptions:
            logger.warning(f"Subscription {subscription_id} bereits vorhanden")
            return False

        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_types=event_types,
            callback=callback,
            agent_id_filter=agent_id_filter,
        )

        self._subscriptions[subscription_id] = subscription
        logger.info(f"Subscription {subscription_id} für {len(event_types)} Event-Typen erstellt")
        return True

    def unsubscribe(self, subscription_id: str) -> bool:
        """Beendet eine Event-Subscription."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"Subscription {subscription_id} beendet")
            return True
        return False

    def replay(self) -> list[AgentEvent]:
        """Gibt alle Events in chronologischer Reihenfolge zurück."""
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[AgentEvent]:
        """Filtert Events nach Typ."""
        with self._lock:
            return [event for event in self._events if event.event_type == event_type]

    def get_events_by_agent(self, agent_id: str) -> list[AgentEvent]:
        """Filtert Events nach Agent-ID."""
        with self._lock:
            return [event for event in self._events if event.agent_id == agent_id]

    def clear_history(self) -> int:
        """Löscht Event-Historie und gibt Anzahl gelöschter Events zurück."""
        with self._lock:
            count = len(self._events)
            self._events.clear()
            self._idempotency_manager.clear()
            logger.info(f"{count} Events aus Historie gelöscht")
            return count

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Event-Bus-Statistiken zurück."""
        with self._lock:
            active_subscriptions = sum(1 for s in self._subscriptions.values() if s.active)
            idempotency_stats = self._idempotency_manager.get_statistics()

            return {
                "total_events": len(self._events),
                "published_count": self._published_counter["value"],
                "duplicate_count": idempotency_stats["duplicates_found"],
                "duplicate_rate": idempotency_stats["duplicate_rate"],
                "total_subscriptions": len(self._subscriptions),
                "active_subscriptions": active_subscriptions,
                "callback_errors": self._callback_error_counter["value"],
                "max_history_size": self._events.maxlen,
                "idempotency_cache_size": idempotency_stats["cache_size"],
            }


__all__ = ["AgentEvent", "AgentEventBus", "EventSubscription"]
