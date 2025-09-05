"""BusService - Fassade für KEI-Bus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kei_logging import get_logger

from .config import BusSettings, bus_settings
from .kafka_provider import KafkaProvider
from .nats_provider import NATSProvider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .envelope import BusEnvelope

logger = get_logger(__name__)


class BusService:
    """Hochohne Fassade für Publish/Subscribe und RPC über Bus."""

    def __init__(self, settings: BusSettings | None = None) -> None:
        self.settings = settings or bus_settings
        self._nats = NATSProvider()
        self._kafka = KafkaProvider()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialisiert Verbindungen und Streams."""
        if self._initialized:
            return
        if self.settings.provider == "nats":
            await self._nats.connect()
        elif self.settings.provider == "kafka":
            await self._kafka.connect()
        self._initialized = True

    async def publish(self, envelope: BusEnvelope) -> None:
        """Veröffentlicht Nachricht."""
        if self.settings.provider == "nats":
            await self._nats.publish(envelope)
        elif self.settings.provider == "kafka":
            await self._kafka.publish(envelope)

    async def subscribe(
        self,
        subject: str,
        queue: str | None,
        handler: Callable[[BusEnvelope], Awaitable[None]],
        *,
        durable: str | None = None,
    ) -> None:
        """Abonniert auf Subject."""
        if self.settings.provider == "nats":
            await self._nats.subscribe(subject, queue, handler, durable=durable)
        elif self.settings.provider == "kafka":
            # Für Kafka verwenden wir group_suffix aus queue/durable
            group_suffix = queue or durable or "default"
            await self._kafka.subscribe(subject, group_suffix, handler)


_global_service: BusService | None = None


def get_bus_service() -> BusService:
    """Gibt globale BusService-Instanz zurück."""
    global _global_service
    if _global_service is None:
        _global_service = BusService()
    return _global_service
