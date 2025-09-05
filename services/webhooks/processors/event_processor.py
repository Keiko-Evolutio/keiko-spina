"""Event-Processor für Outbound-Webhook-Events.

Extrahiert die Event-Processing-Logik aus dem WebhookManager für bessere
Separation of Concerns und Testbarkeit.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from monitoring import record_custom_metric

from ..audit_logger import webhook_audit
from ..exceptions import WebhookExceptionFactory
from ..keys import outbox_key
from ..models import DeliveryRecord, DeliveryStatus, WebhookEvent, WebhookEventMeta
from ..prometheus_metrics import WEBHOOK_ENQUEUED_TOTAL, WEBHOOK_QUEUE_DEPTH

if TYPE_CHECKING:
    from ..targets import TargetRegistry
    from ..utils.redis_manager import RedisManager

logger = get_logger(__name__)


class WebhookEventProcessor:
    """Verarbeitet Outbound-Webhook-Events mit verbesserter Architektur."""

    def __init__(
        self,
        redis_manager: RedisManager,
        target_registry: TargetRegistry,
    ) -> None:
        """Initialisiert den Event-Processor.

        Args:
            redis_manager: Redis-Manager für Queue-Operationen
            target_registry: Target-Registry für Target-Auflösung
        """
        self.redis_manager = redis_manager
        self.target_registry = target_registry

    async def enqueue_event(
        self,
        *,
        target_id: str,
        event_type: str,
        data: dict[str, Any],
        meta: WebhookEventMeta | None = None,
        shard_names: list[str] | None = None,
    ) -> tuple[str, str]:
        """Plant Outbound-Webhook-Event in Queue ein.

        Args:
            target_id: ID des Webhook-Targets
            event_type: Typ des Events
            data: Event-Payload-Daten
            meta: Optionale Event-Metadaten
            shard_names: Liste der verfügbaren Queue-Shards

        Returns:
            Tupel aus (delivery_id, event_id)

        Raises:
            WebhookTargetException: Wenn Target nicht verfügbar
            WebhookDeliveryException: Wenn Queue nicht verfügbar
        """
        # Target validieren
        target = await self.target_registry.get(target_id)
        if not target or not target.enabled:
            raise WebhookExceptionFactory.target_not_found(target_id)

        # Event und Delivery-Record erstellen
        event_id = uuid.uuid4().hex
        event_meta = meta or WebhookEventMeta()
        event = WebhookEvent(
            id=event_id,
            event_type=event_type,
            data=data,
            meta=event_meta
        )

        delivery_id = uuid.uuid4().hex
        tenant_id = event_meta.tenant if hasattr(event_meta, "tenant") else None
        record = DeliveryRecord(
            delivery_id=delivery_id,
            target_id=target_id,
            event_id=event_id,
            tenant_id=tenant_id,
            status=DeliveryStatus.pending,
        )

        # Queue-Payload erstellen und einreihen
        payload = self._create_queue_payload(record, target, event)
        shard = self._select_shard(delivery_id, shard_names or ["default"])
        queue_key = outbox_key(tenant_id, shard)

        success = await self.redis_manager.safe_lpush(queue_key, payload)
        if not success:
            raise WebhookExceptionFactory.circuit_breaker_open("redis_queue")

        # Metriken und Audit-Logging
        await self._record_metrics_and_audit(
            target_id=target_id,
            event_type=event_type,
            tenant_id=tenant_id,
            delivery_id=delivery_id,
            event=event,
        )

        return delivery_id, event_id

    def _create_queue_payload(self, record, target, event) -> str:
        """Erstellt JSON-Payload für Queue.

        Args:
            record: Delivery-Record
            target: Webhook-Target
            event: Webhook-Event

        Returns:
            JSON-String für Queue
        """
        return json.dumps({
            "record": record.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
        })

    def _select_shard(self, delivery_id: str, shard_names: list[str]) -> str:
        """Wählt Shard basierend auf Delivery-ID (deterministische Verteilung).

        Args:
            delivery_id: Delivery-ID für Hash-Berechnung
            shard_names: Verfügbare Shard-Namen

        Returns:
            Gewählter Shard-Name
        """
        if len(shard_names) == 1:
            return shard_names[0]

        try:
            # Nutze die ersten 8 Hex-Zeichen als Hash-Quelle
            idx = int(delivery_id[:8], 16) % len(shard_names)
            return shard_names[idx]
        except (ValueError, IndexError):
            # Fallback auf ersten Shard
            return shard_names[0]

    async def _record_metrics_and_audit(
        self,
        *,
        target_id: str,
        event_type: str,
        tenant_id: str | None,
        delivery_id: str,
        event: WebhookEvent,
    ) -> None:
        """Zeichnet Metriken und Audit-Logs auf (best effort).

        Args:
            target_id: Target-ID
            event_type: Event-Typ
            tenant_id: Tenant-ID
            delivery_id: Delivery-ID
            event: Webhook-Event
        """
        # Custom Metrics
        try:
            record_custom_metric(
                "webhook.enqueued",
                1,
                {"target": target_id, "event_type": event_type}
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Custom metric recording failed: %s", exc)

        # Prometheus Metrics
        try:
            WEBHOOK_ENQUEUED_TOTAL.labels(
                target_id=target_id,
                event_type=event_type,
                tenant_id=tenant_id or ""
            ).inc()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Prometheus metric recording failed: %s", exc)

        # Audit Logging
        try:
            correlation_id = (
                event.meta.correlation_id
                if (event.meta and hasattr(event.meta, "correlation_id")
                    and event.meta.correlation_id)
                else event.id
            )
            await webhook_audit.outbound_enqueued(
                correlation_id=correlation_id,
                delivery_id=delivery_id,
                target_id=target_id,
                event_type=event_type,
                tenant_id=tenant_id,
                user_id=None,
                details={"queue": "default"},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Audit logging failed: %s", exc)

    async def get_queue_depth(self, shard_names: list[str]) -> int:
        """Berechnet die Gesamttiefe aller Outbox-Queues.

        Args:
            shard_names: Liste der Queue-Shards

        Returns:
            Gesamtanzahl der Events in allen Queues
        """
        total_depth = 0

        for shard in shard_names:
            try:
                queue_key = outbox_key(None, shard)
                depth = await self.redis_manager.safe_llen(queue_key)
                total_depth += depth

                # Prometheus Metric pro Shard
                self._update_queue_depth_metric(shard, depth)

            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Queue depth calculation failed for shard %s: %s", shard, exc)

        return total_depth

    def _update_queue_depth_metric(self, shard: str, depth: int) -> None:
        """Aktualisiert Queue-Depth-Metrik für einen Shard.

        Args:
            shard: Shard-Name
            depth: Queue-Tiefe
        """
        try:
            WEBHOOK_QUEUE_DEPTH.labels(queue_name=shard).set(depth)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Queue depth metric failed for shard %s: %s", shard, exc)


__all__ = ["WebhookEventProcessor"]
