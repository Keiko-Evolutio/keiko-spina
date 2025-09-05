"""Webhook-Retry-Scheduler für Backoff-Logic und DLQ-Management.

Verwaltet Retry-Strategien mit exponential backoff und Dead Letter Queue
für fehlgeschlagene Webhook-Deliveries.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from monitoring import record_custom_metric

from ..audit_logger import WebhookAuditEventType, webhook_audit
from ..constants import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    MAX_BACKOFF_SECONDS,
)
from ..keys import dlq_key, outbox_key
from ..models import DeliveryRecord, DeliveryStatus, WebhookEvent, WebhookTarget
from ..prometheus_metrics import WEBHOOK_DLQ_TOTAL

if TYPE_CHECKING:
    from ..utils.redis_manager import RedisManager

logger = get_logger(__name__)


class WebhookRetryScheduler:
    """Verwaltet Retry-Logic und DLQ-Management für Webhook-Deliveries."""

    def __init__(self, redis_manager: RedisManager) -> None:
        """Initialisiert den Retry-Scheduler.

        Args:
            redis_manager: Redis-Manager für Queue-Operationen
        """
        self.redis_manager = redis_manager

    async def schedule_retry(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
        event: WebhookEvent,
        queue_name: str,
    ) -> bool:
        """Plant Retry oder verschiebt in DLQ.

        Args:
            record: Delivery-Record
            target: Webhook-Target
            event: Webhook-Event
            queue_name: Queue-Name für Retry

        Returns:
            True wenn Retry geplant, False wenn in DLQ verschoben
        """
        # Prüfe ob weitere Versuche möglich sind
        max_attempts = target.max_attempts or DEFAULT_MAX_ATTEMPTS
        if record.attempt >= max_attempts:
            await self._move_to_dlq(record, target, event)
            return False

        # Berechne Backoff-Delay
        delay_seconds = self._calculate_backoff_delay(record, target)

        # Warte Backoff-Zeit
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        # Plane Retry
        await self._schedule_retry_attempt(record, target, event, queue_name, delay_seconds)
        return True

    async def _move_to_dlq(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
        event: WebhookEvent,
    ) -> None:
        """Verschiebt fehlgeschlagene Delivery in Dead Letter Queue.

        Args:
            record: Delivery-Record
            target: Webhook-Target
            event: Webhook-Event
        """
        # Status auf DLQ setzen
        record.status = DeliveryStatus.dlq

        # DLQ-Payload erstellen
        dlq_payload = json.dumps({
            "record": record.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
        })

        # In DLQ einreihen
        tenant_id = event.meta.tenant if event.meta else None
        dlq_queue_key = dlq_key(tenant_id)

        success = await self.redis_manager.safe_lpush(dlq_queue_key, dlq_payload)
        if not success:
            logger.error(
                "Failed to move delivery %s to DLQ for target %s",
                record.delivery_id, target.id
            )
            return

        # Metriken aufzeichnen
        await self._record_dlq_metrics(target, event, record)

        # Audit-Logging
        await self._record_dlq_audit(record, target, event)

        logger.info(
            "Moved delivery %s to DLQ after %d attempts (target: %s)",
            record.delivery_id, record.attempt, target.id
        )

    async def _schedule_retry_attempt(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
        event: WebhookEvent,
        queue_name: str,
        delay_seconds: float,
    ) -> None:
        """Plant einen neuen Retry-Versuch.

        Args:
            record: Delivery-Record
            target: Webhook-Target
            event: Webhook-Event
            queue_name: Queue-Name
            delay_seconds: Verwendete Delay-Zeit
        """
        # Retry-Payload erstellen
        retry_payload = json.dumps({
            "record": record.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
        })

        # In Retry-Queue einreihen
        tenant_id = event.meta.tenant if event.meta else None
        retry_queue_key = outbox_key(tenant_id, queue_name)

        success = await self.redis_manager.safe_lpush(retry_queue_key, retry_payload)
        if not success:
            logger.error(
                "Failed to schedule retry for delivery %s (target: %s)",
                record.delivery_id, target.id
            )
            return

        # Audit-Logging für Retry
        await self._record_retry_audit(record, target, event, delay_seconds)

        logger.debug(
            "Scheduled retry for delivery %s (attempt %d/%d, delay: %.1fs, target: %s)",
            record.delivery_id,
            record.attempt + 1,
            target.max_attempts or DEFAULT_MAX_ATTEMPTS,
            delay_seconds,
            target.id
        )

    def _calculate_backoff_delay(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
    ) -> float:
        """Berechnet Backoff-Delay für Retry.

        Args:
            record: Delivery-Record
            target: Webhook-Target

        Returns:
            Delay in Sekunden
        """
        base_backoff = target.backoff_seconds or DEFAULT_BACKOFF_SECONDS

        # Exponential backoff: base * 2^(attempt-1)
        exponential_delay = base_backoff * (2 ** max(0, record.attempt - 1))

        # Begrenze auf Maximum
        return min(exponential_delay, MAX_BACKOFF_SECONDS)

    async def _record_dlq_metrics(
        self,
        target: WebhookTarget,
        event: WebhookEvent,
        record: DeliveryRecord,
    ) -> None:
        """Zeichnet DLQ-Metriken auf.

        Args:
            target: Webhook-Target
            event: Webhook-Event
            record: Delivery-Record
        """
        try:
            # Custom Metrics
            record_custom_metric(
                "webhook.dlq",
                1,
                {
                    "target": target.id,
                    "event_type": event.event_type,
                    "attempts": record.attempt,
                }
            )

            # Prometheus Metrics
            tenant_id = event.meta.tenant if event.meta else ""
            WEBHOOK_DLQ_TOTAL.labels(
                target_id=target.id,
                event_type=event.event_type,
                tenant_id=tenant_id,
            ).inc()

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to record DLQ metrics: %s", exc)

    async def _record_dlq_audit(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
        event: WebhookEvent,
    ) -> None:
        """Zeichnet DLQ-Audit-Log auf.

        Args:
            record: Delivery-Record
            target: Webhook-Target
            event: Webhook-Event
        """
        try:
            tenant_id = event.meta.tenant if event.meta else None
            await webhook_audit.dlq_event(
                event=WebhookAuditEventType.DLQ_MOVE,
                delivery_id=record.delivery_id,
                target_id=target.id,
                tenant_id=tenant_id,
                correlation_id=record.correlation_id,
                details={
                    "attempt": record.attempt,
                    "max_attempts": target.max_attempts or DEFAULT_MAX_ATTEMPTS,
                    "last_error": record.last_error,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to record DLQ audit: %s", exc)

    async def _record_retry_audit(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
        event: WebhookEvent,
        delay_seconds: float,
    ) -> None:
        """Zeichnet Retry-Audit-Log auf.

        Args:
            record: Delivery-Record
            target: Webhook-Target
            event: Webhook-Event
            delay_seconds: Verwendete Delay-Zeit
        """
        try:
            tenant_id = event.meta.tenant if event.meta else None
            await webhook_audit.outbound_retried(
                correlation_id=record.correlation_id,
                delivery_id=record.delivery_id,
                target_id=target.id,
                event_type=event.event_type,
                tenant_id=tenant_id,
                details={
                    "attempt": record.attempt,
                    "next_delay_s": delay_seconds,
                    "last_error": record.last_error,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to record retry audit: %s", exc)

    def get_retry_info(
        self,
        record: DeliveryRecord,
        target: WebhookTarget,
    ) -> dict[str, Any]:
        """Gibt Retry-Informationen zurück.

        Args:
            record: Delivery-Record
            target: Webhook-Target

        Returns:
            Dictionary mit Retry-Informationen
        """
        max_attempts = target.max_attempts or DEFAULT_MAX_ATTEMPTS
        next_delay = self._calculate_backoff_delay(record, target)

        return {
            "current_attempt": record.attempt,
            "max_attempts": max_attempts,
            "remaining_attempts": max(0, max_attempts - record.attempt),
            "next_delay_seconds": next_delay,
            "backoff_base_seconds": target.backoff_seconds or DEFAULT_BACKOFF_SECONDS,
            "will_retry": record.attempt < max_attempts,
            "will_dlq": record.attempt >= max_attempts,
        }


__all__ = ["WebhookRetryScheduler"]
