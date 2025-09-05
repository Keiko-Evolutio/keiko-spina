"""Tests für den WebhookEventProcessor."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from services.webhooks.models import WebhookEventMeta
from services.webhooks.processors.event_processor import WebhookEventProcessor
from services.webhooks.tests.utils import MockRedisClient, async_test, create_webhook_target


class TestWebhookEventProcessor:
    """Tests für die WebhookEventProcessor-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.mock_redis = MockRedisClient()
        self.mock_target_registry = AsyncMock()

        # Mock Redis Manager
        self.mock_redis_manager = AsyncMock()
        self.mock_redis_manager.safe_lpush = AsyncMock(return_value=1)

        self.processor = WebhookEventProcessor(
            redis_manager=self.mock_redis_manager,
            target_registry=self.mock_target_registry,
        )

    @async_test
    async def test_enqueue_event_success(self):
        """Testet erfolgreiche Event-Enqueue-Operation."""
        # Setup
        target = create_webhook_target(target_id="test-target", enabled=True)
        self.mock_target_registry.get.return_value = target

        # Execute
        delivery_id, event_id = await self.processor.enqueue_event(
            target_id="test-target",
            event_type="test.event",
            data={"test": "data"},
        )

        # Verify
        assert delivery_id is not None
        assert event_id is not None
        assert len(delivery_id) == 32  # UUID hex
        assert len(event_id) == 32  # UUID hex

        # Verify Redis call
        self.mock_redis_manager.safe_lpush.assert_called_once()
        call_args = self.mock_redis_manager.safe_lpush.call_args
        assert "kei:webhook:outbox:default:default" in call_args[0][0]

        # Verify payload structure
        payload = json.loads(call_args[0][1])
        assert "record" in payload
        assert "target" in payload
        assert "event" in payload

    @async_test
    async def test_enqueue_event_target_not_found(self):
        """Testet Exception bei nicht gefundenem Target."""
        # Setup
        self.mock_target_registry.get.return_value = None

        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            await self.processor.enqueue_event(
                target_id="nonexistent-target",
                event_type="test.event",
                data={"test": "data"},
            )

        assert "not found" in str(exc_info.value).lower()

    @async_test
    async def test_enqueue_event_target_disabled(self):
        """Testet Exception bei deaktiviertem Target."""
        # Setup
        target = create_webhook_target(target_id="test-target", enabled=False)
        self.mock_target_registry.get.return_value = target

        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            await self.processor.enqueue_event(
                target_id="test-target",
                event_type="test.event",
                data={"test": "data"},
            )

        assert "not found" in str(exc_info.value).lower()

    @async_test
    async def test_enqueue_event_with_meta(self):
        """Testet Event-Enqueue mit Metadaten."""
        # Setup
        target = create_webhook_target(target_id="test-target", enabled=True)
        self.mock_target_registry.get.return_value = target

        meta = WebhookEventMeta(
            tenant="test-tenant",
            correlation_id="test-correlation",
            source="test-source",
        )

        # Execute
        delivery_id, event_id = await self.processor.enqueue_event(
            target_id="test-target",
            event_type="test.event",
            data={"test": "data"},
            meta=meta,
        )

        # Verify
        assert delivery_id is not None
        assert event_id is not None

        # Verify payload contains meta
        call_args = self.mock_redis_manager.safe_lpush.call_args
        payload = json.loads(call_args[0][1])
        assert payload["event"]["meta"]["tenant"] == "test-tenant"
        assert payload["event"]["meta"]["correlation_id"] == "test-correlation"

    @async_test
    async def test_enqueue_event_redis_failure(self):
        """Testet Exception bei Redis-Fehler."""
        # Setup
        target = create_webhook_target(target_id="test-target", enabled=True)
        self.mock_target_registry.get.return_value = target
        self.mock_redis_manager.safe_lpush.return_value = 0  # Failure

        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            await self.processor.enqueue_event(
                target_id="test-target",
                event_type="test.event",
                data={"test": "data"},
            )

        assert "circuit" in str(exc_info.value).lower()

    def test_select_shard_single_shard(self):
        """Testet Shard-Auswahl mit einem Shard."""
        result = self.processor._select_shard("test-delivery-id", ["default"])
        assert result == "default"

    def test_select_shard_multiple_shards(self):
        """Testet Shard-Auswahl mit mehreren Shards."""
        shards = ["shard_0", "shard_1", "shard_2"]

        # Teste deterministische Verteilung
        result1 = self.processor._select_shard("12345678", shards)
        result2 = self.processor._select_shard("12345678", shards)
        assert result1 == result2  # Deterministic

        # Teste verschiedene IDs
        result3 = self.processor._select_shard("87654321", shards)
        # Kann gleich oder unterschiedlich sein, aber sollte gültiger Shard sein
        assert result3 in shards

    def test_select_shard_invalid_delivery_id(self):
        """Testet Shard-Auswahl mit ungültiger Delivery-ID."""
        shards = ["shard_0", "shard_1"]
        result = self.processor._select_shard("invalid-hex", shards)
        assert result == "shard_0"  # Fallback auf ersten Shard

    @async_test
    async def test_get_queue_depth(self):
        """Testet Queue-Depth-Berechnung."""
        # Setup
        self.mock_redis_manager.safe_llen = AsyncMock(side_effect=[5, 3, 2])

        # Execute
        depth = await self.processor.get_queue_depth(["shard_0", "shard_1", "shard_2"])

        # Verify
        assert depth == 10  # 5 + 3 + 2
        assert self.mock_redis_manager.safe_llen.call_count == 3

    @async_test
    async def test_get_queue_depth_with_errors(self):
        """Testet Queue-Depth-Berechnung mit Fehlern."""
        # Setup - Ein Shard wirft Exception
        self.mock_redis_manager.safe_llen = AsyncMock(side_effect=[5, Exception("Redis error"), 2])

        # Execute
        depth = await self.processor.get_queue_depth(["shard_0", "shard_1", "shard_2"])

        # Verify - Fehler werden ignoriert
        assert depth == 7  # 5 + 0 + 2
        assert self.mock_redis_manager.safe_llen.call_count == 3

    @async_test
    @patch("services.webhooks.processors.event_processor.record_custom_metric")
    @patch("services.webhooks.processors.event_processor.webhook_audit")
    async def test_record_metrics_and_audit(self, mock_audit, mock_metric):
        """Testet Metriken- und Audit-Aufzeichnung."""
        # Setup
        from services.webhooks.models import WebhookEvent, WebhookEventMeta
        event = WebhookEvent(
            id="test-event-id",
            event_type="test.event",
            data={"test": "data"},
            meta=WebhookEventMeta(correlation_id="test-correlation")
        )

        # Execute
        await self.processor._record_metrics_and_audit(
            target_id="test-target",
            event_type="test.event",
            tenant_id="test-tenant",
            delivery_id="test-delivery",
            event=event,
        )

        # Verify custom metric
        mock_metric.assert_called_once_with(
            "webhook.enqueued",
            1,
            {"target": "test-target", "event_type": "test.event"}
        )

        # Verify audit logging
        mock_audit.outbound_enqueued.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
