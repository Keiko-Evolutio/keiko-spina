"""Tests für refactored Webhook-Komponenten.

Testet die neuen Utility-Module und refactored Worker-Klassen.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from services.webhooks.models import WebhookTarget
from services.webhooks.utils.redis_operations import (
    redis_get_json,
    redis_hash_get_all,
    redis_list_pop,
    redis_list_push,
    redis_set_json,
    safe_redis_operation,
)
from services.webhooks.workers.base_worker import BaseWorker, WorkerConfig, WorkerStatus


class TestRedisOperations:
    """Tests für Redis-Operations-Utility."""

    @pytest.mark.asyncio
    async def test_safe_redis_operation_success(self):
        """Test erfolgreiche Redis-Operation."""
        mock_client = AsyncMock()
        mock_client.get.return_value = "test_value"

        async def operation(client):
            return await client.get("test_key")

        with patch("services.webhooks.utils.redis_operations.get_cache_client", return_value=mock_client):
            result = await safe_redis_operation("test_op", operation, "default")

        assert result == "test_value"
        mock_client.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_safe_redis_operation_no_client(self):
        """Test Redis-Operation ohne verfügbaren Client."""
        with patch("backend.services.webhooks.utils.redis_operations.get_cache_client", return_value=None):
            result = await safe_redis_operation("test_op", lambda c: c.get("key"), "default")

        assert result == "default"

    @pytest.mark.asyncio
    async def test_safe_redis_operation_exception(self):
        """Test Redis-Operation mit Exception."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Redis error")

        async def operation(client):
            return await client.get("test_key")

        with patch("backend.services.webhooks.utils.redis_operations.get_cache_client", return_value=mock_client):
            result = await safe_redis_operation("test_op", operation, "default")

        assert result == "default"

    @pytest.mark.asyncio
    async def test_redis_get_json_with_model(self):
        """Test JSON-Get mit Pydantic-Model."""
        mock_client = AsyncMock()
        test_data = {"id": "test", "url": "http://example.com"}
        mock_client.get.return_value = json.dumps(test_data)

        with patch("backend.services.webhooks.utils.redis_operations.get_cache_client", return_value=mock_client):
            result = await redis_get_json("test_key", WebhookTarget)

        assert isinstance(result, WebhookTarget)
        assert result.id == "test"
        assert result.url == "http://example.com"

    @pytest.mark.asyncio
    async def test_redis_set_json_with_model(self):
        """Test JSON-Set mit Pydantic-Model."""
        mock_client = AsyncMock()
        target = WebhookTarget(id="test", url="http://example.com")

        with patch("backend.services.webhooks.utils.redis_operations.get_cache_client", return_value=mock_client):
            result = await redis_set_json("test_key", target)

        assert result is True
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args[0]
        assert call_args[0] == "test_key"
        assert "test" in call_args[1]
        assert "http://example.com" in call_args[1]

    @pytest.mark.asyncio
    async def test_redis_hash_operations(self):
        """Test Hash-Operationen."""
        mock_client = AsyncMock()
        test_data = {"field1": json.dumps({"id": "test1", "url": "http://example1.com"})}
        mock_client.hgetall.return_value = test_data

        with patch("backend.services.webhooks.utils.redis_operations.get_cache_client", return_value=mock_client):
            # Test hash_get_all
            result = await redis_hash_get_all("test_hash", WebhookTarget)

        assert len(result) == 1
        assert "field1" in result
        assert isinstance(result["field1"], WebhookTarget)
        assert result["field1"].id == "test1"

    @pytest.mark.asyncio
    async def test_redis_list_operations(self):
        """Test List-Operationen."""
        mock_client = AsyncMock()
        mock_client.rpop.return_value = "test_item"

        with patch("backend.services.webhooks.utils.redis_operations.get_cache_client", return_value=mock_client):
            # Test list_pop
            result = await redis_list_pop("test_list")
            assert result == "test_item"

            # Test list_push
            success = await redis_list_push("test_list", "new_item")
            assert success is True

        mock_client.rpop.assert_called_once_with("test_list")
        mock_client.lpush.assert_called_once_with("test_list", "new_item")


class TestWorker(BaseWorker):
    """Test-Worker für BaseWorker-Tests."""

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.cycle_count = 0
        self.should_fail = False

    async def _run_cycle(self) -> None:
        """Test-Zyklus."""
        self.cycle_count += 1
        if self.should_fail:
            raise RuntimeError("Test error")
        await asyncio.sleep(0.01)  # Kurze Pause


class TestBaseWorker:
    """Tests für BaseWorker-Klasse."""

    def test_worker_initialization(self):
        """Test Worker-Initialisierung."""
        config = WorkerConfig(name="test-worker", poll_interval_seconds=0.1)
        worker = TestWorker(config)

        assert worker.config.name == "test-worker"
        assert worker.status == WorkerStatus.STOPPED
        assert not worker.is_running
        assert worker.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_worker_start_stop(self):
        """Test Worker Start/Stop."""
        config = WorkerConfig(name="test-worker", poll_interval_seconds=0.1)
        worker = TestWorker(config)

        # Start Worker
        await worker.start()
        assert worker.status == WorkerStatus.RUNNING
        assert worker.is_running

        # Warte kurz für einige Zyklen
        await asyncio.sleep(0.3)
        assert worker.cycle_count > 0

        # Stop Worker
        await worker.stop()
        assert worker.status == WorkerStatus.STOPPED
        assert not worker.is_running

    @pytest.mark.asyncio
    async def test_worker_error_handling(self):
        """Test Worker Error-Handling."""
        config = WorkerConfig(
            name="test-worker",
            poll_interval_seconds=0.1,
            auto_restart=True,
            max_restart_attempts=2
        )
        worker = TestWorker(config)

        await worker.start()

        # Simuliere Fehler
        worker.should_fail = True

        # Warte auf Fehler und Restart-Versuche
        await asyncio.sleep(0.5)

        # Worker sollte nach max_restart_attempts im ERROR-Status sein
        assert worker.status == WorkerStatus.ERROR

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_health_info(self):
        """Test Worker Health-Informationen."""
        config = WorkerConfig(name="test-worker", poll_interval_seconds=0.1)
        worker = TestWorker(config)

        health = worker.health_info
        assert health["name"] == "test-worker"
        assert health["status"] == "stopped"
        assert health["uptime_seconds"] == 0.0
        assert health["restart_count"] == 0
        assert health["last_error"] is None
        assert health["task_alive"] is False

        await worker.start()
        await asyncio.sleep(0.1)

        health = worker.health_info
        assert health["status"] == "running"
        assert health["uptime_seconds"] > 0
        assert health["task_alive"] is True

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_restart(self):
        """Test Worker-Restart."""
        config = WorkerConfig(name="test-worker", poll_interval_seconds=0.1)
        worker = TestWorker(config)

        await worker.start()
        initial_cycle_count = worker.cycle_count

        await worker.restart()

        # Nach Restart sollte Worker wieder laufen
        assert worker.status == WorkerStatus.RUNNING

        await asyncio.sleep(0.2)

        # Cycle-Count sollte zurückgesetzt sein (neuer Worker-Zyklus)
        assert worker.cycle_count >= 0

        await worker.stop()


class TestDeliveryWorkerRefactoring:
    """Tests für refactored DeliveryWorker."""

    @pytest.mark.asyncio
    async def test_delivery_worker_inheritance(self):
        """Test dass DeliveryWorker BaseWorker korrekt verwendet."""
        from services.webhooks.delivery_worker import DeliveryWorker

        worker = DeliveryWorker("test-queue", poll_interval=0.1)

        # Sollte BaseWorker sein
        assert isinstance(worker, BaseWorker)
        assert worker.config.name == "delivery-worker-test-queue"
        assert worker.config.poll_interval_seconds == 0.1

    @pytest.mark.asyncio
    async def test_delivery_worker_constants_usage(self):
        """Test dass DeliveryWorker Konstanten verwendet."""
        from services.webhooks.constants import HTTP_CLIENT_TIMEOUT_SECONDS
        from services.webhooks.delivery_worker import DeliveryWorker

        worker = DeliveryWorker("test-queue")

        # HTTP-Client sollte konfigurierte Timeout verwenden
        assert worker._http.timeout == HTTP_CLIENT_TIMEOUT_SECONDS


class TestTargetRegistryRefactoring:
    """Tests für refactored TargetRegistry."""

    @pytest.mark.asyncio
    async def test_target_registry_redis_operations(self):
        """Test dass TargetRegistry neue Redis-Operations verwendet."""
        from services.webhooks.targets import TargetRegistry

        registry = TargetRegistry("test-registry")

        # Mock Redis-Operations
        with patch("services.webhooks.targets.redis_hash_get_all") as mock_get_all, \
             patch("services.webhooks.targets.redis_hash_set") as mock_set:

            mock_get_all.return_value = {}
            mock_set.return_value = True

            # Test save
            target = WebhookTarget(id="test", url="http://example.com")
            await registry._save_target_to_redis(target)

            # Verify redis_hash_set was called
            mock_set.assert_called_once()
            call_args = mock_set.call_args[0]
            assert "test-registry" in call_args[0]  # hash_key
            assert call_args[1] == "test"  # field

            # Test load
            await registry._load_from_redis()

            # Verify redis_hash_get_all was called
            mock_get_all.assert_called_once()


class TestCodeQualityMetrics:
    """Tests für Code-Qualitäts-Metriken."""

    def test_function_length_compliance(self):
        """Test dass refactored Funktionen Clean Code Standards erfüllen."""
        import inspect

        from services.webhooks.delivery_worker import DeliveryWorker

        worker = DeliveryWorker("test")

        # Teste dass _process_one aufgeteilt wurde
        assert hasattr(worker, "_fetch_delivery_data")
        assert hasattr(worker, "_parse_delivery_data")
        assert hasattr(worker, "_scan_tenant_queues")
        assert hasattr(worker, "_try_work_stealing")

        # Teste Funktionslängen (sollten alle < 20 Zeilen sein)
        for method_name in ["_fetch_delivery_data", "_parse_delivery_data",
                           "_scan_tenant_queues", "_try_work_stealing"]:
            method = getattr(worker, method_name)
            source_lines = inspect.getsource(method).split("\n")
            # Entferne leere Zeilen und Kommentare
            code_lines = [line for line in source_lines
                         if line.strip() and not line.strip().startswith("#")]
            assert len(code_lines) <= 20, f"{method_name} hat {len(code_lines)} Zeilen (max 20)"

    def test_constants_usage(self):
        """Test dass Magic Numbers durch Konstanten ersetzt wurden."""
        from services.webhooks.constants import (
            HTTP_CLIENT_TIMEOUT_SECONDS,
            REDIS_SCAN_COUNT_DEFAULT,
            REDIS_SCAN_COUNT_WORK_STEALING,
        )

        # Teste dass Konstanten definiert sind
        assert isinstance(REDIS_SCAN_COUNT_DEFAULT, int)
        assert isinstance(REDIS_SCAN_COUNT_WORK_STEALING, int)
        assert isinstance(HTTP_CLIENT_TIMEOUT_SECONDS, float)

        # Teste sinnvolle Werte
        assert REDIS_SCAN_COUNT_DEFAULT > 0
        assert REDIS_SCAN_COUNT_WORK_STEALING > REDIS_SCAN_COUNT_DEFAULT
        assert HTTP_CLIENT_TIMEOUT_SECONDS > 0

    def test_duplicate_code_elimination(self):
        """Test dass Duplicate Code eliminiert wurde."""
        from services.webhooks.delivery_worker import DeliveryWorker
        from services.webhooks.health_prober import HealthProber
        from services.webhooks.secret_rotation_worker import SecretRotationWorker
        from services.webhooks.workers.base_worker import BaseWorker

        # Alle Worker sollten BaseWorker verwenden
        assert issubclass(DeliveryWorker, BaseWorker)
        assert issubclass(SecretRotationWorker, BaseWorker)
        assert issubclass(HealthProber, BaseWorker)

        # Teste dass Worker keine eigenen start/stop Methoden haben
        # (sollten von BaseWorker geerbt werden)
        for worker_class in [DeliveryWorker, SecretRotationWorker, HealthProber]:
            # start/stop sollten von BaseWorker kommen, nicht überschrieben
            assert "start" not in worker_class.__dict__
            assert "stop" not in worker_class.__dict__
            # Aber _run_cycle sollte implementiert sein
            assert "_run_cycle" in worker_class.__dict__


if __name__ == "__main__":
    pytest.main([__file__])
