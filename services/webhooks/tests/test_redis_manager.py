"""Tests für den Redis Manager."""

from unittest.mock import patch

import pytest

from services.webhooks.tests.utils import MockRedisClient, async_test
from services.webhooks.utils.redis_manager import (
    RedisManager,
    get_redis_client,
    get_redis_manager,
)
from storage.cache.redis_cache import NoOpCache


class TestRedisManager:
    """Tests für die RedisManager-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.manager = RedisManager()

    @async_test
    async def test_get_client_returns_cached_client(self):
        """Testet, dass get_client gecachte Clients zurückgibt."""
        mock_client = MockRedisClient()

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            client1 = await self.manager.get_client()
            client2 = await self.manager.get_client()

            assert client1 is client2
            assert client1 is mock_client

    @async_test
    async def test_get_client_fallback_to_noop_on_error(self):
        """Testet Fallback auf NoOpCache bei Verbindungsfehlern."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  side_effect=ConnectionError("Redis nicht verfügbar")):
            client = await self.manager.get_client()

            assert isinstance(client, NoOpCache)

    @async_test
    async def test_invalidate_cache_clears_cached_client(self):
        """Testet, dass invalidate_cache den Cache leert."""
        mock_client = MockRedisClient()

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            await self.manager.get_client()
            self.manager.invalidate_cache()

            # Nächster Aufruf sollte neuen Client holen
            client = await self.manager.get_client()
            assert client is mock_client

    @async_test
    async def test_is_available_true_for_redis_client(self):
        """Testet is_available für echten Redis-Client."""
        mock_client = MockRedisClient()

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            available = await self.manager.is_available()
            assert available is True

    @async_test
    async def test_is_available_false_for_noop_cache(self):
        """Testet is_available für NoOpCache."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  side_effect=ConnectionError()):
            available = await self.manager.is_available()
            assert available is False

    @async_test
    async def test_execute_with_retry_success_on_first_attempt(self):
        """Testet erfolgreiche Operation beim ersten Versuch."""
        mock_client = MockRedisClient()

        async def test_operation(client):
            await client.set("test", "value")
            return "success"

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            result = await self.manager.execute_with_retry(test_operation)
            assert result == "success"

    @async_test
    async def test_execute_with_retry_fallback_on_noop(self):
        """Testet Fallback-Wert bei NoOpCache."""
        async def test_operation(client):
            return "should not be called"

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  side_effect=ConnectionError()):
            result = await self.manager.execute_with_retry(
                test_operation,
                fallback_value="fallback"
            )
            assert result == "fallback"

    @async_test
    async def test_execute_with_retry_retries_on_connection_error(self):
        """Testet Retry-Verhalten bei Verbindungsfehlern."""
        mock_client = MockRedisClient()
        call_count = 0

        async def failing_operation(client):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            result = await self.manager.execute_with_retry(
                failing_operation,
                max_attempts=3,
                backoff_seconds=0.01  # Schneller für Tests
            )
            assert result == "success"
            assert call_count == 3

    @async_test
    async def test_execute_with_retry_gives_up_after_max_attempts(self):
        """Testet, dass nach max_attempts aufgegeben wird."""
        mock_client = MockRedisClient()

        async def always_failing_operation(client):
            raise ConnectionError("Always fails")

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            result = await self.manager.execute_with_retry(
                always_failing_operation,
                max_attempts=2,
                backoff_seconds=0.01,
                fallback_value="fallback"
            )
            assert result == "fallback"


class TestRedisManagerSafeMethods:
    """Tests für die Safe-Methoden des RedisManagers."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.manager = RedisManager()
        self.mock_client = MockRedisClient()

    @async_test
    async def test_safe_set_success(self):
        """Testet erfolgreiche SET-Operation."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_set("test", "value")
            assert result is True
            assert await self.mock_client.get("test") == "value"

    @async_test
    async def test_safe_set_with_expiration(self):
        """Testet SET-Operation mit Expiration."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_set("test", "value", ex=300)
            assert result is True
            assert "test" in self.mock_client._expirations

    @async_test
    async def test_safe_set_nx_existing_key(self):
        """Testet SET NX mit existierendem Key."""
        await self.mock_client.set("test", "existing")

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_set("test", "new", nx=True)
            assert result is False
            assert await self.mock_client.get("test") == "existing"

    @async_test
    async def test_safe_get_success(self):
        """Testet erfolgreiche GET-Operation."""
        await self.mock_client.set("test", "value")

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_get("test")
            assert result == "value"

    @async_test
    async def test_safe_get_nonexistent_key(self):
        """Testet GET-Operation für nicht existierenden Key."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_get("nonexistent")
            assert result is None

    @async_test
    async def test_safe_delete_success(self):
        """Testet erfolgreiche DELETE-Operation."""
        await self.mock_client.set("test1", "value1")
        await self.mock_client.set("test2", "value2")

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_delete("test1", "test2")
            assert result == 2
            assert await self.mock_client.get("test1") is None
            assert await self.mock_client.get("test2") is None

    @async_test
    async def test_safe_lpush_success(self):
        """Testet erfolgreiche LPUSH-Operation."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_lpush("list", "value1", "value2")
            assert result == 2
            assert await self.mock_client.llen("list") == 2

    @async_test
    async def test_safe_rpop_success(self):
        """Testet erfolgreiche RPOP-Operation."""
        await self.mock_client.lpush("list", "value1", "value2")

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_rpop("list")
            assert result == "value1"

    @async_test
    async def test_safe_hset_success(self):
        """Testet erfolgreiche HSET-Operation."""
        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_hset("hash", "field", "value")
            assert result is True
            assert await self.mock_client.hget("hash", "field") == "value"

    @async_test
    async def test_safe_hgetall_success(self):
        """Testet erfolgreiche HGETALL-Operation."""
        await self.mock_client.hset("hash", "field1", "value1")
        await self.mock_client.hset("hash", "field2", "value2")

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=self.mock_client):
            result = await self.manager.safe_hgetall("hash")
            assert result == {"field1": "value1", "field2": "value2"}


class TestGlobalFunctions:
    """Tests für globale Funktionen."""

    @async_test
    async def test_get_redis_manager_singleton(self):
        """Testet, dass get_redis_manager ein Singleton zurückgibt."""
        manager1 = get_redis_manager()
        manager2 = get_redis_manager()
        assert manager1 is manager2

    @async_test
    async def test_get_redis_client_convenience_function(self):
        """Testet die Convenience-Funktion get_redis_client."""
        mock_client = MockRedisClient()

        with patch("services.webhooks.utils.redis_manager.get_cache_client",
                  return_value=mock_client):
            client = await get_redis_client()
            assert client is mock_client


if __name__ == "__main__":
    pytest.main([__file__])
