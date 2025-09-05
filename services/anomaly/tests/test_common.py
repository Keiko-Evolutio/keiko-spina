"""Unit-Tests für anomaly/common.py.

Testet alle Utility-Funktionen, Konfigurationsklassen und Redis-Helper.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.anomaly.common import (
    BASELINE_CACHE_PREFIX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_TTL,
    DEFAULT_CONTAMINATION_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_LATENT_DIM,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAINING_INTERVAL_SECONDS,
    DEFAULT_WINDOW_SIZE,
    MIN_SAMPLES_THRESHOLD,
    TRAIN_QUEUE_KEY,
    AnomalyRedisHelper,
    BaseAnomalyConfig,
    DetectionConfig,
    LSTMConfig,
    TrainingConfig,
    calculate_statistics,
    deserialize_training_job,
    format_cache_key,
    redis_helper,
    safe_ml_operation,
    safe_redis_operation,
    serialize_training_job,
)


class TestKonfigurationsklassen:
    """Tests für Konfigurationsklassen."""

    def test_base_anomaly_config_defaults(self):
        """Testet Standard-Werte der BaseAnomalyConfig."""
        config = BaseAnomalyConfig()
        assert config.random_state == DEFAULT_RANDOM_STATE
        assert config.min_samples == MIN_SAMPLES_THRESHOLD

    def test_detection_config_defaults(self):
        """Testet Standard-Werte der DetectionConfig."""
        config = DetectionConfig()
        assert config.contamination == DEFAULT_CONTAMINATION_RATE
        assert config.random_state == DEFAULT_RANDOM_STATE
        assert config.min_samples == MIN_SAMPLES_THRESHOLD

    def test_detection_config_custom_values(self):
        """Testet benutzerdefinierte Werte der DetectionConfig."""
        config = DetectionConfig(
            contamination=0.05,
            random_state=123,
            min_samples=100
        )
        assert config.contamination == 0.05
        assert config.random_state == 123
        assert config.min_samples == 100

    def test_lstm_config_defaults(self):
        """Testet Standard-Werte der LSTMConfig."""
        config = LSTMConfig()
        assert config.window_size == DEFAULT_WINDOW_SIZE
        assert config.latent_dim == DEFAULT_LATENT_DIM
        assert config.epochs == DEFAULT_EPOCHS
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.random_state == DEFAULT_RANDOM_STATE
        assert config.min_samples == MIN_SAMPLES_THRESHOLD

    def test_training_config_defaults(self):
        """Testet Standard-Werte der TrainingConfig."""
        config = TrainingConfig()
        assert config.interval_seconds == DEFAULT_TRAINING_INTERVAL_SECONDS
        assert config.queue_key == TRAIN_QUEUE_KEY
        assert config.random_state == DEFAULT_RANDOM_STATE
        assert config.min_samples == MIN_SAMPLES_THRESHOLD


class TestUtilityFunktionen:
    """Tests für Utility-Funktionen."""

    def test_calculate_statistics_empty_list(self):
        """Testet calculate_statistics mit leerer Liste."""
        result = calculate_statistics([])
        expected = {"mean": 0.0, "variance": 0.0, "stddev": 0.0}
        assert result == expected

    def test_calculate_statistics_single_value(self):
        """Testet calculate_statistics mit einem Wert."""
        result = calculate_statistics([5.0])
        assert result["mean"] == 5.0
        assert result["variance"] == 0.0
        assert result["stddev"] == 0.0

    def test_calculate_statistics_multiple_values(self):
        """Testet calculate_statistics mit mehreren Werten."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_statistics(values)

        assert result["mean"] == 3.0
        assert abs(result["variance"] - 2.5) < 1e-10  # Floating-point Vergleich
        assert abs(result["stddev"] - 1.5811388300841898) < 1e-10

    def test_format_cache_key_without_suffix(self):
        """Testet format_cache_key ohne Suffix."""
        key = format_cache_key("tenant1", "cpu_usage")
        expected = f"{BASELINE_CACHE_PREFIX}:tenant1:cpu_usage"
        assert key == expected

    def test_format_cache_key_with_suffix(self):
        """Testet format_cache_key mit Suffix."""
        key = format_cache_key("tenant1", "cpu_usage", "baseline")
        expected = f"{BASELINE_CACHE_PREFIX}:tenant1:cpu_usage:baseline"
        assert key == expected

    def test_serialize_training_job(self):
        """Testet serialize_training_job."""
        result = serialize_training_job("tenant1", "cpu", [1.0, 2.0, 3.0])
        expected = json.dumps({
            "tenant": "tenant1",
            "metric": "cpu",
            "values": [1.0, 2.0, 3.0]
        })
        assert result == expected

    def test_deserialize_training_job_valid(self):
        """Testet deserialize_training_job mit gültigen Daten."""
        job_data = json.dumps({
            "tenant": "tenant1",
            "metric": "cpu",
            "values": [1.0, 2.0, 3.0]
        })
        result = deserialize_training_job(job_data)
        expected = {
            "tenant": "tenant1",
            "metric": "cpu",
            "values": [1.0, 2.0, 3.0]
        }
        assert result == expected

    def test_deserialize_training_job_invalid(self):
        """Testet deserialize_training_job mit ungültigen Daten."""
        result = deserialize_training_job("invalid json")
        expected = {"tenant": "default", "metric": "unknown", "values": []}
        assert result == expected

    def test_deserialize_training_job_none(self):
        """Testet deserialize_training_job mit None."""
        result = deserialize_training_job(None)
        expected = {"tenant": "default", "metric": "unknown", "values": []}
        assert result == expected


class TestSafeOperations:
    """Tests für Safe-Operation-Funktionen."""

    @pytest.mark.asyncio
    async def test_safe_redis_operation_success(self):
        """Testet safe_redis_operation bei erfolgreichem Aufruf."""
        mock_func = AsyncMock(return_value="success")
        result = await safe_redis_operation(
            "test_operation",
            mock_func,
            "arg1", "arg2",
            kwarg1="value1"
        )
        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_safe_redis_operation_exception(self):
        """Testet safe_redis_operation bei Exception."""
        mock_func = AsyncMock(side_effect=Exception("Redis error"))
        result = await safe_redis_operation(
            "test_operation",
            mock_func,
            default_return="fallback"
        )
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_safe_redis_operation_no_logging(self):
        """Testet safe_redis_operation ohne Logging."""
        mock_func = AsyncMock(side_effect=Exception("Redis error"))
        with patch("services.anomaly.common.logger") as mock_logger:
            result = await safe_redis_operation(
                "test_operation",
                mock_func,
                log_errors=False,
                default_return="fallback"
            )
            assert result == "fallback"
            mock_logger.warning.assert_not_called()

    def test_safe_ml_operation_success(self):
        """Testet safe_ml_operation bei erfolgreichem Aufruf."""
        mock_func = MagicMock(return_value="ml_result")
        result = safe_ml_operation(
            "test_ml_operation",
            mock_func,
            "arg1", "arg2",
            kwarg1="value1"
        )
        assert result == "ml_result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    def test_safe_ml_operation_exception(self):
        """Testet safe_ml_operation bei Exception."""
        mock_func = MagicMock(side_effect=Exception("ML error"))
        result = safe_ml_operation(
            "test_ml_operation",
            mock_func,
            default_return="ml_fallback"
        )
        assert result == "ml_fallback"


class TestAnomalyRedisHelper:
    """Tests für AnomalyRedisHelper-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.helper = AnomalyRedisHelper()

    @pytest.mark.asyncio
    async def test_store_baseline_success(self):
        """Testet store_baseline bei erfolgreichem Speichern."""
        mock_client = AsyncMock()
        mock_client.setex = AsyncMock(return_value=True)

        with patch("services.anomaly.common.get_cache_client", return_value=mock_client):
            stats = {"mean": 5.0, "variance": 2.0, "stddev": 1.41}
            result = await self.helper.store_baseline("tenant1", "cpu", stats)

            assert result is True
            expected_key = format_cache_key("tenant1", "cpu", "baseline")
            mock_client.setex.assert_called_once_with(
                expected_key,
                DEFAULT_CACHE_TTL,
                json.dumps(stats)
            )

    @pytest.mark.asyncio
    async def test_store_baseline_no_client(self):
        """Testet store_baseline ohne Redis-Client."""
        with patch("services.anomaly.common.get_cache_client", return_value=None):
            stats = {"mean": 5.0, "variance": 2.0, "stddev": 1.41}
            result = await self.helper.store_baseline("tenant1", "cpu", stats)
            assert result is False

    @pytest.mark.asyncio
    async def test_get_baseline_success(self):
        """Testet get_baseline bei erfolgreichem Laden."""
        mock_client = AsyncMock()
        stats = {"mean": 5.0, "variance": 2.0, "stddev": 1.41}
        mock_client.get = AsyncMock(return_value=json.dumps(stats))

        with patch("services.anomaly.common.get_cache_client", return_value=mock_client):
            result = await self.helper.get_baseline("tenant1", "cpu")

            assert result == stats
            expected_key = format_cache_key("tenant1", "cpu", "baseline")
            mock_client.get.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_get_baseline_not_found(self):
        """Testet get_baseline wenn keine Daten gefunden werden."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        with patch("services.anomaly.common.get_cache_client", return_value=mock_client):
            result = await self.helper.get_baseline("tenant1", "cpu")
            assert result is None

    @pytest.mark.asyncio
    async def test_enqueue_training_job_success(self):
        """Testet enqueue_training_job bei erfolgreichem Hinzufügen."""
        mock_client = AsyncMock()
        mock_client.lpush = AsyncMock(return_value=1)

        with patch("services.anomaly.common.get_cache_client", return_value=mock_client):
            result = await self.helper.enqueue_training_job("tenant1", "cpu", [1.0, 2.0])

            assert result is True
            expected_job = serialize_training_job("tenant1", "cpu", [1.0, 2.0])
            mock_client.lpush.assert_called_once_with(TRAIN_QUEUE_KEY, expected_job)

    @pytest.mark.asyncio
    async def test_dequeue_training_job_success(self):
        """Testet dequeue_training_job bei erfolgreichem Abrufen."""
        mock_client = AsyncMock()
        job_data = serialize_training_job("tenant1", "cpu", [1.0, 2.0])
        mock_client.rpop = AsyncMock(return_value=job_data)

        with patch("services.anomaly.common.get_cache_client", return_value=mock_client):
            result = await self.helper.dequeue_training_job()

            expected = {"tenant": "tenant1", "metric": "cpu", "values": [1.0, 2.0]}
            assert result == expected
            mock_client.rpop.assert_called_once_with(TRAIN_QUEUE_KEY)

    @pytest.mark.asyncio
    async def test_dequeue_training_job_empty_queue(self):
        """Testet dequeue_training_job bei leerer Queue."""
        mock_client = AsyncMock()
        mock_client.rpop = AsyncMock(return_value=None)

        with patch("services.anomaly.common.get_cache_client", return_value=mock_client):
            result = await self.helper.dequeue_training_job()
            assert result is None


class TestGlobaleInstanz:
    """Tests für globale redis_helper-Instanz."""

    def test_redis_helper_instance(self):
        """Testet dass redis_helper eine AnomalyRedisHelper-Instanz ist."""
        assert isinstance(redis_helper, AnomalyRedisHelper)
