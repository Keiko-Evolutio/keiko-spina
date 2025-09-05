"""Unit-Tests für anomaly/service.py.

Testet AnomalyDetectionService mit allen Methoden und Edge-Cases.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.anomaly.common import ANOMALY_THRESHOLD_SIGMA, DetectionConfig
from services.anomaly.service import AnomalyDetectionService


class TestAnomalyDetectionService:
    """Tests für AnomalyDetectionService-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.config = DetectionConfig(
            contamination=0.1,
            random_state=42,
            min_samples=10
        )
        self.service = AnomalyDetectionService(self.config)

    def test_init_with_config(self):
        """Testet Initialisierung mit Konfiguration."""
        assert self.service.config == self.config

    def test_init_without_config(self):
        """Testet Initialisierung ohne Konfiguration (Standard-Werte)."""
        service = AnomalyDetectionService()
        assert isinstance(service.config, DetectionConfig)
        assert service.config.contamination == 0.02  # Standard-Wert

    def test_validate_input_valid_data(self):
        """Testet _validate_input mit gültigen Daten."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        with patch("services.anomaly.service.SKLEARN_AVAILABLE", True):
            result = self.service._validate_input(values)
            assert result is True

    def test_validate_input_insufficient_data(self):
        """Testet _validate_input mit unzureichenden Daten."""
        values = [1.0, 2.0, 3.0]  # Weniger als min_samples
        with patch("services.anomaly.service.SKLEARN_AVAILABLE", True):
            result = self.service._validate_input(values)
            assert result is False

    def test_validate_input_empty_list(self):
        """Testet _validate_input mit leerer Liste."""
        values = []
        with patch("services.anomaly.service.SKLEARN_AVAILABLE", True):
            result = self.service._validate_input(values)
            assert result is False

    def test_validate_input_sklearn_unavailable(self):
        """Testet _validate_input wenn sklearn nicht verfügbar ist."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        with patch("services.anomaly.service.SKLEARN_AVAILABLE", False):
            result = self.service._validate_input(values)
            assert result is False

    @patch("services.anomaly.service.safe_ml_operation")
    def test_perform_detection_success(self, mock_safe_ml):
        """Testet _perform_detection bei erfolgreichem Aufruf."""
        # Mock safe_ml_operation gibt Tuple zurück
        mock_safe_ml.return_value = (-0.8, True)

        values = [1.0, 2.0, 3.0, 10.0]
        result = self.service._perform_detection(values)

        assert result is not None
        last_score, is_outlier = result
        assert last_score == -0.8
        assert is_outlier is True
        mock_safe_ml.assert_called_once()

    @patch("services.anomaly.service.safe_ml_operation")
    def test_perform_detection_exception(self, mock_safe_ml):
        """Testet _perform_detection bei Exception."""
        mock_safe_ml.return_value = None

        values = [1.0, 2.0, 3.0, 4.0]
        result = self.service._perform_detection(values)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_metrics_normal_case(self):
        """Testet _update_metrics für normalen Fall (kein Outlier)."""
        with patch("services.anomaly.service.ANOMALY_SCORE_GAUGE") as mock_gauge:
            mock_labels = MagicMock()
            mock_gauge.labels.return_value = mock_labels

            await self.service._update_metrics("tenant1", "cpu", 0.5, False)

            mock_gauge.labels.assert_called_once_with(tenant="tenant1", metric="cpu")
            mock_labels.set.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_update_metrics_outlier_case(self):
        """Testet _update_metrics für Outlier-Fall."""
        with patch("services.anomaly.service.ANOMALY_SCORE_GAUGE") as mock_score_gauge, \
             patch("services.anomaly.service.ANOMALY_DETECTIONS_TOTAL") as mock_detection_total, \
             patch("services.anomaly.service.emit_warning") as mock_emit_warning:

            mock_score_labels = MagicMock()
            mock_score_gauge.labels.return_value = mock_score_labels

            mock_detection_labels = MagicMock()
            mock_detection_total.labels.return_value = mock_detection_labels

            await self.service._update_metrics("tenant1", "cpu", -0.8, True)

            # Score-Gauge prüfen
            mock_score_gauge.labels.assert_called_with(tenant="tenant1", metric="cpu")
            mock_score_labels.set.assert_called_with(-0.8)

            # Detection-Counter prüfen
            mock_detection_total.labels.assert_called_with(
                tenant="tenant1",
                metric="cpu",
                severity="warning"
            )
            mock_detection_labels.inc.assert_called_once()

            # Webhook-Alert prüfen
            mock_emit_warning.assert_called_once_with(
                "Anomalie erkannt",
                {"tenant": "tenant1", "metric": "cpu", "score": -0.8}
            )

    @pytest.mark.asyncio
    async def test_update_metrics_exception_handling(self):
        """Testet _update_metrics mit Exception-Handling."""
        with patch("services.anomaly.service.ANOMALY_SCORE_GAUGE") as mock_gauge:
            mock_gauge.labels.side_effect = Exception("Prometheus error")

            # Sollte keine Exception werfen
            await self.service._update_metrics("tenant1", "cpu", 0.5, False)

    @pytest.mark.asyncio
    async def test_detect_insufficient_data(self):
        """Testet detect mit unzureichenden Daten."""
        values = [1.0, 2.0]  # Zu wenige Werte
        result = await self.service.detect(tenant="tenant1", metric_name="cpu", values=values)

        expected = {"ready": False, "reason": "insufficient_data_or_no_ml"}
        assert result == expected

    @pytest.mark.asyncio
    async def test_detect_detection_failed(self):
        """Testet detect wenn Detection fehlschlägt."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        with patch.object(self.service, "_validate_input", return_value=True), \
             patch.object(self.service, "_perform_detection", return_value=None):

            result = await self.service.detect(tenant="tenant1", metric_name="cpu", values=values)

            expected = {"ready": False, "reason": "detection_failed"}
            assert result == expected

    @pytest.mark.asyncio
    async def test_detect_success_no_outlier(self):
        """Testet detect bei erfolgreichem Aufruf ohne Outlier."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        with patch.object(self.service, "_validate_input", return_value=True), \
             patch.object(self.service, "_perform_detection", return_value=(0.5, False)), \
             patch.object(self.service, "_update_metrics") as mock_update:

            result = await self.service.detect(tenant="tenant1", metric_name="cpu", values=values)

            expected = {"ready": True, "outlier": False, "score": 0.5}
            assert result == expected
            mock_update.assert_called_once_with("tenant1", "cpu", 0.5, False)

    @pytest.mark.asyncio
    async def test_detect_success_with_outlier(self):
        """Testet detect bei erfolgreichem Aufruf mit Outlier."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0]

        with patch.object(self.service, "_validate_input", return_value=True), \
             patch.object(self.service, "_perform_detection", return_value=(-0.8, True)), \
             patch.object(self.service, "_update_metrics") as mock_update:

            result = await self.service.detect(tenant="tenant1", metric_name="cpu", values=values)

            expected = {"ready": True, "outlier": True, "score": -0.8}
            assert result == expected
            mock_update.assert_called_once_with("tenant1", "cpu", -0.8, True)

    @pytest.mark.asyncio
    async def test_learn_baseline_empty_values(self):
        """Testet learn_baseline mit leeren Werten."""
        result = await self.service.learn_baseline(tenant="tenant1", metric_name="cpu", values=[])
        assert result is False

    @pytest.mark.asyncio
    async def test_learn_baseline_success(self):
        """Testet learn_baseline bei erfolgreichem Aufruf."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        with patch("services.anomaly.service.calculate_statistics") as mock_calc_stats, \
             patch("services.anomaly.service.redis_helper") as mock_redis_helper:

            mock_calc_stats.return_value = {"mean": 3.0, "variance": 2.5, "stddev": 1.58}
            mock_redis_helper.store_baseline = AsyncMock(return_value=True)

            result = await self.service.learn_baseline(tenant="tenant1", metric_name="cpu", values=values)

            assert result is True
            mock_calc_stats.assert_called_once_with(values)
            mock_redis_helper.store_baseline.assert_called_once_with(
                "tenant1",
                "cpu",
                {"mean": 3.0, "variance": 2.5, "stddev": 1.58}
            )

    @pytest.mark.asyncio
    async def test_adaptive_threshold_no_baseline(self):
        """Testet adaptive_threshold ohne gespeicherte Baseline."""
        with patch("services.anomaly.service.redis_helper") as mock_redis_helper:
            mock_redis_helper.get_baseline = AsyncMock(return_value=None)

            result = await self.service.adaptive_threshold(tenant="tenant1", metric_name="cpu")

            assert result is None
            mock_redis_helper.get_baseline.assert_called_once_with("tenant1", "cpu")

    @pytest.mark.asyncio
    async def test_adaptive_threshold_success(self):
        """Testet adaptive_threshold bei erfolgreichem Aufruf."""
        baseline_stats = {"mean": 5.0, "variance": 4.0, "stddev": 2.0}

        with patch("services.anomaly.service.redis_helper") as mock_redis_helper:
            mock_redis_helper.get_baseline = AsyncMock(return_value=baseline_stats)

            result = await self.service.adaptive_threshold(tenant="tenant1", metric_name="cpu")

            # mean ± ANOMALY_THRESHOLD_SIGMA * stddev = 5.0 ± 3.0 * 2.0 = (-1.0, 11.0)
            expected_lower = 5.0 - ANOMALY_THRESHOLD_SIGMA * 2.0
            expected_upper = 5.0 + ANOMALY_THRESHOLD_SIGMA * 2.0
            expected = (expected_lower, expected_upper)

            assert result == expected
            mock_redis_helper.get_baseline.assert_called_once_with("tenant1", "cpu")

    @pytest.mark.asyncio
    async def test_adaptive_threshold_missing_stats(self):
        """Testet adaptive_threshold mit unvollständigen Statistiken."""
        baseline_stats = {"mean": 5.0}  # stddev fehlt

        with patch("services.anomaly.service.redis_helper") as mock_redis_helper:
            mock_redis_helper.get_baseline = AsyncMock(return_value=baseline_stats)

            result = await self.service.adaptive_threshold(tenant="tenant1", metric_name="cpu")

            # Sollte mit stddev=0.0 rechnen
            expected = (5.0, 5.0)  # mean ± 3.0 * 0.0
            assert result == expected
