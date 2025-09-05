"""Unit-Tests für anomaly/lstm_service.py.

Testet LSTMAnomalyService mit TensorFlow-Mocking und Fallback-Verhalten.
"""

from unittest.mock import MagicMock, patch

from services.anomaly.common import LSTMConfig
from services.anomaly.lstm_service import LSTMAnomalyService


class TestLSTMAnomalyService:
    """Tests für LSTMAnomalyService-Klasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.config = LSTMConfig(
            window_size=10,
            latent_dim=8,
            epochs=3,
            batch_size=16
        )
        self.service = LSTMAnomalyService(self.config)

    def test_init_with_config(self):
        """Testet Initialisierung mit Konfiguration."""
        assert self.service.config == self.config
        assert self.service._model is None

    def test_init_without_config(self):
        """Testet Initialisierung ohne Konfiguration (Standard-Werte)."""
        service = LSTMAnomalyService()
        assert isinstance(service.config, LSTMConfig)
        assert service.config.window_size == 20  # Standard-Wert
        assert service._model is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", False)
    def test_build_model_tf_unavailable(self):
        """Testet _build_model wenn TensorFlow nicht verfügbar ist."""
        result = self.service._build_model()
        assert result is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_build_model_success(self, mock_safe_ml):
        """Testet _build_model bei erfolgreichem Aufruf."""
        # Mock TensorFlow-Komponenten
        mock_model = MagicMock()
        mock_safe_ml.return_value = mock_model

        result = self.service._build_model()

        assert result == mock_model
        mock_safe_ml.assert_called_once()

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_build_model_exception(self, mock_safe_ml):
        """Testet _build_model bei Exception."""
        mock_safe_ml.return_value = None

        result = self.service._build_model()

        assert result is None

    def test_create_windows_insufficient_data(self):
        """Testet _create_windows mit unzureichenden Daten."""
        values = [1.0, 2.0, 3.0]  # Weniger als window_size
        result = self.service._create_windows(values)
        assert result is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", False)
    def test_create_windows_tf_unavailable(self):
        """Testet _create_windows wenn TensorFlow nicht verfügbar ist."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        result = self.service._create_windows(values)
        assert result is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_create_windows_success(self, mock_safe_ml):
        """Testet _create_windows bei erfolgreichem Aufruf."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        mock_windows = MagicMock()
        mock_safe_ml.return_value = mock_windows

        result = self.service._create_windows(values)

        assert result == mock_windows
        mock_safe_ml.assert_called_once()

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", False)
    def test_train_tf_unavailable(self):
        """Testet train wenn TensorFlow nicht verfügbar ist."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        with patch("services.anomaly.lstm_service.logger") as mock_logger:
            result = self.service.train(values)

            assert result is False
            mock_logger.info.assert_called_once_with(
                "TensorFlow nicht verfügbar – LSTM-Training übersprungen"
            )

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    def test_train_insufficient_data(self):
        """Testet train mit unzureichenden Daten."""
        values = [1.0, 2.0, 3.0]  # Weniger als window_size

        with patch.object(self.service, "_create_windows", return_value=None):
            result = self.service.train(values)
            assert result is False

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    def test_train_model_build_failed(self):
        """Testet train wenn Model-Build fehlschlägt."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        mock_training_data = MagicMock()

        with patch.object(self.service, "_create_windows", return_value=mock_training_data), \
             patch.object(self.service, "_build_model", return_value=None):

            result = self.service.train(values)
            assert result is False

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_train_success_new_model(self, mock_safe_ml):
        """Testet train bei erfolgreichem Aufruf mit neuem Modell."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        mock_training_data = MagicMock()
        mock_model = MagicMock()

        # safe_ml_operation gibt True zurück für Training
        mock_safe_ml.return_value = True

        with patch.object(self.service, "_create_windows", return_value=mock_training_data), \
             patch.object(self.service, "_build_model", return_value=mock_model):

            result = self.service.train(values)

            assert result is True
            assert self.service._model == mock_model
            mock_safe_ml.assert_called_once()

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_train_success_existing_model(self, mock_safe_ml):
        """Testet train bei erfolgreichem Aufruf mit existierendem Modell."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        mock_training_data = MagicMock()
        mock_model = MagicMock()

        # Setze existierendes Modell
        self.service._model = mock_model
        mock_safe_ml.return_value = True

        with patch.object(self.service, "_create_windows", return_value=mock_training_data):
            result = self.service.train(values)

            assert result is True
            assert self.service._model == mock_model  # Modell bleibt gleich
            mock_safe_ml.assert_called_once()

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_train_training_failed(self, mock_safe_ml):
        """Testet train wenn Training fehlschlägt."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        mock_training_data = MagicMock()
        mock_model = MagicMock()

        mock_safe_ml.return_value = False  # Training fehlgeschlagen

        with patch.object(self.service, "_create_windows", return_value=mock_training_data), \
             patch.object(self.service, "_build_model", return_value=mock_model):

            result = self.service.train(values)
            assert result is False

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", False)
    def test_score_tf_unavailable(self):
        """Testet score wenn TensorFlow nicht verfügbar ist."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.service.score(values)
        assert result is None

    def test_score_no_model(self):
        """Testet score ohne trainiertes Modell."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # _model ist None (Standard)
        result = self.service.score(values)
        assert result is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    def test_score_insufficient_data(self):
        """Testet score mit unzureichenden Daten."""
        values = [1.0, 2.0, 3.0]  # Weniger als window_size
        mock_model = MagicMock()
        self.service._model = mock_model

        with patch.object(self.service, "_create_windows", return_value=None):
            result = self.service.score(values)
            assert result is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_score_success(self, mock_safe_ml):
        """Testet score bei erfolgreichem Aufruf."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        mock_test_data = MagicMock()
        mock_model = MagicMock()
        self.service._model = mock_model

        # safe_ml_operation gibt MSE-Score zurück
        expected_score = 0.123
        mock_safe_ml.return_value = expected_score

        with patch.object(self.service, "_create_windows", return_value=mock_test_data):
            result = self.service.score(values)

            assert result == expected_score
            mock_safe_ml.assert_called_once()

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    @patch("services.anomaly.lstm_service.safe_ml_operation")
    def test_score_scoring_failed(self, mock_safe_ml):
        """Testet score wenn Scoring fehlschlägt."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        mock_test_data = MagicMock()
        mock_model = MagicMock()
        self.service._model = mock_model

        mock_safe_ml.return_value = None  # Scoring fehlgeschlagen

        with patch.object(self.service, "_create_windows", return_value=mock_test_data):
            result = self.service.score(values)
            assert result is None


class TestLSTMServiceIntegration:
    """Integrationstests für LSTMAnomalyService."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.service = LSTMAnomalyService()

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", False)
    def test_full_workflow_tf_unavailable(self):
        """Testet kompletten Workflow ohne TensorFlow."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        # Training sollte fehlschlagen
        train_result = self.service.train(values)
        assert train_result is False

        # Scoring sollte None zurückgeben
        score_result = self.service.score(values)
        assert score_result is None

    @patch("services.anomaly.lstm_service.TF_AVAILABLE", True)
    def test_workflow_insufficient_data(self):
        """Testet Workflow mit unzureichenden Daten."""
        values = [1.0, 2.0, 3.0]  # Zu wenige Werte

        # Training sollte fehlschlagen
        train_result = self.service.train(values)
        assert train_result is False

        # Scoring sollte None zurückgeben
        score_result = self.service.score(values)
        assert score_result is None
