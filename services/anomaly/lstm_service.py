"""TensorFlow LSTM-basierter Anomalie-Service (Fallback-fähig).

Verwendet ein einfaches LSTM-Autoencoder-Modell, um Rekonstruktionsfehler
als Anomalie-Score zu bestimmen. Bei fehlender TensorFlow-Installation
fällt der Service auf No-Op zurück.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np  # type: ignore
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False

from kei_logging import get_logger

from .common import LSTMConfig, safe_ml_operation

logger = get_logger(__name__)


class LSTMAnomalyService:
    """Einfacher LSTM Autoencoder zur Anomalie-Erkennung in Zeitreihen."""

    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.config = config or LSTMConfig()
        self._model: Any | None = None

    def _build_model(self) -> Any | None:
        """Erstellt LSTM-Autoencoder-Modell."""
        if not TF_AVAILABLE:
            return None

        def build_operation():
            window_size = self.config.window_size
            inputs = tf.keras.Input(shape=(window_size, 1))

            # Encoder
            encoded = tf.keras.layers.LSTM(
                self.config.latent_dim,
                return_sequences=False
            )(inputs)

            # Decoder
            decoded = tf.keras.layers.RepeatVector(window_size)(encoded)
            decoded = tf.keras.layers.LSTM(
                self.config.latent_dim,
                return_sequences=True
            )(decoded)
            outputs = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1)
            )(decoded)

            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mse")
            return model

        return safe_ml_operation(
            "build_lstm_model",
            build_operation,
            default_return=None
        )

    def _create_windows(self, values: list[float]) -> Any | None:
        """Erstellt Sliding-Window-Daten für LSTM-Training."""
        if len(values) < self.config.window_size or not TF_AVAILABLE:
            return None

        def window_operation():
            if not TF_AVAILABLE:  # numpy ist nur verfügbar wenn TF verfügbar ist
                return None
            data = np.array(values, dtype="float32").reshape(-1, 1)
            windows = []
            for i in range(len(data) - self.config.window_size + 1):
                windows.append(data[i:i + self.config.window_size])
            return np.stack(windows, axis=0)

        return safe_ml_operation(
            "create_lstm_windows",
            window_operation,
            default_return=None
        )

    def train(self, values: list[float]) -> bool:
        """Trainiert LSTM-Autoencoder mit Zeitreihendaten."""
        if not TF_AVAILABLE:
            logger.info("TensorFlow nicht verfügbar – LSTM-Training übersprungen")
            return False

        training_data = self._create_windows(values)
        if training_data is None:
            return False

        if self._model is None:
            self._model = self._build_model()
            if self._model is None:
                return False

        def training_operation():
            self._model.fit(
                training_data,
                training_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=0
            )
            return True

        return safe_ml_operation(
            "lstm_training",
            training_operation,
            default_return=False
        )

    def score(self, values: list[float]) -> float | None:
        """Berechnet Anomalie-Score basierend auf Rekonstruktionsfehler."""
        if not TF_AVAILABLE or self._model is None:
            return None

        test_data = self._create_windows(values)
        if test_data is None:
            return None

        def scoring_operation():
            reconstruction = self._model.predict(test_data, verbose=0)
            mse = np.mean(np.mean((test_data - reconstruction) ** 2, axis=(1, 2)))
            return float(mse)

        return safe_ml_operation(
            "lstm_scoring",
            scoring_operation,
            default_return=None
        )
