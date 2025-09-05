"""Anomaly Detection Service.

Analysiert Zeitreihen-Metriken und erkennt Anomalien (Baseline + ML).
Implementiert IsolationForest als baseline-freundliche Methode; LSTM kann
optional ergänzt werden. Ergebnisse werden in Prometheus-Metriken gespiegelt
und via Webhook-Alerting publiziert.
"""

from __future__ import annotations

from typing import Any

try:
    from sklearn.ensemble import IsolationForest  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

import contextlib

from kei_logging import get_logger
from monitoring.metrics_definitions import ANOMALY_DETECTIONS_TOTAL, ANOMALY_SCORE_GAUGE
from services.webhooks.alerting import emit_warning

from .common import (
    ANOMALY_THRESHOLD_SIGMA,
    DetectionConfig,
    calculate_statistics,
    redis_helper,
    safe_ml_operation,
)

logger = get_logger(__name__)


class AnomalyDetectionService:
    """Einfacher Service für zeitnahe Anomalieerkennung."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    async def detect(self, *, tenant: str, metric_name: str, values: list[float]) -> dict[str, Any]:
        """Detektiert Anomalien in einer Zeitreihe.

        Args:
            tenant: Tenant-ID
            metric_name: Name der Metrik
            values: Zeitreihenwerte (gleichmäßig gesampelt)

        Returns:
            Ergebnis mit Score/Outlier-Flags
        """
        if not self._validate_input(values):
            return {"ready": False, "reason": "insufficient_data_or_no_ml"}

        detection_result = self._perform_detection(values)
        if detection_result is None:
            return {"ready": False, "reason": "detection_failed"}

        last_score, is_outlier = detection_result

        # Metriken und Alerting
        await self._update_metrics(tenant, metric_name, last_score, is_outlier)

        return {"ready": True, "outlier": bool(is_outlier), "score": last_score}

    def _validate_input(self, values: list[float]) -> bool:
        """Validiert Eingabedaten für Anomalieerkennung."""
        return (
            bool(values) and
            len(values) >= self.config.min_samples and
            SKLEARN_AVAILABLE
        )

    def _perform_detection(self, values: list[float]) -> tuple[float, bool] | None:
        """Führt Anomalieerkennung mit IsolationForest durch."""
        def detection_operation():
            # IsolationForest erwartet 2D Input
            data = [[float(v)] for v in values]
            model = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_state
            )
            predictions = model.fit_predict(data)
            scores = model.decision_function(data)  # Höher = normaler

            # Einfache Heuristik: letzter Punkt bewertet
            last_score = float(scores[-1])
            is_outlier = predictions[-1] == -1

            return last_score, is_outlier

        return safe_ml_operation(
            "isolation_forest_detection",
            detection_operation,
            default_return=None
        )

    async def _update_metrics(self, tenant: str, metric_name: str, score: float, is_outlier: bool) -> None:
        """Aktualisiert Prometheus-Metriken und sendet Alerts."""
        # Prometheus-Metriken (best effort)
        with contextlib.suppress(Exception):
            ANOMALY_SCORE_GAUGE.labels(tenant=tenant, metric=metric_name).set(score)

        if is_outlier:
            with contextlib.suppress(Exception):
                ANOMALY_DETECTIONS_TOTAL.labels(
                    tenant=tenant,
                    metric=metric_name,
                    severity="warning"
                ).inc()

            # Webhook-Benachrichtigung (best effort)
            with contextlib.suppress(Exception):
                await emit_warning(
                    "Anomalie erkannt",
                    {"tenant": tenant, "metric": metric_name, "score": score}
                )

    async def learn_baseline(self, *, tenant: str, metric_name: str, values: list[float]) -> bool:
        """Lernt Baseline-Statistiken und persistiert sie in Redis.

        Returns:
            Erfolg/Fehlschlag
        """
        if not values:
            return False

        # Berechne Statistiken mit gemeinsamer Utility-Funktion
        stats = calculate_statistics(values)

        # Persistierung über Redis-Helper
        return await redis_helper.store_baseline(tenant, metric_name, stats)

    async def adaptive_threshold(self, *, tenant: str, metric_name: str) -> tuple[float, float] | None:
        """Liest gelerntes Baseline-Fenster und gibt adaptiven Schwellwert zurück (mean ± 3*stddev)."""
        stats = await redis_helper.get_baseline(tenant, metric_name)
        if not stats:
            return None

        mean = stats.get("mean", 0.0)
        stddev = stats.get("stddev", 0.0)

        # Anomalie-Schwellwerte basierend auf konfigurierbarer Sigma-Regel
        return (
            mean - ANOMALY_THRESHOLD_SIGMA * stddev,
            mean + ANOMALY_THRESHOLD_SIGMA * stddev
        )
