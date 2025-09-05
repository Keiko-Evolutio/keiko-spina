"""Gemeinsame Utilities und Konfigurationen für Anomaly Detection Service.

Konsolidiert wiederverwendbare Komponenten, Konfigurationen und Utility-Funktionen
für alle Anomaly-Detection-Module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

logger = get_logger(__name__)

T = TypeVar("T")

# =====================================================================
# Konstanten
# =====================================================================

# Machine Learning Konstanten
DEFAULT_CONTAMINATION_RATE: float = 0.02  # 2% erwartete Anomalierate
DEFAULT_RANDOM_STATE: int = 42  # Reproduzierbare Zufallszahlen
MIN_SAMPLES_THRESHOLD: int = 50  # Mindestanzahl Samples für ML

# LSTM Konstanten
DEFAULT_WINDOW_SIZE: int = 20  # Sliding Window Größe
DEFAULT_LATENT_DIM: int = 16  # Latente Dimension des Autoencoders
DEFAULT_EPOCHS: int = 5  # Training-Epochen
DEFAULT_BATCH_SIZE: int = 32  # Batch-Größe für Training

# Training Scheduler Konstanten
DEFAULT_TRAINING_INTERVAL_SECONDS: int = 300  # 5 Minuten
TRAIN_QUEUE_KEY: str = "kei:anomaly:train:queue"

# Cache Konstanten
BASELINE_CACHE_PREFIX: str = "kei:anomaly:baseline"
DEFAULT_CACHE_TTL: int = 3600  # 1 Stunde

# Anomalie-Detection Konstanten
ANOMALY_THRESHOLD_SIGMA: float = 3.0  # 3-Sigma-Regel für Schwellwerte


# =====================================================================
# Basis-Konfigurationsklassen
# =====================================================================

@dataclass
class BaseAnomalyConfig:
    """Basis-Konfiguration für alle Anomaly-Detection-Komponenten."""

    random_state: int = DEFAULT_RANDOM_STATE
    min_samples: int = MIN_SAMPLES_THRESHOLD


@dataclass
class DetectionConfig(BaseAnomalyConfig):
    """Konfiguration für Anomalieerkennung."""

    contamination: float = DEFAULT_CONTAMINATION_RATE


@dataclass
class LSTMConfig(BaseAnomalyConfig):
    """Konfigurationsparameter für LSTM-basiertes Modell."""

    window_size: int = DEFAULT_WINDOW_SIZE
    latent_dim: int = DEFAULT_LATENT_DIM
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class TrainingConfig(BaseAnomalyConfig):
    """Konfiguration für Training-Scheduler."""

    interval_seconds: int = DEFAULT_TRAINING_INTERVAL_SECONDS
    queue_key: str = TRAIN_QUEUE_KEY


# =====================================================================
# Utility-Funktionen
# =====================================================================

def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Berechnet Basis-Statistiken für eine Zeitreihe.

    Args:
        values: Liste von Zeitreihenwerten

    Returns:
        Dictionary mit mean, variance, stddev
    """
    if not values:
        return {"mean": 0.0, "variance": 0.0, "stddev": 0.0}

    mean = float(sum(values) / len(values))
    variance = float(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1))
    stddev = float(math.sqrt(max(0.0, variance)))

    return {"mean": mean, "variance": variance, "stddev": stddev}


def format_cache_key(tenant: str, metric_name: str, suffix: str = "") -> str:
    """Erstellt einheitlichen Cache-Key für Anomaly-Daten.

    Args:
        tenant: Tenant-ID
        metric_name: Name der Metrik
        suffix: Optionaler Suffix

    Returns:
        Formatierter Cache-Key
    """
    key = f"{BASELINE_CACHE_PREFIX}:{tenant}:{metric_name}"
    if suffix:
        key += f":{suffix}"
    return key


def serialize_training_job(tenant: str, metric: str, values: list[float]) -> str:
    """Serialisiert Training-Job für Redis-Queue.

    Args:
        tenant: Tenant-ID
        metric: Metrik-Name
        values: Zeitreihenwerte

    Returns:
        JSON-serialisierter Job
    """
    return json.dumps({
        "tenant": tenant,
        "metric": metric,
        "values": values
    })


def deserialize_training_job(job_data: str) -> dict[str, Any]:
    """Deserialisiert Training-Job aus Redis-Queue.

    Args:
        job_data: JSON-String des Jobs

    Returns:
        Dictionary mit Job-Daten
    """
    try:
        return json.loads(job_data)
    except (json.JSONDecodeError, TypeError):
        return {"tenant": "default", "metric": "unknown", "values": []}


# =====================================================================
# Exception-Handling Utilities
# =====================================================================

async def safe_redis_operation(
    operation: str,
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """Führt Redis-Operation mit Exception-Handling aus.

    Args:
        operation: Name der Operation für Logging
        func: Auszuführende Funktion
        *args: Positionale Argumente für func
        default_return: Rückgabewert bei Fehlern
        log_errors: Ob Fehler geloggt werden sollen
        **kwargs: Keyword-Argumente für func

    Returns:
        Ergebnis der Operation oder default_return bei Fehlern
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"Redis-Operation '{operation}' fehlgeschlagen: {e}")
        return default_return


def safe_ml_operation(
    operation: str,
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """Führt ML-Operation mit Exception-Handling aus.

    Args:
        operation: Name der Operation für Logging
        func: Auszuführende Funktion
        *args: Positionale Argumente für func
        default_return: Rückgabewert bei Fehlern
        log_errors: Ob Fehler geloggt werden sollen
        **kwargs: Keyword-Argumente für func

    Returns:
        Ergebnis der Operation oder default_return bei Fehlern
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"ML-Operation '{operation}' fehlgeschlagen: {e}")
        return default_return


# =====================================================================
# Redis-Utility-Klasse
# =====================================================================

class AnomalyRedisHelper:
    """Vereinfachte Redis-Operationen für Anomaly-Detection."""

    def __init__(self):
        self._client: Any | NoOpCache | None = None

    async def _ensure_client(self) -> Any | NoOpCache | None:
        """Stellt Redis-Client sicher."""
        if self._client is None:
            self._client = await get_cache_client()
        return self._client

    async def store_baseline(
        self,
        tenant: str,
        metric_name: str,
        stats: dict[str, float],
        ttl: int = DEFAULT_CACHE_TTL
    ) -> bool:
        """Speichert Baseline-Statistiken in Redis.

        Args:
            tenant: Tenant-ID
            metric_name: Name der Metrik
            stats: Statistiken-Dictionary
            ttl: Time-to-live in Sekunden

        Returns:
            Erfolg der Operation
        """
        client = await self._ensure_client()
        if client is None or isinstance(client, NoOpCache):
            return False

        cache_key = format_cache_key(tenant, metric_name, "baseline")

        return await safe_redis_operation(
            "store_baseline",
            client.setex,
            cache_key,
            ttl,
            json.dumps(stats),
            default_return=False
        )

    async def get_baseline(
        self,
        tenant: str,
        metric_name: str
    ) -> dict[str, float] | None:
        """Lädt Baseline-Statistiken aus Redis.

        Args:
            tenant: Tenant-ID
            metric_name: Name der Metrik

        Returns:
            Statistiken-Dictionary oder None
        """
        client = await self._ensure_client()
        if client is None or isinstance(client, NoOpCache):
            return None

        cache_key = format_cache_key(tenant, metric_name, "baseline")

        result = await safe_redis_operation(
            "get_baseline",
            client.get,
            cache_key,
            default_return=None
        )

        if result:
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    async def enqueue_training_job(
        self,
        tenant: str,
        metric: str,
        values: list[float]
    ) -> bool:
        """Fügt Training-Job zur Queue hinzu.

        Args:
            tenant: Tenant-ID
            metric: Metrik-Name
            values: Zeitreihenwerte

        Returns:
            Erfolg der Operation
        """
        client = await self._ensure_client()
        if client is None or isinstance(client, NoOpCache):
            return False

        job_data = serialize_training_job(tenant, metric, values)

        return await safe_redis_operation(
            "enqueue_training_job",
            client.lpush,
            TRAIN_QUEUE_KEY,
            job_data,
            default_return=False
        ) is not False

    async def dequeue_training_job(self) -> dict[str, Any] | None:
        """Nimmt Training-Job aus der Queue.

        Returns:
            Job-Dictionary oder None
        """
        client = await self._ensure_client()
        if client is None or isinstance(client, NoOpCache):
            return None

        job_data = await safe_redis_operation(
            "dequeue_training_job",
            client.rpop,
            TRAIN_QUEUE_KEY,
            default_return=None
        )

        if job_data:
            return deserialize_training_job(job_data)
        return None


# Globale Redis-Helper-Instanz
redis_helper = AnomalyRedisHelper()
