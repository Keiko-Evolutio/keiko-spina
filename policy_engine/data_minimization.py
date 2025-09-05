# backend/policy_engine/data_minimization.py
"""Data Minimization Engine für Keiko Personal Assistant

Implementiert automatische Datenminimierung, Smart-Sampling,
Data-Lifecycle-Management und Differential-Privacy-Mechanismen.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class SamplingStrategy(str, Enum):
    """Strategien für Smart-Sampling."""
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    RESERVOIR = "reservoir"
    IMPORTANCE = "importance"


class PrivacyMechanism(str, Enum):
    """Differential-Privacy-Mechanismen."""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"


class DataLifecycleStage(str, Enum):
    """Stadien im Data-Lifecycle."""
    COLLECTION = "collection"
    PROCESSING = "processing"
    STORAGE = "storage"
    ANALYSIS = "analysis"
    ARCHIVAL = "archival"
    DELETION = "deletion"


@dataclass
class DataMinimizationPolicy:
    """Policy für Datenminimierung."""
    policy_id: str
    name: str
    description: str
    data_types: set[str]
    purpose: str
    retention_days: int
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    sampling_rate: float = 1.0  # 0.0 - 1.0
    privacy_mechanism: PrivacyMechanism | None = None
    privacy_budget: float = 1.0  # Epsilon für Differential Privacy
    auto_apply: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MinimizationResult:
    """Ergebnis einer Datenminimierung."""
    original_size: int
    minimized_size: int
    reduction_ratio: float
    applied_policies: list[str] = field(default_factory=list)
    sampling_applied: bool = False
    privacy_applied: bool = False
    processing_time_ms: float = 0.0

    @property
    def size_reduction_percent(self) -> float:
        """Gibt Größenreduktion in Prozent zurück."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.minimized_size / self.original_size) * 100


@dataclass
class DataLifecycleEvent:
    """Event im Data-Lifecycle."""
    event_id: str
    data_id: str
    stage: DataLifecycleStage
    action: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class PurposeBinder:
    """Bindet Datenverarbeitung an spezifische Zwecke."""

    def __init__(self):
        """Initialisiert Purpose Binder."""
        self._purpose_bindings: dict[str, dict[str, Any]] = {}
        self._data_purposes: dict[str, set[str]] = {}  # data_id -> purposes

    def bind_data_to_purpose(
        self,
        data_id: str,
        purpose: str,
        allowed_operations: set[str],
        retention_days: int,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Bindet Daten an spezifischen Zweck."""
        binding = {
            "purpose": purpose,
            "allowed_operations": allowed_operations,
            "retention_days": retention_days,
            "bound_at": datetime.now(UTC),
            "metadata": metadata or {}
        }

        self._purpose_bindings[f"{data_id}:{purpose}"] = binding

        if data_id not in self._data_purposes:
            self._data_purposes[data_id] = set()
        self._data_purposes[data_id].add(purpose)

        logger.info(f"Daten {data_id} an Zweck {purpose} gebunden")

    def is_operation_allowed(
        self,
        data_id: str,
        operation: str,
        purpose: str
    ) -> bool:
        """Prüft, ob Operation für Zweck erlaubt ist."""
        binding_key = f"{data_id}:{purpose}"
        binding = self._purpose_bindings.get(binding_key)

        if not binding:
            return False

        return operation in binding["allowed_operations"]

    def get_data_purposes(self, data_id: str) -> set[str]:
        """Gibt alle Zwecke für Daten zurück."""
        return self._data_purposes.get(data_id, set())

    def is_retention_expired(self, data_id: str, purpose: str) -> bool:
        """Prüft, ob Aufbewahrungsfrist abgelaufen ist."""
        binding_key = f"{data_id}:{purpose}"
        binding = self._purpose_bindings.get(binding_key)

        if not binding:
            return True

        bound_at = binding["bound_at"]
        retention_days = binding["retention_days"]
        expiry_date = bound_at + timedelta(days=retention_days)

        return datetime.now(UTC) > expiry_date


class SmartSampler:
    """Smart-Sampling für große Datensätze."""

    def __init__(self):
        """Initialisiert Smart Sampler."""
        self._sampling_stats = {
            "samples_created": 0,
            "total_data_processed": 0,
            "average_reduction": 0.0
        }

    def sample_data(
        self,
        data: list[Any],
        strategy: SamplingStrategy,
        sample_rate: float,
        metadata: dict[str, Any] | None = None
    ) -> tuple[list[Any], dict[str, Any]]:
        """Führt Smart-Sampling durch."""
        if not data or sample_rate >= 1.0:
            return data, {"strategy": strategy.value, "sample_rate": sample_rate}

        sample_size = max(1, int(len(data) * sample_rate))

        if strategy == SamplingStrategy.RANDOM:
            sampled_data = self._random_sampling(data, sample_size)
        elif strategy == SamplingStrategy.SYSTEMATIC:
            sampled_data = self._systematic_sampling(data, sample_size)
        elif strategy == SamplingStrategy.STRATIFIED:
            sampled_data = self._stratified_sampling(data, sample_size, metadata)
        elif strategy == SamplingStrategy.RESERVOIR:
            sampled_data = self._reservoir_sampling(data, sample_size)
        elif strategy == SamplingStrategy.IMPORTANCE:
            sampled_data = self._importance_sampling(data, sample_size, metadata)
        else:
            sampled_data = self._random_sampling(data, sample_size)

        # Aktualisiere Statistiken
        self._sampling_stats["samples_created"] += 1
        self._sampling_stats["total_data_processed"] += len(data)
        reduction = 1 - len(sampled_data) / len(data)
        self._sampling_stats["average_reduction"] = (
            (self._sampling_stats["average_reduction"] * (self._sampling_stats["samples_created"] - 1) + reduction) /
            self._sampling_stats["samples_created"]
        )

        sampling_info = {
            "strategy": strategy.value,
            "original_size": len(data),
            "sample_size": len(sampled_data),
            "sample_rate": sample_rate,
            "actual_rate": len(sampled_data) / len(data)
        }

        return sampled_data, sampling_info

    def _random_sampling(self, data: list[Any], sample_size: int) -> list[Any]:
        """Zufälliges Sampling."""
        return random.sample(data, min(sample_size, len(data)))

    def _systematic_sampling(self, data: list[Any], sample_size: int) -> list[Any]:
        """Systematisches Sampling."""
        if sample_size >= len(data):
            return data

        step = len(data) // sample_size
        start = random.randint(0, step - 1)

        return [data[i] for i in range(start, len(data), step)][:sample_size]

    def _stratified_sampling(
        self,
        data: list[Any],
        sample_size: int,
        metadata: dict[str, Any] | None
    ) -> list[Any]:
        """Stratifiziertes Sampling."""
        # Vereinfachte Implementierung - gruppiert nach Hash
        if not metadata or "strata_key" not in metadata:
            return self._random_sampling(data, sample_size)

        strata_key = metadata["strata_key"]
        strata = {}

        for item in data:
            # Bestimme Stratum basierend auf strata_key
            if isinstance(item, dict) and strata_key in item:
                stratum = item[strata_key]
            elif hasattr(item, strata_key):
                stratum = getattr(item, strata_key)
            else:
                # Fallback: Hash-basierte Strata-Bestimmung
                stratum = hash(str(item)) % 10

            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(item)

        # Proportionale Aufteilung
        sampled_data = []
        for stratum_data in strata.values():
            stratum_sample_size = max(1, int(len(stratum_data) * sample_size / len(data)))
            sampled_data.extend(self._random_sampling(stratum_data, stratum_sample_size))

        return sampled_data[:sample_size]

    def _reservoir_sampling(self, data: list[Any], sample_size: int) -> list[Any]:
        """Reservoir-Sampling für Streams."""
        reservoir = []

        for i, item in enumerate(data):
            if i < sample_size:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = item

        return reservoir

    def _importance_sampling(
        self,
        data: list[Any],
        sample_size: int,
        metadata: dict[str, Any] | None
    ) -> list[Any]:
        """Importance-Sampling basierend auf Gewichtung."""
        if not metadata or "importance_weights" not in metadata:
            return self._random_sampling(data, sample_size)

        weights = metadata["importance_weights"]
        if len(weights) != len(data):
            return self._random_sampling(data, sample_size)

        # Gewichtete Auswahl
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        sampled_indices = random.choices(range(len(data)), weights=probabilities, k=sample_size)
        return [data[i] for i in sampled_indices]


class DifferentialPrivacyMechanism:
    """Implementiert Differential-Privacy-Mechanismen."""

    def __init__(self):
        """Initialisiert Differential Privacy Mechanism."""
        self._privacy_budget_used = 0.0
        self._max_privacy_budget = 10.0

    def add_noise(
        self,
        data: float | list[float],
        mechanism: PrivacyMechanism,
        epsilon: float,
        sensitivity: float = 1.0
    ) -> float | list[float]:
        """Fügt Differential-Privacy-Noise hinzu."""
        if self._privacy_budget_used + epsilon > self._max_privacy_budget:
            logger.warning("Privacy-Budget überschritten")
            return data

        self._privacy_budget_used += epsilon

        if isinstance(data, int | float):
            return self._add_noise_scalar(data, mechanism, epsilon, sensitivity)
        if isinstance(data, list):
            return [self._add_noise_scalar(x, mechanism, epsilon, sensitivity) for x in data]
        logger.warning(f"Unsupported data type for privacy mechanism: {type(data)}")
        return data

    def _add_noise_scalar(
        self,
        value: float,
        mechanism: PrivacyMechanism,
        epsilon: float,
        sensitivity: float
    ) -> float:
        """Fügt Noise zu Skalar-Wert hinzu."""
        if mechanism == PrivacyMechanism.LAPLACE:
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale)
            return value + noise

        if mechanism == PrivacyMechanism.GAUSSIAN:
            # Gaussian-Mechanismus für (epsilon, delta)-DP
            delta = 1e-5  # Kleine Delta für praktische Anwendung
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            noise = random.gauss(0, sigma)
            return value + noise

        if mechanism == PrivacyMechanism.EXPONENTIAL:
            # Vereinfachte Exponential-Mechanismus-Implementierung
            noise = np.random.exponential(1 / epsilon)
            return value + noise

        if mechanism == PrivacyMechanism.RANDOMIZED_RESPONSE:
            # Randomized Response für binäre Werte
            p = math.exp(epsilon) / (math.exp(epsilon) + 1)
            if random.random() < p:
                return value
            return 1 - value if value in [0, 1] else value

        return value

    def get_remaining_budget(self) -> float:
        """Gibt verbleibendes Privacy-Budget zurück."""
        return max(0, self._max_privacy_budget - self._privacy_budget_used)

    def reset_budget(self) -> None:
        """Setzt Privacy-Budget zurück."""
        self._privacy_budget_used = 0.0


class DataLifecycleManager:
    """Manager für Data-Lifecycle."""

    def __init__(self):
        """Initialisiert Data Lifecycle Manager."""
        self._lifecycle_events: list[DataLifecycleEvent] = []
        self._data_registry: dict[str, dict[str, Any]] = {}
        self._auto_deletion_enabled = True

    def register_data(
        self,
        data_id: str,
        data_type: str,
        purpose: str,
        retention_days: int,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Registriert Daten im Lifecycle."""
        self._data_registry[data_id] = {
            "data_type": data_type,
            "purpose": purpose,
            "retention_days": retention_days,
            "created_at": datetime.now(UTC),
            "metadata": metadata or {},
            "current_stage": DataLifecycleStage.COLLECTION
        }

        self._log_lifecycle_event(
            data_id,
            DataLifecycleStage.COLLECTION,
            "data_registered"
        )

    def transition_stage(
        self,
        data_id: str,
        new_stage: DataLifecycleStage,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Überführt Daten in neues Lifecycle-Stadium."""
        if data_id not in self._data_registry:
            return False

        old_stage = self._data_registry[data_id]["current_stage"]
        self._data_registry[data_id]["current_stage"] = new_stage

        if metadata:
            self._data_registry[data_id]["metadata"].update(metadata)

        self._log_lifecycle_event(
            data_id,
            new_stage,
            f"transition_from_{old_stage.value}",
            metadata
        )

        return True

    def _log_lifecycle_event(
        self,
        data_id: str,
        stage: DataLifecycleStage,
        action: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Loggt Lifecycle-Event."""
        import uuid

        event = DataLifecycleEvent(
            event_id=str(uuid.uuid4()),
            data_id=data_id,
            stage=stage,
            action=action,
            metadata=metadata or {}
        )

        self._lifecycle_events.append(event)

    async def cleanup_expired_data(self) -> int:
        """Bereinigt abgelaufene Daten."""
        if not self._auto_deletion_enabled:
            return 0

        deleted_count = 0
        current_time = datetime.now(UTC)

        for data_id, data_info in list(self._data_registry.items()):
            created_at = data_info["created_at"]
            retention_days = data_info["retention_days"]
            expiry_date = created_at + timedelta(days=retention_days)

            if current_time > expiry_date:
                # Markiere als gelöscht
                self.transition_stage(data_id, DataLifecycleStage.DELETION)

                # Entferne aus Registry
                del self._data_registry[data_id]
                deleted_count += 1

                logger.info(f"Abgelaufene Daten gelöscht: {data_id}")

        return deleted_count

    def get_data_lifecycle(self, data_id: str) -> list[DataLifecycleEvent]:
        """Gibt Lifecycle-Events für Daten zurück."""
        return [event for event in self._lifecycle_events if event.data_id == data_id]


class DataMinimizationEngine:
    """Engine für Datenminimierung."""

    def __init__(self):
        """Initialisiert Data Minimization Engine."""
        self._policies: list[DataMinimizationPolicy] = []
        self.purpose_binder = PurposeBinder()
        self.smart_sampler = SmartSampler()
        self.privacy_mechanism = DifferentialPrivacyMechanism()
        self.lifecycle_manager = DataLifecycleManager()

        # Statistiken
        self._minimizations_performed = 0
        self._total_size_reduction = 0.0

    def add_policy(self, policy: DataMinimizationPolicy) -> None:
        """Fügt Minimierungsrichtlinie hinzu."""
        self._policies.append(policy)
        logger.info(f"Data-Minimization-Policy hinzugefügt: {policy.name}")

    @trace_function("data_minimization.apply")
    async def apply_minimization(
        self,
        data: Any,
        data_type: str,
        purpose: str,
        context: dict[str, Any] | None = None
    ) -> MinimizationResult:
        """Wendet Datenminimierung an."""
        start_time = time.time()
        self._minimizations_performed += 1

        original_size = self._calculate_data_size(data)
        applicable_policies = self._find_applicable_policies(data_type, purpose)

        minimized_data, applied_policies, sampling_applied, privacy_applied = (
            await self._apply_policies(data, applicable_policies, context)
        )

        return self._create_minimization_result(
            data, minimized_data, original_size, applied_policies,
            sampling_applied, privacy_applied, start_time
        )

    def _find_applicable_policies(self, data_type: str, purpose: str) -> list[DataMinimizationPolicy]:
        """Findet anwendbare Policies für den Datentyp und Zweck."""
        return [
            policy for policy in self._policies
            if (data_type in policy.data_types and
                policy.auto_apply and
                policy.purpose == purpose)
        ]

    async def _apply_policies(
        self,
        data: Any,
        policies: list[DataMinimizationPolicy],
        context: dict[str, Any] | None
    ) -> tuple[Any, list[str], bool, bool]:
        """Wendet alle anwendbaren Policies an."""
        minimized_data = data
        applied_policies = []
        sampling_applied = False
        privacy_applied = False

        for policy in policies:
            minimized_data, sampling_applied = self._apply_sampling_if_needed(
                minimized_data, policy, context, sampling_applied
            )

            minimized_data, privacy_applied = self._apply_privacy_if_needed(
                minimized_data, policy, privacy_applied
            )

            applied_policies.append(policy.policy_id)

        return minimized_data, applied_policies, sampling_applied, privacy_applied

    def _apply_sampling_if_needed(
        self,
        data: Any,
        policy: DataMinimizationPolicy,
        context: dict[str, Any] | None,
        sampling_applied: bool
    ) -> tuple[Any, bool]:
        """Wendet Sampling an falls erforderlich."""
        if isinstance(data, list) and policy.sampling_rate < 1.0:
            data, sampling_info = self.smart_sampler.sample_data(
                data,
                policy.sampling_strategy,
                policy.sampling_rate,
                context
            )
            sampling_applied = True

        return data, sampling_applied

    def _apply_privacy_if_needed(
        self,
        data: Any,
        policy: DataMinimizationPolicy,
        privacy_applied: bool
    ) -> tuple[Any, bool]:
        """Wendet Differential Privacy an falls erforderlich."""
        if policy.privacy_mechanism and isinstance(data, list | int | float):
            data = self.privacy_mechanism.add_noise(
                data,
                policy.privacy_mechanism,
                policy.privacy_budget
            )
            privacy_applied = True

        return data, privacy_applied

    def _create_minimization_result(
        self,
        original_data: Any,
        minimized_data: Any,
        original_size: int,
        applied_policies: list[str],
        sampling_applied: bool,
        privacy_applied: bool,
        start_time: float
    ) -> MinimizationResult:
        """Erstellt das finale Minimization-Ergebnis."""
        minimized_size = self._calculate_data_size(minimized_data)
        reduction_ratio = 1 - minimized_size / original_size if original_size > 0 else 0
        processing_time = (time.time() - start_time) * 1000

        # Validiere Datenintegrität
        self._validate_minimization_integrity(original_data, minimized_data)

        # Aktualisiere Statistiken
        self._total_size_reduction += reduction_ratio

        # Logge Minimierungsdetails für Debugging
        logger.debug(
            "Datenminimierung abgeschlossen",
            extra={
                "original_type": type(original_data).__name__,
                "original_size": original_size,
                "minimized_size": minimized_size,
                "reduction_ratio": reduction_ratio,
                "applied_policies": applied_policies,
                "processing_time_ms": processing_time
            }
        )

        return MinimizationResult(
            original_size=original_size,
            minimized_size=minimized_size,
            reduction_ratio=reduction_ratio,
            applied_policies=applied_policies,
            sampling_applied=sampling_applied,
            privacy_applied=privacy_applied,
            processing_time_ms=processing_time
        )

    def _validate_minimization_integrity(self, original_data: Any, minimized_data: Any) -> None:
        """Validiert die Integrität der Datenminimierung."""
        # Prüfe Datentyp-Konsistenz
        if type(original_data) != type(minimized_data):
            logger.warning(
                f"Datentyp-Inkonsistenz nach Minimierung: {type(original_data)} -> {type(minimized_data)}"
            )

        # Prüfe auf leere Ergebnisse bei nicht-leeren Eingaben
        if original_data and not minimized_data:
            logger.warning("Minimierung resultierte in leeren Daten trotz nicht-leerer Eingabe")

        # Prüfe Strukturintegrität für Listen/Dicts
        if isinstance(original_data, list) and isinstance(minimized_data, list):
            if minimized_data and len(minimized_data) > len(original_data):
                logger.error("Minimierte Daten größer als Original - möglicher Fehler")

        elif isinstance(original_data, dict) and isinstance(minimized_data, dict):
            # Prüfe auf kritische Schlüssel-Verluste
            original_keys = set(original_data.keys())
            minimized_keys = set(minimized_data.keys())
            lost_keys = original_keys - minimized_keys
            if lost_keys:
                logger.debug(f"Schlüssel durch Minimierung entfernt: {lost_keys}")

    def _calculate_data_size(self, data: Any) -> int:
        """Berechnet Datengröße."""
        if isinstance(data, list | tuple | dict | str):
            return len(data)
        return 1

    def get_minimization_statistics(self) -> dict[str, Any]:
        """Gibt Minimierungsstatistiken zurück."""
        avg_reduction = (
            self._total_size_reduction / max(self._minimizations_performed, 1)
        )

        return {
            "minimizations_performed": self._minimizations_performed,
            "average_size_reduction": avg_reduction,
            "total_policies": len(self._policies),
            "privacy_budget_remaining": self.privacy_mechanism.get_remaining_budget(),
            "sampling_stats": self.smart_sampler._sampling_stats,
            "registered_data_items": len(self.lifecycle_manager._data_registry)
        }


# Globale Data Minimization Engine Instanz
data_minimization_engine = DataMinimizationEngine()
