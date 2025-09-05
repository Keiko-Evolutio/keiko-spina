# backend/policy_engine/base_engines.py
"""Gemeinsame Base-Klassen für Policy-Engines.

Konsolidiert die gemeinsamen Patterns aus verschiedenen Engine-Implementierungen
in wiederverwendbare Base-Klassen mit Statistik-Tracking und Error-Handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from kei_logging import get_logger

# Import konsolidierter Konstanten
from .constants import DEFAULT_CACHE_TTL, LONG_HASH_LENGTH, SHORT_HASH_LENGTH

logger = get_logger(__name__)


class BaseEngine(ABC):
    """Basis-Klasse für alle Policy-Engines."""

    def __init__(self, name: str | None = None):
        """Initialisiert Base-Engine."""
        self.name = name or self.__class__.__name__

        # Gemeinsame Statistiken
        self._checks_performed = 0
        self._violations_detected = 0
        self._errors_encountered = 0
        self._total_processing_time = 0.0
        self._last_check_time = None

        # Engine-Status
        self._is_enabled = True
        self._initialization_time = time.time()

    @abstractmethod
    async def process(self, content: str, context: dict[str, Any] | None = None) -> Any:
        """Verarbeitet Content und gibt Ergebnis zurück."""

    async def safe_process(self, content: str, context: dict[str, Any] | None = None) -> Any:
        """Sichere Verarbeitung mit Error-Handling und Statistik-Tracking."""
        if not self._is_enabled:
            logger.warning(f"Engine {self.name} ist deaktiviert")
            return self._get_default_result()

        start_time = time.time()
        self._checks_performed += 1
        self._last_check_time = start_time

        try:
            result = await self.process(content, context)

            # Zähle Violations (falls Result eine Liste ist)
            if isinstance(result, list):
                self._violations_detected += len(result)
            elif hasattr(result, "violations") and isinstance(result.violations, list):
                self._violations_detected += len(result.violations)

            return result

        except Exception as e:
            self._errors_encountered += 1
            logger.exception(f"Fehler in Engine {self.name}: {e}")
            return self._handle_error(e, content, context)

        finally:
            processing_time = time.time() - start_time
            self._total_processing_time += processing_time

    def _get_default_result(self) -> Any:
        """Gibt Standard-Ergebnis zurück wenn Engine deaktiviert ist."""
        return []

    def _handle_error(self, error: Exception, content: str, context: dict[str, Any] | None = None) -> Any:
        """Behandelt Fehler und gibt Fallback-Ergebnis zurück."""
        logger.error(
            f"Engine {self.name} Fallback aktiviert: {error}",
            extra={
                "engine_name": self.name,
                "error_type": type(error).__name__,
                "content_length": len(content),
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "context_keys": list(context.keys()) if context else None,
                "error_count": self._errors_encountered
            }
        )
        return self._get_default_result()

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Engine-Statistiken zurück."""
        uptime = time.time() - self._initialization_time
        avg_processing_time = (
            self._total_processing_time / max(self._checks_performed, 1)
        )

        return {
            "engine_name": self.name,
            "is_enabled": self._is_enabled,
            "uptime_seconds": uptime,
            "checks_performed": self._checks_performed,
            "violations_detected": self._violations_detected,
            "errors_encountered": self._errors_encountered,
            "error_rate": self._errors_encountered / max(self._checks_performed, 1),
            "violation_rate": self._violations_detected / max(self._checks_performed, 1),
            "total_processing_time": self._total_processing_time,
            "average_processing_time": avg_processing_time,
            "last_check_time": self._last_check_time
        }

    def enable(self) -> None:
        """Aktiviert die Engine."""
        self._is_enabled = True
        logger.info(f"Engine {self.name} aktiviert")

    def disable(self) -> None:
        """Deaktiviert die Engine."""
        self._is_enabled = False
        logger.info(f"Engine {self.name} deaktiviert")

    def reset_statistics(self) -> None:
        """Setzt alle Statistiken zurück."""
        self._checks_performed = 0
        self._violations_detected = 0
        self._errors_encountered = 0
        self._total_processing_time = 0.0
        self._last_check_time = None
        logger.info(f"Statistiken für Engine {self.name} zurückgesetzt")


class ValidationEngine(BaseEngine):
    """Basis-Klasse für Validierungs-Engines."""

    def __init__(self, name: str | None = None):
        """Initialisiert Validation-Engine."""
        super().__init__(name)
        self._validation_rules: list[Any] = []
        self._strict_mode = False

    def add_validation_rule(self, rule: Any) -> None:
        """Fügt eine Validierungsregel hinzu."""
        self._validation_rules.append(rule)
        logger.debug(f"Validierungsregel hinzugefügt zu {self.name}")

    def set_strict_mode(self, enabled: bool) -> None:
        """Aktiviert/deaktiviert Strict-Mode."""
        self._strict_mode = enabled
        logger.info(f"Strict-Mode für {self.name}: {enabled}")

    async def validate_all_rules(self, content: str, context: dict[str, Any] | None = None) -> list[Any]:
        """Validiert Content gegen alle Regeln."""
        violations = []

        for rule in self._validation_rules:
            try:
                rule_violations = await self._apply_rule(rule, content, context)
                violations.extend(rule_violations)
            except Exception as e:
                if self._strict_mode:
                    raise
                logger.exception(f"Validierungsregel fehlgeschlagen: {e}")

        return violations

    @abstractmethod
    async def _apply_rule(self, rule: Any, content: str, context: dict[str, Any] | None = None) -> list[Any]:
        """Wendet eine einzelne Regel an."""


class CachingEngine(BaseEngine):
    """Basis-Klasse für Engines mit Caching."""

    def __init__(self, name: str | None = None, cache_ttl: int = DEFAULT_CACHE_TTL):
        """Initialisiert Caching-Engine."""
        super().__init__(name)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = cache_ttl
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, content: str, context: dict[str, Any] | None = None) -> str:
        """Generiert Cache-Key für Content und Context."""
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:LONG_HASH_LENGTH]
        context_hash = ""

        if context:
            context_str = str(sorted(context.items()))
            context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:SHORT_HASH_LENGTH]

        return f"{content_hash}_{context_hash}"

    def _get_from_cache(self, cache_key: str) -> Any | None:
        """Holt Ergebnis aus Cache."""
        cached = self._cache.get(cache_key)

        if cached and time.time() - cached["timestamp"] < self._cache_ttl:
            self._cache_hits += 1
            return cached["result"]

        self._cache_misses += 1
        return None

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Speichert Ergebnis in Cache."""
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

    def clear_cache(self) -> int:
        """Löscht Cache und gibt Anzahl gelöschter Einträge zurück."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache für {self.name} geleert: {cache_size} Einträge")
        return cache_size

    def get_statistics(self) -> dict[str, Any]:
        """Gibt erweiterte Statistiken mit Cache-Metriken zurück."""
        base_stats = super().get_statistics()

        cache_stats = {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            "cache_ttl": self._cache_ttl
        }

        return {**base_stats, **cache_stats}

    async def process(self, content: str, context: dict[str, Any] | None = None) -> Any:
        """Implementiert Caching-Wrapper um process_implementation.

        Args:
            content: Zu verarbeitender Content
            context: Zusätzlicher Kontext

        Returns:
            Verarbeitungsergebnis (aus Cache oder neu berechnet)
        """
        cache_key = self._get_cache_key(content, context)

        # Versuche aus Cache zu holen
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Nicht im Cache - berechne neu
        result = await self.process_implementation(content, context)

        # Speichere im Cache
        self._store_in_cache(cache_key, result)

        return result

    @abstractmethod
    async def process_implementation(self, content: str, context: dict[str, Any] | None = None) -> Any:
        """Implementiert die eigentliche Verarbeitungslogik.

        Diese Methode muss von Subklassen implementiert werden.

        Args:
            content: Zu verarbeitender Content
            context: Zusätzlicher Kontext

        Returns:
            Verarbeitungsergebnis
        """


class PolicyEngineManager:
    """Manager für mehrere Policy-Engines."""

    def __init__(self, name: str = "PolicyEngineManager"):
        """Initialisiert Engine-Manager."""
        self.name = name
        self._engines: dict[str, BaseEngine] = {}
        self._execution_order: list[str] = []

    def register_engine(self, engine: BaseEngine, execution_order: int | None = None) -> None:
        """Registriert eine Engine."""
        self._engines[engine.name] = engine

        if execution_order is not None:
            # Füge an spezifischer Position ein
            if execution_order >= len(self._execution_order):
                self._execution_order.append(engine.name)
            else:
                self._execution_order.insert(execution_order, engine.name)
        else:
            # Füge am Ende hinzu
            self._execution_order.append(engine.name)

        logger.info(f"Engine registriert: {engine.name}")

    async def process_all(self, content: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Verarbeitet Content mit allen registrierten Engines."""
        results = {}

        for engine_name in self._execution_order:
            engine = self._engines.get(engine_name)
            if engine:
                try:
                    result = await engine.safe_process(content, context)
                    results[engine_name] = result
                except Exception as e:
                    logger.exception(f"Engine {engine_name} fehlgeschlagen: {e}")
                    results[engine_name] = None

        return results

    def get_combined_statistics(self) -> dict[str, Any]:
        """Gibt kombinierte Statistiken aller Engines zurück."""
        engine_stats = {}
        total_checks = 0
        total_violations = 0
        total_errors = 0

        for engine_name, engine in self._engines.items():
            stats = engine.get_statistics()
            engine_stats[engine_name] = stats
            total_checks += stats.get("checks_performed", 0)
            total_violations += stats.get("violations_detected", 0)
            total_errors += stats.get("errors_encountered", 0)

        return {
            "manager_name": self.name,
            "registered_engines": len(self._engines),
            "execution_order": self._execution_order,
            "total_checks": total_checks,
            "total_violations": total_violations,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / max(total_checks, 1),
            "overall_violation_rate": total_violations / max(total_checks, 1),
            "engine_statistics": engine_stats
        }
