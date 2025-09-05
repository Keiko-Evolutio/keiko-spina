"""Multi-Window Rate Limiter für granulare Zeitfenster-basierte Limits.
Implementiert präzise Rate Limiting mit mehreren Zeitfenstern (Minute/Stunde/Tag).
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime

from kei_logging import get_logger

from .algorithms import (
    FixedWindowAlgorithm,
    LeakyBucketAlgorithm,
    SlidingWindowAlgorithm,
    TokenBucketAlgorithm,
)
from .interfaces import (
    IRateLimitAlgorithm,
    IRateLimitStore,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitResult,
    TimeWindowConfig,
    VoiceOperation,
    VoiceRateLimitContext,
    WindowResult,
)

logger = get_logger(__name__)


@dataclass
class MultiWindowConfig:
    """Konfiguration für Multi-Window Rate Limiting."""
    operation: VoiceOperation
    minute_config: TimeWindowConfig | None = None
    hour_config: TimeWindowConfig | None = None
    day_config: TimeWindowConfig | None = None

    def get_active_configs(self) -> list[tuple[str, TimeWindowConfig]]:
        """Gibt alle aktiven Zeitfenster-Konfigurationen zurück."""
        configs = []
        if self.minute_config and self.minute_config.enabled:
            configs.append(("minute", self.minute_config))
        if self.hour_config and self.hour_config.enabled:
            configs.append(("hour", self.hour_config))
        if self.day_config and self.day_config.enabled:
            configs.append(("day", self.day_config))
        return configs


class MultiWindowRateLimiter:
    """Multi-Window Rate Limiter für granulare Zeitfenster-basierte Limits.

    Unterstützt gleichzeitige Rate Limiting-Prüfungen für:
    - Minütliche Limits (X Requests pro Minute)
    - Stündliche Limits (X Requests pro Stunde)
    - Tägliche Limits (X Requests pro Tag)

    Features:
    - Sliding Window oder Token Bucket Algorithmus
    - Separate Zähler für verschiedene Zeitfenster
    - Graceful Degradation bei Limit-Überschreitung
    - Redis-Backend Unterstützung für verteilte Systeme
    - Performance-optimiert für hohe Durchsatzraten
    """

    def __init__(
        self,
        store: IRateLimitStore,
        default_algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    ):
        self.store = store
        self.default_algorithm = default_algorithm
        self.algorithms: dict[RateLimitAlgorithm, IRateLimitAlgorithm] = {
            RateLimitAlgorithm.SLIDING_WINDOW: SlidingWindowAlgorithm(),
            RateLimitAlgorithm.TOKEN_BUCKET: TokenBucketAlgorithm(),
            RateLimitAlgorithm.FIXED_WINDOW: FixedWindowAlgorithm(),
            RateLimitAlgorithm.LEAKY_BUCKET: LeakyBucketAlgorithm()
        }
        self._lock = asyncio.Lock()

        logger.info(f"Multi-Window Rate Limiter initialisiert mit {default_algorithm.value} Algorithmus")

    def _generate_window_key(
        self,
        base_key: str,
        window_name: str,
        operation: VoiceOperation
    ) -> str:
        """Generiert Schlüssel für spezifisches Zeitfenster."""
        return f"{base_key}:{operation.value}:{window_name}"

    async def check_multi_window_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext,
        config: MultiWindowConfig
    ) -> RateLimitResult:
        """Prüft Rate Limits für alle konfigurierten Zeitfenster.

        Returns:
            RateLimitResult: Aggregiertes Ergebnis aller Zeitfenster-Prüfungen
        """
        async with self._lock:
            base_key = f"user:{context.user_id}"
            window_results = {}
            overall_allowed = True
            earliest_reset_time = datetime.now()
            min_retry_after = None

            # Prüfe alle aktiven Zeitfenster
            active_configs = config.get_active_configs()

            if not active_configs:
                # Keine aktiven Zeitfenster - erlaube Request
                return RateLimitResult(
                    allowed=True,
                    limit=0,
                    remaining=0,
                    reset_time=datetime.now(),
                    window_results={},
                    operation=operation
                )

            for window_name, window_config in active_configs:
                window_key = self._generate_window_key(base_key, window_name, operation)

                # Wähle Algorithmus für dieses Zeitfenster
                algorithm = self.algorithms.get(
                    window_config.algorithm,
                    self.algorithms[self.default_algorithm]
                )

                # Erstelle RateLimitConfig für dieses Zeitfenster
                rate_config = RateLimitConfig(
                    limit=window_config.limit,
                    window_seconds=window_config.window_seconds,
                    algorithm=window_config.algorithm,
                    burst_limit=window_config.burst_limit,
                    enabled=window_config.enabled,
                    operation=operation
                )

                # Prüfe Rate Limit für dieses Zeitfenster
                try:
                    result = await algorithm.check_limit(self.store, window_key, rate_config)

                    # Speichere Zeitfenster-spezifisches Ergebnis
                    window_results[window_name] = WindowResult(
                        window_seconds=window_config.window_seconds,
                        allowed=result.allowed,
                        limit=result.limit,
                        remaining=result.remaining,
                        reset_time=result.reset_time,
                        retry_after_seconds=result.retry_after_seconds,
                        algorithm_used=window_config.algorithm
                    )

                    # Aktualisiere Gesamt-Ergebnis
                    if not result.allowed:
                        overall_allowed = False

                    # Finde früheste Reset-Zeit
                    earliest_reset_time = max(earliest_reset_time, result.reset_time)

                    # Finde minimale Retry-After-Zeit
                    if result.retry_after_seconds is not None:
                        if min_retry_after is None or result.retry_after_seconds < min_retry_after:
                            min_retry_after = result.retry_after_seconds

                    logger.debug(
                        f"Window {window_name} check: allowed={result.allowed}, "
                        f"remaining={result.remaining}/{result.limit}, "
                        f"reset_time={result.reset_time}"
                    )

                except Exception as e:
                    logger.error(f"Fehler bei Rate Limit Prüfung für Zeitfenster {window_name}: {e}")
                    # Bei Fehlern: Erlaube Request (Fail-Open-Strategie)
                    window_results[window_name] = WindowResult(
                        window_seconds=window_config.window_seconds,
                        allowed=True,
                        limit=window_config.limit,
                        remaining=window_config.limit,
                        reset_time=datetime.now(),
                        algorithm_used=window_config.algorithm
                    )

            # Bestimme Gesamt-Limit und Remaining (verwende restriktivste Werte)
            total_limit = min(wr.limit for wr in window_results.values()) if window_results else 0
            total_remaining = min(wr.remaining for wr in window_results.values()) if window_results else 0

            # Erstelle aggregiertes Ergebnis
            result = RateLimitResult(
                allowed=overall_allowed,
                limit=total_limit,
                remaining=total_remaining,
                reset_time=earliest_reset_time,
                retry_after_seconds=min_retry_after,
                window_results=window_results,
                algorithm_used=self.default_algorithm,
                key=base_key,
                operation=operation
            )

            logger.debug(
                f"Multi-Window Rate Limit Check für {operation.value}: "
                f"allowed={overall_allowed}, windows={len(window_results)}"
            )

            return result

    async def consume_multi_window_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext,
        config: MultiWindowConfig,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limits für alle konfigurierten Zeitfenster.

        Args:
            operation: Voice-Operation
            context: Rate Limiting-Kontext
            config: Multi-Window-Konfiguration
            amount: Anzahl zu konsumierender Einheiten

        Returns:
            RateLimitResult: Aggregiertes Ergebnis aller Zeitfenster-Konsumierungen
        """
        async with self._lock:
            base_key = f"user:{context.user_id}"
            window_results = {}
            overall_allowed = True
            earliest_reset_time = datetime.now()
            min_retry_after = None

            # Prüfe alle aktiven Zeitfenster
            active_configs = config.get_active_configs()

            if not active_configs:
                # Keine aktiven Zeitfenster - erlaube Request
                return RateLimitResult(
                    allowed=True,
                    limit=0,
                    remaining=0,
                    reset_time=datetime.now(),
                    window_results={},
                    operation=operation
                )

            # Erste Phase: Prüfe alle Zeitfenster ohne zu konsumieren
            can_consume_all = True
            for window_name, window_config in active_configs:
                window_key = self._generate_window_key(base_key, window_name, operation)

                algorithm = self.algorithms.get(
                    window_config.algorithm,
                    self.algorithms[self.default_algorithm]
                )

                rate_config = RateLimitConfig(
                    limit=window_config.limit,
                    window_seconds=window_config.window_seconds,
                    algorithm=window_config.algorithm,
                    burst_limit=window_config.burst_limit,
                    enabled=window_config.enabled,
                    operation=operation
                )

                # Prüfe ohne zu konsumieren
                check_result = await algorithm.check_limit(self.store, window_key, rate_config)
                if not check_result.allowed or check_result.remaining < amount:
                    can_consume_all = False
                    break

            # Zweite Phase: Konsumiere nur wenn alle Zeitfenster erlauben
            if can_consume_all:
                for window_name, window_config in active_configs:
                    window_key = self._generate_window_key(base_key, window_name, operation)

                    algorithm = self.algorithms.get(
                        window_config.algorithm,
                        self.algorithms[self.default_algorithm]
                    )

                    rate_config = RateLimitConfig(
                        limit=window_config.limit,
                        window_seconds=window_config.window_seconds,
                        algorithm=window_config.algorithm,
                        burst_limit=window_config.burst_limit,
                        enabled=window_config.enabled,
                        operation=operation
                    )

                    try:
                        result = await algorithm.consume(self.store, window_key, rate_config, amount)

                        window_results[window_name] = WindowResult(
                            window_seconds=window_config.window_seconds,
                            allowed=result.allowed,
                            limit=result.limit,
                            remaining=result.remaining,
                            reset_time=result.reset_time,
                            retry_after_seconds=result.retry_after_seconds,
                            algorithm_used=window_config.algorithm
                        )

                        if not result.allowed:
                            overall_allowed = False

                        earliest_reset_time = max(earliest_reset_time, result.reset_time)

                        if result.retry_after_seconds is not None:
                            if min_retry_after is None or result.retry_after_seconds < min_retry_after:
                                min_retry_after = result.retry_after_seconds

                    except Exception as e:
                        logger.error(f"Fehler bei Rate Limit Konsumierung für Zeitfenster {window_name}: {e}")
                        overall_allowed = False
            else:
                # Mindestens ein Zeitfenster würde überschritten - konsumiere nichts
                overall_allowed = False

                # Sammle aktuelle Status für alle Zeitfenster
                for window_name, window_config in active_configs:
                    window_key = self._generate_window_key(base_key, window_name, operation)

                    algorithm = self.algorithms.get(
                        window_config.algorithm,
                        self.algorithms[self.default_algorithm]
                    )

                    rate_config = RateLimitConfig(
                        limit=window_config.limit,
                        window_seconds=window_config.window_seconds,
                        algorithm=window_config.algorithm,
                        burst_limit=window_config.burst_limit,
                        enabled=window_config.enabled,
                        operation=operation
                    )

                    check_result = await algorithm.check_limit(self.store, window_key, rate_config)

                    window_results[window_name] = WindowResult(
                        window_seconds=window_config.window_seconds,
                        allowed=check_result.allowed,
                        limit=check_result.limit,
                        remaining=check_result.remaining,
                        reset_time=check_result.reset_time,
                        retry_after_seconds=check_result.retry_after_seconds,
                        algorithm_used=window_config.algorithm
                    )

            # Bestimme Gesamt-Werte
            total_limit = min(wr.limit for wr in window_results.values()) if window_results else 0
            total_remaining = min(wr.remaining for wr in window_results.values()) if window_results else 0

            result = RateLimitResult(
                allowed=overall_allowed,
                limit=total_limit,
                remaining=total_remaining,
                reset_time=earliest_reset_time,
                retry_after_seconds=min_retry_after,
                window_results=window_results,
                algorithm_used=self.default_algorithm,
                key=base_key,
                operation=operation
            )

            logger.debug(
                f"Multi-Window Rate Limit Consume für {operation.value}: "
                f"allowed={overall_allowed}, consumed={amount if overall_allowed else 0}"
            )

            return result
