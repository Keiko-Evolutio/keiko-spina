"""Voice Rate Limiter Implementation.
Spezialisiertes Rate Limiting für Voice-Operationen.
"""

import uuid
from datetime import datetime
from typing import Any

from kei_logging import get_logger

from .algorithms import create_rate_limit_algorithm
from .interfaces import (
    IRateLimitAlgorithm,
    IRateLimitStore,
    IVoiceRateLimiter,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitResult,
    TimeWindowConfig,
    UserTier,
    VoiceOperation,
    VoiceRateLimitContext,
    VoiceRateLimitSettings,
)
from .multi_window_limiter import MultiWindowConfig, MultiWindowRateLimiter

logger = get_logger(__name__)


class VoiceRateLimiter(IVoiceRateLimiter):
    """Voice-spezifischer Rate Limiter.
    Implementiert Multi-Level Rate Limiting für Voice-Operationen.
    """

    def __init__(
        self,
        store: IRateLimitStore,
        default_configs: dict[VoiceOperation, RateLimitConfig],
        user_tier_multipliers: dict[UserTier, float],
        settings: VoiceRateLimitSettings | None = None
    ):
        self.store = store
        self.default_configs = default_configs
        self.user_tier_multipliers = user_tier_multipliers
        self.settings = settings

        # Algorithm-Cache
        self._algorithms: dict[RateLimitAlgorithm, IRateLimitAlgorithm] = {}

        # Concurrent Slot Tracking
        self._concurrent_slots: dict[str, str] = {}  # slot_id -> key

        # Multi-Window Rate Limiter für granulare Zeitfenster
        self.multi_window_limiter = MultiWindowRateLimiter(
            store=store,
            default_algorithm=RateLimitAlgorithm.SLIDING_WINDOW
        )

        logger.info("Voice rate limiter mit Multi-Window-Unterstützung initialisiert")

    def _create_multi_window_config(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> MultiWindowConfig | None:
        """Erstellt Multi-Window-Konfiguration basierend auf Settings und Operation."""
        if not self.settings:
            return None

        # User-Tier-Multiplier anwenden
        multiplier = self.user_tier_multipliers.get(context.user_tier, 1.0)

        # Bestimme Limits basierend auf Operation
        minute_config = None
        hour_config = None
        day_config = None

        if operation == VoiceOperation.SPEECH_TO_TEXT:
            minute_config = TimeWindowConfig(
                limit=int(self.settings.stt_requests_per_minute * multiplier),
                window_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            hour_config = TimeWindowConfig(
                limit=int(self.settings.stt_requests_per_hour * multiplier),
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            day_config = TimeWindowConfig(
                limit=int(self.settings.stt_requests_per_day * multiplier),
                window_seconds=86400,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
        elif operation == VoiceOperation.VOICE_SYNTHESIS:
            minute_config = TimeWindowConfig(
                limit=int(self.settings.tts_requests_per_minute * multiplier),
                window_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            hour_config = TimeWindowConfig(
                limit=int(self.settings.tts_requests_per_hour * multiplier),
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            day_config = TimeWindowConfig(
                limit=int(self.settings.tts_requests_per_day * multiplier),
                window_seconds=86400,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
        elif operation == VoiceOperation.REALTIME_STREAMING:
            minute_config = TimeWindowConfig(
                limit=int(self.settings.streaming_requests_per_minute * multiplier),
                window_seconds=60,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            )
            hour_config = TimeWindowConfig(
                limit=int(self.settings.streaming_requests_per_hour * multiplier),
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            )
            day_config = TimeWindowConfig(
                limit=int(self.settings.streaming_requests_per_day * multiplier),
                window_seconds=86400,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            )
        elif operation == VoiceOperation.WEBSOCKET_CONNECTION:
            minute_config = TimeWindowConfig(
                limit=int(self.settings.websocket_connections_per_minute * multiplier),
                window_seconds=60,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW
            )
            hour_config = TimeWindowConfig(
                limit=int(self.settings.websocket_connections_per_hour * multiplier),
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW
            )
            day_config = TimeWindowConfig(
                limit=int(self.settings.websocket_connections_per_day * multiplier),
                window_seconds=86400,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW
            )
        elif operation == VoiceOperation.AGENT_EXECUTION:
            minute_config = TimeWindowConfig(
                limit=int(self.settings.agent_executions_per_minute * multiplier),
                window_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            hour_config = TimeWindowConfig(
                limit=int(self.settings.agent_executions_per_hour * multiplier),
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            day_config = TimeWindowConfig(
                limit=int(self.settings.agent_executions_per_day * multiplier),
                window_seconds=86400,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
        elif operation == VoiceOperation.TOOL_CALL:
            minute_config = TimeWindowConfig(
                limit=int(self.settings.tool_calls_per_minute * multiplier),
                window_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            hour_config = TimeWindowConfig(
                limit=int(self.settings.tool_calls_per_hour * multiplier),
                window_seconds=3600,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
            day_config = TimeWindowConfig(
                limit=int(self.settings.tool_calls_per_day * multiplier),
                window_seconds=86400,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )

        # Erstelle Multi-Window-Konfiguration nur wenn mindestens ein Zeitfenster definiert ist
        if any([minute_config, hour_config, day_config]):
            return MultiWindowConfig(
                operation=operation,
                minute_config=minute_config,
                hour_config=hour_config,
                day_config=day_config
            )

        return None

    def _get_algorithm(self, algorithm: RateLimitAlgorithm) -> IRateLimitAlgorithm:
        """Holt oder erstellt Algorithm-Instance."""
        if algorithm not in self._algorithms:
            self._algorithms[algorithm] = create_rate_limit_algorithm(algorithm)
        return self._algorithms[algorithm]

    def _get_rate_limit_config(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> RateLimitConfig:
        """Bestimmt Rate Limit Config für Operation und Kontext."""
        base_config = self.default_configs.get(operation)
        if not base_config:
            # Fallback-Config
            base_config = RateLimitConfig(
                limit=100,
                window_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )

        # User-Tier-Multiplier anwenden
        multiplier = self.user_tier_multipliers.get(context.user_tier, 1.0)

        # Adaptive Anpassungen
        if base_config.adaptive and context.current_load is not None:
            if context.current_load > base_config.load_threshold:
                # Reduziere Limits bei hoher Last
                multiplier *= 0.7

        # Peak Hours Anpassung
        if context.peak_hours:
            multiplier *= 0.8  # 20% Reduktion in Peak Hours

        # Neue Config mit angepassten Limits
        adjusted_limit = max(1, int(base_config.limit * multiplier))
        burst_limit = None
        if base_config.burst_limit:
            burst_limit = max(1, int(base_config.burst_limit * multiplier))

        return RateLimitConfig(
            limit=adjusted_limit,
            window_seconds=base_config.window_seconds,
            algorithm=base_config.algorithm,
            burst_limit=burst_limit,
            enabled=base_config.enabled,
            operation=operation,
            user_tier=context.user_tier,
            adaptive=base_config.adaptive,
            min_limit=base_config.min_limit,
            max_limit=base_config.max_limit,
            load_threshold=base_config.load_threshold
        )

    def _generate_rate_limit_keys(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> list[tuple[str, str]]:
        """Generiert Rate Limit Keys für verschiedene Scopes."""
        # Optimierte List-Creation mit festen und bedingten Einträgen
        keys = [
            # Global Rate Limit
            ("global", f"global:{operation.value}"),
            # User-basierte Rate Limits
            ("user", f"user:{context.user_id}:{operation.value}"),
        ]

        # Bedingte Einträge hinzufügen
        if context.ip_address:
            keys.append(("ip", f"ip:{context.ip_address}:{operation.value}"))
        if context.session_id:
            keys.append(("session", f"session:{context.session_id}:{operation.value}"))
        if context.endpoint:
            keys.append(("endpoint", f"endpoint:{context.endpoint}:{operation.value}"))
        if context.workflow_id:
            keys.append(("workflow", f"workflow:{context.workflow_id}:{operation.value}"))

        return keys

    async def check_rate_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> RateLimitResult:
        """Prüft Rate Limit für Voice-Operation mit Multi-Window-Unterstützung."""
        # Prüfe zuerst Multi-Window-Limits (granulare Zeitfenster)
        multi_window_config = self._create_multi_window_config(operation, context)
        multi_window_result = None  # Initialize to avoid unbound variable
        if multi_window_config:
            multi_window_result = await self.multi_window_limiter.check_multi_window_limit(
                operation, context, multi_window_config
            )

            if not multi_window_result.allowed:
                logger.warning(
                    f"Multi-window rate limit exceeded for {operation.value} "
                    f"(user: {context.user_id})"
                )
                return multi_window_result

        # Fallback auf Legacy-Rate-Limiting für Abwärtskompatibilität
        config = self._get_rate_limit_config(operation, context)

        if not config.enabled:
            # Rate Limiting deaktiviert
            return RateLimitResult(
                allowed=True,
                limit=config.limit,
                remaining=config.limit,
                reset_time=datetime.now(),
                algorithm_used=config.algorithm,
                operation=operation
            )

        algorithm = self._get_algorithm(config.algorithm)
        keys = self._generate_rate_limit_keys(operation, context)

        # Prüfe alle Rate Limit Scopes
        for scope, key in keys:
            result = await algorithm.check_limit(self.store, key, config)
            result.operation = operation

            if not result.allowed:
                # Rate Limit überschritten
                logger.warning(
                    f"Legacy rate limit exceeded for {operation.value} "
                    f"(scope: {scope}, user: {context.user_id})"
                )
                return result

        # Alle Checks bestanden - verwende Multi-Window-Ergebnis falls verfügbar
        if multi_window_config and multi_window_result:
            return multi_window_result

        # Fallback auf Legacy-Ergebnis
        return RateLimitResult(
            allowed=True,
            limit=config.limit,
            remaining=config.limit,  # Approximation
            reset_time=datetime.now(),
            algorithm_used=config.algorithm,
            operation=operation
        )

    async def consume_rate_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Voice-Operation mit Multi-Window-Unterstützung."""
        # Konsumiere zuerst Multi-Window-Limits (granulare Zeitfenster)
        multi_window_config = self._create_multi_window_config(operation, context)
        multi_window_result = None  # Initialize to avoid unbound variable
        if multi_window_config:
            multi_window_result = await self.multi_window_limiter.consume_multi_window_limit(
                operation, context, multi_window_config, amount
            )

            if not multi_window_result.allowed:
                logger.warning(
                    f"Multi-window rate limit exceeded for {operation.value} "
                    f"(user: {context.user_id}, amount: {amount})"
                )
                return multi_window_result

        # Fallback auf Legacy-Rate-Limiting für Abwärtskompatibilität
        config = self._get_rate_limit_config(operation, context)

        if not config.enabled:
            # Rate Limiting deaktiviert
            return RateLimitResult(
                allowed=True,
                limit=config.limit,
                remaining=config.limit,
                reset_time=datetime.now(),
                algorithm_used=config.algorithm,
                operation=operation
            )

        algorithm = self._get_algorithm(config.algorithm)
        keys = self._generate_rate_limit_keys(operation, context)

        # Konsumiere für alle Rate Limit Scopes
        results = []
        for scope, key in keys:
            result = await algorithm.consume(self.store, key, config, amount)
            result.operation = operation
            results.append((scope, result))

            if not result.allowed:
                # Rate Limit überschritten - rollback vorherige Konsumierungen
                await self._rollback_consumption(keys[:len(results)-1], algorithm, config, amount)

                logger.warning(
                    f"Legacy rate limit exceeded for {operation.value} "
                    f"(scope: {scope}, user: {context.user_id}, amount: {amount})"
                )
                return result

        # Alle Konsumierungen erfolgreich - verwende Multi-Window-Ergebnis falls verfügbar
        if multi_window_config and multi_window_result:
            return multi_window_result

        # Fallback auf Legacy-Ergebnis
        # Gib das restriktivste Ergebnis zurück
        most_restrictive = min(results, key=lambda x: x[1].remaining)
        return most_restrictive[1]

    async def _rollback_consumption(
        self,
        consumed_keys: list[tuple[str, str]],
        algorithm: IRateLimitAlgorithm,
        config: RateLimitConfig,
        amount: int
    ) -> None:
        """Rollback von Rate Limit Konsumierungen."""
        # Vereinfachte Rollback-Implementation
        # In einer vollständigen Implementation würde man die Konsumierung rückgängig machen
        logger.debug(f"Rolling back {len(consumed_keys)} rate limit consumptions")

        # Für Sliding Window ist Rollback schwierig, da Timestamps bereits gesetzt sind
        # Für Token Bucket könnte man Tokens zurückgeben
        # Hier implementieren wir einen vereinfachten Ansatz

    async def check_concurrent_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> RateLimitResult:
        """Prüft Concurrent-Limit für Voice-Operation."""
        config = self._get_rate_limit_config(operation, context)

        # Concurrent Limits verwenden das Limit als maximale Anzahl gleichzeitiger Verbindungen
        concurrent_limit = config.limit

        # Generiere Concurrent-Keys
        keys = self._generate_rate_limit_keys(operation, context)

        for scope, key in keys:
            concurrent_key = f"concurrent:{key}"
            current_count = await self.store.get_concurrent_count(concurrent_key)

            if current_count >= concurrent_limit:
                return RateLimitResult(
                    allowed=False,
                    limit=concurrent_limit,
                    remaining=0,
                    reset_time=datetime.now(),
                    retry_after_seconds=60,  # Retry after 1 minute
                    algorithm_used=config.algorithm,
                    key=concurrent_key,
                    operation=operation
                )

        return RateLimitResult(
            allowed=True,
            limit=concurrent_limit,
            remaining=concurrent_limit,  # Approximation
            reset_time=datetime.now(),
            algorithm_used=config.algorithm,
            operation=operation
        )

    async def acquire_concurrent_slot(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> tuple[bool, str | None]:
        """Erwirbt Concurrent-Slot. Gibt (success, slot_id) zurück."""
        # Erst prüfen
        result = await self.check_concurrent_limit(operation, context)
        if not result.allowed:
            return False, None

        # Slot erwerben
        slot_id = str(uuid.uuid4())
        keys = self._generate_rate_limit_keys(operation, context)

        try:
            for scope, key in keys:
                concurrent_key = f"concurrent:{key}"
                await self.store.increment_concurrent(concurrent_key)

                # Slot-Tracking für Cleanup
                self._concurrent_slots[slot_id] = concurrent_key

            logger.debug(f"Acquired concurrent slot {slot_id} for {operation.value}")
            return True, slot_id

        except Exception as e:
            # Rollback bei Fehler
            await self._release_concurrent_slots_by_id(slot_id)
            logger.error(f"Failed to acquire concurrent slot: {e}")
            return False, None

    async def release_concurrent_slot(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext,
        slot_id: str
    ) -> None:
        """Gibt Concurrent-Slot frei."""
        await self._release_concurrent_slots_by_id(slot_id)
        logger.debug(f"Released concurrent slot {slot_id} for {operation.value}")

    async def _release_concurrent_slots_by_id(self, slot_id: str) -> None:
        """Gibt alle Concurrent-Slots für Slot-ID frei."""
        if slot_id in self._concurrent_slots:
            concurrent_key = self._concurrent_slots[slot_id]
            try:
                await self.store.decrement_concurrent(concurrent_key)
            except Exception as e:
                logger.error(f"Failed to release concurrent slot {slot_id}: {e}")
            finally:
                del self._concurrent_slots[slot_id]

    async def get_rate_limit_status(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> dict[str, Any]:
        """Gibt Rate Limit Status für Operation zurück."""
        config = self._get_rate_limit_config(operation, context)
        keys = self._generate_rate_limit_keys(operation, context)

        status = {
            "operation": operation.value,
            "user_id": context.user_id,
            "user_tier": context.user_tier.value,
            "config": {
                "limit": config.limit,
                "window_seconds": config.window_seconds,
                "algorithm": config.algorithm.value,
                "enabled": config.enabled
            },
            "scopes": {}
        }

        for scope, key in keys:
            try:
                current_count = await self.store.get_count(key, config.window_seconds)
                reset_time = await self.store.get_reset_time(key, config.window_seconds)
                concurrent_count = await self.store.get_concurrent_count(f"concurrent:{key}")

                status["scopes"][scope] = {
                    "current_count": current_count,
                    "remaining": max(0, config.limit - current_count),
                    "reset_time": reset_time.isoformat(),
                    "concurrent_count": concurrent_count
                }
            except Exception as e:
                status["scopes"][scope] = {"error": str(e)}

        return status
