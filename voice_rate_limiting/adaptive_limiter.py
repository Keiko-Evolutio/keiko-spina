"""Adaptive Rate Limiter Implementation.
Dynamische Anpassung von Rate Limits basierend auf System-Load und Zeit.
"""

import asyncio
from datetime import datetime

from kei_logging import get_logger

from .interfaces import (
    IAdaptiveRateLimiter,
    RateLimitAlgorithm,
    RateLimitConfig,
    UserTier,
    VoiceOperation,
    VoiceRateLimitSettings,
)

logger = get_logger(__name__)


class AdaptiveRateLimiter(IAdaptiveRateLimiter):
    """Adaptive Rate Limiter Implementation.
    Passt Rate Limits dynamisch basierend auf System-Load und Tageszeit an.
    """

    def __init__(self, settings: VoiceRateLimitSettings):
        self.settings = settings

        # Basis-Konfigurationen für jede Operation
        self._base_configs = self._create_base_configs()

        # Aktuelle adaptive Multiplier
        self._load_multiplier = 1.0
        self._time_multiplier = 1.0

        # System-Load-Tracking
        self._current_load = 0.0
        self._load_history = []
        self._max_load_history = 60  # 60 Datenpunkte

        # Monitoring-Task
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

        logger.info("Adaptive rate limiter initialized")

    def _create_base_configs(self) -> dict[VoiceOperation, dict[UserTier, RateLimitConfig]]:
        """Erstellt Basis-Konfigurationen für alle Operationen und User-Tiers."""
        # Dictionary literal mit allen Voice-Operationen initialisieren
        configs = {
            # Speech-to-Text Limits
            VoiceOperation.SPEECH_TO_TEXT: {
                UserTier.ANONYMOUS: RateLimitConfig(
                    limit=int(self.settings.user_speech_to_text_limit * self.settings.anonymous_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.STANDARD: RateLimitConfig(
                    limit=self.settings.user_speech_to_text_limit,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.PREMIUM: RateLimitConfig(
                    limit=int(self.settings.user_speech_to_text_limit * self.settings.premium_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_speech_to_text_limit * self.settings.premium_multiplier * 1.5),
                    adaptive=True
                ),
                UserTier.ENTERPRISE: RateLimitConfig(
                    limit=int(self.settings.user_speech_to_text_limit * self.settings.enterprise_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_speech_to_text_limit * self.settings.enterprise_multiplier * 2),
                    adaptive=True
                )
            },

            # Voice Synthesis Limits
            VoiceOperation.VOICE_SYNTHESIS: {
                UserTier.ANONYMOUS: RateLimitConfig(
                    limit=int(self.settings.user_voice_synthesis_limit * self.settings.anonymous_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.STANDARD: RateLimitConfig(
                    limit=self.settings.user_voice_synthesis_limit,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.PREMIUM: RateLimitConfig(
                    limit=int(self.settings.user_voice_synthesis_limit * self.settings.premium_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_voice_synthesis_limit * self.settings.premium_multiplier * 1.5),
                    adaptive=True
                ),
                UserTier.ENTERPRISE: RateLimitConfig(
                    limit=int(self.settings.user_voice_synthesis_limit * self.settings.enterprise_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_voice_synthesis_limit * self.settings.enterprise_multiplier * 2),
                    adaptive=True
                )
            },

            # Realtime Streaming Limits
            VoiceOperation.REALTIME_STREAMING: {
                UserTier.ANONYMOUS: RateLimitConfig(
                    limit=int(self.settings.user_realtime_streaming_limit * self.settings.anonymous_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.STANDARD: RateLimitConfig(
                    limit=self.settings.user_realtime_streaming_limit,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.PREMIUM: RateLimitConfig(
                    limit=int(self.settings.user_realtime_streaming_limit * self.settings.premium_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
                    adaptive=True
                ),
                UserTier.ENTERPRISE: RateLimitConfig(
                    limit=int(self.settings.user_realtime_streaming_limit * self.settings.enterprise_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
                    adaptive=True
                )
            },

            # WebSocket Connection Limits
            VoiceOperation.WEBSOCKET_CONNECTION: {
                UserTier.ANONYMOUS: RateLimitConfig(
                    limit=int(self.settings.user_websocket_connections * self.settings.anonymous_multiplier),
                    window_seconds=3600,  # 1 hour window for connections
                    algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                    adaptive=True
                ),
                UserTier.STANDARD: RateLimitConfig(
                    limit=self.settings.user_websocket_connections,
                    window_seconds=3600,
                    algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                    adaptive=True
                ),
                UserTier.PREMIUM: RateLimitConfig(
                    limit=int(self.settings.user_websocket_connections * self.settings.premium_multiplier),
                    window_seconds=3600,
                    algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                    adaptive=True
                ),
                UserTier.ENTERPRISE: RateLimitConfig(
                    limit=int(self.settings.user_websocket_connections * self.settings.enterprise_multiplier),
                    window_seconds=3600,
                    algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                    adaptive=True
                )
            },

            # Agent Execution Limits
            VoiceOperation.AGENT_EXECUTION: {
                UserTier.ANONYMOUS: RateLimitConfig(
                    limit=int(self.settings.user_agent_executions_limit * self.settings.anonymous_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.STANDARD: RateLimitConfig(
                    limit=self.settings.user_agent_executions_limit,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.PREMIUM: RateLimitConfig(
                    limit=int(self.settings.user_agent_executions_limit * self.settings.premium_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_agent_executions_limit * self.settings.premium_multiplier * 1.3),
                    adaptive=True
                ),
                UserTier.ENTERPRISE: RateLimitConfig(
                    limit=int(self.settings.user_agent_executions_limit * self.settings.enterprise_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_agent_executions_limit * self.settings.enterprise_multiplier * 1.5),
                    adaptive=True
                )
            },

            # Tool Call Limits
            VoiceOperation.TOOL_CALL: {
                UserTier.ANONYMOUS: RateLimitConfig(
                    limit=int(self.settings.user_tool_calls_limit * self.settings.anonymous_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.STANDARD: RateLimitConfig(
                    limit=self.settings.user_tool_calls_limit,
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                    adaptive=True
                ),
                UserTier.PREMIUM: RateLimitConfig(
                    limit=int(self.settings.user_tool_calls_limit * self.settings.premium_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_tool_calls_limit * self.settings.premium_multiplier * 1.2),
                    adaptive=True
                ),
                UserTier.ENTERPRISE: RateLimitConfig(
                    limit=int(self.settings.user_tool_calls_limit * self.settings.enterprise_multiplier),
                    window_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                    burst_limit=int(self.settings.user_tool_calls_limit * self.settings.enterprise_multiplier * 1.3),
                    adaptive=True
                )
            }
        }

        return configs

    async def start_monitoring(self) -> None:
        """Startet adaptive Monitoring-Task."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Adaptive rate limiter monitoring started")

    async def stop_monitoring(self) -> None:
        """Stoppt adaptive Monitoring-Task."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Adaptive rate limiter monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Monitoring-Schleife für adaptive Anpassungen."""
        while self._running:
            try:
                # System-Load sammeln
                await self._collect_system_load()

                # Load-basierte Anpassungen
                await self._adjust_for_load()

                # Zeit-basierte Anpassungen
                await self._adjust_for_time()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive rate limiter monitoring: {e}")
                await asyncio.sleep(5)

    async def _collect_system_load(self) -> None:
        """Sammelt aktuelle System-Load."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Kombinierte Load-Metrik
            self._current_load = max(cpu_percent / 100.0, memory_percent / 100.0)

            # Load-History aktualisieren
            self._load_history.append(self._current_load)
            if len(self._load_history) > self._max_load_history:
                self._load_history.pop(0)

        except Exception as e:
            logger.debug(f"Could not collect system load: {e}")
            self._current_load = 0.5  # Fallback

    async def _adjust_for_load(self) -> None:
        """Passt Multiplier basierend auf System-Load an."""
        if not self._load_history:
            return

        # Durchschnittliche Load der letzten Messungen
        avg_load = sum(self._load_history[-10:]) / min(10, len(self._load_history))

        if avg_load > self.settings.cpu_threshold:
            # Hohe Last - reduziere Limits
            self._load_multiplier = max(0.3, 1.0 - (avg_load - self.settings.cpu_threshold) * 2)
        elif avg_load < self.settings.cpu_threshold * 0.5:
            # Niedrige Last - erhöhe Limits leicht
            self._load_multiplier = min(1.2, 1.0 + (self.settings.cpu_threshold * 0.5 - avg_load) * 0.5)
        else:
            # Normale Last
            self._load_multiplier = 1.0

        logger.debug(f"Load-based multiplier adjusted to {self._load_multiplier:.2f} (avg_load: {avg_load:.2f})")

    async def _adjust_for_time(self) -> None:
        """Passt Multiplier basierend auf Tageszeit an."""
        current_hour = datetime.now().hour

        is_peak_hours = self.settings.peak_hours_start <= current_hour <= self.settings.peak_hours_end

        if is_peak_hours:
            self._time_multiplier = self.settings.peak_hours_multiplier
        else:
            self._time_multiplier = 1.0

        logger.debug(f"Time-based multiplier: {self._time_multiplier:.2f} (peak_hours: {is_peak_hours})")

    async def adjust_limits_based_on_load(self, current_load: float) -> None:
        """Passt Limits basierend auf System-Load an."""
        self._current_load = current_load
        await self._adjust_for_load()

    async def adjust_limits_based_on_time(self, is_peak_hours: bool) -> None:
        """Passt Limits basierend auf Tageszeit an."""
        if is_peak_hours:
            self._time_multiplier = self.settings.peak_hours_multiplier
        else:
            self._time_multiplier = 1.0

    async def get_current_limits(self, operation: VoiceOperation, user_tier: UserTier) -> RateLimitConfig:
        """Gibt aktuelle Limits für Operation und User-Tier zurück."""
        base_config = self._base_configs.get(operation, {}).get(user_tier)

        if not base_config:
            # Fallback
            return RateLimitConfig(
                limit=10,
                window_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )

        # Adaptive Multiplier anwenden
        combined_multiplier = self._load_multiplier * self._time_multiplier

        adjusted_limit = max(1, int(base_config.limit * combined_multiplier))
        burst_limit = None
        if base_config.burst_limit:
            burst_limit = max(1, int(base_config.burst_limit * combined_multiplier))

        return RateLimitConfig(
            limit=adjusted_limit,
            window_seconds=base_config.window_seconds,
            algorithm=base_config.algorithm,
            burst_limit=burst_limit,
            enabled=base_config.enabled,
            operation=operation,
            user_tier=user_tier,
            adaptive=base_config.adaptive,
            min_limit=base_config.min_limit,
            max_limit=base_config.max_limit,
            load_threshold=base_config.load_threshold
        )

    def get_adaptation_status(self) -> dict[str, float]:
        """Gibt aktuellen Adaptation-Status zurück."""
        return {
            "load_multiplier": self._load_multiplier,
            "time_multiplier": self._time_multiplier,
            "combined_multiplier": self._load_multiplier * self._time_multiplier,
            "current_load": self._current_load,
            "avg_load": sum(self._load_history[-10:]) / min(10, len(self._load_history)) if self._load_history else 0.0
        }
