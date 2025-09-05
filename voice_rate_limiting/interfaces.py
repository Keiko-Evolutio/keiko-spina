"""Voice Rate Limiting Interfaces für Keiko Personal Assistant.
Definiert abstrakte Interfaces für Voice-spezifisches Rate Limiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RateLimitType(Enum):
    """Typen von Rate Limits."""
    GLOBAL = "global"                    # Globale System-Limits
    USER = "user"                       # Per-User Limits
    IP = "ip"                          # Per-IP Limits
    SESSION = "session"                # Per-Session Limits
    ENDPOINT = "endpoint"              # Per-Endpoint Limits
    CONCURRENT = "concurrent"          # Concurrent Connection Limits


class VoiceOperation(Enum):
    """Voice-spezifische Operationen."""
    SPEECH_TO_TEXT = "speech_to_text"
    VOICE_SYNTHESIS = "voice_synthesis"
    REALTIME_STREAMING = "realtime_streaming"
    WEBSOCKET_CONNECTION = "websocket_connection"
    AGENT_EXECUTION = "agent_execution"
    TOOL_CALL = "tool_call"
    WORKFLOW_START = "workflow_start"
    AUDIO_UPLOAD = "audio_upload"
    TEXT_INPUT = "text_input"


class UserTier(Enum):
    """User-Tier für Quality-of-Service."""
    ANONYMOUS = "anonymous"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class RateLimitAlgorithm(Enum):
    """Rate Limiting Algorithmen."""
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class TimeWindowConfig:
    """Zeitfenster-spezifische Konfiguration."""
    limit: int                          # Anzahl erlaubter Requests
    window_seconds: int                 # Zeitfenster in Sekunden
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: int | None = None   # Burst-Limit für Token Bucket
    enabled: bool = True


@dataclass
class RateLimitConfig:
    """Rate Limit Konfiguration mit granularen Zeitfenstern."""
    # Legacy-Unterstützung (wird zu minute_window gemappt)
    limit: int                          # Anzahl erlaubter Requests
    window_seconds: int                 # Zeitfenster in Sekunden
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: int | None = None   # Burst-Limit für Token Bucket
    enabled: bool = True

    # Granulare Zeitfenster-Konfiguration
    minute_window: TimeWindowConfig | None = None    # Pro Minute
    hour_window: TimeWindowConfig | None = None      # Pro Stunde
    day_window: TimeWindowConfig | None = None       # Pro Tag

    # Voice-spezifische Konfiguration
    operation: VoiceOperation | None = None
    user_tier: UserTier | None = None

    # Adaptive Konfiguration
    adaptive: bool = False
    min_limit: int | None = None
    max_limit: int | None = None
    load_threshold: float = 0.8         # CPU/Memory Threshold für Adaptation

    def __post_init__(self):
        """Initialisiert Zeitfenster-Konfigurationen falls nicht gesetzt."""
        # Legacy-Kompatibilität: Wenn keine granularen Zeitfenster definiert sind,
        # verwende die Legacy-Werte für das entsprechende Zeitfenster
        if not any([self.minute_window, self.hour_window, self.day_window]):
            if self.window_seconds <= 60:
                self.minute_window = TimeWindowConfig(
                    limit=self.limit,
                    window_seconds=self.window_seconds,
                    algorithm=self.algorithm,
                    burst_limit=self.burst_limit,
                    enabled=self.enabled
                )
            elif self.window_seconds <= 3600:
                self.hour_window = TimeWindowConfig(
                    limit=self.limit,
                    window_seconds=self.window_seconds,
                    algorithm=self.algorithm,
                    burst_limit=self.burst_limit,
                    enabled=self.enabled
                )
            else:
                self.day_window = TimeWindowConfig(
                    limit=self.limit,
                    window_seconds=self.window_seconds,
                    algorithm=self.algorithm,
                    burst_limit=self.burst_limit,
                    enabled=self.enabled
                )

    def get_active_windows(self) -> list[TimeWindowConfig]:
        """Gibt alle aktiven Zeitfenster zurück."""
        windows = []
        if self.minute_window and self.minute_window.enabled:
            windows.append(self.minute_window)
        if self.hour_window and self.hour_window.enabled:
            windows.append(self.hour_window)
        if self.day_window and self.day_window.enabled:
            windows.append(self.day_window)
        return windows


@dataclass
class WindowResult:
    """Ergebnis für ein einzelnes Zeitfenster."""
    window_seconds: int
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after_seconds: int | None = None
    algorithm_used: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW


@dataclass
class RateLimitResult:
    """Ergebnis einer Rate Limit Prüfung mit Multi-Window-Unterstützung."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after_seconds: int | None = None

    # Multi-Window-Ergebnisse
    window_results: dict[str, WindowResult] = None  # Key: "minute", "hour", "day"

    # Zusätzliche Metadaten
    algorithm_used: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    key: str = ""
    operation: VoiceOperation | None = None

    # Headers für HTTP Response
    headers: dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = self._generate_headers()
        if self.window_results is None:
            self.window_results = {}

    def _generate_headers(self) -> dict[str, str]:
        """Generiert HTTP Headers für Rate Limiting."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp())),
        }

        if self.retry_after_seconds is not None:
            headers["Retry-After"] = str(self.retry_after_seconds)

        if self.operation:
            headers["X-RateLimit-Operation"] = self.operation.value

        return headers


@dataclass
class VoiceRateLimitContext:
    """Kontext für Voice Rate Limiting."""
    user_id: str
    session_id: str | None = None
    ip_address: str | None = None
    user_tier: UserTier = UserTier.STANDARD
    endpoint: str | None = None

    # Voice-spezifische Kontextdaten
    workflow_id: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None

    # Request-Metadaten
    request_size_bytes: int | None = None
    audio_duration_seconds: float | None = None

    # System-Kontext
    current_load: float | None = None
    peak_hours: bool = False


# =============================================================================
# RATE LIMITING INTERFACES
# =============================================================================

@runtime_checkable
class IRateLimitStore(Protocol):
    """Interface für Rate Limit Storage Backend."""

    async def get_count(self, key: str, window_seconds: int) -> int:
        """Gibt aktuelle Anzahl für Schlüssel zurück."""
        ...

    async def increment(self, key: str, window_seconds: int, amount: int = 1) -> int:
        """Erhöht Counter und gibt neue Anzahl zurück."""
        ...

    async def get_reset_time(self, key: str, window_seconds: int) -> datetime:
        """Gibt Reset-Zeit für Schlüssel zurück."""
        ...

    async def clear(self, key: str) -> None:
        """Löscht Counter für Schlüssel."""
        ...

    async def get_concurrent_count(self, key: str) -> int:
        """Gibt aktuelle Concurrent-Anzahl zurück."""
        ...

    async def increment_concurrent(self, key: str) -> int:
        """Erhöht Concurrent-Counter."""
        ...

    async def decrement_concurrent(self, key: str) -> int:
        """Verringert Concurrent-Counter."""
        ...


@runtime_checkable
class IRateLimitAlgorithm(Protocol):
    """Interface für Rate Limiting Algorithmen."""

    async def check_limit(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Prüft Rate Limit für Schlüssel."""
        ...

    async def consume(
        self,
        store: IRateLimitStore,
        key: str,
        config: RateLimitConfig,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Schlüssel."""
        ...


@runtime_checkable
class IVoiceRateLimiter(Protocol):
    """Interface für Voice-spezifisches Rate Limiting."""

    async def check_rate_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> RateLimitResult:
        """Prüft Rate Limit für Voice-Operation."""
        ...

    async def consume_rate_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext,
        amount: int = 1
    ) -> RateLimitResult:
        """Konsumiert Rate Limit für Voice-Operation."""
        ...

    async def check_concurrent_limit(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> RateLimitResult:
        """Prüft Concurrent-Limit für Voice-Operation."""
        ...

    async def acquire_concurrent_slot(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> tuple[bool, str | None]:
        """Erwirbt Concurrent-Slot. Gibt (success, slot_id) zurück."""
        ...

    async def release_concurrent_slot(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext,
        slot_id: str
    ) -> None:
        """Gibt Concurrent-Slot frei."""
        ...

    async def get_rate_limit_status(
        self,
        operation: VoiceOperation,
        context: VoiceRateLimitContext
    ) -> dict[str, Any]:
        """Gibt Rate Limit Status für Operation und Kontext zurück."""
        ...


@runtime_checkable
class IAdaptiveRateLimiter(Protocol):
    """Interface für Adaptive Rate Limiting."""

    async def adjust_limits_based_on_load(self, current_load: float) -> None:
        """Passt Limits basierend auf System-Load an."""
        ...

    async def adjust_limits_based_on_time(self, is_peak_hours: bool) -> None:
        """Passt Limits basierend auf Tageszeit an."""
        ...

    async def get_current_limits(self, operation: VoiceOperation, user_tier: UserTier) -> RateLimitConfig:
        """Gibt aktuelle Limits für Operation und User-Tier zurück."""
        ...

    def get_adaptation_status(self) -> dict[str, Any]:
        """Gibt aktuellen Adaptations-Status zurück."""
        ...

    async def start_monitoring(self) -> None:
        """Startet Adaptive Rate Limit Monitoring."""
        ...

    async def stop_monitoring(self) -> None:
        """Stoppt Adaptive Rate Limit Monitoring."""
        ...


@runtime_checkable
class IVoiceRateLimitService(Protocol):
    """Haupt-Interface für Voice Rate Limiting Service."""

    @property
    def rate_limiter(self) -> IVoiceRateLimiter:
        """Voice Rate Limiter."""
        ...

    @property
    def adaptive_limiter(self) -> IAdaptiveRateLimiter:
        """Adaptive Rate Limiter."""
        ...

    async def initialize(self) -> None:
        """Initialisiert Rate Limiting Service."""
        ...

    async def shutdown(self) -> None:
        """Fährt Rate Limiting Service herunter."""
        ...

    async def get_rate_limit_status(self, context: VoiceRateLimitContext) -> dict[str, Any]:
        """Gibt Rate Limit Status für Kontext zurück."""
        ...

    async def reset_rate_limits(self, context: VoiceRateLimitContext) -> None:
        """Setzt Rate Limits für Kontext zurück (Admin-Funktion)."""
        ...

    async def get_global_statistics(self) -> dict[str, Any]:
        """Gibt globale Rate Limiting Statistiken zurück."""
        ...

    @property
    def is_running(self) -> bool:
        """Prüft ob der Service läuft."""
        ...

    async def update_user_tier(self, user_id: str, tier: UserTier) -> None:
        """Aktualisiert User-Tier für Rate Limiting."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check für Rate Limiting Service durch."""
        ...


# =============================================================================
# MIDDLEWARE INTERFACES
# =============================================================================

@runtime_checkable
class IVoiceRateLimitMiddleware(Protocol):
    """Interface für Voice Rate Limiting Middleware."""

    async def check_http_rate_limit(self, request: Any, operation: VoiceOperation) -> RateLimitResult:
        """Prüft Rate Limit für HTTP Request."""
        ...

    async def check_websocket_rate_limit(self, websocket: Any, user_id: str) -> RateLimitResult:
        """Prüft Rate Limit für WebSocket Connection."""
        ...

    async def handle_rate_limit_exceeded(self, result: RateLimitResult) -> Any:
        """Behandelt Rate Limit Überschreitung."""
        ...


# =============================================================================
# CONFIGURATION INTERFACES
# =============================================================================

@dataclass
class VoiceRateLimitSettings:
    """Voice Rate Limiting Konfiguration mit granularen Zeitfenstern."""

    # Basis-Konfiguration
    enabled: bool = True
    redis_enabled: bool = True
    redis_url: str | None = None

    # Globale Limits
    global_requests_per_minute: int = 1000
    global_concurrent_connections: int = 100

    # Legacy User-basierte Limits (per Minute) - für Abwärtskompatibilität
    user_speech_to_text_limit: int = 60
    user_voice_synthesis_limit: int = 30
    user_realtime_streaming_limit: int = 10
    user_websocket_connections: int = 3
    user_agent_executions_limit: int = 20
    user_tool_calls_limit: int = 50

    # Granulare Zeitfenster-basierte Limits
    # Speech-to-Text Limits
    stt_requests_per_minute: int = 60
    stt_requests_per_hour: int = 1000
    stt_requests_per_day: int = 10000

    # Text-to-Speech Limits
    tts_requests_per_minute: int = 30
    tts_requests_per_hour: int = 500
    tts_requests_per_day: int = 5000

    # Realtime Streaming Limits
    streaming_requests_per_minute: int = 10
    streaming_requests_per_hour: int = 100
    streaming_requests_per_day: int = 1000

    # WebSocket Connection Limits
    websocket_connections_per_minute: int = 3
    websocket_connections_per_hour: int = 20
    websocket_connections_per_day: int = 100

    # Agent Execution Limits
    agent_executions_per_minute: int = 20
    agent_executions_per_hour: int = 200
    agent_executions_per_day: int = 2000

    # Tool Call Limits
    tool_calls_per_minute: int = 50
    tool_calls_per_hour: int = 500
    tool_calls_per_day: int = 5000

    # Session-basierte Limits
    session_workflow_starts_limit: int = 100
    session_audio_uploads_limit: int = 200
    session_text_inputs_limit: int = 500

    # IP-basierte Limits
    ip_requests_per_minute: int = 200
    ip_concurrent_connections: int = 10

    # User-Tier-spezifische Multiplier
    anonymous_multiplier: float = 0.5
    standard_multiplier: float = 1.0
    premium_multiplier: float = 2.0
    enterprise_multiplier: float = 5.0

    # Adaptive Limits
    adaptive_enabled: bool = True
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8
    peak_hours_start: int = 9  # 9 AM
    peak_hours_end: int = 17   # 5 PM
    peak_hours_multiplier: float = 0.7

    # Burst-Konfiguration
    burst_enabled: bool = True
    burst_multiplier: float = 1.5

    # Monitoring
    monitoring_enabled: bool = True
    alert_threshold: float = 0.9  # Alert bei 90% Limit-Auslastung
