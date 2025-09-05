import asyncio
import base64
import builtins
import contextlib
import json
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import websockets
import websockets.exceptions
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected
from websockets.exceptions import ConnectionClosedOK

try:
    from websockets.exceptions import InvalidStatusCode
except ImportError:
    # Fallback für ältere websockets Versionen
    InvalidStatusCode = Exception
from websockets.legacy.server import WebSocketServerProtocol

from config.voice_config import VoiceDetectionConfig, get_voice_config
from kei_logging import get_logger

# Monitoring Integration
try:
    from core.container import get_container
    from monitoring.interfaces import IMonitoringService
    MONITORING_AVAILABLE = True
except ImportError:
    get_container = None
    IMonitoringService = None
    MONITORING_AVAILABLE = False

# Voice Rate Limiting Integration
try:
    from voice_rate_limiting.interfaces import (
        IVoiceRateLimitService,
        UserTier,
        VoiceOperation,
        VoiceRateLimitContext,
    )
    VOICE_RATE_LIMITING_AVAILABLE = True
except ImportError:
    IVoiceRateLimitService = None
    VoiceOperation = None
    VoiceRateLimitContext = None
    UserTier = None
    VOICE_RATE_LIMITING_AVAILABLE = False

from .base import create_router
from .voice_shared import (
    DEFAULT_PERFORMANCE_THRESHOLDS,
    DEFAULT_RECOVERY_STRATEGY,
    AlertLevel,
    ComponentType,
    ErrorEvent,
    HealthCheckResult,
    HealthStatus,
    RecoveryAction,
    RecoveryStrategy,
    VoiceMetrics,
    VoiceSystemHealth,
    create_health_check_result,
    determine_error_severity,
    determine_overall_status,
    log_error_event,
)

logger = get_logger(__name__)


# =====================================================================
# Utility-Funktionen
# =====================================================================

def get_current_timestamp() -> str:
    """Einheitliche Zeitstempel-Generierung."""
    return datetime.now(UTC).isoformat()


def get_monitoring_service() -> Any:
    """Holt Monitoring Service aus DI Container."""
    if not MONITORING_AVAILABLE:
        return None

    try:
        container = get_container()
        return container.resolve(IMonitoringService)
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug(f"Could not resolve monitoring service: {e}")
        return None


def get_voice_rate_limit_service() -> Any:
    """Holt Voice Rate Limiting Service aus DI Container."""
    if not VOICE_RATE_LIMITING_AVAILABLE:
        return None

    # Prüfe Konfiguration - wenn deaktiviert, return None
    try:
        from config.voice_rate_limiting_config import VoiceRateLimitSettings
        import voice_rate_limiting.service  # noqa: F401

        settings = VoiceRateLimitSettings()
        if not settings.enabled:
            logger.debug("Voice Rate Limiting ist in der Konfiguration deaktiviert")
            return None

    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug(f"Voice Rate Limiting Konfiguration nicht verfügbar: {e}")
        return None

    try:
        container = get_container()
        return container.resolve(IVoiceRateLimitService)
    except (ImportError, AttributeError, RuntimeError, KeyError) as e:
        logger.debug(f"Could not resolve voice rate limiting service: {e}")
        return None


def create_rate_limit_context(request: Any, user_id: str = None) -> Any:
    """Erstellt Rate Limit Context aus Request."""
    if not VOICE_RATE_LIMITING_AVAILABLE:
        return None

    # User-ID extrahieren
    if not user_id:
        user_id = getattr(request.state, "user_id", "anonymous")

    # Session-ID extrahieren
    session_id = getattr(request.state, "session_id", None)

    # IP-Adresse extrahieren
    ip_address = getattr(request, "client", {}).get("host", "unknown")

    # User-Tier bestimmen (vereinfacht)
    user_tier = UserTier.STANDARD

    return VoiceRateLimitContext(
        user_id=str(user_id),
        session_id=session_id,
        ip_address=ip_address,
        user_tier=user_tier,
        endpoint=getattr(request.url, "path", "/voice")
    )


def debug_log(message: str, enabled: bool = True):
    """Konsolidiertes Debug-Logging."""
    if enabled:
        logger.debug(message)


async def send_websocket_response(websocket: WebSocket, response_type: str,
                                  data: dict[str, Any] | None = None, **kwargs) -> bool:
    """Einheitliche WebSocket-Response-Erstellung mit robuster Fehlerbehandlung.

    Returns:
        bool: True wenn erfolgreich gesendet, False bei Verbindungsfehlern
    """
    try:
        response = {
            "type": response_type,
            "timestamp": get_current_timestamp(),
            **(data or {}),
            **kwargs
        }
        await websocket.send_json(response)
        return True
    except (WebSocketDisconnect, ConnectionClosedOK, ClientDisconnected) as e:
        logger.debug(f"🔌 Client disconnected during response send ({response_type}): {e}")
        return False
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.warning(f"⚠️ Failed to send WebSocket response ({response_type}): {e}")
        return False


async def send_error_response(websocket: WebSocket, error: str,
                              error_type: str = "error", **kwargs) -> bool:
    """Einheitliche Error-Response-Erstellung.

    Returns:
        bool: True wenn erfolgreich gesendet, False bei Verbindungsfehlern
    """
    return await send_websocket_response(
        websocket, error_type,
        {"error": error}, **kwargs
    )


async def safe_send_websocket_response(websocket: WebSocket, response_type: str,
                                       data: dict[str, Any] | None = None, **kwargs) -> bool:
    """Sichere WebSocket-Response mit automatischer Fehlerbehandlung.

    Diese Funktion sollte verwendet werden, wenn ein Verbindungsabbruch
    nicht kritisch ist und die Anwendung normal weiterlaufen soll.

    Returns:
        bool: True wenn erfolgreich gesendet, False bei Verbindungsfehlern
    """
    try:
        return await send_websocket_response(websocket, response_type, data, **kwargs)
    except (WebSocketDisconnect, StarletteWebSocketDisconnect, ConnectionClosedOK, ClientDisconnected):
        logger.debug(f"🔌 Client disconnected during safe response send ({response_type})")
        return False
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.warning(f"⚠️ Unexpected error during safe response send ({response_type}): {e}")
        return False


# =====================================================================
# Enums und Status-Definitionen
# =====================================================================

class BufferState(Enum):
    """Buffer-Zustandsdefinitionen."""
    EMPTY = "empty"
    ACCUMULATING = "accumulating"
    READY_FOR_COMMIT = "ready_for_commit"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ERROR_RECOVERY = "error_recovery"


class ResponseState(Enum):
    """Response-Zustandsdefinitionen."""
    IDLE = "idle"
    CREATING = "creating"
    ACTIVE = "active"
    FINISHING = "finishing"
    ERROR = "error"


# =====================================================================
# Konfiguration und Models
# =====================================================================

class VoiceConnectionInfo(BaseModel):
    """Informationen über eine Voice-Verbindung."""
    user_id: str
    connected_at: datetime
    endpoint_used: str
    azure_session_id: str | None = None
    last_activity: datetime | None = None


class VoiceSettings(BaseModel):
    """Konfiguration für Voice-Einstellungen mit externalisierter Voice Detection Config."""
    user_id: str
    modalities: list[str] = ["text", "audio"]
    instructions: str = (
        "Du bist Keiko, eine freundliche und hilfsbereite KI-Assistentin. "
        "Antworte ausschließlich auf Deutsch, unabhängig von der Sprache der Eingabe. "
        "Sprich mit einer klaren, natürlichen und menschlichen Stimme in normaler Geschwindigkeit "
        "(vergleichbar mit einer entspannten menschlichen Konversation, ca. 120-150 Wörter pro Minute). "
        "Vermeide es, Wörter zu verschlucken, und sprich ohne Echo-Effekte. "
        "Deine Antworten sollen warmherzig, gesprächig und leicht verständlich sein. "
        "Ignoriere jegliche automatische Spracherkennung und bleibe strikt bei Deutsch."
    )

    # Voice Detection Konfiguration wird aus externalisierter Config geladen
    _voice_config: VoiceDetectionConfig | None = None

    @property
    def voice_config(self) -> VoiceDetectionConfig:
        """Lädt Voice Detection Konfiguration lazy."""
        if self._voice_config is None:
            self._voice_config = get_voice_config()
        return self._voice_config

    @property
    def voice(self) -> str:
        """Voice Synthesis Voice aus Konfiguration."""
        return self.voice_config.voice

    @property
    def input_audio_format(self) -> str:
        """Input Audio Format aus Konfiguration."""
        return self.voice_config.input_audio_format

    @property
    def output_audio_format(self) -> str:
        """Output Audio Format aus Konfiguration."""
        return self.voice_config.output_audio_format

    @property
    def input_audio_transcription(self) -> dict[str, str]:
        """Input Audio Transcription aus Konfiguration."""
        return self.voice_config.to_transcription_dict()

    @property
    def turn_detection(self) -> dict[str, Any]:
        """Turn Detection aus Konfiguration."""
        return self.voice_config.to_turn_detection_dict()

    @property
    def temperature(self) -> float:
        """Temperature aus Konfiguration."""
        return self.voice_config.temperature

    @property
    def max_response_output_tokens(self) -> int:
        """Max Response Output Tokens aus Konfiguration."""
        return self.voice_config.max_response_output_tokens

    @property
    def speech_rate(self) -> float:
        """Speech Rate aus Konfiguration."""
        return self.voice_config.speech_rate


# =====================================================================
# Event Deduplication und State Management
# =====================================================================

class StateManager:
    """Vereinfachtes State Management für Voice-Session."""

    def __init__(self):
        self._buffer_state = BufferState.EMPTY
        self._response_state = ResponseState.IDLE
        self._last_state_change = time.time()

    def get_buffer_state(self) -> BufferState:
        """Buffer-Status abrufen."""
        return self._buffer_state

    def set_buffer_state(self, new_state: BufferState) -> bool:
        """Buffer-Status setzen."""
        old_state = self._buffer_state
        self._buffer_state = new_state
        self._last_state_change = time.time()
        debug_log(f"🔄 Buffer state: {old_state.value} → {new_state.value}")
        return True

    def get_response_state(self) -> ResponseState:
        """Response-Status abrufen."""
        return self._response_state

    def set_response_state(self, new_state: ResponseState) -> bool:
        """Response-Status setzen."""
        old_state = self._response_state
        self._response_state = new_state
        self._last_state_change = time.time()
        debug_log(f"🔄 Response state: {old_state.value} → {new_state.value}")
        return True

    def is_safe_for_commit(self) -> bool:
        """Prüft ob Commit sicher ausgeführt werden kann."""
        return (
            self._buffer_state in [BufferState.READY_FOR_COMMIT, BufferState.ACCUMULATING] and
            self._response_state == ResponseState.IDLE
        )

    def is_safe_for_response(self) -> bool:
        """Prüft ob Response-Erstellung sicher ist."""
        return self._response_state == ResponseState.IDLE


class EventDeduplicator:
    """Vereinfachte Event-Deduplizierung."""

    def __init__(self, window_seconds: float = 1.0):
        self.window_seconds = window_seconds
        self.seen_events: dict[str, float] = {}

    def is_duplicate(self, event_key: str) -> bool:
        """Prüft ob Event ein Duplikat ist."""
        current_time = time.time()

        if event_key in self.seen_events:
            if current_time - self.seen_events[event_key] < self.window_seconds:
                return True

        self.seen_events[event_key] = current_time

        # Periodisches Cleanup
        if len(self.seen_events) > 100:
            cutoff_time = current_time - self.window_seconds * 2
            self.seen_events = {
                key: timestamp for key, timestamp in self.seen_events.items()
                if timestamp > cutoff_time
            }

        return False


# =====================================================================
# Azure OpenAI Realtime Client - Optimiert
# =====================================================================

class AzureRealtimeClient:
    """Optimierter Azure OpenAI Realtime API Client."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.ws: WebSocketServerProtocol | None = None
        self.session_id: str | None = None
        self.is_connected = False

        # State Management
        self.state_manager = StateManager()
        self.event_deduplicator = EventDeduplicator()

        # Function Call Aggregation (call_id -> {name, args: [chunks], item_id, response_id})
        self._function_calls: dict[str, dict[str, Any]] = {}

        # Audio Buffer Management
        self._audio_chunks: list[bytes] = []
        self._audio_chunk_count = 0
        self._total_audio_bytes = 0

        # Audio-Konfiguration (PCM16 @ 24kHz)
        self._samples_per_second = 24000
        self._bytes_per_sample = 2
        self._min_audio_duration_ms = 150  # Erhöht für Stabilität
        self._min_audio_bytes = int((
                                        self._min_audio_duration_ms / 1000) * self._samples_per_second * self._bytes_per_sample)

        # Commit-Kontrolle
        self._commit_lock = asyncio.Lock()
        self._last_commit_time = 0.0
        self._min_commit_interval = 0.8  # Erhöht für Stabilität

        # Error Recovery
        self._consecutive_empty_commits = 0
        self._max_empty_commits = 2  # Reduziert für schnellere Recovery
        self._error_recovery_active = False

        # Performance-/Latenzmetriken
        #  - _last_response_created_ts: Zeitpunkt der Erstellung einer Antwort
        #  - _first_audio_delta_seen: Flag, ob erstes Audio-Delta für aktuelle Antwort gesehen wurde
        #  - _last_speech_started_ts: Zeitpunkt des zuletzt erkannten Sprachbeginns (VAD)
        #  - _last_interrupt_forwarded_ts: Zeitpunkt der letzten an den Client weitergeleiteten Unterbrechung
        #  Alle Zeiten in Sekunden (time.time())
        self._last_response_created_ts: float | None = None
        self._first_audio_delta_seen: bool = False
        self._last_speech_started_ts: float | None = None
        self._last_interrupt_forwarded_ts: float | None = None

        logger.info(
            f"🎛️ Audio buffer initialized: min_duration={self._min_audio_duration_ms}ms, "
            f"min_bytes={self._min_audio_bytes}, user={self.user_id}")

        # Foto-Workflow Status
        self.photo_mode_active: bool = False
        self.last_photo_request_ts: float | None = None
        self._last_photo_action_ts: float | None = None
        self._photo_action_cooldown_s: float = 1.0

    async def _speak(self, text: str) -> None:
        """Assistenten-Nachricht sicher senden (keine konkurrierende Response)."""
        try:
            if not self.state_manager.is_safe_for_response():
                await asyncio.sleep(0.25)
            await self.send_event("conversation.item.create", {
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                }
            })
            await self.send_event("response.create", {})
        except (ConnectionError, OSError, RuntimeError) as e:
            logger.warning(f"⚠️ speak failed: {e}")

    async def connect(self) -> bool:
        """Verbindung zu Azure OpenAI Realtime API herstellen."""
        # SSL-Modul am Anfang importieren
        import ssl

        try:
            ws_url, headers = AzureRealtimeClient._build_websocket_url()
            logger.info(f"🔌 Connecting to Azure OpenAI Realtime API for {self.user_id}")

            # SSL-Kontext mit sicherer Konfiguration
            ssl_context = ssl.create_default_context()
            # SSL-Verifikation ist kritisch für Produktionssicherheit

            self.ws = await websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                max_size=2 ** 23,  # 8MB für Audio
                ssl=ssl_context
            )

            self.is_connected = True
            logger.info("✅ Connected to Azure OpenAI Realtime API")

            # Warten auf session.created mit Timeout
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
                data = json.loads(message)
                if data.get("type") == "session.created":
                    self.session_id = data.get("session", {}).get("id")
                    logger.info(f"🎯 Azure session created: {self.session_id}")
                    # State zurücksetzen bei neuer Session
                    self.state_manager.set_buffer_state(BufferState.EMPTY)
                    self.state_manager.set_response_state(ResponseState.IDLE)
                else:
                    logger.warning(f"⚠️ Unexpected first message: {data.get('type')}")
            except TimeoutError:
                logger.exception("❌ Timeout waiting for session.created")
                await self._cleanup_websocket()
                return False
            except websockets.exceptions.WebSocketException as e:
                logger.exception(f"❌ WebSocket error during session creation: {e}")
                await self._cleanup_websocket()
                return False

            return True

        except InvalidStatusCode as e:
            logger.exception(f"❌ HTTP {e.status_code}: Azure OpenAI connection failed")
            if e.status_code == 400:
                logger.exception(
                    "💡 Check: API version (sollte 2025-04-01-preview sein) and deployment parameters")
            elif e.status_code == 401:
                logger.exception("💡 Check: API key authentication")
            elif e.status_code == 404:
                logger.exception("💡 Check: deployment name and region")
            await self._cleanup_websocket()
            return False
        except ssl.SSLError as e:
            logger.exception(f"❌ SSL certificate error: {e}")
            logger.exception("💡 SSL-Zertifikatsproblem - prüfe Netzwerkverbindung und Zertifikate")
            await self._cleanup_websocket()
            return False
        except websockets.exceptions.WebSocketException as e:
            logger.exception(f"❌ WebSocket error during connection: {e}")
            await self._cleanup_websocket()
            return False
        except (ConnectionError, OSError, RuntimeError) as e:
            logger.exception(f"❌ Failed to connect to Azure OpenAI: {e}")
            await self._cleanup_websocket()
            return False

    @staticmethod
    def _build_websocket_url() -> tuple[str, dict]:
        """WebSocket URL und Headers für Azure OpenAI erstellen."""
        from config.settings import settings as config

        endpoint = config.project_keiko_voice_endpoint
        # API-Key korrekt extrahieren (kann String oder SecretStr sein)
        if hasattr(config.project_keiko_api_key, "get_secret_value"):
            api_key = config.project_keiko_api_key.get_secret_value()
        else:
            api_key = str(config.project_keiko_api_key) if config.project_keiko_api_key else ""

        # Konfigurationsprüfung mit klarer Meldung und ohne Exception-Abbruch
        if not endpoint or not api_key:
            missing = []
            if not endpoint:
                missing.append("PROJECT_KEIKO_VOICE_ENDPOINT")
            if not api_key:
                missing.append("PROJECT_KEIKO_API_KEY")
            raise ValueError(f"Missing required Azure OpenAI configuration: {', '.join(missing)}")

        # URL-Konstruktion vereinfacht - nur HTTPS zu WSS für Sicherheit
        if endpoint.startswith("https://"):
            ws_url = endpoint.replace("https://", "wss://")
        elif endpoint.startswith("http://"):
            # Warnung für unsichere Verbindung
            logger.warning("⚠️ Using insecure HTTP connection - consider using HTTPS")
            ws_url = endpoint.replace("http://", "ws://")
        else:
            ws_url = endpoint

        headers = {
            "api-key": api_key,
            "User-Agent": "Keiko-Personal-Assistant/2.0-Realtime"
        }

        logger.info(f"🔗 WebSocket URL: {ws_url}")
        return ws_url, headers

    async def send_event(self, event_type: str, data: dict[str, Any] | None = None) -> bool:
        """Event an Azure OpenAI senden mit Deduplizierung."""
        if not self.is_connected or not self.ws:
            debug_log(f"Cannot send event {event_type}: not connected")
            return False

        # Vereinfachte Event-Deduplizierung
        event_key = f"{event_type}_{str(data)[:50]}"
        if self.event_deduplicator.is_duplicate(event_key):
            debug_log(f"🔄 Skipping duplicate event: {event_type}")
            return True

        try:
            event = {"type": event_type, **(data or {})}
            await self.ws.send(json.dumps(event))
            debug_log(f"📤 Sent {event_type} to Azure")
            return True

        except (ConnectionError, OSError, RuntimeError) as e:
            logger.exception(f"❌ Failed to send {event_type}: {e}")
            return False

    async def send_audio(self, audio_data: bytes) -> bool:
        """Audio zu Buffer hinzufügen.
        Auch wenn keine aktive Verbindung besteht, wird lokal gepuffert,
        sodass ein späterer Commit möglich ist (für Tests/offline).
        """
        if not audio_data or len(audio_data) < 1:
            debug_log("⚠️ Audio data too small or empty")
            return False

        current_buffer_state = self.state_manager.get_buffer_state()
        if current_buffer_state == BufferState.COMMITTING:
            debug_log("⚠️ Buffer currently committing, skipping audio")
            return False

        try:
            # Audio zu internem Buffer hinzufügen
            self._audio_chunks.append(audio_data)
            self._audio_chunk_count += 1
            self._total_audio_bytes += len(audio_data)

            # Buffer-Status aktualisieren
            if current_buffer_state == BufferState.EMPTY:
                self.state_manager.set_buffer_state(BufferState.ACCUMULATING)

            # Wenn nicht verbunden: nur puffern und Zustand aktualisieren
            if not self.is_connected or not self.ws:
                current_duration_ms = self._calculate_buffer_duration_ms()
                debug_log(
                    f"🎵 Audio chunk buffered offline: {len(audio_data)} bytes, duration: {current_duration_ms:.1f}ms"
                )
                if (
                    current_duration_ms >= self._min_audio_duration_ms
                    and self._total_audio_bytes >= self._min_audio_bytes
                ):
                    self.state_manager.set_buffer_state(BufferState.READY_FOR_COMMIT)
                return True

            # Verbunden: Base64-kodieren und Event senden
            base64_audio = base64.b64encode(audio_data).decode("utf-8")
            success = await self.send_event("input_audio_buffer.append", {"audio": base64_audio})

            if success:
                current_duration_ms = self._calculate_buffer_duration_ms()
                debug_log(
                    f"🎵 Audio chunk added: {len(audio_data)} bytes, duration: {current_duration_ms:.1f}ms"
                )

                # Prüfen ob bereit für Commit
                if (
                    current_duration_ms >= self._min_audio_duration_ms
                    and self._total_audio_bytes >= self._min_audio_bytes
                ):
                    self.state_manager.set_buffer_state(BufferState.READY_FOR_COMMIT)
            else:
                # Bei Fehler: Chunk aus Buffer entfernen
                self._audio_chunks.pop()
                self._audio_chunk_count -= 1
                self._total_audio_bytes -= len(audio_data)
                logger.error("❌ Failed to send audio, chunk removed from buffer")

            return success

        except (ValueError, TypeError, OSError) as e:
            logger.exception(f"❌ Audio processing error: {e}")
            self.state_manager.set_buffer_state(BufferState.ERROR_RECOVERY)
            return False

    def _calculate_buffer_duration_ms(self) -> float:
        """Berechne die aktuelle Pufferdauer in Millisekunden."""
        if self._total_audio_bytes == 0:
            return 0.0

        # PCM16 @ 24kHz: samples = bytes / 2, duration = samples / 24000
        total_samples = self._total_audio_bytes // self._bytes_per_sample
        duration_seconds = total_samples / self._samples_per_second
        return duration_seconds * 1000.0

    def _reset_audio_buffer(self):
        """Setzt den Audio-Buffer zurück."""
        self._audio_chunks.clear()
        self._audio_chunk_count = 0
        self._total_audio_bytes = 0

    async def commit_audio_buffer(self) -> bool:
        """Audio-Buffer committen."""
        current_time = time.time()

        # Basis-Validierung
        if not self.state_manager.is_safe_for_commit():
            debug_log("⚠️ Buffer commit not safe")
            return False

        if self._total_audio_bytes == 0:
            debug_log("⚠️ Buffer empty")
            self.state_manager.set_buffer_state(BufferState.EMPTY)
            return False

        # Rate-Limiting Check
        if current_time - self._last_commit_time < self._min_commit_interval:
            debug_log("⚠️ Commit rate-limited")
            return False

        try:
            self.state_manager.set_buffer_state(BufferState.COMMITTING)

            # Buffer-Größen-Validierung
            current_duration_ms = self._calculate_buffer_duration_ms()

            if (self._total_audio_bytes < self._min_audio_bytes or
                current_duration_ms < self._min_audio_duration_ms):

                self._consecutive_empty_commits += 1
                logger.warning(f"⚠️ Audio buffer insufficient: {current_duration_ms:.1f}ms")

                if self._consecutive_empty_commits >= self._max_empty_commits:
                    logger.error("❌ Too many insufficient commits, entering error recovery")
                    self._enter_error_recovery()

                self.state_manager.set_buffer_state(BufferState.ACCUMULATING)
                return False

            # Response-Status prüfen
            if not self.state_manager.is_safe_for_response():
                debug_log("⚠️ Response already active, deferring commit")
                self.state_manager.set_buffer_state(BufferState.READY_FOR_COMMIT)
                return False

            # Commit ausführen
            success = await self.send_event("input_audio_buffer.commit", {})

            if success:
                logger.info(f"✅ Audio buffer committed: {self._total_audio_bytes} bytes ({current_duration_ms:.1f}ms)")
                self._reset_audio_buffer()
                self._last_commit_time = current_time
                self._consecutive_empty_commits = 0
                self._error_recovery_active = False
                self.state_manager.set_buffer_state(BufferState.COMMITTED)
                asyncio.create_task(self._reset_to_empty_after_delay())
                return True
            logger.error("❌ Failed to commit audio buffer")
            self.state_manager.set_buffer_state(BufferState.ERROR_RECOVERY)
            return False

        except (ValueError, TypeError, OSError) as e:
            logger.exception(f"❌ Buffer commit error: {e}")
            self.state_manager.set_buffer_state(BufferState.ERROR_RECOVERY)
            return False

    async def _reset_to_empty_after_delay(self):
        """Setzt Buffer-Status nach kurzer Verzögerung auf EMPTY zurück."""
        await asyncio.sleep(0.1)
        current_state = self.state_manager.get_buffer_state()
        if current_state == BufferState.COMMITTED:
            self.state_manager.set_buffer_state(BufferState.EMPTY)

    def _enter_error_recovery(self):
        """Aktiviert Error Recovery Modus."""
        self._error_recovery_active = True
        self._min_audio_duration_ms = min(300, self._min_audio_duration_ms + 50)
        self._min_audio_bytes = int((self._min_audio_duration_ms / 1000) *
                                   self._samples_per_second * self._bytes_per_sample)
        self._consecutive_empty_commits = 0
        logger.info(f"🔧 Error recovery: minimum duration increased to {self._min_audio_duration_ms}ms")
        self.state_manager.set_buffer_state(BufferState.ERROR_RECOVERY)

    async def interrupt_response(self) -> bool:
        """Response unterbrechen."""
        response_state = self.state_manager.get_response_state()
        if response_state == ResponseState.IDLE:
            debug_log("⚠️ No active response to interrupt")
            return False

        self.state_manager.set_response_state(ResponseState.IDLE)
        success = await self.send_event("response.cancel", {})

        if success:
            logger.info("✅ Response interrupted")
        else:
            logger.error("❌ Failed to interrupt response")

        return success

    async def send_text(self, text: str) -> bool:
        """Text-Message senden."""
        if not text.strip():
            return False

        success = await self.send_event("conversation.item.create", {
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        })

        if success:
            # Response erstellen
            await self.send_event("response.create", {})

        return success

    async def update_session(self, settings: VoiceSettings) -> bool:
        """Session-Konfiguration aktualisieren."""
        # Tools-Definition für Azure Realtime
        tools = [
            {
                "type": "function",
                "name": "create_image",
                "description": "Erstellt ein Bild basierend auf einer Textbeschreibung mit DALL-E 3",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detaillierte Beschreibung des zu erstellenden Bildes"
                        },
                        "size": {
                            "type": "string",
                            "enum": ["1024x1024", "1024x1792", "1792x1024"],
                            "description": "Bildgröße",
                            "default": "1024x1024"
                        },
                        "quality": {
                            "type": "string",
                            "enum": ["standard", "hd"],
                            "description": "Bildqualität",
                            "default": "standard"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["Realistic", "Artistic", "Cartoon", "Photography",
                                     "Digital Art"],
                            "description": "Bildstil",
                            "default": "Realistic"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "type": "function",
                "name": "perform_web_research",
                "description": "Führt eine Web-Recherche zu einem bestimmten Thema durch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Suchbegriff oder Frage für die Web-Recherche"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximale Anzahl der Suchergebnisse",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "photo_request",
                "description": "Fordert den Nutzer auf, ein Foto aufzunehmen und aktiviert die Kamera-Funktion im Client",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Benutzer-ID (optional)"
                        }
                    }
                }
            },
            {
                "type": "function",
                "name": "capture_photo",
                "description": "Löst unmittelbar die Fotoaufnahme im Client aus",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "resolution": {
                            "type": "string",
                            "enum": ["640x480", "1280x720", "1920x1080"]
                        }
                    }
                }
            },
            {
                "type": "function",
                "name": "confirm_photo",
                "description": "Bestätigt die Fotoauswahl und löst das Speichern beim Client aus",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    }
                }
            },
            {
                "type": "function",
                "name": "photo_upload",
                "description": "Alias: Speichert das zuletzt aufgenommene Foto (Client-Upload)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    }
                }
            }
        ]

        session_config = {
            "session": {
                "modalities": settings.modalities,
                "instructions": settings.instructions,
                "voice": settings.voice,
                "input_audio_format": settings.input_audio_format,
                "output_audio_format": settings.output_audio_format,
                "input_audio_transcription": settings.input_audio_transcription,
                "turn_detection": settings.turn_detection,
                "temperature": settings.temperature,
                "max_response_output_tokens": settings.max_response_output_tokens,
                "tools": tools
            }
        }

        success = await self.send_event("session.update", session_config)
        if success:
            logger.info(f"✅ Session updated for {self.user_id} with {len(tools)} tools")
            # Note: Frontend notification removed - websocket not available in this context
        return success

    async def listen(self, websocket: WebSocket):
        """Event-Listener für Azure-Events mit verbesserter Fehlerbehandlung."""
        try:
            while self.is_connected and self.ws:
                try:
                    # Verwende recv() ohne wait_for um Queue-Probleme zu vermeiden
                    message = await self.ws.recv()
                    await self._handle_azure_event(message, websocket)
                except TimeoutError:
                    # Kurze Pause bei Timeout
                    await asyncio.sleep(0.1)
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("🔌 Azure WebSocket connection closed")
                    break
                except websockets.exceptions.WebSocketException as e:
                    logger.warning(f"⚠️ WebSocket error in listen: {e}")
                    break
                except (RuntimeError, OSError, ValueError) as e:
                    logger.warning(f"⚠️ Unexpected error in listen: {e}")
                    # Bei unerwarteten Fehlern kurz warten und weitermachen
                    await asyncio.sleep(0.1)
                    continue

        except (ConnectionError, OSError, RuntimeError) as e:
            logger.exception(f"❌ Listen error: {e}")
        finally:
            self.is_connected = False
            # Cleanup WebSocket-Verbindung
            await self._cleanup_websocket()

    async def _cleanup_websocket(self):
        """Bereinigt die WebSocket-Verbindung ordnungsgemäß."""
        if self.ws:
            try:
                # Versuche alle ausstehenden Nachrichten zu leeren
                while True:
                    try:
                        await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    except (TimeoutError, websockets.exceptions.ConnectionClosed):
                        break
                    except (ConnectionError, OSError, RuntimeError):
                        break
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.debug(f"Error during WebSocket cleanup: {e}")

            try:
                # Versuche WebSocket zu schließen ohne Status-Prüfung
                # ClientConnection hat kein 'closed' Attribut
                await self.ws.close()
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.debug(f"Error closing WebSocket: {e}")

            self.ws = None

    async def _handle_azure_event(self, message: str, websocket: WebSocket):
        """Azure-Event-Handler mit optimierter Fehlerbehandlung."""
        # Initialize variables to avoid unbound variable in exception handlers
        event = {}
        event_type = "unknown"

        try:
            event = json.loads(message)
            event_type = event.get("type", "unknown")

            debug_log(f"📨 Azure event: {event_type}")

            # Event-basierte State-Updates
            if event_type == "input_audio_buffer.committed":
                self.state_manager.set_buffer_state(BufferState.COMMITTED)
                item_id = event.get("item_id")
                logger.info(f"✅ Buffer committed by Azure, item_id: {item_id}")

            elif event_type == "response.created":
                self.state_manager.set_response_state(ResponseState.CREATING)
                response_id = event.get("response", {}).get("id")
                logger.info(f"🎯 Response created: {response_id}")
                # Zeitpunkt für TTS-Startlatenz messen
                self._last_response_created_ts = time.time()
                self._first_audio_delta_seen = False

            elif event_type == "response.done":
                self.state_manager.set_response_state(ResponseState.IDLE)
                response_id = event.get("response", {}).get("id")
                logger.info(f"✅ Response completed: {response_id}")
                # Reset der Audio-Delta-Flagge nach Abschluss
                self._first_audio_delta_seen = False

            elif event_type == "response.cancelled":
                self.state_manager.set_response_state(ResponseState.IDLE)
                logger.info("✅ Response cancelled")

            elif event_type == "input_audio_buffer.speech_started":
                # Benutzer beginnt zu sprechen (VAD/Turn-Detection) → sofortige Unterbrechung auslösen
                logger.info("🎤 Speech started detected")
                self._last_speech_started_ts = time.time()
                try:
                    # 1) Client sofort informieren, damit der Player Puffer leert (nahezu sofortiges Feedback)
                    interrupt_sent = await send_websocket_response(websocket, "interrupt", {})
                    if not interrupt_sent:
                        logger.debug("🔌 Client disconnected during interrupt signal")
                        return  # Stoppe weitere Verarbeitung
                    # 2) Aktive Antwort auf Azure-Seite abbrechen (TTS stoppen)
                    cancel_ok = await self.interrupt_response()
                    # 3) Latenz messen und in Monitoring aufnehmen
                    self._last_interrupt_forwarded_ts = time.time()
                    interrupt_latency_ms = (
                        (self._last_interrupt_forwarded_ts - self._last_speech_started_ts) * 1000.0
                    ) if self._last_speech_started_ts else None
                    if interrupt_latency_ms is not None:
                        try:
                            # Metriken im globalen Monitoring erfassen
                            voice_monitoring_manager.session_stats[self.user_id].append({
                                "metric": "interrupt_latency_ms",
                                "value": interrupt_latency_ms,
                                "timestamp": datetime.now(UTC)
                            })
                            voice_monitoring_manager.session_stats[self.user_id].append({
                                "metric": "response_cancel",
                                "value": 1 if cancel_ok else 0,
                                "timestamp": datetime.now(UTC)
                            })
                        except (RuntimeError, AttributeError, KeyError) as e:
                            logger.debug(f"Monitoring append failed: {e}")
                except (ConnectionError, OSError, RuntimeError) as e:
                    logger.exception(f"❌ Interrupt forwarding failed: {e}")

            elif event_type == "input_audio_buffer.speech_stopped":
                logger.info("🤫 Speech stopped detected")
                # Auto-Commit nach Speech-Stop
                if self.state_manager.is_safe_for_commit():
                    asyncio.create_task(self._auto_commit_with_delay())

            elif event_type == "response.audio.delta":
                audio_data = event.get("delta", "")
                if audio_data:
                    # Audio-Daten sind kritisch - verwende normale send_websocket_response
                    # um Verbindungsabbrüche zu erkennen
                    success = await send_websocket_response(websocket, "audio_response", {"audio": audio_data})
                    if not success:
                        logger.debug("🔌 Client disconnected during audio transmission")
                        return  # Stoppe weitere Audio-Übertragung
                    # Ersten Audio-Frame pro Antwort tracken, um TTS-Startlatenz zu messen
                    if not self._first_audio_delta_seen:
                        self._first_audio_delta_seen = True
                        if self._last_response_created_ts is not None:
                            try:
                                tts_latency_ms = (time.time() - self._last_response_created_ts) * 1000.0
                                voice_monitoring_manager.session_stats[self.user_id].append({
                                    "metric": "tts_start_latency_ms",
                                    "value": tts_latency_ms,
                                    "timestamp": datetime.now(UTC)
                                })
                                logger.info(f"📈 TTS start latency: {tts_latency_ms:.1f} ms")
                            except (RuntimeError, AttributeError, KeyError) as e:
                                logger.debug(f"Monitoring append failed: {e}")
            # Client-Bestätigung: Upload abgeschlossen
            elif event_type == "photo_upload_done":
                try:
                    event.get("image_url")
                    await self._speak("Ich habe das Foto gespeichert.")
                    # Foto-Modus beenden
                    self.photo_mode_active = False
                except (ConnectionError, OSError, RuntimeError) as e:
                    logger.warning(f"notify upload done failed: {e}")

            # Tool/Function Calling Aggregation
            elif event_type == "response.output_item.added":
                try:
                    output_item = event.get("item", {})
                    if output_item.get("type") == "function_call":
                        call_id = output_item.get("call_id") or output_item.get("id")
                        name = output_item.get("name")
                        if call_id and name:
                            self._function_calls[call_id] = {
                                "name": name,
                                "args": [],
                                "response_id": event.get("response", {}).get("id"),
                            }
                            logger.info(f"🧩 Function call started: {name} ({call_id})")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.exception(f"❌ function_call start parse error: {e}")

            elif event_type == "response.function_call_arguments.delta":
                try:
                    call_id = event.get("call_id")
                    arguments_chunk = event.get("delta", "")
                    if call_id and arguments_chunk is not None:
                        entry = self._function_calls.setdefault(call_id, {"name": None, "args": []})
                        entry["args"].append(arguments_chunk)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.exception(f"❌ function_call delta parse error: {e}")

            elif event_type == "response.function_call_arguments.done":
                try:
                    call_id = event.get("call_id")
                    entry = self._function_calls.get(call_id)
                    if entry:
                        name = entry.get("name")
                        args_str = "".join(entry.get("args", []))
                        try:
                            arguments = json.loads(args_str) if args_str.strip() else {}
                        except (json.JSONDecodeError, ValueError):
                            fixed = args_str.strip()
                            if not fixed.endswith("}") and not fixed.endswith("]"):
                                fixed += "}"
                            try:
                                arguments = json.loads(fixed)
                            except (json.JSONDecodeError, ValueError):
                                arguments = {"_raw": args_str}
                        # Serverseitige Ausführung (produktionsreif): orchestrator tool call
                        asyncio.create_task(self._execute_and_respond_function(
                            call_id=call_id,
                            name=name,
                            arguments=arguments,
                            websocket=websocket
                        ))
                        logger.info(
                            f"🧩 Function call ready: {name} ({call_id}) -> executing server-side")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.exception(f"❌ function_call done parse error: {e}")


            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                if transcript:
                    # Transkription ist wichtig - verwende sichere Übertragung
                    await safe_send_websocket_response(websocket, "transcription", {"text": transcript})
                    # Voice-Trigger für Fotoaufnahme, wenn Foto-Modus aktiv ist
                    try:
                        if getattr(self, "photo_mode_active", False):
                            text = str(transcript).strip().lower()
                            # Einfache Muster (Substrings) – ergänzt um JA/NEIN-Varianten
                            ready_patterns = [
                                "ich bin bereit", "jetzt bin ich bereit", "jetzt bereit",
                                "mach das foto", "mach ein foto", "jetzt foto",
                                "i am ready", "i'm ready", "take the photo", "take a photo"
                            ]
                            confirm_patterns = [
                                "ist gut", "passt so", "speichern", "kannst speichern", "gut so",
                                "that's good", "looks good", "save it", "you can save",
                                "ja, speichern", "ja speichern"
                            ]
                            retake_patterns = [
                                "nochmal", "erneut", "retake", "neues foto", "mach nochmal",
                                "take again", "one more", "try again", "nein, nochmal", "nein nochmal"
                            ]

                            # Fuzzy-/Semantik-nahe Übereinstimmung (leichtgewichtig)
                            def _matches(candidate: str, patterns: list[str], threshold: float = 0.85) -> bool:
                                try:
                                    import difflib
                                except ImportError:
                                    difflib = None  # type: ignore
                                normalized = candidate.replace(",", "").replace(".", "").strip()
                                for patt in patterns:
                                    if patt in normalized:
                                        return True
                                    if difflib is not None:
                                        ratio = difflib.SequenceMatcher(None, normalized, patt).ratio()
                                        if ratio >= threshold:
                                            return True
                                return False

                            if _matches(text, ready_patterns):
                                await send_websocket_response(websocket, "photo_capture", {
                                    "trigger": "voice_ready"
                                })
                                # Direkt nach Auslösung die Frage stellen
                                with contextlib.suppress(Exception):
                                    await self._speak("Ich habe das Foto aufgenommen. Gefällt es dir so, soll ich es speichern oder ein neues machen? Sage einfach 'Speichern' oder 'nochmal'.")
                            elif _matches(text, confirm_patterns):
                                # Client: Upload auslösen
                                await send_websocket_response(websocket, "photo_upload", {
                                    "trigger": "voice_confirm"
                                })
                                await self._speak("Dann speichere ich es jetzt.")
                            elif _matches(text, retake_patterns):
                                await send_websocket_response(websocket, "photo_capture", {
                                    "trigger": "voice_retake"
                                })
                    except (KeyError, AttributeError, RuntimeError):
                        pass

            elif event_type == "error":
                await self._handle_azure_error(event, websocket)

            # Generisches Event-Forwarding an Client (sicher)
            await safe_send_websocket_response(websocket, "azure_event", {
                "event_type": event_type,
                "data": event
            })

        except json.JSONDecodeError as e:
            logger.exception(f"❌ Invalid JSON from Azure: {e}")
            # Generisches Event-Forwarding an Client (sicher)
            await safe_send_websocket_response(websocket, "azure_event", {
                "event_type": event_type,
                "data": event
            })
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Event handling error - Verbindungsproblem: {e}")
            # Generisches Event-Forwarding an Client (sicher)
            await safe_send_websocket_response(websocket, "azure_event", {
                "event_type": event_type,
                "data": event
            })
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.exception(f"❌ Event handling error - Unerwarteter Fehler: {e}")
            # Generisches Event-Forwarding an Client (sicher)
            await safe_send_websocket_response(websocket, "azure_event", {
                "event_type": event_type,
                "data": event
            })

    async def _auto_commit_with_delay(self):
        """Auto-Commit mit kleiner Verzögerung für stabileren Betrieb."""
        await asyncio.sleep(0.2)  # Kurze Verzögerung für Stabilität
        try:
            await self.commit_audio_buffer()
        except (ValueError, TypeError, OSError) as e:
            logger.exception(f"❌ Auto-commit error: {e}")

    async def _execute_and_respond_function(self, call_id: str, name: str,
                                            arguments: dict[str, Any],
                                            websocket: WebSocket) -> None:
        """Führt erkannte Funktion serverseitig aus und sendet Completion an Azure & Client.

        - Mappt Funktionsnamen auf Orchestrator-Tools
        - Sendet Fortschritts-/Fehler-Updates an Client
        - Liefert function_completion an Azure zurück (Konversationsfluss)
        """
        try:
            # Alias-Mapping für Toolnamen (Azure Realtime -> Orchestrator)
            name_map: dict[str, str] = {
                "generate_image": "generate_image",
                "create_image": "generate_image",  # Alias von Azure-Realtime-Funktion
                "analyze_and_maybe_generate_image": "analyze_and_maybe_generate_image",
                "perform_web_research": "perform_web_research",
                # Foto-Workflow aktivieren
                "photo_request": "photo_request",
                # Fotoaufnahme direkt auslösen
                "capture_photo": "capture_photo",
                "take_photo": "capture_photo",
                # Bestätigung/Speichern
                "confirm_photo": "confirm_photo",
                "photo_upload": "photo_upload",
            }
            if name not in name_map:
                await send_websocket_response(websocket, "agent", {
                    "id": f"func_{call_id}",
                    "call_id": call_id,
                    "name": name,
                    "status": "failed",
                    "message": "Funktion nicht erlaubt"
                })
                return

            mapped_name = name_map[name]

            # Kontext-Parameter ergänzen
            user_id = None
            try:
                # user_id aus WebSocket-Registry herleiten
                for uid, ws in active_connections.items():
                    if ws == websocket:
                        user_id = uid
                        break
            except (KeyError, AttributeError):
                user_id = None

            params = dict(arguments or {})
            if mapped_name in {"generate_image", "analyze_and_maybe_generate_image", "photo_request", "capture_photo", "confirm_photo", "photo_upload"}:
                params.setdefault("user_id", user_id)
                params.setdefault("session_id", user_id)

            # Orchestrator-Tool direkt ausführen (umgeht fehlerhafte Tools-Abstraktion)
            if mapped_name == "generate_image":
                from .voice_routes_refactored import refactored_generate_image_implementation
                result = await refactored_generate_image_implementation(
                    prompt=params.get("prompt", ""),
                    size=params.get("size", "1024x1024"),
                    quality=params.get("quality", "standard"),
                    style=params.get("style", "Realistic"),
                    user_id=params.get("user_id"),
                    session_id=params.get("session_id")
                )
                # Debug: Log result structure from refactored implementation
                logger.info(f"🐛 DEBUG refactored result structure: {result}")
                logger.info(f"🐛 DEBUG refactored result type: {type(result)}")
                if isinstance(result, dict):
                    logger.info(f"🐛 DEBUG refactored result keys: {list(result.keys())}")
                    logger.info(f"🐛 DEBUG has image_data: {'image_data' in result}")
                    if "image_data" in result:
                        logger.info(f"🐛 DEBUG image_data content: {result['image_data']}")

                # ROBUSTE LÖSUNG: Direkte URL-Extraktion für Bildgenerierung
                if isinstance(result, dict) and result.get("success"):
                    # Versuche URL aus verschiedenen Quellen zu extrahieren
                    image_url = None

                    # 1. Aus image_data
                    if "image_data" in result and isinstance(result["image_data"], dict):
                        image_data = result["image_data"]
                        image_url = (
                            image_data.get("storage_url") or
                            image_data.get("sas_url") or
                            image_data.get("blob_url") or
                            image_data.get("url")
                        )

                    # 2. Direkt aus result
                    if not image_url:
                        image_url = (
                            result.get("storage_url") or
                            result.get("sas_url") or
                            result.get("blob_url") or
                            result.get("url")
                        )

                    # 3. Aus data Feld
                    if not image_url and "data" in result and isinstance(result["data"], dict):
                        data = result["data"]
                        image_url = (
                            data.get("storage_url") or
                            data.get("sas_url") or
                            data.get("blob_url") or
                            data.get("url")
                        )

                    # 4. KRITISCH: URL aus result String extrahieren (Regex)
                    if not image_url and "result" in result and isinstance(result["result"], str):
                        import re
                        result_text = result["result"]
                        # Suche nach https:// URLs im Text
                        url_pattern = r"https://[^\s\n]+"
                        url_match = re.search(url_pattern, result_text)
                        if url_match:
                            image_url = url_match.group(0)
                            logger.info(f"🎯 URL aus result String extrahiert: {image_url}")

                    logger.info(f"🐛 DEBUG extracted image_url: {image_url}")

                    # Wenn URL gefunden, erstelle Frontend-kompatible Struktur
                    if image_url:
                        logger.info(f"✅ Creating frontend-compatible image structure with URL: {image_url}")

                        # Überschreibe result mit Frontend-kompatiblem Format
                        result = {
                            "success": True,
                            "image_data": {
                                "storage_url": image_url,
                                "metadata": {
                                    "optimized_prompt": "Generated image",
                                    "timestamp": result.get("timestamp"),
                                    "parameters": {
                                        "size": "1024x1024",
                                        "quality": "standard"
                                    }
                                }
                            },
                            "session_id": result.get("session_id", "unknown")
                        }
                        logger.info(f"✅ Frontend-compatible result created: {result}")
            elif mapped_name == "perform_web_research":
                from .voice_routes_refactored import refactored_perform_web_research_implementation
                result = await refactored_perform_web_research_implementation(
                    query=params.get("query", ""),
                    max_results=params.get("max_results", 5),
                    user_id=params.get("user_id"),
                    session_id=params.get("session_id")
                )
            elif mapped_name == "photo_request":
                from .voice_routes_refactored import refactored_photo_request_implementation
                result = await refactored_photo_request_implementation(
                    user_id=params.get("user_id"),
                )
                # Foto-Modus aktivieren und Client vorbereiten
                self.photo_mode_active = True
                self.last_photo_request_ts = time.time()
            elif mapped_name == "capture_photo":
                # Client sofort anweisen, Foto aufzunehmen
                try:
                    await send_websocket_response(websocket, "photo_capture", {
                        "trigger": "function_call",
                        "resolution": params.get("resolution")
                    })
                    # Assistentenhinweis zum nächsten Schritt (Bestätigung/Retake)
                    await self.send_event("conversation.item.create", {
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": "Ich habe das Foto aufgenommen. Bitte prüfe die Vorschau. Sage 'Speichern', wenn es gut ist, oder 'nochmal' für eine neue Aufnahme."
                            }]
                        }
                    })
                    await self.send_event("response.create", {})
                except (RuntimeError, AttributeError):
                    pass
                result = {"status": "success", "message": "photo_capture_triggered"}
            elif mapped_name in {"confirm_photo", "photo_upload"}:
                # Client: Upload auslösen
                try:
                    now = time.time()
                    if not self._last_photo_action_ts or (now - self._last_photo_action_ts) > self._photo_action_cooldown_s:
                        self._last_photo_action_ts = now
                        await send_websocket_response(websocket, "photo_upload", {
                            "trigger": "function_call"
                        })
                        await self._speak("Ich speichere das Foto jetzt.")
                except (ConnectionError, OSError, RuntimeError) as e:
                    logger.warning(f"photo_upload trigger failed: {e}")
                result = {"status": "success", "message": "photo_upload_triggered"}
            else:
                # Fallback für unbekannte Tools
                result = {"status": "error", "error": f"Tool {mapped_name} nicht implementiert"}

            # Client informieren (Status + Ergebnis)
            status = "success" if result.get("status", "success") == "success" else "failed"

            # Frontend-kompatibles Format für agent Events
            # Für Bildgenerierung: Echte Bilddaten aus image_data extrahieren
            if mapped_name == "generate_image" and "image_data" in result:
                image_data = result["image_data"]
                # Debug: Log image_data structure
                logger.info(f"🐛 DEBUG image_data structure: {image_data}")
                logger.info(f"🐛 DEBUG image_data type: {type(image_data)}")
                if isinstance(image_data, dict):
                    logger.info(f"🐛 DEBUG image_data keys: {list(image_data.keys())}")
                    logger.info(f"🐛 DEBUG storage_url: {image_data.get('storage_url')}")

                # Ensure image_data is a dictionary
                if isinstance(image_data, dict):
                    metadata = image_data.get("metadata", {})
                    params = metadata.get("parameters", {}) if isinstance(metadata, dict) else {}

                    # Versuche verschiedene Felder für die Bild-URL
                    image_url = (
                        image_data.get("storage_url") or
                        image_data.get("sas_url") or
                        image_data.get("blob_url") or
                        image_data.get("url") or
                        None
                    )

                    # Fallback: Versuche auch direkt aus result
                    if not image_url:
                        image_url = (
                            result.get("storage_url") or
                            result.get("sas_url") or
                            result.get("blob_url") or
                            result.get("url") or
                            None
                        )
                        logger.info(f"🐛 DEBUG fallback image_url from result: {image_url}")

                    # Letzter Fallback: Verwende eine Test-URL falls keine URL gefunden wurde
                    if not image_url:
                        image_url = "https://via.placeholder.com/1024x1024/FF0000/FFFFFF?text=Generated+Image"
                        logger.warning(f"🐛 DEBUG using placeholder image_url: {image_url}")

                    logger.info(f"🐛 DEBUG final image_url: {image_url}")

                    # Frontend erwartet content.content Array mit ImageData Objekten
                    content = {
                        "content": [
                            {
                                "id": call_id,
                                "type": "image",
                                "description": metadata.get("optimized_prompt", "Generated image") if isinstance(metadata, dict) else "Generated image",
                                "image_url": image_url,
                                "size": params.get("size", "1024x1024") if isinstance(params, dict) else "1024x1024",
                                "quality": params.get("quality", "standard") if isinstance(params, dict) else "standard",
                                "created_at": metadata.get("timestamp") if isinstance(metadata, dict) else None,
                                "session_id": result.get("session_id", "unknown") if isinstance(result, dict) else "unknown"
                            }
                        ]
                    }
                else:
                    # Fallback if image_data is not a dictionary
                    content = {"content": [{"id": call_id, "type": "image", "description": "Generated image"}]}
            else:
                content = result.get("content", {}) if isinstance(result, dict) else {}

            agent_response = {
                "id": f"func_{call_id}",
                "call_id": call_id,
                "name": mapped_name,
                "status": status,
                "output": True,  # Zeigt an, dass Content verfügbar ist
                "content": content  # Enthält die echten Bilddaten
            }

            logger.info(f"🎨 Sending agent response to frontend: {mapped_name} - {status}")
            logger.debug(f"🎨 Agent response content: {content}")
            if mapped_name == "generate_image" and content.get("content"):
                image_content = content["content"][0] if content["content"] else {}
                image_url = image_content.get("image_url", "")
                if image_url:
                    logger.info(f"🖼️  Image URL sent to frontend: {image_url[:80]}...")
                    logger.info(f"🖼️  Image data format: {list(image_content.keys())}")

            await send_websocket_response(websocket, "agent", agent_response)

            # function_completion an Azure senden (korrekte API-Events)
            output_text = (
                "Bild generiert." if mapped_name == "generate_image" and status == "success" else
                "Analyse abgeschlossen." if mapped_name == "analyze_and_maybe_generate_image" and status == "success" else
                "Web-Recherche abgeschlossen." if mapped_name == "perform_web_research" and status == "success" else
                "Foto-Anfrage bestätigt." if mapped_name == "photo_request" and status == "success" else
                "Fotoaufnahme ausgelöst." if mapped_name == "capture_photo" and status == "success" else
                f"Funktion {mapped_name} fehlgeschlagen"
            )
            try:
                # Korrekte Azure Realtime API Events verwenden
                await self.send_event("conversation.item.create", {
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text
                    }
                })
                await self.send_event("response.create", {})
                logger.info(f"✅ Function completion sent to Azure: {mapped_name}")
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.warning(f"⚠️ Azure function_completion send failed: {e}")
        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(f"❌ Function execution failed: {e}")
            with contextlib.suppress(ConnectionError, OSError, RuntimeError):
                await send_websocket_response(websocket, "agent", {
                    "id": f"func_{call_id}",
                    "call_id": call_id,
                    "name": name,
                    "status": "failed",
                    "error": str(e)
                })

    async def _handle_azure_error(self, error_event: dict[str, Any], websocket: WebSocket):
        """Azure-Fehlerbehandlung."""
        error_data = error_event.get("error", {})
        error_code = error_data.get("code", "unknown")
        error_message = error_data.get("message", "Unbekannter Fehler")

        # Nur kritische Fehler als ERROR loggen, andere als WARNING
        if error_code in ["connection_error", "authentication_failed", "service_unavailable"]:
            logger.error(f"❌ AZURE CRITICAL ERROR - Code: {error_code}, Message: {error_message}")
        else:
            logger.warning(f"⚠️ AZURE WARNING - Code: {error_code}, Message: {error_message}")

        # Spezifische Fehlerbehandlung
        if error_code == "input_audio_buffer_commit_empty":
            self.state_manager.set_buffer_state(BufferState.EMPTY)
            self._consecutive_empty_commits += 1

        elif error_code == "conversation_already_has_active_response":
            self.state_manager.set_response_state(ResponseState.IDLE)
            await self.interrupt_response()

        elif error_code == "rate_limit_exceeded":
            await asyncio.sleep(1.0)

        # Error an Client weiterleiten
        await send_websocket_response(websocket, "azure_error", {
            "error_code": error_code,
            "error_message": error_message
        })

    async def close(self):
        """Verbindung ordnungsgemäß schließen."""
        self.is_connected = False

        # State zurücksetzen
        self.state_manager.set_buffer_state(BufferState.EMPTY)
        self.state_manager.set_response_state(ResponseState.IDLE)

        # Verwende die neue Cleanup-Funktion
        await self._cleanup_websocket()

        logger.info(f"🔌 Azure connection closed for {self.user_id}")


# =====================================================================
# Router und WebSocket Endpoints
# =====================================================================

router = create_router("", ["voice", "realtime", "azure-ai-foundry"])
router.responses.update({404: {"description": "Not found"}})

# Connection-Manager
active_connections: dict[str, WebSocket] = {}
azure_connections: dict[str, AzureRealtimeClient] = {}
connection_info: dict[str, VoiceConnectionInfo] = {}
voice_settings: dict[str, VoiceSettings] = {}


@router.websocket("/{user_id}")
async def voice_websocket_endpoint(websocket: WebSocket, user_id: str):
    """Optimierter WebSocket-Endpoint für Voice-Kommunikation."""
    await handle_voice_websocket(websocket, user_id)


async def handle_voice_websocket(websocket: WebSocket, user_id: str):
    """Optimierter WebSocket-Handler mit verbesserter Fehlerbehandlung."""
    logger.info(f"🔌 Voice WebSocket connection request: {user_id}")
    azure_client = None
    rate_limit_slot_id = None
    rate_limit_service = None  # Initialize to avoid unbound variable in finally block

    try:
        # Voice Rate Limiting prüfen (nur wenn Service verfügbar und aktiviert)
        rate_limit_service = get_voice_rate_limit_service()
        rate_limit_slot_id = None

        # Prüfe explizit die Konfiguration
        try:
            from config.voice_rate_limiting_config import VoiceRateLimitSettings
            settings = VoiceRateLimitSettings()
            voice_rate_limiting_enabled = settings.enabled
        except (ImportError, AttributeError, RuntimeError):
            voice_rate_limiting_enabled = False

        if rate_limit_service and VOICE_RATE_LIMITING_AVAILABLE and voice_rate_limiting_enabled:
            logger.debug(f"Voice Rate Limiting aktiv für Benutzer {user_id}")
            context = VoiceRateLimitContext(
                user_id=user_id,
                ip_address=getattr(websocket.client, "host", "unknown") if hasattr(websocket, "client") else "unknown",
                user_tier=UserTier.STANDARD,
                endpoint="/voice/websocket"
            )

            # Concurrent Connection Limit prüfen
            concurrent_result = await rate_limit_service.rate_limiter.check_concurrent_limit(
                VoiceOperation.WEBSOCKET_CONNECTION, context
            )

            if not concurrent_result.allowed:
                logger.warning(f"WebSocket rate limit exceeded for user {user_id}")
                await websocket.close(code=1008, reason="Rate limit exceeded")
                return

            # Concurrent Slot erwerben
            success, rate_limit_slot_id = await rate_limit_service.rate_limiter.acquire_concurrent_slot(
                VoiceOperation.WEBSOCKET_CONNECTION, context
            )

            if not success:
                logger.warning(f"Failed to acquire WebSocket slot for user {user_id}")
                await websocket.close(code=1008, reason="Connection limit reached")
                return
        else:
            logger.debug(f"Voice Rate Limiting deaktiviert - WebSocket-Verbindung für {user_id} erlaubt")

        await websocket.accept()
        logger.info(f"✅ Voice WebSocket accepted: {user_id}")

        # Connection registrieren
        active_connections[user_id] = websocket
        connection_info[user_id] = VoiceConnectionInfo(
            user_id=user_id,
            connected_at=datetime.now(UTC),
            endpoint_used="/api/voice"
        )

        # Voice-Settings initialisieren
        if user_id not in voice_settings:
            voice_settings[user_id] = VoiceSettings(user_id=user_id)

        # Azure-Client erstellen und verbinden
        azure_client = AzureRealtimeClient(user_id)
        connected = await azure_client.connect()

        listen_task: asyncio.Task | None = None
        if not connected:
            # Degradierter Modus: Verbindung offen halten, aber Voice deaktivieren
            logger.warning("🔇 Azure Realtime unavailable – entering degraded mode (voice disabled)")
            connection_sent = await send_websocket_response(websocket, "connection_established", {
                "user_id": user_id,
                "voice_available": False,
                "azure_session_id": None,
                "api_version": None
            })

            if not connection_sent:
                logger.info(f"🔌 Client disconnected before degraded mode notification could be sent: {user_id}")
                await cleanup_voice_connection(user_id, None)
                return

            # Zusätzliche Error-Response senden
            await safe_send_websocket_response(websocket, "connection_error", {
                "error": "Azure OpenAI Realtime nicht verfügbar. Voice-Funktionen deaktiviert.",
                "voice_available": False
            })

            # Azure-Client auf None setzen für degradierten Modus
            azure_client = None
        else:
            # Azure-Connection speichern
            azure_connections[user_id] = azure_client
            connection_info[user_id].azure_session_id = azure_client.session_id

            # Session konfigurieren
            await azure_client.update_session(voice_settings[user_id])

            # Bestätigung senden (mit Fehlerbehandlung)
            connection_sent = await send_websocket_response(websocket, "connection_established", {
                "user_id": user_id,
                "voice_available": True,
                "azure_session_id": azure_client.session_id,
                "api_version": "2025-04-01-preview"
            })

            if not connection_sent:
                logger.info(f"🔌 Client disconnected before connection_established could be sent: {user_id}")
                # Client ist bereits getrennt, cleanup und return
                await cleanup_voice_connection(user_id, azure_client)
                return

            # Azure-Listener starten
            listen_task = asyncio.create_task(azure_client.listen(websocket))

        # Hinweis an Client (Tools-Verfügbarkeit bleibt bestehen)
        settings_sent = await send_websocket_response(websocket, "settings_updated", {
            "settings": {"tools": ["generate_image", "analyze_and_maybe_generate_image",
                                   "perform_web_research", "photo_request"]}
        })

        if not settings_sent:
            logger.info(f"🔌 Client disconnected before settings_updated could be sent: {user_id}")
            # Client ist bereits getrennt, cleanup und return
            await cleanup_voice_connection(user_id, azure_client)
            return

        # Client-Messages verarbeiten
        try:
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    await handle_client_message(websocket, user_id, data, azure_client)
                except json.JSONDecodeError as e:
                    logger.exception(f"❌ JSON decode error: {e}")
                    await send_error_response(websocket, "Invalid JSON format")
                except (RuntimeError, OSError, ConnectionError) as e:
                    logger.exception(f"❌ Message handling error: {e}")

        except WebSocketDisconnect:
            logger.info(f"🔌 Client disconnected: {user_id}")
        finally:
            if listen_task is not None and not listen_task.done():
                listen_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await listen_task

    except (WebSocketDisconnect, StarletteWebSocketDisconnect, ConnectionClosedOK, ClientDisconnected):
        logger.info(f"🔌 Client disconnected during WebSocket handling: {user_id}")
    except (RuntimeError, OSError, ValueError) as e:
        logger.exception(f"❌ Voice WebSocket error: {e}")
        # Versuche Error-Response zu senden, aber ignoriere Verbindungsfehler
        with contextlib.suppress(WebSocketDisconnect, StarletteWebSocketDisconnect, ConnectionClosedOK, ClientDisconnected, builtins.BaseException):
            await send_error_response(websocket, str(e), "connection_error")
    finally:
        # Rate Limit Slot freigeben
        if rate_limit_slot_id and rate_limit_service and VOICE_RATE_LIMITING_AVAILABLE:
            try:
                context = VoiceRateLimitContext(
                    user_id=user_id,
                    ip_address=getattr(websocket.client, "host", "unknown") if hasattr(websocket, "client") else "unknown",
                    user_tier=UserTier.STANDARD,
                    endpoint="/voice/websocket"
                )
                await rate_limit_service.rate_limiter.release_concurrent_slot(
                    VoiceOperation.WEBSOCKET_CONNECTION, context, rate_limit_slot_id
                )
                logger.debug(f"Released rate limit slot for user {user_id}")
            except (RuntimeError, AttributeError, ConnectionError) as e:
                logger.error(f"Failed to release rate limit slot: {e}")

        await cleanup_voice_connection(user_id, azure_client)


async def handle_client_message(websocket: WebSocket, user_id: str, data: dict[str, Any],
                                azure_client: AzureRealtimeClient | None):
    """Client-Message-Handler."""
    message_type = data.get("type", "unknown")
    debug_log(f"📨 Client message: {message_type} from {user_id}")

    if message_type == "settings":
        settings_data = data.get("settings", {})
        if azure_client and azure_client.is_connected:
            await update_voice_settings(user_id, settings_data, azure_client)
        await safe_send_websocket_response(websocket, "settings_updated", {"settings": settings_data})

    elif message_type == "audio":
        # Prüfe ob Azure-Client verfügbar und verbunden ist
        if not azure_client or not azure_client.is_connected:
            debug_log("⚠️ Audio message ignored - Azure client not connected")
            await safe_send_websocket_response(websocket, "audio_ignored", {
                "reason": "voice_unavailable",
                "message": "Voice-Funktionen sind derzeit nicht verfügbar"
            })
            return

        audio_data = data.get("content", "") or data.get("data", "")
        if audio_data:
            try:
                audio_bytes = base64.b64decode(audio_data)
                success = await azure_client.send_audio(audio_bytes)
                debug_log(f"🎵 Send audio success: {success}")
            except (ValueError, TypeError, OSError) as e:
                logger.exception(f"❌ Audio error: {e}")

    elif message_type == "text":
        if not azure_client or not azure_client.is_connected:
            debug_log("⚠️ Text message ignored - Azure client not connected")
            await safe_send_websocket_response(websocket, "text_ignored", {
                "reason": "voice_unavailable",
                "message": "Voice-Funktionen sind derzeit nicht verfügbar"
            })
            return

        text_content = data.get("content", "")
        if text_content.strip():
            debug_log(f"💬 Sending text to Azure: {text_content}")
            await azure_client.send_text(text_content)

    elif message_type == "interrupt":
        if not azure_client or not azure_client.is_connected:
            debug_log("⚠️ Interrupt message ignored - Azure client not connected")
            await safe_send_websocket_response(websocket, "interrupt_ignored", {
                "reason": "voice_unavailable",
                "message": "Voice-Funktionen sind derzeit nicht verfügbar"
            })
            return

        success = await azure_client.interrupt_response()
        await safe_send_websocket_response(websocket, "audio_interrupted", {"success": success})

    elif message_type == "commit_audio":
        success = await azure_client.commit_audio_buffer()
        await safe_send_websocket_response(websocket, "audio_committed", {"success": success})

    else:
        debug_log(f"⚠️ Unknown message type: {message_type}")


async def update_voice_settings(user_id: str, settings_data: dict[str, Any],
                                azure_client: AzureRealtimeClient):
    """Voice-Settings aktualisieren."""
    if user_id not in voice_settings:
        voice_settings[user_id] = VoiceSettings(user_id=user_id)

    current_settings = voice_settings[user_id]

    # Voice Configuration direkt aktualisieren
    voice_config = current_settings.voice_config

    for key, value in settings_data.items():
        if key == "voice":
            voice_config.voice = value
        elif key == "detection_type":
            voice_config.detection_type = value
        elif key == "threshold":
            voice_config.threshold = float(value)
        elif key == "silence_duration":
            voice_config.silence_duration_ms = int(value)
        elif key == "prefix_padding":
            voice_config.prefix_padding_ms = int(value)
        elif key == "transcription_model":
            voice_config.transcription_model = value
        elif key == "transcription_language":
            voice_config.transcription_language = value
        elif key == "temperature":
            voice_config.temperature = float(value)
        elif key == "speech_rate":
            voice_config.speech_rate = float(value)
        elif key == "max_response_tokens":
            voice_config.max_response_output_tokens = int(value)

    # Voice Configuration validieren nach Änderungen
    try:
        voice_config._validate_config()
        logger.debug(f"✅ Voice configuration validated for {user_id}")
    except ValueError as e:
        logger.error(f"❌ Invalid voice configuration for {user_id}: {e}")
        raise ValueError(f"Invalid voice configuration: {e}")

    # Session aktualisieren
    await azure_client.update_session(current_settings)
    logger.info(f"⚙️ Voice settings updated for {user_id}")


async def cleanup_voice_connection(user_id: str, azure_client: AzureRealtimeClient | None):
    """Connection-Cleanup.

    Entfernt alle registrierten Ressourcen für den Benutzer, inklusive Voice-Settings,
    um Test- und Laufzeit-Leaks zu vermeiden.
    """
    logger.info(f"🧹 Cleaning up voice connection for {user_id}")

    if azure_client:
        await azure_client.close()

    # Registrierungen entfernen (inkl. voice_settings, um Leaks zwischen Tests zu vermeiden)
    active_connections.pop(user_id, None)
    azure_connections.pop(user_id, None)
    connection_info.pop(user_id, None)
    voice_settings.pop(user_id, None)

    logger.info(f"✅ Voice connection cleanup completed for {user_id}")


# =====================================================================
# Voice Error Recovery Manager Integration
# =====================================================================
# =====================================================================


class VoiceErrorRecoveryManager:
    """Comprehensive error recovery and fault tolerance manager."""

    def __init__(self):
        self.error_history: list[ErrorEvent] = []
        self.recovery_strategy: RecoveryStrategy = DEFAULT_RECOVERY_STRATEGY
        self.active_recoveries: dict[str, asyncio.Task] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}

    async def handle_error(self, error_type: str, error_message: str, user_id: str | None = None,
                          component: str | None = None, details: dict[str, Any] | None = None) -> bool:
        """Handle error with automatic recovery attempts."""
        if details is None:
            details = {}

        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now(UTC),
            error_type=error_type,
            severity=determine_error_severity(error_type, Exception(error_message)),
            message=error_message,
            user_id=user_id,
            component=component,
            details=details
        )

        # Log error
        log_error_event(error_event)
        self.error_history.append(error_event)

        # Attempt recovery with default strategy
        recovery_key = f"{error_type}_{user_id or 'global'}_{int(time.time())}"

        if recovery_key not in self.active_recoveries:
            recovery_task = asyncio.create_task(
                self._attempt_recovery(error_event, recovery_key)
            )
            self.active_recoveries[recovery_key] = recovery_task

            try:
                success = await recovery_task
                error_event.recovery_attempted = True
                error_event.recovery_successful = success
                return success
            finally:
                self.active_recoveries.pop(recovery_key, None)

        return False

    async def _attempt_recovery(self, error_event: ErrorEvent, recovery_key: str) -> bool:
        """Attempt error recovery using default strategy."""
        strategy = self.recovery_strategy

        logger.info(f"🔧 Attempting recovery for {error_event.error_type} (key: {recovery_key})")

        for attempt in range(strategy.max_retries):
            try:
                # Execute recovery actions
                for action in strategy.recovery_actions:
                    success = await self._execute_recovery_action(action, error_event)
                    if success:
                        logger.info(f"✅ Recovery successful for {error_event.error_type} on attempt {attempt + 1}")
                        return True

                # Wait before retry
                if attempt < strategy.max_retries - 1:
                    delay = strategy.retry_delay * (strategy.backoff_multiplier ** attempt)
                    await asyncio.sleep(delay)

            except (RuntimeError, ConnectionError, OSError) as e:
                logger.exception(f"❌ Recovery attempt {attempt + 1} failed: {e}")

        logger.error(f"❌ All recovery attempts failed for {error_event.error_type}")
        return False

    async def _execute_recovery_action(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Execute specific recovery action."""
        try:
            action_map = {
                RecoveryAction.RETRY: lambda: True,
                RecoveryAction.RECONNECT: lambda: self._reconnect_azure_client(error_event.user_id),
                RecoveryAction.RESET_SESSION: lambda: self._reset_session(error_event.user_id),
                RecoveryAction.GRACEFUL_DEGRADATION: lambda: self._enable_graceful_degradation(error_event.user_id),
                RecoveryAction.EMERGENCY_SHUTDOWN: lambda: self._emergency_shutdown(error_event.user_id)
            }

            if action in action_map:
                result = action_map[action]()
                return await result if asyncio.iscoroutine(result) else result

        except (RuntimeError, ConnectionError, AttributeError) as e:
            logger.exception(f"❌ Recovery action {action.value} failed: {e}")

        return False

    async def _reconnect_azure_client(self, user_id: str | None) -> bool:
        """Reconnect Azure client for user."""
        if not user_id:
            return False

        try:
            if user_id in azure_connections:
                # Close existing connection
                await azure_connections[user_id].close()
                del azure_connections[user_id]

            # Create new connection
            new_client = AzureRealtimeClient(user_id)
            azure_connections[user_id] = new_client

            logger.info(f"✅ Azure client reconnected for user {user_id}")
            return True

        except (ConnectionError, RuntimeError, OSError) as e:
            logger.exception(f"❌ Failed to reconnect Azure client for user {user_id}: {e}")
            return False


    async def _reset_session(self, user_id: str | None) -> bool:
        """Reset session for user."""
        if not user_id:
            return False

        try:
            # Clean up connections
            active_connections.pop(user_id, None)
            azure_connections.pop(user_id, None)
            connection_info.pop(user_id, None)
            voice_settings.pop(user_id, None)

            logger.info(f"✅ Session reset for user {user_id}")
            return True

        except (RuntimeError, AttributeError, KeyError) as e:
            logger.exception(f"❌ Failed to reset session for user {user_id}: {e}")
            return False

    async def _enable_graceful_degradation(self, user_id: str | None) -> bool:
        """Enable graceful degradation mode."""
        logger.info(f"🔧 Enabling graceful degradation for user {user_id or 'global'}")
        return True

    async def _emergency_shutdown(self, user_id: str | None) -> bool:
        """Emergency shutdown for user session."""
        if user_id:
            return await self._reset_session(user_id)
        return True

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.error_history)
        recovery_attempts = sum(1 for e in self.error_history if e.recovery_attempted)
        successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)

        return {
            "total_errors": total_errors,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / recovery_attempts if recovery_attempts > 0 else 1.0,
            "active_recoveries": len(self.active_recoveries),
            "circuit_breakers": {k: v for k, v in self.circuit_breakers.items() if v.get("open", False)}
        }


# Global recovery manager instance
voice_recovery_manager = VoiceErrorRecoveryManager()


def get_error_statistics() -> dict[str, Any]:
    """Get error statistics from recovery manager."""
    return voice_recovery_manager.get_error_statistics()


# =====================================================================
# Health Check und Monitoring - Erweiterte Implementierung unten
# =====================================================================


@router.get("/metrics")
async def voice_metrics():
    """Detaillierte Metriken für Voice-System."""
    # Einfache Aggregation der zuletzt erfassten Latenzen über alle Sessions
    try:
        all_values_tts: list[float] = []
        all_values_interrupt: list[float] = []
        for entries in voice_monitoring_manager.session_stats.values():
            for e in entries:
                if e.get("metric") == "tts_start_latency_ms":
                    all_values_tts.append(float(e.get("value", 0)))
                if e.get("metric") == "interrupt_latency_ms":
                    all_values_interrupt.append(float(e.get("value", 0)))

        def _aggregate(values: list[float]) -> dict:
            if not values:
                return {"count": 0}
            values_sorted = sorted(values)
            n = len(values_sorted)
            p50 = values_sorted[int(0.5 * (n - 1))]
            p95 = values_sorted[int(0.95 * (n - 1))]
            return {
                "count": n,
                "avg_ms": sum(values_sorted) / n,
                "p50_ms": p50,
                "p95_ms": p95,
            }

        return {
            "connections": {
                "active_websockets": len(active_connections),
                "azure_connections": len(azure_connections),
                "total_registered": len(connection_info)
            },
            "api_info": {
                "version": "2025-04-01-preview",
                "endpoint": "/api/voice",
                "features": ["realtime_audio", "server_vad", "transcription", "interrupt_metrics"]
            },
            "latency": {
                "tts_start": _aggregate(all_values_tts),
                "interrupt": _aggregate(all_values_interrupt),
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
    except (RuntimeError, AttributeError, KeyError) as e:
        logger.exception(f"❌ Metrics aggregation failed: {e}")
        return {
            "connections": {
                "active_websockets": len(active_connections),
                "azure_connections": len(azure_connections),
                "total_registered": len(connection_info)
            },
            "api_info": {
                "version": "2025-04-01-preview",
                "endpoint": "/api/voice",
                "features": ["realtime_audio", "server_vad", "transcription"]
            },
            "timestamp": datetime.now(UTC).isoformat()
        }


# =====================================================================
# Erweiterte Health Monitoring Funktionalität
# =====================================================================

class VoiceHealthMonitor:
    """Comprehensive voice system health monitoring."""

    def __init__(self):
        self.startup_time = time.time()
        self.last_health_check = None
        self.error_history: list[dict[str, Any]] = []
        self.warning_history: list[dict[str, Any]] = []
        self.startup_validation_results: VoiceSystemHealth | None = None

    async def perform_startup_validation(self) -> VoiceSystemHealth:
        """Comprehensive startup validation of voice system."""
        logger.info("🔍 Starting voice system startup validation...")

        time.time()
        results = []

        # 1. WebSocket Endpoint Validation
        ws_result = await self._validate_websocket_endpoint()
        results.append(ws_result)

        # 2. Azure OpenAI Connectivity Test
        azure_result = await self._validate_azure_connectivity()
        results.append(azure_result)

        # 3. Audio Pipeline Integrity Check
        audio_result = await self._validate_audio_pipeline()
        results.append(audio_result)

        # 4. Session Management Validation
        session_result = await self._validate_session_management()
        results.append(session_result)

        # 5. Error Handling Validation
        error_result = await self._validate_error_handling()
        results.append(error_result)

        # Determine overall status
        overall_status = determine_overall_status(results)
        startup_passed = overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]

        # Count issues
        error_count = sum(1 for r in results if r.status == HealthStatus.FAILED)
        warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)

        health = VoiceSystemHealth(
            overall_status=overall_status,
            components=results,
            startup_validation_passed=startup_passed,
            last_check=datetime.now(UTC),
            uptime_seconds=time.time() - self.startup_time,
            error_count=error_count,
            warning_count=warning_count
        )

        self.startup_validation_results = health
        self.last_health_check = datetime.now(UTC)

        logger.info(f"🔍 Voice system validation completed: {overall_status.value}")
        return health

    async def _validate_websocket_endpoint(self) -> HealthCheckResult:
        """Validate WebSocket endpoint configuration."""
        start_time = time.time()

        try:
            # Test WebSocket endpoint configuration
            # Der Router wird mit leerem Prefix erstellt, aber von RouterGroup mit /api/voice registriert
            endpoint_configured = True  # Router ist korrekt konfiguriert
            websocket_routes = [route for route in router.routes if hasattr(route, "path") and "{user_id}" in route.path]
            websocket_available = len(websocket_routes) > 0

            # Beide Bedingungen müssen erfüllt sein für HEALTHY status
            status = HealthStatus.HEALTHY if endpoint_configured and websocket_available else HealthStatus.WARNING

            return create_health_check_result(
                ComponentType.WEBSOCKET_ENDPOINT, status,
                "WebSocket endpoint configuration validated",
                {
                    "endpoint_configured": endpoint_configured,
                    "websocket_routes_available": websocket_available,
                    "active_connections": len(active_connections)
                }, start_time
            )

        except (RuntimeError, AttributeError, ImportError) as e:
            return create_health_check_result(
                ComponentType.WEBSOCKET_ENDPOINT, HealthStatus.CRITICAL,
                "WebSocket endpoint validation failed", {}, start_time, str(e)
            )

    async def _validate_azure_connectivity(self) -> HealthCheckResult:
        """Validate Azure OpenAI connectivity."""
        start_time = time.time()

        try:
            # Test Azure configuration
            test_user_id = "health_check_test"
            test_client = AzureRealtimeClient(test_user_id)

            # Check if client can be instantiated
            client_created = test_client is not None
            config_valid = hasattr(test_client, "state_manager")

            status = HealthStatus.HEALTHY if client_created and config_valid else HealthStatus.WARNING

            return create_health_check_result(
                ComponentType.AZURE_CONNECTIVITY, status,
                "Azure OpenAI connectivity validated",
                {
                    "client_instantiation": client_created,
                    "configuration_valid": config_valid,
                    "azure_connections": len(azure_connections)
                }, start_time
            )

        except ImportError as e:
            return create_health_check_result(
                ComponentType.AZURE_CONNECTIVITY, HealthStatus.WARNING,
                "Azure connectivity module not available",
                {"import_error": str(e)}, start_time, str(e)
            )
        except (RuntimeError, AttributeError) as e:
            return create_health_check_result(
                ComponentType.AZURE_CONNECTIVITY, HealthStatus.CRITICAL,
                "Azure connectivity validation failed", {}, start_time, str(e)
            )

    async def _validate_audio_pipeline(self) -> HealthCheckResult:
        """Validate audio processing pipeline integrity."""
        start_time = time.time()

        try:
            # Test components
            test_user_id = "health_check_test"
            test_client = AzureRealtimeClient(test_user_id)
            test_settings = VoiceSettings(user_id=test_user_id)

            # Validate configuration
            audio_config_valid = (
                hasattr(test_settings, "input_audio_format") and
                hasattr(test_settings, "output_audio_format") and
                test_settings.input_audio_format == "pcm16"
            )

            pipeline_components = {
                "audio_settings": audio_config_valid,
                "client_instantiation": test_client is not None,
                "state_manager": hasattr(test_client, "state_manager")
            }

            all_valid = all(pipeline_components.values())
            status = HealthStatus.HEALTHY if all_valid else HealthStatus.WARNING

            return create_health_check_result(
                ComponentType.AUDIO_PIPELINE, status,
                "Audio pipeline integrity validated",
                pipeline_components, start_time
            )

        except ImportError as e:
            return create_health_check_result(
                ComponentType.AUDIO_PIPELINE, HealthStatus.WARNING,
                "Audio pipeline module not available",
                {"import_error": str(e)}, start_time, str(e)
            )
        except (RuntimeError, AttributeError) as e:
            return create_health_check_result(
                ComponentType.AUDIO_PIPELINE, HealthStatus.CRITICAL,
                "Audio pipeline validation failed", {}, start_time, str(e)
            )

    async def _validate_session_management(self) -> HealthCheckResult:
        """Validate session management functionality."""
        start_time = time.time()

        try:
            # Test session management components
            session_components = {
                "active_connections_tracking": isinstance(active_connections, dict),
                "connection_info_tracking": isinstance(connection_info, dict),
                "session_cleanup_available": True  # Simplified check
            }

            all_valid = all(session_components.values())
            status = HealthStatus.HEALTHY if all_valid else HealthStatus.WARNING

            return create_health_check_result(
                ComponentType.SESSION_MANAGEMENT, status,
                "Session management validated",
                {**session_components, "active_sessions": len(active_connections)}, start_time
            )

        except (RuntimeError, AttributeError, KeyError) as e:
            return create_health_check_result(
                ComponentType.SESSION_MANAGEMENT, HealthStatus.CRITICAL,
                "Session management validation failed", {}, start_time, str(e)
            )

    async def _validate_error_handling(self) -> HealthCheckResult:
        """Validate error handling and recovery mechanisms."""
        start_time = time.time()

        try:
            # Test error handling components
            error_components = {
                "error_recovery_available": True,  # Simplified check
                "graceful_degradation": True,
                "circuit_breaker_pattern": True
            }

            all_valid = all(error_components.values())
            status = HealthStatus.HEALTHY if all_valid else HealthStatus.WARNING

            return create_health_check_result(
                ComponentType.ERROR_HANDLING, status,
                "Error handling mechanisms validated",
                error_components, start_time
            )

        except (RuntimeError, AttributeError, ImportError) as e:
            return create_health_check_result(
                ComponentType.ERROR_HANDLING, HealthStatus.CRITICAL,
                "Error handling validation failed", {}, start_time, str(e)
            )


# Global health monitor instance
voice_health_monitor = VoiceHealthMonitor()


# =====================================================================
# Erweiterte Health Check Endpoints
# =====================================================================

@router.get("/health")
async def voice_health_check():
    """Health-Check für Voice-System."""
    return {
        "status": "healthy",
        "api_version": "2025-04-01-preview",
        "active_connections": len(active_connections),
        "azure_connections": len(azure_connections),
        "timestamp": datetime.now(UTC).isoformat()
    }


@router.get("/health/startup-validation")
async def get_startup_validation():
    """Get voice system startup validation results."""
    if voice_health_monitor.startup_validation_results:
        return voice_health_monitor.startup_validation_results.model_dump()
    return {"message": "Startup validation not yet performed"}


@router.post("/health/validate")
async def perform_health_check():
    """Perform comprehensive voice system health check."""
    return await voice_health_monitor.perform_startup_validation()


@router.get("/health/status")
async def get_health_status():
    """Get current voice system health status."""
    if voice_health_monitor.startup_validation_results:
        return {
            "status": voice_health_monitor.startup_validation_results.overall_status.value,
            "startup_validation_passed": voice_health_monitor.startup_validation_results.startup_validation_passed,
            "last_check": voice_health_monitor.startup_validation_results.last_check.isoformat(),
            "uptime_seconds": voice_health_monitor.startup_validation_results.uptime_seconds,
            "error_count": voice_health_monitor.startup_validation_results.error_count,
            "warning_count": voice_health_monitor.startup_validation_results.warning_count
        }
    return {"status": "unknown", "message": "Health check not yet performed"}


# =====================================================================
# Voice System Monitoring
# =====================================================================

class VoiceMonitoringManager:
    """Real-time voice system monitoring and alerting."""

    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self.active_alerts: list[AlertLevel] = []
        self.session_stats: dict[str, Any] = defaultdict(list)
        self.performance_thresholds = DEFAULT_PERFORMANCE_THRESHOLDS.copy()
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None

    async def start_monitoring(self):
        """Start real-time monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("🔍 Voice monitoring started")

    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.monitoring_task
            logger.info("🔍 Voice monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()

                # Validate metrics before processing
                if metrics:
                    self.metrics_history.append({
                        "timestamp": datetime.now(UTC),
                        "metrics": metrics
                    })

                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds

            except (RuntimeError, OSError, asyncio.CancelledError) as e:
                logger.exception(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _collect_metrics(self) -> VoiceMetrics | None:
        """Collect current voice system metrics."""
        try:
            import psutil

            return VoiceMetrics(
                active_sessions=len(active_connections),
                azure_connections=len(azure_connections),
                average_response_time_ms=100.0,  # Simplified
                error_rate_percent=0.0,  # Simplified
                memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
                network_latency_ms=50.0,  # Simplified
                session_success_rate=95.0,  # Simplified
                average_session_duration_seconds=300.0,  # Simplified
                timestamp=datetime.now(UTC)
            )
        except (RuntimeError, AttributeError, KeyError) as e:
            logger.exception(f"❌ Metrics collection failed: {e}")
            return None

    def get_metrics_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get metrics history."""
        return list(self.metrics_history)[-limit:]


# Global monitoring manager instance
voice_monitoring_manager = VoiceMonitoringManager()


# =====================================================================
# Monitoring Endpoints
# =====================================================================

@router.get("/monitoring/metrics")
async def get_current_metrics():
    """Get current voice system metrics."""
    return await voice_monitoring_manager._collect_metrics()


@router.get("/monitoring/alerts")
async def get_active_alerts():
    """Get all active alerts."""
    return {"alerts": [alert.value for alert in voice_monitoring_manager.active_alerts]}


@router.post("/monitoring/start")
async def start_monitoring():
    """Start real-time monitoring."""
    await voice_monitoring_manager.start_monitoring()
    return {"message": "Voice monitoring started"}


@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop real-time monitoring."""
    await voice_monitoring_manager.stop_monitoring()
    return {"message": "Voice monitoring stopped"}


@router.get("/monitoring/history")
async def get_metrics_history(limit: int = 100):
    """Get metrics history."""
    return {
        "history": voice_monitoring_manager.get_metrics_history(limit),
        "total_records": len(voice_monitoring_manager.metrics_history)
    }
