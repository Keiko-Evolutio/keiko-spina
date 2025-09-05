"""WebRTC Types und Datenstrukturen

Definiert alle Datentypen für WebRTC-Integration im Backend.
Kompatibel mit Frontend WebRTC Types für einheitliche API.

@version 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field

# =============================================================================
# WebRTC Session States
# =============================================================================

class WebRTCSessionState(str, Enum):
    """WebRTC Session States."""
    CREATED = "created"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"

class SignalingState(str, Enum):
    """WebRTC Signaling States."""
    STABLE = "stable"
    HAVE_LOCAL_OFFER = "have-local-offer"
    HAVE_REMOTE_OFFER = "have-remote-offer"
    HAVE_LOCAL_PRANSWER = "have-local-pranswer"
    HAVE_REMOTE_PRANSWER = "have-remote-pranswer"
    CLOSED = "closed"

class IceConnectionState(str, Enum):
    """ICE Connection States."""
    NEW = "new"
    CHECKING = "checking"
    CONNECTED = "connected"
    COMPLETED = "completed"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    CLOSED = "closed"

class IceGatheringState(str, Enum):
    """ICE Gathering States."""
    NEW = "new"
    GATHERING = "gathering"
    COMPLETE = "complete"

# =============================================================================
# Signaling Message Types
# =============================================================================

class SignalingMessageType(str, Enum):
    """Signaling Message Types."""
    OFFER = "offer"
    ANSWER = "answer"
    ICE_CANDIDATE = "ice-candidate"
    ICE_CANDIDATE_ERROR = "ice-candidate-error"
    CONNECTION_STATE_CHANGE = "connection-state-change"
    AUDIO_STATE_CHANGE = "audio-state-change"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

# =============================================================================
# Signaling Messages
# =============================================================================

class BaseSignalingMessage(BaseModel):
    """Basis Signaling Message."""
    type: SignalingMessageType
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = None

class OfferMessage(BaseSignalingMessage):
    """SDP Offer Message."""
    type: SignalingMessageType = SignalingMessageType.OFFER
    offer: dict[str, Any]  # SDP Offer
    audio_constraints: dict[str, Any] | None = None

class AnswerMessage(BaseSignalingMessage):
    """SDP Answer Message."""
    type: SignalingMessageType = SignalingMessageType.ANSWER
    answer: dict[str, Any]  # SDP Answer

class IceCandidateMessage(BaseSignalingMessage):
    """ICE Candidate Message."""
    type: SignalingMessageType = SignalingMessageType.ICE_CANDIDATE
    candidate: dict[str, Any]  # ICE Candidate

class ErrorMessage(BaseSignalingMessage):
    """Error Message."""
    type: SignalingMessageType = SignalingMessageType.ERROR
    error_code: str
    error_message: str
    error_details: dict[str, Any] | None = None

# Union Type für alle Signaling Messages
SignalingMessage = Union[
    OfferMessage,
    AnswerMessage,
    IceCandidateMessage,
    ErrorMessage,
    BaseSignalingMessage
]

# =============================================================================
# WebRTC Configuration
# =============================================================================

@dataclass
class IceServer:
    """ICE Server Konfiguration."""
    urls: list[str]
    username: str | None = None
    credential: str | None = None

@dataclass
class AudioCodec:
    """Audio Codec Konfiguration."""
    codec: str
    priority: int
    bitrate: int | None = None
    parameters: dict[str, str | int] = field(default_factory=dict)

@dataclass
class WebRTCConfiguration:
    """WebRTC Konfiguration."""
    ice_servers: list[IceServer]
    ice_transport_policy: str = "all"
    bundle_policy: str = "max-bundle"
    audio_codecs: list[AudioCodec] = field(default_factory=list)
    dtls_fingerprint: str | None = None

# =============================================================================
# Session Management
# =============================================================================

@dataclass
class WebRTCPeer:
    """WebRTC Peer Information."""
    user_id: str
    session_id: str
    connection_id: str
    ip_address: str
    user_agent: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass
class WebRTCSession:
    """WebRTC Session."""
    session_id: str
    initiator: WebRTCPeer
    responder: WebRTCPeer | None = None
    state: WebRTCSessionState = WebRTCSessionState.CREATED
    signaling_state: SignalingState = SignalingState.STABLE
    ice_connection_state: IceConnectionState = IceConnectionState.NEW
    ice_gathering_state: IceGatheringState = IceGatheringState.NEW
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    connected_at: datetime | None = None
    disconnected_at: datetime | None = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class AudioQualityMetrics:
    """Audio-Qualität Metriken."""
    bitrate: float  # kbps
    packet_loss_rate: float  # 0-1
    round_trip_time: float  # ms
    jitter: float  # ms
    audio_level: float  # 0-1
    codec: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass
class WebRTCMetrics:
    """WebRTC Performance Metriken."""
    session_id: str
    connection_setup_time: float  # ms
    ice_gathering_time: float  # ms
    dtls_handshake_time: float  # ms
    audio_quality: AudioQualityMetrics
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    packets_lost: int
    current_latency: float  # ms
    average_latency: float  # ms
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

# =============================================================================
# Event Types
# =============================================================================

@dataclass
class WebRTCEvent:
    """Basis WebRTC Event."""
    event_type: str
    session_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectionStateChangeEvent(WebRTCEvent):
    """Connection State Change Event."""
    event_type: str = "connection-state-change"
    previous_state: WebRTCSessionState = WebRTCSessionState.CREATED
    new_state: WebRTCSessionState = WebRTCSessionState.CREATED

@dataclass
class AudioTrackEvent(WebRTCEvent):
    """Audio Track Event."""
    event_type: str = "audio-track"
    track_id: str = ""
    track_kind: str = "audio"
    action: str = "added"  # added, removed

@dataclass
class MetricsUpdateEvent(WebRTCEvent):
    """Metrics Update Event."""
    event_type: str = "metrics-update"
    metrics: WebRTCMetrics = field(default_factory=lambda: WebRTCMetrics(
        session_id="",
        connection_setup_time=0,
        ice_gathering_time=0,
        dtls_handshake_time=0,
        audio_quality=AudioQualityMetrics(
            bitrate=0, packet_loss_rate=0, round_trip_time=0,
            jitter=0, audio_level=0, codec=""
        ),
        bytes_sent=0, bytes_received=0, packets_sent=0,
        packets_received=0, packets_lost=0, current_latency=0,
        average_latency=0
    ))

# =============================================================================
# Error Types
# =============================================================================

class WebRTCError(Exception):
    """Basis WebRTC Error."""

    def __init__(
        self,
        message: str,
        code: str = "WEBRTC_ERROR",
        context: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.now(UTC)

class WebRTCSignalingError(WebRTCError):
    """WebRTC Signaling Error."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message, "SIGNALING_ERROR", context)

class WebRTCSessionError(WebRTCError):
    """WebRTC Session Error."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message, "SESSION_ERROR", context)

class WebRTCConnectionError(WebRTCError):
    """WebRTC Connection Error."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message, "CONNECTION_ERROR", context)

class WebRTCAudioError(WebRTCError):
    """WebRTC Audio Error."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message, "AUDIO_ERROR", context)

# =============================================================================
# Utility Functions
# =============================================================================

def create_session_id() -> str:
    """Erstellt eine eindeutige Session ID."""
    import uuid
    return f"webrtc-{uuid.uuid4().hex[:16]}"

def create_connection_id() -> str:
    """Erstellt eine eindeutige Connection ID."""
    import uuid
    return f"conn-{uuid.uuid4().hex[:12]}"

def validate_signaling_message(message_data: dict[str, Any]) -> bool:
    """Validiert eine Signaling Message."""
    required_fields = ["type", "session_id", "user_id", "timestamp"]
    return all(field in message_data for field in required_fields)

def parse_signaling_message(message_data: dict[str, Any]) -> SignalingMessage:
    """Parst eine Signaling Message aus Dictionary."""
    message_type = message_data.get("type")

    if message_type == SignalingMessageType.OFFER:
        return OfferMessage(**message_data)
    if message_type == SignalingMessageType.ANSWER:
        return AnswerMessage(**message_data)
    if message_type == SignalingMessageType.ICE_CANDIDATE:
        return IceCandidateMessage(**message_data)
    if message_type == SignalingMessageType.ERROR:
        return ErrorMessage(**message_data)
    return BaseSignalingMessage(**message_data)

def session_to_dict(session: WebRTCSession) -> dict[str, Any]:
    """Konvertiert WebRTC Session zu Dictionary."""
    return {
        "session_id": session.session_id,
        "initiator": {
            "user_id": session.initiator.user_id,
            "connection_id": session.initiator.connection_id,
            "ip_address": session.initiator.ip_address
        },
        "responder": {
            "user_id": session.responder.user_id,
            "connection_id": session.responder.connection_id,
            "ip_address": session.responder.ip_address
        } if session.responder else None,
        "state": session.state.value,
        "signaling_state": session.signaling_state.value,
        "ice_connection_state": session.ice_connection_state.value,
        "ice_gathering_state": session.ice_gathering_state.value,
        "created_at": session.created_at.isoformat(),
        "connected_at": session.connected_at.isoformat() if session.connected_at else None,
        "disconnected_at": session.disconnected_at.isoformat() if session.disconnected_at else None,
        "last_activity": session.last_activity.isoformat(),
        "metadata": session.metadata
    }
