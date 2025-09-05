"""Streaming Service Package.

Migriert aus kei_stream/ nach services/streaming/ als Teil der kei_*-Module-Konsolidierung.
Konsolidiert auch websocket/ Modul für einheitliche Streaming-Architektur.

Enthält Frame-Schemata, Session-Management, sowie Transports für
WebSocket, gRPC und read-only SSE.

Hauptkomponenten:
- SSE (Server-Sent Events) Transport für Real-time-Updates
- WebSocket-Transport mit mTLS-Unterstützung
- gRPC-Streaming für bidirektionale Kommunikation
- Compression-Policies und Token-Bucket-Rate-Limiting
- Real-time-Metrics und Session-Management
- WebSocket-Manager und Event-Handling (konsolidiert aus websocket/)
"""

# KEI-Stream Core Components
from .frames import (
    AckInfo,
    ChunkInfo,
    ErrorInfo,
    FrameType,
    KEIStreamFrame,
    make_ack,
    make_error,
)
from .session import SessionContext, SessionManager, session_manager

# WebSocket Models (conditional import)
try:
    from data_models.websocket import (
        AgentResponseEvent,
        ConnectionStatusEvent,
        ErrorEvent,
        EventType,
        FunctionCallEvent,
        StatusUpdateEvent,
        VoiceInputEvent,
        WebSocketEvent,
    )
    _WEBSOCKET_MODELS_AVAILABLE = True
except ImportError:
    _WEBSOCKET_MODELS_AVAILABLE = False

# WebSocket Management (konsolidiert aus websocket/)
from .handlers import (
    handle_function_confirmation,
    handle_voice_input,
    register_default_handlers,
)
from .manager import (
    WebSocketConnection,
    WebSocketManager,
    websocket_manager,
)
from .router import router
from .utils import (
    BaseEventHandler,
    EventHandlerFunction,
    EventRouter,
    FunctionEventHandler,
    create_pong_response,
    create_server_timestamp,
    extract_client_time,
    handle_ping_message,
    is_ping_event,
)

# Transport Components (conditional imports)
try:
    from .sse_transport import SSETransport
    _SSE_AVAILABLE = True
except ImportError:
    _SSE_AVAILABLE = False

try:
    from .websocket_transport import WebSocketTransport
    _WS_TRANSPORT_AVAILABLE = True
except ImportError:
    _WS_TRANSPORT_AVAILABLE = False

try:
    from .grpc_transport import KEIStreamService
    _GRPC_TRANSPORT_AVAILABLE = True
except ImportError:
    _GRPC_TRANSPORT_AVAILABLE = False

# Additional Components
try:
    from .compression_policies import CompressionPolicy
    from .token_bucket import TokenBucket
    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False

__all__ = [
    # KEI-Stream Core
    "AckInfo",
    "ChunkInfo",
    "ErrorInfo",
    "FrameType",
    "KEIStreamFrame",
    "SessionContext",
    "SessionManager",
    "make_ack",
    "make_error",
    "session_manager",

    # WebSocket Management (konsolidiert)
    "BaseEventHandler",
    "EventHandlerFunction",
    "EventRouter",
    "FunctionEventHandler",
    "WebSocketConnection",
    "WebSocketManager",
    "websocket_manager",

    # Event Handlers
    "handle_function_confirmation",
    "handle_voice_input",
    "register_default_handlers",

    # Router
    "router",

    # Utils
    "create_pong_response",
    "create_server_timestamp",
    "extract_client_time",
    "handle_ping_message",
    "is_ping_event",
]

# Conditional exports
if _WEBSOCKET_MODELS_AVAILABLE:
    __all__.extend([
        "AgentResponseEvent",
        "ConnectionStatusEvent",
        "ErrorEvent",
        "EventType",
        "FunctionCallEvent",
        "StatusUpdateEvent",
        "VoiceInputEvent",
        "WebSocketEvent",
    ])

if _SSE_AVAILABLE:
    __all__.append("SSETransport")

if _WS_TRANSPORT_AVAILABLE:
    __all__.append("WebSocketTransport")

if _GRPC_TRANSPORT_AVAILABLE:
    __all__.append("KEIStreamService")

if _ADVANCED_AVAILABLE:
    __all__.extend(["CompressionPolicy", "TokenBucket"])

# Paket-Metadaten
__version__ = "0.1.0"
__author__ = "Keiko Development Team"
