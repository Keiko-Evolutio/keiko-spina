"""Zentrale Konstanten für KEI-Stream.

Dieses Modul definiert alle Magic Numbers, Hard-coded Strings und
Default-Werte für das KEI-Stream-System an einem zentralen Ort.
"""

from __future__ import annotations

# ============================================================================
# QUOTA UND LIMITS
# ============================================================================

# Default-Werte für Stream-Quotas
DEFAULT_MAX_STREAMS = 64
DEFAULT_QUOTA_FALLBACK = 64

# Session-Management
DEFAULT_IDLE_TIMEOUT_MINUTES = 10
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 600  # 10 Minuten
DEFAULT_IDEMPOTENCY_MAX_IDS = 10_000

# Token Bucket Defaults
DEFAULT_TOKEN_BUCKET_CAPACITY = 100
DEFAULT_TOKEN_BUCKET_REFILL_RATE = 10.0

# ============================================================================
# TIMEOUTS UND INTERVALS
# ============================================================================

# Reconnect und TTL
DEFAULT_RECONNECT_TTL_SECONDS = 300  # 5 Minuten
DEFAULT_WEBSOCKET_TIMEOUT_SECONDS = 30
DEFAULT_GRPC_TIMEOUT_SECONDS = 30
DEFAULT_SSE_POLLING_INTERVAL_SECONDS = 1.0

# Backoff und Retry
DEFAULT_EXPONENTIAL_BACKOFF_BASE = 2.0
DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_INITIAL_RETRY_DELAY_SECONDS = 1.0

# ============================================================================
# SICHERHEIT UND VERSCHLÜSSELUNG
# ============================================================================

# Default-Secrets (MÜSSEN in Produktion überschrieben werden)
DEFAULT_RECONNECT_SECRET = "change-me-in-production"
DEFAULT_HMAC_SECRET = "change-me-in-production"

# DLP und Redaction
DEFAULT_REDACTION_MASK = "[REDACTED]"
DEFAULT_PII_REDACTION_ENABLED = True

# ============================================================================
# DATEISYSTEM UND STORAGE
# ============================================================================

# Chunk-Storage
DEFAULT_CHUNK_SINK = "memory"
DEFAULT_CHUNK_DIRECTORY = "/tmp/kei_stream_chunks"
DEFAULT_CHUNK_MAX_SIZE_BYTES = 1024 * 1024  # 1 MB
DEFAULT_CHUNK_TTL_SECONDS = 3600  # 1 Stunde

# ============================================================================
# KOMPRESSION
# ============================================================================

# WebSocket-Kompression
DEFAULT_WS_PERMESSAGE_DEFLATE = True

# gRPC-Kompression
DEFAULT_GRPC_COMPRESSION = "gzip"
SUPPORTED_GRPC_COMPRESSIONS = {"gzip", "deflate", "none", "off", "false"}

# ============================================================================
# METRIKEN UND OBSERVABILITY
# ============================================================================

# Prometheus Histogram Buckets
DURATION_BUCKETS_SECONDS = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
    300.0, 600.0, 1800.0, 3600.0, float("inf")
)

SIZE_BUCKETS_BYTES = (
    64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, float("inf")
)

CONNECTION_DURATION_BUCKETS = (
    1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0, float("inf")
)

POLLING_DURATION_BUCKETS = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf")
)

# ============================================================================
# KONVERTIERUNGS-KONSTANTEN
# ============================================================================

# Zeit-Konvertierungen
MILLISECONDS_PER_SECOND = 1000.0
MICROSECONDS_PER_SECOND = 1_000_000.0
NANOSECONDS_PER_SECOND = 1_000_000_000.0

# Größen-Konvertierungen
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

# ============================================================================
# UMGEBUNGSVARIABLEN-NAMEN
# ============================================================================

# KEI-Stream Basis-Konfiguration
ENV_KEI_STREAM_MAX_STREAMS = "KEI_STREAM_MAX_STREAMS"
ENV_KEI_STREAM_QUOTAS = "KEI_STREAM_QUOTAS"
ENV_KEI_STREAM_COMPRESSION = "KEI_STREAM_COMPRESSION"

# DLP und Sicherheit
ENV_KEI_STREAM_DLP_RULES = "KEI_STREAM_DLP_RULES"
ENV_KEI_STREAM_REDACTION_MASK = "KEI_STREAM_REDACTION_MASK"
ENV_KEI_STREAM_RECONNECT_SECRET = "KEI_STREAM_RECONNECT_SECRET"
ENV_KEI_STREAM_RECONNECT_TTL_SECS = "KEI_STREAM_RECONNECT_TTL_SECS"

# Chunk-Storage
ENV_KEI_STREAM_CHUNK_SINK = "KEI_STREAM_CHUNK_SINK"
ENV_KEI_STREAM_CHUNK_DIR = "KEI_STREAM_CHUNK_DIR"

# Kompression
ENV_KEI_STREAM_WS_PERMESSAGE_DEFLATE = "KEI_STREAM_WS_PERMESSAGE_DEFLATE"
ENV_KEI_RPC_DEFAULT_COMPRESSION = "KEI_RPC_DEFAULT_COMPRESSION"

# ============================================================================
# FRAME-TYPEN UND PROTOKOLL
# ============================================================================

# Standard Frame-Typen
FRAME_TYPE_PARTIAL = "partial"
FRAME_TYPE_FINAL = "final"
FRAME_TYPE_TOOL_CALL = "tool_call"
FRAME_TYPE_TOOL_RESULT = "tool_result"
FRAME_TYPE_STATUS = "status"
FRAME_TYPE_ERROR = "error"
FRAME_TYPE_HEARTBEAT = "heartbeat"
FRAME_TYPE_ACK = "ack"
FRAME_TYPE_NACK = "nack"
FRAME_TYPE_RESUME = "resume"
FRAME_TYPE_CHUNK_START = "chunk_start"
FRAME_TYPE_CHUNK_CONTINUE = "chunk_continue"
FRAME_TYPE_CHUNK_END = "chunk_end"

# ============================================================================
# VALIDIERUNGS-KONSTANTEN
# ============================================================================

# Maximale Werte für Validierung
MAX_SESSION_ID_LENGTH = 128
MAX_STREAM_ID_LENGTH = 128
MAX_TENANT_ID_LENGTH = 64
MAX_API_KEY_LENGTH = 256
MAX_FRAME_PAYLOAD_SIZE_BYTES = 10 * BYTES_PER_MB  # 10 MB
MAX_CORRELATION_ID_LENGTH = 128

# Minimale Werte
MIN_QUOTA_VALUE = 1
MIN_TTL_SECONDS = 1
MIN_TIMEOUT_SECONDS = 1

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BYTES_PER_GB",
    "BYTES_PER_KB",
    "BYTES_PER_MB",
    "CONNECTION_DURATION_BUCKETS",
    "DEFAULT_CHUNK_DIRECTORY",
    "DEFAULT_CHUNK_MAX_SIZE_BYTES",
    # Storage
    "DEFAULT_CHUNK_SINK",
    "DEFAULT_CHUNK_TTL_SECONDS",
    "DEFAULT_EXPONENTIAL_BACKOFF_BASE",
    "DEFAULT_GRPC_COMPRESSION",
    "DEFAULT_GRPC_TIMEOUT_SECONDS",
    "DEFAULT_HMAC_SECRET",
    "DEFAULT_IDEMPOTENCY_MAX_IDS",
    "DEFAULT_IDEMPOTENCY_TTL_SECONDS",
    "DEFAULT_IDLE_TIMEOUT_MINUTES",
    "DEFAULT_INITIAL_RETRY_DELAY_SECONDS",
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    # Quota und Limits
    "DEFAULT_MAX_STREAMS",
    "DEFAULT_PII_REDACTION_ENABLED",
    "DEFAULT_QUOTA_FALLBACK",
    # Sicherheit
    "DEFAULT_RECONNECT_SECRET",
    # Timeouts und Intervals
    "DEFAULT_RECONNECT_TTL_SECONDS",
    "DEFAULT_REDACTION_MASK",
    "DEFAULT_SSE_POLLING_INTERVAL_SECONDS",
    "DEFAULT_TOKEN_BUCKET_CAPACITY",
    "DEFAULT_TOKEN_BUCKET_REFILL_RATE",
    "DEFAULT_WEBSOCKET_TIMEOUT_SECONDS",
    # Kompression
    "DEFAULT_WS_PERMESSAGE_DEFLATE",
    # Metriken
    "DURATION_BUCKETS_SECONDS",
    "ENV_KEI_RPC_DEFAULT_COMPRESSION",
    "ENV_KEI_STREAM_CHUNK_DIR",
    "ENV_KEI_STREAM_CHUNK_SINK",
    "ENV_KEI_STREAM_COMPRESSION",
    "ENV_KEI_STREAM_DLP_RULES",
    # Umgebungsvariablen
    "ENV_KEI_STREAM_MAX_STREAMS",
    "ENV_KEI_STREAM_QUOTAS",
    "ENV_KEI_STREAM_RECONNECT_SECRET",
    "ENV_KEI_STREAM_RECONNECT_TTL_SECS",
    "ENV_KEI_STREAM_REDACTION_MASK",
    "ENV_KEI_STREAM_WS_PERMESSAGE_DEFLATE",
    "FRAME_TYPE_ACK",
    "FRAME_TYPE_CHUNK_CONTINUE",
    "FRAME_TYPE_CHUNK_END",
    "FRAME_TYPE_CHUNK_START",
    "FRAME_TYPE_ERROR",
    "FRAME_TYPE_FINAL",
    "FRAME_TYPE_HEARTBEAT",
    "FRAME_TYPE_NACK",
    # Frame-Typen
    "FRAME_TYPE_PARTIAL",
    "FRAME_TYPE_RESUME",
    "FRAME_TYPE_STATUS",
    "FRAME_TYPE_TOOL_CALL",
    "FRAME_TYPE_TOOL_RESULT",
    "MAX_API_KEY_LENGTH",
    "MAX_CORRELATION_ID_LENGTH",
    "MAX_FRAME_PAYLOAD_SIZE_BYTES",
    # Validierung
    "MAX_SESSION_ID_LENGTH",
    "MAX_STREAM_ID_LENGTH",
    "MAX_TENANT_ID_LENGTH",
    "MICROSECONDS_PER_SECOND",
    # Konvertierungen
    "MILLISECONDS_PER_SECOND",
    "MIN_QUOTA_VALUE",
    "MIN_TIMEOUT_SECONDS",
    "MIN_TTL_SECONDS",
    "NANOSECONDS_PER_SECOND",
    "POLLING_DURATION_BUCKETS",
    "SIZE_BUCKETS_BYTES",
    "SUPPORTED_GRPC_COMPRESSIONS",
]
