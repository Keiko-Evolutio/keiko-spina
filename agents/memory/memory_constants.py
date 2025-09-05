"""Konstanten für Memory-Module."""

from __future__ import annotations

from typing import Final

CHAT_MESSAGE_CATEGORY: Final[str] = "chat_message"

DEFAULT_MAX_MESSAGES: Final[int] = 200
DEFAULT_MAX_AGE_HOURS: Final[int] = 168
MIN_RETENTION_HOURS: Final[int] = 24
MAX_RETENTION_HOURS: Final[int] = 8760

CLEANUP_BATCH_SIZE: Final[int] = 100
MAX_CLEANUP_ITERATIONS: Final[int] = 10

CHECKPOINT_ID_PREFIX: Final[str] = "cp-"
DEFAULT_CHECKPOINT_ID: Final[str] = "cp-1"
DEFAULT_THREAD_ID: Final[str] = "default"

MAX_CHECKPOINTS_PER_THREAD: Final[int] = 1000
CHECKPOINT_CLEANUP_THRESHOLD: Final[int] = 800

# Chat Message Queries
CHAT_MESSAGES_LOAD_QUERY: Final[str] = (
    "SELECT c.role, c.content, c.created_at FROM c "
    "WHERE c.category = @cat AND c.session_id = @sid "
    "ORDER BY c.created_at ASC"
)

CHAT_MESSAGES_DELETE_OLD_QUERY: Final[str] = (
    "SELECT c.id FROM c WHERE c.category=@cat AND c.session_id=@sid AND c.created_at < @cut"
)

CHAT_MESSAGES_COUNT_QUERY: Final[str] = (
    "SELECT VALUE COUNT(1) FROM c WHERE c.category=@cat AND c.session_id=@sid"
)

CHAT_MESSAGES_DELETE_EXCESS_QUERY: Final[str] = (
    "SELECT c.id FROM c WHERE c.category=@cat AND c.session_id=@sid ORDER BY c.created_at ASC"
)

# Checkpoint Queries
CHECKPOINTS_LOAD_QUERY: Final[str] = (
    "SELECT c.id, c.thread_id, c.created_at, c.state, c.metadata "
    "FROM c WHERE c.category = @cat AND c.thread_id = @tid ORDER BY c.created_at ASC"
)

CHECKPOINTS_DELETE_QUERY: Final[str] = (
    "SELECT c.id FROM c WHERE c.category = @cat AND c.thread_id = @tid"
)

# =============================================================================
# ERROR MESSAGES
# =============================================================================

# Chat Memory Error Messages
CHAT_MEMORY_LOAD_ERROR: Final[str] = "CosmosChatMemory load_messages Fehler"
CHAT_MEMORY_APPEND_ERROR: Final[str] = "CosmosChatMemory append_messages Fehler"
CHAT_MEMORY_CLEANUP_ERROR: Final[str] = "Cleanup Fehler"
CHAT_MEMORY_DELETE_ERROR: Final[str] = "Cleanup Delete fehlgeschlagen"
CHAT_MEMORY_EXCESS_DELETE_ERROR: Final[str] = "Excess Delete fehlgeschlagen"

# Checkpoint Error Messages
CHECKPOINT_PERSIST_ERROR: Final[str] = "Checkpoint Persist-Start fehlgeschlagen"
CHECKPOINT_COSMOS_PERSIST_ERROR: Final[str] = "Persist nach Cosmos fehlgeschlagen"
CHECKPOINT_LOAD_ERROR: Final[str] = "Cosmos-Ladevorgang fehlgeschlagen"
CHECKPOINT_CLEAR_ERROR: Final[str] = "Cosmos-Clear fehlgeschlagen"
CHECKPOINT_DELETE_ERROR: Final[str] = "Cosmos Delete Checkpoints fehlgeschlagen"
CHECKPOINT_UPSERT_ERROR: Final[str] = "Cosmos Upsert fehlgeschlagen"
CHECKPOINT_DELETE_SINGLE_ERROR: Final[str] = "Delete einzelner Checkpoint fehlgeschlagen"

# General Error Messages
LANGGRAPH_UNAVAILABLE_ERROR: Final[str] = "langgraph MemorySaver nicht verfügbar"
ASYNC_IN_RUNNING_LOOP_ERROR: Final[str] = (
    "Kann Coroutine nicht synchron im laufenden Event Loop ausführen"
)

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Batch Processing
DEFAULT_BATCH_SIZE: Final[int] = 50
MAX_BATCH_SIZE: Final[int] = 500

# Timeout Settings
COSMOS_OPERATION_TIMEOUT: Final[int] = 30  # Sekunden
ASYNC_OPERATION_TIMEOUT: Final[int] = 10   # Sekunden

# Memory Pressure
MEMORY_PRESSURE_THRESHOLD: Final[float] = 0.85
GC_INTERVAL_SECONDS: Final[int] = 60

# =============================================================================
# VALIDATION LIMITS
# =============================================================================

# Content Limits
MAX_MESSAGE_CONTENT_LENGTH: Final[int] = 100_000  # 100KB
MAX_METADATA_SIZE: Final[int] = 10_000            # 10KB
MAX_SESSION_ID_LENGTH: Final[int] = 255
MAX_THREAD_ID_LENGTH: Final[int] = 255

# Role Validation
VALID_MESSAGE_ROLES: Final[frozenset[str]] = frozenset({
    "user", "assistant", "system", "function", "tool"
})

DEFAULT_MESSAGE_ROLE: Final[str] = "user"

__all__ = [
    "ASYNC_IN_RUNNING_LOOP_ERROR",
    "ASYNC_OPERATION_TIMEOUT",
    "CHAT_MEMORY_APPEND_ERROR",
    "CHAT_MEMORY_CLEANUP_ERROR",
    "CHAT_MEMORY_DELETE_ERROR",
    "CHAT_MEMORY_EXCESS_DELETE_ERROR",
    # Error Messages
    "CHAT_MEMORY_LOAD_ERROR",
    "CHAT_MESSAGES_COUNT_QUERY",
    "CHAT_MESSAGES_DELETE_EXCESS_QUERY",
    "CHAT_MESSAGES_DELETE_OLD_QUERY",
    # Queries
    "CHAT_MESSAGES_LOAD_QUERY",
    # Categories
    "CHAT_MESSAGE_CATEGORY",
    "CHECKPOINTS_DELETE_QUERY",
    "CHECKPOINTS_LOAD_QUERY",
    "CHECKPOINT_CLEANUP_THRESHOLD",
    "CHECKPOINT_CLEAR_ERROR",
    "CHECKPOINT_COSMOS_PERSIST_ERROR",
    "CHECKPOINT_DELETE_ERROR",
    "CHECKPOINT_DELETE_SINGLE_ERROR",
    # Checkpoints
    "CHECKPOINT_ID_PREFIX",
    "CHECKPOINT_LOAD_ERROR",
    "CHECKPOINT_PERSIST_ERROR",
    "CHECKPOINT_UPSERT_ERROR",
    "CLEANUP_BATCH_SIZE",
    "COSMOS_OPERATION_TIMEOUT",
    # Performance
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHECKPOINT_ID",
    "DEFAULT_MAX_AGE_HOURS",
    # Retention
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MESSAGE_ROLE",
    "DEFAULT_THREAD_ID",
    "GC_INTERVAL_SECONDS",
    "LANGGRAPH_UNAVAILABLE_ERROR",
    "MAX_BATCH_SIZE",
    "MAX_CHECKPOINTS_PER_THREAD",
    "MAX_CLEANUP_ITERATIONS",
    # Validation
    "MAX_MESSAGE_CONTENT_LENGTH",
    "MAX_METADATA_SIZE",
    "MAX_RETENTION_HOURS",
    "MAX_SESSION_ID_LENGTH",
    "MAX_THREAD_ID_LENGTH",
    "MEMORY_PRESSURE_THRESHOLD",
    "MIN_RETENTION_HOURS",
    "VALID_MESSAGE_ROLES",
]
