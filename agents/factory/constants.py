# backend/agents/factory/constants.py
"""Zentrale Constants und Enums für das Factory-Modul.

Konsolidiert alle Magic Numbers und Hard-coded Strings für bessere Wartbarkeit
und Einhaltung der Clean Code Prinzipien.
"""
from __future__ import annotations

from enum import Enum
from typing import Final

# =============================================================================
# Performance und Timing Constants
# =============================================================================

# Retry und Delay Settings
DEFAULT_RETRY_DELAY: Final[float] = 0.01
DEFAULT_TIMEOUT: Final[float] = 30.0
MAX_RETRY_ATTEMPTS: Final[int] = 3
EXPONENTIAL_BACKOFF_MULTIPLIER: Final[float] = 2.0

# Performance Limits
MAX_REQUEST_COUNT: Final[int] = 2_147_483_647  # Max int32 für Overflow-Schutz
DEFAULT_LATENCY_THRESHOLD: Final[float] = 1.0  # Sekunden
PERFORMANCE_SAMPLE_SIZE: Final[int] = 100

# =============================================================================
# Cache und Memory Management
# =============================================================================

# Cache Settings
DEFAULT_CACHE_TTL: Final[int] = 3600  # 1 Stunde in Sekunden
MAX_CACHE_SIZE: Final[int] = 1000
CACHE_CLEANUP_INTERVAL: Final[int] = 300  # 5 Minuten

# Memory Limits
MAX_AGENT_INSTANCES: Final[int] = 50
MAX_MCP_CLIENTS: Final[int] = 100
MAX_SESSION_COUNT: Final[int] = 200

# =============================================================================
# Network und Transport Constants
# =============================================================================

# URLs und Endpoints
MOCK_SERVER_URL: Final[str] = "http://mock-server.com"
DEFAULT_MCP_ENDPOINT: Final[str] = "http://localhost:8080/mcp"
HEALTH_CHECK_ENDPOINT: Final[str] = "/health"

# Transport Settings
DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10
MAX_CONCURRENT_CONNECTIONS: Final[int] = 50
CONNECTION_TIMEOUT: Final[float] = 10.0
READ_TIMEOUT: Final[float] = 30.0

# =============================================================================
# Agent und Framework Constants
# =============================================================================

# Supported Frameworks
SUPPORTED_FRAMEWORKS: Final[tuple[str, ...]] = (
    "azure_foundry",
    "openai_assistant",
    "langchain",
    "custom"
)

# Default Framework
DEFAULT_FRAMEWORK: Final[str] = "azure_foundry"

# Agent Types
AGENT_TYPE_AZURE: Final[str] = "azure_foundry"
AGENT_TYPE_CUSTOM: Final[str] = "custom"
AGENT_TYPE_MCP: Final[str] = "mcp"

# =============================================================================
# Error und Logging Constants
# =============================================================================

# Error Messages
ERROR_AGENT_NOT_FOUND: Final[str] = "Agent nicht gefunden"
ERROR_SESSION_NOT_FOUND: Final[str] = "Session nicht gefunden"
ERROR_MCP_CLIENT_UNAVAILABLE: Final[str] = "MCP Client nicht verfügbar"
ERROR_FACTORY_NOT_INITIALIZED: Final[str] = "Factory nicht initialisiert"
ERROR_INVALID_FRAMEWORK: Final[str] = "Ungültiges Framework"

# Log Levels und Messages
LOG_AGENT_CREATED: Final[str] = "Agent erfolgreich erstellt"
LOG_AGENT_CLEANUP: Final[str] = "Agent-Cleanup durchgeführt"
LOG_MCP_CLIENT_CONNECTED: Final[str] = "MCP Client verbunden"
LOG_SESSION_INITIALIZED: Final[str] = "Session initialisiert"

# =============================================================================
# Feature Flags
# =============================================================================

# Feature Availability
DEFAULT_FEATURE_FLAGS: Final[dict[str, bool]] = {
    "mcp_client": True,
    "azure_foundry": True,
    "performance_metrics": True,
    "caching": True,
    "health_monitoring": True,
}

# =============================================================================
# Enums für typisierte Constants
# =============================================================================

class AgentFramework(str, Enum):
    """Unterstützte Agent-Frameworks."""
    AZURE_FOUNDRY = "azure_foundry"
    OPENAI_ASSISTANT = "openai_assistant"
    LANGCHAIN = "langchain"
    CUSTOM = "custom"


class FactoryState(str, Enum):
    """Factory-Zustände."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class ClientState(str, Enum):
    """Client-Zustände."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class MetricType(str, Enum):
    """Performance-Metric-Typen."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    CACHE_HIT_RATE = "cache_hit_rate"


class LogLevel(str, Enum):
    """Log-Level für strukturiertes Logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorSeverity(str, Enum):
    """Error-Severity-Level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Validation Constants
# =============================================================================

# String Validation
MIN_AGENT_ID_LENGTH: Final[int] = 3
MAX_AGENT_ID_LENGTH: Final[int] = 100
MIN_PROJECT_ID_LENGTH: Final[int] = 3
MAX_PROJECT_ID_LENGTH: Final[int] = 100

# Numeric Validation
MIN_PORT_NUMBER: Final[int] = 1
MAX_PORT_NUMBER: Final[int] = 65535
MIN_TIMEOUT_SECONDS: Final[float] = 0.1
MAX_TIMEOUT_SECONDS: Final[float] = 300.0

# =============================================================================
# Export für einfachen Import
# =============================================================================

__all__ = [
    "AGENT_TYPE_AZURE",
    "AGENT_TYPE_CUSTOM",
    "AGENT_TYPE_MCP",
    "CACHE_CLEANUP_INTERVAL",
    "CONNECTION_TIMEOUT",
    # Cache Constants
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CONNECTION_POOL_SIZE",
    # Feature Flags
    "DEFAULT_FEATURE_FLAGS",
    "DEFAULT_FRAMEWORK",
    "DEFAULT_LATENCY_THRESHOLD",
    "DEFAULT_MCP_ENDPOINT",
    # Performance Constants
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_TIMEOUT",
    # Error Constants
    "ERROR_AGENT_NOT_FOUND",
    "ERROR_FACTORY_NOT_INITIALIZED",
    "ERROR_INVALID_FRAMEWORK",
    "ERROR_MCP_CLIENT_UNAVAILABLE",
    "ERROR_SESSION_NOT_FOUND",
    "EXPONENTIAL_BACKOFF_MULTIPLIER",
    "HEALTH_CHECK_ENDPOINT",
    "LOG_AGENT_CLEANUP",
    # Log Constants
    "LOG_AGENT_CREATED",
    "LOG_MCP_CLIENT_CONNECTED",
    "LOG_SESSION_INITIALIZED",
    "MAX_AGENT_ID_LENGTH",
    "MAX_AGENT_INSTANCES",
    "MAX_CACHE_SIZE",
    "MAX_CONCURRENT_CONNECTIONS",
    "MAX_MCP_CLIENTS",
    "MAX_PORT_NUMBER",
    "MAX_PROJECT_ID_LENGTH",
    "MAX_REQUEST_COUNT",
    "MAX_RETRY_ATTEMPTS",
    "MAX_SESSION_COUNT",
    "MAX_TIMEOUT_SECONDS",
    # Validation Constants
    "MIN_AGENT_ID_LENGTH",
    "MIN_PORT_NUMBER",
    "MIN_PROJECT_ID_LENGTH",
    "MIN_TIMEOUT_SECONDS",
    # Network Constants
    "MOCK_SERVER_URL",
    "PERFORMANCE_SAMPLE_SIZE",
    "READ_TIMEOUT",
    # Agent Constants
    "SUPPORTED_FRAMEWORKS",
    # Enums
    "AgentFramework",
    "ClientState",
    "ErrorSeverity",
    "FactoryState",
    "LogLevel",
    "MetricType",
]
