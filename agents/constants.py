# backend/agents/constants.py
"""Konsolidierte Konstanten für das Agents-Modul.

Zentrale Definition aller gemeinsamen Magic Numbers, Timeouts und Konfigurationswerte
für bessere Wartbarkeit und Konsistenz. Konsolidiert aus:
- kei_agents/shared_constants.py
- agents/agents_constants.py
- agents/base_agent_constants.py
- Verschiedene *_constants.py Dateien aus Submodulen

Eliminiert alle Code-Duplikate zwischen den ursprünglichen Constants-Dateien.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final

# ============================================================================
# FRAMEWORK-ENUMS (Konsolidiert aus kei_agents/shared_constants.py)
# ============================================================================

class AgentFramework(Enum):
    """Unterstützte Agent-Frameworks."""
    FOUNDRY = "foundry"
    AUTOGEN = "autogen"
    SEMANTIC_KERNEL = "semantic_kernel"
    CUSTOM = "custom"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"


class AgentStatus(Enum):
    """Status-Enum für Agents (konsolidiert aus mehreren Dateien)."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class AgentType(Enum):
    """Typ-Enum für Agents (konsolidiert aus mehreren Dateien)."""
    CUSTOM = "custom"
    FOUNDRY = "foundry"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"


class ExecutionStatus(Enum):
    """Status-Codes für Agent-Ausführung."""
    SUCCESS = "success"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LogLevel(Enum):
    """Log-Level für Agent-Framework."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# TIMEOUT-KONSTANTEN (Konsolidiert)
# ============================================================================

class TimeoutConstants:
    """Konsolidierte Timeout-Konfigurationen für alle Operationen."""

    # Execution Timeouts (30 Sekunden Standard überall)
    DEFAULT_EXECUTION_TIMEOUT_SECONDS: Final[int] = 30
    DEFAULT_TASK_TIMEOUT_SECONDS: Final[float] = 30.0
    DEFAULT_TIMEOUT: Final[int] = 30

    # Tool und Endpoint Timeouts
    TOOL_ENDPOINT_TIMEOUT_SECONDS: Final[int] = 30

    # Health Check Timeouts
    DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 30
    DEFAULT_READINESS_CHECK_INTERVAL_SECONDS: Final[int] = 10
    HEALTH_CHECK_TIMEOUT: Final[float] = 5.0

    # Circuit Breaker Timeouts
    CIRCUIT_BREAKER_TIMEOUT: Final[int] = 60
    DEFAULT_RECOVERY_TIMEOUT: Final[float] = 60.0


# HTTP Timeouts (aus agents_constants.py)
HTTP_TIMEOUTS: dict[str, float] = {
    "default": 10.0,
    "short": 5.0,
    "long": 30.0,
    "web_research": 5.0,
}


# ============================================================================
# PERFORMANCE-KONSTANTEN (Konsolidiert)
# ============================================================================

# Performance-Schwellenwerte in Millisekunden
PERFORMANCE_WARNING_THRESHOLD_MS: Final[int] = 1000  # 1 Sekunde
PERFORMANCE_ERROR_THRESHOLD_MS: Final[int] = 5000    # 5 Sekunden

class PerformanceThresholds:
    """Konsolidierte Performance-Schwellenwerte für Monitoring."""

    # Execution Time Thresholds (in seconds)
    FAST_TASK_THRESHOLD: Final[float] = 1.0
    NORMAL_TASK_THRESHOLD: Final[float] = 5.0
    SLOW_TASK_THRESHOLD: Final[float] = 30.0

    # Success Rate Thresholds (as percentage)
    EXCELLENT_SUCCESS_RATE: Final[float] = 0.99
    GOOD_SUCCESS_RATE: Final[float] = 0.95
    ACCEPTABLE_SUCCESS_RATE: Final[float] = 0.90

    # Concurrency Limits
    DEFAULT_MAX_CONCURRENT_EXECUTIONS: Final[int] = 5
    DEFAULT_MAX_CONCURRENT_TASKS: Final[int] = 10
    MAX_CONCURRENT_REQUESTS: Final[int] = 10
    REQUEST_QUEUE_SIZE: Final[int] = 100


# ============================================================================
# DEFAULT-WERTE (Konsolidiert)
# ============================================================================

class DefaultValues:
    """Konsolidierte Default-Werte für alle Konfigurationen."""

    # Framework Defaults
    DEFAULT_FRAMEWORK: Final[str] = AgentFramework.FOUNDRY.value
    DEFAULT_MODEL_NAME: Final[str] = "gpt-4o"
    DEFAULT_TEMPERATURE: Final[float] = 0.7
    DEFAULT_MAX_TOKENS: Final[int] = 1000
    DEFAULT_LOG_LEVEL: Final[str] = LogLevel.INFO.value

    # Feature Flags
    DEFAULT_ENABLE_METRICS: Final[bool] = True
    METRICS_ENABLED: Final[bool] = True
    TRACING_ENABLED: Final[bool] = True
    DEFAULT_COMPONENT_SINGLETON: Final[bool] = True

    # Batch Processing
    DEFAULT_BATCH_SIZE: Final[int] = 50
    MAX_BATCH_SIZE: Final[int] = 500
    MIN_BATCH_SIZE: Final[int] = 1


# ============================================================================
# ADAPTER-KONSTANTEN (Konsolidiert aus adapter/constants.py)
# ============================================================================

# Framework-Konstanten
FRAMEWORK_FOUNDRY: Final[str] = "foundry"
FRAMEWORK_AUTOGEN: Final[str] = "autogen"
FRAMEWORK_SEMANTIC_KERNEL: Final[str] = "semantic_kernel"

# Unterstützte Frameworks
SUPPORTED_FRAMEWORKS: Final[list[str]] = [FRAMEWORK_FOUNDRY]

# Framework-zu-Modul-Mapping
FRAMEWORK_MODULE_MAP: Final[dict[str, str]] = {
    FRAMEWORK_FOUNDRY: "foundry_adapter",
}

# Framework-Anzeigenamen
FRAMEWORK_DISPLAY_NAMES: Final[dict[str, str]] = {
    FRAMEWORK_FOUNDRY: "Azure AI Foundry",
    FRAMEWORK_AUTOGEN: "AutoGen",
    FRAMEWORK_SEMANTIC_KERNEL: "Semantic Kernel",
}

# Task-Verarbeitungskonstanten
TASK_PREVIEW_MAX_LENGTH: Final[int] = 100
TASK_PREVIEW_SUFFIX: Final[str] = "..."

# Status-Konstanten
STATUS_SUCCESS: Final[str] = "success"
STATUS_ERROR: Final[str] = "error"
STATUS_PENDING: Final[str] = "pending"

# Thread-Konstanten
DEFAULT_THREAD_ROLE: Final[str] = "user"

# Adapter-Typ-Konstanten
ADAPTER_TYPE_FOUNDRY: Final[str] = "foundry"

# Validierungs-Konstanten
VALIDATION_AVAILABLE: Final[bool] = True

# ============================================================================
# CIRCUIT BREAKER KONSTANTEN (Konsolidiert aus circuit_breaker/constants.py)
# ============================================================================

# Default Circuit Breaker Konfiguration
DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_SUCCESS_THRESHOLD: Final[int] = 3
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_RECOVERY_ATTEMPTS: Final[int] = 3
DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
DEFAULT_MAX_BACKOFF_SECONDS: Final[int] = 300

# Agent-Type-spezifische Konfiguration
VOICE_AGENT_FAILURE_THRESHOLD: Final[int] = 3
VOICE_AGENT_TIMEOUT_SECONDS: Final[float] = 10.0
TOOL_AGENT_FAILURE_THRESHOLD: Final[int] = 5
TOOL_AGENT_TIMEOUT_SECONDS: Final[float] = 30.0
WORKFLOW_AGENT_FAILURE_THRESHOLD: Final[int] = 2
WORKFLOW_AGENT_TIMEOUT_SECONDS: Final[float] = 60.0
ORCHESTRATOR_AGENT_FAILURE_THRESHOLD: Final[int] = 2
ORCHESTRATOR_AGENT_TIMEOUT_SECONDS: Final[float] = 15.0

# Fallback Konfiguration
DEFAULT_FALLBACK_TIMEOUT_SECONDS: Final[float] = 15.0
DEFAULT_MAX_FALLBACK_ATTEMPTS: Final[int] = 2
DEFAULT_FALLBACK_EXECUTION_TIME_MS: Final[float] = 50.0

# Metrics und Monitoring
DEFAULT_SLIDING_WINDOW_SIZE: Final[int] = 100
DEFAULT_SLIDING_WINDOW_DURATION_SECONDS: Final[float] = 300.0
DEFAULT_MINIMUM_THROUGHPUT: Final[int] = 10
DEFAULT_FAILURE_RATE_THRESHOLD: Final[float] = 0.5

# Fallback Agent Namen
FALLBACK_AGENTS: dict[str, list[str]] = {
    "voice_agent": [
        "voice_fallback_agent",
        "simple_text_agent",
        "echo_agent"
    ],
    "tool_agent": [
        "generic_tool_agent",
        "fallback_tool_agent"
    ],
    "workflow_agent": [
        "simple_workflow_agent",
        "linear_workflow_agent"
    ],
    "custom_agent": [
        "generic_agent",
        "echo_agent"
    ]
}

# Adaptive Backoff Konfiguration
HIGH_FAILURE_RATE_THRESHOLD: Final[float] = 0.8
MEDIUM_FAILURE_RATE_THRESHOLD: Final[float] = 0.5
HIGH_FAILURE_RATE_MULTIPLIER: Final[int] = 3
MEDIUM_FAILURE_RATE_MULTIPLIER: Final[int] = 2
RECENT_CALLS_WINDOW_MINUTES: Final[int] = 5

# Circuit Breaker Error Messages
ERROR_CIRCUIT_BREAKER_OPEN: Final[str] = "Circuit breaker {name} is {state}"
ERROR_SERVICE_NOT_INITIALIZED: Final[str] = "Service not initialized"
ERROR_INVALID_STATE: Final[str] = "Invalid state: {state}. Use 'open', 'closed', or 'reset'"

# Circuit Breaker Logging Messages
LOG_CIRCUIT_BREAKER_OPENED: Final[str] = "Circuit breaker {name} opened (failures: {failures}, next attempt: {next_attempt})"
LOG_CIRCUIT_BREAKER_CLOSED: Final[str] = "Circuit breaker {name} closed (successes: {successes})"
LOG_CIRCUIT_BREAKER_HALF_OPEN: Final[str] = "Circuit breaker {name} transitioned to half-open"
LOG_FAILURE_RECORDED: Final[str] = "Circuit breaker {name}: Failure recorded (type: {type}, consecutive: {consecutive}, error: {error})"
LOG_SUCCESS_RECORDED: Final[str] = "Circuit breaker {name}: Success recorded (time: {time}ms, consecutive: {consecutive})"

# Circuit Breaker Cache Konfiguration
DEFAULT_CACHE_MAX_SIZE: Final[int] = 1000
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 300
DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS: Final[int] = 60

# ============================================================================
# LOGGING UND METRICS (Konsolidiert aus base_agent_constants.py)
# ============================================================================

class LogEvents:
    """Log-Event-Konstanten."""
    AGENT_INIT: Final[str] = "agent.init"
    AGENT_HANDLE_START: Final[str] = "agent.handle.start"
    AGENT_HANDLE_COMPLETE: Final[str] = "agent.handle.complete"
    TASK_EXECUTION_ERROR: Final[str] = "task.execution.error"
    PERFORMANCE_WARNING: Final[str] = "performance.warning"
    PERFORMANCE_ERROR: Final[str] = "performance.error"


class MetricsNames:
    """Metrik-Namen-Konstanten."""
    TASK_EXECUTION_TIME: Final[str] = "task_execution_time"
    TASK_SUCCESS_RATE: Final[str] = "task_success_rate"
    AGENT_AVAILABILITY: Final[str] = "agent_availability"
    TOTAL_REQUESTS: Final[str] = "total_requests"
    SUCCESSFUL_REQUESTS: Final[str] = "successful_requests"
    FAILED_REQUESTS: Final[str] = "failed_requests"
    ACTIVE_REQUESTS: Final[str] = "active_requests"
    ERROR_RATE: Final[str] = "error_rate"
    RESPONSE_TIME: Final[str] = "response_time"
    AVERAGE_LATENCY: Final[str] = "average_latency"
    STORAGE_UPLOAD_LATENCY: Final[str] = "storage_upload_latency"


# ============================================================================
# ERROR MESSAGES (Konsolidiert)
# ============================================================================

class ErrorMessages:
    """Fehler-Nachrichten-Konstanten."""
    AGENT_NOT_AVAILABLE: Final[str] = "Agent ist nicht verfügbar"
    TASK_EXECUTION_FAILED: Final[str] = "Task-Ausführung fehlgeschlagen"
    METRICS_RECORDING_FAILED: Final[str] = "Metrics-Aufzeichnung fehlgeschlagen"
    AGENT_NOT_INITIALIZED: Final[str] = "Agent {agent_id} ist nicht initialisiert"
    HEALTH_CHECK_FAILED: Final[str] = "Health-Check fehlgeschlagen: {status}"
    INVALID_PARAMETERS: Final[str] = "Ungültige Parameter"

    # Adapter-spezifische Error Messages
    ERROR_ADAPTER_FACTORY_UNAVAILABLE: Final[str] = "AdapterFactory nicht verfügbar"
    ERROR_AGENT_NAME_REQUIRED: Final[str] = "Agent-Name ist in Konfiguration erforderlich"
    ERROR_AGENT_ID_EMPTY: Final[str] = "Agent-ID darf nicht leer sein"
    ERROR_TASK_EMPTY: Final[str] = "Task darf nicht leer sein"
    ERROR_UNKNOWN_FRAMEWORK: Final[str] = "Unbekanntes Framework"


# ============================================================================
# ADAPTER-SPEZIFISCHE KONSTANTEN (Konsolidiert aus adapter/constants.py)
# ============================================================================

# Adapter Error Messages (direkte Konstanten für Backward Compatibility)
ERROR_ADAPTER_FACTORY_UNAVAILABLE: Final[str] = "AdapterFactory nicht verfügbar"
ERROR_AGENT_NAME_REQUIRED: Final[str] = "Agent-Name ist in Konfiguration erforderlich"
ERROR_AGENT_ID_EMPTY: Final[str] = "Agent-ID darf nicht leer sein"
ERROR_TASK_EMPTY: Final[str] = "Task darf nicht leer sein"
ERROR_UNKNOWN_FRAMEWORK: Final[str] = "Unbekanntes Framework"

# Adapter-spezifische Konfiguration
DEFAULT_API_VERSION: Final[str] = "2024-05-01-preview"

# Adapter Log Messages
LOG_ADAPTER_CREATED: Final[str] = "Adapter für {framework} erstellt"
LOG_AGENT_CREATED: Final[str] = "Agent {name} erstellt"
LOG_AGENT_OVERWRITTEN: Final[str] = "Agent {name} überschrieben"
LOG_FOUNDRY_INITIALIZED: Final[str] = "Azure AI Foundry initialisiert"
LOG_TASK_COMPLETED: Final[str] = "Task abgeschlossen: {task}"
LOG_TASK_EXECUTING: Final[str] = "Task wird ausgeführt: {task}"

# Adapter Status
STATUS_SUCCESS: Final[str] = "success"
STATUS_ERROR: Final[str] = "error"
STATUS_PENDING: Final[str] = "pending"

# Task Preview
TASK_PREVIEW_MAX_LENGTH: Final[int] = 100
TASK_PREVIEW_SUFFIX: Final[str] = "..."

# Thread Configuration
DEFAULT_THREAD_ROLE: Final[str] = "user"

# Adapter Types
ADAPTER_TYPE_FOUNDRY: Final[str] = "foundry"

# Validation
VALIDATION_AVAILABLE: Final[bool] = True


# ============================================================================
# CACHE UND MEMORY (Konsolidiert aus memory_constants.py)
# ============================================================================

# Cache-Konfiguration
CACHE_TTL: Final[int] = 300  # 5 Minuten
CACHE_MAX_SIZE: Final[int] = 1000
CACHE_CLEANUP_INTERVAL: Final[int] = 60  # 1 Minute

# Memory-Management
MEMORY_PRESSURE_THRESHOLD: Final[float] = 0.85
GC_INTERVAL_SECONDS: Final[int] = 60


# ============================================================================
# VALIDATION UND LIMITS (Konsolidiert)
# ============================================================================

# Text-Limits
MAX_TEXT_LENGTH: Final[int] = 100_000  # 100KB
MAX_SEARCH_RESULTS: Final[int] = 1000
MAX_RETRY_ATTEMPTS: Final[int] = 3

# Validation Patterns
VALIDATION_PATTERNS: dict[str, str] = {
    "agent_id": r"^[a-zA-Z0-9_-]+$",
    "task_name": r"^[a-zA-Z0-9_\s-]+$",
    "capability": r"^[a-zA-Z0-9_.-]+$",
}


# ============================================================================
# MCP UND TOOLS (Konsolidiert aus tools_constants.py)
# ============================================================================

# MCP Bridge Settings
MCP_BRIDGE_SETTINGS: dict[str, Any] = {
    "capability_mappings": {
        "web_research": ["search", "browse", "fetch"],
        "file_operations": ["read", "write", "list"],
        "data_analysis": ["analyze", "compute", "visualize"],
    },
    "default_timeout": 30,
    "max_retries": 3,
}

# Cache Settings für Tools
CACHE_SETTINGS: dict[str, Any] = {
    "ttl": CACHE_TTL,
    "max_size": CACHE_MAX_SIZE,
    "cleanup_interval": CACHE_CLEANUP_INTERVAL,
}


# ============================================================================
# CUSTOM AGENTS (Konsolidiert aus custom/constants.py)
# ============================================================================

from enum import Enum

# =============================================================================
# Image Generation Enums
# =============================================================================

class ImageSize(str, Enum):
    """Unterstützte Bildgrößen für DALL·E-3."""

    SQUARE = "1024x1024"
    PORTRAIT = "1024x1792"
    LANDSCAPE = "1792x1024"


class ImageQuality(str, Enum):
    """Bildqualitäts-Optionen."""

    STANDARD = "standard"
    HD = "hd"


class ImageStyle(str, Enum):
    """Verfügbare Bildstile."""

    REALISTIC = "Realistic"
    ARTISTIC = "Artistic"
    CARTOON = "Cartoon"
    PHOTOGRAPHY = "Photography"
    DIGITAL_ART = "Digital Art"


# =============================================================================
# Agent Configuration Enums
# =============================================================================


class AgentFunctionType(str, Enum):
    """Agent-Funktionstypen."""

    IMAGE_GENERATION = "image_generation"
    TEXT_PROCESSING = "text_processing"
    DATA_ANALYSIS = "data_analysis"
    CUSTOM = "custom"


# =============================================================================
# Content Safety Constants
# =============================================================================

# Blacklist für Prompt-Sanitization
CONTENT_SAFETY_BLACKLIST: Final[set[str]] = {
    "gore",
    "child sexual",
    "csam",
    "terror",
    "how to make a bomb",
    "vergewaltigung",
    "vergewaltige",
    "sprengstoffbau",
    "anschlag",
    "pornografisch",
    "kinderporn",
    "nsfw",
}

# Replacement-Marker für problematische Inhalte
REDACTED_MARKER: Final[str] = "[redacted]"

# Safety-Score Schwellenwerte
SAFETY_SCORE_THRESHOLD: Final[float] = 0.5
SAFETY_CONFIDENCE_THRESHOLD: Final[float] = 0.7


# =============================================================================
# Storage Constants
# =============================================================================

# Standard-Container für Bilder
DEFAULT_IMAGE_CONTAINER: Final[str] = "generated-images"

# SAS-URL Gültigkeitsdauer (Minuten)
DEFAULT_SAS_EXPIRY_MINUTES: Final[int] = 60

# Unterstützte Content-Types
SUPPORTED_IMAGE_CONTENT_TYPES: Final[set[str]] = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

# Maximale Dateigröße (Bytes)
MAX_IMAGE_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB

# =============================================================================
# Direkte Konstanten für Kompatibilität
# =============================================================================

# Extrahierte Konstanten aus DefaultValues für direkten Import
DEFAULT_MAX_TOKENS: Final[int] = DefaultValues.DEFAULT_MAX_TOKENS
DEFAULT_TEMPERATURE: Final[float] = DefaultValues.DEFAULT_TEMPERATURE
DEFAULT_MODEL_NAME: Final[str] = DefaultValues.DEFAULT_MODEL_NAME

# =============================================================================
# Performance und Retry Constants
# =============================================================================

# Timeout-Werte (Sekunden)
DEFAULT_REQUEST_TIMEOUT: Final[int] = 30
CONTENT_SAFETY_TIMEOUT: Final[int] = 10
IMAGE_GENERATION_TIMEOUT: Final[int] = 60
STORAGE_UPLOAD_TIMEOUT: Final[int] = 30

# Retry-Konfiguration (verwendet globale MAX_RETRY_ATTEMPTS aus Zeile 374)
RETRY_BASE_DELAY: Final[float] = 0.5
RETRY_MAX_DELAY: Final[float] = 5.0

# Performance-Schwellenwerte (verwendet globale Definitionen aus Zeilen 110-111)
# Image Generator spezifische Performance-Schwellenwerte
IMAGE_GENERATION_WARNING_THRESHOLD_MS: Final[int] = 15000  # 15 Sekunden
IMAGE_GENERATION_ERROR_THRESHOLD_MS: Final[int] = 30000   # 30 Sekunden

# Konvertierungs-Konstanten
SECONDS_TO_MILLISECONDS: Final[int] = 1000
DEFAULT_SAFETY_SCORE: Final[float] = 0.0
METRICS_INCREMENT: Final[int] = 1


# =============================================================================
# Prompt Optimization Constants
# =============================================================================

# Stil-spezifische Prompt-Verbesserungen
STYLE_OPTIMIZATION_MAP: Final[dict[str, str]] = {
    ImageStyle.REALISTIC: "photorealistic, high detail, realistic lighting",
    ImageStyle.ARTISTIC: "painting style, rich textures, expressive brushstrokes, abstract elements",
    ImageStyle.CARTOON: "cartoon, bold lines, vibrant colors, clean shapes",
    ImageStyle.PHOTOGRAPHY: "professional photography, depth of field, bokeh, 50mm lens, high dynamic range",
    ImageStyle.DIGITAL_ART: "digital art, highly detailed, volumetric lighting, 8k, trending on artstation",
}

# Qualitäts-Tags für Prompt-Optimierung
QUALITY_ENHANCEMENT_TAGS: Final[str] = "ultra-detailed, sharp focus, high quality"

# Prompt-Längen (verwende spezifischere Werte aus custom/constants.py)
MAX_PROMPT_LENGTH: Final[int] = 1000
MAX_OPTIMIZED_PROMPT_LENGTH: Final[int] = 1500
MIN_PROMPT_LENGTH: Final[int] = 3  # Korrigiert von 10 auf 3


# =============================================================================
# Validation Constants
# =============================================================================

# Erlaubte Werte für Validierung (als Sets für O(1) Lookup)
ALLOWED_IMAGE_SIZES: Final[set[str]] = {size.value for size in ImageSize}
ALLOWED_IMAGE_QUALITIES: Final[set[str]] = {quality.value for quality in ImageQuality}
ALLOWED_IMAGE_STYLES: Final[set[str]] = {style.value for style in ImageStyle}


# =============================================================================
# Error Messages
# =============================================================================

class ErrorMessagesImageGenerator:
    """Standardisierte Fehlermeldungen für Image Generator."""

    # Validation Errors
    INVALID_SIZE = "Ungültige Bildgröße"
    INVALID_QUALITY = "Ungültige Bildqualität"
    INVALID_STYLE = "Ungültiger Bildstil"
    PROMPT_REQUIRED = "Prompt erforderlich"
    PROMPT_TOO_SHORT = "Prompt zu kurz"
    PROMPT_TOO_LONG = "Prompt zu lang"

    # Service Errors
    CONTENT_SAFETY_BLOCKED = "Inhalt durch Content Safety blockiert"
    IMAGE_GENERATION_FAILED = "Bildgenerierung fehlgeschlagen"
    STORAGE_UPLOAD_FAILED = "Storage-Upload fehlgeschlagen"
    AZURE_STORAGE_NOT_CONFIGURED = "Azure Storage nicht konfiguriert"

    # Agent Errors
    AGENT_NOT_AVAILABLE = "Agent nicht verfügbar"
    AGENT_INITIALIZATION_FAILED = "Agent-Initialisierung fehlgeschlagen"


# =============================================================================
# Logging Event Names
# =============================================================================

class LogEventsImageGenerator:
    """Standardisierte Logging-Event-Namen für Image Generator."""

    # Agent Events
    AGENT_INIT = "agent_init"
    AGENT_HANDLE_START = "agent_handle_start"
    AGENT_HANDLE_COMPLETE = "agent_handle_complete"

    # Content Safety Events
    CONTENT_SAFETY_CHECK = "content_safety_check"
    CONTENT_SAFETY_BLOCKED = "content_safety_blocked"
    CONTENT_SAFETY_ERROR = "content_safety_error"

    # Image Generation Events
    IMAGE_GENERATION_START = "image_generation_start"
    IMAGE_GENERATION_COMPLETE = "image_generation_complete"
    IMAGE_GENERATION_ERROR = "image_generation_error"

    # Storage Events
    STORAGE_UPLOAD_START = "storage_upload_start"
    STORAGE_UPLOAD_COMPLETE = "storage_upload_complete"
    STORAGE_UPLOAD_ERROR = "storage_upload_error"

    # Performance Events
    PERFORMANCE_WARNING = "performance_warning"
    PERFORMANCE_ERROR = "performance_error"


# =============================================================================
# Metrics Names
# =============================================================================

class MetricsNamesImageGenerator:
    """Standardisierte Metrics-Namen für Image Generator."""

    # Latenz-Metriken
    CONTENT_SAFETY_LATENCY = "content_safety.latency_ms"
    IMAGE_GENERATION_LATENCY = "image_generation.latency_ms"
    STORAGE_UPLOAD_LATENCY = "storage_upload.latency_ms"
    TOTAL_REQUEST_LATENCY = "agent.total_request_latency_ms"

    # Counter-Metriken
    REQUESTS_TOTAL = "agent.requests_total"
    REQUESTS_SUCCESS = "agent.requests_success"
    REQUESTS_FAILED = "agent.requests_failed"
    CONTENT_SAFETY_BLOCKED = "content_safety.blocked_total"

    # Gauge-Metriken
    ACTIVE_REQUESTS = "agent.active_requests"
    QUEUE_LENGTH = "agent.queue_length"


# ============================================================================
# MEMORY (Konsolidiert aus memory_constants.py)
# ============================================================================

# Cosmos DB Categories
CHAT_MESSAGE_CATEGORY: Final[str] = "chat_message"
LANGGRAPH_CHECKPOINT_CATEGORY: Final[str] = "langgraph_checkpoint"

# Retention Policies
DEFAULT_RETENTION_DAYS: Final[int] = 30
MAX_RETENTION_DAYS: Final[int] = 365


# ============================================================================
# WORKFLOWS (Konsolidiert aus workflows_constants.py)
# ============================================================================

# LangGraph Workflow Constants
DEFAULT_WORKFLOW_TIMEOUT: Final[int] = 300  # 5 Minuten
MAX_WORKFLOW_STEPS: Final[int] = 50
WORKFLOW_CHECKPOINT_INTERVAL: Final[int] = 10


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timeout(timeout_type: str = "default") -> float:
    """Gibt den konfigurierten Timeout-Wert zurück."""
    return HTTP_TIMEOUTS.get(timeout_type, HTTP_TIMEOUTS["default"])


def get_error_message(error_type: str) -> str:
    """Gibt eine standardisierte Fehlermeldung zurück."""
    error_messages = {
        "agent_not_available": ErrorMessages.AGENT_NOT_AVAILABLE,
        "task_failed": ErrorMessages.TASK_EXECUTION_FAILED,
        "metrics_failed": ErrorMessages.METRICS_RECORDING_FAILED,
    }
    return error_messages.get(error_type, "Unbekannter Fehler")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "AgentFramework",
    "AgentStatus",
    "AgentType",
    "AgentFunctionType",
    "ExecutionStatus",
    "LogLevel",
    # Constants Classes
    "TimeoutConstants",
    "PerformanceThresholds",
    "DefaultValues",
    "LogEvents",
    "MetricsNames",
    "ErrorMessages",
    "ErrorMessagesImageGenerator",
    "LogEventsImageGenerator",
    "MetricsNamesImageGenerator",
    # Individual Constants
    "HTTP_TIMEOUTS",
    "PERFORMANCE_WARNING_THRESHOLD_MS",
    "PERFORMANCE_ERROR_THRESHOLD_MS",
    "CACHE_TTL",
    "CACHE_MAX_SIZE",
    "CACHE_CLEANUP_INTERVAL",
    "MEMORY_PRESSURE_THRESHOLD",
    "GC_INTERVAL_SECONDS",
    "MAX_TEXT_LENGTH",
    "MAX_SEARCH_RESULTS",
    "MAX_RETRY_ATTEMPTS",
    "VALIDATION_PATTERNS",
    # Direkte Konstanten für Kompatibilität
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MODEL_NAME",
    # MCP and Tools
    "MCP_BRIDGE_SETTINGS",
    "CACHE_SETTINGS",
    # Custom Agents
    "MAX_PROMPT_LENGTH",
    "MIN_PROMPT_LENGTH",
    "DEFAULT_IMAGE_CONTAINER",
    "IMAGE_GENERATION_TIMEOUT",
    "CONTENT_SAFETY_TIMEOUT",
    # Memory
    "CHAT_MESSAGE_CATEGORY",
    # Adapter Constants
    "FRAMEWORK_FOUNDRY",
    "FRAMEWORK_AUTOGEN",
    "FRAMEWORK_SEMANTIC_KERNEL",
    "SUPPORTED_FRAMEWORKS",
    "FRAMEWORK_MODULE_MAP",
    "FRAMEWORK_DISPLAY_NAMES",
    "TASK_PREVIEW_MAX_LENGTH",
    "TASK_PREVIEW_SUFFIX",
    "STATUS_SUCCESS",
    "STATUS_ERROR",
    "STATUS_PENDING",
    # Adapter Error Messages
    "ERROR_ADAPTER_FACTORY_UNAVAILABLE",
    "ERROR_AGENT_NAME_REQUIRED",
    "ERROR_AGENT_ID_EMPTY",
    "ERROR_TASK_EMPTY",
    "ERROR_UNKNOWN_FRAMEWORK",
    # Adapter Configuration
    "DEFAULT_API_VERSION",
    # Adapter Log Messages
    "LOG_ADAPTER_CREATED",
    "LOG_AGENT_CREATED",
    "LOG_AGENT_OVERWRITTEN",
    "LOG_FOUNDRY_INITIALIZED",
    "LOG_TASK_COMPLETED",
    "LOG_TASK_EXECUTING",
    # Adapter Types and Config
    "DEFAULT_THREAD_ROLE",
    "ADAPTER_TYPE_FOUNDRY",
    "VALIDATION_AVAILABLE",
    "DEFAULT_THREAD_ROLE",
    "ADAPTER_TYPE_FOUNDRY",
    "VALIDATION_AVAILABLE",
    # Circuit Breaker Constants
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RECOVERY_TIMEOUT_SECONDS",
    "DEFAULT_SUCCESS_THRESHOLD",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RECOVERY_ATTEMPTS",
    "DEFAULT_BACKOFF_MULTIPLIER",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "VOICE_AGENT_FAILURE_THRESHOLD",
    "VOICE_AGENT_TIMEOUT_SECONDS",
    "TOOL_AGENT_FAILURE_THRESHOLD",
    "TOOL_AGENT_TIMEOUT_SECONDS",
    "WORKFLOW_AGENT_FAILURE_THRESHOLD",
    "WORKFLOW_AGENT_TIMEOUT_SECONDS",
    "ORCHESTRATOR_AGENT_FAILURE_THRESHOLD",
    "ORCHESTRATOR_AGENT_TIMEOUT_SECONDS",
    "DEFAULT_FALLBACK_TIMEOUT_SECONDS",
    "DEFAULT_MAX_FALLBACK_ATTEMPTS",
    "DEFAULT_FALLBACK_EXECUTION_TIME_MS",
    "DEFAULT_SLIDING_WINDOW_SIZE",
    "DEFAULT_SLIDING_WINDOW_DURATION_SECONDS",
    "DEFAULT_MINIMUM_THROUGHPUT",
    "DEFAULT_FAILURE_RATE_THRESHOLD",
    "FALLBACK_AGENTS",
    "HIGH_FAILURE_RATE_THRESHOLD",
    "MEDIUM_FAILURE_RATE_THRESHOLD",
    "HIGH_FAILURE_RATE_MULTIPLIER",
    "MEDIUM_FAILURE_RATE_MULTIPLIER",
    "RECENT_CALLS_WINDOW_MINUTES",
    # Circuit Breaker Error and Log Messages
    "ERROR_CIRCUIT_BREAKER_OPEN",
    "ERROR_SERVICE_NOT_INITIALIZED",
    "ERROR_INVALID_STATE",
    "LOG_CIRCUIT_BREAKER_OPENED",
    "LOG_CIRCUIT_BREAKER_CLOSED",
    "LOG_CIRCUIT_BREAKER_HALF_OPEN",
    "LOG_FAILURE_RECORDED",
    "LOG_SUCCESS_RECORDED",
    # Circuit Breaker Cache Configuration
    "DEFAULT_CACHE_MAX_SIZE",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS",
    "LANGGRAPH_CHECKPOINT_CATEGORY",
    "DEFAULT_RETENTION_DAYS",
    "MAX_RETENTION_DAYS",
    # Workflows
    "DEFAULT_WORKFLOW_TIMEOUT",
    "MAX_WORKFLOW_STEPS",
    "WORKFLOW_CHECKPOINT_INTERVAL",
    # Image Generator Constants
    "ImageSize",
    "ImageQuality",
    "ImageStyle",
    "ALLOWED_IMAGE_SIZES",
    "ALLOWED_IMAGE_QUALITIES",
    "ALLOWED_IMAGE_STYLES",
    "DEFAULT_SAFETY_SCORE",
    "IMAGE_GENERATION_WARNING_THRESHOLD_MS",
    "IMAGE_GENERATION_ERROR_THRESHOLD_MS",
    "METRICS_INCREMENT",
    "SECONDS_TO_MILLISECONDS",
    "MIN_PROMPT_LENGTH",
    # Utility Functions
    "get_timeout",
    "get_error_message",
]
