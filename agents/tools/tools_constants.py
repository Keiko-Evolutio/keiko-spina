"""Zentrale Constants für das tools-Modul.

Konsolidiert alle Magic Numbers, Hard-coded Strings und Konfigurationswerte
aus dem gesamten tools-Modul zur Verbesserung der Wartbarkeit und Konsistenz.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# HTTP-Konfiguration
# =============================================================================

HTTP_TIMEOUTS: dict[str, float] = {
    "default": 10.0,
    "short": 5.0,
    "long": 30.0,
    "web_research": 5.0,
    "health_check": 10.0,  # Für MCP Health Checks
}

HTTP_HEADERS: dict[str, str] = {
    "content_type": "Content-Type",
    "api_key": "api-key",
    "json": "application/json",
    "user_agent": "Keiko-Agent-Tools/1.0",
    "authorization": "Authorization",
    "x_api_key": "X-API-Key",
}

HTTP_STATUS_CODES: dict[str, int] = {
    "ok": 200,
    "created": 201,
    "bad_request": 400,
    "unauthorized": 401,
    "not_found": 404,
    "server_error": 500,
}

# =============================================================================
# API-Versionen und URLs
# =============================================================================

API_VERSIONS: dict[str, str] = {
    "azure_search": "2023-07-01-Preview",
    "cosmos_db": "2023-11-15",
    "openai": "2023-12-01-preview",
}

DEFAULT_URLS: dict[str, str] = {
    "duckduckgo_search": "https://duckduckgo.com/html/?q=",
    "azure_search_endpoint": "/indexes/{index_name}/docs/search",
}

# =============================================================================
# Cache-Einstellungen
# =============================================================================

CACHE_SETTINGS: dict[str, Any] = {
    "ttl": 300,  # 5 Minuten
    "max_size": 1000,
    "cleanup_interval": 600,  # 10 Minuten
}

# =============================================================================
# Score-Gewichtungen und Schwellenwerte
# =============================================================================

SCORE_WEIGHTS: dict[str, float] = {
    "vector": 0.6,
    "keyword": 0.4,
    "hybrid": 0.5,
    "semantic": 0.7,
    "lexical": 0.3,
}

SCORE_THRESHOLDS: dict[str, float] = {
    "min_similarity": 0.1,
    "good_match": 0.7,
    "excellent_match": 0.9,
}

# =============================================================================
# Standard-Feldnamen
# =============================================================================

FIELD_NAMES: dict[str, str] = {
    "content": "content",
    "text": "text",
    "embedding": "embedding",
    "score": "score",
    "metadata": "metadata",
    "id": "id",
    "title": "title",
    "url": "url",
    "timestamp": "timestamp",
}

# =============================================================================
# Retrieval-Konfiguration
# =============================================================================

RETRIEVAL_DEFAULTS: dict[str, Any] = {
    "top_k": 5,
    "max_top_k": 100,
    "min_top_k": 1,
    "default_multiplier": 10,  # Für Kandidaten-Suche
}

# =============================================================================
# MCP-Bridge-Konfiguration
# =============================================================================

MCP_BRIDGE_SETTINGS: dict[str, Any] = {
    "capability_mappings": {
        "search": {"search", "query", "find", "lookup"},
        "weather": {"weather", "forecast", "climate", "temperature"},
        "calendar": {"calendar", "schedule", "appointment", "event"},
        "email": {"email", "mail", "message", "send"},
        "database": {"database", "db", "query", "sql", "data"},
        "file": {"file", "document", "upload", "download", "storage"},
        "image": {"image", "photo", "picture", "visual", "generate"},
        "text": {"text", "content", "write", "generate", "translate"},
        "math": {"math", "calculate", "compute", "formula"},
        "web": {"web", "http", "api", "request", "scrape"},
    },
    "scoring_weights": {
        "name_match": 3,
        "description_match": 1,
        "capability_match": 2,
    },
}

# =============================================================================
# Error-Messages
# =============================================================================

ERROR_MESSAGES: dict[str, str] = {
    "invalid_tool_id": "Ungültige Tool-ID: {tool_id}. Format: 'server:tool'",
    "tool_execution_error": "Tool-Ausführung fehlgeschlagen",
    "image_service_unavailable": "Image-Service nicht verfügbar",
    "embedding_service_unavailable": "Embedding-Service nicht verfügbar",
    "retrieval_fallback": "{retriever_name} Fallback aktiv: {error}",
    "http_request_failed": "HTTP-Request fehlgeschlagen: {url}",
    "invalid_config": "Ungültige Konfiguration: {details}",
}

# =============================================================================
# Logging-Konfiguration
# =============================================================================

LOG_EVENTS: dict[str, str] = {
    "tool_discovery": "tools.discovery",
    "tool_execution": "tools.execution",
    "retrieval_start": "retrieval.start",
    "retrieval_complete": "retrieval.complete",
    "retrieval_error": "retrieval.error",
    "cache_hit": "cache.hit",
    "cache_miss": "cache.miss",
    "fallback_activated": "fallback.activated",
}

# =============================================================================
# File-Extensions und MIME-Types
# =============================================================================

SUPPORTED_FILE_TYPES: dict[str, str] = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".json": "application/json",
}

# =============================================================================
# Validation-Patterns
# =============================================================================

VALIDATION_PATTERNS: dict[str, str] = {
    "tool_id": r"^[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+$",
    "server_name": r"^[a-zA-Z0-9_-]+$",
    "capability_name": r"^[a-zA-Z0-9_-]+$",
    "url": r"^https?://[^\s/$.?#].[^\s]*$",
}

# =============================================================================
# Performance-Limits
# =============================================================================

PERFORMANCE_LIMITS: dict[str, int] = {
    "max_concurrent_requests": 10,
    "max_text_length": 100000,  # 100KB
    "max_embedding_batch_size": 100,
    "max_search_results": 1000,
    "max_retry_attempts": 3,
}

# =============================================================================
# Default-Werte für verschiedene Retriever-Types
# =============================================================================

RETRIEVER_DEFAULTS: dict[str, dict[str, Any]] = {
    "azure_search": {
        "top_k": 5,
        "timeout": HTTP_TIMEOUTS["default"],
        "api_version": API_VERSIONS["azure_search"],
    },
    "cosmos_vector": {
        "top_k": 5,
        "collection": "documents",
        "candidate_multiplier": RETRIEVAL_DEFAULTS["default_multiplier"],
    },
    "hybrid": {
        "top_k": 5,
        "vector_weight": SCORE_WEIGHTS["vector"],
        "keyword_weight": SCORE_WEIGHTS["keyword"],
    },
    "multimodal": {
        "top_k": 5,
        "text_enabled": True,
        "image_enabled": False,
    },
}

# =============================================================================
# Pragma-Kommentare für Code-Coverage
# =============================================================================

PRAGMA_COMMENTS: dict[str, str] = {
    "no_cover_optional": "pragma: no cover - optional dependency",
    "no_cover_defensive": "pragma: no cover - defensive fallback",
    "no_cover_interface": "pragma: no cover - interface method",
    "no_cover_fallback": "pragma: no cover - fallback implementation",
}

# =============================================================================
# Utility-Funktionen für Constants
# =============================================================================

def get_timeout(timeout_type: str = "default") -> float:
    """Gibt den konfigurierten Timeout-Wert zurück."""
    return HTTP_TIMEOUTS.get(timeout_type, HTTP_TIMEOUTS["default"])


def get_api_version(service: str) -> str:
    """Gibt die API-Version für einen Service zurück."""
    return API_VERSIONS.get(service, "latest")


def get_field_name(field_type: str) -> str:
    """Gibt den standardisierten Feldnamen zurück."""
    return FIELD_NAMES.get(field_type, field_type)


def get_error_message(error_type: str, **kwargs: Any) -> str:
    """Gibt eine formatierte Error-Message zurück."""
    template = ERROR_MESSAGES.get(error_type, "Unbekannter Fehler: {error_type}")
    return template.format(error_type=error_type, **kwargs)


def get_retriever_default(retriever_type: str, setting: str) -> Any:
    """Gibt Default-Werte für Retriever-Konfiguration zurück."""
    return RETRIEVER_DEFAULTS.get(retriever_type, {}).get(setting)


__all__ = [
    "API_VERSIONS",
    "CACHE_SETTINGS",
    "DEFAULT_URLS",
    "ERROR_MESSAGES",
    "FIELD_NAMES",
    "HTTP_HEADERS",
    "HTTP_STATUS_CODES",
    # Constants
    "HTTP_TIMEOUTS",
    "LOG_EVENTS",
    "MCP_BRIDGE_SETTINGS",
    "PERFORMANCE_LIMITS",
    "PRAGMA_COMMENTS",
    "RETRIEVAL_DEFAULTS",
    "RETRIEVER_DEFAULTS",
    "SCORE_THRESHOLDS",
    "SCORE_WEIGHTS",
    "SUPPORTED_FILE_TYPES",
    "VALIDATION_PATTERNS",
    "get_api_version",
    "get_error_message",
    "get_field_name",
    "get_retriever_default",
    # Utility Functions
    "get_timeout",
]
