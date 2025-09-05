# backend/agents/metadata/constants.py
"""Konstanten für Agent Metadata Module.

Konsolidiert alle Konfigurationswerte für bessere Wartbarkeit.
"""

from __future__ import annotations

from typing import Final

# Framework-Versionen
AZURE_AI_FOUNDRY_VERSION: Final[str] = "2025.03"
AZURE_AI_FOUNDRY_API_VERSION: Final[str] = "2025-03-01-preview"
SEMANTIC_KERNEL_VERSION: Final[str] = "1.0.0"
AUTOGEN_VERSION: Final[str] = "0.4.0"
CUSTOM_MCP_VERSION: Final[str] = "1.0.0"

# Default-Konfigurationswerte
DEFAULT_MAX_TOKENS: Final[int] = 4000
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_AUTO_FUNCTION_INVOCATION: Final[bool] = True

# Deprecation-Konfiguration
DEFAULT_MIN_MINOR_GAP: Final[int] = 2
DEFAULT_MINOR_GRACE_PERIOD: Final[int] = 2

# Agent-Versioning
DEFAULT_AGENT_VERSION: Final[str] = "1.0.0"

# Framework-spezifische Konfigurationen
AZURE_FOUNDRY_CONFIG_TEMPLATE: Final[dict[str, str]] = {
    "api_version": AZURE_AI_FOUNDRY_API_VERSION,
}

SEMANTIC_KERNEL_CONFIG_TEMPLATE: Final[dict[str, str | int | bool]] = {
    "auto_function_invocation": DEFAULT_AUTO_FUNCTION_INVOCATION,
    "max_tokens": DEFAULT_MAX_TOKENS,
}

AUTOGEN_CONFIG_TEMPLATE: Final[dict[str, float]] = {
    "temperature": DEFAULT_TEMPERATURE,
}

# Framework-Typ-Mappings
FRAMEWORK_VERSION_MAP: Final[dict[str, str]] = {
    "AZURE_AI_FOUNDRY": AZURE_AI_FOUNDRY_VERSION,
    "SEMANTIC_KERNEL": SEMANTIC_KERNEL_VERSION,
    "AUTOGEN": AUTOGEN_VERSION,
    "CUSTOM_MCP": CUSTOM_MCP_VERSION,
}

# Validierungs-Konstanten
MIN_AGENT_ID_LENGTH: Final[int] = 3
MAX_AGENT_ID_LENGTH: Final[int] = 100
MIN_AGENT_NAME_LENGTH: Final[int] = 1
MAX_AGENT_NAME_LENGTH: Final[int] = 255

# Logging-Nachrichten
LOG_MESSAGES: Final[dict[str, str]] = {
    "azure_foundry_created": "Azure AI Foundry AgentMetadata erstellt: {}",
    "semantic_kernel_created": "Semantic Kernel AgentMetadata erstellt: {}",
    "autogen_created": "AutoGen AgentMetadata erstellt: {}",
    "metadata_registered": "AgentMetadata registriert: {}",
    "metadata_unregistered": "AgentMetadata entfernt: {}",
    "capability_deprecated": "Capability '{}' ist veraltet (Agent {} vs Capability {}). Bitte auf eine neuere Version migrieren.",
}

# Error-Nachrichten
ERROR_MESSAGES: Final[dict[str, str]] = {
    "unknown_framework": "Unbekannter Framework-Typ: {}",
    "invalid_agent_id": "Ungültige Agent-ID: {}",
    "invalid_agent_name": "Ungültiger Agent-Name: {}",
    "metadata_not_found": "AgentMetadata nicht gefunden: {}",
    "framework_inference_failed": "Framework-Typ konnte nicht abgeleitet werden für Agent: {}",
}

# Framework-Erkennungs-Patterns
FRAMEWORK_DETECTION_PATTERNS: Final[dict[str, list[str]]] = {
    "AZURE_AI_FOUNDRY": ["azure", "foundry", "project_id", "endpoint"],
    "SEMANTIC_KERNEL": ["semantic", "kernel", "model_deployment", "auto_function"],
    "AUTOGEN": ["autogen", "model_config", "conversation"],
    "CUSTOM_MCP": ["mcp", "custom", "server"],
}

__all__ = [
    # Framework-Versionen
    "AZURE_AI_FOUNDRY_VERSION",
    "AZURE_AI_FOUNDRY_API_VERSION",
    "SEMANTIC_KERNEL_VERSION",
    "AUTOGEN_VERSION",
    "CUSTOM_MCP_VERSION",

    # Default-Werte
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_AUTO_FUNCTION_INVOCATION",
    "DEFAULT_MIN_MINOR_GAP",
    "DEFAULT_MINOR_GRACE_PERIOD",
    "DEFAULT_AGENT_VERSION",

    # Konfiguration-Templates
    "AZURE_FOUNDRY_CONFIG_TEMPLATE",
    "SEMANTIC_KERNEL_CONFIG_TEMPLATE",
    "AUTOGEN_CONFIG_TEMPLATE",
    "FRAMEWORK_VERSION_MAP",

    # Validierung
    "MIN_AGENT_ID_LENGTH",
    "MAX_AGENT_ID_LENGTH",
    "MIN_AGENT_NAME_LENGTH",
    "MAX_AGENT_NAME_LENGTH",

    # Nachrichten
    "LOG_MESSAGES",
    "ERROR_MESSAGES",
    "FRAMEWORK_DETECTION_PATTERNS",
]
