"""Zentrale Konstanten für Capabilities-Modul.

Eliminiert Magic Numbers und Hard-coded Strings.
"""

from typing import Any

# ============================================================================
# VERSION MANAGEMENT
# ============================================================================

DEFAULT_API_VERSION: str = "2.0.0"
SUPPORTED_API_VERSIONS: list[str] = ["2.0.0", "2.1.0"]

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

DEFAULT_SERVER_CONFIG: dict[str, str] = {
    "name": "KEI-MCP API",
    "environment": "development",
    "region": "local",
    "instance_id": "local-instance"
}

# Environment-Variable-Namen
ENV_VARS: dict[str, str] = {
    "environment": "ENVIRONMENT",
    "region": "REGION",
    "instance_id": "INSTANCE_ID"
}

# ============================================================================
# FEATURE STATUS STRINGS
# ============================================================================

FEATURE_STATUS_STRINGS: dict[str, str] = {
    "experimental": "experimental",
    "stable": "stable",
    "deprecated": "deprecated",
    "disabled": "disabled"
}

# ============================================================================
# FEATURE SCOPE STRINGS
# ============================================================================

FEATURE_SCOPE_STRINGS: dict[str, str] = {
    "global": "global",
    "per_client": "per_client",
    "per_server": "per_server"
}

# ============================================================================
# CAPABILITY CATEGORY STRINGS
# ============================================================================

CAPABILITY_CATEGORY_STRINGS: dict[str, str] = {
    "core": "core",
    "tools": "tools",
    "resources": "resources",
    "prompts": "prompts",
    "monitoring": "monitoring",
    "security": "security",
    "experimental": "experimental"
}

# ============================================================================
# DEFAULT FEATURE FLAGS
# ============================================================================

DEFAULT_FEATURE_FLAGS: list[dict[str, Any]] = [
    {
        "name": "enhanced_mcp_registry",
        "description": "Erweiterte MCP-Registry mit Versionierung und Health-Checks",
        "enabled": True,
        "scope": "global"
    },
    {
        "name": "advanced_monitoring",
        "description": "Erweiterte Monitoring-Features mit Metriken und Alerting",
        "enabled": True,
        "scope": "global"
    },
    {
        "name": "experimental_ai_features",
        "description": "Experimentelle KI-Features und Modell-Integrationen",
        "enabled": False,
        "scope": "global"
    },
    {
        "name": "beta_ui_features",
        "description": "Beta-UI-Features für ausgewählte Clients",
        "enabled": False,
        "scope": "per_client"
    },
    {
        "name": "rate_limiting_v2",
        "description": "Erweiterte Rate-Limiting-Features",
        "enabled": True,
        "scope": "global"
    }
]

# ============================================================================
# DEFAULT CAPABILITIES
# ============================================================================

DEFAULT_CAPABILITIES: list[dict[str, Any]] = [
    {
        "name": "server_management",
        "description": "MCP-Server-Registrierung und -Verwaltung",
        "category": "core",
        "endpoints": [
            "POST /api/v1/mcp/external/servers/register",
            "GET /api/v1/mcp/external/servers",
            "GET /api/v1/mcp/external/servers/{server_name}/stats",
            "DELETE /api/v1/mcp/external/servers/{server_name}"
        ]
    },
    {
        "name": "tool_management",
        "description": "Tool-Discovery und -Ausführung",
        "category": "core",
        "endpoints": [
            "GET /api/v1/mcp/external/servers/{server_name}/tools",
            "POST /api/v1/mcp/external/servers/{server_name}/tools/{tool_name}/call"
        ]
    },
    {
        "name": "resource_management",
        "description": "Resource-Discovery und -Zugriff",
        "category": "core",
        "endpoints": [
            "GET /api/v1/mcp/external/servers/{server_name}/resources",
            "GET /api/v1/mcp/external/servers/{server_name}/resources/{resource_uri}"
        ]
    },
    {
        "name": "prompt_management",
        "description": "Prompt-Discovery und -Abruf",
        "category": "core",
        "endpoints": [
            "GET /api/v1/mcp/external/servers/{server_name}/prompts",
            "GET /api/v1/mcp/external/servers/{server_name}/prompts/{prompt_name}"
        ]
    },
    {
        "name": "advanced_monitoring",
        "description": "Erweiterte Monitoring-Features mit Metriken",
        "category": "experimental",
        "endpoints": [
            "GET /api/v1/monitoring/metrics",
            "GET /api/v1/monitoring/health/detailed"
        ],
        "feature_flags": ["advanced_monitoring"]
    },
    {
        "name": "enhanced_registry",
        "description": "Erweiterte Registry mit Versionierung",
        "category": "experimental",
        "endpoints": [
            "GET /api/v1/registry/enhanced/servers",
            "POST /api/v1/registry/enhanced/servers/{server_name}/versions"
        ],
        "feature_flags": ["enhanced_mcp_registry"]
    }
]

# ============================================================================
# LOGGING MESSAGES
# ============================================================================

LOG_MESSAGES: dict[str, str] = {
    "unknown_feature": "Unknown feature flag: {feature_name}",
    "cannot_enable_unknown": "Cannot enable unknown feature: {feature_name}",
    "cannot_disable_unknown": "Cannot disable unknown feature: {feature_name}",
    "cannot_remove_unknown": "Cannot remove unknown feature: {feature_name}",
    "feature_exists": "Feature flag '{feature_name}' already exists",
    "enabled_for_client": "Enabled feature '{feature_name}' for client '{client_id}'",
    "disabled_for_client": "Disabled feature '{feature_name}' for client '{client_id}'",
    "enabled_for_server": "Enabled feature '{feature_name}' for server '{server_name}'",
    "disabled_for_server": "Disabled feature '{feature_name}' for server '{server_name}'",
    "added_feature": "Added feature flag: {feature_name}",
    "removed_feature": "Removed feature flag: {feature_name}",
    "loaded_features": "Loaded {count} default feature flags",
    "loaded_capabilities": "Loaded {count} default capabilities"
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

VALIDATION_RULES: dict[str, Any] = {
    "max_feature_name_length": 100,
    "max_description_length": 500,
    "max_endpoints_per_capability": 20,
    "max_feature_flags_per_capability": 10,
    "max_requirements_per_capability": 10
}
