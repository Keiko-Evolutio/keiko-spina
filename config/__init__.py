# backend/config/__init__.py
"""Konfigurationsmanagement für Keiko Personal Assistant.

Die Konfiguration ist in spezialisierte Module aufgeteilt.
Legacy-Kompatibilität wird durch die Haupt-Settings-Klasse gewährleistet.
"""

# Haupt-Settings
# Legacy Config Modules
from .api_versioning_config import DEPRECATION_RULES, DeprecationRule
from .azure_settings import AzureSettings, azure_settings
from .communication_settings import CommunicationSettings, communication_settings

# Utilities
from .constants import *  # Alle Konstanten exportieren

# Spezialisierte Settings-Module
from .core_settings import CoreSettings, core_settings
from .database_settings import DatabaseSettings, database_settings
from .domain_validation_config import DomainValidationConfig, load_domain_validation_config
from .env_utils import (
    get_env_bool,
    get_env_enum,
    get_env_float,
    get_env_int,
    get_env_list,
    get_env_str,
    load_env_config,
    validate_required_env_vars,
)
from .kei_mcp_config import KEI_MCP_SETTINGS, KEIMCPConfig, KEIMCPSettings
from .monitoring_settings import MonitoringSettings, monitoring_settings
from .mtls_config import MTLS_SETTINGS, MTLSSettings
from .oidc_config import OIDC_SETTINGS, OIDCSettings
from .security_settings import SecuritySettings, security_settings
from .settings import Settings, create_test_settings, get_settings, settings

# Einheitliche Rate-Limiting-Konfiguration
from .unified_rate_limiting import (
    RateLimitPolicy,
    RateLimitTier,
    UnifiedRateLimitConfig,
    get_unified_rate_limit_config,
    reload_unified_rate_limit_config,
)
from .voice_config import (
    VoiceDetectionConfig,
    VoiceServiceSettings,
    get_voice_config,
    reload_voice_config,
)
from .websocket_auth_config import WEBSOCKET_AUTH_CONFIG, WebSocketAuthConfig

__version__ = "2.0.0"
__author__ = "Keiko Development Team"

__all__ = [
    "DEPRECATION_RULES",
    "KEI_MCP_SETTINGS",
    "MTLS_SETTINGS",
    "OIDC_SETTINGS",
    "WEBSOCKET_AUTH_CONFIG",
    "AzureSettings",
    "CommunicationSettings",
    # Spezialisierte Settings
    "CoreSettings",
    "DatabaseSettings",
    # Legacy Config Modules
    "DeprecationRule",
    "DomainValidationConfig",
    "KEIMCPConfig",
    "KEIMCPSettings",
    "MTLSSettings",
    "MonitoringSettings",
    "OIDCSettings",
    "RateLimitPolicy",
    "RateLimitTier",
    "SecuritySettings",
    # Haupt-Settings
    "Settings",
    # Rate Limiting
    "UnifiedRateLimitConfig",
    # Voice Configuration
    "VoiceDetectionConfig",
    "VoiceServiceSettings",
    "WebSocketAuthConfig",
    "azure_settings",
    "communication_settings",
    "core_settings",
    "create_test_settings",
    "database_settings",
    "get_env_bool",
    "get_env_enum",
    "get_env_float",
    "get_env_int",
    "get_env_list",
    # Utilities
    "get_env_str",
    "get_settings",
    "get_unified_rate_limit_config",
    "get_voice_config",
    "load_domain_validation_config",
    "load_env_config",
    "monitoring_settings",
    "reload_unified_rate_limit_config",
    "reload_voice_config",
    "security_settings",
    "settings",
    "validate_required_env_vars",
]
