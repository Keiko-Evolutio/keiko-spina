# backend/voice/common/__init__.py
"""Voice Common Package - Refactored und Enterprise-Ready.

Zentrale Voice-Utilities mit Clean Code Standards, Dependency Injection
und umfassender Error-Handling-Strategie.
"""

from __future__ import annotations

# =============================================================================
# Configuration Management
# =============================================================================
from .config_manager import (
    # Protocols
    ConfigurationSource,
    # Classes
    DatabaseConfigurationSource,
    FallbackConfigurationSource,
    FileConfigurationSource,
    VoiceConfigurationManager,
    # Factory Functions
    create_voice_configuration_manager,
    # Legacy Compatibility
    get_default_configuration,
    get_default_configuration_data,
    get_global_config_manager,
)

# =============================================================================
# Constants und Configuration
# =============================================================================
from .constants import (
    CONFIG_CATEGORY_KEY,
    CONFIG_CONTENT_KEY,
    CONFIG_DEFAULT_KEY,
    # Configuration Keys
    CONFIG_ID_KEY,
    CONFIG_NAME_KEY,
    CONFIG_TOOLS_KEY,
    COSMOS_ENABLE_CROSS_PARTITION,
    # Cosmos DB Konstanten
    COSMOS_PARTITION_KEY_PATH,
    COSMOS_QUERY_DEFAULT_CONFIG,
    DEFAULT_CATEGORY,
    DEFAULT_FILE_ENCODING,
    DEFAULT_PROMPTY_FILENAME,
    DEFAULT_QUERY_TIMEOUT,
    DEFAULT_TOOLS,
    ERROR_COSMOSDB_CONNECT,
    # Error Messages
    ERROR_COSMOSDB_CONNECTION,
    ERROR_COSMOSDB_QUERY,
    ERROR_FALLBACK_CONFIG,
    ERROR_LOAD_DEFAULT_CONFIG,
    ERROR_PROMPTY_NOT_FOUND,
    ERROR_PROMPTY_PARSE,
    ERROR_PROMPTY_READ,
    INLINE_PATH_NAME,
    # Performance
    MAX_FILE_SIZE_BYTES,
    MAX_QUERY_RESULTS,
    # Validation
    MIN_PROMPTY_PARTS,
    PROMPTY_CATEGORY_KEY,
    PROMPTY_DIR,
    PROMPTY_MODEL_KEY,
    PROMPTY_NAME_KEY,
    PROMPTY_TOOLS_KEY,
    READ_CHUNK_SIZE,
    REQUIRED_FRONT_MATTER_KEYS,
    VALID_CATEGORIES,
    # File Konstanten
    VOICE_MODULE_DIR,
    # YAML Konstanten
    YAML_DELIMITER,
    YAML_DELIMITER_COUNT,
    # Konfiguration
    VoiceConfig,
    VoiceFeatureFlags,
)

# =============================================================================
# Cosmos DB Operations
# =============================================================================
from .cosmos_operations import (
    # Classes
    CosmosDBManager,
    # Protocols
    CosmosSettings,
    VoiceCosmosDBManager,
    _query_scalar,
    # Factory Functions
    create_voice_cosmos_manager,
    # Legacy Compatibility
    get_cosmos_container,
)

# =============================================================================
# Exceptions
# =============================================================================
from .exceptions import (
    # Configuration Exceptions
    ConfigurationError,
    ConfigurationNotFoundError,
    CosmosDBConnectionError,
    CosmosDBQueryError,
    # Database Exceptions
    DatabaseError,
    DefaultConfigurationError,
    # File Processing Exceptions
    FileProcessingError,
    InvalidCategoryError,
    PromptyFileNotFoundError,
    PromptyParsingError,
    # Validation Exceptions
    ValidationError,
    # Base Exceptions
    VoiceError,
    YAMLParsingError,
    # Utilities
    wrap_exception,
)

# =============================================================================
# Prompty Parser
# =============================================================================
from .prompty_parser import (
    PromptyParser,
    PromptyValidator,
    # Classes
    YAMLParser,
    # Legacy Compatibility
    _extract_config_from_prompty,
    # Factory Functions
    create_prompty_parser,
    load_prompty_config,
    load_prompty_file,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CONFIG_CATEGORY_KEY",
    "CONFIG_CONTENT_KEY",
    "CONFIG_DEFAULT_KEY",
    "CONFIG_ID_KEY",
    "CONFIG_NAME_KEY",
    "CONFIG_TOOLS_KEY",
    "COSMOS_ENABLE_CROSS_PARTITION",
    "COSMOS_PARTITION_KEY_PATH",
    "COSMOS_QUERY_DEFAULT_CONFIG",
    "DEFAULT_CATEGORY",
    "DEFAULT_FILE_ENCODING",
    "DEFAULT_PROMPTY_FILENAME",
    "DEFAULT_QUERY_TIMEOUT",
    "DEFAULT_TOOLS",
    "ERROR_COSMOSDB_CONNECT",
    "ERROR_COSMOSDB_CONNECTION",
    "ERROR_COSMOSDB_QUERY",
    "ERROR_FALLBACK_CONFIG",
    "ERROR_LOAD_DEFAULT_CONFIG",
    "ERROR_PROMPTY_NOT_FOUND",
    "ERROR_PROMPTY_PARSE",
    "ERROR_PROMPTY_READ",
    "INLINE_PATH_NAME",
    "MAX_FILE_SIZE_BYTES",
    "MAX_QUERY_RESULTS",
    "MIN_PROMPTY_PARTS",
    "PROMPTY_CATEGORY_KEY",
    "PROMPTY_DIR",
    "PROMPTY_MODEL_KEY",
    "PROMPTY_NAME_KEY",
    "PROMPTY_TOOLS_KEY",
    "READ_CHUNK_SIZE",
    "REQUIRED_FRONT_MATTER_KEYS",
    "VALID_CATEGORIES",
    # Constants und Configuration
    "VOICE_MODULE_DIR",
    "YAML_DELIMITER",
    "YAML_DELIMITER_COUNT",
    "ConfigurationError",
    "ConfigurationNotFoundError",
    # Configuration Management
    "ConfigurationSource",
    "CosmosDBConnectionError",
    "CosmosDBManager",
    "CosmosDBQueryError",
    # Cosmos DB Operations
    "CosmosSettings",
    "DatabaseConfigurationSource",
    "DatabaseError",
    "DefaultConfigurationError",
    "FallbackConfigurationSource",
    "FileConfigurationSource",
    "FileProcessingError",
    "InvalidCategoryError",
    "PromptyFileNotFoundError",
    "PromptyParser",
    "PromptyParsingError",
    "PromptyValidator",
    "ValidationError",
    "VoiceConfig",
    "VoiceConfigurationManager",
    "VoiceCosmosDBManager",
    # Exceptions
    "VoiceError",
    "VoiceFeatureFlags",
    # Prompty Parser
    "YAMLParser",
    "YAMLParsingError",
    "_extract_config_from_prompty",
    "_query_scalar",
    "create_prompty_parser",
    "create_voice_configuration_manager",
    "create_voice_cosmos_manager",
    "get_cosmos_container",
    "get_default_configuration",
    "get_default_configuration_data",
    "get_global_config_manager",
    "load_prompty_config",
    "load_prompty_file",
    "wrap_exception",
]

# =============================================================================
# Backward Compatibility
# =============================================================================

# Stelle sicher, dass alle Legacy-Funktionen verfügbar sind
# für bestehenden Code, der das Modul verwendet

# Legacy-Imports für bestehende APIs
__legacy_exports__ = [
    "get_default_configuration",
    "get_default_configuration_data",
    "load_prompty_config",
    "load_prompty_file",
    "get_cosmos_container",
    "_query_scalar",
    "_extract_config_from_prompty",
]

# Füge Legacy-Exports zu __all__ hinzu falls nicht bereits vorhanden
for export in __legacy_exports__:
    if export not in __all__:
        __all__.append(export)

# =============================================================================
# Module Metadata
# =============================================================================

__version__ = "2.0.0"
__author__ = "Keiko Personal Assistant Team"
__description__ = "Voice Common Package - Enterprise-Ready"
