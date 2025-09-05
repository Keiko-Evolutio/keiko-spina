"""Voice Prompty Module.

Dediziertes Modul für Prompty-Template-Management und -Verarbeitung
mit Enterprise-Grade Code-Qualität und klarer Separation of Concerns.

Dieses Modul bietet:
- Template-Loading und -Parsing
- Konfigurierbare Prompty-Templates
- Validation und Error-Handling
- Performance-optimierte Template-Verarbeitung
"""

from __future__ import annotations

from .constants import (
    ALLOWED_TEMPLATE_CATEGORIES,
    # Configuration Constants
    DEFAULT_TEMPLATE_CONFIG,
    # Template Constants
    DEFAULT_TEMPLATE_NAME,
    # Validation Constants
    MAX_TEMPLATE_SIZE_BYTES,
    MIN_TEMPLATE_SIZE_BYTES,
    OPTIONAL_METADATA_FIELDS,
    REQUIRED_METADATA_FIELDS,
    SUPPORTED_TEMPLATE_VERSIONS,
    TEMPLATE_FILE_EXTENSION,
)
from .exceptions import (
    InvalidTemplateError,
    MetadataError,
    # Base Exceptions
    PromptyError,
    PromptyParsingError,
    PromptyTemplateError,
    PromptyValidationError,
    RenderingError,
    # Specific Exceptions
    TemplateNotFoundError,
)

# =============================================================================
# Core Exports
# =============================================================================
from .parser import (
    ParseResult,
    # Classes
    PromptyParser,
    ValidationResult,
    # Factory Functions
    create_parser,
    extract_metadata,
    # Utility Functions
    parse_prompty_file,
    validate_prompty_content,
)

# =============================================================================
# Version Information
# =============================================================================

__version__ = "1.0.0"
__author__ = "Keiko Personal Assistant"
__description__ = "Enterprise-Grade Prompty Template Management"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ALLOWED_TEMPLATE_CATEGORIES",
    "DEFAULT_TEMPLATE_CONFIG",
    # Constants
    "DEFAULT_TEMPLATE_NAME",
    "MAX_TEMPLATE_SIZE_BYTES",
    "MIN_TEMPLATE_SIZE_BYTES",
    "OPTIONAL_METADATA_FIELDS",
    "REQUIRED_METADATA_FIELDS",
    "SUPPORTED_TEMPLATE_VERSIONS",
    "TEMPLATE_FILE_EXTENSION",
    "InvalidTemplateError",
    "MetadataError",
    "ParseResult",
    # Exceptions
    "PromptyError",
    # Parsing
    "PromptyParser",
    "PromptyParsingError",
    "PromptyTemplateError",
    "PromptyValidationError",
    "RenderingError",
    "TemplateNotFoundError",
    "ValidationResult",
    "create_parser",
    "extract_metadata",
    "parse_prompty_file",
    "validate_prompty_content",
]
