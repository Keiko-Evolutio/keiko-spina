"""API-Spezifikation-Generator-Paket für Keiko Personal Assistant

Automatische API-Spezifikation-Generierung für Keiko Personal Assistant

Generiert vollständige OpenAPI, AsyncAPI und MCP-Spezifikationen basierend auf
der aktuellen Codebase und integriert diese in die FastAPI-Anwendung.
"""

# Import actual classes and functions that exist
from .api_spec_generator import APISpecificationGenerator, get_api_spec_generator
from .asyncapi_generator import build_asyncapi_dict
from .base_generator import BaseSpecGenerator, BaseSpecificationError
from .constants import (
    ContactInfo,
    DirectoryNames,
    ErrorMessages,
    FileNames,
    SpecConstants,
    get_contact_info,
    get_license_info,
    get_security_schemes,
)
from .openapi_generator import generate_openapi_dict
from .publisher import SpecPublisher
from .utils import (
    ensure_directory,
    extract_spec_metadata,
    read_spec_file,
    validate_required_fields,
    write_spec_file,
)

__all__ = [
    # Generators
    "APISpecificationGenerator",
    "get_api_spec_generator",
    "build_asyncapi_dict",
    "BaseSpecGenerator",
    "BaseSpecificationError",
    "generate_openapi_dict",
    # Publisher
    "SpecPublisher",
    # Constants
    "ContactInfo",
    "DirectoryNames",
    "ErrorMessages",
    "FileNames",
    "SpecConstants",
    "get_contact_info",
    "get_license_info",
    "get_security_schemes",
    # Utils
    "ensure_directory",
    "read_spec_file",
    "write_spec_file",
    "validate_required_fields",
    "extract_spec_metadata",
]
