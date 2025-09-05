"""Zentrale Konstanten für das specs-Modul.

Enthält alle Magic Numbers, Versionsnummern, URLs und andere Hard-coded Values
die im specs-Modul verwendet werden.
"""

from __future__ import annotations

from typing import Any


class SpecConstants:
    """Zentrale Konstanten-Klasse für Spezifikations-Generierung."""

    # API-Versionen
    API_VERSION = "1.0.0"
    OPENAPI_VERSION = "3.1.0"
    ASYNCAPI_VERSION = "3.0.0"

    # Generator-Versionen
    OPENAPI_GENERATOR_VERSION = "6.6.0"
    ASYNCAPI_GENERATOR_VERSION = "1.9.0"

    # HTTP-Konfiguration
    HTTP_TIMEOUT = 10.0
    HTTP_STATUS_BAD_REQUEST = 400

    # Server-URLs
    LOCAL_DEV_SERVER = "http://localhost:8000"
    PRODUCTION_SERVER = "https://api.Keiko Personal Assistantcom"

    # NATS/Kafka-Konfiguration
    NATS_DEV_HOST = "localhost:4222"
    KAFKA_DEV_HOST = "localhost:9092"

    # GitHub-URLs
    GITHUB_BASE_URL = "https://github.com/kei-agent-framework"
    GITHUB_DOCS_URL = "https://docs.Keiko Personal Assistantcom"

    # Package-Konfiguration
    PYTHON_MIN_VERSION = "3.8"
    PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]

    # File-Encoding
    DEFAULT_ENCODING = "utf-8"

    # Beispiel-Werte
    EXAMPLE_AGENT_ID = "agent:chatbot-pro:1.0.0"

    # Webhook-Konfiguration
    WEBHOOK_INBOUND_PATTERN = "kei.webhook.inbound.*.v1"

    # Bus-Patterns
    BUS_EVENTS_PATTERN = "kei.events.*"
    BUS_RPC_PATTERN = "kei.rpc.*"
    BUS_TASKS_PATTERN = "kei.tasks.*"
    BUS_A2A_PATTERN = "kei.a2a.*"
    BUS_DLQ_PATTERN = "kei.dlq.>"
    BUS_PARKING_PATTERN = "kei.parking.*"


class ErrorMessages:
    """Zentrale Error-Messages für das specs-Modul."""

    OPENAPI_GENERATION_FAILED = "OpenAPI-Generierung fehlgeschlagen"
    ASYNCAPI_GENERATION_FAILED = "AsyncAPI-Generierung fehlgeschlagen"
    SDK_GENERATION_FAILED = "SDK-Generierung fehlgeschlagen"
    SPEC_GENERATION_FAILED = "Spezifikations-Generierung fehlgeschlagen"

    PYTHON_SDK_GENERATION_FAILED = "Python-SDK-Generierung fehlgeschlagen"
    TYPESCRIPT_SDK_GENERATION_FAILED = "TypeScript-SDK-Generierung fehlgeschlagen"

    MISSING_ENDPOINT = "missing_endpoint"
    UNKNOWN_TARGET = "unknown_target"

    ASYNCAPI_BASE_NOT_FOUND = "AsyncAPI-Basis-Spezifikation nicht gefunden"
    UNSUPPORTED_OPENAPI_VERSION = "Unsupported OpenAPI version"
    REQUIRED_FIELD_MISSING = "Erforderliches Feld fehlt in OpenAPI-Spezifikation"


class FileNames:
    """Zentrale Dateinamen für das specs-Modul."""

    # OpenAPI-Dateien
    OPENAPI_GENERATED_YAML = "kei_api_generated.yaml"
    OPENAPI_MANAGEMENT_YAML = "management_api.yaml"
    OPENAPI_JSON = "openapi.json"
    OPENAPI_YAML = "openapi.yaml"

    # AsyncAPI-Dateien
    ASYNCAPI_BASE_YAML = "events_api.yaml"
    ASYNCAPI_GENERATED_YAML = "kei_events_generated.yaml"
    ASYNCAPI_JSON = "asyncapi.json"
    ASYNCAPI_YAML = "asyncapi.yaml"

    # SDK-Dateien
    PYPROJECT_TOML = "pyproject.toml"
    MAKEFILE = "Makefile"
    GITIGNORE = ".gitignore"
    TSCONFIG_JSON = "tsconfig.json"
    PACKAGE_JSON = "package.json"

    # Beispiel-Dateien
    BASIC_EXAMPLE_PY = "basic_example.py"
    DISCOVERY_EXAMPLE_PY = "discovery_example.py"
    EXAMPLES_README_MD = "README.md"


class DirectoryNames:
    """Zentrale Verzeichnisnamen für das specs-Modul."""

    SPECS_DIR = "specs"
    OPENAPI_DIR = "openapi"
    ASYNCAPI_DIR = "asyncapi"
    MCP_DIR = "mcp"
    GENERATED_DIR = "generated"
    TEMPLATES_DIR = "templates"
    CAPABILITIES_DIR = "capabilities"
    EXAMPLES_DIR = "examples"
    GENERATED_SDKS_DIR = "generated_sdks"


class SecuritySchemes:
    """Security-Schema-Definitionen."""

    BEARER_AUTH = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    }

    API_KEY_AUTH = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API-Key für Service-to-Service-Authentifizierung"
    }

    MTLS = {
        "type": "mutualTLS"
    }

    CLIENT_CERTIFICATE = {
        "type": "clientCertificate",
        "description": "mTLS Client‑Zertifikate"
    }

    HMAC_HEADER = {
        "type": "httpApiKey",
        "name": "x-kei-signature",
        "in": "header",
        "description": "HMAC‑Signatur des Payloads"
    }


class ContactInfo:
    """Kontakt-Informationen für API-Spezifikationen."""

    TEAM_NAME = "KEI-Agent-Framework Development Team"
    TEAM_EMAIL = "dev@Keiko Personal Assistantcom"
    TEAM_URL = "https://github.com/kei-agent-framework"

    LICENSE_NAME = "MIT License"
    LICENSE_URL = "https://opensource.org/licenses/MIT"

    TERMS_OF_SERVICE = "https://Keiko Personal Assistantcom/terms"


class PythonRequirements:
    """Python-Package-Requirements."""

    CORE_DEPENDENCIES = [
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0"
    ]

    ASYNC_DEPENDENCIES = [
        "aiohttp>=3.8.0",
        "asyncio-mqtt>=0.11.0"
    ]

    DEV_DEPENDENCIES = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0"
    ]


class TypeScriptConfig:
    """TypeScript-Konfiguration."""

    COMPILER_OPTIONS = {
        "target": "ES2020",
        "module": "commonjs",
        "lib": ["ES2020"],
        "outDir": "./dist",
        "rootDir": "./src",
        "strict": True,
        "esModuleInterop": True,
        "skipLibCheck": True,
        "forceConsistentCasingInFileNames": True,
        "declaration": True,
        "declarationMap": True,
        "sourceMap": True
    }


# Convenience-Funktionen für häufig verwendete Konstanten
def get_api_version() -> str:
    """Gibt die aktuelle API-Version zurück."""
    return SpecConstants.API_VERSION


def get_openapi_version() -> str:
    """Gibt die OpenAPI-Version zurück."""
    return SpecConstants.OPENAPI_VERSION


def get_asyncapi_version() -> str:
    """Gibt die AsyncAPI-Version zurück."""
    return SpecConstants.ASYNCAPI_VERSION


def get_security_schemes() -> dict[str, Any]:
    """Gibt alle Security-Schemas zurück."""
    return {
        "BearerAuth": SecuritySchemes.BEARER_AUTH,
        "ApiKeyAuth": SecuritySchemes.API_KEY_AUTH,
        "mtls": SecuritySchemes.MTLS,
        "clientCertificate": SecuritySchemes.CLIENT_CERTIFICATE,
        "hmacHeader": SecuritySchemes.HMAC_HEADER
    }


def get_contact_info() -> dict[str, str]:
    """Gibt Kontakt-Informationen zurück."""
    return {
        "name": ContactInfo.TEAM_NAME,
        "email": ContactInfo.TEAM_EMAIL,
        "url": ContactInfo.TEAM_URL
    }


def get_license_info() -> dict[str, str]:
    """Gibt Lizenz-Informationen zurück."""
    return {
        "name": ContactInfo.LICENSE_NAME,
        "url": ContactInfo.LICENSE_URL
    }
