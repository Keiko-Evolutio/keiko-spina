"""API-Spezifikations-Generator für Keiko Personal Assistant

Automatische API-Spezifikations-Generierung für Keiko Personal Assistant

Generiert vollständige OpenAPI, AsyncAPI und MCP-Spezifikationen basierend auf
der aktuellen Codebase und integriert diese in die FastAPI-Anwendung.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

# Import ValidationError from api.common for proper parameter compatibility
from api.common.error_handlers import ValidationError
from kei_logging import LogLinkedError, get_logger, with_log_links

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

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)


class APISpecificationError(LogLinkedError):
    """Fehler bei API-Spezifikations-Generierung."""

    def __init__(self, message: str, spec_type: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.spec_type = spec_type


class SpecificationMetadata(BaseModel):
    """Metadaten für API-Spezifikationen."""
    title: str
    version: str
    description: str
    generated_at: datetime
    generator_version: str
    source_files: list[str]
    checksum: str


class APISpecificationGenerator:
    """Generator für vollständige API-Spezifikationen."""

    def __init__(self, app: FastAPI, specs_dir: str = DirectoryNames.SPECS_DIR):
        """Initialisiert Spezifikations-Generator.

        Args:
            app: FastAPI-Anwendung
            specs_dir: Verzeichnis für Spezifikations-Dateien
        """
        self.app = app
        self.specs_dir = Path(specs_dir)
        self.specs_dir.mkdir(exist_ok=True)

        # Erstelle Unterverzeichnisse
        (self.specs_dir / DirectoryNames.OPENAPI_DIR).mkdir(exist_ok=True)
        (self.specs_dir / DirectoryNames.ASYNCAPI_DIR).mkdir(exist_ok=True)
        (self.specs_dir / DirectoryNames.MCP_DIR).mkdir(exist_ok=True)
        (self.specs_dir / DirectoryNames.GENERATED_DIR).mkdir(exist_ok=True)

        self._base_openapi_spec = None
        self._enhanced_openapi_spec = None
        self._asyncapi_spec = None
        self._mcp_profiles = {}

    @with_log_links(component="api_spec_generator", operation="generate_openapi")
    def generate_openapi_spec(
        self,
        include_enhanced: bool = True,
        include_examples: bool = True,
        include_security: bool = True
    ) -> dict[str, Any]:
        """Generiert vollständige OpenAPI-Spezifikation.

        Args:
            include_enhanced: Enhanced Management APIs einschließen
            include_examples: Beispiele einschließen
            include_security: Security-Schemas einschließen

        Returns:
            Vollständige OpenAPI-Spezifikation
        """
        try:
            # Basis-OpenAPI von FastAPI
            base_spec = get_openapi(
                title="KEI-Agent-Framework API",
                version=SpecConstants.API_VERSION,
                description="Vollständige API für das KEI-Agent-Framework",
                routes=self.app.routes,
            )

            # Erweitere mit benutzerdefinierten Schemas
            enhanced_spec = self._enhance_openapi_spec(
                base_spec,
                include_enhanced=include_enhanced,
                include_examples=include_examples,
                include_security=include_security
            )

            # Lade externe Schema-Definitionen
            external_schemas = self._load_external_schemas()
            if external_schemas:
                enhanced_spec.setdefault("components", {}).setdefault("schemas", {}).update(
                    external_schemas
                )

            # Validiere Spezifikation
            self._validate_openapi_spec(enhanced_spec)

            # Speichere generierte Spezifikation
            self._save_spec(enhanced_spec, DirectoryNames.OPENAPI_DIR, FileNames.OPENAPI_GENERATED_YAML)

            logger.info(
                "OpenAPI-Spezifikation generiert",
                extra={
                    "paths_count": len(enhanced_spec.get("paths", {})),
                    "schemas_count": len(enhanced_spec.get("components", {}).get("schemas", {})),
                    "include_enhanced": include_enhanced,
                    "include_examples": include_examples
                }
            )

            return enhanced_spec

        except Exception as e:
            raise APISpecificationError(
                message=f"{ErrorMessages.OPENAPI_GENERATION_FAILED}: {e}",
                spec_type="openapi",
                cause=e
            )

    def _enhance_openapi_spec(
        self,
        base_spec: dict[str, Any],
        include_enhanced: bool = True,
        include_examples: bool = True,
        include_security: bool = True
    ) -> dict[str, Any]:
        """Erweitert Basis-OpenAPI-Spezifikation.

        Args:
            base_spec: Basis-Spezifikation von FastAPI
            include_enhanced: Enhanced APIs einschließen
            include_examples: Beispiele einschließen
            include_security: Security einschließen

        Returns:
            Erweiterte OpenAPI-Spezifikation
        """
        enhanced_spec = base_spec.copy()

        # Erweitere Info-Sektion
        enhanced_spec["info"].update({
            "contact": get_contact_info(),
            "license": get_license_info(),
            "termsOfService": ContactInfo.TERMS_OF_SERVICE
        })

        # Füge Server hinzu
        enhanced_spec["servers"] = [
            {
                "url": SpecConstants.LOCAL_DEV_SERVER,
                "description": "Lokale Entwicklungsumgebung"
            },
            {
                "url": SpecConstants.PRODUCTION_SERVER,
                "description": "Produktionsumgebung"
            }
        ]

        # Füge Security-Schemas hinzu
        if include_security:
            enhanced_spec["components"] = enhanced_spec.get("components", {})
            enhanced_spec["components"]["securitySchemes"] = get_security_schemes()

            # Globale Security
            enhanced_spec["security"] = [
                {"BearerAuth": []},
                {"ApiKeyAuth": []}
            ]

        # Füge Error-Response-Schemas hinzu
        enhanced_spec["components"] = enhanced_spec.get("components", {})
        enhanced_spec["components"]["responses"] = self._get_error_responses()

        # Füge Enhanced Management APIs hinzu
        if include_enhanced:
            enhanced_paths = self._get_enhanced_api_paths()
            enhanced_spec["paths"].update(enhanced_paths)

        # Füge Beispiele hinzu
        if include_examples:
            self._add_examples_to_spec(enhanced_spec)

        # Füge Tags hinzu
        enhanced_spec["tags"] = [
            {
                "name": "Agent Registry",
                "description": "Agent-Registrierung und -Management"
            },
            {
                "name": "Agent Discovery",
                "description": "Erweiterte Agent-Discovery"
            },
            {
                "name": "Tenant Management",
                "description": "Multi-Tenant-Verwaltung"
            },
            {
                "name": "Rollout Management",
                "description": "Deployment und Rollout-Strategien"
            },
            {
                "name": "Statistics",
                "description": "Statistiken und Metriken"
            },
            {
                "name": "Health",
                "description": "Gesundheitschecks"
            },
            {
                "name": "Maintenance",
                "description": "Wartung und Cleanup"
            }
        ]

        return enhanced_spec

    def _get_error_responses(self) -> dict[str, Any]:
        """Erstellt Standard-Error-Response-Schemas.

        Returns:
            Error-Response-Definitionen
        """
        return {
            "ValidationError": {
                "description": "Validierungsfehler",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["error", "error_code", "message"],
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "ValidationError"
                                },
                                "error_code": {
                                    "type": "string",
                                    "example": "VALIDATION_FAILED"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Eingabedaten sind ungültig"
                                },
                                "details": {
                                    "type": "object",
                                    "description": "Detaillierte Validierungsfehler"
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri",
                                    "description": "Link zu detaillierten Logs"
                                },
                                "timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                }
                            }
                        }
                    }
                }
            },
            "UnauthorizedError": {
                "description": "Authentifizierung erforderlich",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "Unauthorized"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Authentifizierung erforderlich"
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    }
                }
            },
            "ForbiddenError": {
                "description": "Keine Berechtigung",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "Forbidden"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Keine Berechtigung für diese Operation"
                                },
                                "required_permission": {
                                    "type": "string",
                                    "example": "agent:write"
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    }
                }
            },
            "NotFoundError": {
                "description": "Ressource nicht gefunden",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "NotFound"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Ressource nicht gefunden"
                                },
                                "resource_type": {
                                    "type": "string",
                                    "example": "agent"
                                },
                                "resource_id": {
                                    "type": "string",
                                    "example": "chatbot-pro"
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    }
                }
            },
            "ConflictError": {
                "description": "Konflikt mit bestehendem Zustand",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "Conflict"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Ressource bereits vorhanden"
                                },
                                "conflicting_resource": {
                                    "type": "string",
                                    "example": "agent:chatbot-pro:1.0.0"
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    }
                }
            },
            "UnprocessableEntityError": {
                "description": "Unverarbeitbare Entität",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "UnprocessableEntity"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Eingabedaten können nicht verarbeitet werden"
                                },
                                "validation_errors": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "field": {"type": "string"},
                                            "message": {"type": "string"},
                                            "value": {"type": "string"}
                                        }
                                    }
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    }
                }
            },
            "InternalServerError": {
                "description": "Interner Serverfehler",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {
                                    "type": "string",
                                    "example": "InternalServerError"
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Ein interner Fehler ist aufgetreten"
                                },
                                "error_id": {
                                    "type": "string",
                                    "format": "uuid",
                                    "description": "Eindeutige Fehler-ID für Support"
                                },
                                "log_link": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                        }
                    }
                }
            }
        }

    def _get_enhanced_api_paths(self) -> dict[str, Any]:
        """Holt Enhanced API-Pfade aus externen Spezifikationen.

        Returns:
            Enhanced API-Pfade
        """
        try:
            # Lade externe OpenAPI-Spezifikation
            external_spec_path = self.specs_dir / "openapi" / "management_api.yaml"

            if external_spec_path.exists():
                with open(external_spec_path, encoding="utf-8") as f:
                    external_spec = yaml.safe_load(f)

                return external_spec.get("paths", {})

            return {}

        except Exception as e:
            logger.warning(f"Konnte externe API-Pfade nicht laden: {e}")
            return {}

    def _load_external_schemas(self) -> dict[str, Any]:
        """Lädt externe Schema-Definitionen.

        Returns:
            Externe Schemas
        """
        try:
            schemas_path = self.specs_dir / "openapi" / "components" / "schemas.yaml"

            if schemas_path.exists():
                with open(schemas_path, encoding="utf-8") as f:
                    schemas_data = yaml.safe_load(f)

                return schemas_data.get("components", {}).get("schemas", {})

            return {}

        except Exception as e:
            logger.warning(f"Konnte externe Schemas nicht laden: {e}")
            return {}

    def _add_examples_to_spec(self, spec: dict[str, Any]) -> None:
        """Fügt Beispiele zu OpenAPI-Spezifikation hinzu.

        Args:
            spec: OpenAPI-Spezifikation
        """
        # Beispiele für Request/Response-Bodies hinzufügen
        for path_item in spec.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    # Request-Body-Beispiele
                    if "requestBody" in operation:
                        self._add_request_examples(operation["requestBody"])

                    # Response-Beispiele
                    if "responses" in operation:
                        self._add_response_examples(operation["responses"])

    def _add_request_examples(self, request_body: dict[str, Any]) -> None:
        """Fügt Beispiele zu Request-Body hinzu."""
        content = request_body.get("content", {})

        for media_content in content.values():
            if "schema" in media_content and "examples" not in media_content:
                # Generiere Beispiele basierend auf Schema
                schema = media_content["schema"]
                examples = self._generate_examples_from_schema(schema)

                if examples:
                    media_content["examples"] = examples

    def _add_response_examples(self, responses: dict[str, Any]) -> None:
        """Fügt Beispiele zu Responses hinzu."""
        for response in responses.values():
            if isinstance(response, dict) and "content" in response:
                content = response["content"]

                for media_content in content.values():
                    if "schema" in media_content and "examples" not in media_content:
                        schema = media_content["schema"]
                        examples = self._generate_examples_from_schema(schema)

                        if examples:
                            media_content["examples"] = examples

    def _generate_examples_from_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generiert Beispiele basierend auf JSON-Schema.

        Args:
            schema: JSON-Schema

        Returns:
            Generierte Beispiele
        """
        # Vereinfachte Beispiel-Generierung
        # In einer vollständigen Implementierung würde hier eine
        # sophistiziertere Schema-zu-Beispiel-Konvertierung stattfinden

        if "example" in schema:
            return {
                "default": {
                    "summary": "Standard-Beispiel",
                    "value": schema["example"]
                }
            }

        return {}

    def _validate_openapi_spec(self, spec: dict[str, Any]) -> None:
        """Validiert OpenAPI-Spezifikation.

        Args:
            spec: OpenAPI-Spezifikation

        Raises:
            ValidationError: Bei ungültiger Spezifikation
        """
        required_fields = ["openapi", "info", "paths"]

        for field in required_fields:
            if field not in spec:
                raise ValidationError(
                    message=f"Erforderliches Feld '{field}' fehlt in OpenAPI-Spezifikation",
                    field=field,
                    spec_type="openapi"
                )

        # Validiere OpenAPI-Version
        openapi_version = spec.get("openapi", "")
        if not openapi_version.startswith("3."):
            raise ValidationError(
                message=f"Unsupported OpenAPI version: {openapi_version}",
                field="openapi",
                value=openapi_version,
                spec_type="openapi"
            )

    def _save_spec(
        self,
        spec: dict[str, Any],
        spec_type: str,
        filename: str
    ) -> None:
        """Speichert Spezifikation in Datei.

        Args:
            spec: Spezifikation
            spec_type: Spezifikations-Typ
            filename: Dateiname
        """
        output_path = self.specs_dir / "generated" / filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"{spec_type.upper()}-Spezifikation gespeichert: {output_path}")

        except Exception as e:
            logger.exception(f"Fehler beim Speichern der {spec_type}-Spezifikation: {e}")
            raise

    @with_log_links(component="api_spec_generator", operation="generate_asyncapi")
    def generate_asyncapi_spec(self) -> dict[str, Any]:
        """Generiert AsyncAPI-Spezifikation für Event-Streaming.

        Returns:
            AsyncAPI-Spezifikation
        """
        try:
            # Lade Basis-AsyncAPI-Spezifikation
            asyncapi_path = self.specs_dir / "asyncapi" / "events_api.yaml"

            if not asyncapi_path.exists():
                raise APISpecificationError(
                    message="AsyncAPI-Basis-Spezifikation nicht gefunden",
                    spec_type="asyncapi",
                    file_path=str(asyncapi_path)
                )

            with open(asyncapi_path, encoding="utf-8") as f:
                asyncapi_spec = yaml.safe_load(f)

            # Erweitere mit dynamischen Event-Schemas
            self._enhance_asyncapi_spec(asyncapi_spec)

            # Speichere generierte Spezifikation
            self._save_spec(asyncapi_spec, "asyncapi", "kei_events_generated.yaml")

            logger.info(
                "AsyncAPI-Spezifikation generiert",
                extra={
                    "channels_count": len(asyncapi_spec.get("channels", {})),
                    "operations_count": len(asyncapi_spec.get("operations", {}))
                }
            )

            return asyncapi_spec

        except Exception as e:
            raise APISpecificationError(
                message=f"AsyncAPI-Generierung fehlgeschlagen: {e}",
                spec_type="asyncapi",
                cause=e
            )

    def _enhance_asyncapi_spec(self, spec: dict[str, Any]) -> None:
        """Erweitert AsyncAPI-Spezifikation mit dynamischen Inhalten.

        Args:
            spec: AsyncAPI-Spezifikation
        """
        # Füge aktuelle Timestamp hinzu
        spec["info"]["x-generated-at"] = datetime.now(UTC).isoformat()

        # Erweitere mit aktuellen Event-Schemas aus der Codebase
        # Dynamische Schema-Extraktion wird in zukünftigen Versionen implementiert

        # Füge Server-Konfiguration basierend auf Umgebung hinzu
        # Umgebungsbasierte Server-Konfiguration wird in zukünftigen Versionen implementiert

    def generate_all_specs(self) -> dict[str, dict[str, Any]]:
        """Generiert alle API-Spezifikationen.

        Returns:
            Dictionary mit allen generierten Spezifikationen
        """
        specs = {}

        try:
            # OpenAPI-Spezifikation
            specs["openapi"] = self.generate_openapi_spec()

            # AsyncAPI-Spezifikation
            specs["asyncapi"] = self.generate_asyncapi_spec()

            # MCP-Profile
            specs["mcp"] = self.generate_mcp_profiles()

            logger.info(
                "Alle API-Spezifikationen generiert",
                extra={
                    "openapi_paths": len(specs["openapi"].get("paths", {})),
                    "asyncapi_channels": len(specs["asyncapi"].get("channels", {})),
                    "mcp_profiles": len(specs["mcp"])
                }
            )

            return specs

        except Exception as e:
            raise APISpecificationError(
                message=f"Spezifikations-Generierung fehlgeschlagen: {e}",
                cause=e
            )

    def generate_mcp_profiles(self) -> dict[str, Any]:
        """Generiert MCP-Profile für alle Capabilities.

        Returns:
            Dictionary mit MCP-Profilen
        """
        profiles = {}

        # Lade bestehende MCP-Profile
        mcp_dir = self.specs_dir / "mcp" / "capabilities"

        if mcp_dir.exists():
            for profile_file in mcp_dir.glob("*.yaml"):
                try:
                    with open(profile_file, encoding="utf-8") as f:
                        profile = yaml.safe_load(f)

                    capability_name = profile.get("capability", {}).get("name", profile_file.stem)
                    profiles[capability_name] = profile

                except Exception as e:
                    logger.warning(f"Konnte MCP-Profil nicht laden: {profile_file}: {e}")

        return profiles


# Globale Generator-Instanz
_api_spec_generator: APISpecificationGenerator | None = None


def get_api_spec_generator(app: FastAPI) -> APISpecificationGenerator:
    """Holt oder erstellt API-Spezifikations-Generator.

    Args:
        app: FastAPI-Anwendung

    Returns:
        API-Spezifikations-Generator
    """
    global _api_spec_generator

    if _api_spec_generator is None:
        _api_spec_generator = APISpecificationGenerator(app)

    return _api_spec_generator
