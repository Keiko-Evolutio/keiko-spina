#!/usr/bin/env python3
"""API Versioning und Documentation für Issue #56 Messaging-first Architecture
Implementiert automatische OpenAPI/AsyncAPI Dokumentation und API-Versionierung

ARCHITEKTUR-COMPLIANCE:
- Automatische Schema-Validierung für API-Requests
- API-Versionierung (v1, v2, etc.)
- OpenAPI/AsyncAPI Dokumentation-Generation
- Backward-Compatibility Management
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, field_validator

from kei_logging import get_logger

logger = get_logger(__name__)

class APIVersion(BaseModel):
    """API Version Information"""
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic Version")
    release_date: datetime = Field(..., description="Release-Datum")
    status: str = Field(..., description="Version Status")
    deprecated: bool = Field(default=False, description="Deprecated Flag")
    sunset_date: datetime | None = Field(None, description="Sunset-Datum")
    changelog: list[str] = Field(default_factory=list, description="Changelog")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed_statuses = ["development", "beta", "stable", "deprecated", "sunset"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v

class APIVersionManager:
    """Manager für API-Versionierung"""

    def __init__(self, versions_config_path: str = "api-contracts/versions.yaml"):
        self.versions_config_path = Path(versions_config_path)
        self.versions: dict[str, APIVersion] = {}
        self.current_version = "v1"
        self.supported_versions = ["v1"]

        # Lade Versions-Konfiguration
        self._load_versions_config()

    def _load_versions_config(self):
        """Lädt API-Versions-Konfiguration"""
        try:
            if self.versions_config_path.exists():
                with open(self.versions_config_path) as f:
                    config = yaml.safe_load(f)

                for version_key, version_data in config.get("versions", {}).items():
                    self.versions[version_key] = APIVersion(**version_data)

                self.current_version = config.get("current_version", "v1")
                self.supported_versions = config.get("supported_versions", ["v1"])

                logger.info(f"API Versions geladen: {list(self.versions.keys())}")
            else:
                # Default-Konfiguration erstellen
                self._create_default_versions_config()

        except Exception as e:
            logger.error(f"Fehler beim Laden der API-Versions-Konfiguration: {e}")
            self._create_default_versions_config()

    def _create_default_versions_config(self):
        """Erstellt Default-Versions-Konfiguration"""
        default_config = {
            "current_version": "v1",
            "supported_versions": ["v1"],
            "versions": {
                "v1": {
                    "version": "1.0.0",
                    "release_date": datetime.now(UTC).isoformat(),
                    "status": "stable",
                    "deprecated": False,
                    "changelog": [
                        "Initial API release",
                        "Platform-SDK Events API",
                        "Platform-SDK Management API",
                        "gRPC Streaming Services"
                    ]
                }
            }
        }

        # Speichere Default-Konfiguration
        try:
            self.versions_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.versions_config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

            # Lade Default-Konfiguration
            self.versions = {
                "v1": APIVersion(**default_config["versions"]["v1"])
            }
            self.current_version = "v1"
            self.supported_versions = ["v1"]

            logger.info("Default API-Versions-Konfiguration erstellt")

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Default-Konfiguration: {e}")

    def get_version_info(self, version: str) -> APIVersion | None:
        """Gibt Version-Informationen zurück"""
        return self.versions.get(version)

    def is_version_supported(self, version: str) -> bool:
        """Prüft ob Version unterstützt wird"""
        return version in self.supported_versions

    def is_version_deprecated(self, version: str) -> bool:
        """Prüft ob Version deprecated ist"""
        version_info = self.get_version_info(version)
        return version_info.deprecated if version_info else False

    def get_all_versions(self) -> dict[str, APIVersion]:
        """Gibt alle Versionen zurück"""
        return self.versions.copy()

class OpenAPIDocumentationGenerator:
    """Generator für OpenAPI-Dokumentation"""

    def __init__(self, app: FastAPI, version_manager: APIVersionManager):
        self.app = app
        self.version_manager = version_manager

    def generate_openapi_schema(self, version: str = "v1") -> dict[str, Any]:
        """Generiert OpenAPI Schema für spezifische Version"""
        try:
            version_info = self.version_manager.get_version_info(version)
            if not version_info:
                raise ValueError(f"Version {version} not found")

            # Basis OpenAPI Schema generieren
            openapi_schema = get_openapi(
                title="Keiko Platform-SDK API",
                version=version_info.version,
                description=self._get_api_description(version_info),
                routes=self.app.routes,
                servers=[
                    {"url": f"/api/{version}", "description": f"API {version}"}
                ]
            )

            # Erweitere Schema mit zusätzlichen Informationen
            openapi_schema["info"]["contact"] = {
                "name": "Keiko Development Team",
                "url": "https://github.com/keiko-dev-team/keiko-personal-assistant"
            }

            openapi_schema["info"]["license"] = {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }

            # Version-spezifische Informationen
            if version_info.deprecated:
                openapi_schema["info"]["x-deprecated"] = True
                if version_info.sunset_date:
                    openapi_schema["info"]["x-sunset-date"] = version_info.sunset_date.isoformat()

            # Changelog hinzufügen
            openapi_schema["info"]["x-changelog"] = version_info.changelog

            # Security Schemes
            openapi_schema["components"]["securitySchemes"] = {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }

            # Global Security
            openapi_schema["security"] = [
                {"ApiKeyAuth": []},
                {"BearerAuth": []}
            ]

            return openapi_schema

        except Exception as e:
            logger.error(f"Fehler beim Generieren des OpenAPI Schemas: {e}")
            raise

    def _get_api_description(self, version_info: APIVersion) -> str:
        """Generiert API-Beschreibung"""
        description = f"""
# Keiko Platform-SDK API {version_info.version}

**Architektur-konforme API für Event-Kommunikation zwischen Platform und SDK**

Diese API gewährleistet strikte Unabhängigkeit zwischen Platform und SDK durch:
- Versionierte HTTP/REST Endpoints
- Schema-validierte Event-Strukturen
- Backward-kompatible API-Evolution
- Keine direkten Messaging-Dependencies

**Status:** {version_info.status.title()}
"""

        if version_info.deprecated:
            description += """
⚠️ **DEPRECATED**: Diese API-Version ist deprecated.
"""
            if version_info.sunset_date:
                description += f"**Sunset-Datum:** {version_info.sunset_date.strftime('%Y-%m-%d')}"

        description += """

## Changelog

"""
        for change in version_info.changelog:
            description += f"- {change}\n"

        return description.strip()

    def save_openapi_schema(self, version: str = "v1", output_path: str | None = None):
        """Speichert OpenAPI Schema in Datei"""
        try:
            schema = self.generate_openapi_schema(version)

            if output_path is None:
                output_path = f"api-contracts/generated/openapi-{version}.yaml"

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

            logger.info(f"OpenAPI Schema gespeichert: {output_file}")

        except Exception as e:
            logger.error(f"Fehler beim Speichern des OpenAPI Schemas: {e}")
            raise

class AsyncAPIDocumentationGenerator:
    """Generator für AsyncAPI-Dokumentation"""

    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager

    def generate_asyncapi_schema(self, version: str = "v1") -> dict[str, Any]:
        """Generiert AsyncAPI Schema für Event-Streaming"""
        try:
            version_info = self.version_manager.get_version_info(version)
            if not version_info:
                raise ValueError(f"Version {version} not found")

            asyncapi_schema = {
                "asyncapi": "3.0.0",
                "info": {
                    "title": "Keiko Platform-SDK Events",
                    "version": version_info.version,
                    "description": "Real-time Event-Streaming zwischen Platform und SDK",
                    "contact": {
                        "name": "Keiko Development Team",
                        "url": "https://github.com/keiko-dev-team/keiko-personal-assistant"
                    },
                    "license": {
                        "name": "MIT",
                        "url": "https://opensource.org/licenses/MIT"
                    }
                },
                "servers": {
                    "platform-events": {
                        "host": "localhost:8000",
                        "protocol": "sse",
                        "description": "Platform Event Streaming via Server-Sent Events"
                    },
                    "platform-websocket": {
                        "host": "localhost:8000",
                        "protocol": "ws",
                        "description": "Platform Event Streaming via WebSocket"
                    }
                },
                "channels": {
                    "/events/agent/{agent_id}/stream": {
                        "address": "/events/agent/{agent_id}/stream",
                        "messages": {
                            "agentEvent": {
                                "$ref": "#/components/messages/AgentEvent"
                            }
                        },
                        "description": "Agent-spezifische Event-Streams",
                        "parameters": {
                            "agent_id": {
                                "description": "Agent-ID",
                                "schema": {
                                    "type": "string",
                                    "pattern": "^agent_[a-zA-Z0-9_]+$"
                                }
                            }
                        }
                    }
                },
                "operations": {
                    "subscribeAgentEvents": {
                        "action": "receive",
                        "channel": {
                            "$ref": "#/channels/~1events~1agent~1{agent_id}~1stream"
                        },
                        "summary": "Abonniere Agent Events",
                        "description": "Empfängt Real-time Agent Events via Server-Sent Events"
                    }
                },
                "components": {
                    "messages": {
                        "AgentEvent": {
                            "name": "AgentEvent",
                            "title": "Agent Event",
                            "summary": "Event von einem Agent",
                            "contentType": "application/json",
                            "payload": {
                                "$ref": "#/components/schemas/AgentEventPayload"
                            }
                        }
                    },
                    "schemas": {
                        "AgentEventPayload": {
                            "type": "object",
                            "properties": {
                                "event_id": {
                                    "type": "string",
                                    "description": "Eindeutige Event-ID"
                                },
                                "event_type": {
                                    "type": "string",
                                    "enum": ["agent.created", "agent.updated", "agent.status_changed", "agent.deleted"]
                                },
                                "agent_id": {
                                    "type": "string",
                                    "pattern": "^agent_[a-zA-Z0-9_]+$"
                                },
                                "timestamp": {
                                    "type": "string",
                                    "format": "date-time"
                                },
                                "data": {
                                    "type": "object",
                                    "description": "Event-spezifische Daten"
                                }
                            },
                            "required": ["event_id", "event_type", "agent_id", "timestamp", "data"]
                        }
                    }
                }
            }

            return asyncapi_schema

        except Exception as e:
            logger.error(f"Fehler beim Generieren des AsyncAPI Schemas: {e}")
            raise

    def save_asyncapi_schema(self, version: str = "v1", output_path: str | None = None):
        """Speichert AsyncAPI Schema in Datei"""
        try:
            schema = self.generate_asyncapi_schema(version)

            if output_path is None:
                output_path = f"api-contracts/generated/asyncapi-{version}.yaml"

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

            logger.info(f"AsyncAPI Schema gespeichert: {output_file}")

        except Exception as e:
            logger.error(f"Fehler beim Speichern des AsyncAPI Schemas: {e}")
            raise

class APIVersionMiddleware:
    """Middleware für API-Versionierung"""

    def __init__(self, version_manager: APIVersionManager):
        self.version_manager = version_manager

    async def __call__(self, request: Request, call_next):
        """Middleware-Handler"""
        try:
            # Extrahiere Version aus URL
            path_parts = request.url.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] == "api":
                version = path_parts[1]

                # Prüfe ob Version unterstützt wird
                if not self.version_manager.is_version_supported(version):
                    raise HTTPException(
                        status_code=404,
                        detail=f"API version {version} not supported. Supported versions: {self.version_manager.supported_versions}"
                    )

                # Prüfe ob Version deprecated ist
                if self.version_manager.is_version_deprecated(version):
                    version_info = self.version_manager.get_version_info(version)
                    warning_header = f"API version {version} is deprecated"
                    if version_info and version_info.sunset_date:
                        warning_header += f". Sunset date: {version_info.sunset_date.strftime('%Y-%m-%d')}"

                    # Füge Warning-Header hinzu
                    response = await call_next(request)
                    response.headers["Warning"] = f'299 - "{warning_header}"'
                    return response

            # Normale Request-Verarbeitung
            response = await call_next(request)
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Fehler in API Version Middleware: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

def setup_api_versioning(app: FastAPI) -> tuple[APIVersionManager, OpenAPIDocumentationGenerator, AsyncAPIDocumentationGenerator]:
    """Setup für API-Versionierung und Dokumentation"""
    # Version Manager
    version_manager = APIVersionManager()

    # Documentation Generators
    openapi_generator = OpenAPIDocumentationGenerator(app, version_manager)
    asyncapi_generator = AsyncAPIDocumentationGenerator(version_manager)

    # Middleware hinzufügen
    app.middleware("http")(APIVersionMiddleware(version_manager))

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In Production einschränken
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom OpenAPI Schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        # Generiere Schema für aktuelle Version
        current_version = version_manager.current_version
        app.openapi_schema = openapi_generator.generate_openapi_schema(current_version)
        return app.openapi_schema

    app.openapi = custom_openapi

    logger.info("API Versioning und Documentation Setup abgeschlossen")

    return version_manager, openapi_generator, asyncapi_generator
