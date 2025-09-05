# backend/api/specs/common_responses.py
"""Gemeinsame OpenAPI-Response-Definitionen.

Zentralisiert wiederverwendbare Response-Definitionen für Standard-HTTP-Responses,
Error-Handling und Response-Header zur Elimination von Code-Duplikaten.
"""

from __future__ import annotations

from typing import Any

from .constants import (
    COMPONENT_REF_BAD_REQUEST,
    COMPONENT_REF_FORBIDDEN,
    COMPONENT_REF_INTERNAL_ERROR,
    COMPONENT_REF_RATE_LIMITED,
    COMPONENT_REF_TIMEOUT,
    COMPONENT_REF_UNAUTHORIZED,
    CONTENT_TYPE_JSON,
    HEADER_AGENT_ID,
    HEADER_CORRELATION_ID,
    HEADER_DURATION_MS,
    HEADER_OPERATION_ID,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_INTERNAL_ERROR,
    HTTP_STATUS_OK,
    HTTP_STATUS_RATE_LIMITED,
    HTTP_STATUS_SERVICE_UNAVAILABLE,
    HTTP_STATUS_TIMEOUT,
    HTTP_STATUS_UNAUTHORIZED,
    SCHEMA_REF_ERROR_RESPONSE,
)


def create_response_header(description: str, schema_type: str = "string") -> dict[str, Any]:
    """Erstellt eine standardisierte Response-Header-Definition.

    Args:
        description: Header-Beschreibung
        schema_type: Schema-Typ (default: "string")

    Returns:
        OpenAPI-Response-Header-Definition
    """
    return {
        "description": description,
        "schema": {"type": schema_type}
    }


def get_standard_response_headers() -> dict[str, dict[str, Any]]:
    """Gibt Standard-Response-Header zurück.

    Returns:
        Dictionary mit Standard-Response-Header-Definitionen
    """
    return {
        HEADER_CORRELATION_ID: create_response_header(
            "Korrelations-ID für Request-Tracking"
        ),
        HEADER_OPERATION_ID: create_response_header(
            "Eindeutige Operation-ID"
        ),
        HEADER_AGENT_ID: create_response_header(
            "ID des ausführenden Agents"
        ),
        HEADER_DURATION_MS: create_response_header(
            "Ausführungsdauer in Millisekunden"
        )
    }


def create_json_response(
    description: str,
    schema_ref: str,
    headers: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Erstellt eine JSON-Response-Definition.

    Args:
        description: Response-Beschreibung
        schema_ref: Schema-Referenz für Response-Body
        headers: Zusätzliche Response-Header

    Returns:
        OpenAPI-Response-Definition
    """
    response = {
        "description": description,
        "content": {
            CONTENT_TYPE_JSON: {
                "schema": {"$ref": schema_ref}
            }
        }
    }

    if headers:
        response["headers"] = headers

    return response


def create_success_response(
    description: str,
    schema_ref: str,
    include_standard_headers: bool = True
) -> dict[str, Any]:
    """Erstellt eine Success-Response mit Standard-Headern.

    Args:
        description: Response-Beschreibung
        schema_ref: Schema-Referenz für Response-Body
        include_standard_headers: Ob Standard-Header eingeschlossen werden sollen

    Returns:
        OpenAPI-Success-Response-Definition
    """
    headers = get_standard_response_headers() if include_standard_headers else None
    return create_json_response(description, schema_ref, headers)


def get_error_response_definition() -> dict[str, Any]:
    """Gibt Standard-Error-Response-Definition zurück.

    Returns:
        OpenAPI-Error-Response-Definition
    """
    return {
        "description": "Fehler bei der Request-Verarbeitung",
        "content": {
            CONTENT_TYPE_JSON: {
                "schema": {"$ref": SCHEMA_REF_ERROR_RESPONSE}
            }
        }
    }


def get_standard_error_responses() -> dict[str, dict[str, Any]]:
    """Gibt alle Standard-Error-Response-Definitionen zurück.

    Returns:
        Dictionary mit Standard-Error-Response-Definitionen
    """
    error_response = get_error_response_definition()

    return {
        "BadRequest": {
            "description": "Ungültige Anfrage - Request-Parameter oder -Body fehlerhaft",
            **error_response
        },
        "Unauthorized": {
            "description": "Nicht autorisiert - Authentifizierung erforderlich",
            **error_response
        },
        "Forbidden": {
            "description": "Zugriff verweigert - Unzureichende Berechtigung",
            **error_response
        },
        "Timeout": {
            "description": "Operation-Timeout - Request-Verarbeitung zu lange",
            **error_response
        },
        "RateLimited": {
            "description": "Rate-Limit überschritten - Zu viele Requests",
            **error_response
        },
        "InternalError": {
            "description": "Interner Server-Fehler - Unerwarteter Fehler",
            **error_response
        }
    }


def get_standard_error_response_references() -> dict[str, dict[str, str]]:
    """Gibt Standard-Error-Response-Referenzen zurück.

    Returns:
        Dictionary mit Error-Response-Referenzen
    """
    return {
        HTTP_STATUS_BAD_REQUEST: {"$ref": COMPONENT_REF_BAD_REQUEST},
        HTTP_STATUS_UNAUTHORIZED: {"$ref": COMPONENT_REF_UNAUTHORIZED},
        HTTP_STATUS_FORBIDDEN: {"$ref": COMPONENT_REF_FORBIDDEN},
        HTTP_STATUS_TIMEOUT: {"$ref": COMPONENT_REF_TIMEOUT},
        HTTP_STATUS_RATE_LIMITED: {"$ref": COMPONENT_REF_RATE_LIMITED},
        HTTP_STATUS_INTERNAL_ERROR: {"$ref": COMPONENT_REF_INTERNAL_ERROR}
    }


def create_operation_responses(
    success_description: str,
    success_schema_ref: str,
    include_standard_errors: bool = True,
    additional_responses: dict[str, dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    """Erstellt vollständige Response-Definition für Operation.

    Args:
        success_description: Beschreibung der Success-Response
        success_schema_ref: Schema-Referenz für Success-Response
        include_standard_errors: Ob Standard-Error-Responses eingeschlossen werden sollen
        additional_responses: Zusätzliche Response-Definitionen

    Returns:
        Dictionary mit vollständigen Response-Definitionen
    """
    responses = {
        HTTP_STATUS_OK: create_success_response(success_description, success_schema_ref)
    }

    if include_standard_errors:
        responses.update(get_standard_error_response_references())

    if additional_responses:
        responses.update(additional_responses)

    return responses


def create_health_responses() -> dict[str, dict[str, Any]]:
    """Erstellt Response-Definitionen für Health-Endpunkte.

    Returns:
        Dictionary mit Health-Response-Definitionen
    """
    return {
        HTTP_STATUS_OK: {
            "description": "Service ist gesund",
            "content": {
                CONTENT_TYPE_JSON: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["healthy"]},
                            "service": {"type": "string"},
                            "version": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "components": {
                                "type": "object",
                                "properties": {
                                    "agent_registry": {"type": "boolean"},
                                    "capability_manager": {"type": "boolean"}
                                }
                            }
                        }
                    }
                }
            }
        },
        HTTP_STATUS_SERVICE_UNAVAILABLE: {
            "description": "Service ist nicht verfügbar",
            "content": {
                CONTENT_TYPE_JSON: {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["unhealthy"]},
                            "service": {"type": "string"},
                            "error": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }
    }


def get_response_components() -> dict[str, dict[str, Any]]:
    """Gibt alle Response-Komponenten für OpenAPI-Spezifikation zurück.

    Returns:
        Dictionary mit Response-Komponenten-Definitionen
    """
    return get_standard_error_responses()
