# backend/api/specs/common_schemas.py
"""Gemeinsame OpenAPI-Schema-Definitionen.

Zentralisiert wiederverwendbare Schema-Definitionen für bessere Konsistenz
und Wartbarkeit. Konsolidiert Security-Schemas mit anderen Modulen.
"""

from __future__ import annotations

from typing import Any

from .constants import (
    OPERATION_STATUS_FAILED,
    SECURITY_BEARER_AUTH,
    SERVICE_STATUS_HEALTHY,
    SERVICE_STATUS_UNHEALTHY,
)


def get_security_schemas() -> dict[str, dict[str, Any]]:
    """Gibt standardisierte Security-Schema-Definitionen zurück.

    Konsolidiert Security-Schemas für Wiederverwendung in verschiedenen
    OpenAPI-Spezifikationen.

    Returns:
        Dictionary mit Security-Schema-Definitionen
    """
    return {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer Token für API-Authentifizierung"
        },
        "mtls": {
            "type": "mutualTLS",
            "description": "Mutual TLS Client-Zertifikat-Authentifizierung"
        }
    }


def get_error_response_schema() -> dict[str, Any]:
    """Gibt standardisiertes Error-Response-Schema zurück.

    Returns:
        OpenAPI-Schema für Error-Responses
    """
    return {
        "type": "object",
        "properties": {
            "operation_id": {
                "type": "string",
                "description": "Eindeutige Operation-ID für Tracking"
            },
            "correlation_id": {
                "type": "string",
                "description": "Korrelations-ID für Request-Verfolgung"
            },
            "status": {
                "type": "string",
                "enum": [OPERATION_STATUS_FAILED],
                "description": "Operation-Status"
            },
            "error": {
                "type": "object",
                "properties": {
                    "error_code": {
                        "type": "string",
                        "description": "Maschinenlesbarer Error-Code"
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Menschenlesbare Fehlermeldung"
                    },
                    "error_type": {
                        "type": "string",
                        "description": "Kategorie des Fehlers"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Zeitstempel des Fehlers"
                    }
                },
                "required": ["error_code", "error_message", "error_type", "timestamp"],
                "description": "Detaillierte Fehlerinformationen"
            }
        },
        "required": ["operation_id", "correlation_id", "status", "error"],
        "description": "Standardisierte Error-Response-Struktur"
    }


def get_health_status_schema() -> dict[str, Any]:
    """Gibt Health-Status-Schema zurück.

    Returns:
        OpenAPI-Schema für Health-Status-Responses
    """
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": [SERVICE_STATUS_HEALTHY, SERVICE_STATUS_UNHEALTHY],
                "description": "Service-Health-Status"
            },
            "service": {
                "type": "string",
                "description": "Service-Name"
            },
            "version": {
                "type": "string",
                "description": "Service-Version"
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "Zeitstempel der Health-Prüfung"
            },
            "components": {
                "type": "object",
                "properties": {
                    "agent_registry": {
                        "type": "boolean",
                        "description": "Agent Registry Verfügbarkeit"
                    },
                    "capability_manager": {
                        "type": "boolean",
                        "description": "Capability Manager Verfügbarkeit"
                    }
                },
                "description": "Status der Service-Komponenten"
            },
            "error": {
                "type": "string",
                "description": "Fehlermeldung bei unhealthy Status"
            }
        },
        "required": ["status", "service", "timestamp"],
        "description": "Service Health Status Information"
    }


def get_service_status_schema() -> dict[str, Any]:
    """Gibt detailliertes Service-Status-Schema zurück.

    Returns:
        OpenAPI-Schema für Service-Status-Responses
    """
    return {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "Service-Name"
            },
            "version": {
                "type": "string",
                "description": "Service-Version"
            },
            "status": {
                "type": "string",
                "description": "Aktueller Service-Status"
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "Zeitstempel der Status-Abfrage"
            },
            "statistics": {
                "type": "object",
                "properties": {
                    "operation_cache_size": {
                        "type": "integer",
                        "description": "Anzahl gecachter Operationen"
                    },
                    "available_agents": {
                        "type": "integer",
                        "description": "Anzahl verfügbarer Agents"
                    }
                },
                "description": "Service-Statistiken"
            },
            "capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Liste verfügbarer Capabilities"
            }
        },
        "required": ["service", "version", "status", "timestamp"],
        "description": "Detaillierte Service-Status-Informationen"
    }


def get_common_schemas() -> dict[str, dict[str, Any]]:
    """Gibt alle gemeinsamen Schema-Definitionen zurück.

    Returns:
        Dictionary mit allen wiederverwendbaren Schema-Definitionen
    """
    return {
        "ErrorResponse": get_error_response_schema(),
        "HealthStatus": get_health_status_schema(),
        "ServiceStatus": get_service_status_schema()
    }


def get_security_requirements() -> list[dict[str, list[str]]]:
    """Gibt Standard-Security-Requirements zurück.

    Returns:
        Liste von Security-Requirements für OpenAPI
    """
    return [
        {SECURITY_BEARER_AUTH: []}
    ]
