# backend/api/specs/kei_rpc_paths.py
"""KEI-RPC spezifische OpenAPI-Pfad-Definitionen.

Implementiert Endpoint-Builder für KEI-RPC Operationen mit gemeinsamen
Patterns und wiederverwendbaren Komponenten.
"""

from __future__ import annotations

from typing import Any

from .common_parameters import create_endpoint_parameter_references
from .common_responses import create_health_responses, create_operation_responses
from .common_schemas import get_security_requirements
from .constants import (
    CONTENT_TYPE_JSON,
    HEALTH_TAG,
    KEI_RPC_HEALTH_PATH,
    KEI_RPC_STATUS_PATH,
    KEI_RPC_TAG,
    MONITORING_TAG,
    SCHEMA_REF_ACT_REQUEST,
    SCHEMA_REF_ACT_RESPONSE,
    SCHEMA_REF_EXPLAIN_REQUEST,
    SCHEMA_REF_EXPLAIN_RESPONSE,
    SCHEMA_REF_OBSERVE_REQUEST,
    SCHEMA_REF_OBSERVE_RESPONSE,
    SCHEMA_REF_PLAN_REQUEST,
    SCHEMA_REF_PLAN_RESPONSE,
    get_kei_rpc_endpoint_path,
)


def create_request_body(schema_ref: str, examples: dict[str, Any] | None = None) -> dict[str, Any]:
    """Erstellt Request-Body-Definition für KEI-RPC Operation.

    Args:
        schema_ref: Schema-Referenz für Request-Body
        examples: Beispiel-Definitionen

    Returns:
        OpenAPI-Request-Body-Definition
    """
    request_body = {
        "required": True,
        "content": {
            CONTENT_TYPE_JSON: {
                "schema": {"$ref": schema_ref}
            }
        }
    }

    if examples:
        request_body["content"][CONTENT_TYPE_JSON]["examples"] = examples

    return request_body


def create_kei_rpc_operation(
    operation: str,
    description: str,
    schema_ref_request: str,
    schema_ref_response: str,
    examples: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt KEI-RPC Operation-Definition.

    Args:
        operation: Operation-Name (plan, act, observe, explain)
        description: Operation-Beschreibung
        schema_ref_request: Schema-Referenz für Request
        schema_ref_response: Schema-Referenz für Response
        examples: Request-Beispiele

    Returns:
        OpenAPI-Operation-Definition
    """
    return {
        "post": {
            "summary": f"{operation.title()} Operation",
            "description": description,
            "operationId": f"{operation}_operation",
            "tags": [KEI_RPC_TAG],
            "security": get_security_requirements(),
            "parameters": create_endpoint_parameter_references(),
            "requestBody": create_request_body(schema_ref_request, examples),
            "responses": create_operation_responses(
                f"{operation.title()} erfolgreich ausgeführt",
                schema_ref_response
            )
        }
    }


def get_plan_operation_examples() -> dict[str, Any]:
    """Gibt Beispiele für Plan-Operation zurück."""
    return {
        "simple_plan": {
            "summary": "Einfacher Plan",
            "value": {
                "objective": "Erstelle einen Marketingplan für ein neues Produkt",
                "constraints": {
                    "budget": 10000,
                    "timeline": "3 months"
                },
                "resources": {
                    "team_size": 5,
                    "tools": ["analytics", "design"]
                },
                "success_criteria": [
                    "Erhöhung der Markenbekanntheit um 20%",
                    "Generierung von 1000 Leads"
                ]
            }
        },
        "technical_plan": {
            "summary": "Technischer Plan",
            "value": {
                "objective": "Implementiere ein neues Feature in der Anwendung",
                "constraints": {
                    "technology_stack": "Python/FastAPI",
                    "deadline": "2024-02-01"
                },
                "agent_context": {
                    "required_capabilities": ["software_development", "testing"]
                }
            }
        }
    }


def get_act_operation_examples() -> dict[str, Any]:
    """Gibt Beispiele für Act-Operation zurück."""
    return {
        "file_operation": {
            "summary": "Datei-Operation",
            "value": {
                "action": "Erstelle eine CSV-Datei mit Verkaufsdaten",
                "parameters": {
                    "filename": "sales_data.csv",
                    "columns": ["date", "product", "amount"],
                    "data_source": "database"
                }
            }
        },
        "api_call": {
            "summary": "API-Aufruf",
            "value": {
                "action": "Rufe Wetterdaten für Berlin ab",
                "parameters": {
                    "city": "Berlin",
                    "api_key": "weather_api_key"
                },
                "plan_reference": "weather_plan_123"
            }
        }
    }


def get_observe_operation_examples() -> dict[str, Any]:
    """Gibt Beispiele für Observe-Operation zurück."""
    return {
        "system_monitoring": {
            "summary": "System-Monitoring",
            "value": {
                "observation_target": "Server-Performance",
                "observation_type": "metrics",
                "filters": {
                    "time_range": "last_hour",
                    "metrics": ["cpu", "memory", "disk"]
                },
                "include_history": True
            }
        },
        "log_analysis": {
            "summary": "Log-Analyse",
            "value": {
                "observation_target": "Application Logs",
                "observation_type": "error_analysis",
                "filters": {
                    "log_level": "ERROR",
                    "time_range": "last_24h"
                }
            }
        }
    }


def get_explain_operation_examples() -> dict[str, Any]:
    """Gibt Beispiele für Explain-Operation zurück."""
    return {
        "technical_explanation": {
            "summary": "Technische Erklärung",
            "value": {
                "subject": "Wie funktioniert Machine Learning?",
                "explanation_type": "technical",
                "audience": "developers",
                "detail_level": "high",
                "context_references": ["ml_basics", "algorithms"]
            }
        },
        "business_explanation": {
            "summary": "Business-Erklärung",
            "value": {
                "subject": "ROI-Berechnung für Marketing-Kampagnen",
                "explanation_type": "business",
                "audience": "management",
                "detail_level": "medium"
            }
        }
    }


def create_health_operation() -> dict[str, Any]:
    """Erstellt Health-Check-Operation."""
    return {
        "get": {
            "summary": "KEI-RPC Health Check",
            "description": "Gibt Health-Status des KEI-RPC Service zurück",
            "operationId": "kei_rpc_health",
            "tags": [KEI_RPC_TAG, HEALTH_TAG],
            "responses": create_health_responses()
        }
    }


def create_status_operation() -> dict[str, Any]:
    """Erstellt Status-Operation."""
    return {
        "get": {
            "summary": "KEI-RPC Service Status",
            "description": "Gibt detaillierten Service-Status zurück",
            "operationId": "kei_rpc_status",
            "tags": [KEI_RPC_TAG, MONITORING_TAG],
            "security": get_security_requirements(),
            "responses": create_operation_responses(
                "Service-Status erfolgreich abgerufen",
                "#/components/schemas/ServiceStatus"
            )
        }
    }


def get_kei_rpc_paths() -> dict[str, dict[str, Any]]:
    """Gibt alle KEI-RPC Pfad-Definitionen zurück.

    Returns:
        Dictionary mit KEI-RPC OpenAPI-Pfad-Definitionen
    """
    return {
        get_kei_rpc_endpoint_path("plan"): create_kei_rpc_operation(
            "plan",
            "Erstellt einen strukturierten Plan basierend auf Zielbeschreibung",
            SCHEMA_REF_PLAN_REQUEST,
            SCHEMA_REF_PLAN_RESPONSE,
            get_plan_operation_examples()
        ),
        get_kei_rpc_endpoint_path("act"): create_kei_rpc_operation(
            "act",
            "Führt eine spezifische Aktion aus",
            SCHEMA_REF_ACT_REQUEST,
            SCHEMA_REF_ACT_RESPONSE,
            get_act_operation_examples()
        ),
        get_kei_rpc_endpoint_path("observe"): create_kei_rpc_operation(
            "observe",
            "Beobachtet und analysiert Zustände oder Prozesse",
            SCHEMA_REF_OBSERVE_REQUEST,
            SCHEMA_REF_OBSERVE_RESPONSE,
            get_observe_operation_examples()
        ),
        get_kei_rpc_endpoint_path("explain"): create_kei_rpc_operation(
            "explain",
            "Generiert strukturierte Erklärungen",
            SCHEMA_REF_EXPLAIN_REQUEST,
            SCHEMA_REF_EXPLAIN_RESPONSE,
            get_explain_operation_examples()
        ),
        KEI_RPC_HEALTH_PATH: create_health_operation(),
        KEI_RPC_STATUS_PATH: create_status_operation()
    }
