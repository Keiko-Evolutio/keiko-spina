"""Gemeinsame Pydantic-Modelle für das API-Versionierungsmodul.

Konsolidierte Base-Klassen und Modelle zur Reduzierung von Code-Duplikation
und Verbesserung der Konsistenz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_CATEGORY,
    HEALTH_STATUS_HEALTHY,
    MAX_DESCRIPTION_LENGTH,
    MAX_FUNCTION_NAME_LENGTH,
    MIN_NAME_LENGTH,
    PARAM_TYPE_ANY,
    WEBHOOK_STATUS_ACCEPTED,
)

if TYPE_CHECKING:
    from datetime import datetime


class BaseVersionedModel(BaseModel):
    """Basis-Modell für alle versionierten API-Modelle."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )


class HealthComponent(BaseVersionedModel):
    """Komponentenstatus für Health-Checks."""

    name: str = Field(..., min_length=MIN_NAME_LENGTH)
    status: str = Field(
        default=HEALTH_STATUS_HEALTHY,
        description="healthy|degraded|unhealthy|unavailable"
    )
    details: dict[str, Any] | None = Field(default=None)


class BaseHealthResponse(BaseVersionedModel):
    """Basis-Klasse für Health-Responses."""

    service: str = Field(..., min_length=MIN_NAME_LENGTH)
    version: str = Field(..., min_length=MIN_NAME_LENGTH)
    status: str = Field(..., min_length=MIN_NAME_LENGTH)
    components: list[HealthComponent] = Field(default_factory=list)


class HealthV2Response(BaseHealthResponse):
    """Erweiterte Health-Response für API v2."""

    sla_metrics: dict[str, Any] | None = Field(default=None)
    trend_data: dict[str, Any] | None = Field(default=None)


class BaseAgentModel(BaseVersionedModel):
    """Basis-Modell für Agent-Informationen."""

    id: str = Field(..., min_length=MIN_NAME_LENGTH)
    name: str = Field(..., min_length=MIN_NAME_LENGTH)


class AgentSummary(BaseAgentModel):
    """Agent-Summary für API v2."""

    category: str | None = Field(default=None)


class BaseWebhookModel(BaseVersionedModel):
    """Basis-Modell für Webhook-Responses."""

    status: str = Field(default=WEBHOOK_STATUS_ACCEPTED)


class WebhookAck(BaseWebhookModel):
    """Webhook-Bestätigung für API v2."""

    received: int = Field(default=1, ge=0)


class ParameterSpec(BaseVersionedModel):
    """Spezifikation eines Funktionsparameters."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "text",
                    "type": "string",
                    "description": "Textinhalt",
                    "required": True,
                    "default": None
                }
            ]
        }
    )

    name: str = Field(..., min_length=MIN_NAME_LENGTH, max_length=MAX_FUNCTION_NAME_LENGTH)
    type: str = Field(
        default=PARAM_TYPE_ANY,
        pattern=r"^(string|number|integer|boolean|object|array|any)$"
    )
    description: str = Field(default="", max_length=MAX_DESCRIPTION_LENGTH)
    required: bool = Field(default=True)
    default: Any | None = Field(default=None)


class BaseFunctionModel(BaseVersionedModel):
    """Basis-Modell für Funktions-Schemas."""

    name: str = Field(..., min_length=MIN_NAME_LENGTH, max_length=MAX_FUNCTION_NAME_LENGTH)
    description: str = Field(..., max_length=MAX_DESCRIPTION_LENGTH)
    parameters: list[ParameterSpec] = Field(default_factory=list)


class V2FunctionSchema(BaseFunctionModel):
    """Funktionsschema für API v2."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "echo",
                    "description": "Spiegelt Text wider",
                    "parameters": [
                        {
                            "name": "text",
                            "type": "string",
                            "description": "Inhalt",
                            "required": True
                        }
                    ],
                    "return_type": "string",
                    "category": "built_in",
                    "tags": ["demo"]
                }
            ]
        }
    )

    return_type: str = Field(default=PARAM_TYPE_ANY)
    category: str = Field(default=DEFAULT_CATEGORY)
    tags: list[str] = Field(default_factory=list)


class BaseExecutionRequest(BaseVersionedModel):
    """Basis-Modell für Ausführungsanfragen."""

    function: str = Field(..., min_length=MIN_NAME_LENGTH, max_length=MAX_FUNCTION_NAME_LENGTH)
    parameters: dict[str, Any] = Field(default_factory=dict)


class V2ExecuteRequest(BaseExecutionRequest):
    """Ausführungsanfrage für API v2."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"function": "add_numbers", "parameters": {"a": 1, "b": 2}},
                {"function": "echo", "parameters": {"text": "Hello"}},
            ]
        }
    )


class ExecutionError(BaseVersionedModel):
    """Fehlerinformationen für Ausführungen."""

    code: str = Field(..., min_length=MIN_NAME_LENGTH)
    message: str = Field(..., min_length=MIN_NAME_LENGTH)
    details: dict[str, Any] | None = Field(default=None)


class BaseExecutionResult(BaseVersionedModel):
    """Basis-Modell für Ausführungsergebnisse."""

    execution_id: str = Field(..., min_length=MIN_NAME_LENGTH)
    function: str = Field(..., min_length=MIN_NAME_LENGTH)
    status: str = Field(..., min_length=MIN_NAME_LENGTH)
    executed_at: datetime = Field(...)
    duration_ms: int = Field(..., ge=0)


class V2ExecuteResult(BaseExecutionResult):
    """Ausführungsergebnis für API v2."""

    result: Any | None = Field(default=None)
    error: ExecutionError | None = Field(default=None)


__all__ = [
    # Agent Models
    "AgentSummary",
    "BaseAgentModel",
    "BaseExecutionRequest",
    "BaseExecutionResult",
    "BaseFunctionModel",
    "BaseHealthResponse",
    # Base Classes
    "BaseVersionedModel",
    "BaseWebhookModel",
    "ExecutionError",
    # Health Models
    "HealthComponent",
    "HealthV2Response",
    # Function Models
    "ParameterSpec",
    # Execution Models
    "V2ExecuteRequest",
    "V2ExecuteResult",
    "V2FunctionSchema",
    # Webhook Models
    "WebhookAck",
]
