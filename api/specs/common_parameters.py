# backend/api/specs/common_parameters.py
"""Gemeinsame OpenAPI-Parameter-Definitionen.

Zentralisiert wiederverwendbare Parameter-Definitionen für Header,
Query-Parameter und Path-Parameter zur Elimination von Code-Duplikaten.
"""

from __future__ import annotations

from typing import Any

from .constants import (
    DEFAULT_PRIORITY,
    DEFAULT_TIMEOUT_SECONDS,
    EXAMPLE_TRACEPARENT,
    HEADER_IDEMPOTENCY_KEY,
    HEADER_PRIORITY,
    HEADER_TIMEOUT,
    HEADER_TRACEPARENT,
    HEADER_TRACESTATE,
    IDEMPOTENCY_KEY_MAX_LENGTH,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    PARAM_REF_IDEMPOTENCY_KEY,
    PARAM_REF_PRIORITY,
    PARAM_REF_TIMEOUT,
    PARAM_REF_TRACEPARENT,
    PARAM_REF_TRACESTATE,
    PRIORITY_LEVELS,
)


def create_header_parameter(
    name: str,
    description: str,
    required: bool = False,
    schema_type: str = "string",
    **schema_kwargs: Any
) -> dict[str, Any]:
    """Erstellt einen standardisierten Header-Parameter.

    Args:
        name: Header-Name
        description: Parameter-Beschreibung
        required: Ob Parameter erforderlich ist
        schema_type: Schema-Typ (default: "string")
        **schema_kwargs: Zusätzliche Schema-Eigenschaften

    Returns:
        OpenAPI-Parameter-Definition
    """
    return {
        "name": name,
        "in": "header",
        "description": description,
        "required": required,
        "schema": {"type": schema_type, **schema_kwargs}
    }



def get_traceparent_parameter() -> dict[str, Any]:
    """Gibt W3C Trace Context Traceparent Header-Parameter zurück.

    Returns:
        OpenAPI-Parameter-Definition für Traceparent
    """
    param = create_header_parameter(
        name=HEADER_TRACEPARENT,
        description="W3C Trace Context Traceparent für Request-Tracing",
        required=False
    )
    param["example"] = EXAMPLE_TRACEPARENT
    return param


def get_tracestate_parameter() -> dict[str, Any]:
    """Gibt W3C Trace Context Tracestate Header-Parameter zurück.

    Returns:
        OpenAPI-Parameter-Definition für Tracestate
    """
    return create_header_parameter(
        name=HEADER_TRACESTATE,
        description="W3C Trace Context Tracestate für erweiterte Trace-Informationen",
        required=False
    )


def get_idempotency_key_parameter() -> dict[str, Any]:
    """Gibt Idempotency-Key Header-Parameter zurück.

    Returns:
        OpenAPI-Parameter-Definition für Idempotency-Key
    """
    return create_header_parameter(
        name=HEADER_IDEMPOTENCY_KEY,
        description="Idempotenz-Schlüssel für wiederholbare Operationen",
        required=False,
        maxLength=IDEMPOTENCY_KEY_MAX_LENGTH
    )


def get_priority_parameter() -> dict[str, Any]:
    """Gibt Priority Header-Parameter zurück.

    Returns:
        OpenAPI-Parameter-Definition für Priority
    """
    return create_header_parameter(
        name=HEADER_PRIORITY,
        description="Operation-Priorität für Request-Scheduling",
        required=False,
        enum=list(PRIORITY_LEVELS),
        default=DEFAULT_PRIORITY
    )


def get_timeout_parameter() -> dict[str, Any]:
    """Gibt Timeout Header-Parameter zurück.

    Returns:
        OpenAPI-Parameter-Definition für Timeout
    """
    return create_header_parameter(
        name=HEADER_TIMEOUT,
        description="Operation-Timeout in Sekunden",
        required=False,
        schema_type="integer",
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=MAX_TIMEOUT_SECONDS,
        default=DEFAULT_TIMEOUT_SECONDS
    )


def get_trace_headers() -> list[dict[str, Any]]:
    """Gibt alle Trace-Context Header-Parameter zurück.

    Returns:
        Liste von Trace-Context Parameter-Definitionen
    """
    return [
        get_traceparent_parameter(),
        get_tracestate_parameter()
    ]


def get_operation_headers() -> list[dict[str, Any]]:
    """Gibt alle Operation-spezifischen Header-Parameter zurück.

    Returns:
        Liste von Operation-Header Parameter-Definitionen
    """
    return [
        get_idempotency_key_parameter(),
        get_priority_parameter(),
        get_timeout_parameter()
    ]


def get_common_headers() -> list[dict[str, Any]]:
    """Gibt alle gemeinsamen Header-Parameter zurück.

    Returns:
        Liste aller gemeinsamen Parameter-Definitionen
    """
    return get_trace_headers() + get_operation_headers()


def get_common_parameter_references() -> list[dict[str, str]]:
    """Gibt Parameter-Referenzen für Wiederverwendung zurück.

    Returns:
        Liste von Parameter-Referenzen
    """
    return [
        {"$ref": PARAM_REF_TRACEPARENT},
        {"$ref": PARAM_REF_TRACESTATE},
        {"$ref": PARAM_REF_IDEMPOTENCY_KEY},
        {"$ref": PARAM_REF_PRIORITY},
        {"$ref": PARAM_REF_TIMEOUT}
    ]


def get_parameter_components() -> dict[str, dict[str, Any]]:
    """Gibt alle Parameter-Komponenten für OpenAPI-Spezifikation zurück.

    Returns:
        Dictionary mit Parameter-Komponenten-Definitionen
    """
    return {
        "TraceparentHeader": get_traceparent_parameter(),
        "TracestateHeader": get_tracestate_parameter(),
        "IdempotencyKeyHeader": get_idempotency_key_parameter(),
        "PriorityHeader": get_priority_parameter(),
        "TimeoutHeader": get_timeout_parameter()
    }


def create_endpoint_parameters(
    include_trace: bool = True,
    include_idempotency: bool = True,
    include_priority: bool = True,
    include_timeout: bool = True,
    additional_parameters: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Erstellt Parameter-Liste für Endpunkt-Definition.

    Args:
        include_trace: Ob Trace-Context Parameter eingeschlossen werden sollen
        include_idempotency: Ob Idempotency-Key Parameter eingeschlossen werden soll
        include_priority: Ob Priority Parameter eingeschlossen werden soll
        include_timeout: Ob Timeout Parameter eingeschlossen werden soll
        additional_parameters: Zusätzliche Parameter-Definitionen

    Returns:
        Liste von Parameter-Definitionen für Endpunkt
    """
    parameters = []

    if include_trace:
        parameters.extend(get_trace_headers())

    if include_idempotency:
        parameters.append(get_idempotency_key_parameter())

    if include_priority:
        parameters.append(get_priority_parameter())

    if include_timeout:
        parameters.append(get_timeout_parameter())

    if additional_parameters:
        parameters.extend(additional_parameters)

    return parameters


def create_endpoint_parameter_references(
    include_trace: bool = True,
    include_idempotency: bool = True,
    include_priority: bool = True,
    include_timeout: bool = True,
    additional_refs: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Erstellt Parameter-Referenz-Liste für Endpunkt-Definition.

    Args:
        include_trace: Ob Trace-Context Parameter-Referenzen eingeschlossen werden sollen
        include_idempotency: Ob Idempotency-Key Parameter-Referenz eingeschlossen werden soll
        include_priority: Ob Priority Parameter-Referenz eingeschlossen werden soll
        include_timeout: Ob Timeout Parameter-Referenz eingeschlossen werden soll
        additional_refs: Zusätzliche Parameter-Referenzen

    Returns:
        Liste von Parameter-Referenzen für Endpunkt
    """
    references = []

    if include_trace:
        references.extend([
            {"$ref": PARAM_REF_TRACEPARENT},
            {"$ref": PARAM_REF_TRACESTATE}
        ])

    if include_idempotency:
        references.append({"$ref": PARAM_REF_IDEMPOTENCY_KEY})

    if include_priority:
        references.append({"$ref": PARAM_REF_PRIORITY})

    if include_timeout:
        references.append({"$ref": PARAM_REF_TIMEOUT})

    if additional_refs:
        references.extend(additional_refs)

    return references
