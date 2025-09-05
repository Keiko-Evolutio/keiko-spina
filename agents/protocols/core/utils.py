"""Protocol Utilities für MCP-Capability-Validierung und Konfiguration.

Dieses Modul stellt Utility-Funktionen für die Validierung von MCP-Capabilities,
Normalisierung von Protocol-Konfigurationen und Erstellung standardisierter
Response-Objekte bereit.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .dataclasses import MCPCapability

logger = get_logger(__name__)


def validate_mcp_capability(capability: MCPCapability) -> bool:
    """Validiert MCP-Capability gegen Spezifikation.

    Args:
        capability: MCP-Capability zur Validierung

    Returns:
        True wenn Capability valide ist

    Raises:
        ValueError: Bei ungültiger Capability
    """
    if not isinstance(capability, MCPCapability):
        logger.error(
            "Ungültiger Parameter-Typ für MCP-Capability-Validierung",
            extra={
                "expected_type": "MCPCapability",
                "actual_type": type(capability).__name__,
                "operation": "capability_validation"
            }
        )
        raise ValueError("Parameter muss MCPCapability-Instanz sein")

    # Prüfe erforderliche Felder
    required_fields = ["id", "type", "name", "description", "schema", "transport"]
    for field in required_fields:
        if not hasattr(capability, field):
            raise ValueError(
                f"Erforderliches Feld fehlt: {field}",
                {"missing_field": field, "capability_id": getattr(capability, "id", "unknown")}
            )

        value = getattr(capability, field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"Feld '{field}' darf nicht leer sein",
                {"empty_field": field, "capability_id": getattr(capability, "id", "unknown")}
            )

    # Prüfe Schema-Format
    if not isinstance(capability.schema, dict):
        raise ValueError(
            "Schema muss ein Dictionary sein",
            {
                "schema_type": type(capability.schema).__name__,
                "capability_id": capability.id,
                "expected_type": "dict"
            }
        )

    # Prüfe Endpoint bei bestimmten Transport-Typen
    if capability.transport.value in ["http_sse", "streamable_http"]:
        if not getattr(capability, "endpoint", None):
            raise ValueError(
                f"Endpoint erforderlich für Transport: {capability.transport.value}",
                {
                    "transport_type": capability.transport.value,
                    "capability_id": capability.id,
                    "missing_field": "endpoint"
                }
            )

    logger.debug(
        f"MCP-Capability erfolgreich validiert: {capability.id}",
        extra={
            "capability_id": capability.id,
            "capability_type": capability.type.value if hasattr(capability.type, "value") else str(capability.type),
            "transport_type": capability.transport.value if hasattr(capability.transport, "value") else str(capability.transport),
            "operation": "capability_validation_success"
        }
    )

    return True


def normalize_protocol_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalisiert Protocol-Konfiguration.

    Args:
        raw_config: Rohe Konfiguration

    Returns:
        Normalisierte Konfiguration

    Example:
        >>> input_config = {"timeout": "60", "retries": "3"}
        >>> result = normalize_protocol_config(input_config)
        >>> assert result["timeout"] == 60.0
        >>> assert result["retries"] == 3
    """
    if not raw_config:
        return {}

    result = {}

    for key, value in raw_config.items():
        # Normalisiere bekannte Felder
        try:
            if key == "timeout" and isinstance(value, (str, int, float)):
                result[key] = float(value)
            elif key in ["retries", "max_connections"] and isinstance(value, (str, int)):
                result[key] = int(value)
            elif key in ["enabled", "debug"] and isinstance(value, (str, bool)):
                if isinstance(value, bool):
                    result[key] = value
                else:
                    result[key] = str(value).lower() in ("true", "1", "yes", "on")
            else:
                result[key] = value
        except (ValueError, TypeError) as e:
            # Bei Konvertierungsfehlern den ursprünglichen Wert beibehalten
            logger.warning(
                f"Konvertierungsfehler für Konfigurationsfeld '{key}': {e}",
                extra={
                    "field_name": key,
                    "field_value": str(value),
                    "field_type": type(value).__name__,
                    "error_type": type(e).__name__,
                    "operation": "config_normalization"
                }
            )
            result[key] = value

    logger.debug(
        f"Protocol-Konfiguration normalisiert: {len(result)} Felder",
        extra={
            "original_fields": len(raw_config),
            "normalized_fields": len(result),
            "field_names": list(result.keys()),
            "operation": "config_normalization_success"
        }
    )

    return result


def create_error_result(error_message: str, **kwargs: Any) -> dict[str, Any]:
    """Erstellt standardisierte Error-Response.

    Args:
        error_message: Fehlermeldung
        **kwargs: Zusätzliche Error-Daten

    Returns:
        Standardisierte Error-Response

    Example:
        >>> error = create_error_result("Connection failed", code=500)
        >>> assert error["success"] == False
        >>> assert error["error"] == "Connection failed"
        >>> assert error["code"] == 500
    """
    logger.debug(
        f"Error-Response erstellt: {error_message}",
        extra={
            "error_message": error_message,
            "additional_data": list(kwargs.keys()),
            "operation": "error_response_creation"
        }
    )

    return {
        "success": False,
        "error": error_message,
        "result": None,
        **kwargs
    }


def create_success_result(result: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Erstellt standardisierte Success-Response.

    Args:
        result: Erfolgs-Daten
        **kwargs: Zusätzliche Response-Daten

    Returns:
        Standardisierte Success-Response

    Example:
        >>> success = create_success_result({"data": "test"}, execution_time=0.5)
        >>> assert success["success"] == True
        >>> assert success["result"]["data"] == "test"
        >>> assert success["execution_time"] == 0.5
    """
    logger.debug(
        "Success-Response erstellt",
        extra={
            "result_type": type(result).__name__ if result is not None else "None",
            "additional_data": list(kwargs.keys()),
            "operation": "success_response_creation"
        }
    )

    return {
        "success": True,
        "error": None,
        "result": result,
        **kwargs
    }


__all__ = [
    "create_error_result",
    "create_success_result",
    "normalize_protocol_config",
    "validate_mcp_capability",
]
