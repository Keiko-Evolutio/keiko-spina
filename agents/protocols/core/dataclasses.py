"""Protocol Dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import MCPPrimitiveType, MCPTransportType


@dataclass(slots=True, frozen=True)
class MCPCapability:
    """MCP-Capability als first-class citizen."""

    id: str
    type: MCPPrimitiveType
    name: str
    description: str
    schema: dict[str, Any]
    transport: MCPTransportType
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_request(self, request: dict[str, Any]) -> bool:
        """Validiert Request gegen JSON Schema.

        Args:
            request: Request-Daten zur Validierung

        Returns:
            True wenn Request valide ist

        Raises:
            ValueError: Bei ungültigen Request-Daten
        """
        if not isinstance(request, dict):
            raise ValueError("Request muss ein Dictionary sein")

        # Basis-Validierung: Prüfe erforderliche Felder
        required_fields = self.schema.get("required", [])
        for field_name in required_fields:
            if field_name not in request:
                raise ValueError(f"Erforderliches Feld fehlt: {field_name}")

        return True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
            "transport": self.transport.value,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AgentExecutionContext:
    """Execution-Context für Agent-Operationen."""

    session_id: str
    agent_id: str
    thread_id: str | None = None
    user_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)

    # MCP Session Management
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    active_tools: set[str] = field(default_factory=set)
    resource_cache: dict[str, Any] = field(default_factory=dict)

    # Performance Tracking
    operation_count: int = 0
    total_latency: float = 0.0
    error_count: int = 0

    # Metadata für Observability
    trace_id: str | None = None
    parent_span_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_operation_metrics(self, latency: float, success: bool = True) -> None:
        """Fügt Operation-Metriken hinzu.

        Args:
            latency: Ausführungszeit in Sekunden
            success: Ob Operation erfolgreich war
        """
        self.operation_count += 1
        self.total_latency += latency
        if not success:
            self.error_count += 1

    def get_average_latency(self) -> float:
        """Berechnet durchschnittliche Latenz.

        Returns:
            Durchschnittliche Latenz in Sekunden
        """
        if self.operation_count == 0:
            return 0.0
        return self.total_latency / self.operation_count

    def get_error_rate(self) -> float:
        """Berechnet Fehlerrate.

        Returns:
            Fehlerrate als Prozentsatz (0.0-1.0)
        """
        if self.operation_count == 0:
            return 0.0
        return self.error_count / self.operation_count


@dataclass(slots=True, frozen=True)
class AgentOperationResult:
    """Type-sichere Result-Struktur für Agent-Operationen."""

    success: bool
    result: Any | None = None
    error: str | None = None
    execution_time: float = 0.0

    # MCP-spezifische Results
    mcp_calls: list[dict[str, Any]] = field(default_factory=list)
    resources_accessed: list[str] = field(default_factory=list)

    # Observability Data
    trace_data: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "mcp_calls": self.mcp_calls,
            "resources_accessed": self.resources_accessed,
            "trace_data": self.trace_data,
            "performance_metrics": self.performance_metrics,
        }

    def to_json(self) -> str:
        """Konvertiert zu JSON-String."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


__all__ = [
    "AgentExecutionContext",
    "AgentOperationResult",
    "MCPCapability",
]
