"""MCP (Model Context Protocol) Integration Module.

Stellt die Integration mit externen MCP Servern bereit und ermöglicht
die Kommunikation über standardisierte HTTP-Endpoints.

Hauptkomponenten:
- MCPClient: HTTP-Client für externe MCP Server
- MCPRegistry: Registry für Server-Management und Health Monitoring
- CircuitBreaker: Fehlerbehandlung und automatische Wiederherstellung
- SchemaValidator: Validierung von Tool-Parametern und Responses
"""

from .kei_mcp_circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerException,
    CircuitState,
)
from .kei_mcp_client import (
    ExternalMCPConfig,
    KEIMCPClient,
    MCPToolDefinition,
    MCPToolResult,
)
from .kei_mcp_registry import (
    KEIMCPRegistry,
    MCPResourceResult,
    RegisteredMCPServer,
    kei_mcp_registry,
)

# Kompatibilitäts-Alias
mcp_registry = kei_mcp_registry
from .schema_validator import (
    SchemaValidationError,
    ValidationResult,
    schema_validator,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerException",
    "CircuitState",
    "ExternalMCPConfig",
    # Client-Klassen
    "KEIMCPClient",
    # Registry
    "KEIMCPRegistry",
    "MCPResourceResult",
    "MCPToolDefinition",
    "MCPToolResult",
    "RegisteredMCPServer",
    "SchemaValidationError",
    "ValidationResult",
    "kei_mcp_registry",
    "mcp_registry",  # Alias für Kompatibilität
    # Schema Validation
    "schema_validator",
]
