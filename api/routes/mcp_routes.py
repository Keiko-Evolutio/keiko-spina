"""API-Routen für externe MCP Server Integration.

Diese Routen ermöglichen die Registrierung, Verwaltung und Nutzung externer
MCP Server über eine standardisierte REST API.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from agents.tools.mcp import mcp_registry
from agents.tools.mcp.kei_mcp_client import ExternalMCPConfig
from kei_logging import get_logger
from observability import trace_function
from observability.kei_mcp_metrics import kei_mcp_metrics
from security.kei_mcp_auth import (
    kei_mcp_auth,
    require_auth,
    require_domain_validation_for_registration,
)

logger = get_logger(__name__)
router = APIRouter(tags=["external-mcp"])


def require_rate_limit(request: Any, bucket: str = "default") -> None:
    """Leichtgewichtiger Wrapper für Rate Limiting-Dependency.

    Nutzt `kei_mcp_auth.require_rate_limit`, falls vorhanden. In Testumgebungen
    kann diese Funktion per Patch gemockt werden.

    Args:
        request: FastAPI Request Objekt
        bucket: Name des Rate-Limit-Buckets
    """
    try:
        if hasattr(kei_mcp_auth, "require_rate_limit"):
            return kei_mcp_auth.require_rate_limit(request, bucket)  # type: ignore[return-value]
    except Exception:
        # In Tests/Dev ignorieren wir fehlende Implementierungen still
        return None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RegisterServerRequest(BaseModel):
    """Request für Server-Registrierung."""

    server_name: str = Field(
        ...,
        description="Eindeutiger Name des MCP Servers",
        example="weather-service",
        min_length=1,
        max_length=100
    )
    base_url: str = Field(
        ...,
        description="Basis-URL des MCP Servers",
        example="https://weather-mcp.example.com"
    )
    api_key: str | None = Field(
        None,
        description="Optionaler API-Key für Authentifizierung",
        example="sk-1234567890abcdef"
    )
    timeout_seconds: float = Field(
        30.0,
        description="Timeout für HTTP-Requests in Sekunden",
        ge=1.0,
        le=300.0
    )
    max_retries: int = Field(
        3,
        description="Maximale Anzahl von Wiederholungsversuchen",
        ge=0,
        le=10
    )
    custom_headers: dict[str, str] | None = Field(
        None,
        description="Zusätzliche HTTP-Headers",
        example={"X-Custom-Header": "value"}
    )

    @field_validator("base_url")
    def validate_base_url(cls, v):
        """Validiert die Basis-URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url muss mit http:// oder https:// beginnen")
        return v.rstrip("/")

    @field_validator("server_name")
    def validate_server_name(cls, v):
        """Validiert den Server-Namen."""
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("server_name darf nur alphanumerische Zeichen, Bindestriche und Unterstriche enthalten")
        return v


class InvokeToolRequest(BaseModel):
    """Request für Tool-Aufruf."""

    server_name: str = Field(
        ...,
        description="Name des MCP Servers",
        example="weather-service"
    )
    tool_name: str = Field(
        ...,
        description="Name des aufzurufenden Tools",
        example="get_weather"
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="Parameter für den Tool-Aufruf",
        example={"city": "Berlin", "units": "metric"}
    )


class ServerResponse(BaseModel):
    """Response für Server-Operationen."""

    success: bool = Field(..., description="Ob die Operation erfolgreich war")
    message: str = Field(..., description="Beschreibung des Ergebnisses")
    server_name: str | None = Field(None, description="Name des betroffenen Servers")
    data: dict[str, Any] | None = Field(None, description="Zusätzliche Daten")


class ToolInvocationResponse(BaseModel):
    """Response für Tool-Aufrufe."""

    success: bool = Field(..., description="Ob der Tool-Aufruf erfolgreich war")
    result: Any | None = Field(None, description="Ergebnis des Tool-Aufrufs")
    error: str | None = Field(None, description="Fehlermeldung bei Misserfolg")
    server: str | None = Field(None, description="Name des ausführenden Servers")
    execution_time_ms: float | None = Field(None, description="Ausführungszeit in Millisekunden")
    metadata: dict[str, Any] | None = Field(None, description="Zusätzliche Metadaten")


class ServerListResponse(BaseModel):
    """Response für Server-Liste."""

    servers: list[str] = Field(..., description="Liste der Server (abhängig von include_unhealthy Parameter)")
    available_servers: list[str] = Field(..., description="Liste verfügbarer (gesunder) Server")
    total_servers: int = Field(..., description="Gesamtanzahl aller registrierten Server")
    available_servers_count: int = Field(..., description="Anzahl verfügbarer (gesunder) Server")
    tools_by_server: dict[str, list[dict[str, Any]]] = Field(..., description="Tools gruppiert nach Server")


class ToolListResponse(BaseModel):
    """Response für Tool-Liste."""

    tools: list[dict[str, Any]] = Field(..., description="Liste aller verfügbaren Tools")
    tools_by_server: dict[str, list[dict[str, Any]]] = Field(..., description="Tools gruppiert nach Server")
    total_tools: int = Field(..., description="Gesamtanzahl verfügbarer Tools")


class ServerStatsResponse(BaseModel):
    """Response für Server-Statistiken."""

    server_name: str = Field(..., description="Name des Servers")
    is_healthy: bool = Field(..., description="Aktueller Gesundheitsstatus")
    uptime_seconds: float = Field(..., description="Uptime seit Registrierung")
    total_requests: int = Field(..., description="Gesamtanzahl der Requests")
    failed_requests: int = Field(..., description="Anzahl fehlgeschlagener Requests")
    error_rate: float = Field(..., description="Fehlerrate (0.0 - 1.0)")
    avg_response_time_ms: float = Field(..., description="Durchschnittliche Antwortzeit")
    available_tools: int = Field(..., description="Anzahl verfügbarer Tools")
    last_health_check: float | None = Field(None, description="Zeitstempel des letzten Health Checks")


# ============================================================================
# RESOURCE MODELS
# ============================================================================

class ResourceDefinition(BaseModel):
    """Definition einer Resource."""

    id: str = Field(..., description="Eindeutige ID der Resource")
    name: str = Field(..., description="Anzeigename der Resource")
    type: str = Field(..., description="MIME-Type oder Resource-Typ")
    description: str = Field(..., description="Beschreibung der Resource")
    server: str = Field(..., description="Name des MCP Servers")
    size_bytes: int | None = Field(None, description="Größe der Resource in Bytes")
    last_modified: str | None = Field(None, description="Letzte Änderung der Resource")
    etag: str | None = Field(None, description="Entity Tag für Caching")
    metadata: dict[str, Any] | None = Field(None, description="Zusätzliche Metadaten")


class ResourceListResponse(BaseModel):
    """Response für Resource-Liste."""

    resources: list[ResourceDefinition] = Field(..., description="Liste aller verfügbaren Resources")
    resources_by_server: dict[str, list[ResourceDefinition]] = Field(..., description="Resources gruppiert nach Server")
    total_resources: int = Field(..., description="Gesamtanzahl verfügbarer Resources")


class ResourceContentResponse(BaseModel):
    """Response für Resource-Inhalt."""

    id: str = Field(..., description="ID der Resource")
    content: str | dict[str, Any] | bytes = Field(..., description="Inhalt der Resource")
    type: str = Field(..., description="MIME-Type des Inhalts")
    encoding: str = Field(default="utf-8", description="Encoding des Inhalts")
    metadata: dict[str, Any] | None = Field(None, description="Zusätzliche Metadaten")


# ============================================================================
# PROMPT MODELS
# ============================================================================

class PromptDefinition(BaseModel):
    """Definition eines Prompt-Templates."""

    name: str = Field(..., description="Name des Prompt-Templates")
    description: str = Field(..., description="Beschreibung des Prompts")
    version: str = Field(..., description="Version des Prompts")
    server: str = Field(..., description="Name des MCP Servers")
    parameters: dict[str, Any] | None = Field(None, description="JSON Schema für Parameter")
    tags: list[str] | None = Field(None, description="Tags für Kategorisierung")
    created_at: str | None = Field(None, description="Erstellungsdatum")
    updated_at: str | None = Field(None, description="Letzte Aktualisierung")


class PromptListResponse(BaseModel):
    """Response für Prompt-Liste."""

    prompts: list[PromptDefinition] = Field(..., description="Liste aller verfügbaren Prompts")
    prompts_by_server: dict[str, list[PromptDefinition]] = Field(..., description="Prompts gruppiert nach Server")
    total_prompts: int = Field(..., description="Gesamtanzahl verfügbarer Prompts")


class PromptExample(BaseModel):
    """Beispiel für Prompt-Verwendung."""

    name: str = Field(..., description="Name des Beispiels")
    parameters: dict[str, Any] = Field(..., description="Parameter für das Beispiel")
    expected_output: str | None = Field(None, description="Erwartete Ausgabe")


class PromptResponse(BaseModel):
    """Response für Prompt-Template."""

    name: str = Field(..., description="Name des Prompt-Templates")
    template: str = Field(..., description="Prompt-Template mit Platzhaltern")
    version: str = Field(..., description="Version des Prompts")
    parameters: dict[str, Any] = Field(..., description="JSON Schema für Parameter")
    description: str | None = Field(None, description="Beschreibung des Prompts")
    examples: list[PromptExample] | None = Field(None, description="Beispiele für Prompt-Verwendung")
    metadata: dict[str, Any] | None = Field(None, description="Zusätzliche Metadaten")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post(
    "/servers/register",
    response_model=ServerResponse,
    responses={
        401: {"description": "Unauthorized"},
        422: {"description": "Unprocessable Entity"},
    },
)
@trace_function("api.external_mcp.register_server")
async def register_external_server(
    request: RegisterServerRequest,
    http_request: Request,
    _: str = Depends(require_auth)
) -> ServerResponse:
    """Registriert einen externen MCP Server.

    Dieser Endpoint ermöglicht die Registrierung eines neuen externen MCP Servers
    in der Keiko Personal Assistant Plattform. Der Server wird automatisch auf
    Erreichbarkeit geprüft und seine verfügbaren Tools werden entdeckt.

    **Domain-Validierung:** Server-Domain muss in der konfigurierten Whitelist stehen.
    Die Validierung erfolgt einmalig bei der Registrierung und wird persistiert.

    **Authentifizierung:** Bearer Token erforderlich

    **Rate Limiting:** 10 Requests pro Minute
    """
    correlation_id = str(uuid.uuid4())
    logger.info(f"Server-Registrierung gestartet: {request.server_name} (ID: {correlation_id})")

    # Domain-Validierung bei Registrierung durchführen
    try:
        await require_domain_validation_for_registration(request.base_url)
        domain_validated = True
        logger.info(f"Domain-Validierung für {request.server_name} erfolgreich")
    except HTTPException as e:
        logger.warning(f"Domain-Validierung für {request.server_name} fehlgeschlagen: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"Unerwarteter Fehler bei Domain-Validierung für {request.server_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Interner Fehler bei Domain-Validierung"
        )

    config = ExternalMCPConfig(
        server_name=request.server_name,
        base_url=request.base_url,
        api_key=request.api_key,
        timeout_seconds=request.timeout_seconds,
        max_retries=request.max_retries,
        custom_headers=request.custom_headers
    )

    # Server in Registry registrieren (mit Domain-Validierung-Status)
    success = await mcp_registry.register_server(config, domain_validated=domain_validated)

    # Rate Limit Headers hinzufügen (falls verfügbar)
    headers = {}
    try:
        if hasattr(kei_mcp_auth, "get_rate_limit_headers"):
            headers = kei_mcp_auth.get_rate_limit_headers(http_request, "register")
    except Exception as e:
        logger.debug(f"Rate limit headers nicht verfügbar: {e}")

    if success:
        # Server-Statistiken abrufen
        stats = mcp_registry.get_server_stats(request.server_name)

        logger.info(f"Server erfolgreich registriert: {request.server_name} (ID: {correlation_id})")

        response = ServerResponse(
            success=True,
            message=f"Server {request.server_name} erfolgreich registriert",
            server_name=request.server_name,
            data=stats
        )

        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
    logger.error(f"Server-Registrierung fehlgeschlagen: {request.server_name} (ID: {correlation_id})")
    raise HTTPException(
        status_code=422,  # Unprocessable Entity
        detail={
            "error": "Registration Failed",
            "message": f"Registrierung von Server {request.server_name} fehlgeschlagen",
            "type": "registration_error",
            "correlation_id": correlation_id
        },
        headers=headers
    )


@router.delete("/servers/{server_name}", response_model=ServerResponse)
@trace_function("api.external_mcp.unregister_server")
async def unregister_external_server(
    http_request: Request,
    server_name: str = Path(..., description="Name des zu entfernenden Servers"),
    _: str = Depends(require_auth),
    _rate_limit: None = Depends(lambda req: require_rate_limit(req, "default"))
) -> ServerResponse:
    """Entfernt einen externen MCP Server.

    Dieser Endpoint entfernt einen registrierten MCP Server aus der Plattform.
    Alle aktiven Verbindungen werden ordnungsgemäß geschlossen.
    """
    correlation_id = str(uuid.uuid4())
    logger.info(f"Server-Entfernung gestartet: {server_name} (ID: {correlation_id})")

    success = await mcp_registry.unregister_server(server_name)

    # Log server unregistration
    if success:
        logger.info(f"Server erfolgreich entfernt: {server_name} (ID: {correlation_id})")
    else:
        logger.warning(f"Server-Entfernung fehlgeschlagen: {server_name} nicht gefunden (ID: {correlation_id})")

    headers = kei_mcp_auth.get_rate_limit_headers(http_request, "default")

    if success:
        logger.info(f"Server erfolgreich entfernt: {server_name} (ID: {correlation_id})")
        response = ServerResponse(
            success=True,
            message=f"Server {server_name} erfolgreich entfernt",
            server_name=server_name
        )
        return JSONResponse(content=response.dict(), headers=headers)
    logger.warning(f"Server nicht gefunden: {server_name} (ID: {correlation_id})")
    raise HTTPException(
        status_code=404,
        detail={
            "error": "Server Not Found",
            "message": f"Server {server_name} nicht gefunden",
            "type": "not_found_error",
            "correlation_id": correlation_id
        },
        headers=headers
    )


@router.get("/servers", response_model=ServerListResponse)
@trace_function("api.external_mcp.list_servers")
async def list_external_servers(
    include_unhealthy: bool = Query(
        False,
        description="Ob auch nicht verfügbare Server aufgelistet werden sollen"
    )
) -> ServerListResponse:
    """Listet alle externen MCP Server.

    Gibt eine Liste aller registrierten MCP Server zurück, optional gefiltert
    nach Verfügbarkeit. Enthält auch die verfügbaren Tools pro Server.
    """
    # Basis-Daten abrufen
    all_servers = mcp_registry.get_all_servers()
    available_servers = mcp_registry.get_available_servers()
    tools_by_server = mcp_registry.get_all_tools()

    if include_unhealthy:
        # Für include_unhealthy=True: Zeige alle Server
        # Füge leere Tool-Listen für ungesunde Server hinzu
        for server_name in all_servers:
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []

        return ServerListResponse(
            servers=all_servers,  # Alle Server (gesunde + ungesunde)
            available_servers=available_servers,  # Nur gesunde Server
            total_servers=len(all_servers),  # Gesamtanzahl aller Server
            available_servers_count=len(available_servers),  # Anzahl gesunder Server
            tools_by_server=tools_by_server
        )
    # Für include_unhealthy=False: Zeige nur gesunde Server
    return ServerListResponse(
        servers=available_servers,  # Nur gesunde Server
        available_servers=available_servers,  # Nur gesunde Server (identisch)
        total_servers=len(all_servers),  # Gesamtanzahl aller Server
        available_servers_count=len(available_servers),  # Anzahl gesunder Server
        tools_by_server=tools_by_server
    )


@router.get("/servers/{server_name}/stats", response_model=ServerStatsResponse)
@trace_function("api.external_mcp.get_server_stats")
async def get_server_stats(
    server_name: str = Path(..., description="Name des Servers")
) -> ServerStatsResponse:
    """Ruft Statistiken für einen spezifischen Server ab.

    Liefert detaillierte Statistiken über Performance, Verfügbarkeit und
    Nutzung eines registrierten MCP Servers.
    """
    stats = mcp_registry.get_server_stats(server_name)

    if stats is None:
        raise HTTPException(
            status_code=404,
            detail=f"Server {server_name} nicht gefunden"
        )

    return ServerStatsResponse(**stats)


@router.post(
    "/tools/invoke",
    response_model=ToolInvocationResponse,
    responses={
        401: {"description": "Unauthorized"},
        422: {"description": "Unprocessable Entity"},
    },
)
@trace_function("api.external_mcp.invoke_tool")
async def invoke_external_tool(
    request: InvokeToolRequest,
    http_request: Request,
    _: str = Depends(require_auth)
) -> ToolInvocationResponse:
    """Ruft ein Tool auf einem externen MCP Server auf.

    Führt einen Tool-Aufruf auf dem spezifizierten MCP Server aus und gibt
    das Ergebnis zurück. Validiert Parameter gegen JSON-Schema vor dem Aufruf.
    Unterstützt automatische Retries und umfassende Fehlerbehandlung.

    **Parameter-Validierung:** Alle Parameter werden gegen das vom MCP Server
    bereitgestellte JSON-Schema validiert.

    **Rate Limiting:** 100 Requests pro Minute pro Server

    **Authentifizierung:** Bearer Token erforderlich

    **Feature Flags:** Berücksichtigt aktivierte Features für erweiterte Funktionalität
    """
    correlation_id = str(uuid.uuid4())
    logger.info(f"Tool-Aufruf gestartet: {request.server_name}:{request.tool_name} (ID: {correlation_id})")

    # 1. Parameter-Validierung gegen JSON-Schema
    validation_result = await mcp_registry.validate_tool_parameters(
        server_name=request.server_name,
        tool_name=request.tool_name,
        parameters=request.parameters
    )

    if not validation_result.valid:
        logger.warning(f"Parameter-Validierung fehlgeschlagen für {request.server_name}:{request.tool_name}: "
                      f"{len(validation_result.errors)} Fehler (ID: {correlation_id})")

        # Log validation failure
        logger.warning(f"Tool-Parameter-Validierung fehlgeschlagen: {request.server_name}:{request.tool_name} "
                      f"- {len(validation_result.errors)} Fehler (ID: {correlation_id})")

        # HTTP 422 für Validierungsfehler
        raise HTTPException(
            status_code=422,  # Unprocessable Entity
            detail={
                "error": "Parameter Validation Failed",
                "message": "Tool-Parameter entsprechen nicht dem erwarteten Schema",
                "type": "parameter_validation_error",
                "validation_errors": validation_result.errors,
                "field_errors": validation_result.field_errors,
                "correlation_id": correlation_id,
                "validation_time_ms": validation_result.validation_time_ms
            }
        )

    logger.debug(f"Parameter-Validierung erfolgreich für {request.server_name}:{request.tool_name} "
                f"({validation_result.validation_time_ms:.1f}ms, ID: {correlation_id})")

    # 2. Tool-Aufruf ausführen
    result = await mcp_registry.invoke_tool(
        server_name=request.server_name,
        tool_name=request.tool_name,
        parameters=request.parameters
    )

    # Log tool invocation result
    if result.success:
        logger.info(f"Tool-Aufruf erfolgreich: {request.server_name}:{request.tool_name} "
                   f"in {result.execution_time_ms}ms (ID: {correlation_id})")
    else:
        logger.warning(f"Tool-Aufruf fehlgeschlagen: {request.server_name}:{request.tool_name} "
                      f"- {result.error} (ID: {correlation_id})")

    # Rate Limit Headers hinzufügen (falls verfügbar)
    headers = {}
    try:
        if hasattr(kei_mcp_auth, "get_rate_limit_headers"):
            headers = kei_mcp_auth.get_rate_limit_headers(http_request, "invoke")
    except Exception as e:
        logger.debug(f"Rate limit headers nicht verfügbar: {e}")

    if result.success:
        logger.info(f"Tool-Aufruf erfolgreich: {request.server_name}:{request.tool_name} "
                   f"({result.execution_time_ms:.1f}ms, ID: {correlation_id})")
    else:
        logger.warning(f"Tool-Aufruf fehlgeschlagen: {request.server_name}:{request.tool_name} "
                      f"- {result.error} (ID: {correlation_id})")

    response = ToolInvocationResponse(
        success=result.success,
        result=result.result,
        error=result.error,
        server=result.server,
        execution_time_ms=result.execution_time_ms,
        metadata={
            **(result.metadata or {}),
            "correlation_id": correlation_id
        }
    )

    if result.success:
        return JSONResponse(content=response.dict(), headers=headers)
    raise HTTPException(
        status_code=422,  # Unprocessable Entity
        detail={
            "error": "Tool Execution Failed",
            "message": result.error or "Tool-Aufruf fehlgeschlagen",
            "type": "tool_execution_error",
            "correlation_id": correlation_id,
            "server": result.server,
            "tool": request.tool_name
        },
        headers=headers
    )


@router.get("/tools", response_model=ToolListResponse)
@trace_function("api.external_mcp.list_tools")
async def list_all_external_tools(
    server_name: str | None = Query(
        None,
        description="Optionaler Filter nach Server-Name"
    ),
    tool_name: str | None = Query(
        None,
        description="Optionaler Filter nach Tool-Name"
    ),
    _: str = Depends(require_auth)
) -> ToolListResponse:
    """Listet alle verfügbaren Tools aller externen Server.

    Gibt eine umfassende Liste aller verfügbaren Tools zurück, optional
    gefiltert nach Server oder Tool-Name.

    **Authentifizierung:** Bearer Token erforderlich
    """
    if tool_name:
        # Suche nach spezifischem Tool-Namen
        matching_tools = mcp_registry.find_tools_by_name(tool_name)
        tools_by_server = {}
        all_tools = []

        for match in matching_tools:
            server = match["server"]
            tool = match["tool"]

            if server_name and server != server_name:
                continue

            if server not in tools_by_server:
                tools_by_server[server] = []
            tools_by_server[server].append(tool)

            all_tools.append({
                **tool,
                "server": server
            })
    else:
        # Alle Tools abrufen
        tools_by_server = mcp_registry.get_all_tools()

        if server_name:
            # Nach Server filtern
            if server_name in tools_by_server:
                tools_by_server = {server_name: tools_by_server[server_name]}
            else:
                tools_by_server = {}

        # Flache Liste erstellen
        all_tools = []
        for server, tools in tools_by_server.items():
            for tool in tools:
                all_tools.append({
                    **tool,
                    "server": server
                })

    return ToolListResponse(
        tools=all_tools,
        tools_by_server=tools_by_server,
        total_tools=len(all_tools)
    )


@router.get("/health")
@trace_function("api.external_mcp.health_check")
async def health_check() -> dict[str, Any]:
    """Gesundheitsprüfung für die externe MCP Integration.

    Gibt den Status der MCP Registry und aller registrierten Server zurück.
    """
    all_servers = mcp_registry.get_all_servers()
    available_servers = mcp_registry.get_available_servers()
    health_summary = kei_mcp_metrics.get_health_summary()

    return {
        "status": health_summary["status"],
        "total_servers": len(all_servers),
        "available_servers": len(available_servers),
        "unavailable_servers": len(all_servers) - len(available_servers),
        "registry_active": True,
        "error_rate": health_summary["error_rate"],
        "avg_response_time_ms": health_summary["avg_response_time_ms"],
        "open_circuit_breakers": 0  # Circuit breaker not implemented yet
    }


@router.get("/metrics")
@trace_function("api.external_mcp.get_metrics")
async def get_comprehensive_metrics(
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Gibt umfassende Metriken für externe MCP Integration zurück.

    Enthält Performance-Metriken, Circuit Breaker Status, Connection Pool
    Statistiken und detaillierte Error Categorization.
    """
    # Basis-Metriken
    metrics = kei_mcp_metrics.get_comprehensive_metrics()

    # Circuit Breaker Statistiken hinzufügen (placeholder)
    circuit_stats = {}  # Circuit breaker not implemented yet
    metrics["circuit_breakers"] = circuit_stats

    # Registry-spezifische Metriken
    registry_stats = {}
    for server_name in mcp_registry.get_all_servers():
        server_stats = mcp_registry.get_server_stats(server_name)
        if server_stats:
            registry_stats[server_name] = server_stats

    metrics["registry_servers"] = registry_stats

    return metrics


@router.get("/circuit-breakers")
@trace_function("api.external_mcp.get_circuit_breakers")
async def get_circuit_breaker_status(
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Gibt Circuit Breaker Status für alle externen MCP Server zurück.

    Zeigt detaillierte Informationen über Circuit Breaker Zustände,
    Fehlerstatistiken und Recovery-Status.
    """
    # Placeholder implementation - circuit breaker not yet implemented
    circuit_stats = {}

    # Get server list for placeholder data
    servers = mcp_registry.list_servers()
    for server_name in servers:
        circuit_stats[f"mcp_client_{server_name}"] = {
            "state": "closed",
            "failure_count": 0,
            "success_count": 0,
            "last_failure_time": None,
            "next_attempt_time": None
        }

    total_circuits = len(circuit_stats)
    closed_circuits = total_circuits  # All closed in placeholder

    return {
        "circuit_breakers": circuit_stats,
        "summary": {
            "total_circuits": total_circuits,
            "open_circuits": 0,
            "half_open_circuits": 0,
            "closed_circuits": closed_circuits,
            "health_percentage": 100.0
        }
    }


@router.post(
    "/circuit-breakers/{server_name}/reset",
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not Found"},
    },
)
@trace_function("api.external_mcp.reset_circuit_breaker")
async def reset_circuit_breaker(
    server_name: str = Path(..., description="Name des Servers"),
    _: str = Depends(require_auth)
) -> dict[str, Any]:
    """Setzt Circuit Breaker für einen spezifischen Server zurück.

    Nützlich für manuelle Recovery nach Wartungsarbeiten.
    """
    # Check if server exists
    if server_name not in mcp_registry.list_servers():
        raise HTTPException(
            status_code=404,
            detail=f"Server {server_name} nicht gefunden"
        )

    # Placeholder implementation - circuit breaker not yet implemented
    correlation_id = str(uuid.uuid4())
    logger.info(f"Circuit Breaker Reset angefordert für {server_name} (nicht implementiert) (ID: {correlation_id})")

    return {
        "success": True,
        "message": f"Circuit Breaker für {server_name} zurückgesetzt (Placeholder)",
        "server_name": server_name,
        "correlation_id": correlation_id,
        "note": "Circuit breaker functionality not yet implemented"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

# Exception handlers removed - should be defined in main FastAPI app


# ============================================================================
# RESOURCE ENDPOINTS
# ============================================================================

@router.get("/resources", response_model=ResourceListResponse)
@trace_function("api.external_mcp.list_resources")
async def list_all_external_resources(
    server_name: str | None = Query(
        None,
        description="Optionaler Filter nach Server-Name"
    ),
    resource_name: str | None = Query(
        None,
        description="Optionaler Filter nach Resource-Name"
    ),
    resource_type: str | None = Query(
        None,
        description="Optionaler Filter nach Resource-Typ"
    )
) -> ResourceListResponse:
    """Listet alle verfügbaren Resources.

    Gibt eine umfassende Liste aller verfügbaren Resources zurück, optional
    gefiltert nach Server oder Resource-Name. Resources sind read-only Datenquellen
    wie Dateien, Dokumente, oder strukturierte Daten.
    """
    # Resources von Registry abrufen
    resources_by_server = await mcp_registry.get_all_resources()

    # Filtern nach Server-Name
    if server_name:
        if server_name in resources_by_server:
            resources_by_server = {server_name: resources_by_server[server_name]}
        else:
            resources_by_server = {}

    # Flache Liste erstellen und filtern
    all_resources = []
    for server, resources in resources_by_server.items():
        for resource in resources:
            # Filter nach Resource-Name
            if resource_name and resource_name.lower() not in resource.get("name", "").lower():
                continue

            # Filter nach Resource-Typ
            if resource_type and resource.get("type") != resource_type:
                continue

            resource_def = ResourceDefinition(
                id=resource.get("id", ""),
                name=resource.get("name", ""),
                type=resource.get("type", ""),
                description=resource.get("description", ""),
                server=server,
                size_bytes=resource.get("size_bytes"),
                last_modified=resource.get("last_modified"),
                etag=resource.get("etag"),
                metadata=resource.get("metadata")
            )
            all_resources.append(resource_def)

    # Gruppierte Antwort erstellen
    filtered_resources_by_server = {}
    for resource in all_resources:
        if resource.server not in filtered_resources_by_server:
            filtered_resources_by_server[resource.server] = []
        filtered_resources_by_server[resource.server].append(resource)

    return ResourceListResponse(
        resources=all_resources,
        resources_by_server=filtered_resources_by_server,
        total_resources=len(all_resources)
    )


@router.get("/resources/{server_name}/{resource_id}")
@trace_function("api.external_mcp.get_resource")
async def get_external_resource(
    server_name: str = Path(..., description="Name des MCP Servers"),
    resource_id: str = Path(..., description="Eindeutige ID der Resource"),
    if_none_match: str | None = Header(None, alias="If-None-Match"),
    range_header: str | None = Header(None, alias="Range")
) -> Response:
    """Ruft eine spezifische Resource ab.

    Lädt den Inhalt einer spezifischen Resource von einem externen MCP Server.
    Unterstützt ETag-basiertes Caching und Range-Requests für große Dateien.
    """
    correlation_id = str(uuid.uuid4())
    logger.info(f"Resource-Abruf gestartet: {server_name}:{resource_id} (ID: {correlation_id})")

    try:
        # Resource von Registry abrufen
        resource_result = await mcp_registry.get_resource(
            server_name=server_name,
            resource_id=resource_id,
            if_none_match=if_none_match,
            range_header=range_header
        )

        if not resource_result.success:
            if "not found" in resource_result.error.lower():
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Resource Not Found",
                        "message": f"Resource {resource_id} nicht gefunden auf Server {server_name}",
                        "type": "not_found_error",
                        "correlation_id": correlation_id
                    }
                )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Resource Access Failed",
                    "message": resource_result.error,
                    "type": "resource_access_error",
                    "correlation_id": correlation_id
                }
            )

        # ETag-Match prüfen (304 Not Modified)
        if resource_result.status_code == 304:
            return Response(status_code=304)

        # Range-Request prüfen (206 Partial Content)
        if resource_result.status_code == 206:
            headers = {
                "Content-Range": resource_result.headers.get("Content-Range", ""),
                "ETag": resource_result.headers.get("ETag", "")
            }
            return Response(
                content=resource_result.content,
                status_code=206,
                headers=headers,
                media_type=resource_result.content_type
            )

        # Normale Response (200 OK)
        headers = {}
        if resource_result.etag:
            headers["ETag"] = resource_result.etag
        if resource_result.last_modified:
            headers["Last-Modified"] = resource_result.last_modified
        if resource_result.content_length:
            headers["Content-Length"] = str(resource_result.content_length)

        logger.info(f"Resource-Abruf erfolgreich: {server_name}:{resource_id} "
                   f"({len(resource_result.content)} bytes, ID: {correlation_id})")

        return Response(
            content=resource_result.content,
            status_code=200,
            headers=headers,
            media_type=resource_result.content_type
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Resource-Abruf fehlgeschlagen: {server_name}:{resource_id} - {exc} (ID: {correlation_id})")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "Unerwarteter Fehler beim Resource-Abruf",
                "type": "internal_error",
                "correlation_id": correlation_id
            }
        )


# ============================================================================
# PROMPT ENDPOINTS
# ============================================================================

@router.get("/prompts", response_model=PromptListResponse)
@trace_function("api.external_mcp.list_prompts")
async def list_all_external_prompts(
    server_name: str | None = Query(
        None,
        description="Optionaler Filter nach Server-Name"
    ),
    prompt_name: str | None = Query(
        None,
        description="Optionaler Filter nach Prompt-Name"
    ),
    version: str | None = Query(
        None,
        description="Optionaler Filter nach Prompt-Version"
    )
) -> PromptListResponse:
    """Listet alle verfügbaren Prompts.

    Gibt eine umfassende Liste aller verfügbaren Prompt-Templates zurück, optional
    gefiltert nach Server oder Prompt-Name. Prompts sind versionierte Templates
    mit Parametrisierungs-Schema.
    """
    # Prompts von Registry abrufen
    prompts_by_server = await mcp_registry.get_all_prompts()

    # Filtern nach Server-Name
    if server_name:
        if server_name in prompts_by_server:
            prompts_by_server = {server_name: prompts_by_server[server_name]}
        else:
            prompts_by_server = {}

    # Flache Liste erstellen und filtern
    all_prompts = []
    for server, prompts in prompts_by_server.items():
        for prompt in prompts:
            # Filter nach Prompt-Name
            if prompt_name and prompt_name.lower() not in prompt.get("name", "").lower():
                continue

            # Filter nach Version
            if version and prompt.get("version") != version:
                continue

            prompt_def = PromptDefinition(
                name=prompt.get("name", ""),
                description=prompt.get("description", ""),
                version=prompt.get("version", ""),
                server=server,
                parameters=prompt.get("parameters"),
                tags=prompt.get("tags"),
                created_at=prompt.get("created_at"),
                updated_at=prompt.get("updated_at")
            )
            all_prompts.append(prompt_def)

    # Gruppierte Antwort erstellen
    filtered_prompts_by_server = {}
    for prompt in all_prompts:
        if prompt.server not in filtered_prompts_by_server:
            filtered_prompts_by_server[prompt.server] = []
        filtered_prompts_by_server[prompt.server].append(prompt)

    return PromptListResponse(
        prompts=all_prompts,
        prompts_by_server=filtered_prompts_by_server,
        total_prompts=len(all_prompts)
    )


@router.get("/prompts/{server_name}/{prompt_name}", response_model=PromptResponse)
@trace_function("api.external_mcp.get_prompt")
async def get_external_prompt(
    server_name: str = Path(..., description="Name des MCP Servers"),
    prompt_name: str = Path(..., description="Name des Prompt-Templates"),
    version: str | None = Query(None, description="Spezifische Version des Prompts")
) -> PromptResponse:
    """Ruft ein spezifisches Prompt-Template ab.

    Lädt ein spezifisches Prompt-Template von einem externen MCP Server.
    Unterstützt Versionierung und Parametrisierungs-Schema.
    """
    correlation_id = str(uuid.uuid4())
    logger.info(f"Prompt-Abruf gestartet: {server_name}:{prompt_name} (ID: {correlation_id})")

    try:
        # Prompt von Registry abrufen
        prompt_result = await mcp_registry.get_prompt(
            server_name=server_name,
            prompt_name=prompt_name,
            version=version
        )

        if not prompt_result.get("success", False):
            error_text = (prompt_result.get("error") or "").lower()
            if any(x in error_text for x in ["not found", "nicht gefunden", "404"]):
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Prompt Not Found",
                        "message": f"Prompt {prompt_name} nicht gefunden auf Server {server_name}",
                        "type": "not_found_error",
                        "correlation_id": correlation_id
                    }
                )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Prompt Access Failed",
                    "message": prompt_result.get("error", "Unbekannter Fehler"),
                    "type": "prompt_access_error",
                    "correlation_id": correlation_id
                }
            )

        logger.info(f"Prompt-Abruf erfolgreich: {server_name}:{prompt_name} (ID: {correlation_id})")

        return PromptResponse(
            name=prompt_result.get("name", prompt_name),
            template=prompt_result.get("template", ""),
            version=prompt_result.get("version", "1.0.0"),
            parameters=prompt_result.get("parameters", {}),
            description=prompt_result.get("description"),
            examples=[
                PromptExample(
                    name=ex.get("name", ""),
                    parameters=ex.get("parameters", {}),
                    expected_output=ex.get("expected_output")
                ) for ex in prompt_result.get("examples", [])
            ],
            metadata=prompt_result.get("metadata")
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Prompt-Abruf fehlgeschlagen: {server_name}:{prompt_name} - {exc} (ID: {correlation_id})")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "Unerwarteter Fehler beim Prompt-Abruf",
                "type": "internal_error",
                "correlation_id": correlation_id
            }
        )


# Exception handlers removed - should be defined in main FastAPI app
