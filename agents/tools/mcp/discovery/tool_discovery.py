# backend/kei_mcp/discovery/tool_discovery.py
"""Vollständige Tool-Discovery für KEI-MCP Interface.

Implementiert automatische Tool-Registrierung, Schema-Validierung,
Parameter-Mapping und dynamische Verfügbarkeitsprüfung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function, trace_span

from ..schema_validator import ValidationResult, schema_validator

if TYPE_CHECKING:
    from ..kei_mcp_registry import MCPToolDefinition, RegisteredMCPServer

logger = get_logger(__name__)


class ToolAvailabilityStatus(str, Enum):
    """Status der Tool-Verfügbarkeit."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ToolCategory(str, Enum):
    """Kategorien für Tool-Klassifizierung."""
    DATA_PROCESSING = "data_processing"
    FILE_OPERATIONS = "file_operations"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    UTILITY = "utility"
    CUSTOM = "custom"


@dataclass
class ToolSchema:
    """Schema-Definition für Tool-Parameter."""
    type: str
    properties: dict[str, Any]
    required: list[str] = field(default_factory=list)
    additionalProperties: bool = False
    examples: list[dict[str, Any]] = field(default_factory=list)

    def validate_parameters(self, parameters: dict[str, Any]) -> ValidationResult:
        """Validiert Parameter gegen Schema."""
        return schema_validator.validate_tool_parameters(
            parameters=parameters,
            schema=self.__dict__
        )


@dataclass
class ToolMetadata:
    """Erweiterte Metadaten für Tools."""
    category: ToolCategory
    tags: set[str] = field(default_factory=set)
    version: str = "1.0.0"
    author: str | None = None
    documentation_url: str | None = None
    source_url: str | None = None
    license: str | None = None
    dependencies: list[str] = field(default_factory=list)
    rate_limit: int | None = None  # Requests per minute
    cost_per_call: float | None = None
    estimated_runtime_ms: int | None = None
    security_level: str = "medium"
    data_sensitivity: str = "public"


@dataclass
class DiscoveredTool:
    """Vollständig entdecktes Tool mit erweiterten Informationen."""
    id: str
    name: str
    description: str
    server_name: str
    schema: ToolSchema
    metadata: ToolMetadata
    availability_status: ToolAvailabilityStatus = ToolAvailabilityStatus.AVAILABLE
    last_validated: datetime | None = None
    validation_errors: list[str] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 1.0
    avg_response_time_ms: float = 0.0
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_available(self) -> bool:
        """Prüft, ob Tool verfügbar ist."""
        return self.availability_status == ToolAvailabilityStatus.AVAILABLE

    def matches_capability(self, capability: str) -> bool:
        """Prüft, ob Tool eine Capability erfüllt."""
        capability_lower = capability.lower()

        # Direkte Tag-Übereinstimmung
        if capability_lower in [tag.lower() for tag in self.metadata.tags]:
            return True

        # Kategorie-Übereinstimmung
        if capability_lower == self.metadata.category.value:
            return True

        # Name/Beschreibung-Übereinstimmung
        return bool(capability_lower in self.name.lower() or capability_lower in self.description.lower())


class ToolDiscoveryEngine:
    """Engine für vollständige Tool-Discovery."""

    def __init__(self) -> None:
        """Initialisiert Tool-Discovery-Engine."""
        self._discovered_tools: dict[str, DiscoveredTool] = {}
        self._tools_by_server: dict[str, list[str]] = {}
        self._tools_by_category: dict[ToolCategory, list[str]] = {}
        self._capability_mappings: dict[str, list[str]] = self._load_capability_mappings()
        self._last_discovery_run: datetime | None = None
        self._discovery_interval_seconds = 300  # 5 Minuten
        self._validation_cache: dict[str, tuple[ValidationResult, datetime]] = {}
        self._cache_ttl_seconds = 3600  # 1 Stunde

    @trace_function("mcp.tool_discovery.discover_all")
    async def discover_all_tools(
        self,
        servers: dict[str, RegisteredMCPServer],
        force_refresh: bool = False
    ) -> list[DiscoveredTool]:
        """Entdeckt alle Tools von registrierten MCP-Servern.

        Args:
            servers: Dictionary der registrierten Server
            force_refresh: Erzwingt Neuentdeckung

        Returns:
            Liste aller entdeckten Tools
        """
        current_time = datetime.now(UTC)

        # Prüfe, ob Discovery erforderlich ist
        if (not force_refresh and
            self._last_discovery_run and
            (current_time - self._last_discovery_run).total_seconds() < self._discovery_interval_seconds):
            return list(self._discovered_tools.values())

        logger.info(f"Starte Tool-Discovery für {len(servers)} Server")

        discovered_tools = []

        for server_name, server in servers.items():
            if not server.is_healthy:
                logger.warning(f"Server {server_name} ist nicht gesund, überspringe Tool-Discovery")
                continue

            try:
                server_tools = await self._discover_server_tools(server_name, server)
                discovered_tools.extend(server_tools)

                # Server-Mapping aktualisieren
                self._tools_by_server[server_name] = [tool.id for tool in server_tools]

            except Exception as e:
                logger.exception(f"Tool-Discovery für Server {server_name} fehlgeschlagen: {e}")

        # Tools nach Kategorie organisieren
        self._organize_tools_by_category(discovered_tools)

        # Discovery-Cache aktualisieren
        self._discovered_tools = {tool.id: tool for tool in discovered_tools}
        self._last_discovery_run = current_time

        logger.info(f"Tool-Discovery abgeschlossen: {len(discovered_tools)} Tools entdeckt")
        return discovered_tools

    async def _discover_server_tools(
        self,
        server_name: str,
        server: RegisteredMCPServer
    ) -> list[DiscoveredTool]:
        """Entdeckt Tools eines spezifischen Servers.

        Args:
            server_name: Name des Servers
            server: Server-Instanz

        Returns:
            Liste der entdeckten Tools
        """
        tools = []

        with trace_span("mcp.tool_discovery.server", {"server": server_name}):
            for mcp_tool in server.available_tools:
                try:
                    discovered_tool = await self._convert_mcp_tool_to_discovered(
                        mcp_tool, server_name
                    )

                    # Schema-Validierung
                    validation_result = await self._validate_tool_schema(discovered_tool)
                    if not validation_result.valid:
                        discovered_tool.validation_errors = validation_result.errors
                        discovered_tool.availability_status = ToolAvailabilityStatus.ERROR
                        logger.warning(f"Tool {discovered_tool.id} hat Schema-Validierungsfehler")

                    # Verfügbarkeitsprüfung
                    availability = await self._check_tool_availability(discovered_tool, server)
                    discovered_tool.availability_status = availability

                    tools.append(discovered_tool)

                except Exception as e:
                    logger.exception(f"Fehler beim Verarbeiten von Tool {mcp_tool.name}: {e}")

        return tools

    async def _convert_mcp_tool_to_discovered(
        self,
        mcp_tool: MCPToolDefinition,
        server_name: str
    ) -> DiscoveredTool:
        """Konvertiert MCP-Tool zu DiscoveredTool.

        Args:
            mcp_tool: MCP-Tool-Definition
            server_name: Name des Servers

        Returns:
            DiscoveredTool-Instanz
        """
        # Schema aus MCP-Tool extrahieren
        schema = ToolSchema(
            type="object",
            properties=mcp_tool.parameters.get("properties", {}),
            required=mcp_tool.parameters.get("required", []),
            additionalProperties=mcp_tool.parameters.get("additionalProperties", False),
            examples=mcp_tool.examples or []
        )

        # Metadaten ableiten
        metadata = self._extract_tool_metadata(mcp_tool)

        # Tool-ID generieren
        tool_id = f"{server_name}:{mcp_tool.name}"

        return DiscoveredTool(
            id=tool_id,
            name=mcp_tool.name,
            description=mcp_tool.description,
            server_name=server_name,
            schema=schema,
            metadata=metadata
        )

    def _extract_tool_metadata(self, mcp_tool: MCPToolDefinition) -> ToolMetadata:
        """Extrahiert Metadaten aus MCP-Tool-Definition.

        Args:
            mcp_tool: MCP-Tool-Definition

        Returns:
            ToolMetadata-Instanz
        """
        # Kategorie aus Name/Beschreibung ableiten
        category = self._classify_tool_category(mcp_tool.name, mcp_tool.description)

        # Tags aus Beschreibung extrahieren
        tags = self._extract_tags_from_description(mcp_tool.description)

        # Metadaten aus MCP-Tool extrahieren (MCPToolDefinition hat kein metadata Attribut)
        metadata_dict = {}

        return ToolMetadata(
            category=category,
            tags=tags,
            version=metadata_dict.get("version", "1.0.0"),
            author=metadata_dict.get("author"),
            documentation_url=metadata_dict.get("documentation_url"),
            source_url=metadata_dict.get("source_url"),
            license=metadata_dict.get("license"),
            dependencies=metadata_dict.get("dependencies", []),
            rate_limit=metadata_dict.get("rate_limit"),
            cost_per_call=metadata_dict.get("cost_per_call"),
            estimated_runtime_ms=metadata_dict.get("estimated_runtime_ms"),
            security_level=metadata_dict.get("security_level", "medium"),
            data_sensitivity=metadata_dict.get("data_sensitivity", "public")
        )

    def _classify_tool_category(self, name: str, description: str) -> ToolCategory:
        """Klassifiziert Tool in Kategorie.

        Args:
            name: Tool-Name
            description: Tool-Beschreibung

        Returns:
            ToolCategory
        """
        text = f"{name} {description}".lower()

        # Kategorie-Keywords
        category_keywords = {
            ToolCategory.DATA_PROCESSING: ["data", "process", "transform", "parse", "convert"],
            ToolCategory.FILE_OPERATIONS: ["file", "read", "write", "upload", "download"],
            ToolCategory.WEB_SCRAPING: ["scrape", "crawl", "web", "html", "extract"],
            ToolCategory.API_INTEGRATION: ["api", "rest", "http", "request", "endpoint"],
            ToolCategory.ANALYSIS: ["analyze", "analysis", "statistics", "report"],
            ToolCategory.COMMUNICATION: ["email", "message", "send", "notify", "chat"],
            ToolCategory.AUTOMATION: ["automate", "schedule", "workflow", "trigger"],
            ToolCategory.UTILITY: ["utility", "helper", "tool", "format", "validate"]
        }

        # Finde beste Kategorie-Übereinstimmung
        best_category = ToolCategory.CUSTOM
        best_score = 0

        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def _extract_tags_from_description(self, description: str) -> set[str]:
        """Extrahiert Tags aus Tool-Beschreibung.

        Args:
            description: Tool-Beschreibung

        Returns:
            Set von Tags
        """
        tags = set()

        # Einfache Keyword-Extraktion
        keywords = [
            "json", "xml", "csv", "pdf", "image", "text", "data",
            "web", "api", "database", "file", "email", "chat",
            "analysis", "report", "automation", "utility"
        ]

        description_lower = description.lower()
        for keyword in keywords:
            if keyword in description_lower:
                tags.add(keyword)

        return tags

    async def _validate_tool_schema(self, tool: DiscoveredTool) -> ValidationResult:
        """Validiert Tool-Schema.

        Args:
            tool: Zu validierendes Tool

        Returns:
            ValidationResult
        """
        # Cache-Prüfung
        cache_key = f"{tool.id}:{tool.metadata.version}"
        if cache_key in self._validation_cache:
            cached_result, cached_time = self._validation_cache[cache_key]
            if (datetime.now(UTC) - cached_time).total_seconds() < self._cache_ttl_seconds:
                return cached_result

        # Schema-Validierung durchführen
        try:
            result = schema_validator.validate_tool_schema(tool.schema.__dict__)

            # Cache aktualisieren
            self._validation_cache[cache_key] = (result, datetime.now(UTC))

            return result

        except Exception as e:
            logger.exception(f"Schema-Validierung für Tool {tool.id} fehlgeschlagen: {e}")
            return ValidationResult(
                valid=False,
                sanitized_value=None,
                errors=[f"Schema-Validierung fehlgeschlagen: {e!s}"]
            )

    async def _check_tool_availability(
        self,
        tool: DiscoveredTool,
        server: RegisteredMCPServer
    ) -> ToolAvailabilityStatus:
        """Prüft Tool-Verfügbarkeit.

        Args:
            tool: Zu prüfendes Tool
            server: Server-Instanz

        Returns:
            ToolAvailabilityStatus
        """
        try:
            # Einfacher Ping-Test (kann erweitert werden)
            if not server.is_healthy:
                return ToolAvailabilityStatus.UNAVAILABLE

            # Prüfe, ob Tool in Server-Tool-Liste vorhanden ist
            tool_names = [t.name for t in server.available_tools]
            if tool.name not in tool_names:
                return ToolAvailabilityStatus.UNAVAILABLE

            # Weitere Verfügbarkeitsprüfungen können hier hinzugefügt werden

            return ToolAvailabilityStatus.AVAILABLE

        except Exception as e:
            logger.exception(f"Verfügbarkeitsprüfung für Tool {tool.id} fehlgeschlagen: {e}")
            return ToolAvailabilityStatus.ERROR

    def _organize_tools_by_category(self, tools: list[DiscoveredTool]) -> None:
        """Organisiert Tools nach Kategorien.

        Args:
            tools: Liste der Tools
        """
        self._tools_by_category.clear()

        for tool in tools:
            category = tool.metadata.category
            if category not in self._tools_by_category:
                self._tools_by_category[category] = []
            self._tools_by_category[category].append(tool.id)

    def _load_capability_mappings(self) -> dict[str, list[str]]:
        """Lädt Capability-Mappings.

        Returns:
            Dictionary mit Capability-Mappings
        """
        return {
            "data_processing": ["data", "process", "transform", "convert"],
            "file_operations": ["file", "read", "write", "upload", "download"],
            "web_scraping": ["scrape", "crawl", "web", "extract"],
            "api_integration": ["api", "rest", "http", "request"],
            "analysis": ["analyze", "statistics", "report", "insights"],
            "communication": ["email", "message", "send", "notify"],
            "automation": ["automate", "schedule", "workflow"],
            "utility": ["utility", "helper", "format", "validate"]
        }

    # Public Interface Methods

    def get_tool_by_id(self, tool_id: str) -> DiscoveredTool | None:
        """Gibt Tool anhand ID zurück."""
        return self._discovered_tools.get(tool_id)

    def get_tools_by_server(self, server_name: str) -> list[DiscoveredTool]:
        """Gibt alle Tools eines Servers zurück."""
        tool_ids = self._tools_by_server.get(server_name, [])
        return [self._discovered_tools[tool_id] for tool_id in tool_ids
                if tool_id in self._discovered_tools]

    def get_tools_by_category(self, category: ToolCategory) -> list[DiscoveredTool]:
        """Gibt alle Tools einer Kategorie zurück."""
        tool_ids = self._tools_by_category.get(category, [])
        return [self._discovered_tools[tool_id] for tool_id in tool_ids
                if tool_id in self._discovered_tools]

    def get_available_tools(self) -> list[DiscoveredTool]:
        """Gibt alle verfügbaren Tools zurück."""
        return [tool for tool in self._discovered_tools.values() if tool.is_available()]

    def search_tools_by_capability(self, capability: str) -> list[DiscoveredTool]:
        """Sucht Tools nach Capability."""
        matching_tools = []

        for tool in self._discovered_tools.values():
            if tool.matches_capability(capability) and tool.is_available():
                matching_tools.append(tool)

        # Sortiere nach Erfolgsrate und Antwortzeit
        matching_tools.sort(key=lambda t: (t.success_rate, -t.avg_response_time_ms), reverse=True)

        return matching_tools


# Globale Tool-Discovery-Engine
tool_discovery_engine = ToolDiscoveryEngine()
