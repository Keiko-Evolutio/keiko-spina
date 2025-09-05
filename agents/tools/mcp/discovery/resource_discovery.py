# backend/kei_mcp/discovery/resource_discovery.py
"""Resource-Discovery für KEI-MCP Interface.

Implementiert Resource-Discovery, Caching, Synchronisation,
Access-Control und Metadaten-Management für MCP-Server-Ressourcen.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function, trace_span

if TYPE_CHECKING:
    import asyncio

    from ..kei_mcp_registry import MCPResourceDefinition, RegisteredMCPServer

logger = get_logger(__name__)


class ResourceType(str, Enum):
    """Typen von MCP-Ressourcen."""
    FILE = "file"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    STREAM = "stream"
    CONFIGURATION = "configuration"
    TEMPLATE = "template"
    SCHEMA = "schema"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class ResourceAccessLevel(str, Enum):
    """Zugriffslevel für Ressourcen."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    RESTRICTED = "restricted"
    PRIVATE = "private"


class ResourceStatus(str, Enum):
    """Status einer Ressource."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    CACHED = "cached"
    STALE = "stale"
    ERROR = "error"
    SYNCING = "syncing"


@dataclass
class ResourceMetadata:
    """Erweiterte Metadaten für Ressourcen."""
    resource_type: ResourceType
    access_level: ResourceAccessLevel = ResourceAccessLevel.PUBLIC
    content_type: str | None = None
    encoding: str | None = None
    language: str | None = None
    tags: set[str] = field(default_factory=set)
    version: str = "1.0.0"
    checksum: str | None = None
    dependencies: list[str] = field(default_factory=list)
    expiry_time: datetime | None = None
    cache_ttl_seconds: int = 3600
    compression: str | None = None
    encryption: str | None = None

    def is_expired(self) -> bool:
        """Prüft, ob Ressource abgelaufen ist."""
        if not self.expiry_time:
            return False
        return datetime.now(UTC) > self.expiry_time


@dataclass
class CachedResource:
    """Gecachte Ressource mit Metadaten."""
    id: str
    content: bytes
    metadata: ResourceMetadata
    cached_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    last_accessed: datetime | None = None
    etag: str | None = None

    def is_stale(self) -> bool:
        """Prüft, ob Cache-Eintrag veraltet ist."""
        if self.metadata.is_expired():
            return True

        cache_age = datetime.now(UTC) - self.cached_at
        return cache_age.total_seconds() > self.metadata.cache_ttl_seconds

    def mark_accessed(self) -> None:
        """Markiert Ressource als zugegriffen."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)


@dataclass
class DiscoveredResource:
    """Entdeckte Ressource mit vollständigen Informationen."""
    id: str
    name: str
    description: str
    server_name: str
    uri: str
    metadata: ResourceMetadata
    status: ResourceStatus = ResourceStatus.AVAILABLE
    size_bytes: int | None = None
    last_modified: datetime | None = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    access_permissions: set[str] = field(default_factory=set)
    usage_count: int = 0
    error_count: int = 0

    def is_accessible(self, user_permissions: set[str]) -> bool:
        """Prüft, ob Ressource für Benutzer zugänglich ist."""
        if self.metadata.access_level == ResourceAccessLevel.PUBLIC:
            return True

        if not self.access_permissions:
            return self.metadata.access_level == ResourceAccessLevel.PUBLIC

        return bool(self.access_permissions & user_permissions)

    def matches_criteria(self, criteria: dict[str, Any]) -> bool:
        """Prüft, ob Ressource Suchkriterien erfüllt."""
        # Typ-Filter
        if "type" in criteria and self.metadata.resource_type != criteria["type"]:
            return False

        # Tag-Filter
        if "tags" in criteria:
            required_tags = set(criteria["tags"])
            if not required_tags.issubset(self.metadata.tags):
                return False

        # Content-Type-Filter
        if "content_type" in criteria and self.metadata.content_type != criteria["content_type"]:
            return False

        # Größen-Filter
        return not ("max_size" in criteria and self.size_bytes and self.size_bytes > criteria["max_size"])


class ResourceAccessController:
    """Controller für Ressourcen-Zugriffskontrolle."""

    def __init__(self) -> None:
        """Initialisiert Access Controller."""
        self._access_policies: dict[str, dict[str, Any]] = {}
        self._user_permissions: dict[str, set[str]] = {}
        self._resource_permissions: dict[str, set[str]] = {}

    def set_user_permissions(self, user_id: str, permissions: set[str]) -> None:
        """Setzt Benutzer-Berechtigungen."""
        self._user_permissions[user_id] = permissions

    def set_resource_permissions(self, resource_id: str, permissions: set[str]) -> None:
        """Setzt Ressourcen-Berechtigungen."""
        self._resource_permissions[resource_id] = permissions

    def check_access(self, user_id: str, resource: DiscoveredResource) -> bool:
        """Prüft Zugriffsberechtigung."""
        user_perms = self._user_permissions.get(user_id, set())
        resource_perms = self._resource_permissions.get(resource.id, resource.access_permissions)

        return resource.is_accessible(user_perms) and (
            not resource_perms or bool(user_perms & resource_perms)
        )


class ResourceCache:
    """Cache für MCP-Ressourcen."""

    def __init__(self, max_size_mb: int = 100) -> None:
        """Initialisiert Resource Cache.

        Args:
            max_size_mb: Maximale Cache-Größe in MB
        """
        self._cache: dict[str, CachedResource] = {}
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0

    def get(self, resource_id: str) -> CachedResource | None:
        """Gibt gecachte Ressource zurück."""
        cached = self._cache.get(resource_id)
        if cached and not cached.is_stale():
            cached.mark_accessed()
            return cached
        if cached and cached.is_stale():
            self.remove(resource_id)
        return None

    def put(self, resource_id: str, content: bytes, metadata: ResourceMetadata) -> bool:
        """Fügt Ressource zum Cache hinzu."""
        content_size = len(content)

        # Prüfe Cache-Größe
        if content_size > self._max_size_bytes:
            logger.warning(f"Ressource {resource_id} zu groß für Cache ({content_size} bytes)")
            return False

        # Mache Platz im Cache
        self._make_space(content_size)

        # Erstelle Cache-Eintrag
        cached_resource = CachedResource(
            id=resource_id,
            content=content,
            metadata=metadata,
            etag=hashlib.md5(content).hexdigest()
        )

        self._cache[resource_id] = cached_resource
        self._current_size_bytes += content_size

        logger.debug(f"Ressource {resource_id} zu Cache hinzugefügt ({content_size} bytes)")
        return True

    def remove(self, resource_id: str) -> bool:
        """Entfernt Ressource aus Cache."""
        if resource_id in self._cache:
            cached = self._cache[resource_id]
            self._current_size_bytes -= len(cached.content)
            del self._cache[resource_id]
            return True
        return False

    def _make_space(self, required_bytes: int) -> None:
        """Macht Platz im Cache für neue Ressource."""
        while (self._current_size_bytes + required_bytes > self._max_size_bytes and
               self._cache):
            # Entferne am wenigsten genutzte Ressource
            lru_resource_id = min(
                self._cache.keys(),
                key=lambda rid: (
                    self._cache[rid].last_accessed or self._cache[rid].cached_at,
                    self._cache[rid].access_count
                )
            )
            self.remove(lru_resource_id)

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        return {
            "total_resources": len(self._cache),
            "total_size_bytes": self._current_size_bytes,
            "max_size_bytes": self._max_size_bytes,
            "utilization": self._current_size_bytes / self._max_size_bytes,
            "stale_resources": sum(1 for r in self._cache.values() if r.is_stale())
        }


class ResourceDiscoveryEngine:
    """Engine für Resource-Discovery und -Management."""

    def __init__(self, cache_size_mb: int = 100) -> None:
        """Initialisiert Resource-Discovery-Engine."""
        self._discovered_resources: dict[str, DiscoveredResource] = {}
        self._resources_by_server: dict[str, list[str]] = {}
        self._resources_by_type: dict[ResourceType, list[str]] = {}
        self._cache = ResourceCache(cache_size_mb)
        self._access_controller = ResourceAccessController()
        self._last_discovery_run: datetime | None = None
        self._discovery_interval_seconds = 600  # 10 Minuten
        self._sync_tasks: dict[str, asyncio.Task] = {}

    @trace_function("mcp.resource_discovery.discover_all")
    async def discover_all_resources(
        self,
        servers: dict[str, RegisteredMCPServer],
        force_refresh: bool = False
    ) -> list[DiscoveredResource]:
        """Entdeckt alle Ressourcen von registrierten MCP-Servern.

        Args:
            servers: Dictionary der registrierten Server
            force_refresh: Erzwingt Neuentdeckung

        Returns:
            Liste aller entdeckten Ressourcen
        """
        current_time = datetime.now(UTC)

        # Prüfe, ob Discovery erforderlich ist
        if (not force_refresh and
            self._last_discovery_run and
            (current_time - self._last_discovery_run).total_seconds() < self._discovery_interval_seconds):
            return list(self._discovered_resources.values())

        logger.info(f"Starte Resource-Discovery für {len(servers)} Server")

        discovered_resources = []

        for server_name, server in servers.items():
            if not server.is_healthy:
                logger.warning(f"Server {server_name} ist nicht gesund, überspringe Resource-Discovery")
                continue

            try:
                server_resources = await self._discover_server_resources(server_name, server)
                discovered_resources.extend(server_resources)

                # Server-Mapping aktualisieren
                self._resources_by_server[server_name] = [res.id for res in server_resources]

            except Exception as e:
                logger.exception(f"Resource-Discovery für Server {server_name} fehlgeschlagen: {e}")

        # Ressourcen nach Typ organisieren
        self._organize_resources_by_type(discovered_resources)

        # Discovery-Cache aktualisieren
        self._discovered_resources = {res.id: res for res in discovered_resources}
        self._last_discovery_run = current_time

        logger.info(f"Resource-Discovery abgeschlossen: {len(discovered_resources)} Ressourcen entdeckt")
        return discovered_resources

    async def _discover_server_resources(
        self,
        server_name: str,
        server: RegisteredMCPServer
    ) -> list[DiscoveredResource]:
        """Entdeckt Ressourcen eines spezifischen Servers."""
        resources = []

        with trace_span("mcp.resource_discovery.server", {"server": server_name}):
            for mcp_resource in server.available_resources:
                try:
                    discovered_resource = await self._convert_mcp_resource_to_discovered(
                        mcp_resource, server_name
                    )

                    # Status-Prüfung
                    status = await self._check_resource_status(discovered_resource, server)
                    discovered_resource.status = status

                    resources.append(discovered_resource)

                except Exception as e:
                    logger.exception(f"Fehler beim Verarbeiten von Ressource {mcp_resource.name}: {e}")

        return resources

    async def _convert_mcp_resource_to_discovered(
        self,
        mcp_resource: MCPResourceDefinition,
        server_name: str
    ) -> DiscoveredResource:
        """Konvertiert MCP-Ressource zu DiscoveredResource."""
        # Metadaten ableiten
        metadata = self._extract_resource_metadata(mcp_resource)

        # Ressourcen-ID generieren
        resource_id = f"{server_name}:{mcp_resource.id}"

        # URI konstruieren
        uri = f"mcp://{server_name}/{mcp_resource.id}"

        # Last-Modified parsen
        last_modified = None
        if mcp_resource.last_modified:
            with contextlib.suppress(ValueError):
                last_modified = datetime.fromisoformat(mcp_resource.last_modified.replace("Z", "+00:00"))

        return DiscoveredResource(
            id=resource_id,
            name=mcp_resource.name,
            description=mcp_resource.description,
            server_name=server_name,
            uri=uri,
            metadata=metadata,
            size_bytes=mcp_resource.size_bytes,
            last_modified=last_modified
        )

    def _extract_resource_metadata(self, mcp_resource: MCPResourceDefinition) -> ResourceMetadata:
        """Extrahiert Metadaten aus MCP-Ressource."""
        # Typ aus MCP-Typ ableiten
        resource_type = self._classify_resource_type(mcp_resource.type)

        # Metadaten aus MCP-Ressource extrahieren
        metadata_dict = mcp_resource.metadata or {}

        # Tags aus Beschreibung extrahieren
        tags = self._extract_tags_from_description(mcp_resource.description)

        return ResourceMetadata(
            resource_type=resource_type,
            access_level=ResourceAccessLevel(metadata_dict.get("access_level", "public")),
            content_type=metadata_dict.get("content_type"),
            encoding=metadata_dict.get("encoding"),
            language=metadata_dict.get("language"),
            tags=tags,
            version=metadata_dict.get("version", "1.0.0"),
            checksum=mcp_resource.etag,
            dependencies=metadata_dict.get("dependencies", []),
            cache_ttl_seconds=metadata_dict.get("cache_ttl_seconds", 3600),
            compression=metadata_dict.get("compression"),
            encryption=metadata_dict.get("encryption")
        )

    def _classify_resource_type(self, mcp_type: str) -> ResourceType:
        """Klassifiziert MCP-Typ zu ResourceType."""
        type_mapping = {
            "file": ResourceType.FILE,
            "database": ResourceType.DATABASE,
            "api": ResourceType.API_ENDPOINT,
            "stream": ResourceType.STREAM,
            "config": ResourceType.CONFIGURATION,
            "template": ResourceType.TEMPLATE,
            "schema": ResourceType.SCHEMA,
            "doc": ResourceType.DOCUMENTATION,
            "documentation": ResourceType.DOCUMENTATION
        }

        return type_mapping.get(mcp_type.lower(), ResourceType.UNKNOWN)

    def _extract_tags_from_description(self, description: str) -> set[str]:
        """Extrahiert Tags aus Ressourcen-Beschreibung."""
        tags = set()

        # Einfache Keyword-Extraktion
        keywords = [
            "json", "xml", "csv", "pdf", "image", "text", "data",
            "config", "template", "schema", "documentation", "api"
        ]

        description_lower = description.lower()
        for keyword in keywords:
            if keyword in description_lower:
                tags.add(keyword)

        return tags

    async def _check_resource_status(
        self,
        resource: DiscoveredResource,
        server: RegisteredMCPServer
    ) -> ResourceStatus:
        """Prüft Ressourcen-Status."""
        try:
            if not server.is_healthy:
                return ResourceStatus.UNAVAILABLE

            # Prüfe, ob Ressource in Server-Ressourcen-Liste vorhanden ist
            if resource.name not in [r.name for r in server.available_resources]:
                return ResourceStatus.UNAVAILABLE

            # Prüfe Cache-Status
            if self._cache.get(resource.id):
                return ResourceStatus.CACHED

            return ResourceStatus.AVAILABLE

        except Exception as e:
            logger.exception(f"Status-Prüfung für Ressource {resource.id} fehlgeschlagen: {e}")
            return ResourceStatus.ERROR

    def _organize_resources_by_type(self, resources: list[DiscoveredResource]) -> None:
        """Organisiert Ressourcen nach Typ."""
        self._resources_by_type.clear()

        for resource in resources:
            resource_type = resource.metadata.resource_type
            if resource_type not in self._resources_by_type:
                self._resources_by_type[resource_type] = []
            self._resources_by_type[resource_type].append(resource.id)

    # Public Interface Methods

    def get_resource_by_id(self, resource_id: str) -> DiscoveredResource | None:
        """Gibt Ressource anhand ID zurück."""
        return self._discovered_resources.get(resource_id)

    def get_resources_by_server(self, server_name: str) -> list[DiscoveredResource]:
        """Gibt alle Ressourcen eines Servers zurück."""
        resource_ids = self._resources_by_server.get(server_name, [])
        return [self._discovered_resources[rid] for rid in resource_ids
                if rid in self._discovered_resources]

    def get_resources_by_type(self, resource_type: ResourceType) -> list[DiscoveredResource]:
        """Gibt alle Ressourcen eines Typs zurück."""
        resource_ids = self._resources_by_type.get(resource_type, [])
        return [self._discovered_resources[rid] for rid in resource_ids
                if rid in self._discovered_resources]

    def search_resources(self, criteria: dict[str, Any]) -> list[DiscoveredResource]:
        """Sucht Ressourcen nach Kriterien."""
        matching_resources = []

        for resource in self._discovered_resources.values():
            if resource.matches_criteria(criteria):
                matching_resources.append(resource)

        return matching_resources

    async def get_resource_content(
        self,
        resource_id: str,
        user_id: str | None = None
    ) -> bytes | None:
        """Gibt Ressourcen-Inhalt zurück."""
        resource = self.get_resource_by_id(resource_id)
        if not resource:
            return None

        # Zugriffskontrolle
        if user_id and not self._access_controller.check_access(user_id, resource):
            logger.warning(f"Zugriff auf Ressource {resource_id} für Benutzer {user_id} verweigert")
            return None

        # Cache-Prüfung
        cached = self._cache.get(resource_id)
        if cached:
            return cached.content

        # Ressource von Server laden (vereinfacht)
        # In echter Implementierung würde hier der MCP-Client verwendet
        logger.info(f"Lade Ressource {resource_id} von Server {resource.server_name}")
        return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        return self._cache.get_cache_stats()


# Globale Resource-Discovery-Engine
resource_discovery_engine = ResourceDiscoveryEngine()
