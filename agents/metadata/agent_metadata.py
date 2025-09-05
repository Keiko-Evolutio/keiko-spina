# backend/agents/metadata/agent_metadata.py
"""Agent Metadata Core Model.

Zentrales Metadata-System für multi-agent Architekturen.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MCPSpecVersion(str, Enum):
    """MCP Specification Versions."""

    V2025_06_18 = "2025-06-18"

    @classmethod
    def get_latest(cls) -> MCPSpecVersion:
        """Gibt die neueste MCP-Version zurück."""
        return cls.V2025_06_18


class FrameworkType(str, Enum):
    """Unterstützte Agent-Frameworks."""

    AZURE_AI_FOUNDRY = "azure_ai_foundry"
    AZURE_FOUNDRY = "azure_foundry"
    SEMANTIC_KERNEL = "semantic_kernel"
    AUTOGEN = "autogen"
    CUSTOM_MCP = "custom_mcp"


class TransportType(str, Enum):
    """MCP Transport-Typen."""

    STREAMABLE_HTTP = "streamable_http"
    HTTP = "http"
    WEBSOCKET = "websocket"
    STDIO = "stdio"


class AuthMethod(str, Enum):
    """Authentifizierungsmethoden."""

    OAUTH2_1 = "oauth2.1"
    API_KEY = "api_key"  # pragma: allowlist secret
    NONE = "none"


class CapabilityCategory(str, Enum):
    """Kategorien von Agent-Capabilities gemäß KEI-Agent Spezifikation."""

    TOOLS = "tools"  # Tool-Management und -Ausführung
    SKILLS = "skills"  # Fähigkeiten und Kompetenzen
    DOMAINS = "domains"  # Domänen-spezifisches Wissen
    POLICIES = "policies"  # Governance und Compliance-Regeln


class HealthStatus(str, Enum):
    """Health-Status für Capabilities."""

    OK = "ok"  # Vollständig funktionsfähig
    DEGRADED = "degraded"  # Eingeschränkt funktionsfähig
    UNAVAILABLE = "unavailable"  # Nicht verfügbar


class ReadinessStatus(str, Enum):
    """Readiness-Status für Capabilities."""

    READY = "ready"  # Bereit für Anfragen
    STARTING = "starting"  # Startet gerade
    DRAINING = "draining"  # Wird heruntergefahren


class CapabilityStatus(str, Enum):
    """Status von Capabilities."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEPRECATED = "deprecated"


# Data Models
class MCPServerEndpoint(BaseModel):
    """MCP Server Endpoint-Definition."""

    url: str = Field(..., description="Server-URL")
    transport: TransportType = Field(..., description="Transport-Typ")
    auth_method: AuthMethod = Field(
        default=AuthMethod.NONE, description="Authentifizierungsmethode"
    )


class MCPCapabilityDescriptor(BaseModel):
    """Beschreibt eine MCP-Capability."""

    id: str = Field(..., description="Eindeutige Capability-ID")
    name: str = Field(..., description="Human-readable Name")
    description: str | None = Field(None, description="Capability-Beschreibung")
    status: CapabilityStatus = Field(default=CapabilityStatus.AVAILABLE, description="Status")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Capability-Parameter")
    # Interop-Referenzen (optionale, standardisierte Felder)
    endpoints: dict[str, Any] | None = Field(
        default=None, description="Endpoints/Topics/Services für Interop"
    )
    versions: dict[str, Any] | None = Field(
        default=None, description="Versionshinweise pro Interface"
    )


class MCPProfile(BaseModel):
    """Formales MCP-Profil (versioniert) mit Tools/Resources/Prompts.

    Dieses Objekt kann zur Validierung/Versionierung pro Agent/Capability genutzt werden.
    """

    profile_version: MCPSpecVersion = Field(
        default=MCPSpecVersion.get_latest(), description="MCP Profilversion"
    )
    tools: list[str] = Field(default_factory=list, description="Werkzeuge")
    resources: list[str] = Field(default_factory=list, description="Ressourcen")
    prompts: list[str] = Field(default_factory=list, description="Prompts")


class MCPServerDescriptor(BaseModel):
    """Beschreibt einen MCP Server."""

    server_id: str = Field(..., description="Eindeutige Server-ID")
    name: str = Field(..., description="Server-Name")
    endpoint: MCPServerEndpoint = Field(..., description="Server-Endpoint")
    capabilities: list[MCPCapabilityDescriptor] = Field(
        default_factory=list, description="Server-Capabilities"
    )
    supported_versions: list[MCPSpecVersion] = Field(
        default_factory=lambda: [MCPSpecVersion.V2025_06_18],
        description="Unterstützte MCP-Versionen",
    )


class AgentMetadata(BaseModel):
    """Core Agent Metadata Model.

    Zentrales Metadata-System für multi-agent Architekturen mit:
    - MCP Server Discovery und Management
    - Framework-agnostische Konfiguration
    - Dynamische Tool-Registrierung
    """

    # Agent Basis-Identifikation
    agent_id: str = Field(..., description="Eindeutige Agent-ID")
    agent_name: str = Field(..., description="Human-readable Agent-Name")
    agent_version: str = Field(default="1.0.0", description="Agent-Version (SemVer)")

    # Identität / Governance
    owner: str | None = Field(default=None, description="Verantwortliche/r Owner des Agents")
    tenant: str | None = Field(default=None, description="Mandant / Tenant-Zuordnung")
    tags: list[str] = Field(default_factory=list, description="Tags für Suche/Governance")

    # Framework Konfiguration
    framework_type: FrameworkType = Field(
        default=FrameworkType.AZURE_FOUNDRY, description="Zugrunde liegendes Framework"
    )
    framework_version: str = Field(..., description="Framework-Version")
    framework_config: dict[str, Any] = Field(
        default_factory=dict, description="Framework-spezifische Konfiguration"
    )

    # MCP Server Registry
    mcp_servers: dict[str, MCPServerDescriptor] = Field(
        default_factory=dict, description="Registrierte MCP Server"
    )

    # Capability Management
    available_capabilities: dict[str, MCPCapabilityDescriptor] = Field(
        default_factory=dict, description="Alle verfügbaren Capabilities"
    )

    # MCP Client Version
    mcp_client_version: MCPSpecVersion = Field(
        default_factory=MCPSpecVersion.get_latest, description="Unterstützte MCP Client-Version"
    )

    # Azure AI Foundry Integration
    foundry_project_id: str | None = Field(None, description="Azure AI Foundry Projekt-ID")
    connected_agents: set[str] = Field(
        default_factory=set, description="Verbundene Agents für Multi-Agent-Workflows"
    )

    # Discovery Konfiguration
    discovery_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "auto_discovery_enabled": True,
            "discovery_interval_seconds": 300,
            "max_discovery_depth": 3,
        },
        description="Auto-Discovery-Konfiguration",
    )

    # Temporal Metadaten
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_discovery_run: datetime | None = None

    def __init__(self, **data: Any) -> None:
        # Custom pre-validation to satisfy unit tests expecting ValueError
        required = ["agent_id", "agent_name", "framework_version"]
        missing = [k for k in required if k not in data or data.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        ft = data.get("framework_type")
        if isinstance(ft, str):
            try:
                data["framework_type"] = FrameworkType(ft)
            except Exception:
                raise ValueError("Invalid framework_type")
        super().__init__(**data)

    # SemVer / Deprecation / Docs
    migration_guide_url: str | None = Field(
        default=None, description="Link zu Migrationshinweisen"
    )
    changelog_url: str | None = Field(default=None, description="Link zum Changelog")
    deprecation_warning: str | None = Field(
        default=None, description="Berechnete Deprecation-Warnung"
    )

    class Config:
        """Pydantic-Konfiguration."""

        json_encoders = {datetime: lambda v: v.isoformat(), set: lambda v: list(v)}
        use_enum_values = False
        validate_assignment = True

    # -------------------------
    # Pre Root Validator to enforce ValueError for required fields and enum validation
    # -------------------------
    @model_validator(mode="before")
    @classmethod
    def _pre_validate_required_and_enums(cls, values):
        # Ensure required fields present; raise ValueError (tests expect ValueError)
        required = ["agent_id", "agent_name", "framework_version"]
        missing = [k for k in required if k not in values or values.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        # Validate framework_type when provided as string
        ft = values.get("framework_type")
        if isinstance(ft, str):
            try:
                FrameworkType(ft)
            except Exception:
                raise ValueError("Invalid framework_type")
        return values

    # -------------------------
    # Pydantic Feld-Validierungen
    # -------------------------
    @field_validator("agent_version")
    @classmethod
    def _validate_agent_version(cls, v: str) -> str:
        """Validiert SemVer für agent_version."""
        if not cls._is_valid_semver(v):
            raise ValueError("Ungültige SemVer-Version für agent_version")
        return v

    @field_validator("owner")
    @classmethod
    def _validate_owner_field(cls, v: str | None) -> str | None:
        """Validiert Owner-Feld (optional)."""
        if not cls._validate_owner(v):
            raise ValueError("Ungültiger Owner (1..128 sichtbare Zeichen)")
        return v

    @field_validator("tenant")
    @classmethod
    def _validate_tenant_field(cls, v: str | None) -> str | None:
        """Validiert Tenant-Feld (optional)."""
        if not cls._validate_tenant(v):
            raise ValueError("Ungültiger Tenant (Regex: [a-z0-9_-]{1,64})")
        if v is not None:
            v = v.lower()
        return v

    @field_validator("tags")
    @classmethod
    def _validate_tags_list(cls, tags: list[str]) -> list[str]:
        """Validiert Tag-Liste (max 50, eindeutige Reihenfolge) und einzelne Tags."""
        if tags is None:
            return []
        if len(tags) > 50:
            raise ValueError("Zu viele Tags (maximal 50)")

        # Validiere einzelne Tags
        for tag in tags:
            if not cls._validate_tag(tag):
                raise ValueError(f"Ungültiger Tag (Regex: [a-zA-Z0-9_-]{{1,50}}): {tag}")

        # Eindeutig bei Erhalt der Reihenfolge
        seen = set()
        unique: list[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    # Validierungen
    @classmethod
    def _is_valid_semver(cls, value: str) -> bool:
        """Prüft einfache SemVer-Konformität major.minor.patch."""
        try:
            from core.versioning import SemanticVersion

            version = SemanticVersion.parse(value)
            return (version.major, version.minor, version.patch) != (0, 0, 0) or value.startswith("0.")
        except Exception:
            return False

    @classmethod
    def _validate_tag(cls, tag: str) -> bool:
        """Validiert Tag-Format: a-z0-9-_ zwischen 1 und 50 Zeichen."""
        import re

        return bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,50}", tag))

    @classmethod
    def _validate_owner(cls, owner: str | None) -> bool:
        """Validiert Owner: 1..128 sichtbare Zeichen (ohne Steuerzeichen)."""
        if owner is None:
            return True
        if not (1 <= len(owner) <= 128):
            return False
        return all(31 < ord(ch) < 127 for ch in owner)

    @classmethod
    def _validate_tenant(cls, tenant: str | None) -> bool:
        """Validiert Tenant: a-z0-9-_ zwischen 1 und 64 Zeichen."""
        if tenant is None:
            return True
        import re

        return bool(re.fullmatch(r"[a-z0-9_-]{1,64}", tenant))

    def assign_identity(
        self,
        *,
        owner: str | None,
        tenant: str | None,
        tags: list[str] | None,
        request_tenant: str | None = None,
    ) -> None:
        """Setzt Identität nach Validierung (Owner/Tenant/Tags)."""
        if not self._validate_owner(owner):
            raise ValueError("Ungültiger Owner")
        if not self._validate_tenant(tenant):
            raise ValueError("Ungültiger Tenant")
        # Tenant-Isolation: Wenn request_tenant gesetzt, muss tenant übereinstimmen
        if request_tenant and tenant and tenant.lower() != request_tenant.lower():
            raise ValueError("Tenant-Isolation verletzt: tenant != X-Tenant-Id")
        if tags is not None:
            if len(tags) > 50:
                raise ValueError("Zu viele Tags (max 50)")
            invalid = [t for t in tags if not self._validate_tag(t)]
            if invalid:
                raise ValueError("Ungültige Tags: " + ", ".join(invalid))
            self.tags = list(dict.fromkeys(tags))
        self.owner = owner
        self.tenant = tenant.lower() if tenant else None

    def set_version(self, version: str) -> None:
        """Setzt die Agent-Version nach SemVer-Prüfung."""
        if not self._is_valid_semver(version):
            raise ValueError("Ungültige SemVer-Version")
        self.agent_version = version

    # Utility Methods
    def get_capability(self, capability_id: str) -> MCPCapabilityDescriptor | None:
        """Gibt Capability anhand der ID zurück."""
        return self.available_capabilities.get(capability_id)

    def add_mcp_server(self, server: MCPServerDescriptor) -> None:
        """Fügt MCP Server hinzu."""
        self.mcp_servers[server.server_id] = server

        # Capabilities des Servers hinzufügen
        for capability in server.capabilities:
            self.available_capabilities[capability.id] = capability

        self.updated_at = datetime.now(UTC)

    def evaluate_deprecation_against(self, target_version: str) -> None:
        """Berechnet Deprecation-Warnung gemäß SemVer-Regel (≥ 2 Minor)."""
        try:
            from core.versioning import evaluate_deprecation

            info = evaluate_deprecation(
                current_version=self.agent_version,
                target_version=target_version,
                migration_link=self.migration_guide_url,
                changelog_link=self.changelog_url,
            )
            self.deprecation_warning = info.warning if info.deprecated else None
        except Exception:
            self.deprecation_warning = None

    def remove_mcp_server(self, server_id: str) -> bool:
        """Entfernt MCP Server und dessen Capabilities."""
        if server_id not in self.mcp_servers:
            return False

        server = self.mcp_servers.pop(server_id)

        # Server-Capabilities entfernen
        for capability in server.capabilities:
            self.available_capabilities.pop(capability.id, None)

        self.updated_at = datetime.now(UTC)
        return True

    # ---------------------------------
    # Deprecation Policy – Capabilities
    # ---------------------------------
    def evaluate_capability_deprecation(
        self, capability: MCPCapabilityDescriptor, *, minor_grace: int = 2
    ) -> str | None:
        """Berechnet Deprecation‑Hinweis für Capability basierend auf SemVer Minor‑Differenz.

        Policy:
          - Wenn Agent‑Version um >= minor_grace Minor‑Versionen neuer ist als Capability‑Version,
            wird Capability als deprecated markiert und ein Hinweis erzeugt.
        """
        try:
            from core.versioning import SemanticVersion

            # Agent Version
            agent_version = SemanticVersion.parse(self.agent_version)
            # Capability Version – aus parameters.versions.capability oder versions.rpc falls vorhanden
            cap_ver = None
            if capability.versions and isinstance(capability.versions, dict):
                cap_ver = (
                    capability.versions.get("capability")
                    or capability.versions.get("rpc")
                    or capability.versions.get("grpc")
                )
            if not cap_ver and isinstance(capability.parameters, dict):
                ver_info = capability.parameters.get("versions") or {}
                if isinstance(ver_info, dict):
                    cap_ver = (
                        ver_info.get("capability") or ver_info.get("rpc") or ver_info.get("grpc")
                    )
            if not cap_ver:
                return None
            capability_version = SemanticVersion.parse(str(cap_ver))
            if (agent_version.major == capability_version.major and
                (agent_version.minor - capability_version.minor) >= minor_grace):
                hint = (
                    f"Capability '{capability.id}' ist veraltet (Agent {self.agent_version} vs Capability {cap_ver}). "
                    f"Bitte auf eine neuere Version migrieren."
                )
                return hint
        except Exception:
            return None
        return None

    def connect_agent(self, agent_id: str) -> None:
        """Verbindet einen anderen Agent."""
        self.connected_agents.add(agent_id)
        self.updated_at = datetime.now(UTC)

    def disconnect_agent(self, agent_id: str) -> bool:
        """Trennt Verbindung zu einem Agent."""
        if agent_id in self.connected_agents:
            self.connected_agents.remove(agent_id)
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def update_framework_config(self, config_updates: dict[str, Any]) -> None:
        """Aktualisiert Framework-Konfiguration."""
        self.framework_config.update(config_updates)
        self.updated_at = datetime.now(UTC)

    def add_capability(self, capability: MCPCapabilityDescriptor) -> None:
        """Fügt Capability hinzu."""
        self.available_capabilities[capability.id] = capability
        self.updated_at = datetime.now(UTC)

    def remove_capability(self, capability_id: str) -> bool:
        """Entfernt Capability."""
        if capability_id in self.available_capabilities:
            del self.available_capabilities[capability_id]
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def list_capabilities_by_status(
        self, status: CapabilityStatus
    ) -> list[MCPCapabilityDescriptor]:
        """Gibt Capabilities nach Status gefiltert zurück."""
        return [
            capability
            for capability in self.available_capabilities.values()
            if capability.status == status
        ]

    def get_mcp_server(self, server_id: str) -> MCPServerDescriptor | None:
        """Gibt MCP Server anhand der ID zurück."""
        return self.mcp_servers.get(server_id)
