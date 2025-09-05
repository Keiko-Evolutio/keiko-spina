# backend/agents/metadata/service.py
"""AgentMetadata Service - Factory und Management.

Service für Agent-Metadata-Erstellung und -Management.
Kombiniert Factory-Pattern und Lifecycle-Management.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .agent_metadata import AgentMetadata, FrameworkType, MCPSpecVersion
from .constants import (
    AUTOGEN_CONFIG_TEMPLATE,
    AUTOGEN_VERSION,
    AZURE_AI_FOUNDRY_VERSION,
    AZURE_FOUNDRY_CONFIG_TEMPLATE,
    DEFAULT_AGENT_VERSION,
    ERROR_MESSAGES,
    FRAMEWORK_DETECTION_PATTERNS,
    FRAMEWORK_VERSION_MAP,
    LOG_MESSAGES,
    SEMANTIC_KERNEL_CONFIG_TEMPLATE,
    SEMANTIC_KERNEL_VERSION,
)

if TYPE_CHECKING:
    from data_models.core.core import Agent

    from ..registry.dynamic_registry import DynamicAgentRegistry

from kei_logging import get_logger

logger = get_logger(__name__)


class AgentMetadataService:
    """Einheitlicher Service für Agent-Metadata-Erstellung und -Management."""

    def __init__(self, registry: DynamicAgentRegistry | None = None):
        self.registry = registry
        self.metadata_instances: dict[str, AgentMetadata] = {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    # Factory Methods
    async def create_for_azure_foundry(
        self, agent_id: str, agent_name: str, project_id: str, endpoint: str
    ) -> AgentMetadata:
        """Erstellt AgentMetadata für Azure AI Foundry Agent."""
        framework_config = {
            **AZURE_FOUNDRY_CONFIG_TEMPLATE,
            "project_id": project_id,
            "endpoint": endpoint,
        }

        return await self._create_metadata_template(
            agent_id=agent_id,
            agent_name=agent_name,
            framework_type=FrameworkType.AZURE_AI_FOUNDRY,
            framework_version=AZURE_AI_FOUNDRY_VERSION,
            framework_config=framework_config,
            foundry_project_id=project_id,
            log_key="azure_foundry_created"
        )

    async def create_for_semantic_kernel(
        self, agent_id: str, agent_name: str, model_deployment: str
    ) -> AgentMetadata:
        """Erstellt AgentMetadata für Semantic Kernel Agent."""
        framework_config = {
            **SEMANTIC_KERNEL_CONFIG_TEMPLATE,
            "model_deployment": model_deployment,
        }

        return await self._create_metadata_template(
            agent_id=agent_id,
            agent_name=agent_name,
            framework_type=FrameworkType.SEMANTIC_KERNEL,
            framework_version=SEMANTIC_KERNEL_VERSION,
            framework_config=framework_config,
            log_key="semantic_kernel_created"
        )

    async def create_for_autogen(
        self, agent_id: str, agent_name: str, model_config: dict[str, Any]
    ) -> AgentMetadata:
        """Erstellt AgentMetadata für AutoGen Agent."""
        framework_config = {
            **AUTOGEN_CONFIG_TEMPLATE,
            "model_config": model_config,
            "temperature": model_config.get("temperature", AUTOGEN_CONFIG_TEMPLATE["temperature"]),
        }

        return await self._create_metadata_template(
            agent_id=agent_id,
            agent_name=agent_name,
            framework_type=FrameworkType.AUTOGEN,
            framework_version=AUTOGEN_VERSION,
            framework_config=framework_config,
            log_key="autogen_created"
        )

    async def create_from_agent(self, agent: Agent) -> AgentMetadata:
        """Erstellt AgentMetadata aus bestehenden Agent-Objekten."""
        agent_id = getattr(agent, "id", getattr(agent, "agent_id", "unknown"))
        agent_name = getattr(agent, "name", getattr(agent, "agent_name", "Unnamed Agent"))

        # Framework-Typ aus Agent-Eigenschaften ableiten
        framework_type = AgentMetadataService._infer_framework_type(agent)
        framework_version = AgentMetadataService._get_framework_version(framework_type)
        framework_config = AgentMetadataService._get_basic_config(agent, framework_type)

        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_name=agent_name,
            framework_type=framework_type,
            framework_version=framework_version,
            framework_config=framework_config,
            mcp_client_version=MCPSpecVersion.V2025_06_18,
        )

        await self._enhance_with_registry(metadata)
        self.logger.info(
            f"AgentMetadata aus Agent erstellt: {agent_name}",
            extra={
                "agent_id": agent_id,
                "agent_name": agent_name,
                "framework_type": framework_type.value if hasattr(framework_type, "value") else str(framework_type),
                "framework_version": framework_version,
                "operation": "metadata_creation_from_agent"
            }
        )
        return metadata

    async def _create_metadata_template(
        self,
        agent_id: str,
        agent_name: str,
        framework_type: FrameworkType,
        framework_version: str,
        framework_config: dict[str, Any],
        log_key: str,
        foundry_project_id: str | None = None,
    ) -> AgentMetadata:
        """Template-Methode für AgentMetadata-Erstellung.

        Args:
            agent_id: Agent-ID
            agent_name: Agent-Name
            framework_type: Framework-Typ
            framework_version: Framework-Version
            framework_config: Framework-spezifische Konfiguration
            log_key: Schlüssel für Log-Nachricht
            foundry_project_id: Optional Azure Foundry Projekt-ID

        Returns:
            Konfigurierte AgentMetadata
        """
        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_name=agent_name,
            framework_type=framework_type,
            framework_version=framework_version,
            framework_config=framework_config,
            foundry_project_id=foundry_project_id,
            mcp_client_version=MCPSpecVersion.V2025_06_18,
        )

        await self._enhance_with_registry(metadata)
        self.logger.info(
            LOG_MESSAGES[log_key].format(agent_name),
            extra={
                "agent_id": agent_id,
                "agent_name": agent_name,
                "framework_type": framework_type.value if hasattr(framework_type, "value") else str(framework_type),
                "framework_version": framework_version,
                "foundry_project_id": foundry_project_id,
                "operation": "metadata_creation_template"
            }
        )
        return metadata

    # Management Methods
    def register_metadata(self, metadata: AgentMetadata) -> None:
        """Registriert AgentMetadata-Instanz im Service."""
        self.metadata_instances[metadata.agent_id] = metadata
        metadata.updated_at = datetime.now(UTC)
        self.logger.info(
            LOG_MESSAGES["metadata_registered"].format(metadata.agent_id),
            extra={
                "agent_id": metadata.agent_id,
                "agent_name": metadata.agent_name,
                "framework_type": metadata.framework_type.value if hasattr(metadata.framework_type, "value") else str(metadata.framework_type),
                "total_registered": len(self.metadata_instances),
                "operation": "metadata_registration"
            }
        )

    def get_metadata(self, agent_id: str) -> AgentMetadata | None:
        """Gibt AgentMetadata für gegebene Agent-ID zurück."""
        return self.metadata_instances.get(agent_id)

    def list_all_capabilities(self) -> dict[str, list[str]]:
        """Listet alle verfügbaren Capabilities nach Agent auf."""
        return {
            agent_id: list(metadata.available_capabilities.keys())
            for agent_id, metadata in self.metadata_instances.items()
        }

    def find_agents_with_capability(self, capability_id: str) -> list[str]:
        """Findet alle Agents die eine bestimmte Capability haben."""
        return [
            agent_id
            for agent_id, metadata in self.metadata_instances.items()
            if capability_id in metadata.available_capabilities
        ]

    # Private Helper Methods
    async def _enhance_with_registry(self, metadata: AgentMetadata) -> None:
        """Erweitert Metadata mit Registry-Informationen."""
        if not self.registry:
            return

        try:
            related_agents = await self.registry.find_agents_for_task(
                task_description="general assistance", preferred_category="general"
            )

            # Füge bis zu 3 Connected Agents hinzu
            for agent_match in related_agents[:3]:
                if agent_match.agent.id != metadata.agent_id:
                    metadata.connected_agents.add(agent_match.agent.id)

            self.logger.info(
                f"Registry-Integration: {len(metadata.connected_agents)} Connected Agents",
                extra={
                    "agent_id": metadata.agent_id,
                    "connected_agents_count": len(metadata.connected_agents),
                    "connected_agents": list(metadata.connected_agents),
                    "operation": "registry_enhancement"
                }
            )

        except Exception as e:
            self.logger.warning(
                f"Registry-Integration fehlgeschlagen: {e}",
                extra={
                    "agent_id": metadata.agent_id,
                    "error_type": type(e).__name__,
                    "operation": "registry_integration"
                }
            )

    @staticmethod
    def _infer_framework_type(agent: Agent) -> FrameworkType:
        """Leitet Framework-Typ aus Agent-Eigenschaften ab.

        Args:
            agent: Agent-Objekt zur Framework-Erkennung

        Returns:
            Erkannter Framework-Typ
        """
        agent_type = str(type(agent).__name__).lower()

        # Pattern-Erkennung
        for framework, patterns in FRAMEWORK_DETECTION_PATTERNS.items():
            if any(pattern in agent_type for pattern in patterns):
                return getattr(FrameworkType, framework)

        return FrameworkType.CUSTOM_MCP

    @staticmethod
    def _get_framework_version(framework_type: FrameworkType) -> str:
        """Gibt Standard-Framework-Version zurück.

        Args:
            framework_type: Framework-Typ

        Returns:
            Framework-Version
        """
        return FRAMEWORK_VERSION_MAP.get(framework_type.name, DEFAULT_AGENT_VERSION)

    @staticmethod
    def _get_basic_config(agent: Agent, _framework_type: FrameworkType) -> dict[str, Any]:
        """Erstellt Basic-Konfiguration für Framework.

        Args:
            agent: Agent-Objekt
            _framework_type: Framework-Typ (aktuell nicht verwendet)

        Returns:
            Basic-Konfiguration für Framework
        """
        config = {}

        # Sichere Attribut-Zugriffe mit getattr
        model = getattr(agent, "model", None)
        if model:
            config["model"] = model

        temperature = getattr(agent, "temperature", None)
        if temperature is not None:
            config["temperature"] = temperature

        return config


# Globaler Service
metadata_service = AgentMetadataService()


# Convenience-Funktion
async def create_metadata(
    framework_type: str,
    agent_id: str,
    agent_name: str,
    auto_register: bool = True,
    **kwargs: Any
) -> AgentMetadata:
    """Generische Funktion zur Agent-Metadata-Erstellung.

    Args:
        framework_type: "azure_foundry", "semantic_kernel", oder "autogen"
        agent_id: Agent-ID
        agent_name: Agent-Name
        auto_register: Automatisch im Service registrieren
        **kwargs: Framework-spezifische Parameter

    Returns:
        Konfigurierte AgentMetadata

    Raises:
        ValueError: Bei unbekanntem Framework-Typ
    """
    if framework_type == "azure_foundry":
        metadata = await metadata_service.create_for_azure_foundry(
            agent_id, agent_name, kwargs.get("project_id", ""), kwargs.get("endpoint", "")
        )
    elif framework_type == "semantic_kernel":
        metadata = await metadata_service.create_for_semantic_kernel(
            agent_id, agent_name, kwargs.get("model_deployment", "")
        )
    elif framework_type == "autogen":
        metadata = await metadata_service.create_for_autogen(
            agent_id, agent_name, kwargs.get("model_config", {})
        )
    else:
        from ...core.exceptions import KeikoValidationError

        logger.error(
            f"Unbekannter Framework-Typ: {framework_type}",
            extra={
                "framework_type": framework_type,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "available_frameworks": ["azure_foundry", "semantic_kernel", "autogen"],
                "operation": "metadata_creation"
            }
        )
        raise KeikoValidationError(
            ERROR_MESSAGES["unknown_framework"].format(framework_type),
            details={"framework_type": str(framework_type)}
        )

    if auto_register:
        metadata_service.register_metadata(metadata)

    logger.info(
        f"Metadata erfolgreich erstellt für {framework_type} Agent: {agent_name}",
        extra={
            "framework_type": framework_type,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "auto_register": auto_register,
            "operation": "metadata_creation_success"
        }
    )

    return metadata
