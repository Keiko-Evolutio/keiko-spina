"""Mixin für Agent-Loading-Funktionalität.

Agent-Loading-Logik aus verschiedenen Registry-Klassen.
"""

from typing import Any

from kei_logging import get_logger

from ..utils.types import AgentInstance

logger = get_logger(__name__)


class AgentLoadingMixin:
    """Mixin für Agent-Loading-Funktionalität."""

    def __init__(self, *args, **kwargs):
        """Initialisiert das Mixin."""
        super().__init__(*args, **kwargs)

        self._foundry_available = False
        self._custom_available = False
        self._loading_errors: list[str] = []

    async def check_adapters(self) -> None:
        """Prüft verfügbare Adapter."""
        await self._check_foundry_adapter()
        await self._check_custom_adapter()

        logger.debug(
            f"Adapter-Status: Foundry={self._foundry_available}, "
            f"Custom={self._custom_available}"
        )

    async def _check_foundry_adapter(self) -> None:
        """Prüft Foundry-Adapter-Verfügbarkeit."""
        try:
            from agents.adapter import is_foundry_available
            self._foundry_available = is_foundry_available()
        except ImportError as e:
            self._foundry_available = False
            self._loading_errors.append(f"Foundry-Adapter nicht verfügbar: {e}")

    async def _check_custom_adapter(self) -> None:
        """Prüft Custom-Adapter-Verfügbarkeit."""
        try:
            from agents.custom_loader.custom_loader import load_custom_agents
            self._custom_available = True
        except ImportError as e:
            self._custom_available = False
            self._loading_errors.append(f"Custom-Adapter nicht verfügbar: {e}")

    async def load_foundry_agents(self) -> list[AgentInstance]:
        """Lädt Azure AI Foundry Agents.

        Returns:
            Liste der geladenen Agents
        """
        if not self._foundry_available:
            logger.debug("Foundry-Adapter nicht verfügbar")
            return []

        try:
            from agents.adapter import create_foundry_adapter

            adapter = await create_foundry_adapter()
            if not adapter:
                logger.warning("Foundry-Adapter konnte nicht erstellt werden")
                return []

            # Verschiedene API-Varianten unterstützen
            agents = await self._get_agents_from_adapter(adapter)

            logger.info(f"Foundry-Agents geladen: {len(agents)}")
            return agents

        except Exception as e:
            error_msg = f"Foundry-Agents laden fehlgeschlagen: {e}"
            self._loading_errors.append(error_msg)
            logger.warning(error_msg)
            return []

    async def _get_agents_from_adapter(self, adapter: Any) -> list[AgentInstance]:
        """Holt Agents aus Adapter mit verschiedenen API-Varianten.

        Args:
            adapter: Adapter-Instanz

        Returns:
            Liste der Agents
        """
        # Verschiedene API-Varianten unterstützen
        if hasattr(adapter, "list_agents"):
            return await adapter.list_agents()
        if hasattr(adapter, "get_agents"):
            return await adapter.get_agents()
        if hasattr(adapter, "agents"):
            return list(adapter.agents.values())
        logger.warning("Adapter hat keine bekannte Agent-API")
        return []

    async def load_custom_agents(self) -> list[AgentInstance]:
        """Lädt Custom Agents.

        Returns:
            Liste der geladenen Agents
        """
        if not self._custom_available:
            logger.debug("Custom-Adapter nicht verfügbar")
            return []

        try:
            from agents.custom_loader.custom_loader import load_custom_agents

            agents = await load_custom_agents()
            logger.info(f"Custom-Agents geladen: {len(agents)}")
            return agents

        except Exception as e:
            error_msg = f"Custom-Agents laden fehlgeschlagen: {e}"
            self._loading_errors.append(error_msg)
            logger.warning(error_msg)
            return []

    async def register_specialized_agents(self) -> None:
        """Registriert spezialisierte Agents.

        Konsolidierte Logik für die Registrierung von spezialisierten Agents
        wie Web Research und Image Generator.
        """
        await self._register_web_research_agent()
        await self._register_image_generator_agent()

    async def _register_web_research_agent(self) -> None:
        """Registriert Web Research Agent."""
        try:
            from config.settings import settings
            from data_models.core.core import Agent as CoreAgent

            agent_id = getattr(settings, "agent_bing_search_id", "")
            if not agent_id:
                return

            agent = CoreAgent(
                id=agent_id,
                name="Web Research Agent",
                type="web_research",
                description="Spezialisierter Agent für externe Web-Recherche",
                status="available" if self._foundry_available else "unavailable",
                capabilities=["web_research", "function_calling"],
            )

            # Registriere in der Registry (falls verfügbar)
            if hasattr(self, "_agents"):
                self._agents[agent_id] = agent
                logger.debug(f"Web Research Agent registriert: {agent_id}")

        except Exception as e:
            error_msg = f"Web Research Agent Registrierung fehlgeschlagen: {e}"
            self._loading_errors.append(error_msg)
            logger.warning(error_msg)

    async def _register_image_generator_agent(self) -> None:
        """Registriert Image Generator Agent."""
        try:
            from agents.custom.image_generator_agent import ImageGeneratorAgent
            from config.settings import settings as _settings
            from data_models.core.core import Agent as CoreAgent

            agent_id = getattr(_settings, "agent_image_generator_id", "") or "agent_image_generator"

            # Nur registrieren wenn Endpoint konfiguriert ist
            if not getattr(_settings, "project_keiko_image_endpoint", ""):
                logger.debug("Image Generator übersprungen: Kein Endpoint konfiguriert")
                return

            agent_impl = ImageGeneratorAgent()

            # Metadaten erstellen
            agent_meta = CoreAgent(
                id=agent_id,
                name="Image Generator Agent",
                type="image_generation",
                description="Erstellt Bilder mit DALL·E-3",
                status="available" if agent_impl.status == "available" else "unavailable",
                capabilities=["image_generation", "content_safety", "storage_upload"],
            )

            # Registriere beide in der Registry (falls verfügbar)
            if hasattr(self, "_agents"):
                self._agents[agent_id] = agent_impl
                self._agents[f"{agent_id}__meta"] = agent_meta
                logger.debug(f"Image Generator Agent registriert: {agent_id}")

        except Exception as e:
            error_msg = f"Image Generator Registrierung fehlgeschlagen: {e}"
            self._loading_errors.append(error_msg)
            logger.warning(error_msg)

    def get_loading_errors(self) -> list[str]:
        """Gibt Loading-Fehler zurück.

        Returns:
            Liste der Fehler
        """
        return self._loading_errors.copy()

    def clear_loading_errors(self) -> None:
        """Löscht Loading-Fehler."""
        self._loading_errors.clear()

    def get_adapter_status(self) -> dict[str, bool]:
        """Gibt Adapter-Status zurück.

        Returns:
            Status-Dictionary
        """
        return {
            "foundry_available": self._foundry_available,
            "custom_available": self._custom_available,
        }
