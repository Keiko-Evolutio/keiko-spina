"""Dynamic Agent Registry für intelligente Agent-Discovery.

Agent-Registry mit:
- Intelligente Agent-Discovery und Matching
- Capability-basierte Suche
- Performance-Tracking und Metriken
- Automatische Agent-Kategorisierung
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)

__all__ = [
    "AgentCapability",
    "AgentCategory",
    "AgentMatch",
    "DynamicAgentRegistry",
    "dynamic_registry"
]

_CAPABILITY_KEYWORDS = {
    "code_interpreter": ["code", "python", "script", "program"],
    "file_search": ["search", "file", "document", "find"],
    "function_calling": ["function", "api", "call", "tool"],
    "web_research": ["research", "web", "internet", "external", "sources", "links"],
}

_CATEGORY_KEYWORDS = {
    "assistant": ["assistant", "helper", "support"],
    "analysis": ["analyst", "research", "data", "analyze"],
    "automation": ["automation", "workflow", "process"],
    "specialist": ["specialist", "expert", "domain"],
}


class AgentCategory(Enum):
    """Agent-Kategorien für Klassifizierung und Discovery.

    Definiert die verschiedenen Kategorien von Agents basierend auf
    ihrer primären Funktionalität und ihrem Einsatzbereich.
    """

    ASSISTANT = "assistant"
    SPECIALIST = "specialist"
    AUTOMATION = "automation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    INTEGRATION = "integration"
    ORCHESTRATION = "orchestration"
    CUSTOM = "custom"


@dataclass
class AgentCapability:
    """Agent-Fähigkeit mit Metadaten und Validierung.

    Repräsentiert eine spezifische Fähigkeit eines Agents mit
    Kategorisierung und Confidence-Score für Matching-Algorithmen.
    """

    name: str
    description: str
    category: AgentCategory
    confidence_score: float = 1.0

    def __post_init__(self) -> None:
        """Post-Initialisierung mit Validierung."""
        if not self.name or not self.name.strip():
            raise ValueError("Capability-Name darf nicht leer sein")

        if not self.description or not self.description.strip():
            raise ValueError("Capability-Beschreibung darf nicht leer sein")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score muss zwischen 0.0 und 1.0 liegen, "
                f"erhalten: {self.confidence_score}"
            )


@dataclass
class AgentMatch:
    """Agent-Suchergebnis mit Matching-Details und Validierung.

    Repräsentiert das Ergebnis einer Agent-Suche mit Relevanz-Score
    und detaillierten Matching-Informationen.
    """

    agent_id: str
    agent_name: str
    match_score: float
    capabilities: list[str]
    agent_type: str
    metadata: dict[str, Any] = field(default_factory=dict)  # Zusätzliche Metadaten

    def __post_init__(self) -> None:
        """Post-Initialisierung mit Validierung."""
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("Agent-ID darf nicht leer sein")

        if not self.agent_name or not self.agent_name.strip():
            raise ValueError("Agent-Name darf nicht leer sein")

        if not 0.0 <= self.match_score <= 1.0:
            raise ValueError(
                f"match_score muss zwischen 0.0 und 1.0 liegen, "
                f"erhalten: {self.match_score}"
            )

        if not isinstance(self.capabilities, list):
            raise ValueError("capabilities muss eine Liste sein")


class DynamicAgentRegistry:
    """Enterprise Dynamic Agent Registry für intelligente Agent-Discovery.

    Implementiert umfassende Agent-Discovery mit:
    - Capability-basierte Suche und Matching
    - Automatische Agent-Kategorisierung
    - Performance-Tracking und Caching
    - Multi-Adapter-Unterstützung
    - Thread-sichere Singleton-Implementation
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_cache_age: timedelta | None = None):
        """Thread-sichere Singleton-Implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DynamicAgentRegistry, cls).__new__(cls)
                    cls._instance._singleton_initialized = False
        return cls._instance

    def __init__(self, max_cache_age: timedelta | None = None):
        """Initialisiert Dynamic Agent Registry (nur einmal für Singleton).

        Args:
            max_cache_age: Maximales Cache-Alter für Agent-Daten
        """
        # Singleton: Nur einmal initialisieren
        if hasattr(self, "_singleton_initialized") and self._singleton_initialized:
            return

        self.agents: dict[str, Any] = {}
        self.max_cache_age = max_cache_age or timedelta(hours=1)
        self.last_refresh: datetime | None = None
        self.last_error: str | None = None
        self._initialized: bool = False

        # Adapter-Verfügbarkeit
        self._foundry_available = False
        self._custom_available = False

        # Performance-Metriken
        self._search_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        # Test-Agent Support für persistente Registration
        self._test_agents_registered = False

        self._singleton_initialized = True
        logger.debug("Dynamic Agent Registry erstellt (Singleton)")

    def register_test_agents(self) -> bool:
        """Registriert Test-Agents für Pipeline-Tests."""
        if self._test_agents_registered:
            logger.debug("Test-Agents bereits registriert")
            return True

        try:
            # Fallback Test Agent Implementation
            @dataclass
            class PersistentTestAgent:
                """Fallback Test Agent für Registry-Tests."""
                agent_id: str
                name: str
                description: str
                capabilities: list[str]
                category: str

                @property
                def id(self) -> str:
                    return self.agent_id

            # Test-Agents erstellen
            test_agents = [
                PersistentTestAgent(
                    agent_id="test_web_search_agent",
                    name="Test Web Search Agent",
                    description="Test-Agent für Web-Suche und Recherche",
                    capabilities=["web_search", "research", "web_research"],
                    category="research"
                ),
                PersistentTestAgent(
                    agent_id="test_image_generation_agent",
                    name="Test Image Generation Agent",
                    description="Test-Agent für Bildgenerierung",
                    capabilities=["image_generation", "dalle", "visual_creation"],
                    category="creative"
                ),
                PersistentTestAgent(
                    agent_id="test_conversation_agent",
                    name="Test Conversation Agent",
                    description="Test-Agent für Konversation",
                    capabilities=["conversation", "chat", "assistant"],
                    category="assistant"
                ),
                PersistentTestAgent(
                    agent_id="test_multi_capability_agent",
                    name="Test Multi-Capability Agent",
                    description="Test-Agent mit mehreren Capabilities",
                    capabilities=["web_search", "conversation", "research"],
                    category="general"
                ),
                PersistentTestAgent(
                    agent_id="test_specialized_dalle_agent",
                    name="Test Specialized DALLE Agent",
                    description="Spezialisierter Test-Agent für DALLE",
                    capabilities=["dalle", "image_generation"],
                    category="creative"
                )
            ]

            # Test-Agents registrieren
            for test_agent in test_agents:
                self.agents[test_agent.id] = test_agent

            self._test_agents_registered = True
            logger.info(f"Test-Agents registriert: {len(test_agents)} Agents")
            return True

        except Exception as e:
            logger.error(f"Test-Agent Registration fehlgeschlagen: {e}")
            return False

    @trace_function("registry.initialize")
    async def initialize(self) -> bool:
        """Initialisiert Registry und lädt Agents.

        Returns:
            True wenn Initialisierung erfolgreich

        Raises:
            RuntimeError: Wenn Initialisierung fehlschlägt
        """
        try:
            logger.debug("Registry-Initialisierung gestartet")

            # Adapter-Verfügbarkeit prüfen
            await self._check_adapters()

            # Agents laden
            await self.refresh_agents()

            self._initialized = True

            logger.info(
                f"Dynamic Agent Registry initialisiert - "
                f"Foundry: {self._foundry_available}, "
                f"Custom: {self._custom_available}, "
                f"Agents: {len(self.agents)}"
            )
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Registry-Initialisierung fehlgeschlagen: {e}")
            return False

    async def _check_adapters(self) -> None:
        """Prüft verfügbare Adapter."""
        # Foundry Adapter
        try:
            from agents.adapter import is_foundry_available
            self._foundry_available = is_foundry_available()
        except ImportError:
            is_foundry_available = lambda: False  # type: ignore
            self._foundry_available = False

        # Custom Agents
        try:
            import agents.custom_loader.custom_loader  # noqa: F401

            self._custom_available = True
        except ImportError:
            self._custom_available = False

    async def refresh_agents(self) -> None:
        """Lädt alle verfügbaren Agents."""
        if self.last_refresh and datetime.now() - self.last_refresh < self.max_cache_age:
            logger.debug(
                {
                    "event": "registry_refresh_skipped",
                    "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
                }
            )
            return

        logger.debug({"event": "registry_refresh_start"})
        self.agents.clear()

        # Foundry Agents laden
        if self._foundry_available:
            logger.debug({"event": "load_foundry_agents"})
            await self._load_foundry_agents()

        # Deep Research/Web Research Agent registrieren (falls konfiguriert)
        await self._register_web_research_agent_if_configured()

        # Image Generator Agent registrieren (falls konfiguriert)
        await self._register_image_generator_if_configured()

        # Custom Agents laden
        if self._custom_available:
            logger.debug({"event": "load_custom_agents"})
            await self._load_custom_agents()

        self.last_refresh = datetime.now()
        logger.info(f"Registry refreshed: {len(self.agents)} agents geladen")

    async def _load_foundry_agents(self) -> None:
        """Lädt Azure AI Foundry Agents."""
        try:
            from agents.adapter import create_foundry_adapter

            adapter = await create_foundry_adapter()
            if adapter:
                # Verschiedene API-Varianten unterstützen
                if hasattr(adapter, "list_agents"):
                    foundry_agents = await adapter.list_agents()
                elif hasattr(adapter, "get_agents"):
                    foundry_agents = await adapter.get_agents()
                elif hasattr(adapter, "agents"):
                    foundry_agents = adapter.agents.values()
                else:
                    foundry_agents = []

                for agent in foundry_agents:
                    agent_id = getattr(agent, "id", str(uuid4()))
                    self.agents[agent_id] = agent
        except Exception as e:
            logger.warning(f"Foundry Agents laden fehlgeschlagen: {e}")

    async def _load_custom_agents(self) -> None:
        """Lädt Custom Agents."""
        try:
            from agents.custom_loader.custom_loader import load_custom_agents

            custom_agents = await load_custom_agents()
            for agent in custom_agents:
                agent_id = getattr(agent, "id", str(uuid4()))
                self.agents[agent_id] = agent
        except Exception as e:
            logger.warning(f"Custom Agents laden fehlgeschlagen: {e}")

    async def _register_image_generator_if_configured(self) -> None:
        """Registriert nativen Image Generator Agent wenn konfiguriert."""
        try:
            from agents.custom.image_generator_agent import ImageGeneratorAgent
            from config.settings import settings as _settings
            from data_models.core.core import Agent as CoreAgent

            agent_id = getattr(_settings, "agent_image_generator_id", "") or "agent_image_generator"
            # Nur registrieren, wenn Endpoint konfiguriert ist
            if not getattr(_settings, "project_keiko_image_endpoint", ""):
                logger.debug(
                    {
                        "event": "image_agent_skip_no_endpoint",
                        "reason": "project_keiko_image_endpoint not set",
                    }
                )
                return

            logger.debug({"event": "image_agent_create", "agent_id": agent_id})
            agent_impl = ImageGeneratorAgent()
            # Note: agent_id is already set during ImageGeneratorAgent initialization via settings

            # Metadaten in Registry verfügbar machen
            agent_meta = CoreAgent(
                id=agent_id,
                name="Image Generator Agent",
                type="image_generation",
                description="Erstellt Bilder mit DALL·E-3 und speichert diese in Azure Storage",
                status="available" if agent_impl.status == "available" else "unavailable",
                capabilities=["image_generation", "content_safety", "storage_upload"],
            )
            # Beide registrieren: Metadaten und Implementierung (unter gleicher ID)
            self.agents[agent_id] = agent_impl
            self.agents[f"{agent_id}__meta"] = agent_meta
            logger.debug(
                {
                    "event": "image_agent_registered",
                    "status": agent_impl.status,
                    "capabilities": agent_impl.capabilities,
                }
            )
        except Exception as e:
            logger.warning(f"Image Generator Registrierung fehlgeschlagen: {e}")

    async def _register_web_research_agent_if_configured(self) -> None:
        """Registriert einen Web Research Agent basierend auf .env-Konfiguration.

        Nutzt Azure AI Foundry Deep Research Fähigkeit semantisch; falls die
        Foundry-Integration nicht verfügbar ist, wird ein Platzhalter-Agent
        registriert, der als nicht verfügbar markiert ist.
        """
        try:
            from config.settings import settings
            from data_models.core.core import Agent as CoreAgent

            agent_id = getattr(settings, "agent_bing_search_id", "")
            if not agent_id:
                return

            # Agent-Metadaten erstellen (stark typisiert)
            agent = CoreAgent(
                id=agent_id,
                name="Web Research Agent",
                type="web_research",
                description="Spezialisierter Agent für externe Web-Recherche mit Azure AI Foundry Deep Research",
                status="available" if self._foundry_available else "unavailable",
                capabilities=["web_research", "function_calling"],
            )

            self.agents[agent_id] = agent
        except Exception as e:  # pragma: no cover - defensiv
            logger.warning(f"Web Research Agent Registrierung fehlgeschlagen: {e}")

    async def search_agents(
        self,
        query: str | None = None,
        capabilities: list[str] | None = None,
        category: AgentCategory | None = None,
        limit: int = 10,
    ) -> list[AgentMatch]:
        """Sucht Agents basierend auf Kriterien."""
        if not self.agents or self._is_cache_expired():
            await self.refresh_agents()

        matches = []
        for agent_id, agent in self.agents.items():
            score = self._calculate_match_score(agent, query, capabilities, category)
            if score > 0:
                matches.append(
                    AgentMatch(
                        agent_id=agent_id,
                        agent_name=getattr(agent, "name", "Unknown"),
                        match_score=score,
                        capabilities=self._extract_capabilities(agent),
                        agent_type=type(agent).__name__,
                        metadata={"last_updated": self.last_refresh},
                    )
                )

        return sorted(matches, key=lambda x: x.match_score, reverse=True)[:limit]

    def _calculate_match_score(
        self,
        agent: Any,
        query: str | None,
        capabilities: list[str] | None,
        category: AgentCategory | None,
    ) -> float:
        """Berechnet Match-Score für Agent."""
        score = 0.0

        # Text-Matching
        if query:
            agent_text = ""
            if hasattr(agent, "name") and agent.name:
                agent_text += agent.name.lower() + " "
            if hasattr(agent, "description") and agent.description:
                agent_text += agent.description.lower() + " "

            if query.lower() in agent_text:
                score += 0.8

        # Capabilities-Matching
        if capabilities:
            agent_caps = self._extract_capabilities(agent)
            matching_caps = set(capabilities) & set(agent_caps)
            if matching_caps:
                score += 0.6 * (len(matching_caps) / len(capabilities))

        # Category-Matching
        if category:
            agent_category = self._infer_category(agent)
            if agent_category == category:
                score += 0.4

        return min(score, 1.0)

    @staticmethod
    def _extract_capabilities(agent: Any) -> list[str]:
        """Extrahiert Capabilities aus Agent."""
        capabilities = []

        # Tool-basierte Capabilities
        if hasattr(agent, "tools") and agent.tools:
            for tool in agent.tools:
                if isinstance(tool, dict) and "type" in tool:
                    capabilities.append(tool["type"])
                elif hasattr(tool, "type"):
                    capabilities.append(tool.type)

        # Beschreibungsbasierte Capabilities
        if hasattr(agent, "description") and agent.description:
            desc_lower = agent.description.lower()
            for capability, keywords in _CAPABILITY_KEYWORDS.items():
                if any(keyword in desc_lower for keyword in keywords):
                    capabilities.append(capability)

        return capabilities

    @staticmethod
    def _infer_category(agent: Any) -> AgentCategory:
        """Inferiert Agent-Kategorie."""
        if hasattr(agent, "name") and agent.name:
            name_lower = agent.name.lower()
            for category_name, keywords in _CATEGORY_KEYWORDS.items():
                if any(keyword in name_lower for keyword in keywords):
                    return AgentCategory(category_name)
        return AgentCategory.CUSTOM

    def _is_cache_expired(self) -> bool:
        """Prüft Cache-Ablauf."""
        if not self.last_refresh:
            return True
        return datetime.now() - self.last_refresh > self.max_cache_age

    async def get_agent_by_id(self, agent_id: str) -> Any | None:
        """Holt Agent by ID."""
        if self._is_cache_expired():
            await self.refresh_agents()
        return self.agents.get(agent_id)

    async def get_agent(self, agent_id: str) -> Any | None:
        """Holt Agent by ID - Alias für get_agent_by_id."""
        return await self.get_agent_by_id(agent_id)

    async def list_agents(self) -> list[Any]:
        """Gibt Liste aller verfügbaren Agents zurück."""
        if self._is_cache_expired():
            await self.refresh_agents()
        return list(self.agents.values())

    async def start(self) -> None:
        """Startet Registry - Alias für initialize()."""
        await self.initialize()

    async def stop(self) -> None:
        """Bereinigt Registry."""
        self.agents.clear()
        self.last_refresh = None
        self._initialized = False
        logger.info("Dynamic Agent Registry gestoppt")

    async def find_agents_for_task(
        self,
        task_description: str | None = None,
        required_capabilities: list[str] | None = None,
        preferred_category: str | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Kompatibilitätsmethode: liefert Ergebnisse mit.agent, .match_score usw."""
        if not self.agents or self._is_cache_expired():
            await self.refresh_agents()

        # Hilfsobjekt für Rückgabe
        @dataclass
        class _CompatMatch:
            agent: Any
            match_score: float
            load_factor: float
            estimated_response_time: float
            matched_capabilities: list[str]

        results: list[_CompatMatch] = []

        req_caps_lower = [c.lower() for c in (required_capabilities or [])]
        for agent in self.agents.values():
            caps = self._extract_capabilities(agent)
            caps_lower = [c.lower() for c in caps]
            matched_caps = [c for c in req_caps_lower if c in caps_lower] if req_caps_lower else []

            # Score basierend auf Text und Capabilities
            score = 0.0
            if task_description:
                score += self._calculate_match_score(agent, task_description, None, None)
            if matched_caps:
                score = min(1.0, score + 0.6 * (len(matched_caps) / (len(req_caps_lower) or 1)))

            # Kategoriepräferenz leicht gewichten
            if preferred_category:
                inferred = self._infer_category(agent).value
                if inferred == preferred_category:
                    score = min(1.0, score + 0.1)

            # Einfache Heuristiken für Auslastung und ETA
            load_factor = max(0.05, 1.0 - score * 0.8)
            eta_seconds = max(0.1, 5.0 - score * 4.0)

            results.append(
                _CompatMatch(
                    agent=agent,
                    match_score=round(score, 4),
                    load_factor=round(load_factor, 4),
                    estimated_response_time=round(eta_seconds, 3),
                    matched_capabilities=matched_caps,
                )
            )

        # Sortiert nach Score, besten zuerst
        results.sort(key=lambda r: r.match_score, reverse=True)
        return results[:limit]

    def get_status(self) -> dict[str, Any]:
        """Gibt Registry-Status zurück."""
        return {
            "total_agents": len(self.agents),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "foundry_available": self._foundry_available,
            "custom_available": self._custom_available,
            "cache_expired": self._is_cache_expired(),
            "last_error": self.last_error,
        }

    def is_initialized(self) -> bool:
        """Public API für Initialisierungs-Status.

        Returns:
            True wenn Registry initialisiert ist
        """
        return self._initialized


# Globale Registry-Instanz
dynamic_registry = DynamicAgentRegistry()
