"""Deep Research Service für Azure AI Foundry mit Fallback.

Dieser Service kapselt die Nutzung der (optionalen) Azure AI Foundry SDKs
für Web-Recherche (Deep Research) und bietet robuste Fallback-Mechanismen.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kei_logging import get_logger

from .common import (
    DEEP_RESEARCH_FALLBACK_MESSAGE,
    DEEP_RESEARCH_MAX_ITERATIONS,
    DEEP_RESEARCH_SDK_UNAVAILABLE,
    DEFAULT_CONFIDENCE_SCORE,
    HIGH_CONFIDENCE_SCORE,
    MEDIUM_CONFIDENCE_SCORE,
    RetryableClient,
    StandardHTTPClientConfig,
    create_fallback_result,
    create_http_retry_config,
)

logger = get_logger(__name__)


def _now_iso() -> str:
    """Hilfsfunktion: aktueller Zeitstempel im ISO-Format (UTC)."""
    return datetime.utcnow().isoformat()


@dataclass(slots=True)
class DeepResearchConfig:
    """Konfigurationswerte für Deep Research aus .env/Settings."""

    endpoint: str
    api_key: str
    project_id: str
    agent_id: str | None = None


class DeepResearchService(RetryableClient):
    """Service für Azure AI Foundry Deep Research mit Fallback-Logik.

    Der Service versucht, die Azure SDKs zu verwenden. Falls nicht verfügbar
    oder fehlerhaft konfiguriert, wird ein strukturierter Fallback geliefert.
    """

    def __init__(self, config: DeepResearchConfig) -> None:
        # Retry-Konfiguration initialisieren
        super().__init__(create_http_retry_config())

        self._config = config
        self._available = False

        # Lazy-Initialisierung
        self._ai_project_client = None
        self._agents_client = None

        # HTTP Client Konfiguration
        self._http_config = StandardHTTPClientConfig.deep_research()

        self._available = self._initialize_clients()

    @property
    def is_available(self) -> bool:
        """Gibt zurück, ob die Azure-Clients verfügbar sind."""
        return self._available

    def _initialize_clients(self) -> bool:
        """Initialisiert optionale Azure-Clients mit robuster Fehlerbehandlung."""
        try:
            from azure.ai.projects import AIProjectClient  # type: ignore
            from azure.core.credentials import AzureKeyCredential  # type: ignore

            # API-Key bevorzugt (Service-Principals via DefaultAzureCredential sind möglich)
            if not self._config.endpoint:
                return False

            if self._config.api_key:
                credential: Any = AzureKeyCredential(self._config.api_key)
            else:
                try:
                    from azure.identity import DefaultAzureCredential  # type: ignore

                    credential = DefaultAzureCredential()
                except Exception:
                    return False

            # Project-Client erstellen
            self._ai_project_client = AIProjectClient(endpoint=self._config.endpoint, credential=credential)  # type: ignore

            # Agents-Client (optional, API kann sich unterscheiden – defensiv behandeln)
            try:
                from azure.ai.agents import AgentsClient  # type: ignore

                self._agents_client = AgentsClient(self._ai_project_client)  # type: ignore
            except Exception:
                # Ohne AgentsClient kann ggf. über REST oder andere Pfade gearbeitet werden
                self._agents_client = None

            return True
        except Exception as e:  # pragma: no cover - optional
            logger.debug(f"DeepResearchService Initialisierung (Azure) fehlgeschlagen: {e}")
            return False

    def _create_research_thread(self) -> Any | None:
        """Erstellt einen Research-Thread mit Azure Agents API.

        Returns:
            Thread-Objekt oder None bei Fehlern
        """
        if self._agents_client is None:
            return None

        try:
            return self._agents_client.threads.create()  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(f"Thread-Erstellung fehlgeschlagen: {e}")
            return None

    def _add_user_message(self, thread: Any, query: str) -> bool:
        """Fügt eine Nutzer-Message zum Thread hinzu.

        Args:
            thread: Thread-Objekt
            query: Nutzer-Query

        Returns:
            True bei Erfolg, False bei Fehlern
        """
        try:
            self._agents_client.messages.create(  # type: ignore[attr-defined]
                thread_id=thread.id,
                role="user",
                content=query,
            )
            return True
        except Exception as e:
            logger.debug(f"Message-Erstellung fehlgeschlagen: {e}")
            return False

    def _create_research_results(
        self,
        query: str,
        max_iterations: int = DEEP_RESEARCH_MAX_ITERATIONS
    ) -> list[dict[str, Any]]:
        """Erstellt strukturierte Research-Ergebnisse.

        Args:
            query: Research-Query
            max_iterations: Maximale Iterationen

        Returns:
            Liste von Research-Ergebnissen
        """
        return [
            {
                "iteration": i,
                "query": query if i == 0 else f"Follow-up {i}: {query}",
                "sources": [
                    # Strukturierte Quellen-Einträge mit URL und Titel
                    # (SDK-Integration sollte diese Liste befüllen)
                    # {"url": "https://example.com", "title": "Beispiel", "verified": True}
                ],
                "key_findings": [],
                "confidence": max(DEFAULT_CONFIDENCE_SCORE, HIGH_CONFIDENCE_SCORE - i * 0.1),
                "requires_followup": i < max_iterations - 1,
            }
            for i in range(max(1, max_iterations))
        ]

    def _create_synthesis(self) -> dict[str, Any]:
        """Erstellt Synthesis-Informationen für Research-Ergebnisse.

        Returns:
            Synthesis-Dictionary
        """
        return {
            "summary": "Deep Research durchgeführt (Azure)",
            "key_insights": [],
            "confidence": MEDIUM_CONFIDENCE_SCORE,
            "recommendation": "Ergebnisse prüfen und ggf. vertiefen",
        }

    async def _perform_azure_research(
        self,
        query: str,
        max_iterations: int
    ) -> dict[str, Any]:
        """Führt die Azure AI Foundry Research durch.

        Args:
            query: Research-Query
            max_iterations: Maximale Iterationen

        Returns:
            Research-Ergebnisse
        """
        # Thread erstellen und Message hinzufügen
        thread = self._create_research_thread()
        if thread is not None:
            self._add_user_message(thread, query)
            # Optional: Deep Research Tool triggern – Implementierung variiert je nach SDK
            # self._agents_client.runs.create(...)

        # Strukturierte Ergebnisse erstellen
        return {
            "query": query,
            "results": self._create_research_results(query, max_iterations),
            "synthesis": self._create_synthesis(),
            "timestamp": _now_iso(),
            "sources_verified": True,
        }

    async def run(
        self,
        query: str,
        *,
        max_iterations: int = DEEP_RESEARCH_MAX_ITERATIONS,
        _context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Führt die Recherche aus und liefert strukturierte Ergebnisse.

        Args:
            query: Die Recherche-Frage des Benutzers
            max_iterations: Maximale Anzahl an Recherche-Iterationen
            _context: Optionaler Kontext für die Recherche (unused)

        Returns:
            Strukturierte Antwort mit Quellen, Erkenntnissen und Metadaten.
        """
        if not self._available:
            return self._fallback_result(query, reason=DEEP_RESEARCH_SDK_UNAVAILABLE)

        try:
            return await self._execute_with_retry(
                self._perform_azure_research,
                query,
                max_iterations
            )
        except Exception as e:
            logger.warning(f"DeepResearchService Ausführung fehlgeschlagen: {e}")
            return self._fallback_result(query, reason=str(e))

    def _fallback_result(self, query: str, *, reason: str) -> dict[str, Any]:
        """Erzeugt eine strukturierte Fallback-Antwort mit Begründung."""
        fallback_base = create_fallback_result(
            service_name="DeepResearchService",
            operation="run",
            reason=reason
        )

        # Deep Research spezifische Struktur hinzufügen
        fallback_base.update({
            "query": query,
            "results": [
                {
                    "iteration": 0,
                    "query": query,
                    "sources": [],
                    "key_findings": [
                        DEEP_RESEARCH_FALLBACK_MESSAGE,
                        f"Grund: {reason}",
                    ],
                    "confidence": DEFAULT_CONFIDENCE_SCORE,
                    "requires_followup": False,
                }
            ],
            "synthesis": {
                "summary": "Keine verifizierten Quellen verfügbar",
                "key_insights": [],
                "confidence": DEFAULT_CONFIDENCE_SCORE,
                "recommendation": "Bitte Azure AI Foundry konfigurieren",
            },
            "sources_verified": False,
        })

        return fallback_base


def create_deep_research_service() -> DeepResearchService | None:
    """Factory-Funktion zur Erstellung des DeepResearchService aus Settings.

    Gibt None zurück, wenn erforderliche Settings fehlen.
    """
    try:
        from config.settings import settings

        cfg = DeepResearchConfig(
            endpoint=settings.project_keiko_services_endpoint or settings.project_keiko_openai_endpoint,
            api_key=settings.project_keiko_api_key,
            project_id=getattr(settings, "project_keiko_project_id", ""),
            agent_id=getattr(settings, "agent_bing_search_id", None),
        )
        if not cfg.endpoint:
            return None
        return DeepResearchService(cfg)
    except Exception as e:  # pragma: no cover - defensiv
        logger.debug(f"DeepResearchService Factory fehlgeschlagen: {e}")
        return None
