"""Fabrik für dynamische LangGraph-Nodes basierend auf Agent-Capabilities.

Diese Komponente untersucht die Dynamic Registry und erzeugt zur Laufzeit
Node-Funktionen für die aktuell verfügbaren Agents, gefiltert nach
Capabilities. Sie ermöglicht eine anpassbare Anzahl dynamischer Nodes und
stellt Fallbacks bereit, falls keine passenden Agents verfügbar sind.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .workflows_constants import (
    DEFAULT_MAX_DYNAMIC_NODES,
    ERROR_MESSAGES,
    NODE_PREFIXES,
    STATE_KEYS,
    WORKFLOW_NODES,
)
from .workflows_utils import run_sync

logger = get_logger(__name__)


DynamicNode = tuple[str, Callable[[Any], Any]]
DynamicEdges = list[tuple[str, str]]


@dataclass(slots=True)
class DynamicNodeFactoryConfig:
    """Konfiguration für die dynamische Node-Fabrik.

    Attributes:
        max_dynamic_nodes: Obergrenze der erzeugten dynamischen Nodes
        connect_from: Optionaler Vorgänger-Node für erzeugte Kanten
        connect_to: Optionaler Nachfolger-Node für erzeugte Kanten
    """

    max_dynamic_nodes: int = DEFAULT_MAX_DYNAMIC_NODES
    connect_from: str | None = None
    connect_to: str | None = None


class DynamicNodeFactory:
    """Erzeugt dynamische Nodes aus der Dynamic Registry.

    Diese Klasse abstrahiert die Laufzeit-Erzeugung von LangGraph-Nodefunktionen
    für gefundene Agents. Sie filtert anhand Capabilities und begrenzt die
    Anzahl der erzeugten Nodes gemäß Konfiguration.
    """

    def __init__(self, *, config: DynamicNodeFactoryConfig | None = None) -> None:
        """Initialisiert die Fabrik.

        Args:
            config: Optionale Konfiguration
        """
        self.config = config or DynamicNodeFactoryConfig()

    async def create_nodes(
        self,
        *,
        required_capabilities: list[str] | None = None,
        registry: Any,
    ) -> tuple[list[DynamicNode], DynamicEdges]:
        """Erzeugt dynamische Nodes und optionale Kanten.

        Args:
            required_capabilities: Liste benötigter Capabilities
            registry: Dynamic Registry Instanz

        Returns:
            Tuple aus Liste dynamischer Nodes und den dazugehörigen Kanten
        """
        try:
            if not getattr(registry, "_initialized", False):
                await registry.start()  # type: ignore[func-returns-value]
        except Exception as e:  # pragma: no cover - defensiv
            logger.warning(f"Registry-Start fehlgeschlagen: {e}")

        nodes: list[DynamicNode] = []
        edges: DynamicEdges = []

        try:
            candidates = self._filter_agents(
                agents=getattr(registry, "agents", {}),
                required_capabilities=required_capabilities or [],
            )
        except Exception as e:  # pragma: no cover - defensiv
            logger.warning(f"Agent-Filter fehlgeschlagen: {e}")
            candidates = []

        # Begrenze Anzahl dynamischer Nodes
        limit = max(0, int(self.config.max_dynamic_nodes))
        selected = candidates[:limit]

        if not selected:
            # Graceful Degradation: Fallback-Node erzeugen
            def _noop(state: Any) -> Any:
                # Setzt nur einen Fehlerhinweis in den State-Extras
                try:
                    extras = getattr(state, STATE_KEYS["EXTRAS"], {})
                    if isinstance(extras, dict):
                        extras[STATE_KEYS["DYNAMIC_FALLBACK"]] = True
                        extras[STATE_KEYS["DYNAMIC_REASON"]] = ERROR_MESSAGES["NO_MATCHING_AGENTS"]
                        setattr(state, STATE_KEYS["EXTRAS"], extras)
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Fehler beim Setzen der State-Extras für Fallback: {e}")
                except Exception as e:
                    logger.warning(f"Unerwarteter Fehler beim Fallback-State-Setup: {e}")
                return state

            nodes.append((WORKFLOW_NODES["DYNAMIC_NOOP"], _noop))
            self._append_edges(edges, from_node=self.config.connect_from, to_node=WORKFLOW_NODES["DYNAMIC_NOOP"])
            if self.config.connect_to:
                edges.append((WORKFLOW_NODES["DYNAMIC_NOOP"], self.config.connect_to))
            return nodes, edges

        # Für jeden Kandidaten eine Node-Funktion bauen
        for agent_id, _meta in selected:
            node_id = f"{NODE_PREFIXES['INVOKE']}{agent_id}"
            fn = self._make_agent_invoke_node(agent_id)
            nodes.append((node_id, fn))
            self._append_edges(edges, from_node=self.config.connect_from, to_node=node_id)
            if self.config.connect_to:
                edges.append((node_id, self.config.connect_to))

        return nodes, edges

    @staticmethod
    def _append_edges(edges: DynamicEdges, *, from_node: str | None, to_node: str) -> None:
        """Fügt Kanten hinzu, falls ein Vorgänger definiert ist.

        Args:
            edges: Kantenliste
            from_node: Vorgänger-Node-ID oder None
            to_node: Ziel-Node-ID
        """
        if from_node:
            edges.append((from_node, to_node))

    @staticmethod
    def _filter_agents(
        *, agents: dict[str, Any], required_capabilities: Iterable[str]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Filtert Agents anhand Capabilities und liefert geordnete Kandidaten.

        Args:
            agents: Registry-Agents (ID -> Objekt/Metadaten)
            required_capabilities: Muss-Capabilities

        Returns:
            Liste von (agent_id, metadata) Kandidaten, absteigend nach Score
        """
        req = {str(c).lower() for c in required_capabilities}
        scored: list[tuple[str, dict[str, Any], int]] = []
        for agent_id, agent in agents.items():
            caps = []
            if hasattr(agent, "capabilities"):
                try:
                    caps = [str(c).lower() for c in agent.capabilities]  # type: ignore[arg-type]
                except Exception:
                    caps = []
            score = len(req.intersection(set(caps))) if req else 1
            if score > 0:
                meta = {
                    "id": agent_id,
                    "name": getattr(agent, "name", agent_id),
                    "capabilities": getattr(agent, "capabilities", []),
                    "score": score,
                }
                scored.append((agent_id, meta, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(a_id, meta) for a_id, meta, _ in scored]

    @staticmethod
    def _make_agent_invoke_node(agent_id: str) -> Callable[[Any], Any]:
        """Erzeugt eine Node-Funktion, die einen spezifischen Agent ausführt.

        Args:
            agent_id: Ziel-Agent-ID
        """

        def _invoke(state: Any) -> Any:
            # Defensive Imports innerhalb der Node-Funktion, um Importkosten zu minimieren
            try:
                from agents.common.operations import execute_agent_task  # type: ignore

                # Nachricht aus State extrahieren
                message = getattr(state, STATE_KEYS["MESSAGE"], None)
                result = run_sync(
                    execute_agent_task(agent_id=agent_id, task=message, framework=None)
                )
                # Ergebnis in State ablegen
                if hasattr(state, STATE_KEYS["EXTRAS"]) and isinstance(getattr(state, STATE_KEYS["EXTRAS"]), dict):
                    state.extras[f"{NODE_PREFIXES['RESULT']}{agent_id}"] = str(result)
                else:
                    with contextlib.suppress(AttributeError, TypeError):
                        setattr(state, STATE_KEYS["EXTRAS"], {f"{NODE_PREFIXES['RESULT']}{agent_id}": str(result)})
            except (ValueError, TypeError) as e:  # pragma: no cover - defensiv
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(state, STATE_KEYS["ERROR"], f"{ERROR_MESSAGES['DYNAMIC_INVOKE_FAILED']}:{agent_id}:{e}")
            except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(state, STATE_KEYS["ERROR"], f"{ERROR_MESSAGES['DYNAMIC_INVOKE_FAILED']}:{agent_id}:Verbindungsproblem:{e}")
            except Exception as e:  # pragma: no cover - defensiv
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(state, STATE_KEYS["ERROR"], f"{ERROR_MESSAGES['DYNAMIC_INVOKE_FAILED']}:{agent_id}:Unerwarteter Fehler:{e}")
            return state

        return _invoke

__all__ = [
    "DynamicNodeFactory",
    "DynamicNodeFactoryConfig",
]
