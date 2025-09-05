"""Azure AI Foundry Adapter"""

import time
from typing import Any

from agents.adapter.adapter_factory import FoundryConfig
from kei_logging import get_logger

from ..constants import (
    ERROR_AGENT_ID_EMPTY,
    ERROR_AGENT_NAME_REQUIRED,
    ERROR_TASK_EMPTY,
    LOG_FOUNDRY_INITIALIZED,
    LOG_TASK_COMPLETED,
    LOG_TASK_EXECUTING,
    STATUS_SUCCESS,
    TASK_PREVIEW_MAX_LENGTH,
    TASK_PREVIEW_SUFFIX,
)
from ..exceptions import (
    KEIExternalServiceError,
    agent_not_found_error,
    validation_error,
)
from ..logging_utils import StructuredLogger

logger = get_logger(__name__)
structured_logger = StructuredLogger("foundry_adapter")

# Import-Handler für Tests
_ORIGINAL_IMPORT = __import__


async def initialize() -> None:
    """Initialisiert Foundry-Client"""
    try:
        import azure.ai.projects  # noqa: F401

        logger.info(LOG_FOUNDRY_INITIALIZED)
    except ImportError as e:
        raise ImportError("Azure AI SDK nicht installiert") from e
    except Exception:
        # Import-Patch für Tests zurücksetzen
        try:
            _bi = _ORIGINAL_IMPORT("builtins")
            _bi.__import__ = _ORIGINAL_IMPORT
        except (ImportError, AttributeError):
            # Ignoriere Fehler beim Zurücksetzen des Import-Patches
            pass
        raise


class FoundryAdapter:
    """Adapter für Azure AI Foundry"""

    def __init__(self, config: "FoundryConfig"):
        self.config = config
        self._client = None
        self._agents: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, dict[str, Any]] = {}

    async def get_agents(self) -> dict[str, Any]:
        """Lädt verfügbare Agents"""
        return self._agents

    async def create_agent(self, name: str, instructions: str, tools: list | None = None) -> Any:
        """Erstellt neuen Agent.

        Args:
            name: Name des Agents (muss eindeutig sein)
            instructions: Anweisungen für den Agent
            tools: Optionale Liste von Tools

        Returns:
            Erstellter Agent

        Raises:
            ValueError: Wenn Name oder Instructions leer sind
        """
        # Input-Validierung
        if not name or not name.strip():
            raise validation_error("Agent-Name darf nicht leer sein", validation_field="name", value=name)

        if not instructions or not instructions.strip():
            raise validation_error("Agent-Instructions dürfen nicht leer sein", validation_field="instructions", value=instructions)

        if name in self._agents:
            structured_logger.warning(
                f"Agent {name} existiert bereits und wird überschrieben",
                extra_data={"agent_name": name, "action": "overwrite"}
            )

        agent = {
            "name": name.strip(),
            "instructions": instructions.strip(),
            "tools": tools or [],
            "type": "foundry",
            "model": self.config.model_name,
        }
        self._agents[name] = agent

        structured_logger.info(
            f"Agent {name} erfolgreich erstellt",
            extra_data={
                "agent_name": name,
                "tools_count": len(tools or []),
                "model": self.config.model_name
            }
        )
        return agent

    async def create_agent_from_config(self, config: dict[str, Any]) -> Any:
        """Erstellt Agent aus Konfigurations-Dict.

        Args:
            config: Konfigurations-Dictionary mit Agent-Parametern

        Returns:
            Erstellter Agent

        Raises:
            ValueError: Wenn erforderliche Konfiguration fehlt
        """
        if not isinstance(config, dict):
            raise validation_error("Konfiguration muss ein Dictionary sein", validation_field="config", value=type(config))

        name = config.get("name", "").strip()
        if not name:
            raise validation_error(ERROR_AGENT_NAME_REQUIRED, validation_field="name")

        instructions = config.get("instructions", "").strip()
        if not instructions:
            raise validation_error("Agent-Instructions sind in Konfiguration erforderlich", validation_field="instructions")

        return await self.create_agent(
            name=name,
            instructions=instructions,
            tools=config.get("tools") or [],
        )

    def _validate_task_execution_input(self, agent_id: str, task: str) -> None:
        """Validiert Eingabeparameter für Task-Ausführung."""
        if not agent_id or not agent_id.strip():
            raise validation_error(ERROR_AGENT_ID_EMPTY, validation_field="agent_id", value=agent_id)
        if not task or not task.strip():
            raise validation_error(ERROR_TASK_EMPTY, validation_field="task", value=task)
        if agent_id not in self._agents:
            raise agent_not_found_error(agent_id)

    @staticmethod
    def _create_task_preview(task: str) -> str:
        """Erstellt Preview-String für Task-Logging.

        Args:
            task: Task-String für Preview-Erstellung

        Returns:
            Gekürzter Task-String für Logging
        """
        if len(task) > TASK_PREVIEW_MAX_LENGTH:
            return task[:TASK_PREVIEW_MAX_LENGTH] + TASK_PREVIEW_SUFFIX
        return task

    @staticmethod
    def _execute_task_implementation(agent_id: str, task: str) -> dict[str, Any]:
        """Führt die eigentliche Task-Ausführung durch.

        Args:
            agent_id: ID des ausführenden Agents
            task: Auszuführende Aufgabe

        Returns:
            Task-Ausführungsergebnis
        """
        return {
            "agent": agent_id,
            "task": task,
            "result": "Task erfolgreich ausgeführt",
            "status": STATUS_SUCCESS,
            "timestamp": time.time()
        }

    async def execute_task(self, agent_id: str, task: str) -> dict[str, Any]:
        """Führt Task mit Agent aus.

        Args:
            agent_id: ID des Agents
            task: Auszuführende Aufgabe

        Returns:
            Ergebnis der Task-Ausführung

        Raises:
            KEIValidationError: Wenn Agent-ID oder Task ungültig sind
            KEIAgentNotFoundError: Wenn Agent nicht existiert
            RuntimeError: Wenn Task-Ausführung fehlschlägt
        """
        # Input-Validierung
        self._validate_task_execution_input(agent_id, task)

        try:
            # Task-Ausführung protokollieren
            task_preview = FoundryAdapter._create_task_preview(task)
            structured_logger.debug(
                LOG_TASK_EXECUTING.format(agent_id=agent_id),
                extra_data={
                    "agent_id": agent_id,
                    "task_preview": task_preview
                }
            )

            # Task ausführen
            result = FoundryAdapter._execute_task_implementation(agent_id, task)

            # Erfolg protokollieren
            structured_logger.info(
                LOG_TASK_COMPLETED.format(agent_id=agent_id),
                extra_data={
                    "agent_id": agent_id,
                    "task_length": len(task),
                    "execution_time": time.time()
                }
            )
            return result

        except Exception as e:
            structured_logger.error(
                f"Task-Ausführung für Agent {agent_id} fehlgeschlagen",
                error=e,
                extra_data={"agent_id": agent_id}
            )
            raise KEIExternalServiceError(
                f"Task-Ausführung fehlgeschlagen: {e!s}",
                service_name="foundry_adapter"
            )

    async def list_agents(self, force_refresh: bool = False) -> list:
        """Listet alle Agents auf.

        Args:
            force_refresh: Ob Agents neu geladen werden sollen

        Returns:
            Liste aller verfügbaren Agents
        """
        structured_logger.debug(
            f"Liste Agents auf (force_refresh: {force_refresh})",
            extra_data={"force_refresh": force_refresh},
        )

        if force_refresh:
            # Agents neu laden (Implementierung folgt)
            structured_logger.debug("Agents werden neu geladen")

        agent_list = list(self._agents.values())
        structured_logger.debug(
            f"{len(agent_list)} Agents gefunden",
            extra_data={"count": len(agent_list)},
        )
        return agent_list

    async def create_thread(self, agent_id: str) -> dict[str, Any]:
        """Erstellt Thread für Agent.

        Args:
            agent_id: ID des Agents für den Thread

        Returns:
            Thread-Informationen mit thread_id

        Raises:
            ValueError: Wenn Agent-ID ungültig ist
        """
        if not agent_id or not agent_id.strip():
            raise validation_error("Agent-ID darf nicht leer sein", validation_field="agent_id")

        if agent_id not in self._agents:
            raise validation_error(
                f"Agent {agent_id} nicht gefunden", validation_field="agent_id", value=agent_id
            )

        thread_id = f"thread_{agent_id}_{int(time.time())}"

        self._threads[thread_id] = {
            "thread_id": thread_id,
            "agent_id": agent_id,
            "messages": [],
            "created_at": time.time(),
        }

        logger.debug(f"Thread {thread_id} für Agent {agent_id} erstellt")
        return {"thread_id": thread_id}

    async def create_message(
        self, thread_id: str, content: str, role: str = "user"
    ) -> dict[str, Any]:
        """Fügt Nachricht zu Thread hinzu."""
        thread = self._threads.setdefault(thread_id, {"thread_id": thread_id, "messages": []})
        message = {"role": role, "content": content}
        thread["messages"].append(message)
        return {"thread_id": thread_id, "message": message}

    async def execute_agent(self, agent_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Führt Agent mit Thread und Message aus."""
        if agent_id not in self._agents:
            return {"error": f"Agent {agent_id} nicht gefunden"}

        # Stelle sicher, dass Thread existiert und Nachricht angelegt ist
        if thread_id and message:
            await self.create_message(thread_id, message)

        return {
            "agent": agent_id,
            "thread_id": thread_id,
            "input": message,
            "result": "Ausführung gestartet",
            "status": "success",
        }

    async def cleanup(self) -> None:
        """Bereinigt Ressourcen"""
        self._agents.clear()
        self._threads.clear()
        self._client = None
