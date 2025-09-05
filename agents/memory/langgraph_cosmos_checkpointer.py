"""LangGraph-kompatibler Checkpointer für Azure Cosmos DB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..constants import (
    LANGGRAPH_CHECKPOINT_CATEGORY,
)
from .async_utils import fire_and_forget_async, run_async_safe
from .cosmos_base import BaseCosmosMemory, CosmosOperationError
from .memory_constants import (
    CHECKPOINT_CLEAR_ERROR,
    CHECKPOINT_DELETE_ERROR,
    CHECKPOINT_ID_PREFIX,
    CHECKPOINT_LOAD_ERROR,
    CHECKPOINT_PERSIST_ERROR,
    CHECKPOINTS_DELETE_QUERY,
    CHECKPOINTS_LOAD_QUERY,
    DEFAULT_CHECKPOINT_ID,
    DEFAULT_THREAD_ID,
    LANGGRAPH_UNAVAILABLE_ERROR,
    MAX_CHECKPOINTS_PER_THREAD,
    MAX_THREAD_ID_LENGTH,
)

if TYPE_CHECKING:
    import builtins

try:  # pragma: no cover - optional import
    from azure.cosmos.aio import ContainerProxy
except Exception:  # pragma: no cover - test/runtime ohne cosmos
    ContainerProxy = object  # type: ignore

try:  # pragma: no cover - optional import
    # Optional: Falls verfügbar, verwenden wir Typen und MemorySaver aus langgraph
    from langgraph.checkpoint.base import Checkpoint
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover - fallback ohne harte Abhängigkeit
    # Mock MemorySaver für Tests/Runtime ohne LangGraph
    class MemorySaver:  # type: ignore[no-redef]
        """Mock MemorySaver für Tests ohne LangGraph."""

        def put(self, config, checkpoint, metadata=None):
            """Mock put method."""

        def get(self, _):
            """Mock get method."""
            return

        def list(self, _):
            """Mock list method."""
            return []

        def clear(self, config):
            """Mock clear method."""

    @dataclass
    class Checkpoint:  # type: ignore
        """Einfaches Fallback-Checkpoint-Modell."""

        id: str
        thread_id: str
        created_at: str
        state: dict[str, Any]
        metadata: dict[str, Any]


class CosmosCheckpointSaver(BaseCosmosMemory):
    """Checkpoint-Speicher für LangGraph mit Cosmos-Persistenz.

    Die Methoden sind absichtlich synchron, um die `MemorySaver`-API zu spiegeln.
    Interne Cosmos-Operationen werden bei Bedarf asynchron ausgeführt.
    """

    def __init__(self, container: ContainerProxy | None = None) -> None:
        """Initialisiert den Saver.

        Args:
            container: Optionaler direkter Cosmos Container. Wenn nicht gesetzt,
                wird bei Aufruf von Persist-Operationen der globale Factory genutzt.

        Raises:
            ValueError: Wenn LangGraph MemorySaver nicht verfügbar ist
        """
        super().__init__(container=container)

        # Delegate an MemorySaver für vollständige Kompatibilität der API
        if MemorySaver is None:
            raise ValueError(LANGGRAPH_UNAVAILABLE_ERROR)

        self._delegate = MemorySaver()

        # In-Memory Index: thread_id -> List[Checkpoint] für Cosmos-Replay
        self._memory: dict[str, list[Checkpoint]] = {}

    # ---------------------------------------------------------------------
    # Öffentliche API – kompatibel zu MemorySaver
    # ---------------------------------------------------------------------
    def put(
        self,
        config: dict[str, Any],
        value: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Speichert einen neuen Checkpoint.

        Args:
            config: LangGraph-Konfiguration mit `configurable.thread_id`
            value: Zustands-Dict, das gespeichert werden soll
            metadata: Optionale Metadaten

        Returns:
            Dict mit `checkpoint_id` und `thread_id` für Replay

        Raises:
            ValueError: Bei ungültigen Parametern
        """
        # Validiere Eingaben
        if not isinstance(config, dict):
            raise ValueError("Config muss ein Dict sein")
        if not isinstance(value, dict):
            raise ValueError("Value muss ein Dict sein")

        # Zuerst MemorySaver verwenden (liefert versionssichere Struktur)
        # MemorySaver.put erwartet nur: config, checkpoint, metadata
        result = self._delegate.put(config, value, metadata)

        # Für Cosmos Persist spiegeln wir einfache Metadaten
        thread_id = self._extract_thread_id(config)
        # Handle RunnableConfig object - extract checkpoint_id via attribute access or dict conversion
        if hasattr(result, "__dict__"):
            result_dict = result.__dict__
            checkpoint_id = str(
                result_dict.get("checkpoint_id") or CosmosCheckpointSaver._next_checkpoint_id(self._memory.get(thread_id))
            )
        elif hasattr(result, "checkpoint_id"):
            checkpoint_id = str(
                getattr(result, "checkpoint_id", None) or CosmosCheckpointSaver._next_checkpoint_id(self._memory.get(thread_id))
            )
        else:
            # Fallback: generate new checkpoint_id
            checkpoint_id = str(CosmosCheckpointSaver._next_checkpoint_id(self._memory.get(thread_id)))

        cp = Checkpoint(
            id=checkpoint_id,
            thread_id=thread_id,
            created_at=datetime.now(UTC).isoformat(),
            state=value,
            metadata=metadata or {},
        )
        self._memory.setdefault(thread_id, []).append(cp)

        # Best-effort Persist in Cosmos asynchron anstoßen
        try:
            fire_and_forget_async(self._persist_checkpoint(cp))
        except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_PERSIST_ERROR} - Verbindungsproblem: {e}")
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_PERSIST_ERROR} - Unerwarteter Fehler: {e}")

        return result

    def get(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Liefert den letzten Checkpoint für den Thread (Delegate-Quelle).

        Args:
            config: LangGraph-Konfiguration mit `configurable.thread_id`

        Returns:
            Checkpoint-Dict oder None wenn nicht gefunden
        """
        if not isinstance(config, dict):
            return None

        # Primär aus MemorySaver beziehen
        got = self._delegate.get(config)
        if got:
            # Konvertiere RunnableConfig zu dict falls nötig
            if hasattr(got, "__dict__"):
                return got.__dict__
            return got

        # Optional: Rehydrieren aus Cosmos, um Delegate zu füllen
        thread_id = self._extract_thread_id(config)
        try:
            records = run_async_safe(self._load_checkpoints_from_cosmos(thread_id))
            if records:
                self._memory[thread_id] = records
                # Delegate kann keine Fremd-Checkpoints übernehmen; wir geben das letzte Item roh zurück
                latest = records[-1]
                return {
                    "checkpoint_id": latest.id,
                    "thread_id": latest.thread_id,
                    "created_at": latest.created_at,
                    "state": latest.state,
                    "metadata": latest.metadata,
                }
        except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_LOAD_ERROR} - Verbindungsproblem: {e}")
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_LOAD_ERROR} - Unerwarteter Fehler: {e}")

        return None

    def list(self, config: dict[str, Any]) -> builtins.list[dict[str, Any]]:
        """Listet alle Checkpoints des Threads (Delegate-Quelle).

        Args:
            config: LangGraph-Konfiguration mit `configurable.thread_id`

        Returns:
            Liste aller Checkpoints für den Thread
        """
        if not isinstance(config, dict):
            return []
        # Konvertiere RunnableConfig zu dict falls nötig
        if hasattr(config, "__dict__"):
            config = config.__dict__
        result = self._delegate.list(config)
        # Konvertiere CheckpointTuple zu dict falls nötig
        if result and hasattr(result[0], "__dict__"):
            return [item.__dict__ if hasattr(item, "__dict__") else item for item in result]
        return result

    def clear(self, config: dict[str, Any]) -> None:
        """Löscht Checkpoints in Memory (Cosmos Bereinigung optional).

        Args:
            config: LangGraph-Konfiguration mit `configurable.thread_id`
        """
        if not isinstance(config, dict):
            return

        self._delegate.clear(config)

        thread_id = self._extract_thread_id(config)
        self._memory.pop(thread_id, None)
        try:
            fire_and_forget_async(self._delete_checkpoints_from_cosmos(thread_id))
        except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_CLEAR_ERROR} - Verbindungsproblem: {e}")
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_CLEAR_ERROR} - Unerwarteter Fehler: {e}")

    # Unbekannte Attribute an Delegate weiterreichen (z. B. get_next_version)
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - einfaches Delegat
        return getattr(self._delegate, item)

    # ---------------------------------------------------------------------
    # Cosmos Helpers (async)
    # ---------------------------------------------------------------------
    async def _persist_checkpoint(self, cp: Checkpoint) -> None:
        """Persistiert einen Checkpoint in Cosmos DB.

        Args:
            cp: Checkpoint zum Persistieren
        """
        try:
            item = {
                "id": cp.id,
                "category": LANGGRAPH_CHECKPOINT_CATEGORY,
                "thread_id": cp.thread_id,
                "created_at": cp.created_at,
                "state": cp.state,
                "metadata": cp.metadata,
            }

            await self._upsert_item(
                item,
                LANGGRAPH_CHECKPOINT_CATEGORY,
                "persist_checkpoint"
            )

        except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
            self._logger.warning(f"Persist nach Cosmos fehlgeschlagen - Verbindungsproblem: {e}")
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.warning(f"Persist nach Cosmos fehlgeschlagen - Unerwarteter Fehler: {e}")

    async def cleanup(self) -> None:
        """Führt Bereinigung alter/überschüssiger Checkpoints durch.

        Implementiert die abstrakte Methode der Basis-Klasse.
        """
        try:
            # Bereinige Checkpoints die das Limit überschreiten
            await self._cleanup_excess_checkpoints()

        except Exception as exc:
            self._logger.debug(f"Checkpoint cleanup Fehler: {exc}")

    async def _cleanup_excess_checkpoints(self) -> None:
        """Löscht überschüssige Checkpoints pro Thread."""
        try:
            # Finde alle Threads mit zu vielen Checkpoints
            threads_query = (
                "SELECT c.thread_id, COUNT(1) as checkpoint_count FROM c "
                "WHERE c.category=@cat GROUP BY c.thread_id"
            )
            params = [{"name": "@cat", "value": LANGGRAPH_CHECKPOINT_CATEGORY}]

            thread_counts = await self._execute_query(
                threads_query,
                params,
                "cleanup_excess_checkpoints"
            )

            for thread_info in thread_counts:
                thread_id = thread_info.get("thread_id")
                checkpoint_count = int(thread_info.get("checkpoint_count", 0))

                if checkpoint_count > MAX_CHECKPOINTS_PER_THREAD:
                    await self._cleanup_thread_excess(thread_id, checkpoint_count)

        except CosmosOperationError as e:
            self._logger.warning(f"Checkpoint cleanup fehlgeschlagen: {e}")

    async def _cleanup_thread_excess(self, thread_id: str, current_count: int) -> None:
        """Löscht überschüssige Checkpoints eines spezifischen Threads.

        Args:
            thread_id: Thread-ID
            current_count: Aktuelle Anzahl Checkpoints
        """
        delete_count = current_count - MAX_CHECKPOINTS_PER_THREAD

        try:
            params = [
                {"name": "@cat", "value": LANGGRAPH_CHECKPOINT_CATEGORY},
                {"name": "@tid", "value": thread_id},
            ]

            # Finde älteste Checkpoints zum Löschen
            old_query = (
                "SELECT c.id FROM c WHERE c.category=@cat AND c.thread_id=@tid "
                "ORDER BY c.created_at ASC"
            )

            old_items = await self._execute_query(
                old_query,
                params,
                "cleanup_thread_excess",
                max_results=delete_count
            )

            if old_items:
                item_ids = [item["id"] for item in old_items]
                deleted_count = await self._batch_delete_items(
                    item_ids,
                    LANGGRAPH_CHECKPOINT_CATEGORY,
                    "cleanup_thread_excess"
                )
                self._logger.debug(
                    f"Überschüssige Checkpoints gelöscht (Thread {thread_id}): {deleted_count}"
                )

        except CosmosOperationError as e:
            self._logger.warning(f"Thread cleanup fehlgeschlagen: {e}")

    async def _load_checkpoints_from_cosmos(self, thread_id: str) -> builtins.list[Checkpoint]:
        """Lädt Checkpoints eines Threads aus Cosmos DB.

        Args:
            thread_id: Thread-ID

        Returns:
            Liste der Checkpoints für den Thread
        """
        if len(thread_id) > MAX_THREAD_ID_LENGTH:
            self._logger.warning(f"Thread-ID zu lang: {thread_id}")
            return []

        try:
            params = [
                {"name": "@cat", "value": LANGGRAPH_CHECKPOINT_CATEGORY},
                {"name": "@tid", "value": thread_id},
            ]

            items = await self._execute_query(
                CHECKPOINTS_LOAD_QUERY,
                params,
                "load_checkpoints"
            )

            results: list[Checkpoint] = []
            for item in items:
                results.append(
                    Checkpoint(
                        id=str(item.get("id", "")),
                        thread_id=str(item.get("thread_id", "")),
                        created_at=str(item.get("created_at", "")),
                        state=dict(item.get("state", {})),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
            return results

        except CosmosOperationError:
            self._logger.warning(f"Cosmos Load Checkpoints fehlgeschlagen: {thread_id}")
            return []
        except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
            self._logger.warning(f"Cosmos Load Checkpoints fehlgeschlagen - Verbindungsproblem: {e}")
            return []
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.warning(f"Cosmos Load Checkpoints fehlgeschlagen - Unerwarteter Fehler: {e}")
            return []

    async def _delete_checkpoints_from_cosmos(self, thread_id: str) -> None:
        """Löscht alle Checkpoints eines Threads in Cosmos DB (best effort).

        Args:
            thread_id: Thread-ID
        """
        if len(thread_id) > MAX_THREAD_ID_LENGTH:
            self._logger.warning(f"Thread-ID zu lang: {thread_id}")
            return

        try:
            params = [
                {"name": "@cat", "value": LANGGRAPH_CHECKPOINT_CATEGORY},
                {"name": "@tid", "value": thread_id},
            ]

            items = await self._execute_query(
                CHECKPOINTS_DELETE_QUERY,
                params,
                "delete_checkpoints"
            )

            if items:
                item_ids = [item["id"] for item in items]
                deleted_count = await self._batch_delete_items(
                    item_ids,
                    LANGGRAPH_CHECKPOINT_CATEGORY,
                    "delete_checkpoints"
                )
                self._logger.debug(f"Checkpoints gelöscht (Thread {thread_id}): {deleted_count}")

        except CosmosOperationError:
            self._logger.warning(f"{CHECKPOINT_DELETE_ERROR}: {thread_id}")
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.warning(f"{CHECKPOINT_DELETE_ERROR}: {e}")


    # -------------------------------------------------------------------------
    # Private Helper-Methoden
    # -------------------------------------------------------------------------

    def _extract_thread_id(self, config: dict[str, Any]) -> str:
        """Extrahiert `thread_id` aus LangGraph-Konfiguration.

        Args:
            config: LangGraph-Konfiguration

        Returns:
            Thread-ID oder Default-Wert
        """
        cfg = config or {}
        conf = cfg.get("configurable", {}) if isinstance(cfg.get("configurable"), dict) else {}
        thread_id = str(conf.get("thread_id") or cfg.get("thread_id") or DEFAULT_THREAD_ID)

        # Validiere Thread-ID Länge
        if len(thread_id) > MAX_THREAD_ID_LENGTH:
            self._logger.warning(f"Thread-ID zu lang, verwende Default: {thread_id}")
            return DEFAULT_THREAD_ID

        return thread_id

    @staticmethod
    def _next_checkpoint_id(existing: builtins.list[Checkpoint] | None) -> str:
        """Erzeugt eine sequentielle Checkpoint-ID.

        Args:
            existing: Bestehende Checkpoints

        Returns:
            Neue Checkpoint-ID
        """
        if not existing:
            return DEFAULT_CHECKPOINT_ID

        try:
            last = existing[-1].id
            if isinstance(last, str) and last.startswith(CHECKPOINT_ID_PREFIX):
                num = int(last.split("-", 1)[1])
                return f"{CHECKPOINT_ID_PREFIX}{num + 1}"
        except (ValueError, IndexError):
            pass

        return f"{CHECKPOINT_ID_PREFIX}{len(existing) + 1}"


__all__ = ["CosmosCheckpointSaver"]
