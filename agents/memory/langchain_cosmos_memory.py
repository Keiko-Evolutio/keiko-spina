"""LangChain-kompatibler Chat-Memory auf Basis von Azure Cosmos DB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from .async_utils import MICROSECOND_OFFSET, create_unique_id, validate_session_id
from .cosmos_base import BaseCosmosMemory, CosmosOperationError
from .memory_constants import (
    CHAT_MEMORY_APPEND_ERROR,
    CHAT_MEMORY_CLEANUP_ERROR,
    CHAT_MEMORY_DELETE_ERROR,
    CHAT_MEMORY_EXCESS_DELETE_ERROR,
    CHAT_MEMORY_LOAD_ERROR,
    CHAT_MESSAGE_CATEGORY,
    CHAT_MESSAGES_DELETE_EXCESS_QUERY,
    CHAT_MESSAGES_LOAD_QUERY,
    DEFAULT_MAX_AGE_HOURS,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MESSAGE_ROLE,
    MAX_MESSAGE_CONTENT_LENGTH,
    MAX_SESSION_ID_LENGTH,
    VALID_MESSAGE_ROLES,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

try:  # pragma: no cover - optional import
    from azure.cosmos.aio import ContainerProxy
except Exception:  # pragma: no cover - test/runtime ohne cosmos
    ContainerProxy = object  # type: ignore


@dataclass(slots=True)
class MemoryRetention:
    """Konfiguration für Aufbewahrung und Bereinigung."""

    max_messages: int = DEFAULT_MAX_MESSAGES
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS


class CosmosChatMemory(BaseCosmosMemory):
    """Persistenter Chat-Speicher für Konversationshistorie.

    Nachrichten werden als flache Items mit Kategorie `chat_message` abgelegt:
    - id: zusammengesetzt aus session_id und timestamp
    - session_id: Partition Key
    - role: user|assistant|system
    - content: Textinhalt
    - created_at: ISO-8601 Zeitstempel
    """

    def __init__(
        self,
        *,
        retention: MemoryRetention | None = None,
        container: ContainerProxy | None = None
    ) -> None:
        """Initialisiert den Speicher.

        Args:
            retention: Optionale Retention-Policy
            container: Optionaler direkter Cosmos Container
        """
        super().__init__(container=container)
        self._retention = retention or MemoryRetention()

    async def load_messages(self, session_id: str) -> list[dict[str, str]]:
        """Lädt Nachrichten für eine Session, sortiert nach Zeit.

        Args:
            session_id: Eindeutige Session-ID

        Returns:
            Liste von Nachrichten im Format {"role": str, "content": str}

        Raises:
            ValueError: Bei ungültiger session_id
        """
        validate_session_id(session_id, MAX_SESSION_ID_LENGTH)

        try:
            params = [
                {"name": "@cat", "value": CHAT_MESSAGE_CATEGORY},
                {"name": "@sid", "value": session_id},
            ]

            items = await self._execute_query(
                CHAT_MESSAGES_LOAD_QUERY,
                params,
                "load_messages"
            )

            results: list[dict[str, str]] = []
            for item in items:
                role = str(item.get("role", DEFAULT_MESSAGE_ROLE))
                content = str(item.get("content", ""))

                # Validiere Role
                if role not in VALID_MESSAGE_ROLES:
                    role = DEFAULT_MESSAGE_ROLE

                results.append({"role": role, "content": content})

            return results

        except CosmosOperationError:
            self._logger.warning(f"{CHAT_MEMORY_LOAD_ERROR}: {session_id}")
            return []
        except Exception as exc:
            self._logger.warning(f"{CHAT_MEMORY_LOAD_ERROR}: {exc}")
            return []

    @staticmethod
    def _normalize_message(msg: dict[str, str]) -> tuple[str, str]:
        """Normalisiert und validiert eine einzelne Nachricht.

        Args:
            msg: Nachrichten-Dict mit role und content

        Returns:
            Tuple aus (role, content) nach Normalisierung
        """
        role = str(msg.get("role", DEFAULT_MESSAGE_ROLE))
        content = str(msg.get("content", ""))

        if role not in VALID_MESSAGE_ROLES:
            role = DEFAULT_MESSAGE_ROLE

        if len(content) > MAX_MESSAGE_CONTENT_LENGTH:
            content = content[:MAX_MESSAGE_CONTENT_LENGTH]

        return role, content

    @staticmethod
    def _create_message_item(
        session_id: str,
        role: str,
        content: str,
        timestamp: float
    ) -> dict[str, str]:
        """Erstellt ein Cosmos DB Item für eine Nachricht.

        Args:
            session_id: Session-ID
            role: Nachrichten-Rolle
            content: Nachrichten-Inhalt
            timestamp: Zeitstempel

        Returns:
            Cosmos DB Item-Dict
        """
        unique_id = create_unique_id("", session_id, timestamp, role[0])

        return {
            "id": unique_id,
            "category": CHAT_MESSAGE_CATEGORY,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(UTC).isoformat(),
        }

    async def append_messages(
        self,
        session_id: str,
        messages: Iterable[dict[str, str]]
    ) -> None:
        """Persistiert Nachrichten sequentiell und triggert Bereinigung.

        Args:
            session_id: Eindeutige Session-ID
            messages: Iterable von Nachrichten-Dicts

        Raises:
            ValueError: Bei ungültigen Parametern
        """
        validate_session_id(session_id, MAX_SESSION_ID_LENGTH)

        message_list = list(messages)
        if not message_list:
            return

        success_count = await self._persist_messages(session_id, message_list)
        self._logger.debug(f"Nachrichten gespeichert: {success_count}/{len(message_list)}")

        # Nach dem Schreiben Bereinigung anstoßen (best effort)
        try:
            await self.cleanup()
        except (ConnectionError, TimeoutError) as e:  # pragma: no cover - defensiv
            self._logger.debug(f"Memory-Cleanup fehlgeschlagen - Verbindungsproblem: {e}")
        except Exception as e:  # pragma: no cover - defensiv
            self._logger.debug(f"Memory-Cleanup fehlgeschlagen - Unerwarteter Fehler: {e}")

    async def _persist_messages(
        self,
        session_id: str,
        message_list: list[dict[str, str]]
    ) -> int:
        """Persistiert eine Liste von Nachrichten.

        Args:
            session_id: Session-ID
            message_list: Liste der zu persistierenden Nachrichten

        Returns:
            Anzahl erfolgreich gespeicherter Nachrichten
        """
        now = datetime.now(UTC)
        success_count = 0

        try:
            for i, msg in enumerate(message_list):
                role, content = CosmosChatMemory._normalize_message(msg)

                # Erstelle eindeutige ID mit Mikrosekunden für Sortierung
                timestamp = now.timestamp() + (i * MICROSECOND_OFFSET)
                item = CosmosChatMemory._create_message_item(session_id, role, content, timestamp)

                if await self._upsert_item(item, CHAT_MESSAGE_CATEGORY, "append_message"):
                    success_count += 1

        except Exception as exc:
            self._logger.warning(f"{CHAT_MEMORY_APPEND_ERROR}: {exc}")

        return success_count

    async def cleanup(self) -> None:
        """Führt Bereinigung alter/überschüssiger Daten durch.

        Implementiert die abstrakte Methode der Basis-Klasse.
        Bereinigt alle Sessions, nicht nur eine spezifische.
        """
        try:
            # Bereinige alle Sessions - in Produktionsumgebung könnte das
            # auf spezifische Sessions beschränkt werden
            await self._cleanup_old_messages()
            await self._cleanup_excess_messages()

        except Exception as exc:
            self._logger.debug(f"{CHAT_MEMORY_CLEANUP_ERROR}: {exc}")

    def _calculate_retention_cutoff(self) -> str:
        """Berechnet den Cutoff-Zeitpunkt für die Retention-Policy.

        Returns:
            ISO-formatierter Cutoff-Zeitstempel
        """
        cutoff = datetime.now(UTC) - timedelta(hours=self._retention.max_age_hours)
        return cutoff.isoformat()

    async def _find_old_message_ids(self, cutoff_iso: str) -> list[str]:
        """Findet IDs aller Nachrichten die älter als der Cutoff sind.

        Args:
            cutoff_iso: ISO-formatierter Cutoff-Zeitstempel

        Returns:
            Liste der zu löschenden Message-IDs
        """
        params = [
            {"name": "@cat", "value": CHAT_MESSAGE_CATEGORY},
            {"name": "@cut", "value": cutoff_iso},
        ]

        old_query = (
            "SELECT c.id FROM c WHERE c.category=@cat AND c.created_at < @cut"
        )

        old_items = await self._execute_query(
            old_query,
            params,
            "find_old_messages"
        )

        return [item["id"] for item in old_items]

    async def _cleanup_old_messages(self) -> None:
        """Löscht Nachrichten die älter als die Retention-Policy sind."""
        try:
            cutoff_iso = self._calculate_retention_cutoff()
            item_ids = await self._find_old_message_ids(cutoff_iso)

            if item_ids:
                deleted_count = await self._batch_delete_items(
                    item_ids,
                    CHAT_MESSAGE_CATEGORY,
                    "cleanup_old_messages"
                )
                self._logger.debug(f"Alte Nachrichten gelöscht: {deleted_count}")

        except CosmosOperationError as e:
            self._logger.warning(f"{CHAT_MEMORY_DELETE_ERROR}: {e}")

    async def _get_session_message_counts(self) -> list[dict[str, Any]]:
        """Ermittelt Nachrichten-Anzahl pro Session.

        Returns:
            Liste mit Session-IDs und Message-Counts
        """
        sessions_query = (
            "SELECT c.session_id, COUNT(1) as msg_count FROM c "
            "WHERE c.category=@cat GROUP BY c.session_id"
        )
        params = [{"name": "@cat", "value": CHAT_MESSAGE_CATEGORY}]

        return await self._execute_query(
            sessions_query,
            params,
            "get_session_counts"
        )

    async def _cleanup_excess_messages(self) -> None:
        """Löscht überschüssige Nachrichten pro Session."""
        try:
            session_counts = await self._get_session_message_counts()

            for session_info in session_counts:
                session_id = session_info.get("session_id")
                msg_count = int(session_info.get("msg_count", 0))

                if msg_count > self._retention.max_messages:
                    await self._cleanup_session_excess(session_id, msg_count)

        except CosmosOperationError as e:
            self._logger.warning(f"{CHAT_MEMORY_EXCESS_DELETE_ERROR}: {e}")

    async def _cleanup_session_excess(self, session_id: str, current_count: int) -> None:
        """Löscht überschüssige Nachrichten einer spezifischen Session.

        Args:
            session_id: Session-ID
            current_count: Aktuelle Anzahl Nachrichten
        """
        delete_count = current_count - self._retention.max_messages

        try:
            params = [
                {"name": "@cat", "value": CHAT_MESSAGE_CATEGORY},
                {"name": "@sid", "value": session_id},
            ]

            # Finde älteste Nachrichten zum Löschen
            excess_items = await self._execute_query(
                CHAT_MESSAGES_DELETE_EXCESS_QUERY,
                params,
                "cleanup_session_excess",
                max_results=delete_count
            )

            if excess_items:
                item_ids = [item["id"] for item in excess_items]
                deleted_count = await self._batch_delete_items(
                    item_ids,
                    CHAT_MESSAGE_CATEGORY,
                    "cleanup_session_excess"
                )
                self._logger.debug(
                    f"Überschüssige Nachrichten gelöscht (Session {session_id}): {deleted_count}"
                )

        except CosmosOperationError as e:
            self._logger.warning(f"{CHAT_MEMORY_EXCESS_DELETE_ERROR}: {e}")


__all__ = ["CosmosChatMemory", "MemoryRetention"]
