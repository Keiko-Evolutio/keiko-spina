"""Core-Operations für Voice Session Management.

Zentrale Operationen für Update-Sending und Thread-Forwarding
mit verbesserter Error-Handling und Type-Safety.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from kei_logging import get_logger

from .content_utils import should_send_content
from .session_constants import (
    IMPORT_ERROR_MSG,
    MESSAGE_FORWARDED_MSG,
    THREAD_FORWARDING_ERROR_MSG,
    UPDATE_SENDING_ERROR_MSG,
)
from .session_exceptions import (
    SessionImportError,
    ThreadForwardingError,
    UpdateSendingError,
)

if TYPE_CHECKING:
    from data_models import Update

logger = get_logger(__name__)


class ClientProtocol(Protocol):
    """Protocol für Client-Interface."""

    async def send_update(self, update: Update) -> None:
        """Sendet Update an Client."""
        ...


class UpdateSender:
    """Service für sicheres Senden von Updates an Client.

    Extrahierte und verbesserte Version der ursprünglichen _send_safe_update() Methode
    mit besserer Error-Handling und Type-Safety.
    """

    def __init__(self, client: ClientProtocol | None = None) -> None:
        """Initialisiert Update-Sender.

        Args:
            client: Client für Update-Sending
        """
        self.client = client

    async def send_update(self, update: Update) -> bool:
        """Sendet Update sicher an Client.

        Args:
            update: Zu sendendes Update

        Returns:
            True wenn erfolgreich gesendet, False sonst

        Raises:
            UpdateSendingError: Bei kritischen Sending-Fehlern
        """
        if not self.client:
            logger.debug("Kein Client verfügbar für Update-Sending")
            return False

        if not update:
            logger.warning("Update ist None oder leer")
            return False

        try:
            await self.client.send_update(update)
            logger.debug("Update erfolgreich gesendet: %s", update.type.value if hasattr(update.type, "value") else update.type)
            return True

        except Exception as e:
            logger.exception("%s: %s", UPDATE_SENDING_ERROR_MSG, e)

            # Für kritische Fehler Exception werfen
            if self._is_critical_error(e):
                raise UpdateSendingError(
                    message=UPDATE_SENDING_ERROR_MSG,
                    update_type=str(update.type) if update.type else None,
                    update_id=getattr(update, "update_id", None),
                    cause=e
                ) from e

            return False

    async def send_update_safe(self, update: Update) -> bool:
        """Sendet Update mit vollständiger Error-Suppression.

        Args:
            update: Zu sendendes Update

        Returns:
            True wenn erfolgreich gesendet, False sonst
        """
        try:
            return await self.send_update(update)
        except Exception as e:
            logger.exception("Fehler beim sicheren Update-Sending: %s", e)
            return False

    def set_client(self, client: ClientProtocol | None) -> None:
        """Setzt neuen Client.

        Args:
            client: Neuer Client
        """
        self.client = client

    def has_client(self) -> bool:
        """Prüft ob Client verfügbar ist."""
        return self.client is not None

    def _is_critical_error(self, error: Exception) -> bool:
        """Prüft ob Fehler kritisch ist.

        Args:
            error: Zu prüfender Fehler

        Returns:
            True wenn kritisch, False sonst
        """
        # Connection-Errors sind meist nicht kritisch
        critical_error_types = (
            TypeError,
            AttributeError,
            ValueError,
        )

        return isinstance(error, critical_error_types)


class ThreadForwarder:
    """Service für Thread-Forwarding von Nachrichten.

    Extrahierte und verbesserte Version der ursprünglichen _forward_to_thread() Methode
    mit besserer Error-Handling und Type-Safety.
    """

    def __init__(self, thread_id: str | None = None) -> None:
        """Initialisiert Thread-Forwarder.

        Args:
            thread_id: Thread-ID für Forwarding
        """
        self.thread_id = thread_id
        self._create_thread_message = None
        self._import_attempted = False

    async def forward_to_thread(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """Leitet Nachricht an Thread weiter.

        Args:
            role: Role der Nachricht
            content: Content der Nachricht
            metadata: Optionale Metadaten

        Returns:
            Message-ID wenn erfolgreich, None sonst

        Raises:
            ThreadForwardingError: Bei kritischen Forwarding-Fehlern
        """
        if not self.thread_id:
            logger.debug("Keine Thread-ID für Forwarding verfügbar")
            return None

        if not should_send_content(content):
            logger.debug("Content nicht gültig für Thread-Forwarding")
            return None

        try:
            # Lazy Import der create_thread_message Funktion
            create_func = await self._get_create_thread_message()
            if not create_func:
                return None

            message_id = await create_func(
                thread_id=self.thread_id,
                role=role,
                content=content,
                metadata=metadata,
            )

            if message_id:
                logger.debug(MESSAGE_FORWARDED_MSG, message_id)
                return message_id
            logger.warning("Thread-Forwarding lieferte keine Message-ID")
            return None

        except ImportError as e:
            logger.exception("%s: %s", IMPORT_ERROR_MSG, e)
            raise SessionImportError(
                message=IMPORT_ERROR_MSG,
                module_name="agents",
                function_name="create_thread_message",
                cause=e
            ) from e

        except Exception as e:
            logger.exception("%s: %s", THREAD_FORWARDING_ERROR_MSG, e)

            # Für kritische Fehler Exception werfen
            if self._is_critical_forwarding_error(e):
                raise ThreadForwardingError(
                    message=THREAD_FORWARDING_ERROR_MSG,
                    thread_id=self.thread_id,
                    role=role,
                    cause=e
                ) from e

            return None

    async def forward_to_thread_safe(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """Leitet Nachricht mit vollständiger Error-Suppression weiter.

        Args:
            role: Role der Nachricht
            content: Content der Nachricht
            metadata: Optionale Metadaten

        Returns:
            Message-ID wenn erfolgreich, None sonst
        """
        try:
            return await self.forward_to_thread(role, content, metadata)
        except Exception as e:
            logger.exception("Fehler beim sicheren Thread-Forwarding: %s", e)
            return None

    def set_thread_id(self, thread_id: str | None) -> None:
        """Setzt neue Thread-ID.

        Args:
            thread_id: Neue Thread-ID
        """
        self.thread_id = thread_id

    def has_thread_id(self) -> bool:
        """Prüft ob Thread-ID verfügbar ist."""
        return bool(self.thread_id)

    async def _get_create_thread_message(self):
        """Lazy Import der create_thread_message Funktion.

        Returns:
            create_thread_message Funktion oder None
        """
        if self._create_thread_message is not None:
            return self._create_thread_message

        if self._import_attempted:
            return None

        try:
            from agents import create_thread_message
            self._create_thread_message = create_thread_message
            self._import_attempted = True
            return create_thread_message

        except ImportError as e:
            logger.exception("%s: %s", IMPORT_ERROR_MSG, e)
            self._import_attempted = True
            return None

    def _is_critical_forwarding_error(self, error: Exception) -> bool:
        """Prüft ob Forwarding-Fehler kritisch ist.

        Args:
            error: Zu prüfender Fehler

        Returns:
            True wenn kritisch, False sonst
        """
        # Network/Connection-Errors sind meist nicht kritisch
        critical_error_types = (
            TypeError,
            AttributeError,
            ValueError,
        )

        return isinstance(error, critical_error_types)


# =============================================================================
# Combined Operations Service
# =============================================================================

class SessionOperations:
    """Kombinierter Service für alle Core-Operations.

    Vereint UpdateSender und ThreadForwarder für einfachere Verwendung.
    """

    def __init__(
        self,
        client: ClientProtocol | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Initialisiert Session-Operations.

        Args:
            client: Client für Update-Sending
            thread_id: Thread-ID für Forwarding
        """
        self.update_sender = UpdateSender(client)
        self.thread_forwarder = ThreadForwarder(thread_id)

    async def send_update(self, update: Update) -> bool:
        """Sendet Update an Client."""
        return await self.update_sender.send_update(update)

    async def forward_to_thread(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """Leitet Nachricht an Thread weiter."""
        return await self.thread_forwarder.forward_to_thread(role, content, metadata)

    def set_client(self, client: ClientProtocol | None) -> None:
        """Setzt neuen Client."""
        self.update_sender.set_client(client)

    def set_thread_id(self, thread_id: str | None) -> None:
        """Setzt neue Thread-ID."""
        self.thread_forwarder.set_thread_id(thread_id)

    def has_client(self) -> bool:
        """Prüft ob Client verfügbar ist."""
        return self.update_sender.has_client()

    def has_thread_id(self) -> bool:
        """Prüft ob Thread-ID verfügbar ist."""
        return self.thread_forwarder.has_thread_id()
