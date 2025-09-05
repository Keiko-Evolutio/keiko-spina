"""Gemeinsame Basis-Klasse für KEI-Bus Provider.

Konsolidiert duplizierte Logik zwischen NATS und Kafka Providern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .config import bus_settings
from .idempotency import is_duplicate, remember
from .metrics import BusMetrics, inject_trace
from .outbox import Inbox, Outbox
from .privacy import decrypt_fields, encrypt_fields
from .security import authorize_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .envelope import BusEnvelope

logger = get_logger(__name__)

# Konstanten für Provider
DEFAULT_OUTBOX_NAME = "default"
PUBLISH_OPERATION = "publish"
CONSUME_OPERATION = "consume"


class BaseProvider(ABC):
    """Abstrakte Basis-Klasse für Bus-Provider.

    Konsolidiert gemeinsame Funktionalitäten wie:
    - Outbox/Inbox-Handling
    - Idempotency-Checks
    - Security/Privacy-Verarbeitung
    - Metrics-Erfassung
    - Error-Handling
    """

    def __init__(self) -> None:
        """Initialisiert gemeinsame Provider-Komponenten."""
        self.metrics = BusMetrics()
        self._initialized = False

    @abstractmethod
    async def connect(self) -> None:
        """Stellt Verbindung zum Message Broker her."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Schließt Verbindung zum Message Broker."""
        raise NotImplementedError

    @abstractmethod
    async def _publish_message(self, envelope: BusEnvelope) -> None:
        """Provider-spezifische Publish-Implementierung."""
        raise NotImplementedError

    @abstractmethod
    async def _subscribe_to_subject(
        self,
        subject: str,
        queue: str | None,
        handler: Callable[[BusEnvelope], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        """Provider-spezifische Subscribe-Implementierung."""
        raise NotImplementedError

    async def publish(self, envelope: BusEnvelope) -> None:
        """Veröffentlicht Nachricht mit gemeinsamer Vor- und Nachverarbeitung.

        Args:
            envelope: Bus-Envelope mit Nachrichtendaten

        Raises:
            KeikoServiceError: Bei Provider-spezifischen Fehlern
        """
        # Outbox persist vor Publish
        await self._persist_to_outbox(envelope)

        # Idempotency-Check
        if await self._check_idempotency(envelope):
            logger.debug("Duplicate publish ignoriert: %s", envelope.id)
            return

        # Security-Autorisierung
        self._authorize_publish(envelope)

        # Privacy-Verschlüsselung
        encrypted_envelope = await self._encrypt_envelope(envelope)

        # Trace-Headers hinzufügen
        encrypted_envelope.headers = inject_trace(encrypted_envelope.headers)

        # Provider-spezifisches Publish
        await self._publish_message(encrypted_envelope)

        # Idempotency merken
        await self._remember_operation(PUBLISH_OPERATION, envelope.id)

        # Metrics erfassen
        self.metrics.mark_publish(envelope.subject, envelope.tenant)

        # Outbox cleanup
        await self._cleanup_outbox(envelope.id)

    async def subscribe(
        self,
        subject: str,
        queue: str | None,
        handler: Callable[[BusEnvelope], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        """Abonniert Subject mit gemeinsamer Nachrichtenverarbeitung.

        Args:
            subject: Subject/Topic zum Abonnieren
            queue: Queue-Name für Load Balancing
            handler: Handler-Funktion für Nachrichten
            **kwargs: Provider-spezifische Parameter
        """
        async def wrapped_handler(envelope: BusEnvelope) -> None:
            """Wrapper für Handler mit gemeinsamer Verarbeitung."""
            try:
                # Security-Autorisierung
                self._authorize_consume(envelope)

                # Idempotency-Check
                if await self._check_consume_idempotency(envelope):
                    self.metrics.mark_redelivery(envelope.subject, envelope.tenant)
                    return

                # Privacy-Entschlüsselung
                decrypted_envelope = await self._decrypt_envelope(envelope)

                # Inbox acknowledgment
                await self._acknowledge_inbox(envelope.id)

                # Handler ausführen
                await handler(decrypted_envelope)

                # Idempotency merken
                await self._remember_operation(CONSUME_OPERATION, envelope.id)

                # Metrics erfassen
                self.metrics.mark_consume(envelope.subject, envelope.tenant)

            except Exception as exc:
                logger.exception(f"Handler-Fehler für {envelope.id}: {exc}")
                self.metrics.mark_error(envelope.subject, envelope.tenant)
                raise

        # Provider-spezifisches Subscribe
        await self._subscribe_to_subject(subject, queue, wrapped_handler, **kwargs)

    async def _persist_to_outbox(self, envelope: BusEnvelope) -> None:
        """Persistiert Nachricht in Outbox für Reliability."""
        try:
            outbox = Outbox(DEFAULT_OUTBOX_NAME)
            await outbox.persist(envelope.id, envelope.model_dump())
        except Exception as exc:
            logger.debug(f"Outbox persist fehlgeschlagen: {exc}")

    async def _cleanup_outbox(self, message_id: str) -> None:
        """Entfernt Nachricht aus Outbox nach erfolgreichem Publish."""
        try:
            outbox = Outbox(DEFAULT_OUTBOX_NAME)
            await outbox.remove(message_id)
        except Exception as exc:
            logger.debug(f"Outbox cleanup fehlgeschlagen: {exc}")

    async def _acknowledge_inbox(self, message_id: str) -> None:
        """Markiert Nachricht als verarbeitet in Inbox."""
        try:
            inbox = Inbox(DEFAULT_OUTBOX_NAME)
            await inbox.ack(message_id)
        except Exception as exc:
            logger.debug(f"Inbox ack fehlgeschlagen: {exc}")

    async def _check_idempotency(self, envelope: BusEnvelope) -> bool:
        """Prüft Idempotency für Publish-Operation."""
        try:
            return await is_duplicate(PUBLISH_OPERATION, envelope.id)
        except Exception as exc:
            logger.debug(f"Idempotency check fehlgeschlagen: {exc}")
            return False

    async def _check_consume_idempotency(self, envelope: BusEnvelope) -> bool:
        """Prüft Idempotency für Consume-Operation."""
        try:
            return await is_duplicate(CONSUME_OPERATION, envelope.id)
        except Exception as exc:
            logger.debug(f"Consume idempotency check fehlgeschlagen: {exc}")
            return False

    async def _remember_operation(self, operation: str, message_id: str) -> None:
        """Merkt sich Operation für Idempotency."""
        try:
            await remember(operation, message_id)
        except Exception as exc:
            logger.debug(f"Idempotency remember fehlgeschlagen: {exc}")

    def _authorize_publish(self, envelope: BusEnvelope) -> None:
        """Autorisiert Publish-Operation."""
        try:
            authorize_message(envelope.headers, envelope.subject, action="publish")
        except PermissionError:
            logger.warning(f"Publish abgelehnt (ACL): {envelope.subject}")
            raise

    def _authorize_consume(self, envelope: BusEnvelope) -> None:
        """Autorisiert Consume-Operation."""
        try:
            authorize_message(envelope.headers, envelope.subject, action="consume")
        except PermissionError:
            logger.warning(f"Consume abgelehnt (ACL): {envelope.subject}")
            raise

    async def _encrypt_envelope(self, envelope: BusEnvelope) -> BusEnvelope:
        """Verschlüsselt Envelope-Payload falls konfiguriert."""
        try:
            if bus_settings.privacy.enable_field_encryption:
                # Verwende Privacy-Konfiguration für Feldverschlüsselung
                fields_to_encrypt = bus_settings.privacy.encrypted_fields or []
                key_id = bus_settings.privacy.encryption_key_id or "default"
                encrypted_payload = encrypt_fields(envelope.payload, fields_to_encrypt, key_id)
                # Erstelle neue Envelope-Instanz mit verschlüsseltem Payload
                return envelope.model_copy(update={"payload": encrypted_payload})
            return envelope
        except Exception as exc:
            logger.debug("Encryption fehlgeschlagen: %s", exc)
            return envelope

    async def _decrypt_envelope(self, envelope: BusEnvelope) -> BusEnvelope:
        """Entschlüsselt Envelope-Payload falls verschlüsselt."""
        try:
            # Verwende Privacy-Konfiguration für Feldentschlüsselung
            fields_to_decrypt = bus_settings.privacy.encrypted_fields or []
            key_id = bus_settings.privacy.encryption_key_id or "default"
            decrypted_payload = decrypt_fields(envelope.payload)
            # Erstelle neue Envelope-Instanz mit entschlüsseltem Payload
            return envelope.model_copy(update={"payload": decrypted_payload})
        except Exception as exc:
            logger.debug("Decryption fehlgeschlagen: %s", exc)
            return envelope


__all__ = [
    "CONSUME_OPERATION",
    "DEFAULT_OUTBOX_NAME",
    "PUBLISH_OPERATION",
    "BaseProvider",
]
