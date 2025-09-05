"""Kafka Provider mit Transaktionen und Idempotenz-Key (Exactly-Once Semantik).

Verwendet aiokafka Producer mit `enable_idempotence=True` und `transactional_id`.
Consumer verarbeitet Offsets in Transaktionen (consume-transform-produce) optional.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

from .config import bus_settings
from .envelope import BusEnvelope
from .idempotency import is_duplicate, remember
from .metrics import BusMetrics, inject_trace
from .outbox import Inbox, Outbox
from .privacy import decrypt_fields, encrypt_fields, get_active_kms_key_id, redact_payload
from .security import authorize_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

try:  # pragma: no cover - optional until installed
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except Exception:  # pragma: no cover
    # Fallback-Typen für bessere IDE-Unterstützung
    class AIOKafkaProducer:  # type: ignore
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def send(self, topic: str, value: bytes, **kwargs) -> Any: ...
        async def init_transactions(self) -> None: ...
        async def begin_transaction(self) -> None: ...
        async def commit_transaction(self) -> None: ...
        async def abort_transaction(self) -> None: ...

    class AIOKafkaConsumer:  # type: ignore
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        def __aiter__(self): ...
        async def __anext__(self): ...

    KAFKA_AVAILABLE = False


class KafkaProvider:
    """Kafka Provider mit TX/Idempotenz für KEI-Bus."""

    def __init__(self) -> None:
        self.producer: AIOKafkaProducer | None = None
        self.metrics = BusMetrics()
        self._loop = asyncio.get_event_loop()

    async def connect(self) -> None:
        """Initialisiert Kafka Producer (idempotent, transactional)."""
        if not KAFKA_AVAILABLE:
            from core.exceptions import KeikoServiceError
            raise KeikoServiceError("aiokafka nicht verfügbar")
        tx_id = f"{bus_settings.kafka_transactional_id_prefix}-{hash(asyncio.current_task()) & 0xFFFF:x}"
        self.producer = AIOKafkaProducer(
            loop=self._loop,
            bootstrap_servers=bus_settings.kafka_bootstrap_servers,
            enable_idempotence=True,
            transactional_id=tx_id,
            acks="all",
        )
        await self.producer.start()
        try:
            if self.producer is not None:
                await self.producer.init_transactions()
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Kafka-Transaktions-Initialisierung fehlgeschlagen - Verbindungsproblem: {e}")
            # Falls TX nicht unterstützt, weiter als idempotenter Producer
        except Exception as e:
            logger.debug(f"Kafka-Transaktions-Initialisierung fehlgeschlagen: {e}")
            # Falls TX nicht unterstützt, weiter als idempotenter Producer
        logger.info("✅ Kafka Producer initialisiert (idempotent, tx optional)")

    async def close(self) -> None:
        if self.producer:
            await self.producer.stop()
            self.producer = None

    async def publish(self, envelope: BusEnvelope) -> None:
        """Veröffentlicht Nachricht mit Idempotenz-Key und Trace-Headern."""
        if not self.producer:
            from core.exceptions import KeikoServiceError
            raise KeikoServiceError("Kafka Producer nicht initialisiert")

        # Initialize outbox to avoid unbound variable
        outbox = None

        # Outbox persist before publish
        try:
            outbox = Outbox("default")
            await outbox.persist(envelope.id, envelope.model_dump())
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"Outbox-Persistierung fehlgeschlagen - Verbindungsproblem: {e}")
        except Exception as e:
            logger.warning(f"Outbox-Persistierung fehlgeschlagen: {e}")

        # Idempotenz auf Applikationsebene
        if await is_duplicate("kafka_publish", envelope.id):
            logger.debug("Duplicate kafka publish ignoriert")
            return

        # Redaction & Encryption
        if bus_settings.redact_payload_before_send:
            envelope.payload = redact_payload(envelope.payload)
        if bus_settings.enable_field_encryption and bus_settings.encryption_fields:
            envelope.payload = encrypt_fields(envelope.payload, bus_settings.encryption_fields, get_active_kms_key_id())

        headers: list[tuple[str, bytes]] = []
        # Trace/ACL
        kheaders = inject_trace(envelope.headers)
        authorize_message(kheaders, envelope.subject, action="publish")
        for k, v in kheaders.items():
            try:
                headers.append((k, str(v).encode("utf-8")))
            except Exception:
                continue

        # Message Key (Ordering/Partition)
        key_bytes = (envelope.key or "").encode("utf-8")
        # Idempotenz-Key/Messaging-Key
        msg_id_source = f"{envelope.subject}:{envelope.key or ''}:{envelope.id}"
        msg_id = hashlib.sha1(msg_id_source.encode("utf-8")).hexdigest()
        headers.append(("message_id", envelope.id.encode("utf-8")))
        headers.append(("dedup_id", msg_id.encode("utf-8")))
        if envelope.corr_id:
            headers.append(("correlation_id", envelope.corr_id.encode("utf-8")))
        if envelope.causation_id:
            headers.append(("causation_id", envelope.causation_id.encode("utf-8")))
        if envelope.tenant:
            headers.append(("tenant", envelope.tenant.encode("utf-8")))

        value = json.dumps(envelope.model_dump()).encode("utf-8")

        # Transaktion optional
        try:
            await self.producer.begin_transaction()
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"Kafka-Transaktion konnte nicht gestartet werden - Verbindungsproblem: {e}")
            # TX nicht verfügbar
        except Exception as e:
            logger.debug(f"Kafka-Transaktion konnte nicht gestartet werden: {e}")
            # TX nicht verfügbar

        try:
            # AIOKafkaProducer.send() returns a Future that we need to await
            future = await self.producer.send(
                topic=envelope.subject,
                key=key_bytes if key_bytes else None,
                value=value,
                headers=headers,
            )
            # Wait for the message to be sent
            await future
            with contextlib.suppress(ConnectionError, TimeoutError, AttributeError):
                await self.producer.commit_transaction()
            self.metrics.mark_publish(envelope.subject, envelope.tenant)
            await remember("kafka_publish", envelope.id)
            with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                if outbox is not None:
                    await outbox.remove(envelope.id)
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Kafka-Publish fehlgeschlagen - Verbindungsproblem: {e}")
            with contextlib.suppress(ConnectionError, TimeoutError, AttributeError):
                await self.producer.abort_transaction()
            raise
        except Exception as e:
            logger.error(f"Kafka-Publish fehlgeschlagen - Unerwarteter Fehler: {e}")
            with contextlib.suppress(ConnectionError, TimeoutError, AttributeError):
                await self.producer.abort_transaction()
            raise

    async def subscribe(
        self,
        topic: str,
        group_suffix: str,
        handler: Callable[[BusEnvelope], Awaitable[None]],
    ) -> None:
        """Einfacher Consumer (ohne TX-Verbund für Offsets), mit Entschlüsselung/ACL."""
        if not KAFKA_AVAILABLE:
            from core.exceptions import KeikoServiceError
            raise KeikoServiceError("aiokafka nicht verfügbar")
        group_id = f"{bus_settings.kafka_group_id_prefix}-{group_suffix}"
        consumer = AIOKafkaConsumer(
            topic,
            loop=self._loop,
            bootstrap_servers=bus_settings.kafka_bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await consumer.start()

        async def _loop() -> None:
            try:
                async for msg in consumer:
                    try:
                        # Verarbeite Message-Headers
                        headers = {}
                        for k, v in (msg.headers or []):
                            if isinstance(v, (bytes, bytearray)):
                                headers[k] = v.decode("utf-8")
                            else:
                                headers[k] = str(v)
                        payload = json.loads(msg.value.decode("utf-8"))
                        env = BusEnvelope(**payload)
                        try:
                            authorize_message(headers, env.subject, action="consume")
                        except PermissionError:
                            await consumer.commit()
                            continue
                        with contextlib.suppress(ValueError, TypeError, KeyError):
                            env.payload = decrypt_fields(env.payload)
                        try:
                            inbox = Inbox("default")
                            await inbox.ack(env.id)
                        except (ConnectionError, TimeoutError) as e:
                            logger.debug(f"Inbox-ACK fehlgeschlagen - Verbindungsproblem: {e}")
                        except Exception as e:
                            logger.warning(f"Inbox-ACK fehlgeschlagen: {e}")
                        await handler(env)
                        self.metrics.mark_consume(env.subject, env.tenant)
                        await consumer.commit()
                    except (ValueError, TypeError, KeyError) as e:
                        logger.error(f"Kafka-Message-Handler fehlgeschlagen - Daten-/Validierungsfehler: {e}")
                        # Bei Fehlern wird auf DLQ-Topic gepublished
                        await self._send_to_dlq(topic, msg.value)
                        await consumer.commit()
                    except Exception as e:
                        logger.exception(f"Kafka-Message-Handler fehlgeschlagen - Unerwarteter Fehler: {e}")
                        # Bei Fehlern wird auf DLQ-Topic gepublished
                        await self._send_to_dlq(topic, msg.value)
                        await consumer.commit()
            finally:
                await consumer.stop()

        asyncio.create_task(_loop())

    async def _send_to_dlq(self, original_topic: str, message_value: bytes) -> None:
        """Sendet fehlgeschlagene Nachricht an Dead Letter Queue.

        Args:
            original_topic: Ursprüngliches Topic
            message_value: Nachrichtenwert
        """
        if not self.producer:
            logger.warning("Kann nicht an DLQ senden - Producer nicht verfügbar")
            return

        dlq_topic = f"kei.dlq.{original_topic}"
        try:
            with contextlib.suppress(Exception):
                # AIOKafkaProducer.send() returns a Future that we need to await
                future = await self.producer.send(dlq_topic, value=message_value)
                await future
                logger.debug(f"Nachricht an DLQ gesendet: {dlq_topic}")
        except Exception as e:
            logger.error(f"DLQ-Send fehlgeschlagen: {e}")
