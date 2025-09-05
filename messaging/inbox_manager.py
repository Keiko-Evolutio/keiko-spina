#!/usr/bin/env python3
"""Platform Inbox Manager für Issue #56 Messaging-first Architecture
Implementiert Inbox Pattern für Platform-interne Idempotenz und Duplikat-Erkennung

ARCHITEKTUR-COMPLIANCE:
- Nur für Platform-interne Message-Verarbeitung
- Keine SDK-Dependencies oder -Exports
- Gewährleistet Exactly-Once Processing für Platform Events
"""

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import asyncpg

from kei_logging import get_logger

from .platform_event_bus import PlatformEvent

logger = get_logger(__name__)

class InboxMessageStatus(Enum):
    """Status von Inbox Messages"""
    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DUPLICATE = "duplicate"

@dataclass
class InboxMessage:
    """Inbox Message für Idempotenz-Garantien"""
    id: str
    message_hash: str
    event_type: str
    event_data: dict[str, Any]
    status: InboxMessageStatus
    received_at: datetime
    processed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: str | None = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = InboxMessageStatus(self.status)

@dataclass
class InboxConfig:
    """Konfiguration für Inbox Manager"""
    database_url: str
    table_name: str = "platform_inbox_messages"
    batch_size: int = 100
    retry_delay_seconds: int = 30
    max_retries: int = 3
    cleanup_after_days: int = 30
    processing_interval_seconds: int = 5
    enable_deduplication: bool = True

class PlatformInboxManager:
    """Platform Inbox Manager für idempotente Message-Verarbeitung"""

    def __init__(self, config: InboxConfig):
        self.config = config
        self.db_pool: asyncpg.Pool | None = None
        self.processing_task: asyncio.Task | None = None
        self.running = False

        # Message Handlers
        self.handlers: dict[str, Callable[[PlatformEvent], Any]] = {}

        # Metriken
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_failed = 0
        self.duplicates_detected = 0

    async def start(self) -> bool:
        """Startet den Inbox Manager"""
        try:
            logger.info("Starte Platform Inbox Manager...")

            # Database Pool erstellen - asyncpg.create_pool ist async
            self.db_pool: asyncpg.Pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=2,
                max_size=10
            )

            # Inbox Tabelle erstellen
            await self._create_inbox_table()

            # Background Processing starten
            self.running = True
            self.processing_task = asyncio.create_task(self._process_inbox_messages())

            logger.info("Platform Inbox Manager erfolgreich gestartet")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Starten des Platform Inbox Managers: {e}")
            return False

    async def stop(self):
        """Stoppt den Inbox Manager"""
        try:
            logger.info("Stoppe Platform Inbox Manager...")

            self.running = False

            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            if self.db_pool:
                await self.db_pool.close()

            logger.info("Platform Inbox Manager gestoppt")

        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Platform Inbox Managers: {e}")

    async def receive_message(self, event: PlatformEvent) -> bool:
        """Empfängt und speichert Platform Event für idempotente Verarbeitung"""
        try:
            # Message Hash für Duplikat-Erkennung erstellen
            message_hash = self._create_message_hash(event)

            # Prüfe auf Duplikat
            if self.config.enable_deduplication:
                if await self._is_duplicate(message_hash):
                    self.duplicates_detected += 1
                    logger.debug(f"Platform Event Duplikat erkannt: {event.event_type}")
                    return True  # Duplikat ist kein Fehler

            # Inbox Message erstellen
            inbox_message = InboxMessage(
                id=event.event_id,
                message_hash=message_hash,
                event_type=event.event_type,
                event_data=asdict(event),
                status=InboxMessageStatus.RECEIVED,
                received_at=datetime.now(UTC)
            )

            # Message in Inbox speichern
            async with self.db_pool.acquire() as conn:
                await self._insert_inbox_message(conn, inbox_message)

            self.messages_received += 1
            logger.debug(f"Platform Event in Inbox empfangen: {event.event_type}")

            return True

        except Exception as e:
            logger.error(f"Fehler beim Empfangen des Platform Events in Inbox: {e}")
            return False

    def register_handler(self, event_type: str, handler: Callable[[PlatformEvent], Any]):
        """Registriert Handler für Event-Typ"""
        self.handlers[event_type] = handler
        logger.info(f"Platform Event Handler registriert: {event_type}")

    async def _process_inbox_messages(self):
        """Background Task für Inbox Message Processing"""
        while self.running:
            try:
                await self._process_received_messages()
                await self._retry_failed_messages()
                await self._cleanup_old_messages()

                await asyncio.sleep(self.config.processing_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten der Platform Inbox Messages: {e}")
                await asyncio.sleep(self.config.processing_interval_seconds)

    async def _process_received_messages(self):
        """Verarbeitet empfangene Inbox Messages"""
        try:
            async with self.db_pool.acquire() as conn:
                # Hole received Messages
                rows = await conn.fetch(
                    f"""
                    SELECT id, message_hash, event_type, event_data, status, received_at, retry_count, max_retries
                    FROM {self.config.table_name}
                    WHERE status = $1
                    ORDER BY received_at
                    LIMIT $2
                    """,
                    InboxMessageStatus.RECEIVED.value,
                    self.config.batch_size
                )

                for row in rows:
                    await self._process_inbox_message(conn, row)

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der received Platform Inbox Messages: {e}")

    async def _retry_failed_messages(self):
        """Verarbeitet failed Messages für Retry"""
        try:
            async with self.db_pool.acquire() as conn:
                # Hole failed Messages die für Retry bereit sind
                cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.retry_delay_seconds)

                rows = await conn.fetch(
                    f"""
                    SELECT id, message_hash, event_type, event_data, status, received_at, retry_count, max_retries
                    FROM {self.config.table_name}
                    WHERE status = $1 AND retry_count < max_retries AND received_at <= $2
                    ORDER BY received_at
                    LIMIT $3
                    """,
                    InboxMessageStatus.FAILED.value,
                    cutoff_time,
                    self.config.batch_size
                )

                for row in rows:
                    await self._process_inbox_message(conn, row)

        except Exception as e:
            logger.error(f"Fehler beim Retry der failed Platform Inbox Messages: {e}")

    async def _process_inbox_message(self, conn: asyncpg.Connection, row):
        """Verarbeitet einzelne Inbox Message"""
        try:
            # Message als processing markieren
            await conn.execute(
                f"""
                UPDATE {self.config.table_name}
                SET status = $1
                WHERE id = $2
                """,
                InboxMessageStatus.PROCESSING.value,
                row["id"]
            )

            # Event aus Row erstellen
            event_data = json.loads(row["event_data"]) if isinstance(row["event_data"], str) else row["event_data"]
            event = PlatformEvent(**event_data)

            # Handler ausführen
            handler = self.handlers.get(event.event_type)
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)

                    # Message als processed markieren
                    await conn.execute(
                        f"""
                        UPDATE {self.config.table_name}
                        SET status = $1, processed_at = $2
                        WHERE id = $3
                        """,
                        InboxMessageStatus.PROCESSED.value,
                        datetime.now(UTC),
                        row["id"]
                    )

                    self.messages_processed += 1
                    logger.debug(f"Platform Inbox Message verarbeitet: {event.event_type}")

                except Exception as e:
                    # Handler-Fehler
                    retry_count = row["retry_count"] + 1

                    await conn.execute(
                        f"""
                        UPDATE {self.config.table_name}
                        SET status = $1, retry_count = $2, error_message = $3
                        WHERE id = $4
                        """,
                        InboxMessageStatus.FAILED.value,
                        retry_count,
                        str(e)[:1000],  # Begrenzte Fehlermeldung
                        row["id"]
                    )

                    self.messages_failed += 1
                    logger.error(f"Fehler beim Verarbeiten der Platform Inbox Message: {e}")

            else:
                # Kein Handler gefunden
                logger.warning(f"Kein Handler für Platform Event-Typ: {event.event_type}")

                await conn.execute(
                    f"""
                    UPDATE {self.config.table_name}
                    SET status = $1, error_message = $2
                    WHERE id = $3
                    """,
                    InboxMessageStatus.FAILED.value,
                    f"No handler for event type: {event.event_type}",
                    row["id"]
                )

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der Platform Inbox Message: {e}")

            # Message zurück auf received setzen für späteren Retry
            try:
                await conn.execute(
                    f"""
                    UPDATE {self.config.table_name}
                    SET status = $1
                    WHERE id = $2
                    """,
                    InboxMessageStatus.RECEIVED.value,
                    row["id"]
                )
            except:
                pass

    async def _is_duplicate(self, message_hash: str) -> bool:
        """Prüft ob Message ein Duplikat ist"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    f"""
                    SELECT COUNT(*) FROM {self.config.table_name}
                    WHERE message_hash = $1
                    """,
                    message_hash
                )

                return result > 0

        except Exception as e:
            logger.error(f"Fehler bei Duplikat-Prüfung: {e}")
            return False

    def _create_message_hash(self, event: PlatformEvent) -> str:
        """Erstellt Hash für Message-Duplikat-Erkennung"""
        # Hash basierend auf Event-ID und Event-Typ
        hash_input = f"{event.event_id}:{event.event_type}:{event.correlation_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    async def _insert_inbox_message(self, conn: asyncpg.Connection, inbox_message: InboxMessage):
        """Fügt Inbox Message in Datenbank ein"""
        await conn.execute(
            f"""
            INSERT INTO {self.config.table_name}
            (id, message_hash, event_type, event_data, status, received_at, retry_count, max_retries)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            inbox_message.id,
            inbox_message.message_hash,
            inbox_message.event_type,
            json.dumps(inbox_message.event_data),
            inbox_message.status.value,
            inbox_message.received_at,
            inbox_message.retry_count,
            inbox_message.max_retries
        )

    async def _cleanup_old_messages(self):
        """Bereinigt alte processed Messages"""
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=self.config.cleanup_after_days)

            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    DELETE FROM {self.config.table_name}
                    WHERE status = $1 AND processed_at < $2
                    """,
                    InboxMessageStatus.PROCESSED.value,
                    cutoff_date
                )

                if result != "DELETE 0":
                    logger.debug(f"Platform Inbox Messages bereinigt: {result}")

        except Exception as e:
            logger.error(f"Fehler beim Bereinigen der Platform Inbox Messages: {e}")

    async def _create_inbox_table(self):
        """Erstellt Inbox Tabelle falls nicht vorhanden"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        message_hash VARCHAR(64) UNIQUE NOT NULL,
                        event_type VARCHAR(255) NOT NULL,
                        event_data JSONB NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        received_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        processed_at TIMESTAMP WITH TIME ZONE,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        error_message TEXT
                    )
                """)

                # Indizes erstellen
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_status
                    ON {self.config.table_name} (status)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_hash
                    ON {self.config.table_name} (message_hash)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_received
                    ON {self.config.table_name} (received_at)
                """)

                logger.debug(f"Platform Inbox Tabelle {self.config.table_name} erstellt/verifiziert")

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Platform Inbox Tabelle: {e}")
            raise

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Inbox Manager Metriken zurück"""
        return {
            "running": self.running,
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "duplicates_detected": self.duplicates_detected,
            "registered_handlers": len(self.handlers),
            "processing_rate": self.messages_processed / max(self.messages_received, 1) * 100
        }
