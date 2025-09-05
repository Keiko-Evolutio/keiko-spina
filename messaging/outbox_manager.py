#!/usr/bin/env python3
"""Platform Outbox Manager für Issue #56 Messaging-first Architecture
Implementiert Outbox Pattern für Platform-interne Transaktionsgarantien

ARCHITEKTUR-COMPLIANCE:
- Nur für Platform-interne Transaktionen
- Keine SDK-Dependencies oder -Exports
- Gewährleistet Eventual Consistency für Platform Events
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import asyncpg

from kei_logging import get_logger

from .platform_event_bus import PlatformEvent

logger = get_logger(__name__)

class OutboxEventStatus(Enum):
    """Status von Outbox Events"""
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
    RETRY = "retry"

@dataclass
class OutboxEvent:
    """Outbox Event für transaktionale Garantien"""
    id: str
    event_type: str
    event_data: dict[str, Any]
    status: OutboxEventStatus
    created_at: datetime
    published_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: datetime | None = None
    error_message: str | None = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = OutboxEventStatus(self.status)

@dataclass
class OutboxConfig:
    """Konfiguration für Outbox Manager"""
    database_url: str
    table_name: str = "platform_outbox_events"
    batch_size: int = 100
    retry_delay_seconds: int = 60
    max_retries: int = 3
    cleanup_after_days: int = 7
    processing_interval_seconds: int = 10

class PlatformOutboxManager:
    """Platform Outbox Manager für transaktionale Event-Publikation"""

    def __init__(self, config: OutboxConfig):
        self.config = config
        self.db_pool: asyncpg.Pool | None = None
        self.processing_task: asyncio.Task | None = None
        self.running = False

        # Metriken
        self.events_stored = 0
        self.events_published = 0
        self.events_failed = 0
        self.retry_attempts = 0

    async def start(self) -> bool:
        """Startet den Outbox Manager"""
        try:
            logger.info("Starte Platform Outbox Manager...")

            # Database Pool erstellen - asyncpg.create_pool ist async
            self.db_pool: asyncpg.Pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=2,
                max_size=10
            )

            # Outbox Tabelle erstellen
            await self._create_outbox_table()

            # Background Processing starten
            self.running = True
            self.processing_task = asyncio.create_task(self._process_outbox_events())

            logger.info("Platform Outbox Manager erfolgreich gestartet")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Starten des Platform Outbox Managers: {e}")
            return False

    async def stop(self):
        """Stoppt den Outbox Manager"""
        try:
            logger.info("Stoppe Platform Outbox Manager...")

            self.running = False

            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            if self.db_pool:
                await self.db_pool.close()

            logger.info("Platform Outbox Manager gestoppt")

        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Platform Outbox Managers: {e}")

    async def store_event(self, event: PlatformEvent, transaction_conn: asyncpg.Connection | None = None) -> str:
        """Speichert Event in Outbox für transaktionale Publikation"""
        try:
            outbox_event = OutboxEvent(
                id=str(uuid.uuid4()),
                event_type=event.event_type,
                event_data=asdict(event),
                status=OutboxEventStatus.PENDING,
                created_at=datetime.now(UTC)
            )

            # Verwende bereitgestellte Transaktion oder erstelle neue
            if transaction_conn:
                await self._insert_outbox_event(transaction_conn, outbox_event)
            else:
                async with self.db_pool.acquire() as conn:
                    await self._insert_outbox_event(conn, outbox_event)

            self.events_stored += 1
            logger.debug(f"Platform Event in Outbox gespeichert: {event.event_type}")

            return outbox_event.id

        except Exception as e:
            logger.error(f"Fehler beim Speichern des Platform Events in Outbox: {e}")
            raise

    async def _insert_outbox_event(self, conn: asyncpg.Connection, outbox_event: OutboxEvent):
        """Fügt Outbox Event in Datenbank ein"""
        await conn.execute(
            f"""
            INSERT INTO {self.config.table_name}
            (id, event_type, event_data, status, created_at, retry_count, max_retries)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            outbox_event.id,
            outbox_event.event_type,
            json.dumps(outbox_event.event_data),
            outbox_event.status.value,
            outbox_event.created_at,
            outbox_event.retry_count,
            outbox_event.max_retries
        )

    async def _process_outbox_events(self):
        """Background Task für Outbox Event Processing"""
        while self.running:
            try:
                await self._process_pending_events()
                await self._process_retry_events()
                await self._cleanup_old_events()

                await asyncio.sleep(self.config.processing_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten der Platform Outbox Events: {e}")
                await asyncio.sleep(self.config.processing_interval_seconds)

    async def _process_pending_events(self):
        """Verarbeitet pending Outbox Events"""
        try:
            async with self.db_pool.acquire() as conn:
                # Hole pending Events
                rows = await conn.fetch(
                    f"""
                    SELECT id, event_type, event_data, status, created_at, retry_count, max_retries
                    FROM {self.config.table_name}
                    WHERE status = $1
                    ORDER BY created_at
                    LIMIT $2
                    """,
                    OutboxEventStatus.PENDING.value,
                    self.config.batch_size
                )

                for row in rows:
                    await self._publish_outbox_event(conn, row)

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der pending Platform Outbox Events: {e}")

    async def _process_retry_events(self):
        """Verarbeitet Retry Outbox Events"""
        try:
            async with self.db_pool.acquire() as conn:
                # Hole Events die für Retry bereit sind
                now = datetime.now(UTC)
                rows = await conn.fetch(
                    f"""
                    SELECT id, event_type, event_data, status, created_at, retry_count, max_retries, next_retry_at
                    FROM {self.config.table_name}
                    WHERE status = $1 AND next_retry_at <= $2
                    ORDER BY next_retry_at
                    LIMIT $3
                    """,
                    OutboxEventStatus.RETRY.value,
                    now,
                    self.config.batch_size
                )

                for row in rows:
                    await self._publish_outbox_event(conn, row)

        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der Retry Platform Outbox Events: {e}")

    async def _publish_outbox_event(self, conn: asyncpg.Connection, row):
        """Publiziert einzelnes Outbox Event"""
        try:
            # Event aus Row erstellen
            event_data = json.loads(row["event_data"])
            event = PlatformEvent(**event_data)

            # Event publizieren (hier würde der Event Bus verwendet)
            # Für jetzt simulieren wir erfolgreiche Publikation
            success = await self._simulate_event_publication(event)

            if success:
                # Event als published markieren
                await conn.execute(
                    f"""
                    UPDATE {self.config.table_name}
                    SET status = $1, published_at = $2
                    WHERE id = $3
                    """,
                    OutboxEventStatus.PUBLISHED.value,
                    datetime.now(UTC),
                    row["id"]
                )

                self.events_published += 1
                logger.debug(f"Platform Outbox Event publiziert: {row['event_type']}")

            else:
                # Retry oder Failed markieren
                retry_count = row["retry_count"] + 1
                max_retries = row["max_retries"]

                if retry_count <= max_retries:
                    # Für Retry markieren
                    next_retry_at = datetime.now(UTC) + timedelta(
                        seconds=self.config.retry_delay_seconds * (2 ** retry_count)  # Exponential backoff
                    )

                    await conn.execute(
                        f"""
                        UPDATE {self.config.table_name}
                        SET status = $1, retry_count = $2, next_retry_at = $3
                        WHERE id = $4
                        """,
                        OutboxEventStatus.RETRY.value,
                        retry_count,
                        next_retry_at,
                        row["id"]
                    )

                    self.retry_attempts += 1
                    logger.warning(f"Platform Outbox Event für Retry markiert: {row['event_type']} (Versuch {retry_count})")

                else:
                    # Als failed markieren
                    await conn.execute(
                        f"""
                        UPDATE {self.config.table_name}
                        SET status = $1, error_message = $2
                        WHERE id = $3
                        """,
                        OutboxEventStatus.FAILED.value,
                        "Max retries exceeded",
                        row["id"]
                    )

                    self.events_failed += 1
                    logger.error(f"Platform Outbox Event endgültig fehlgeschlagen: {row['event_type']}")

        except Exception as e:
            logger.error(f"Fehler beim Publizieren des Platform Outbox Events: {e}")

    async def _simulate_event_publication(self, event: PlatformEvent) -> bool:
        """Simuliert Event-Publikation (wird durch echten Event Bus ersetzt)"""
        try:
            # Hier würde der echte Event Bus verwendet:
            # return await self.event_bus.publish(event)

            # Für jetzt simulieren wir 95% Erfolgsrate
            import random
            return random.random() < 0.95

        except Exception as e:
            logger.error(f"Fehler bei simulierter Platform Event Publikation: {e}")
            return False

    async def _cleanup_old_events(self):
        """Bereinigt alte published Events"""
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=self.config.cleanup_after_days)

            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    DELETE FROM {self.config.table_name}
                    WHERE status = $1 AND published_at < $2
                    """,
                    OutboxEventStatus.PUBLISHED.value,
                    cutoff_date
                )

                if result != "DELETE 0":
                    logger.debug(f"Platform Outbox Events bereinigt: {result}")

        except Exception as e:
            logger.error(f"Fehler beim Bereinigen der Platform Outbox Events: {e}")

    async def _create_outbox_table(self):
        """Erstellt Outbox Tabelle falls nicht vorhanden"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                        id UUID PRIMARY KEY,
                        event_type VARCHAR(255) NOT NULL,
                        event_data JSONB NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        published_at TIMESTAMP WITH TIME ZONE,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        next_retry_at TIMESTAMP WITH TIME ZONE,
                        error_message TEXT
                    )
                """)

                # Indizes erstellen
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_status
                    ON {self.config.table_name} (status)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_retry
                    ON {self.config.table_name} (status, next_retry_at)
                    WHERE status = 'retry'
                """)

                logger.debug(f"Platform Outbox Tabelle {self.config.table_name} erstellt/verifiziert")

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Platform Outbox Tabelle: {e}")
            raise

    @asynccontextmanager
    async def transaction(self):
        """Context Manager für transaktionale Outbox Operations"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Outbox Manager Metriken zurück"""
        return {
            "running": self.running,
            "events_stored": self.events_stored,
            "events_published": self.events_published,
            "events_failed": self.events_failed,
            "retry_attempts": self.retry_attempts,
            "success_rate": self.events_published / max(self.events_stored, 1) * 100
        }
