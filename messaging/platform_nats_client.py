#!/usr/bin/env python3
"""Platform NATS Client für Issue #56 Messaging-first Architecture
Implementiert NATS JetStream Client ausschließlich für Platform-interne Kommunikation

ARCHITEKTUR-COMPLIANCE:
- Nur für Platform-interne Nutzung
- Keine SDK-Dependencies oder -Exports
- Vollständig isoliert von externen Systemen
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import nats
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig, RetentionPolicy, StorageType, StreamConfig

from kei_logging import get_logger

logger = get_logger(__name__)

class PlatformStreamType(Enum):
    """Platform-interne Stream-Typen"""
    AGENTS = "platform.agents"
    TASKS = "platform.tasks"
    WORKFLOWS = "platform.workflows"
    SYSTEM = "platform.system"

@dataclass
class PlatformNATSConfig:
    """Konfiguration für Platform NATS Client"""
    servers: list[str]
    cluster_name: str
    max_reconnect_attempts: int = 10
    reconnect_time_wait: int = 2
    max_outstanding_acks: int = 1000
    max_deliver: int = 3
    ack_wait_seconds: int = 30

    # JetStream Konfiguration
    jetstream_enabled: bool = True
    max_memory: str = "2Gi"
    max_storage: str = "100Gi"

    # Stream-spezifische Konfiguration
    stream_configs: dict[str, dict[str, Any]] = None

    def __post_init__(self):
        if self.stream_configs is None:
            self.stream_configs = {
                PlatformStreamType.AGENTS.value: {
                    "subjects": ["platform.agents.>"],
                    "retention": "7d",
                    "max_msgs": 1000000,
                    "max_bytes": 1024 * 1024 * 1024,  # 1GB
                    "storage": StorageType.FILE
                },
                PlatformStreamType.TASKS.value: {
                    "subjects": ["platform.tasks.>"],
                    "retention": "30d",
                    "max_msgs": 10000000,
                    "max_bytes": 10 * 1024 * 1024 * 1024,  # 10GB
                    "storage": StorageType.FILE
                },
                PlatformStreamType.WORKFLOWS.value: {
                    "subjects": ["platform.workflows.>"],
                    "retention": "7d",
                    "max_msgs": 500000,
                    "max_bytes": 512 * 1024 * 1024,  # 512MB
                    "storage": StorageType.FILE
                },
                PlatformStreamType.SYSTEM.value: {
                    "subjects": ["platform.system.>"],
                    "retention": "3d",
                    "max_msgs": 100000,
                    "max_bytes": 100 * 1024 * 1024,  # 100MB
                    "storage": StorageType.MEMORY
                }
            }

@dataclass
class PlatformMessage:
    """Platform-interne Nachricht"""
    subject: str
    data: dict[str, Any]
    headers: dict[str, str] | None = None
    message_id: str | None = None
    correlation_id: str | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        if self.message_id is None:
            self.message_id = f"msg_{int(time.time() * 1000000)}"

class PlatformNATSClient:
    """Platform NATS Client für interne Messaging"""

    def __init__(self, config: PlatformNATSConfig):
        self.config = config
        self.client: NATSClient | None = None
        self.jetstream: JetStreamContext | None = None
        self.connected = False
        self.subscriptions: dict[str, Any] = {}

        # Metriken
        self.messages_published = 0
        self.messages_received = 0
        self.connection_errors = 0

    async def connect(self) -> bool:
        """Verbindet mit NATS Server und initialisiert JetStream"""
        try:
            logger.info(f"Verbinde mit Platform NATS: {self.config.servers}")

            # NATS Client-Konfiguration
            self.client = await nats.connect(
                servers=self.config.servers,
                name=f"platform-{self.config.cluster_name}",
                max_reconnect_attempts=self.config.max_reconnect_attempts,
                reconnect_time_wait=self.config.reconnect_time_wait,
                error_cb=self._error_callback,
                disconnected_cb=self._disconnected_callback,
                reconnected_cb=self._reconnected_callback
            )

            # JetStream initialisieren
            if self.config.jetstream_enabled:
                self.jetstream = self.client.jetstream()
                await self._setup_streams()

            self.connected = True
            logger.info("Platform NATS Client erfolgreich verbunden")
            return True

        except Exception as e:
            self.connection_errors += 1
            logger.error(f"Fehler beim Verbinden mit Platform NATS: {e}")
            return False

    async def disconnect(self):
        """Trennt Verbindung zu NATS"""
        try:
            if self.client and self.connected:
                # Alle Subscriptions beenden
                for sub_name, subscription in self.subscriptions.items():
                    try:
                        await subscription.unsubscribe()
                        logger.debug(f"Subscription {sub_name} beendet")
                    except Exception as e:
                        logger.warning(f"Fehler beim Beenden der Subscription {sub_name}: {e}")

                await self.client.close()
                self.connected = False
                logger.info("Platform NATS Client getrennt")

        except Exception as e:
            logger.error(f"Fehler beim Trennen von Platform NATS: {e}")

    async def _setup_streams(self):
        """Erstellt Platform-interne JetStream Streams"""
        try:
            for stream_name, stream_config in self.config.stream_configs.items():
                try:
                    # Prüfe ob Stream bereits existiert
                    try:
                        await self.jetstream.stream_info(stream_name)
                        logger.debug(f"Platform Stream {stream_name} bereits vorhanden")
                        continue
                    except:
                        pass

                    # Erstelle neuen Stream
                    config = StreamConfig(
                        name=stream_name,
                        subjects=stream_config["subjects"],
                        retention=RetentionPolicy.LIMITS,
                        max_msgs=stream_config["max_msgs"],
                        max_bytes=stream_config["max_bytes"],
                        storage=stream_config["storage"],
                        max_age=self._parse_retention(stream_config["retention"])
                    )

                    await self.jetstream.add_stream(config)
                    logger.info(f"Platform Stream {stream_name} erstellt")

                except Exception as e:
                    logger.error(f"Fehler beim Erstellen von Stream {stream_name}: {e}")

        except Exception as e:
            logger.error(f"Fehler beim Setup der Platform Streams: {e}")

    def _parse_retention(self, retention_str: str) -> int:
        """Konvertiert Retention-String zu Sekunden"""
        if retention_str.endswith("d"):
            return int(retention_str[:-1]) * 24 * 3600
        if retention_str.endswith("h"):
            return int(retention_str[:-1]) * 3600
        if retention_str.endswith("m"):
            return int(retention_str[:-1]) * 60
        return int(retention_str)

    async def publish(self, message: PlatformMessage, stream: str | None = None) -> bool:
        """Publiziert Platform-interne Nachricht"""
        if not self.connected or not self.jetstream:
            logger.error("Platform NATS Client nicht verbunden")
            return False

        try:
            # Bestimme Stream basierend auf Subject
            if stream is None:
                stream = self._determine_stream(message.subject)

            # Erstelle NATS-Message
            headers = message.headers or {}
            headers.update({
                "message-id": message.message_id,
                "correlation-id": message.correlation_id or message.message_id,
                "timestamp": message.timestamp.isoformat(),
                "platform-internal": "true"  # Markierung für Platform-interne Messages
            })

            # Publiziere Message
            ack = await self.jetstream.publish(
                subject=message.subject,
                payload=json.dumps(message.data).encode(),
                headers=headers,
                stream=stream
            )

            self.messages_published += 1
            logger.debug(f"Platform Message publiziert: {message.subject} -> {stream}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Publizieren der Platform Message: {e}")
            return False

    async def subscribe(self,
                       subject: str,
                       callback: Callable[[PlatformMessage], Any],  # Can be sync or async
                       consumer_name: str | None = None,
                       durable: bool = True) -> str | None:
        """Abonniert Platform-interne Messages"""
        if not self.connected or not self.jetstream:
            logger.error("Platform NATS Client nicht verbunden")
            return None

        try:
            # Bestimme Stream und Consumer
            stream = self._determine_stream(subject)
            if consumer_name is None:
                consumer_name = f"platform-consumer-{subject.replace('.', '-')}-{int(time.time())}"

            # Consumer-Konfiguration
            consumer_config = ConsumerConfig(
                name=consumer_name,
                durable_name=consumer_name if durable else None,
                ack_policy="explicit",
                max_deliver=self.config.max_deliver,
                ack_wait=self.config.ack_wait_seconds
            )

            # Erstelle Subscription
            subscription = await self.jetstream.subscribe(
                subject=subject,
                cb=self._create_message_handler(callback),
                stream=stream,
                config=consumer_config
            )

            subscription_id = f"{stream}-{consumer_name}"
            self.subscriptions[subscription_id] = subscription

            logger.info(f"Platform Subscription erstellt: {subject} -> {consumer_name}")
            return subscription_id

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Platform Subscription: {e}")
            return None

    def _create_message_handler(self, callback: Callable[[PlatformMessage], Any]):
        """Erstellt Message-Handler für Subscription"""
        async def handler(msg):
            try:
                # Parse Message
                data = json.loads(msg.data.decode())
                headers = dict(msg.headers) if msg.headers else {}

                platform_message = PlatformMessage(
                    subject=msg.subject,
                    data=data,
                    headers=headers,
                    message_id=headers.get("message-id"),
                    correlation_id=headers.get("correlation-id"),
                    timestamp=datetime.fromisoformat(headers.get("timestamp", datetime.now(UTC).isoformat()))
                )

                # Callback ausführen (handle both sync and async)
                import inspect
                if inspect.iscoroutinefunction(callback):
                    await callback(platform_message)
                else:
                    callback(platform_message)

                # Message bestätigen
                await msg.ack()
                self.messages_received += 1

            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten der Platform Message: {e}")
                await msg.nak()

        return handler

    def _determine_stream(self, subject: str) -> str:
        """Bestimmt Stream basierend auf Subject"""
        if subject.startswith("platform.agents."):
            return PlatformStreamType.AGENTS.value
        if subject.startswith("platform.tasks."):
            return PlatformStreamType.TASKS.value
        if subject.startswith("platform.workflows."):
            return PlatformStreamType.WORKFLOWS.value
        if subject.startswith("platform.system."):
            return PlatformStreamType.SYSTEM.value
        # Default zu System Stream
        return PlatformStreamType.SYSTEM.value

    async def _error_callback(self, error):
        """Callback für NATS Fehler"""
        self.connection_errors += 1
        logger.error(f"Platform NATS Fehler: {error}")

    async def _disconnected_callback(self):
        """Callback für NATS Disconnection"""
        self.connected = False
        logger.warning("Platform NATS Verbindung getrennt")

    async def _reconnected_callback(self):
        """Callback für NATS Reconnection"""
        self.connected = True
        logger.info("Platform NATS Verbindung wiederhergestellt")

    def get_metrics(self) -> dict[str, Any]:
        """Gibt Client-Metriken zurück"""
        return {
            "connected": self.connected,
            "messages_published": self.messages_published,
            "messages_received": self.messages_received,
            "connection_errors": self.connection_errors,
            "active_subscriptions": len(self.subscriptions),
            "jetstream_enabled": self.config.jetstream_enabled
        }
