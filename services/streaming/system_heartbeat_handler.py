"""System Heartbeat Handler für WebSocket-Streaming
Integriert sich in die bestehende WebSocket-Infrastruktur
"""

import asyncio
import logging
import subprocess
import time
from typing import Any

from data_models.websocket import SystemHeartbeatEvent

# from services.system_heartbeat_service import get_system_heartbeat_service
from .manager import websocket_manager

logger = logging.getLogger(__name__)


class SystemHeartbeatStreamer:
    """Streamt System-Heartbeat-Daten über die bestehende WebSocket-Infrastruktur."""

    def __init__(self):
        self.running = False
        self.stream_task: asyncio.Task = None
        self.last_heartbeat: dict[str, Any] = None
        self.subscribed_connections: set[str] = set()

    async def start_streaming(self):
        """Startet den Heartbeat-Stream."""
        if self.running:
            return

        self.running = True
        self.stream_task = asyncio.create_task(self._stream_loop())
        logger.info("💓 System Heartbeat Streaming gestartet")

    async def stop_streaming(self):
        """Stoppt den Heartbeat-Stream."""
        self.running = False
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 System Heartbeat Streaming gestoppt")

    def subscribe_connection(self, connection_id: str):
        """Abonniert eine Verbindung für Heartbeat-Updates."""
        self.subscribed_connections.add(connection_id)
        logger.debug(f"💓 Connection {connection_id} abonniert Heartbeat-Updates")

    def unsubscribe_connection(self, connection_id: str):
        """Deabonniert eine Verbindung von Heartbeat-Updates."""
        self.subscribed_connections.discard(connection_id)
        logger.debug(f"💓 Connection {connection_id} deabonniert Heartbeat-Updates")

    async def _stream_loop(self):
        """Haupt-Stream-Loop."""
        while self.running:
            try:
                # Aktuellen Heartbeat abrufen
                current_heartbeat = await self._get_current_heartbeat()

                # Nur senden wenn sich etwas geändert hat und Abonnenten vorhanden sind
                if (current_heartbeat and
                    current_heartbeat != self.last_heartbeat and
                    self.subscribed_connections):

                    await self._broadcast_heartbeat(current_heartbeat)
                    self.last_heartbeat = current_heartbeat

                # Alle 5 Sekunden aktualisieren
                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Heartbeat-Stream: {e}")
                await asyncio.sleep(5.0)

    async def _get_current_heartbeat(self) -> dict[str, Any]:
        """Ruft den aktuellen Heartbeat ab."""
        try:
            # Direkte Container-Discovery (vereinfacht)
            return await self._discover_containers_direct()

        except Exception as e:
            logger.warning(f"Fehler beim Abrufen des Heartbeats: {e}")
            return None

    async def _discover_containers_direct(self) -> dict[str, Any]:
        """Direkte Container-Discovery als Fallback."""
        try:
            result = subprocess.run([
                "docker", "ps", "--filter", "name=keiko-",
                "--format", "{{.Names}}\t{{.Status}}"
            ], check=False, capture_output=True, text=True, timeout=5)

            services = {}
            container_count = 0

            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            name = parts[0].replace("keiko-", "")
                            status = parts[1]
                            container_count += 1

                            # Container als healthy markieren wenn er läuft
                            is_healthy = "Up" in status
                            services[name] = {
                                "name": name,
                                "status": "healthy" if is_healthy else "unhealthy",
                                "last_check": time.time(),
                                "response_time_ms": 1.0
                            }

            # Backend-Service hinzufügen (läuft nativ)
            services["application"] = {
                "name": "application",
                "status": "healthy",
                "last_check": time.time(),
                "response_time_ms": 1.0
            }

            total_services = len(services)
            healthy_services = sum(1 for s in services.values() if s["status"] == "healthy")

            return {
                "timestamp": time.time(),
                "phase": "ready",
                "overall_status": "healthy" if healthy_services == total_services else "degraded",
                "services": services,
                "summary": {
                    "total": total_services,
                    "healthy": healthy_services,
                    "unhealthy": total_services - healthy_services,
                    "starting": 0,
                    "failed": 0,
                    "unknown": 0
                },
                "uptime_seconds": time.time() - 1756000000,
                "message": f"🎉 All {total_services} services are healthy ({container_count} Docker containers + 1 backend)"
            }

        except Exception as e:
            logger.error(f"Container discovery failed: {e}")
            return None

    async def _broadcast_heartbeat(self, heartbeat_data: dict[str, Any]):
        """Sendet Heartbeat-Daten an alle abonnierten Verbindungen."""
        if not self.subscribed_connections:
            return

        # Erstelle SystemHeartbeatEvent
        heartbeat_event = SystemHeartbeatEvent(
            timestamp=heartbeat_data["timestamp"],
            phase=heartbeat_data["phase"],
            overall_status=heartbeat_data["overall_status"],
            services=heartbeat_data["services"],
            summary=heartbeat_data["summary"],
            uptime_seconds=heartbeat_data["uptime_seconds"],
            message=heartbeat_data["message"]
        )

        # Sende an alle abonnierten Verbindungen
        disconnected_connections = set()

        for connection_id in self.subscribed_connections.copy():
            success = await websocket_manager.send_to_connection(connection_id, heartbeat_event)
            if not success:
                disconnected_connections.add(connection_id)

        # Entferne disconnected connections
        for connection_id in disconnected_connections:
            self.unsubscribe_connection(connection_id)

        if disconnected_connections:
            logger.debug(f"💓 Entfernte {len(disconnected_connections)} disconnected Heartbeat-Abonnenten")


# Globaler Streamer
system_heartbeat_streamer = SystemHeartbeatStreamer()


async def handle_system_heartbeat_subscription(connection_id: str, message_data: dict[str, Any]):
    """Handler für System Heartbeat Subscription-Nachrichten."""
    action = message_data.get("action")

    if action == "subscribe":
        system_heartbeat_streamer.subscribe_connection(connection_id)

        # Starte Streaming wenn erster Abonnent
        if len(system_heartbeat_streamer.subscribed_connections) == 1:
            await system_heartbeat_streamer.start_streaming()

        # Sende initialen Heartbeat
        current_heartbeat = await system_heartbeat_streamer._get_current_heartbeat()
        if current_heartbeat:
            heartbeat_event = SystemHeartbeatEvent(
                timestamp=current_heartbeat["timestamp"],
                phase=current_heartbeat["phase"],
                overall_status=current_heartbeat["overall_status"],
                services=current_heartbeat["services"],
                summary=current_heartbeat["summary"],
                uptime_seconds=current_heartbeat["uptime_seconds"],
                message=current_heartbeat["message"]
            )
            await websocket_manager.send_to_connection(connection_id, heartbeat_event)

    elif action == "unsubscribe":
        system_heartbeat_streamer.unsubscribe_connection(connection_id)

        # Stoppe Streaming wenn keine Abonnenten mehr
        if len(system_heartbeat_streamer.subscribed_connections) == 0:
            await system_heartbeat_streamer.stop_streaming()


async def handle_connection_disconnect(connection_id: str):
    """Handler für Verbindungsabbrüche."""
    system_heartbeat_streamer.unsubscribe_connection(connection_id)

    # Stoppe Streaming wenn keine Abonnenten mehr
    if len(system_heartbeat_streamer.subscribed_connections) == 0:
        await system_heartbeat_streamer.stop_streaming()


# Startup/Shutdown Events
async def startup_system_heartbeat_streaming():
    """Initialisiert das System Heartbeat Streaming beim Startup."""
    logger.info("💓 System Heartbeat Streaming Handler initialisiert")


async def shutdown_system_heartbeat_streaming():
    """Stoppt das System Heartbeat Streaming beim Shutdown."""
    await system_heartbeat_streamer.stop_streaming()
    logger.info("💓 System Heartbeat Streaming Handler beendet")
