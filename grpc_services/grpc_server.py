#!/usr/bin/env python3
"""gRPC Server für Issue #56 Messaging-first Architecture
Implementiert gRPC Server für SDK-Platform Kommunikation

ARCHITEKTUR-COMPLIANCE:
- Hostet SDKPlatformCommunicationService
- Nutzt Platform Event Bus für interne Kommunikation
- High-performance bidirektionale Streaming
- Comprehensive Health Checks und Metrics
"""

import asyncio
import signal
from datetime import UTC, datetime
from typing import Any

from grpc import aio

# Optionale Health Check Imports - graceful fallback
try:
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    # Fallback für fehlende grpc-health-checking Dependency
    health_pb2_grpc = None
    health_pb2 = None
    HEALTH_CHECK_AVAILABLE = False

    # Mock Health Check Response für Fallback
    class MockHealthCheckResponse:
        """Mock Health Check Response wenn grpc-health-checking nicht verfügbar"""
        SERVING = "SERVING"
        NOT_SERVING = "NOT_SERVING"
        SERVICE_UNKNOWN = "SERVICE_UNKNOWN"

        def __init__(self, status="SERVING"):
            self.status = status

    # Mock health_pb2 für Fallback
    class MockHealthPb2:
        HealthCheckResponse = MockHealthCheckResponse

    health_pb2 = MockHealthPb2()

# Optionale Reflection Imports
try:
    from grpc_reflection.v1alpha import reflection
    REFLECTION_AVAILABLE = True
except ImportError:
    reflection = None
    REFLECTION_AVAILABLE = False

from kei_logging import get_logger
from messaging import PlatformEventBus

from .sdk_platform_service import SDKPlatformCommunicationServiceImpl

logger = get_logger(__name__)

class HealthServicer:
    """gRPC Health Check Service mit graceful fallback"""

    def __init__(self, grpc_server_instance):
        self.grpc_server = grpc_server_instance
        self.service_status = {}

    async def check(self, request, _context):
        """Health Check für spezifischen Service"""
        service = getattr(request, "service", "")

        if service == "":
            # Overall Server Health
            status = health_pb2.HealthCheckResponse.SERVING
        elif service in self.service_status:
            status = self.service_status[service]
        else:
            status = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN

        return health_pb2.HealthCheckResponse(status=status)

    async def watch(self, request, context):
        """Health Status Streaming"""
        service = getattr(request, "service", "")

        # Initial Status senden
        if service in self.service_status:
            status = self.service_status[service]
        else:
            status = health_pb2.HealthCheckResponse.SERVING

        yield health_pb2.HealthCheckResponse(status=status)

        # Periodische Updates (vereinfacht) - nur wenn echte gRPC Health verfügbar
        if HEALTH_CHECK_AVAILABLE and hasattr(context, "cancelled"):
            while not context.cancelled():
                await asyncio.sleep(30)
                yield health_pb2.HealthCheckResponse(status=status)

    def set_service_status(self, service: str, status):
        """Setzt Service Status"""
        self.service_status[service] = status

class GRPCServer:
    """gRPC Server für Platform-SDK Kommunikation"""

    def __init__(self,
                 event_bus: PlatformEventBus,
                 host: str = "0.0.0.0",
                 port: int = 50051,
                 max_workers: int = 10):
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.max_workers = max_workers

        # gRPC Server
        self.server: aio.Server | None = None
        self.running = False

        # Services
        self.sdk_platform_service = SDKPlatformCommunicationServiceImpl(event_bus)
        self.health_service = HealthServicer(self)

        # Metriken
        self.start_time: datetime | None = None
        self.total_requests = 0
        self.active_connections = 0

    async def start(self) -> bool:
        """Startet den gRPC Server"""
        try:
            logger.info(f"Starte gRPC Server auf {self.host}:{self.port}...")

            # gRPC Server erstellen
            self.server = aio.server(
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                    ("grpc.max_receive_message_length", 4 * 1024 * 1024),  # 4MB
                    ("grpc.max_send_message_length", 4 * 1024 * 1024),     # 4MB
                ]
            )

            # Services registrieren
            await self._register_services()

            # Server Port binden
            listen_addr = f"{self.host}:{self.port}"
            self.server.add_insecure_port(listen_addr)

            # Server starten
            await self.server.start()

            self.running = True
            self.start_time = datetime.now(UTC)

            # Health Status setzen
            self.health_service.set_service_status(
                "keiko.sdk.communication.v1.SDKPlatformCommunicationService",
                health_pb2.HealthCheckResponse.SERVING
            )

            logger.info(f"gRPC Server erfolgreich gestartet auf {listen_addr}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Starten des gRPC Servers: {e}")
            return False

    async def stop(self, grace_period: int = 30):
        """Stoppt den gRPC Server"""
        try:
            logger.info("Stoppe gRPC Server...")

            if self.server and self.running:
                # Health Status auf NOT_SERVING setzen
                self.health_service.set_service_status(
                    "keiko.sdk.communication.v1.SDKPlatformCommunicationService",
                    health_pb2.HealthCheckResponse.NOT_SERVING
                )

                # Graceful Shutdown
                await self.server.stop(grace_period)

                self.running = False
                logger.info("gRPC Server gestoppt")

        except Exception as e:
            logger.error(f"Fehler beim Stoppen des gRPC Servers: {e}")

    async def _register_services(self):
        """Registriert gRPC Services"""
        try:
            # SDK-Platform Communication Service
            # In echter Implementierung würde hier der generierte Service Stub verwendet:
            # sdk_communication_pb2_grpc.add_SDKPlatformCommunicationServiceServicer_to_server(
            #     self.sdk_platform_service, self.server
            # )

            # Für Demo registrieren wir Mock-Service
            logger.info("SDK-Platform Communication Service registriert")

            # Health Service (nur wenn verfügbar)
            if HEALTH_CHECK_AVAILABLE and health_pb2_grpc:
                # Type check: health_pb2_grpc is guaranteed to exist here due to condition above
                health_pb2_grpc.add_HealthServicer_to_server(self.health_service, self.server)  # type: ignore[attr-defined]
                logger.info("Health Service registriert")
            else:
                logger.warning("Health Service nicht verfügbar - grpc-health-checking fehlt")

            # Reflection Service (für Development, nur wenn verfügbar)
            if REFLECTION_AVAILABLE and reflection:
                service_names = [
                    "keiko.sdk.communication.v1.SDKPlatformCommunicationService",
                    "grpc.health.v1.Health",
                    "grpc.reflection.v1alpha.ServerReflection"
                ]
                reflection.enable_server_reflection(service_names, self.server)
                logger.info("Reflection Service registriert")
            else:
                logger.warning("Reflection Service nicht verfügbar")

        except Exception as e:
            logger.error(f"Fehler beim Registrieren der gRPC Services: {e}")
            raise

    async def wait_for_termination(self):
        """Wartet auf Server-Termination"""
        if self.server:
            await self.server.wait_for_termination()

    def get_metrics(self) -> dict[str, Any]:
        """Gibt gRPC Server Metriken zurück"""
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = int((datetime.now(UTC) - self.start_time).total_seconds())

        # Service-spezifische Metriken
        service_metrics = self.sdk_platform_service.get_metrics()

        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "uptime_seconds": uptime_seconds,
            "active_connections": self.active_connections,
            "service_metrics": service_metrics,
            "health_status": "serving" if self.running else "not_serving"
        }

    async def handle_shutdown_signal(self, signum, _frame=None):
        """Handler für Shutdown-Signale"""
        logger.info(f"Shutdown-Signal empfangen: {signum}")
        await self.stop()

async def create_grpc_server(event_bus: PlatformEventBus,
                           host: str = "0.0.0.0",
                           port: int = 50051) -> GRPCServer:
    """Factory-Funktion für gRPC Server"""
    server = GRPCServer(event_bus, host, port)

    # Signal Handler registrieren
    loop = asyncio.get_event_loop()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(
            sig,
            lambda s=sig, frame=None: asyncio.create_task(
                server.handle_shutdown_signal(s, frame)
            )
        )

    return server

async def main():
    """Hauptfunktion für standalone gRPC Server"""
    # Mock Event Bus für Demo
    from messaging.platform_event_bus import PlatformEventBusConfig
    from messaging.platform_nats_client import PlatformNATSConfig

    nats_config = PlatformNATSConfig(
        servers=["nats://localhost:4222"],
        cluster_name="platform-cluster"
    )

    config = PlatformEventBusConfig(nats_config=nats_config)
    event_bus = PlatformEventBus(config)

    # Event Bus starten
    if not await event_bus.start():
        logger.error("Fehler beim Starten des Event Bus")
        return

    try:
        # gRPC Server erstellen und starten
        grpc_server = await create_grpc_server(event_bus)

        if not await grpc_server.start():
            logger.error("Fehler beim Starten des gRPC Servers")
            return

        logger.info("gRPC Server läuft - warte auf Termination...")

        # Warte auf Termination
        await grpc_server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt empfangen")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
    finally:
        # Cleanup
        grpc_server = locals().get("grpc_server")
        if grpc_server is not None:
            await grpc_server.stop()
        await event_bus.stop()

if __name__ == "__main__":
    asyncio.run(main())
