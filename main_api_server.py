#!/usr/bin/env python3
"""Main API Server für Issue #56 Messaging-first Architecture
integriert HTTP/REST APIs, gRPC Services und Event-Streaming

ARCHITEKTUR-COMPLIANCE:
- Implementiert vollständige API-first Kommunikationsstrategie
- Nutzt Platform Event Bus aus Phase 1
- Keine direkten NATS-Zugriffe für externe Systeme
- Comprehensive API-Versionierung und Documentation
"""

import asyncio

# Import setup_api_versioning directly from the versioning module file
import importlib.util
import signal
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# HTTP APIs (Phase 2)
from api.v1 import events_router, management_router

# Platform Messaging (Phase 1)
from messaging import PlatformEventBus
from messaging.platform_event_bus import PlatformEventBusConfig
from messaging.platform_nats_client import PlatformNATSConfig

# Load the versioning module directly from the file
versioning_path = Path(__file__).parent / "api" / "versioning.py"
spec = importlib.util.spec_from_file_location("api_versioning_module", versioning_path)
versioning_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(versioning_module)
setup_api_versioning = versioning_module.setup_api_versioning

# gRPC Services (Phase 2)
from grpc_services import GRPCServer
from kei_logging import get_logger

logger = get_logger(__name__)

class KeikoAPIServer:
    """Hauptklasse für Keiko Platform-SDK API Server"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or KeikoAPIServer._get_default_config()

        # Platform Event Bus (Phase 1)
        self.event_bus: PlatformEventBus | None = None

        # FastAPI App
        self.app: FastAPI | None = None

        # gRPC Server
        self.grpc_server: GRPCServer | None = None

        # Server Status
        self.running = False
        self.start_time: datetime | None = None

        # Metriken
        self.http_requests = 0
        self.grpc_requests = 0
        self.websocket_connections = 0

    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        """Standard-Konfiguration"""
        return {
            "http": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
                "workers": 1
            },
            "grpc": {
                "host": "0.0.0.0",
                "port": 50051,
                "max_workers": 10
            },
            "nats": {
                "servers": ["nats://localhost:4222"],
                "cluster_name": "platform-cluster",
                "jetstream_enabled": True
            },
            "api": {
                "title": "Keiko Platform-SDK API",
                "description": "API-first Kommunikation zwischen Platform und SDK",
                "version": "1.0.0",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "openapi_url": "/openapi.json"
            },
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            }
        }

    async def start(self) -> bool:
        """Startet den kompletten API Server"""
        try:
            logger.info("🚀 Starte Keiko Platform-SDK API Server...")

            # 1. Platform Event Bus starten (Phase 1)
            if not await self._start_event_bus():
                return False

            # 2. FastAPI App erstellen und konfigurieren
            self._create_fastapi_app()

            # 3. gRPC Server starten
            if not await self._start_grpc_server():
                return False

            self.running = True
            self.start_time = datetime.now(UTC)

            logger.info("✅ Keiko Platform-SDK API Server erfolgreich gestartet")
            logger.info(f"📋 HTTP API: {self.config['http']['host']}:{self.config['http']['port']}")
            logger.info(f"🔌 gRPC API: {self.config['grpc']['host']}:{self.config['grpc']['port']}")
            logger.info(f"📚 API Docs: {self.config['http']['host']}:{self.config['http']['port']}/docs")

            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Starten des API Servers: {e}")
            await self.stop()
            return False

    async def stop(self):
        """Stoppt den API Server"""
        try:
            logger.info("🛑 Stoppe Keiko Platform-SDK API Server...")

            # gRPC Server stoppen
            if self.grpc_server:
                await self.grpc_server.stop()

            # Event Bus stoppen
            if self.event_bus:
                await self.event_bus.stop()

            self.running = False
            logger.info("✅ API Server gestoppt")

        except Exception as e:
            logger.error(f"❌ Fehler beim Stoppen des API Servers: {e}")

    async def _start_event_bus(self) -> bool:
        """Startet Platform Event Bus"""
        try:
            logger.info("📡 Starte Platform Event Bus...")

            # NATS Konfiguration
            nats_config = PlatformNATSConfig(
                servers=self.config["nats"]["servers"],
                cluster_name=self.config["nats"]["cluster_name"],
                jetstream_enabled=self.config["nats"]["jetstream_enabled"]
            )

            # Event Bus Konfiguration
            event_bus_config = PlatformEventBusConfig(
                nats_config=nats_config,
                enable_schema_validation=True,
                enable_dead_letter_queue=True
            )

            # Event Bus erstellen und starten
            self.event_bus = PlatformEventBus(event_bus_config)

            if await self.event_bus.start():
                logger.info("✅ Platform Event Bus gestartet")
                return True
            logger.error("❌ Platform Event Bus konnte nicht gestartet werden")
            return False

        except Exception as e:
            logger.error(f"❌ Fehler beim Starten des Platform Event Bus: {e}")
            return False

    def _create_fastapi_app(self):
        """Erstellt und konfiguriert FastAPI App"""
        try:
            logger.info("🌐 Erstelle FastAPI App...")

            # Lifespan Event Handler
            @asynccontextmanager
            async def lifespan(app: FastAPI):
                # Startup
                logger.info("🚀 FastAPI Startup Event")
                yield
                # Shutdown
                logger.info("🛑 FastAPI Shutdown Event")
                await self.stop()

            # FastAPI App erstellen
            self.app = FastAPI(
                title=self.config["api"]["title"],
                description=self.config["api"]["description"],
                version=self.config["api"]["version"],
                docs_url=self.config["api"]["docs_url"],
                redoc_url=self.config["api"]["redoc_url"],
                openapi_url=self.config["api"]["openapi_url"],
                lifespan=lifespan
            )

            # API Versioning und Documentation Setup
            setup_api_versioning(self.app)

            # Middleware
            self._setup_middleware()

            # Routes
            self._setup_routes()

            logger.info("✅ FastAPI App erstellt und konfiguriert")

        except Exception as e:
            logger.error(f"❌ Fehler beim Erstellen der FastAPI App: {e}")
            raise

    def _setup_middleware(self):
        """Setup Middleware"""
        # CORS Middleware
        self.app.add_middleware(
            CORSMiddleware,
            **self.config["cors"]
        )

        # Request Logging Middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = datetime.now()

            # Request verarbeiten
            response = await call_next(request)

            # Metriken aktualisieren
            self.http_requests += 1

            # Log Request
            process_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"HTTP {request.method} {request.url.path} "
                f"-> {response.status_code} ({process_time:.3f}s)"
            )

            # Response Headers hinzufügen
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-API-Version"] = "v1"

            return response

    def _setup_routes(self):
        """Setup API Routes"""
        # API v1 Routes
        self.app.include_router(events_router, prefix="/api/v1")
        self.app.include_router(management_router, prefix="/api/v1")

        # Root Route
        @self.app.get("/")
        async def root():
            return RedirectResponse(url="/docs")

        # Health Check
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.running else "unhealthy",
                "timestamp": datetime.now(UTC).isoformat(),
                "version": self.config["api"]["version"],
                "uptime_seconds": int((datetime.now(UTC) - self.start_time).total_seconds()) if self.start_time else 0,
                "services": {
                    "event_bus": self.event_bus.started if self.event_bus else False,
                    "grpc_server": self.grpc_server.running if self.grpc_server else False
                }
            }

        # Metrics Endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            metrics = {
                "server": {
                    "running": self.running,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "uptime_seconds": int((datetime.now(UTC) - self.start_time).total_seconds()) if self.start_time else 0,
                    "http_requests": self.http_requests,
                    "grpc_requests": self.grpc_requests,
                    "websocket_connections": self.websocket_connections
                }
            }

            # Event Bus Metriken
            if self.event_bus:
                metrics["event_bus"] = self.event_bus.get_metrics()

            # gRPC Server Metriken
            if self.grpc_server:
                metrics["grpc_server"] = self.grpc_server.get_metrics()

            return metrics



    async def _start_grpc_server(self) -> bool:
        """Startet gRPC Server"""
        try:
            logger.info("🔌 Starte gRPC Server...")

            if not self.event_bus:
                logger.error("❌ Event Bus muss vor gRPC Server gestartet werden")
                return False

            # gRPC Server erstellen
            from grpc_services.grpc_server import create_grpc_server

            self.grpc_server = await create_grpc_server(
                event_bus=self.event_bus,
                host=self.config["grpc"]["host"],
                port=self.config["grpc"]["port"]
            )

            # gRPC Server starten
            if await self.grpc_server.start():
                logger.info("✅ gRPC Server gestartet")
                return True
            logger.error("❌ gRPC Server konnte nicht gestartet werden")
            return False

        except Exception as e:
            logger.error(f"❌ Fehler beim Starten des gRPC Servers: {e}")
            return False

    async def run_http_server(self):
        """Startet HTTP Server mit Uvicorn"""
        config = uvicorn.Config(
            app=self.app,
            host=self.config["http"]["host"],
            port=self.config["http"]["port"],
            reload=self.config["http"]["reload"],
            workers=self.config["http"]["workers"],
            log_level="info"
        )

        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """Hauptfunktion"""
    # Konfiguration
    config = {
        "http": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False,
            "workers": 1
        },
        "grpc": {
            "host": "0.0.0.0",
            "port": 50051
        },
        "nats": {
            "servers": ["nats://localhost:4222"],
            "cluster_name": "platform-cluster",
            "jetstream_enabled": True
        }
    }

    # API Server erstellen
    api_server = KeikoAPIServer(config)

    # Signal Handler für Graceful Shutdown
    def signal_handler(signum, _frame):
        logger.info(f"Signal {signum} empfangen - starte Shutdown...")
        asyncio.create_task(api_server.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Server starten
        if await api_server.start():
            # HTTP Server laufen lassen
            await api_server.run_http_server()
        else:
            logger.error("❌ API Server konnte nicht gestartet werden")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("👋 Keyboard Interrupt empfangen")
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)
    finally:
        await api_server.stop()

if __name__ == "__main__":
    asyncio.run(main())
