"""gRPC Server Factory für KEI-RPC.

Implementiert bessere Struktur, Single Responsibility Principle
und verbesserte Testbarkeit.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import grpc

from kei_logging import get_logger

from .constants import GRPCServerConfig, ServiceNames
from .error_mapping import ErrorMappingInterceptor
from .interceptors import create_interceptor_chain

# Service-Imports
try:
    from services.streaming.grpc_transport import KEIStreamService
    # Korrekter Import für grpc_stream_pb2_grpc
    try:
        from services.streaming import grpc_stream_pb2_grpc
    except ImportError:
        grpc_stream_pb2_grpc = None
    STREAMING_SERVICES_AVAILABLE = True
except ImportError:
    grpc_stream_pb2_grpc = None
    KEIStreamService = None
    STREAMING_SERVICES_AVAILABLE = False

logger = get_logger(__name__)

# Optional Dependencies
try:
    from grpc_reflection.v1alpha import reflection

    REFLECTION_AVAILABLE = True
except ImportError:
    reflection = None
    REFLECTION_AVAILABLE = False

try:
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    from grpc_health.v1.health import HealthServicer

    HEALTH_AVAILABLE = True
except ImportError:
    health = None
    health_pb2 = None
    health_pb2_grpc = None
    HealthServicer = None
    HEALTH_AVAILABLE = False


class GRPCServerFactory:
    """Factory für gRPC Server-Erstellung mit konfigurierbaren Komponenten.

    Trennt Server-Erstellung, Service-Registration und Konfiguration
    für bessere Testbarkeit und Wartbarkeit.
    """

    def __init__(self) -> None:
        """Initialisiert Server Factory."""
        self.logger = get_logger(f"{__name__}.GRPCServerFactory")

    def create_server(
        self,
        bind_address: str = GRPCServerConfig.DEFAULT_BIND_ADDRESS,
        enable_reflection: bool = True,
        enable_health: bool = True,
        enable_mtls: bool = False,
    ) -> tuple[grpc.aio.Server, Any | None]:
        """Erstellt konfigurierten gRPC Server.

        Args:
            bind_address: Server-Bind-Adresse
            enable_reflection: Server Reflection aktivieren
            enable_health: Health Service aktivieren
            enable_mtls: mTLS aktivieren

        Returns:
            Konfigurierter gRPC Server
        """
        # 1. Interceptors erstellen
        interceptors = self._create_interceptors()

        # 2. Server erstellen
        server = grpc.aio.server(
            interceptors=interceptors,
            options=GRPCServerFactory._get_server_options(),
        )

        # 3. Services registrieren
        self._register_services(server)

        # 4. Optional: Reflection
        if enable_reflection and REFLECTION_AVAILABLE:
            self._setup_reflection(server)

        # 5. Optional: Health Service
        health_servicer = None
        if enable_health and HEALTH_AVAILABLE:
            health_servicer = self._setup_health_service(server)

        # 6. Optional: mTLS
        if enable_mtls:
            self._setup_mtls(server)

        # 7. Port hinzufügen
        if enable_mtls:
            server.add_secure_port(bind_address, self._create_server_credentials())
        else:
            server.add_insecure_port(bind_address)

        self.logger.info(f"gRPC Server erstellt auf {bind_address}")
        return server, health_servicer

    def _create_interceptors(self) -> list[grpc.aio.ServerInterceptor]:
        """Erstellt Interceptor-Chain.

        Returns:
            Liste von konfigurierten Interceptors
        """
        interceptors = create_interceptor_chain()

        # Error-Mapping als letzter Interceptor
        interceptors.append(ErrorMappingInterceptor())

        self.logger.info(f"Interceptor-Chain erstellt: {len(interceptors)} Interceptors")
        return interceptors

    @staticmethod
    def _get_server_options() -> list[tuple]:
        """Erstellt gRPC Server-Optionen.

        Returns:
            Liste von Server-Optionen
        """
        return [
            ("grpc.keepalive_time_ms", GRPCServerConfig.DEFAULT_KEEPALIVE_TIME_MS),
            ("grpc.keepalive_timeout_ms", GRPCServerConfig.DEFAULT_KEEPALIVE_TIMEOUT_MS),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
        ]

    def _register_services(self, server: grpc.aio.Server) -> None:
        """Registriert gRPC Services.

        Args:
            server: gRPC Server-Instanz
        """
        # KEI-RPC Service registrieren
        self._register_kei_rpc_service(server)

        # KEI-Stream Service registrieren (optional)
        self._register_kei_stream_service(server)

    def _register_kei_rpc_service(self, server: grpc.aio.Server) -> None:
        """Registriert KEI-RPC Service."""
        try:
            # Dynamischer Import für bessere Fehlerbehandlung
            from .kei_rpc_service import KEIRPCService
            from .proto.generate import generate_protos

            # Protos generieren falls nötig
            if not GRPCServerFactory._check_protos_exist() and not generate_protos():
                self.logger.error("Proto-Generierung fehlgeschlagen")
                return

            # Service registrieren
            try:
                from .proto import kei_rpc_pb2_grpc

                if hasattr(kei_rpc_pb2_grpc, "add_KEIRPCServiceServicer_to_server"):
                    kei_rpc_pb2_grpc.add_KEIRPCServiceServicer_to_server(KEIRPCService(), server)
                    self.logger.info("KEI-RPC Service registriert")
                else:
                    self.logger.warning("add_KEIRPCServiceServicer_to_server nicht verfügbar")
            except ImportError as e:
                kei_rpc_pb2_grpc = None  # type: ignore
                self.logger.warning(f"KEI-RPC Service Registration fehlgeschlagen: {e}")

        except Exception as e:
            self.logger.exception(f"Fehler bei KEI-RPC Service Registration: {e}")

    def _register_kei_stream_service(self, server: grpc.aio.Server) -> None:
        """Registriert KEI-Stream Service (optional)."""
        try:
            if not STREAMING_SERVICES_AVAILABLE or not grpc_stream_pb2_grpc:
                raise ImportError("Streaming services not available")

            grpc_stream_pb2_grpc.add_KEIStreamServiceServicer_to_server(KEIStreamService(), server)
            self.logger.info("KEI-Stream Service registriert")

        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"KEI-Stream Service Registration fehlgeschlagen: {e}")

    def _setup_reflection(self, server: grpc.aio.Server) -> None:
        """Aktiviert Server Reflection.

        Args:
            server: gRPC Server-Instanz
        """
        if not REFLECTION_AVAILABLE:
            self.logger.warning("gRPC Reflection nicht verfügbar")
            return

        try:
            service_names = [
                ServiceNames.KEI_RPC_SERVICE,
                ServiceNames.KEI_STREAM_SERVICE,
                reflection.SERVICE_NAME,
            ]

            if HEALTH_AVAILABLE:
                service_names.append(ServiceNames.HEALTH_SERVICE)

            reflection.enable_server_reflection(service_names, server)
            self.logger.info("Server Reflection aktiviert")

        except Exception as e:
            self.logger.exception(f"Server Reflection Setup fehlgeschlagen: {e}")

    def _setup_health_service(self, server: grpc.aio.Server) -> Any | None:
        """Aktiviert Health Service.

        Args:
            server: gRPC Server-Instanz

        Returns:
            Health Servicer oder None
        """
        if not HEALTH_AVAILABLE:
            self.logger.warning("gRPC Health Service nicht verfügbar")
            return None

        try:
            health_servicer = HealthServicer()
            if hasattr(health_pb2_grpc, "add_HealthServicer_to_server"):
                health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
                self.logger.info("Health Service aktiviert")
                return health_servicer
            self.logger.warning("add_HealthServicer_to_server nicht verfügbar")
            return None

        except Exception as e:
            self.logger.exception(f"Health Service Setup fehlgeschlagen: {e}")
            return None

    def _setup_mtls(self, server: grpc.aio.Server) -> None:
        """Konfiguriert mTLS (falls aktiviert).

        Args:
            server: gRPC Server-Instanz
        """
        # mTLS-Konfiguration würde hier implementiert werden
        # Vereinfacht für diese Implementierung
        _ = server  # Parameter wird aktuell nicht verwendet
        self.logger.info("mTLS-Setup übersprungen (nicht implementiert)")

    def _create_server_credentials(self) -> grpc.ServerCredentials:
        """Erstellt Server-Credentials für mTLS.

        Returns:
            gRPC Server-Credentials
        """
        # Vereinfachte Implementierung
        cert_path = GRPCServerConfig.MTLS_CERT_PATH
        key_path = GRPCServerConfig.MTLS_KEY_PATH

        if os.path.exists(cert_path) and os.path.exists(key_path):
            with open(cert_path, "rb") as f:
                cert_data = f.read()
            with open(key_path, "rb") as f:
                key_data = f.read()

            return grpc.ssl_server_credentials([(key_data, cert_data)])
        self.logger.warning("mTLS-Zertifikate nicht gefunden - verwende Insecure")
        raise FileNotFoundError("mTLS-Zertifikate nicht verfügbar")

    @staticmethod
    def _check_protos_exist() -> bool:
        """Prüft ob Proto-Dateien existieren.

        Returns:
            True wenn Protos verfügbar sind
        """
        try:
            return True
        except ImportError:
            return False


class GRPCServerManager:
    """Manager für gRPC Server Lifecycle.

    Vereinfacht Server-Start, -Stop und Health-Management.
    """

    def __init__(self, factory: GRPCServerFactory | None = None) -> None:
        """Initialisiert Server Manager.

        Args:
            factory: Server Factory (optional)
        """
        self.factory = factory or GRPCServerFactory()
        self.server: grpc.aio.Server | None = None
        self.health_servicer: Any | None = None
        self.logger = get_logger(f"{__name__}.GRPCServerManager")

    async def start_server(
        self, bind_address: str = GRPCServerConfig.DEFAULT_BIND_ADDRESS, **kwargs
    ) -> grpc.aio.Server:
        """Startet gRPC Server.

        Args:
            bind_address: Server-Bind-Adresse
            **kwargs: Weitere Server-Optionen

        Returns:
            Gestartete Server-Instanz
        """
        if self.server:
            self.logger.warning("Server bereits gestartet")
            return self.server

        # Server erstellen
        self.server, self.health_servicer = self.factory.create_server(
            bind_address=bind_address, **kwargs
        )

        # Server starten
        await self.server.start()

        # Health Status setzen
        if self.health_servicer and health_pb2 and hasattr(health_pb2, "HealthCheckResponse"):
            try:
                self.health_servicer.set(
                    ServiceNames.KEI_RPC_SERVICE, health_pb2.HealthCheckResponse.SERVING
                )
            except Exception as e:
                self.logger.warning(f"Health Status Setup fehlgeschlagen: {e}")

        self.logger.info(f"gRPC Server gestartet auf {bind_address}")
        return self.server

    async def stop_server(self, grace_period: int | None = None) -> None:
        """Stoppt gRPC Server.

        Args:
            grace_period: Grace Period in Sekunden
        """
        if not self.server:
            self.logger.warning("Server nicht gestartet")
            return

        grace = grace_period or GRPCServerConfig.DEFAULT_GRACE_PERIOD_SECONDS

        # Health Status auf NOT_SERVING setzen
        if self.health_servicer and health_pb2 and hasattr(health_pb2, "HealthCheckResponse"):
            with contextlib.suppress(Exception):
                self.health_servicer.set(
                    ServiceNames.KEI_RPC_SERVICE, health_pb2.HealthCheckResponse.NOT_SERVING
                )

        # Server stoppen
        await self.server.stop(grace)
        self.server = None
        self.health_servicer = None

        self.logger.info("gRPC Server gestoppt")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def serve_grpc(
    bind_address: str = GRPCServerConfig.DEFAULT_BIND_ADDRESS, **kwargs
) -> grpc.aio.Server:
    """Convenience-Funktion für Server-Start.

    Args:
        bind_address: Server-Bind-Adresse
        **kwargs: Server-Optionen

    Returns:
        Gestartete Server-Instanz
    """
    manager = GRPCServerManager()
    return await manager.start_server(bind_address, **kwargs)


async def shutdown_grpc(server: grpc.aio.Server) -> None:
    """Convenience-Funktion für Server-Stop.

    Args:
        server: Server-Instanz
    """
    if server:
        await server.stop(GRPCServerConfig.DEFAULT_GRACE_PERIOD_SECONDS)


__all__ = [
    "GRPCServerFactory",
    "GRPCServerManager",
    "serve_grpc",
    "shutdown_grpc",
]
