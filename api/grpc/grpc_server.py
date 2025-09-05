"""KEI-RPC gRPC Server v1.

Startet einen gRPC-Server mit optionalem mTLS, Prometheus/OTLP-Instrumentierung
und Reflection. Der Server implementiert Platzhalter-CRUD/Batch sowie ein
bidi-Streaming für Long-running/Realtime Operationen.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import grpc

from kei_logging import get_logger

# Logger früh initialisieren für Import-Fehlerbehandlung
logger = get_logger(__name__)

try:  # Optional: Reflection nur wenn Paket vorhanden
    from grpc_reflection.v1alpha import reflection  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    reflection = None  # type: ignore
except Exception as reflection_error:  # pragma: no cover - unexpected import error
    logger.debug(f"Unerwarteter Fehler beim Import von grpc_reflection: {reflection_error}")
    reflection = None  # type: ignore

try:  # Optional Health Service
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    health = None  # type: ignore
    health_pb2 = None  # type: ignore
    health_pb2_grpc = None  # type: ignore
except Exception as health_error:  # pragma: no cover - unexpected import error
    logger.debug(f"Unerwarteter Fehler beim Import von grpc_health: {health_error}")
    health = None  # type: ignore
    health_pb2 = None  # type: ignore
    health_pb2_grpc = None  # type: ignore

import contextlib

from services.streaming.grpc_transport import KEIStreamService

from .error_mapping import ErrorMappingInterceptor
from .interceptors import (
    AuthInterceptor,
    DeadlineInterceptor,
    DLPInterceptor,
    MetricsInterceptor,
    RateLimitInterceptor,
    TracingInterceptor,
)

# Importiere für __all__ Export
try:
    from .interceptors.auth import AuthInterceptor as _AuthInterceptor
except ImportError:
    _AuthInterceptor = None

try:
    from .interceptors.rate_limit import RateLimitInterceptor as _RateLimitInterceptor
except ImportError:
    _RateLimitInterceptor = None
from .kei_rpc_service import KEIRPCService
from .proto.generate import generate_protos


class _InMemoryStore:
    """Einfacher In-Memory Store analog zu REST-Demo."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def list(self, page: int, per_page: int, q: str | None, sort: str) -> tuple[list[dict], int]:
        data = list(self.items.values())
        if q:
            data = [d for d in data if q.lower() in d.get("name", "").lower()]
        reverse = sort.startswith("-")
        key = sort.lstrip("-") or "updated_at"
        data.sort(key=lambda d: d.get(key, ""), reverse=reverse)
        total = len(data)
        start = (page - 1) * per_page
        end = start + per_page
        return data[start:end], total

    def create(self, name: str, idem: str | None) -> dict:
        now = datetime.now(UTC).isoformat()
        new_id = idem or f"res_{len(self.items)+1}"
        if new_id in self.items:
            return self.items[new_id]
        obj = {"id": new_id, "name": name, "created_at": now, "updated_at": now}
        self.items[new_id] = obj
        return obj

    def get(self, item_id: str) -> dict | None:
        return self.items.get(item_id)

    def patch(self, item_id: str, name: str | None) -> dict | None:
        obj = self.items.get(item_id)
        if not obj:
            return None
        if name is not None:
            obj["name"] = name
            obj["updated_at"] = datetime.now(UTC).isoformat()
        return obj


_STORE = _InMemoryStore()


async def serve_grpc(
    bind_address: str = "0.0.0.0:50051", *, enable_reflection: bool = True
) -> grpc.aio.Server:
    """Startet gRPC-Server und gibt Instanz zurück.

    Args:
        bind_address: Bind-Adresse mit Port
        enable_reflection: Ob Server Reflection aktiviert werden soll

    Returns:
        Gestartete gRPC Serverinstanz (asyncio)
    """
    # Reihenfolge: Auth → Tracing → DLP → RateLimit → Deadline → Metrics (Manager)
    interceptors = [
        AuthInterceptor(),
        TracingInterceptor(),
        DLPInterceptor(),
        RateLimitInterceptor(),
        DeadlineInterceptor(),
        MetricsInterceptor(),
        # Einheitliches Error-Mapping für KeikoExceptions
        ErrorMappingInterceptor(),
    ]
    # gRPC Kompression standardmäßig aktivierbar
    default_compression = os.getenv("KEI_RPC_DEFAULT_COMPRESSION", "gzip").lower()
    if default_compression == "gzip":
        server_compression = grpc.Compression.Gzip
    elif default_compression == "deflate":
        server_compression = grpc.Compression.Deflate
    elif default_compression in {"none", "off", "false"}:
        server_compression = grpc.Compression.NoCompression
    else:
        server_compression = grpc.Compression.Gzip

    # Server-Optionen inkl. optionaler Keepalive-Einstellungen aus ENV
    # Hinweis: gRPC Heartbeats werden nicht aktiv vom Applikationslayer gesendet;
    # Liveness erfolgt über HTTP/2 Keepalive/Pings auf Transportebene.
    options: list[tuple[str, Any]] = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]

    def _get_int(env_name: str) -> int | None:
        """Liest eine Integer-ENV-Variable, gibt None bei Fehlschlag zurück."""
        try:
            val = os.getenv(env_name)
            return int(val) if val is not None and val != "" else None
        except (ValueError, TypeError):
            return None
        except Exception as parse_error:
            logger.debug(f"Unerwarteter Fehler beim Parsen der ENV-Variable {env_name}: {parse_error}")
            return None

    def _get_bool(env_name: str) -> bool | None:
        """Liest eine Bool-ENV-Variable, gibt None bei Fehlschlag zurück."""
        val = os.getenv(env_name)
        if val is None:
            return None
        return val.lower() in {"1", "true", "yes", "on"}

    # Keepalive-Optionen (optional; nur setzen, wenn ENV vorhanden)
    keepalive_time = _get_int("KEI_RPC_KEEPALIVE_TIME_MS")
    if keepalive_time is not None:
        options.append(("grpc.keepalive_time_ms", keepalive_time))

    keepalive_timeout = _get_int("KEI_RPC_KEEPALIVE_TIMEOUT_MS")
    if keepalive_timeout is not None:
        options.append(("grpc.keepalive_timeout_ms", keepalive_timeout))

    keepalive_permit_no_calls = _get_bool("KEI_RPC_KEEPALIVE_PERMIT_WITHOUT_CALLS")
    if keepalive_permit_no_calls is not None:
        options.append(("grpc.keepalive_permit_without_calls", keepalive_permit_no_calls))

    max_pings_wo_data = _get_int("KEI_RPC_HTTP2_MAX_PINGS_WITHOUT_DATA")
    if max_pings_wo_data is not None:
        options.append(("grpc.http2.max_pings_without_data", max_pings_wo_data))

    min_time_between_pings = _get_int("KEI_RPC_HTTP2_MIN_TIME_BETWEEN_PINGS_MS")
    if min_time_between_pings is not None:
        options.append(("grpc.http2.min_time_between_pings_ms", min_time_between_pings))

    min_ping_interval_wo_data = _get_int("KEI_RPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS")
    if min_ping_interval_wo_data is not None:
        options.append(("grpc.http2.min_ping_interval_without_data_ms", min_ping_interval_wo_data))

    max_ping_strikes = _get_int("KEI_RPC_HTTP2_MAX_PING_STRIKES")
    if max_ping_strikes is not None:
        options.append(("grpc.http2.max_ping_strikes", max_ping_strikes))

    server = grpc.aio.server(
        interceptors=interceptors, compression=server_compression, options=options
    )

    # TLS/mTLS optional laden
    server_credentials = None
    try:
        cert = os.getenv("KEI_RPC_TLS_CERT")
        key = os.getenv("KEI_RPC_TLS_KEY")
        ca = os.getenv("KEI_RPC_TLS_CLIENT_CA")
        if cert and key:
            private_key = open(key, "rb").read()
            certificate_chain = open(cert, "rb").read()
            if ca:
                root_certificates = open(ca, "rb").read()
                server_credentials = grpc.ssl_server_credentials(
                    [(private_key, certificate_chain)],
                    root_certificates=root_certificates,
                    require_client_auth=True,
                )
                logger.info("gRPC TLS aktiviert (mTLS)")
            else:
                server_credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
                logger.info("gRPC TLS aktiviert")
    except (FileNotFoundError, PermissionError) as file_error:
        logger.warning(f"TLS-Konfiguration fehlerhaft - Datei-/Berechtigungsfehler: {file_error}")
    except (ValueError, TypeError) as type_error:
        logger.warning(f"TLS-Konfiguration fehlerhaft - Ungültige Parameter: {type_error}")
    except Exception as tls_error:
        logger.warning(f"TLS-Konfiguration fehlerhaft - Unerwarteter Fehler: {tls_error}")

    # KEI-RPC Servicer registrieren (nur wenn Protos generiert sind)
    try:
        from .proto import kei_rpc_pb2_grpc as rpc_pb2_grpc  # type: ignore
    except ImportError:
        # Versuche zur Laufzeit zu generieren
        if generate_protos():
            try:
                from .proto import kei_rpc_pb2_grpc as rpc_pb2_grpc  # type: ignore
            except ImportError:
                logger.debug("KEI-RPC Proto-Module konnten nicht generiert werden")
                rpc_pb2_grpc = None  # type: ignore
            except Exception as proto_import_error:
                logger.warning(f"Unerwarteter Fehler beim Import der generierten KEI-RPC Protos: {proto_import_error}")
                rpc_pb2_grpc = None  # type: ignore
        else:
            rpc_pb2_grpc = None  # type: ignore
    except Exception as rpc_error:
        logger.warning(f"Unerwarteter Fehler beim Import der KEI-RPC Proto-Module: {rpc_error}")
        rpc_pb2_grpc = None  # type: ignore

    if rpc_pb2_grpc is not None:
        # Registriert den Servicer: add_KEIRPCServiceServicer_to_server
        if hasattr(rpc_pb2_grpc, "add_KEIRPCServiceServicer_to_server"):
            rpc_pb2_grpc.add_KEIRPCServiceServicer_to_server(KEIRPCService(), server)
        else:
            logger.warning("add_KEIRPCServiceServicer_to_server nicht verfügbar in rpc_pb2_grpc")

    # KEI-Stream gRPC Service registrieren (separates Proto-Modul)
    try:
        from services.streaming import grpc_stream_pb2_grpc as stream_pb2_grpc  # type: ignore
    except ImportError:
        # Versuche Stream-Protos zu generieren
        try:
            # Prüfe ob generate_stream_protos verfügbar ist
            try:
                from services.streaming.generate import generate_stream_protos
                if generate_stream_protos():
                    from services.streaming import (
                        grpc_stream_pb2_grpc as stream_pb2_grpc,  # type: ignore
                    )
                else:
                    stream_pb2_grpc = None  # type: ignore
            except ImportError:
                logger.debug("generate_stream_protos nicht verfügbar")
                generate_stream_protos = None  # type: ignore
                stream_pb2_grpc = None  # type: ignore
        except ImportError:
            logger.debug("Stream Proto-Module konnten nicht generiert werden")
            stream_pb2_grpc = None  # type: ignore
        except Exception as stream_gen_error:
            logger.warning(f"Unerwarteter Fehler beim Generieren der Stream-Protos: {stream_gen_error}")
            stream_pb2_grpc = None  # type: ignore
    except Exception as stream_import_error:
        logger.warning(f"Unerwarteter Fehler beim Import der Stream Proto-Module: {stream_import_error}")
        stream_pb2_grpc = None  # type: ignore

    if stream_pb2_grpc is not None:  # type: ignore
        try:
            if hasattr(stream_pb2_grpc, "add_KEIStreamServiceServicer_to_server"):
                stream_pb2_grpc.add_KEIStreamServiceServicer_to_server(KEIStreamService(), server)  # type: ignore
                logger.info("KEI-Stream gRPC Service registriert")
            else:
                logger.warning("add_KEIStreamServiceServicer_to_server nicht verfügbar in stream_pb2_grpc")
        except (AttributeError, TypeError) as stream_attr_error:
            logger.warning(f"KEI-Stream gRPC Service konnte nicht registriert werden - Attribut-/Typ-Fehler: {stream_attr_error}")
        except Exception as stream_service_error:
            logger.warning(f"KEI-Stream gRPC Service konnte nicht registriert werden - Unerwarteter Fehler: {stream_service_error}")

    # Health Service registrieren (falls Paket vorhanden)
    health_servicer = None
    if health is not None and health_pb2_grpc is not None:
        try:
            if hasattr(health, "HealthServicer") and hasattr(health_pb2_grpc, "add_HealthServicer_to_server"):
                health_servicer = health.HealthServicer()
                health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
            else:
                logger.warning("HealthServicer oder add_HealthServicer_to_server nicht verfügbar")
        except (AttributeError, TypeError) as health_attr_error:
            logger.debug(f"Health Service konnte nicht registriert werden - Attribut-/Typ-Fehler: {health_attr_error}")
            health_servicer = None
        except Exception as health_service_error:
            logger.warning(f"Health Service konnte nicht registriert werden - Unerwarteter Fehler: {health_service_error}")
            health_servicer = None

    # Reflection in PROD standardmäßig deaktivieren
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if enable_reflection and environment != "production" and reflection is not None:
        service_names = [
            reflection.SERVICE_NAME,
            "kei.rpc.v1.KEIRPCService",
            "grpc.health.v1.Health",
        ]
        reflection.enable_server_reflection(service_names, server)

    # Bind
    if server_credentials:
        server.add_secure_port(bind_address, server_credentials)
    else:
        server.add_insecure_port(bind_address)

    await server.start()
    # Health Status setzen
    if health_servicer is not None and health_pb2 is not None:
        with contextlib.suppress(Exception):
            if hasattr(health_pb2, "HealthCheckResponse") and hasattr(health_pb2.HealthCheckResponse, "SERVING"):
                health_servicer.set("kei.rpc.v1.KEIRPCService", health_pb2.HealthCheckResponse.SERVING)
    logger.info("KEI-RPC gRPC Server gestartet auf %s", bind_address)
    return server


async def shutdown_grpc(server: grpc.aio.Server) -> None:
    """Beendet gRPC-Server.

    Args:
        server: gRPC Serverinstanz
    """
    if not server:
        return
    await server.stop(grace=None)
    logger.info("gRPC Server gestoppt")


__all__ = ["serve_grpc", "shutdown_grpc"]
