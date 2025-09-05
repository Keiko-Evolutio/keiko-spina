# backend/services/__init__.py
"""Services Paket mit zentralem Service- und Pool-Management.

Konsolidiert verschiedene Services inklusive Messaging (migriert aus kei_bus/).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

logger = get_logger(__name__)

# =====================================================================
# Core Components Import
# =====================================================================

from .core.features import features
from .core.manager import service_manager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# =====================================================================
# Optional Imports mit Fallback-Pattern
# =====================================================================

def _safe_import_messaging() -> dict[str, Any]:
    """Sichere Messaging-Imports (migriert aus kei_bus/)."""
    try:
        from .messaging import (
            BaseProvider,
            BusEnvelope,
            BusService,
            BusSettings,
            KafkaProvider,
            NATSProvider,
            OutboxPattern,
            SagaPattern,
            get_messaging_service,
        )
        return {
            "available": True,
            "BaseProvider": BaseProvider,
            "BusEnvelope": BusEnvelope,
            "BusService": BusService,
            "BusSettings": BusSettings,
            "KafkaProvider": KafkaProvider,
            "NATSProvider": NATSProvider,
            "OutboxPattern": OutboxPattern,
            "SagaPattern": SagaPattern,
            "get_messaging_service": get_messaging_service,
        }
    except ImportError as e:
        logger.debug(f"Messaging Services nicht verfügbar: {e}")
        return {"available": False}


def _safe_import_streaming() -> dict[str, Any]:
    """Sichere Streaming-Imports (migriert aus kei_stream/ und websocket/)."""
    try:
        from .streaming import (
            GRPCStreamTransport,
            KEIStreamFrame,
            SessionManager,
            SSETransport,
            WebSocketManager,
            WebSocketTransport,
            session_manager,
            websocket_manager,
        )
        return {
            "available": True,
            "KEIStreamFrame": KEIStreamFrame,
            "SessionManager": SessionManager,
            "WebSocketManager": WebSocketManager,
            "SSETransport": SSETransport,
            "WebSocketTransport": WebSocketTransport,
            "GRPCStreamTransport": GRPCStreamTransport,
            "websocket_manager": websocket_manager,
            "session_manager": session_manager,
        }
    except ImportError as e:
        logger.debug(f"Streaming Services nicht verfügbar: {e}")
        return {"available": False}


def _safe_import_webhooks() -> dict[str, Any]:
    """Sichere Webhook-Imports (vollständig migriert nach services/webhooks/)."""
    try:
        from .webhooks import (
            WebhookManager,
            get_webhook_manager,
            set_webhook_manager,
        )
        return {
            "available": True,
            "WebhookManager": WebhookManager,
            "get_webhook_manager": get_webhook_manager,
            "set_webhook_manager": set_webhook_manager,
        }
    except ImportError as e:
        logger.debug(f"Webhook Services nicht verfügbar: {e}")
        return {"available": False}

def _safe_import_clients() -> dict[str, Any]:
    """Sichere Client-Imports."""
    try:
        from .clients import (
            Services,
            foundry_credential,
        )
        from .clients import (
            http_client as _base_http_client,
        )
        return {
            "available": True,
            "Services": Services,
            "http_client": _base_http_client,
            "foundry_credential": foundry_credential,
        }
    except ImportError as e:
        logger.debug(f"Client Services nicht verfügbar: {e}")
        return {"available": False}


def _safe_import_pools() -> dict[str, Any]:
    """Sichere Pool-Imports."""
    if not features.is_available("pools"):
        return {"available": False}

    try:
        from .pools import (
            BaseResourcePool,
            PoolHealth,
            PoolMetrics,
            azure_pools,
            get_http_client,
        )
        from .pools import (
            cleanup_pools as _cleanup_pools,
        )
        from .pools import (
            get_health_status as _pools_health_status,
        )
        from .pools import (
            initialize_pools as _initialize_pools,
        )
        return {
            "available": True,
            "azure_pools": azure_pools,
            "BaseResourcePool": BaseResourcePool,
            "PoolHealth": PoolHealth,
            "PoolMetrics": PoolMetrics,
            "get_http_client": get_http_client,
            "get_health_status": _pools_health_status,
            "initialize_pools": _initialize_pools,
            "cleanup_pools": _cleanup_pools,
        }
    except ImportError:
        return {"available": False}


# Module-Imports durchführen
_clients = _safe_import_clients()
_pools = _safe_import_pools()
_messaging = _safe_import_messaging()
_streaming = _safe_import_streaming()
_webhooks = _safe_import_webhooks()

# Service Manager Status
_service_manager_initialization_attempted = False


# =====================================================================
# Public API Functions
# =====================================================================

@asynccontextmanager
async def http_client() -> AsyncIterator[Any]:
    """HTTP-Client Context Manager.

    Gibt einen asynchronen Kontextmanager zurück, der eine HTTP-Verbindung
    aus einem Pool oder dem Standard-Client bereitstellt.
    """
    # Priorisiere Pool-Client wenn verfügbar
    if _pools["available"]:
        pool_client = _pools["get_http_client"]()
        if pool_client:
            async with pool_client.get_connection() as conn:
                yield conn
                return

    # Fallback zu Standard-Client
    if _clients["available"]:
        client = _clients["http_client"]()
        if client:
            async with client.session() as c:
                yield c
            return

    # Letzter Fallback: Service Manager
    await _ensure_service_manager_initialized()
    client = service_manager.get_service("http_client")
    if client:
        async with client.session() as c:
            yield c
        return
    raise RuntimeError("Kein HTTP-Client verfügbar")


async def get_service_health() -> dict[str, Any]:
    """Gesamter Service-Health-Status."""
    await _ensure_service_manager_initialized()
    health = await service_manager.get_health_status()

    # Pool-Health hinzufügen
    if _pools["available"]:
        try:
            pool_health = await _pools["get_health_status"]()
            health["pools"] = pool_health
        except Exception as e:
            health["pools"] = {"error": str(e)}

    health.update({
        "clients_available": _clients["available"],
        "pools_available": _pools["available"],
    })

    return health


def is_services_healthy() -> bool:
    """Prüft grundlegende Service-Verfügbarkeit (synchron)."""
    # Basis-Checks ohne Service Manager Initialisierung
    basic_health = (
        features.is_available("http_clients") and
        _clients["available"]
    )

    # Service Manager Check nur wenn bereits initialisiert
    if service_manager._initialized:
        return basic_health and service_manager.is_healthy()
    # Wenn noch nicht initialisiert, verwende Basis-Health
    # aber vermerke, dass vollständige Initialisierung noch aussteht
    return basic_health


async def is_services_healthy_async() -> bool:
    """Prüft umfassende Service-Verfügbarkeit (asynchron)."""
    await _ensure_service_manager_initialized()
    return (
        features.is_available("http_clients") and
        _clients["available"] and
        service_manager.is_healthy()
    )


async def _ensure_service_manager_initialized() -> None:
    """Stellt sicher, dass der Service Manager initialisiert ist."""
    global _service_manager_initialization_attempted

    if not service_manager._initialized and not _service_manager_initialization_attempted:
        _service_manager_initialization_attempted = True
        try:
            await service_manager.initialize()
            logger.info("Service Manager erfolgreich initialisiert")
        except Exception as e:
            logger.exception(f"Service Manager Initialisierung fehlgeschlagen: {e}")


async def initialize_services() -> None:
    """Services initialisieren."""
    await _ensure_service_manager_initialized()
    if _pools["available"]:
        await _pools["initialize_pools"]()


async def shutdown_services() -> None:
    """Services bereinigen."""
    if _pools["available"]:
        await _pools["cleanup_pools"]()
    await service_manager.cleanup()


def get_optimized_http_client() -> Any | None:
    """HTTP-Client mit Pool-Optimierung."""
    if _pools["available"]:
        return _pools["get_http_client"]()
    if _clients["available"]:
        return _clients["http_client"]()
    return None


# =====================================================================
# Dynamic Exports basierend auf verfügbaren Features
# =====================================================================

_base_exports = [
    "features",
    "service_manager",
    "http_client",
    "get_service_health",
    "is_services_healthy",
    "is_services_healthy_async",
    "initialize_services",
    "shutdown_services",
    "get_optimized_http_client",
]

_client_exports = [
    "Services",
    "foundry_credential",
    # Zusätzliche bequeme Exporte für Azure AI Foundry
] if _clients["available"] else []

_pool_exports = [
    "BaseResourcePool",
    "PoolHealth",
    "PoolMetrics",
    "azure_pools",
] if _pools["available"] else []

_messaging_exports = [
    "BaseProvider",
    "BusEnvelope",
    "BusService",
    "BusSettings",
    "KafkaProvider",
    "NATSProvider",
    "get_messaging_service",
] if _messaging["available"] else []

_streaming_exports = [
    "KEIStreamFrame",
    "SessionManager",
    "WebSocketManager",
    "SSETransport",
    "WebSocketTransport",
    "GRPCStreamTransport",
    "websocket_manager",
    "session_manager",
] if _streaming["available"] else []

_webhooks_exports = [
    "WebhookManager",
    "get_webhook_manager",
    "set_webhook_manager",
] if _webhooks["available"] else []

# Add conditional exports
if _messaging["available"]:
    if "OutboxPattern" in _messaging:
        _messaging_exports.append("OutboxPattern")
    if "SagaPattern" in _messaging:
        _messaging_exports.append("SagaPattern")

__all__ = _base_exports + _client_exports + _pool_exports + _messaging_exports + _streaming_exports + _webhooks_exports

# =====================================================================
# Module Globals für verfügbare Services
# =====================================================================

if _clients["available"]:
    Services = _clients["Services"]
    foundry_credential = _clients["foundry_credential"]
    # Kein direkter Export für ai_project_client, Zugriff über service_manager oder Services-Instanz

if _pools["available"]:
    azure_pools = _pools["azure_pools"]
    BaseResourcePool = _pools["BaseResourcePool"]
    PoolHealth = _pools["PoolHealth"]
    PoolMetrics = _pools["PoolMetrics"]

if _messaging["available"]:
    BaseProvider = _messaging["BaseProvider"]
    BusEnvelope = _messaging["BusEnvelope"]
    BusService = _messaging["BusService"]
    BusSettings = _messaging["BusSettings"]
    KafkaProvider = _messaging["KafkaProvider"]
    NATSProvider = _messaging["NATSProvider"]
    get_messaging_service = _messaging["get_messaging_service"]

    # Conditional advanced features
    if "OutboxPattern" in _messaging:
        OutboxPattern = _messaging["OutboxPattern"]
    if "SagaPattern" in _messaging:
        SagaPattern = _messaging["SagaPattern"]

if _streaming["available"]:
    KEIStreamFrame = _streaming["KEIStreamFrame"]
    SessionManager = _streaming["SessionManager"]
    WebSocketManager = _streaming["WebSocketManager"]
    SSETransport = _streaming["SSETransport"]
    WebSocketTransport = _streaming["WebSocketTransport"]
    GRPCStreamTransport = _streaming["GRPCStreamTransport"]
    websocket_manager = _streaming["websocket_manager"]
    session_manager = _streaming["session_manager"]

if _webhooks["available"]:
    WebhookManager = _webhooks["WebhookManager"]
    get_webhook_manager = _webhooks["get_webhook_manager"]
    set_webhook_manager = _webhooks["set_webhook_manager"]


# =====================================================================
# Logging der Initialisierung
# =====================================================================

def _log_services_status() -> None:
    """Loggt Service-Status mit verbesserter Logik."""
    available = [k for k, v in features.all_features.items() if v]

    # Verwende Basis-Health für initiales Logging
    basic_healthy = (
        features.is_available("http_clients") and
        _clients["available"]
    )

    if basic_healthy:
        if service_manager._initialized:
            # Vollständiger Health-Check möglich
            status = "✅" if service_manager.is_healthy() else "⚠️"
        else:
            # Service Manager noch nicht initialisiert - verwende tentative Status
            status = "🔄"  # Initialisierung läuft
    else:
        status = "⚠️"

    logger.info(f"{status} Services - Features: {', '.join(available)}")

    # Zusätzliche Debug-Information
    if not basic_healthy:
        logger.debug(
            f"Health Details - http_clients: {features.is_available('http_clients')}, "
            f"clients_available: {_clients['available']}, "
            f"service_manager_initialized: {service_manager._initialized}"
        )


# Initialisierung beim Import
_log_services_status()

# Package Metadaten
__version__ = "0.0.1"
__author__ = "Keiko Development Team"
