"""Messaging Service Package.

Migriert aus kei_bus/ nach services/messaging/ als Teil der kei_*-Module-Konsolidierung.

Stellt ein skalierbares, sicheres und beobachtbares Messaging-/Event-Bus-Subsystem bereit
mit Unterstützung für Pub/Sub, Request/Reply, Sagas und Outbox/Inbox Patterns.

Hauptkomponenten:
- Kafka/NATS Provider für verschiedene Message-Broker
- Outbox-Pattern für transactional messaging
- Saga-Pattern für distributed transactions
- Dead Letter Queue (DLQ) für Fehlerbehandlung
- Schema-Registry für Message-Validierung
"""

from __future__ import annotations

from .base_provider import BaseProvider
from .config import BusSettings, bus_settings
from .constants import (
    CONSUME_OPERATION,
    DEFAULT_OUTBOX_NAME,
    DEFAULT_VERSION,
    PUBLISH_OPERATION,
    RPC_REQUEST_TYPE,
)
from .envelope import BusEnvelope
from .kafka_provider import KafkaProvider
from .naming import subject_for_event, subject_for_rpc
from .nats_provider import NATSProvider
from .service import BusService, get_bus_service

# Erweiterte Komponenten
try:
    from .outbox import OutboxPattern
    _OUTBOX_AVAILABLE = True
except ImportError:
    _OUTBOX_AVAILABLE = False

try:
    from .sagas import SagaPattern
    _SAGA_AVAILABLE = True
except ImportError:
    _SAGA_AVAILABLE = False

try:
    from .dlq import DeadLetterQueue
    _DLQ_AVAILABLE = True
except ImportError:
    _DLQ_AVAILABLE = False

try:
    from .schema_registry import SchemaRegistry
    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

# Alias für Kompatibilität
get_messaging_service = get_bus_service

__all__ = [
    # Constants
    "CONSUME_OPERATION",
    "DEFAULT_OUTBOX_NAME",
    "DEFAULT_VERSION",
    "PUBLISH_OPERATION",
    "RPC_REQUEST_TYPE",

    # Core Components
    "BaseProvider",
    "BusEnvelope",
    "BusService",
    "BusSettings",
    "bus_settings",

    # Providers
    "KafkaProvider",
    "NATSProvider",

    # Service Functions
    "get_bus_service",
    "get_messaging_service",

    # Naming Utilities
    "subject_for_event",
    "subject_for_rpc",
]

# Conditional exports
if _OUTBOX_AVAILABLE:
    __all__.append("OutboxPattern")

if _SAGA_AVAILABLE:
    __all__.append("SagaPattern")

if _DLQ_AVAILABLE:
    __all__.append("DeadLetterQueue")

if _SCHEMA_AVAILABLE:
    __all__.append("SchemaRegistry")
