# backend/messaging/__init__.py
"""Platform-interne Messaging-Infrastruktur für Issue #56
Implementiert NATS JetStream basierte Event-driven Architecture

WICHTIG: Dieses Modul ist ausschließlich für Platform-interne Kommunikation.
Keine Abhängigkeiten zum SDK - Kommunikation erfolgt über API-Contracts.
"""

from .inbox_manager import PlatformInboxManager
from .outbox_manager import PlatformOutboxManager
from .platform_event_bus import PlatformEvent, PlatformEventBus
from .platform_nats_client import PlatformNATSClient
from .platform_schema_registry import PlatformSchemaRegistry

__all__ = [
    "PlatformEvent",
    "PlatformEventBus",
    "PlatformInboxManager",
    "PlatformNATSClient",
    "PlatformOutboxManager",
    "PlatformSchemaRegistry"
]
