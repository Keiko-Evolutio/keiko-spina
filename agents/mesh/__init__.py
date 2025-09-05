"""Mesh-Subsystem für Agent-Kommunikation.

Das Mesh-Subsystem bietet eine konsolidierte Infrastruktur für:

## Event-System
- **AgentEvent**: Event-Datenstruktur mit Metadaten
- **AgentEventBus**: Event-Bus mit Idempotenz und Subscriptions
- **EventSubscription**: Callback-basiertes Subscription-System

## Utility-Komponenten
- **HashGenerator**: Zentrale Hash-Generierung für Idempotenz
- **ThreadSafeCache**: Thread-sicherer Cache
- **IdempotencyManager**: Duplikatserkennung mit TTL-Support

## Konstanten
- Policy-Entscheidungen (ALLOW, DENY, UNKNOWN)
- Hash- und Encoding-Konfiguration
- EventBus-Parameter und Cache-TTL

Alle Komponenten sind thread-safe.
"""

from __future__ import annotations

from .agent_event_bus import AgentEvent, AgentEventBus, EventSubscription

# Konstanten und Konfiguration
from .mesh_constants import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_ENCODING,
    HASH_ALGORITHM,
    MAX_EVENT_HISTORY_SIZE,
    POLICY_DECISION_ALLOW,
    POLICY_DECISION_DENY,
    POLICY_DECISION_UNKNOWN,
)

# Utility-Komponenten
from .utils import HashGenerator, IdempotencyManager, ThreadSafeCache

# Version und Metadaten
__version__ = "2.0.0"
__author__ = "Development Team"
__description__ = "Agent Mesh Subsystem"

# Hauptexporte
__all__ = [
    # Konstanten
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_ENCODING",
    "HASH_ALGORITHM",
    "MAX_EVENT_HISTORY_SIZE",
    "POLICY_DECISION_ALLOW",
    "POLICY_DECISION_DENY",
    "POLICY_DECISION_UNKNOWN",
    # Event-System
    "AgentEvent",
    "AgentEventBus",
    "EventSubscription",
    # Utilities
    "HashGenerator",
    "IdempotencyManager",
    "ThreadSafeCache",
]
