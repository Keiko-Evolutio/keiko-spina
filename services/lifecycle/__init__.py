"""Services Lifecycle Management.

Dieses Modul stellt Lifecycle-Management für Services bereit.
Es bietet sowohl die neue Enterprise-Grade Implementierung als auch
Backward-Compatibility für bestehenden Code.
"""

from __future__ import annotations

# Neue Enterprise-Grade Implementierung (empfohlen)
try:
    from agents.lifecycle import AgentLifecycleManager, agent_lifecycle_manager
except ImportError:
    # Fallback if lifecycle module is not available
    AgentLifecycleManager = None
    agent_lifecycle_manager = None

# Backward-Compatibility Adapter
from .lifecycle_adapter import InstanceLifecycle, LifecycleAdapter, lifecycle

# Legacy-Implementierung (deprecated, nur für Migration)
try:
    from .instance_lifecycle import lifecycle as legacy_lifecycle
except ImportError:
    legacy_lifecycle = None

__all__ = [
    "AgentLifecycleManager",
    "InstanceLifecycle",
    "LifecycleAdapter",
    # Neue Implementierung (empfohlen)
    "agent_lifecycle_manager",
    # Legacy (deprecated)
    "legacy_lifecycle",
    # Backward-Compatibility
    "lifecycle",
]
