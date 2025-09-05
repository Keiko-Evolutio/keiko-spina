"""Event Handler Package für Agent-Events."""

from .._compat import UnavailableClass as _UnavailableClass

try:
    from .constants import (
        ContentType,
        EventHandlerErrorCode,
        EventType,
        MessageStatus,
        RunStatus,
        StepStatus,
        StepType,
    )
    from .event_handler import AgentEventHandler
    _CONSTANTS_AVAILABLE = True
except ImportError:
    AgentEventHandler = _UnavailableClass  # type: ignore[assignment]
    _CONSTANTS_AVAILABLE = False

__all__ = ["AgentEventHandler"]

if _CONSTANTS_AVAILABLE:
    __all__.extend([
        "ContentType",
        "EventHandlerErrorCode",
        "EventType",
        "MessageStatus",
        "RunStatus",
        "StepStatus",
        "StepType",
    ])

__version__ = "0.1.0"
__author__ = "Development Team"
__description__ = "Azure AI Foundry Event-Handler für Personal Assistant"
