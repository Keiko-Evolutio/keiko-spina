"""Gemeinsame Type-Definitionen für Service-Interfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

# Gemeinsame Type Aliases
ServiceId = str
ChannelName = str
SubjectName = str
AgentId = str
TaskPayload = dict[str, Any]
EventData = dict[str, Any]
HealthStatus = dict[str, Any]

# Handler Types
EventHandler = Callable[[bytes], Awaitable[None]]
MessageHandler = Callable[[EventData], Awaitable[None]]

# Result Types
ServiceResult = dict[str, Any]
OperationResult = bool | dict[str, Any]

# Configuration Types
ServiceConfig = dict[str, Any]
CapabilityConfig = dict[str, Any]

# Optional Types
OptionalConfig = ServiceConfig | None
OptionalTimeout = float | None
OptionalQueue = str | None
