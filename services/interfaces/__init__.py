"""Service-Interfaces (ABCs) für das DI-System.

Organisiert nach Service-Kategorien:
- Core Services: Kern-Funktionalität (Agent, Bus, Stream)
- Infrastructure Services: System-Management (ServiceManager, DomainRevalidation)
- Feature Services: Feature-spezifisch (Voice, WebhookManager)
- Utility Services: Hilfsfunktionen (RateLimiter)
"""

from __future__ import annotations

# Base classes and types
from ._base import (
    CoreService,
    FeatureService,
    InfrastructureService,
    LifecycleService,
    ServiceStatus,
    UtilityService,
)
from ._types import (
    AgentId,
    CapabilityConfig,
    ChannelName,
    EventData,
    EventHandler,
    HealthStatus,
    MessageHandler,
    OperationResult,
    OptionalConfig,
    OptionalQueue,
    OptionalTimeout,
    ServiceConfig,
    ServiceId,
    ServiceResult,
    SubjectName,
    TaskPayload,
)

# Core Services
from .agent_service import AgentService
from .bus_service import BusService

# Infrastructure Services
from .domain_revalidation_service import DomainRevalidationService

# Utility Services
# Backward compatibility aliases
from .rate_limiter import RateLimiterBackend, RateLimiterService, RateLimitResult
from .service_manager import ServiceManagerService
from .stream_service import StreamService

# Feature Services
from .voice_service import VoiceService
from .webhook_manager import WebhookManagerService

__all__ = [
    # Types
    "AgentId",
    # Core Services
    "AgentService",
    "BusService",
    "CapabilityConfig",
    "ChannelName",
    "CoreService",
    # Infrastructure Services
    "DomainRevalidationService",
    "EventData",
    "EventHandler",
    "FeatureService",
    "HealthStatus",
    "InfrastructureService",
    # Base classes
    "LifecycleService",
    "MessageHandler",
    "OperationResult",
    "OptionalConfig",
    "OptionalQueue",
    "OptionalTimeout",
    "RateLimitResult",
    # Backward compatibility
    "RateLimiterBackend",
    # Utility Services
    "RateLimiterService",
    "ServiceConfig",
    "ServiceId",
    "ServiceManagerService",
    "ServiceResult",
    "ServiceStatus",
    "StreamService",
    "SubjectName",
    "TaskPayload",
    "UtilityService",
    # Feature Services
    "VoiceService",
    "WebhookManagerService",
]
