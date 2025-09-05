"""Type-Definitionen für das Registry-System.

Type Hints und semantische Type Aliases.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeAlias
from uuid import UUID

AgentID: TypeAlias = str
TenantID: TypeAlias = str
VersionConstraint: TypeAlias = str
InstanceID: TypeAlias = str
ServiceID: TypeAlias = str

CapabilityList: TypeAlias = list[str]
AgentMetadata: TypeAlias = dict[str, Any]
HealthMetrics: TypeAlias = dict[str, float | int | str]
LoadMetrics: TypeAlias = dict[str, float | int]

MatchScore: TypeAlias = float
HealthScore: TypeAlias = float
LoadFactor: TypeAlias = float
ConfidenceScore: TypeAlias = float
Timestamp: TypeAlias = datetime
OptionalTimestamp: TypeAlias = datetime | None

AgentInstance: TypeAlias = Any
AgentCollection: TypeAlias = dict[AgentID, AgentInstance]
CapabilitySet: TypeAlias = set[str]

DiscoveryResult: TypeAlias = dict[str, Any]
DiscoveryResults: TypeAlias = list[DiscoveryResult]
SearchQuery: TypeAlias = str
SearchFilters: TypeAlias = dict[str, Any]

HealthStatus: TypeAlias = str
HealthReport: TypeAlias = dict[str, Any]
HealthHistory: TypeAlias = list[HealthReport]

EndpointURL: TypeAlias = str
LoadBalancingDecision: TypeAlias = dict[str, Any]
TrafficWeight: TypeAlias = float
RolloutID: TypeAlias = str
RolloutPhase: TypeAlias = str
RolloutProgress: TypeAlias = float  # 0.0 - 1.0 (Prozent)

# Error-Types
ErrorCode: TypeAlias = str
ErrorDetails: TypeAlias = dict[str, Any]

# Configuration-Types
RegistryConfig: TypeAlias = dict[str, Any]
DiscoveryConfig: TypeAlias = dict[str, Any]
HealthConfig: TypeAlias = dict[str, Any]
LoadBalancingConfig: TypeAlias = dict[str, Any]

HealthCallback: TypeAlias = Callable[[ServiceID, HealthStatus], None]
LoadCallback: TypeAlias = Callable[[ServiceID, LoadFactor], None]
DiscoveryCallback: TypeAlias = Callable[[AgentID, DiscoveryResult], None]

# Union-Types für Flexibilität
AgentIdentifier: TypeAlias = AgentID | UUID | int
VersionSpecifier: TypeAlias = str | int | tuple[int, int, int]
TimeSpecifier: TypeAlias = datetime | int | float  # datetime, timestamp, seconds
