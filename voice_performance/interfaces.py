"""Voice Performance Optimization Interfaces.
Definiert abstrakte Interfaces für Parallel Discovery und Performance-Optimierung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DiscoveryType(Enum):
    """Typen von Discovery-Operationen."""
    AGENT_DISCOVERY = "agent_discovery"
    TOOL_DISCOVERY = "tool_discovery"
    SERVICE_DISCOVERY = "service_discovery"
    CAPABILITY_DISCOVERY = "capability_discovery"
    SCHEMA_DISCOVERY = "schema_discovery"


class ProcessingStage(Enum):
    """Voice Processing Pipeline Stages."""
    SPEECH_TO_TEXT = "speech_to_text"
    INTENT_RECOGNITION = "intent_recognition"
    AGENT_DISCOVERY = "agent_discovery"
    AGENT_SELECTION = "agent_selection"
    AGENT_EXECUTION = "agent_execution"
    RESPONSE_GENERATION = "response_generation"
    TEXT_TO_SPEECH = "text_to_speech"


class CacheStrategy(Enum):
    """Caching-Strategien für Performance-Optimierung."""
    NO_CACHE = "no_cache"
    MEMORY_CACHE = "memory_cache"
    DISTRIBUTED_CACHE = "distributed_cache"
    PREDICTIVE_CACHE = "predictive_cache"
    ADAPTIVE_CACHE = "adaptive_cache"


@dataclass
class VoiceWorkflowContext:
    """Kontext für Voice Workflow Processing."""
    workflow_id: str
    user_id: str
    session_id: str

    # Voice Input
    audio_data: bytes | None = None
    text_input: str | None = None
    language: str = "de-DE"

    # Processing Preferences
    max_latency_ms: int = 500
    parallel_processing: bool = True
    cache_enabled: bool = True

    # Quality Requirements
    min_confidence: float = 0.7
    max_agents: int = 5
    timeout_seconds: float = 30.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical

    # Context Data
    previous_interactions: list[str] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    session_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryRequest:
    """Request für Parallel Discovery."""
    discovery_type: DiscoveryType
    query: str
    context: VoiceWorkflowContext

    # Discovery Parameters
    max_results: int = 10
    timeout_seconds: float = 5.0
    parallel_execution: bool = True

    # Filtering
    required_capabilities: list[str] = field(default_factory=list)
    excluded_agents: list[str] = field(default_factory=list)
    region_preference: str | None = None

    # Quality Requirements
    min_confidence: float = 0.5
    max_latency_ms: int | None = None
    health_check_required: bool = True


@dataclass
class DiscoveryResult:
    """Ergebnis einer Discovery-Operation."""
    discovery_type: DiscoveryType
    query: str

    # Results
    items: list[dict[str, Any]]
    total_found: int
    confidence_scores: list[float]

    # Performance Metrics
    discovery_time_ms: float
    cache_hit: bool = False
    parallel_executions: int = 1

    # Quality Metrics
    average_confidence: float = 0.0
    best_match_confidence: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "parallel_discovery"


@dataclass
class ParallelProcessingResult:
    """Ergebnis von Parallel Processing."""
    workflow_id: str
    stage: ProcessingStage

    # Results
    success: bool
    results: list[Any]

    # Performance Metrics
    total_time_ms: float
    parallel_time_ms: float
    sequential_time_ms: float

    # Optional fields with defaults
    errors: list[str] = field(default_factory=list)
    speedup_factor: float = 1.0

    # Resource Usage
    max_concurrent_tasks: int = 1
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Quality Metrics
    success_rate: float = 1.0
    average_confidence: float = 0.0


@dataclass
class CacheEntry:
    """Cache-Eintrag für Performance-Optimierung."""
    key: str
    value: Any

    # Cache Metadata
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    # Quality Metrics
    confidence: float = 1.0
    freshness_score: float = 1.0

    # Cache Strategy
    strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE
    ttl_seconds: int = 300

    # Predictive Data
    predicted_usage: float = 0.0
    usage_pattern: str = "unknown"


# =============================================================================
# PARALLEL DISCOVERY INTERFACES
# =============================================================================

@runtime_checkable
class IParallelDiscovery(Protocol):
    """Interface für Parallel Discovery Operations."""

    async def discover_agents(
        self,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Führt parallele Agent Discovery durch."""
        ...

    async def discover_tools(
        self,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Führt parallele Tool Discovery durch."""
        ...

    async def discover_services(
        self,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Führt parallele Service Discovery durch."""
        ...

    async def discover_capabilities(
        self,
        request: DiscoveryRequest
    ) -> DiscoveryResult:
        """Führt parallele Capability Discovery durch."""
        ...

    async def discover_all(
        self,
        requests: list[DiscoveryRequest]
    ) -> list[DiscoveryResult]:
        """Führt mehrere Discovery-Operationen parallel durch."""
        ...


@runtime_checkable
class IVoicePerformanceOptimizer(Protocol):
    """Interface für Voice Performance Optimization."""

    async def optimize_speech_to_text(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert Speech-to-Text Processing."""
        ...

    async def optimize_agent_discovery(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert Agent Discovery mit Parallel Processing."""
        ...

    async def optimize_agent_execution(
        self,
        context: VoiceWorkflowContext,
        agents: list[str]
    ) -> ParallelProcessingResult:
        """Optimiert Agent Execution mit Parallel Processing."""
        ...

    async def optimize_response_generation(
        self,
        context: VoiceWorkflowContext,
        results: list[Any]
    ) -> ParallelProcessingResult:
        """Optimiert Response Generation."""
        ...

    async def optimize_full_pipeline(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Optimiert komplette Voice Pipeline."""
        ...


@runtime_checkable
class IPerformanceCache(Protocol):
    """Interface für Performance Caching."""

    async def get(self, key: str) -> CacheEntry | None:
        """Holt Cache-Eintrag."""
        ...

    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 300,
        strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE
    ) -> None:
        """Speichert Cache-Eintrag."""
        ...

    async def invalidate(self, key: str) -> None:
        """Invalidiert Cache-Eintrag."""
        ...

    async def warm_up(
        self,
        context: VoiceWorkflowContext
    ) -> None:
        """Führt Cache Warm-up durch."""
        ...

    async def predict_and_preload(
        self,
        context: VoiceWorkflowContext
    ) -> None:
        """Führt Predictive Preloading durch."""
        ...

    async def get_statistics(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        ...


@runtime_checkable
class IVoiceWorkflowOrchestrator(Protocol):
    """Interface für optimierte Voice Workflow Orchestration."""

    async def process_voice_input(
        self,
        context: VoiceWorkflowContext
    ) -> ParallelProcessingResult:
        """Verarbeitet Voice Input mit Performance-Optimierung."""
        ...

    async def execute_parallel_pipeline(
        self,
        context: VoiceWorkflowContext,
        stages: list[ProcessingStage]
    ) -> list[ParallelProcessingResult]:
        """Führt Pipeline-Stages parallel aus."""
        ...

    async def handle_concurrent_requests(
        self,
        contexts: list[VoiceWorkflowContext]
    ) -> list[ParallelProcessingResult]:
        """Behandelt mehrere Voice Requests concurrent."""
        ...

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Gibt Performance-Metriken zurück."""
        ...


# =============================================================================
# CONFIGURATION INTERFACES
# =============================================================================

@dataclass
class VoicePerformanceSettings:
    """Voice Performance Optimization Konfiguration."""

    # Parallel Processing
    enabled: bool = True
    max_concurrent_discoveries: int = 10
    max_concurrent_agents: int = 5
    max_concurrent_tools: int = 8

    # Discovery Timeouts
    agent_discovery_timeout_seconds: float = 3.0
    tool_discovery_timeout_seconds: float = 2.0
    service_discovery_timeout_seconds: float = 5.0
    capability_discovery_timeout_seconds: float = 1.0

    # Performance Targets
    target_latency_ms: int = 500
    max_latency_ms: int = 2000
    target_throughput_rps: int = 100

    # Caching Configuration
    cache_enabled: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE_CACHE
    cache_ttl_seconds: int = 300
    cache_max_size: int = 10000

    # Predictive Features
    predictive_loading_enabled: bool = True
    warm_up_enabled: bool = True
    pattern_learning_enabled: bool = True

    # Resource Limits
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: float = 80.0
    max_concurrent_workflows: int = 50

    # Quality Settings
    min_discovery_confidence: float = 0.5
    min_agent_confidence: float = 0.7
    max_discovery_results: int = 20

    # Monitoring
    monitoring_enabled: bool = True
    metrics_collection_enabled: bool = True
    performance_alerts_enabled: bool = True

    # Circuit Breaker Integration
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 30

    # Rate Limiting Integration
    rate_limiting_enabled: bool = True
    rate_limiting_coordination: bool = True
