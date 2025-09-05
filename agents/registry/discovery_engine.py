# backend/kei_agents/registry/discovery_engine.py
"""Advanced Agent Discovery Engine für Keiko Personal Assistant

Implementiert Service Discovery, Health-based Discovery, Capability-based Discovery,
Geographic Discovery und Load-based Discovery.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import (
    get_logger,
    with_log_links,
)

from .enhanced_models import AgentStatus, AgentVersionMetadata

logger = get_logger(__name__)


class DiscoveryStrategy(str, Enum):
    """Discovery-Strategien."""

    CAPABILITY_BASED = "capability_based"
    HEALTH_BASED = "health_based"
    LOAD_BASED = "load_based"
    GEOGRAPHIC = "geographic"
    HYBRID = "hybrid"


class LoadBalancingStrategy(str, Enum):
    """Load-Balancing-Strategien."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"


@dataclass
class GeographicLocation:
    """Geografische Location für Agent-Deployment."""

    region: str
    zone: str
    latitude: float | None = None
    longitude: float | None = None

    def distance_to(self, other: GeographicLocation) -> float:
        """Berechnet Distanz zu anderer Location.

        Args:
            other: Andere Location

        Returns:
            Distanz in Kilometern (oder 0 wenn Koordinaten fehlen)
        """
        if (
            self.latitude is None
            or self.longitude is None
            or other.latitude is None
            or other.longitude is None
        ):
            # Fallback: Region/Zone-basierte Distanz
            if self.region == other.region:
                return 0 if self.zone == other.zone else 100
            return 1000

        # Haversine-Formel für Distanz-Berechnung
        r = 6371  # Erdradius in km

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return r * c


@dataclass
class AgentInstance:
    """Agent-Instanz mit Runtime-Informationen."""

    instance_id: str
    agent_metadata: AgentVersionMetadata

    # Runtime-Status
    status: AgentStatus = AgentStatus.AVAILABLE
    health_score: float = 1.0  # 0.0 - 1.0
    load_factor: float = 0.0  # 0.0 - 1.0
    response_time_ms: float = 0.0

    # Geographic Information
    location: GeographicLocation | None = None

    # Connection Information
    endpoint_url: str | None = None
    connection_count: int = 0
    max_connections: int = 100

    # Performance Metrics
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0

    # Timestamps
    last_health_check: datetime = field(default_factory=lambda: datetime.now(UTC))
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_request_at: datetime | None = None

    def __post_init__(self):
        """Validiert Agent-Instanz."""
        if not self.instance_id:
            self.instance_id = str(uuid.uuid4())

        if not 0.0 <= self.health_score <= 1.0:
            raise ValueError("health_score muss zwischen 0.0 und 1.0 liegen")

        if not 0.0 <= self.load_factor <= 1.0:
            raise ValueError("load_factor muss zwischen 0.0 und 1.0 liegen")

    def is_healthy(self) -> bool:
        """Prüft ob Instanz gesund ist."""
        return (
            self.status == AgentStatus.AVAILABLE
            and self.health_score >= 0.7
            and self.load_factor < 0.9
            and self.error_rate < 0.1
        )

    def is_overloaded(self) -> bool:
        """Prüft ob Instanz überlastet ist."""
        return (
            self.load_factor > 0.8
            or self.connection_count >= self.max_connections
            or self.error_rate > 0.2
        )

    def calculate_score(
        self, strategy: DiscoveryStrategy, client_location: GeographicLocation | None = None
    ) -> float:
        """Berechnet Score für Discovery-Strategie.

        Args:
            strategy: Discovery-Strategie
            client_location: Client-Location für Geographic Discovery

        Returns:
            Score zwischen 0.0 und 1.0
        """
        if not self.is_healthy():
            return 0.0

        if strategy == DiscoveryStrategy.HEALTH_BASED:
            return self.health_score * (1.0 - self.error_rate)

        if strategy == DiscoveryStrategy.LOAD_BASED:
            return 1.0 - self.load_factor

        if strategy == DiscoveryStrategy.GEOGRAPHIC:
            if not client_location or not self.location:
                return 0.5  # Neutral score wenn Location fehlt

            distance = self.location.distance_to(client_location)
            # Normalisiere Distanz (max 10000km = 0.0 score)
            distance_score = max(0.0, 1.0 - (distance / 10000.0))
            return distance_score * self.health_score

        if strategy == DiscoveryStrategy.CAPABILITY_BASED:
            # Basis-Score basierend auf Health und Load
            return self.health_score * (1.0 - self.load_factor * 0.5)

        if strategy == DiscoveryStrategy.HYBRID:
            # Gewichtete Kombination aller Faktoren
            health_weight = 0.4
            load_weight = 0.3
            response_weight = 0.2
            location_weight = 0.1

            health_score = self.health_score * (1.0 - self.error_rate)
            load_score = 1.0 - self.load_factor
            response_score = max(0.0, 1.0 - (self.response_time_ms / 1000.0))

            location_score = 0.5
            if client_location and self.location:
                distance = self.location.distance_to(client_location)
                location_score = max(0.0, 1.0 - (distance / 10000.0))

            return (
                health_score * health_weight
                + load_score * load_weight
                + response_score * response_weight
                + location_score * location_weight
            )

        return 0.5  # Default score


@dataclass
class DiscoveryQuery:
    """Query für Agent-Discovery."""

    agent_id: str | None = None
    version_constraint: str | None = None
    capabilities: list[str] = field(default_factory=list)
    tenant_id: str | None = None

    # Discovery-Konfiguration
    strategy: DiscoveryStrategy = DiscoveryStrategy.HYBRID
    max_results: int = 10
    min_health_score: float = 0.7
    max_load_factor: float = 0.9

    # Geographic Filtering
    client_location: GeographicLocation | None = None
    max_distance_km: float | None = None
    preferred_regions: list[str] = field(default_factory=list)

    # Load Balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN

    # Filtering
    exclude_instances: set[str] = field(default_factory=set)
    require_endpoints: bool = False


@dataclass
class DiscoveryResult:
    """Ergebnis einer Agent-Discovery."""

    instance: AgentInstance
    score: float
    match_reasons: list[str] = field(default_factory=list)
    distance_km: float | None = None

    def __post_init__(self):
        """Validiert Discovery-Result."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score muss zwischen 0.0 und 1.0 liegen")


class AdvancedDiscoveryEngine:
    """Erweiterte Discovery Engine für Agent-Suche."""

    def __init__(self):
        """Initialisiert Discovery Engine."""
        self._agent_instances: dict[str, list[AgentInstance]] = {}  # agent_id -> instances
        self._instance_registry: dict[str, AgentInstance] = {}  # instance_id -> instance
        self._round_robin_counters: dict[str, int] = {}

        # Health Check Configuration
        self._health_check_interval = 30  # Sekunden
        self._health_check_timeout = 10  # Sekunden
        self._last_health_check = datetime.now(UTC)

    @with_log_links(component="discovery_engine", operation="register_instance")
    def register_agent_instance(self, instance: AgentInstance) -> None:
        """Registriert Agent-Instanz.

        Args:
            instance: Agent-Instanz
        """
        agent_id = instance.agent_metadata.agent_id

        # Füge zu Agent-Instanzen hinzu
        if agent_id not in self._agent_instances:
            self._agent_instances[agent_id] = []

        self._agent_instances[agent_id].append(instance)
        self._instance_registry[instance.instance_id] = instance

        logger.info(
            f"Agent-Instanz registriert: {instance.instance_id} für {agent_id}",
            extra={
                "instance_id": instance.instance_id,
                "agent_id": agent_id,
                "agent_version": str(instance.agent_metadata.version),
                "location": (
                    f"{instance.location.region}/{instance.location.zone}"
                    if instance.location
                    else None
                ),
                "endpoint": instance.endpoint_url,
            },
        )

    def unregister_agent_instance(self, instance_id: str) -> None:
        """Entfernt Agent-Instanz.

        Args:
            instance_id: Instanz-ID
        """
        if instance_id not in self._instance_registry:
            return

        instance = self._instance_registry[instance_id]
        agent_id = instance.agent_metadata.agent_id

        # Entferne aus Agent-Instanzen
        if agent_id in self._agent_instances:
            self._agent_instances[agent_id] = [
                inst for inst in self._agent_instances[agent_id] if inst.instance_id != instance_id
            ]

            # Entferne Agent komplett wenn keine Instanzen mehr
            if not self._agent_instances[agent_id]:
                del self._agent_instances[agent_id]

        # Entferne aus Instance Registry
        del self._instance_registry[instance_id]

        logger.info(
            f"Agent-Instanz entfernt: {instance_id} für {agent_id}",
            extra={"instance_id": instance_id, "agent_id": agent_id},
        )

    @with_log_links(component="discovery_engine", operation="discover_agents")
    async def discover_agents(self, query: DiscoveryQuery) -> list[DiscoveryResult]:
        """Führt Agent-Discovery durch.

        Args:
            query: Discovery-Query

        Returns:
            Liste von Discovery-Ergebnissen
        """
        # Sammle passende Instanzen
        candidate_instances = await self._collect_candidate_instances(query)

        if not candidate_instances:
            logger.info(
                "Keine passenden Agent-Instanzen gefunden",
                extra={
                    "query_agent_id": query.agent_id,
                    "query_capabilities": query.capabilities,
                    "query_tenant_id": query.tenant_id,
                },
            )
            return []

        # Berechne Scores und erstelle Results
        results = []
        for instance in candidate_instances:
            score = instance.calculate_score(query.strategy, query.client_location)

            # Prüfe Mindest-Score
            if score < 0.1:  # Sehr niedrige Scores ausschließen
                continue

            # Berechne Distanz falls Geographic Query
            distance_km = None
            if query.client_location and instance.location:
                distance_km = instance.location.distance_to(query.client_location)

                # Prüfe Max-Distanz
                if query.max_distance_km and distance_km > query.max_distance_km:
                    continue

            # Erstelle Match-Reasons
            match_reasons = self._generate_match_reasons(instance, query)

            results.append(
                DiscoveryResult(
                    instance=instance,
                    score=score,
                    match_reasons=match_reasons,
                    distance_km=distance_km,
                )
            )

        # Sortiere nach Score
        results.sort(key=lambda x: x.score, reverse=True)

        # Wende Load-Balancing an
        results = self._apply_load_balancing(results, query)

        # Limitiere Ergebnisse
        results = results[: query.max_results]

        logger.info(
            f"Agent-Discovery abgeschlossen: {len(results)} Ergebnisse",
            extra={
                "query_agent_id": query.agent_id,
                "strategy": query.strategy.value,
                "candidates_found": len(candidate_instances),
                "results_returned": len(results),
                "top_score": results[0].score if results else 0.0,
            },
        )

        return results

    async def _collect_candidate_instances(self, query: DiscoveryQuery) -> list[AgentInstance]:
        """Sammelt Kandidaten-Instanzen basierend auf Query.

        Args:
            query: Discovery-Query

        Returns:
            Liste von Kandidaten-Instanzen
        """
        candidates = []

        # Bestimme relevante Agent-IDs
        if query.agent_id:
            target_agent_ids = [query.agent_id]
        else:
            target_agent_ids = list(self._agent_instances.keys())

        for agent_id in target_agent_ids:
            if agent_id not in self._agent_instances:
                continue

            for instance in self._agent_instances[agent_id]:
                # Basis-Filterung
                if not self._passes_basic_filters(instance, query):
                    continue

                # Capability-Filterung
                if query.capabilities and not self._matches_capabilities(
                    instance, query.capabilities
                ):
                    continue

                # Tenant-Filterung
                if query.tenant_id and not instance.agent_metadata.is_accessible_by_tenant(
                    query.tenant_id
                ):
                    continue

                # Geographic Filterung
                if query.preferred_regions and instance.location:
                    if instance.location.region not in query.preferred_regions:
                        continue

                candidates.append(instance)

        return candidates

    def _passes_basic_filters(self, instance: AgentInstance, query: DiscoveryQuery) -> bool:
        """Prüft Basis-Filter.

        Args:
            instance: Agent-Instanz
            query: Discovery-Query

        Returns:
            True wenn Filter bestanden
        """
        # Exclude-Liste
        if instance.instance_id in query.exclude_instances:
            return False

        # Health-Filter
        if instance.health_score < query.min_health_score:
            return False

        # Load-Filter
        if instance.load_factor > query.max_load_factor:
            return False

        # Endpoint-Requirement
        if query.require_endpoints and not instance.endpoint_url:
            return False

        # Status-Filter
        if instance.status not in [AgentStatus.AVAILABLE, AgentStatus.CANARY]:
            return False

        return True

    def _matches_capabilities(
        self, instance: AgentInstance, required_capabilities: list[str]
    ) -> bool:
        """Prüft Capability-Match.

        Args:
            instance: Agent-Instanz
            required_capabilities: Erforderliche Capabilities

        Returns:
            True wenn alle Capabilities vorhanden
        """
        agent_capabilities = set(cap.lower() for cap in instance.agent_metadata.capabilities)
        required_caps = set(cap.lower() for cap in required_capabilities)

        return required_caps.issubset(agent_capabilities)

    def _generate_match_reasons(self, instance: AgentInstance, query: DiscoveryQuery) -> list[str]:
        """Generiert Match-Reasons für Ergebnis.

        Args:
            instance: Agent-Instanz
            query: Discovery-Query

        Returns:
            Liste von Match-Reasons
        """
        reasons = []

        if instance.health_score >= 0.9:
            reasons.append("Excellent health score")
        elif instance.health_score >= 0.8:
            reasons.append("Good health score")

        if instance.load_factor <= 0.3:
            reasons.append("Low load")
        elif instance.load_factor <= 0.6:
            reasons.append("Moderate load")

        if instance.response_time_ms <= 100:
            reasons.append("Fast response time")
        elif instance.response_time_ms <= 500:
            reasons.append("Good response time")

        if query.capabilities:
            matched_caps = [
                cap
                for cap in query.capabilities
                if cap.lower() in [c.lower() for c in instance.agent_metadata.capabilities]
            ]
            if matched_caps:
                reasons.append(f"Matches capabilities: {', '.join(matched_caps)}")

        if query.client_location and instance.location:
            distance = instance.location.distance_to(query.client_location)
            if distance <= 100:
                reasons.append("Very close location")
            elif distance <= 1000:
                reasons.append("Close location")

        return reasons

    def _apply_load_balancing(
        self, results: list[DiscoveryResult], query: DiscoveryQuery
    ) -> list[DiscoveryResult]:
        """Wendet Load-Balancing-Strategie an.

        Args:
            results: Discovery-Ergebnisse
            query: Discovery-Query

        Returns:
            Load-balanced Ergebnisse
        """
        if not results or query.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            import random

            random.shuffle(results)
            return results

        if query.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Einfaches Round-Robin basierend auf Agent-ID
            agent_id = query.agent_id or "global"
            counter = self._round_robin_counters.get(agent_id, 0)

            if results:
                selected_index = counter % len(results)
                selected = results[selected_index]
                remaining = results[:selected_index] + results[selected_index + 1 :]
                results = [selected] + remaining

                self._round_robin_counters[agent_id] = counter + 1

        elif query.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Sortiere nach Connection-Count
            results.sort(key=lambda x: x.instance.connection_count)

        elif query.load_balancing_strategy == LoadBalancingStrategy.RESPONSE_TIME:
            # Sortiere nach Response-Time
            results.sort(key=lambda x: x.instance.response_time_ms)

        elif query.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Gewichtetes Round-Robin basierend auf Score
            # Höhere Scores bekommen höhere Wahrscheinlichkeit
            total_weight = sum(result.score for result in results)
            if total_weight > 0:
                import random

                rand_val = random.uniform(0, total_weight)
                cumulative_weight = 0

                for i, result in enumerate(results):
                    cumulative_weight += result.score
                    if rand_val <= cumulative_weight:
                        # Bewege ausgewähltes Element nach vorne
                        selected = results.pop(i)
                        results.insert(0, selected)
                        break

        return results

    async def update_instance_metrics(self, instance_id: str, metrics: dict[str, Any]) -> None:
        """Aktualisiert Instanz-Metriken.

        Args:
            instance_id: Instanz-ID
            metrics: Neue Metriken
        """
        if instance_id not in self._instance_registry:
            return

        instance = self._instance_registry[instance_id]

        # Aktualisiere Metriken
        if "health_score" in metrics:
            instance.health_score = max(0.0, min(1.0, metrics["health_score"]))

        if "load_factor" in metrics:
            instance.load_factor = max(0.0, min(1.0, metrics["load_factor"]))

        if "response_time_ms" in metrics:
            instance.response_time_ms = max(0.0, metrics["response_time_ms"])

        if "connection_count" in metrics:
            instance.connection_count = max(0, metrics["connection_count"])

        if "requests_per_second" in metrics:
            instance.requests_per_second = max(0.0, metrics["requests_per_second"])

        if "error_rate" in metrics:
            instance.error_rate = max(0.0, min(1.0, metrics["error_rate"]))

        if "uptime_percentage" in metrics:
            instance.uptime_percentage = max(0.0, min(100.0, metrics["uptime_percentage"]))

        instance.last_health_check = datetime.now(UTC)

        logger.debug(
            f"Instanz-Metriken aktualisiert: {instance_id}",
            extra={
                "instance_id": instance_id,
                "health_score": instance.health_score,
                "load_factor": instance.load_factor,
                "response_time_ms": instance.response_time_ms,
            },
        )

    def get_discovery_statistics(self) -> dict[str, Any]:
        """Holt Discovery-Statistiken.

        Returns:
            Statistiken-Dictionary
        """
        total_instances = len(self._instance_registry)
        healthy_instances = sum(1 for inst in self._instance_registry.values() if inst.is_healthy())
        overloaded_instances = sum(
            1 for inst in self._instance_registry.values() if inst.is_overloaded()
        )

        # Durchschnittliche Metriken
        if total_instances > 0:
            avg_health = (
                sum(inst.health_score for inst in self._instance_registry.values())
                / total_instances
            )
            avg_load = (
                sum(inst.load_factor for inst in self._instance_registry.values()) / total_instances
            )
            avg_response_time = (
                sum(inst.response_time_ms for inst in self._instance_registry.values())
                / total_instances
            )
        else:
            avg_health = avg_load = avg_response_time = 0.0

        return {
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "overloaded_instances": overloaded_instances,
            "health_percentage": (
                (healthy_instances / total_instances * 100) if total_instances > 0 else 0
            ),
            "average_health_score": avg_health,
            "average_load_factor": avg_load,
            "average_response_time_ms": avg_response_time,
            "total_agents": len(self._agent_instances),
            "last_health_check": self._last_health_check.isoformat(),
        }


# Globale Discovery Engine Instanz
discovery_engine = AdvancedDiscoveryEngine()
