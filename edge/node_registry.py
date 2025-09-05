"""Edge Node Registry für Keiko Personal Assistant.

Dieses Modul implementiert die Edge-Node-Registry für die Verwaltung
von Edge-Computing-Nodes im verteilten System.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
from .edge_types import (
    EdgeConfiguration,
    EdgeNodeInfo,
    EdgeNodeStatus,
    EdgeNodeType,
    EdgeProcessingCapability,
)

logger = get_logger(__name__)


@dataclass
class NodeHealthMetrics:
    """Gesundheitsmetriken für Edge-Nodes."""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    active_tasks: int = 0
    last_heartbeat: datetime | None = None
    uptime_seconds: int = 0


@dataclass
class RegisteredNode:
    """Registrierter Edge-Node mit erweiterten Informationen."""
    node_info: EdgeNodeInfo
    health_metrics: NodeHealthMetrics = field(default_factory=NodeHealthMetrics)
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    capabilities: set[EdgeProcessingCapability] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


class EdgeNodeRegistry:
    """Enterprise Edge-Node-Registry für intelligente Node-Verwaltung.

    Implementiert umfassende Node-Verwaltung mit:
    - Automatische Node-Discovery und Registrierung
    - Health-Monitoring und Heartbeat-Verwaltung
    - Capability-basierte Node-Auswahl
    - Load-Balancing und Failover-Unterstützung
    - Performance-Tracking und Metriken
    """

    def __init__(self, config: EdgeConfiguration | None = None):
        """Initialisiert die Edge-Node-Registry.

        Args:
            config: Edge-Konfiguration
        """
        self.config = config or EdgeConfiguration()
        self._nodes: dict[str, RegisteredNode] = {}
        self._node_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task | None = None
        self._running = False

        # Konfiguration
        self.heartbeat_timeout = timedelta(seconds=self.config.node_timeout_seconds)
        self.heartbeat_interval = timedelta(seconds=self.config.node_heartbeat_interval_seconds)

        logger.info("Edge-Node-Registry initialisiert")

    async def start(self) -> None:
        """Startet die Registry und Hintergrund-Tasks."""
        if self._running:
            logger.warning("Edge-Node-Registry bereits gestartet")
            return

        self._running = True

        # Heartbeat-Monitoring starten
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        logger.info("Edge-Node-Registry gestartet")

    async def stop(self) -> None:
        """Stoppt die Registry und alle Hintergrund-Tasks."""
        if not self._running:
            return

        self._running = False

        # Heartbeat-Task stoppen
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        logger.info("Edge-Node-Registry gestoppt")

    async def register_node(
        self,
        node_id: str,
        node_info: EdgeNodeInfo,
        capabilities: set[EdgeProcessingCapability] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Registriert einen neuen Edge-Node.

        Args:
            node_id: Eindeutige Node-ID
            node_info: Node-Informationen
            capabilities: Node-Capabilities
            metadata: Zusätzliche Metadaten

        Returns:
            True wenn erfolgreich registriert
        """
        try:
            async with self._node_lock:
                # Node erstellen
                registered_node = RegisteredNode(
                    node_info=node_info,
                    capabilities=capabilities or set(),
                    metadata=metadata or {}
                )

                # Heartbeat initialisieren
                registered_node.health_metrics.last_heartbeat = datetime.now(UTC)

                # Node speichern
                self._nodes[node_id] = registered_node

                logger.info(
                    f"Edge-Node registriert: {node_id} "
                    f"(Typ: {node_info.node_type}, Region: {node_info.region})"
                )

                return True

        except Exception as e:
            logger.error(f"Fehler beim Registrieren von Node {node_id}: {e}")
            return False

    async def unregister_node(self, node_id: str) -> bool:
        """Entfernt einen Edge-Node aus der Registry.

        Args:
            node_id: Node-ID

        Returns:
            True wenn erfolgreich entfernt
        """
        try:
            async with self._node_lock:
                if node_id in self._nodes:
                    del self._nodes[node_id]
                    logger.info(f"Edge-Node entfernt: {node_id}")
                    return True
                logger.warning(f"Node {node_id} nicht in Registry gefunden")
                return False

        except Exception as e:
            logger.error(f"Fehler beim Entfernen von Node {node_id}: {e}")
            return False

    async def update_heartbeat(
        self,
        node_id: str,
        health_metrics: NodeHealthMetrics | None = None
    ) -> bool:
        """Aktualisiert Heartbeat für einen Node.

        Args:
            node_id: Node-ID
            health_metrics: Aktuelle Gesundheitsmetriken

        Returns:
            True wenn erfolgreich aktualisiert
        """
        try:
            async with self._node_lock:
                if node_id not in self._nodes:
                    logger.warning(f"Heartbeat für unbekannten Node: {node_id}")
                    return False

                node = self._nodes[node_id]
                node.health_metrics.last_heartbeat = datetime.now(UTC)
                node.last_updated = datetime.now(UTC)

                # Gesundheitsmetriken aktualisieren
                if health_metrics:
                    node.health_metrics = health_metrics
                    node.health_metrics.last_heartbeat = datetime.now(UTC)

                # Node-Status aktualisieren
                node.node_info.status = EdgeNodeStatus.HEALTHY

                return True

        except Exception as e:
            logger.error(f"Fehler beim Heartbeat-Update für Node {node_id}: {e}")
            return False

    async def get_healthy_nodes(
        self,
        node_type: EdgeNodeType | None = None,
        capabilities: set[EdgeProcessingCapability] | None = None,
        region: str | None = None
    ) -> list[EdgeNodeInfo]:
        """Gibt Liste gesunder Nodes zurück.

        Args:
            node_type: Gewünschter Node-Typ (optional)
            capabilities: Erforderliche Capabilities (optional)
            region: Gewünschte Region (optional)

        Returns:
            Liste gesunder Edge-Nodes
        """
        healthy_nodes = []
        current_time = datetime.now(UTC)

        async with self._node_lock:
            for node_id, registered_node in self._nodes.items():
                # Heartbeat-Timeout prüfen
                if not self._is_node_healthy(registered_node, current_time):
                    continue

                # Filter anwenden
                node_info = registered_node.node_info

                if node_type and node_info.node_type != node_type:
                    continue

                if region and node_info.region != region:
                    continue

                if capabilities and not capabilities.issubset(registered_node.capabilities):
                    continue

                healthy_nodes.append(node_info)

        return healthy_nodes

    async def get_node_by_capability(
        self,
        capability: EdgeProcessingCapability,
        region: str | None = None
    ) -> EdgeNodeInfo | None:
        """Findet besten Node für spezifische Capability.

        Args:
            capability: Gewünschte Capability
            region: Bevorzugte Region (optional)

        Returns:
            Bester verfügbarer Node oder None
        """
        candidates = await self.get_healthy_nodes(
            capabilities={capability},
            region=region
        )

        if not candidates:
            # Fallback: Suche ohne Region-Filter
            if region:
                candidates = await self.get_healthy_nodes(capabilities={capability})

        if not candidates:
            return None

        # Besten Kandidaten auswählen (niedrigste CPU-Auslastung)
        best_node = None
        best_score = float("inf")

        async with self._node_lock:
            for candidate in candidates:
                registered_node = self._nodes.get(candidate.node_id)
                if registered_node:
                    # Score basierend auf CPU-Auslastung und aktiven Tasks
                    score = (
                        registered_node.health_metrics.cpu_usage_percent +
                        registered_node.health_metrics.active_tasks * 10
                    )

                    if score < best_score:
                        best_score = score
                        best_node = candidate

        return best_node

    def _is_node_healthy(self, node: RegisteredNode, current_time: datetime) -> bool:
        """Prüft ob ein Node gesund ist.

        Args:
            node: Registrierter Node
            current_time: Aktuelle Zeit

        Returns:
            True wenn Node gesund ist
        """
        if not node.health_metrics.last_heartbeat:
            return False

        time_since_heartbeat = current_time - node.health_metrics.last_heartbeat
        return time_since_heartbeat < self.heartbeat_timeout

    async def _heartbeat_monitor(self) -> None:
        """Überwacht Heartbeats und markiert ungesunde Nodes."""
        while self._running:
            try:
                current_time = datetime.now(UTC)
                unhealthy_nodes = []

                async with self._node_lock:
                    for node_id, registered_node in self._nodes.items():
                        if not self._is_node_healthy(registered_node, current_time):
                            registered_node.node_info.status = EdgeNodeStatus.UNHEALTHY
                            unhealthy_nodes.append(node_id)

                # Ungesunde Nodes loggen
                for node_id in unhealthy_nodes:
                    logger.warning(f"Node {node_id} als ungesund markiert (Heartbeat-Timeout)")

                # Warten bis zum nächsten Check
                await asyncio.sleep(self.heartbeat_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im Heartbeat-Monitor: {e}")
                await asyncio.sleep(5)  # Kurze Pause bei Fehlern

    async def get_registry_status(self) -> dict[str, Any]:
        """Gibt Registry-Status zurück.

        Returns:
            Dictionary mit Registry-Statistiken
        """
        async with self._node_lock:
            total_nodes = len(self._nodes)
            healthy_nodes = len(await self.get_healthy_nodes())

            node_types = {}
            regions = set()

            for registered_node in self._nodes.values():
                node_type = registered_node.node_info.node_type.value
                node_types[node_type] = node_types.get(node_type, 0) + 1
                regions.add(registered_node.node_info.region)

        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "unhealthy_nodes": total_nodes - healthy_nodes,
            "node_types": node_types,
            "regions": list(regions),
            "heartbeat_timeout_seconds": self.heartbeat_timeout.total_seconds(),
            "running": self._running
        }


def create_node_registry(config: EdgeConfiguration | None = None) -> EdgeNodeRegistry:
    """Factory-Funktion für Edge-Node-Registry.

    Args:
        config: Edge-Konfiguration

    Returns:
        Neue EdgeNodeRegistry-Instanz
    """
    return EdgeNodeRegistry(config)
