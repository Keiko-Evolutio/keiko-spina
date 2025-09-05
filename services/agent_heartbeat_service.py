"""Agent Heartbeat Service.

Überwacht die Erreichbarkeit von registrierten Agents durch regelmäßige Heartbeat-Checks.
Entfernt Agents automatisch aus der Registry, wenn sie nicht mehr erreichbar sind.
"""

import asyncio
import time
from typing import Any

import httpx

# Optional import - Registry-System ist nicht immer verfügbar
try:
    from agents.capabilities import get_capability_manager
    from agents.capabilities.dynamic_registry import dynamic_registry  # type: ignore
except Exception:  # pragma: no cover - Fallback für Testumgebung
    dynamic_registry = None  # type: ignore
    get_capability_manager = None  # type: ignore

from kei_logging import get_logger
from services.core.base_service import MonitoringService
from services.core.constants import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_MAX_HEARTBEAT_FAILURES,
    HTTP_STATUS_OK,
    SERVICE_STATUS_AVAILABLE,
)

logger = get_logger(__name__)


class AgentHeartbeatService(MonitoringService):
    """Service für Agent-Heartbeat-Monitoring."""

    def __init__(
        self,
        check_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        max_failures: int = DEFAULT_MAX_HEARTBEAT_FAILURES
    ):
        """Initialisiert den Heartbeat-Service.

        Args:
            check_interval: Intervall zwischen Checks in Sekunden
            timeout: Timeout für Heartbeat-Requests in Sekunden
            max_failures: Maximale Anzahl fehlgeschlagener Checks vor Entfernung
        """
        super().__init__(
            service_name="AgentHeartbeatService",
            interval_seconds=check_interval,
            timeout_seconds=timeout,
            max_failures=max_failures
        )

    async def _initialize(self) -> None:
        """Service-spezifische Initialisierung."""
        logger.debug("Agent Heartbeat Service initialisiert")

    async def _cleanup(self) -> None:
        """Service-spezifische Bereinigung."""
        self.failure_counts.clear()
        logger.debug("Agent Heartbeat Service bereinigt")

    async def _perform_periodic_task(self) -> None:
        """Führt periodische Heartbeat-Checks durch."""
        await self._check_all_agents()

    async def _check_all_agents(self):
        """Prüft alle registrierten Agents."""
        agents_to_check = []

        # Sammle alle Agents mit Heartbeat-URL
        for agent_id, agent in get_capability_manager()._agent_capabilities.items():
            if hasattr(agent, "heartbeat_url") and agent.heartbeat_url:
                agents_to_check.append((agent_id, agent))

        if not agents_to_check:
            return

        logger.debug(f"Prüfe {len(agents_to_check)} Agents")

        # Führe Heartbeat-Checks parallel aus
        tasks = [
            self._check_agent_heartbeat(agent_id, agent)
            for agent_id, agent in agents_to_check
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_agent_heartbeat(self, agent_id: str, agent: Any) -> None:
        """Prüft einen einzelnen Agent.

        Args:
            agent_id: Agent-ID
            agent: Agent-Objekt
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(agent.heartbeat_url)

                if response.status_code == HTTP_STATUS_OK:
                    await self._handle_heartbeat_success(agent_id, agent)
                else:
                    await self._handle_monitoring_failure(agent_id, f"HTTP {response.status_code}")

        except Exception as e:
            await self._handle_monitoring_failure(agent_id, str(e))

    async def _handle_heartbeat_success(self, agent_id: str, agent: Any) -> None:
        """Behandelt erfolgreiche Heartbeats.

        Args:
            agent_id: Agent-ID
            agent: Agent-Objekt
        """
        # Reset Failure-Counter über Basis-Klasse
        self._reset_failure_count(agent_id)

        # Update last_heartbeat
        if hasattr(agent, "last_heartbeat"):
            agent.last_heartbeat = time.time()

        # Status auf available setzen
        if hasattr(agent, "status"):
            agent.status = SERVICE_STATUS_AVAILABLE

        logger.debug(f"💓 Heartbeat OK: {agent_id}")

    async def _escalate_failure(self, target: str) -> None:
        """Eskaliert anhaltende Agent-Fehler durch Entfernung."""
        await self._remove_agent(target)

    async def _remove_agent(self, agent_id: str):
        """Entfernt einen Agent aus der Registry.

        Args:
            agent_id: Agent-ID
        """
        try:
            if agent_id in get_capability_manager()._agent_capabilities:
                agent = get_capability_manager()._agent_capabilities[agent_id]
                del get_capability_manager()._agent_capabilities[agent_id]

                logger.info(
                    f"🗑️ Agent entfernt (Heartbeat ausgefallen): {agent_id}",
                    extra={
                        "agent_id": agent_id,
                        "agent_name": getattr(agent, "name", "Unknown"),
                        "failure_count": self.failure_counts.get(agent_id, 0),
                        "event_type": "agent_removed_heartbeat_failure"
                    }
                )

            # Entferne aus Failure-Counter
            if agent_id in self.failure_counts:
                del self.failure_counts[agent_id]

        except Exception as e:
            logger.exception(f"Fehler beim Entfernen von Agent {agent_id}: {e}")

    def get_status(self) -> dict:
        """Gibt den Status des Heartbeat-Services zurück."""
        return {
            "running": self.running,
            "check_interval": self.interval_seconds,
            "timeout": self.timeout_seconds,
            "max_failures": self.max_failures,
            "monitored_agents": len([
                agent_id for agent_id, agent in get_capability_manager()._agent_capabilities.items()
                if hasattr(agent, "heartbeat_url") and agent.heartbeat_url
            ]),
            "agents_with_failures": len(self.failure_counts),
            "failure_counts": dict(self.failure_counts)
        }


# Globale Service-Instanz
heartbeat_service = AgentHeartbeatService()


async def start_heartbeat_service():
    """Startet den globalen Heartbeat-Service."""
    await heartbeat_service.start()


async def stop_heartbeat_service():
    """Stoppt den globalen Heartbeat-Service."""
    await heartbeat_service.stop()


def get_heartbeat_status() -> dict:
    """Gibt den Status des Heartbeat-Services zurück."""
    return heartbeat_service.get_status()
