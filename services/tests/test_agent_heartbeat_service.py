"""Unit-Tests für services.agent_heartbeat_service.

Testet den refactored AgentHeartbeatService mit neuer Basis-Klasse.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

# Mock die problematischen Module vor dem Import
sys.modules["agents.capabilities.get_capability_manager()"] = Mock()
sys.modules["kei_agents.registry"] = Mock()

# Jetzt können wir den Service importieren
from services.agent_heartbeat_service import AgentHeartbeatService
from services.core.constants import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_MAX_HEARTBEAT_FAILURES,
    HTTP_STATUS_OK,
    SERVICE_STATUS_AVAILABLE,
)


class TestAgentHeartbeatService:
    """Tests für AgentHeartbeatService."""

    def test_service_initialization(self):
        """Testet Service-Initialisierung mit Standard-Werten."""
        service = AgentHeartbeatService()

        assert service.service_name == "AgentHeartbeatService"
        assert service.interval_seconds == DEFAULT_HEARTBEAT_INTERVAL
        assert service.timeout_seconds == DEFAULT_HEARTBEAT_TIMEOUT
        assert service.max_failures == DEFAULT_MAX_HEARTBEAT_FAILURES
        assert not service.running

    def test_service_initialization_custom_values(self):
        """Testet Service-Initialisierung mit benutzerdefinierten Werten."""
        service = AgentHeartbeatService(
            check_interval=60.0,
            timeout=10.0,
            max_failures=5
        )

        assert service.interval_seconds == 60.0
        assert service.timeout_seconds == 10.0
        assert service.max_failures == 5

    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Testet Service-Lifecycle (Start/Stop)."""
        service = AgentHeartbeatService(check_interval=0.1)  # Kurzes Intervall für Tests

        # Start
        await service.start()
        assert service.running
        assert service.status == SERVICE_STATUS_AVAILABLE

        # Stop
        await service.stop()
        assert not service.running

    @pytest.mark.asyncio
    async def test_check_all_agents_empty_registry(self):
        """Testet _check_all_agents mit leerer Registry."""
        service = AgentHeartbeatService()

        with patch("services.agent_heartbeat_service.dynamic_registry") as mock_registry:
            mock_registry.agents = {}

            # Sollte keine Exception werfen
            await service._check_all_agents()

    @pytest.mark.asyncio
    async def test_check_all_agents_with_agents(self):
        """Testet _check_all_agents mit Agents in Registry."""
        service = AgentHeartbeatService()

        # Mock Agent
        mock_agent = MagicMock()
        mock_agent.heartbeat_url = "http://localhost:8080/health"

        with patch("services.agent_heartbeat_service.dynamic_registry") as mock_registry:
            mock_registry.agents = {"agent1": mock_agent}

            with patch.object(service, "_check_agent_heartbeat") as mock_check:
                await service._check_all_agents()
                mock_check.assert_called_once_with("agent1", mock_agent)

    @pytest.mark.asyncio
    async def test_check_agent_heartbeat_success(self):
        """Testet erfolgreichen Agent-Heartbeat-Check."""
        service = AgentHeartbeatService()

        mock_agent = MagicMock()
        mock_agent.heartbeat_url = "http://localhost:8080/health"

        # Mock HTTP Response
        mock_response = MagicMock()
        mock_response.status_code = HTTP_STATUS_OK

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with patch.object(service, "_handle_heartbeat_success") as mock_success:
                await service._check_agent_heartbeat("agent1", mock_agent)
                mock_success.assert_called_once_with("agent1", mock_agent)

    @pytest.mark.asyncio
    async def test_check_agent_heartbeat_http_error(self):
        """Testet Agent-Heartbeat-Check mit HTTP-Fehler."""
        service = AgentHeartbeatService()

        mock_agent = MagicMock()
        mock_agent.heartbeat_url = "http://localhost:8080/health"

        # Mock HTTP Response mit Fehler
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with patch.object(service, "_handle_monitoring_failure") as mock_failure:
                await service._check_agent_heartbeat("agent1", mock_agent)
                mock_failure.assert_called_once_with("agent1", "HTTP 500")

    @pytest.mark.asyncio
    async def test_check_agent_heartbeat_connection_error(self):
        """Testet Agent-Heartbeat-Check mit Verbindungsfehler."""
        service = AgentHeartbeatService()

        mock_agent = MagicMock()
        mock_agent.heartbeat_url = "http://localhost:8080/health"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with patch.object(service, "_handle_monitoring_failure") as mock_failure:
                await service._check_agent_heartbeat("agent1", mock_agent)
                mock_failure.assert_called_once_with("agent1", "Connection failed")

    @pytest.mark.asyncio
    async def test_handle_heartbeat_success(self):
        """Testet Behandlung erfolgreicher Heartbeats."""
        service = AgentHeartbeatService()

        # Setze Failure-Count
        service.failure_counts["agent1"] = 2

        mock_agent = MagicMock()
        mock_agent.last_heartbeat = None
        mock_agent.status = "unknown"

        await service._handle_heartbeat_success("agent1", mock_agent)

        # Failure-Count sollte zurückgesetzt sein
        assert "agent1" not in service.failure_counts

        # Agent-Attribute sollten aktualisiert sein
        assert mock_agent.last_heartbeat is not None
        assert mock_agent.status == SERVICE_STATUS_AVAILABLE

    @pytest.mark.asyncio
    async def test_handle_heartbeat_success_without_attributes(self):
        """Testet Behandlung erfolgreicher Heartbeats ohne Agent-Attribute."""
        service = AgentHeartbeatService()

        # Mock Agent ohne last_heartbeat und status Attribute
        mock_agent = MagicMock(spec=[])  # Leere spec = keine Attribute

        # Sollte keine Exception werfen
        await service._handle_heartbeat_success("agent1", mock_agent)

    @pytest.mark.asyncio
    async def test_escalate_failure(self):
        """Testet Eskalation von Agent-Fehlern."""
        service = AgentHeartbeatService()

        with patch.object(service, "_remove_agent") as mock_remove:
            await service._escalate_failure("agent1")
            mock_remove.assert_called_once_with("agent1")

    @pytest.mark.asyncio
    async def test_remove_agent(self):
        """Testet Entfernung von Agents."""
        service = AgentHeartbeatService()

        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"

        with patch("services.agent_heartbeat_service.dynamic_registry") as mock_registry:
            mock_registry.agents = {"agent1": mock_agent}

            await service._remove_agent("agent1")

            # Agent sollte aus der Registry entfernt worden sein
            assert "agent1" not in mock_registry.agents

    def test_get_status(self):
        """Testet Status-Abfrage des Services."""
        service = AgentHeartbeatService()
        service.failure_counts = {"agent1": 2, "agent2": 1}

        with patch("services.agent_heartbeat_service.dynamic_registry") as mock_registry:
            # Mock Agents mit Heartbeat-URLs
            mock_agent1 = MagicMock()
            mock_agent1.heartbeat_url = "http://localhost:8080/health"
            mock_agent2 = MagicMock()
            mock_agent2.heartbeat_url = None
            mock_agent3 = MagicMock()
            mock_agent3.heartbeat_url = "http://localhost:8081/health"

            mock_registry.agents = {
                "agent1": mock_agent1,
                "agent2": mock_agent2,  # Ohne heartbeat_url
                "agent3": mock_agent3,
            }

            status = service.get_status()

            assert status["running"] is False
            assert status["check_interval"] == DEFAULT_HEARTBEAT_INTERVAL
            assert status["timeout"] == DEFAULT_HEARTBEAT_TIMEOUT
            assert status["max_failures"] == DEFAULT_MAX_HEARTBEAT_FAILURES
            assert status["monitored_agents"] == 2  # Nur agent1 und agent3
            assert status["agents_with_failures"] == 2
            assert status["failure_counts"] == {"agent1": 2, "agent2": 1}

    @pytest.mark.asyncio
    async def test_periodic_task_execution(self):
        """Testet periodische Ausführung von Heartbeat-Checks."""
        service = AgentHeartbeatService(check_interval=0.05)  # Sehr kurzes Intervall

        with patch.object(service, "_check_all_agents") as mock_check:
            await service.start()
            await asyncio.sleep(0.15)  # Warte auf mehrere Ausführungen
            await service.stop()

            # _check_all_agents sollte mehrfach aufgerufen worden sein
            assert mock_check.call_count > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Testet Health-Check des Services."""
        service = AgentHeartbeatService()
        await service.start()

        health = await service.health_check()

        assert health["service"] == "AgentHeartbeatService"
        assert health["status"] == SERVICE_STATUS_AVAILABLE
        assert health["running"] is True
        assert "response_time_ms" in health

        await service.stop()


class TestAgentHeartbeatServiceIntegration:
    """Integrations-Tests für AgentHeartbeatService."""

    @pytest.mark.asyncio
    async def test_full_heartbeat_cycle(self):
        """Testet vollständigen Heartbeat-Zyklus."""
        service = AgentHeartbeatService(
            check_interval=0.1,
            timeout=1.0,
            max_failures=2
        )

        # Mock Agent
        mock_agent = MagicMock()
        mock_agent.heartbeat_url = "http://localhost:8080/health"
        mock_agent.status = "unknown"

        with patch("services.agent_heartbeat_service.dynamic_registry") as mock_registry:
            mock_registry.agents = {"agent1": mock_agent}

            # Mock HTTP Client für Fehler
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.side_effect = httpx.ConnectError("Connection failed")
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await service.start()
                await asyncio.sleep(0.3)  # Warte auf mehrere Fehler
                await service.stop()

                # Agent sollte nach max_failures entfernt worden sein
                assert "agent1" not in mock_registry.agents
