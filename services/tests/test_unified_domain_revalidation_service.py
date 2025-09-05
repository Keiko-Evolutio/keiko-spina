"""Unit-Tests für services.unified_domain_revalidation_service.

Testet die Migration von DomainRevalidationService zu UnifiedDomainRevalidationService.
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.unified_domain_revalidation_service import UnifiedDomainRevalidationService


class TestUnifiedDomainRevalidationService:
    """Tests für UnifiedDomainRevalidationService."""

    def test_service_initialization(self):
        """Testet Service-Initialisierung."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=True,
                revalidation_interval_hours=24,
                revalidation_on_startup=False,
                remove_invalid_servers=True
            )

            service = UnifiedDomainRevalidationService()

            assert service.service_name == "UnifiedDomainRevalidationService"
            assert service.interval_seconds == 24 * 3600  # 24 Stunden in Sekunden
            assert service.last_revalidation is None
            assert service.last_config_reload is None

    @pytest.mark.asyncio
    async def test_initialize_with_startup_revalidation(self):
        """Testet Initialisierung mit Startup-Revalidierung."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=True,
                revalidation_interval_hours=24,
                revalidation_on_startup=True,
                remove_invalid_servers=True
            )

            service = UnifiedDomainRevalidationService()

            with patch.object(service, "_perform_revalidation") as mock_revalidation:
                await service._initialize()
                mock_revalidation.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_disabled_service(self):
        """Testet Initialisierung mit deaktiviertem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=False,
                revalidation_interval_hours=24,
                revalidation_on_startup=False
            )

            service = UnifiedDomainRevalidationService()

            with patch.object(service, "_perform_revalidation") as mock_revalidation:
                await service._initialize()
                mock_revalidation.assert_not_called()

    @pytest.mark.asyncio
    async def test_perform_periodic_task_enabled(self):
        """Testet periodische Task bei aktiviertem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=True,
                revalidation_interval_hours=24
            )

            service = UnifiedDomainRevalidationService()
            service.last_config_reload = time.time() - 7200  # 2 Stunden alt

            with patch.object(service, "_reload_config") as mock_reload:
                with patch.object(service, "_perform_revalidation") as mock_revalidation:
                    await service._perform_periodic_task()

                    mock_reload.assert_called_once()
                    mock_revalidation.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_periodic_task_disabled(self):
        """Testet periodische Task bei deaktiviertem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(enabled=False)

            service = UnifiedDomainRevalidationService()

            with patch.object(service, "_perform_revalidation") as mock_revalidation:
                await service._perform_periodic_task()
                mock_revalidation.assert_not_called()

    @pytest.mark.asyncio
    async def test_perform_health_check_running_enabled(self):
        """Testet Health-Check bei laufendem und aktiviertem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=True,
                revalidation_interval_hours=24
            )

            service = UnifiedDomainRevalidationService()
            service.running = True
            service.last_revalidation = time.time() - 3600  # 1 Stunde alt

            is_healthy = await service._perform_health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_perform_health_check_running_disabled(self):
        """Testet Health-Check bei laufendem aber deaktiviertem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(enabled=False)

            service = UnifiedDomainRevalidationService()
            service.running = True

            is_healthy = await service._perform_health_check()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_perform_health_check_not_running(self):
        """Testet Health-Check bei nicht laufendem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config"):
            service = UnifiedDomainRevalidationService()
            service.running = False

            is_healthy = await service._perform_health_check()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_reload_config_success(self):
        """Testet erfolgreiches Config-Reload."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_get_config:
            with patch("services.unified_domain_revalidation_service.reload_domain_validation_config") as mock_reload:
                mock_get_config.return_value = MagicMock(
                    revalidation_interval_hours=12  # Geändert von 24 auf 12
                )

                service = UnifiedDomainRevalidationService()
                service.interval_seconds = 24 * 3600  # Alter Wert

                await service._reload_config()

                mock_reload.assert_called_once()
                assert service.interval_seconds == 12 * 3600  # Neuer Wert

    @pytest.mark.asyncio
    async def test_reload_config_failure(self):
        """Testet fehlgeschlagenes Config-Reload."""
        with patch("services.unified_domain_revalidation_service.reload_domain_validation_config") as mock_reload:
            mock_reload.side_effect = Exception("Config reload failed")

            service = UnifiedDomainRevalidationService()

            # Sollte keine Exception werfen, nur loggen
            await service._reload_config()
            mock_reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_revalidation_success(self):
        """Testet erfolgreiche Revalidierung."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            with patch("services.unified_domain_revalidation_service.mcp_registry") as mock_registry:
                mock_config.return_value = MagicMock(revalidation_interval_hours=24)
                mock_registry.revalidate_domains_if_needed = AsyncMock(return_value={
                    "server1": True,
                    "server2": True,
                    "server3": False
                })

                service = UnifiedDomainRevalidationService()

                with patch.object(service, "_handle_validation_failures") as mock_handle_failures:
                    await service._perform_revalidation()

                    mock_registry.revalidate_domains_if_needed.assert_called_once_with(24)
                    mock_handle_failures.assert_called_once()
                    assert service.last_revalidation is not None

    @pytest.mark.asyncio
    async def test_perform_revalidation_all_success(self):
        """Testet Revalidierung ohne Fehler."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            with patch("services.unified_domain_revalidation_service.mcp_registry") as mock_registry:
                mock_config.return_value = MagicMock(revalidation_interval_hours=24)
                mock_registry.revalidate_domains_if_needed = AsyncMock(return_value={
                    "server1": True,
                    "server2": True
                })

                service = UnifiedDomainRevalidationService()

                with patch.object(service, "_handle_validation_failures") as mock_handle_failures:
                    await service._perform_revalidation()

                    mock_handle_failures.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_validation_failures_remove_enabled(self):
        """Testet Behandlung von Validierung-Fehlern mit Entfernung."""
        with patch("services.unified_domain_revalidation_service.mcp_registry") as mock_registry:
            config = MagicMock(remove_invalid_servers=True)
            results = {"server1": True, "server2": False, "server3": False}

            service = UnifiedDomainRevalidationService()

            await service._handle_validation_failures(results, config)

            # Sollte failed servers entfernen
            assert mock_registry.remove_server.call_count == 2
            mock_registry.remove_server.assert_any_call("server2")
            mock_registry.remove_server.assert_any_call("server3")

    @pytest.mark.asyncio
    async def test_handle_validation_failures_remove_disabled(self):
        """Testet Behandlung von Validierung-Fehlern ohne Entfernung."""
        with patch("services.unified_domain_revalidation_service.mcp_registry") as mock_registry:
            config = MagicMock(remove_invalid_servers=False)
            results = {"server1": True, "server2": False}

            service = UnifiedDomainRevalidationService()

            await service._handle_validation_failures(results, config)

            # Sollte keine Server entfernen
            mock_registry.remove_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_revalidation(self):
        """Testet manuelle Revalidierung."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            with patch("services.unified_domain_revalidation_service.mcp_registry") as mock_registry:
                with patch("services.unified_domain_revalidation_service.FORCE_REVALIDATION_INTERVAL", 0):
                    mock_config.return_value = MagicMock()
                    mock_registry.revalidate_domains_if_needed = AsyncMock(return_value={
                        "server1": True,
                        "server2": False
                    })

                    service = UnifiedDomainRevalidationService()

                    with patch.object(service, "_handle_validation_failures") as mock_handle_failures:
                        results = await service.force_revalidation()

                        assert results == {"server1": True, "server2": False}
                        mock_registry.revalidate_domains_if_needed.assert_called_once_with(0)
                        mock_handle_failures.assert_called_once()

    def test_get_status(self):
        """Testet Status-Abfrage."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=True,
                revalidation_interval_hours=24,
                remove_invalid_servers=True,
                revalidation_on_startup=False
            )

            service = UnifiedDomainRevalidationService()
            service.running = True
            service.last_revalidation = 1234567890.0
            service.last_config_reload = 1234567800.0

            status = service.get_status()

            assert status["running"] is True
            assert status["enabled"] is True
            assert status["last_revalidation"] == 1234567890.0
            assert status["revalidation_interval_hours"] == 24
            assert status["service_name"] == "UnifiedDomainRevalidationService"
            assert status["health_status"] == "unknown"

    @pytest.mark.asyncio
    async def test_get_next_revalidation_time(self):
        """Testet Abfrage der nächsten Revalidierung."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(revalidation_interval_hours=24)

            service = UnifiedDomainRevalidationService()
            service.running = True
            service.last_revalidation = 1234567890.0

            next_time = await service.get_next_revalidation_time()

            assert next_time is not None
            assert isinstance(next_time, datetime)

    @pytest.mark.asyncio
    async def test_get_next_revalidation_time_not_running(self):
        """Testet Abfrage der nächsten Revalidierung bei nicht laufendem Service."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config"):
            service = UnifiedDomainRevalidationService()
            service.running = False

            next_time = await service.get_next_revalidation_time()
            assert next_time is None

    @pytest.mark.asyncio
    async def test_get_revalidation_history(self):
        """Testet Abfrage der Revalidierung-Historie."""
        with patch("services.unified_domain_revalidation_service.get_domain_validation_config"):
            service = UnifiedDomainRevalidationService()
            service.last_revalidation = 1234567890.0

            history = await service.get_revalidation_history()

            assert len(history) == 1
            assert history[0]["timestamp"] == 1234567890.0
            assert history[0]["status"] == "completed"


class TestBackwardCompatibility:
    """Tests für Backward-Compatibility."""

    def test_domain_revalidation_service_alias(self):
        """Testet DomainRevalidationService Alias."""
        from services.unified_domain_revalidation_service import DomainRevalidationService

        with patch("services.unified_domain_revalidation_service.get_domain_validation_config") as mock_config:
            mock_config.return_value = MagicMock(
                enabled=True,
                revalidation_interval_hours=24,
                revalidation_on_startup=False
            )

            service = DomainRevalidationService()
            assert isinstance(service, UnifiedDomainRevalidationService)
