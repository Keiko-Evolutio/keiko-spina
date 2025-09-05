"""Test-Konfiguration für Reporting-Service Tests.

Stellt gemeinsame Fixtures und Mock-Objekte bereit.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_settings():
    """Mock für Settings-Konfiguration."""
    settings_mock = MagicMock()
    settings_mock.reporting_interval_minutes = 60
    settings_mock.reporting_enabled = True
    settings_mock.reporting_default_recipients = "test@example.com,admin@example.com"
    settings_mock.grafana_url = "https://grafana.example.com"
    settings_mock.grafana_api_token.get_secret_value.return_value = "test-token"
    return settings_mock


@pytest.fixture
def mock_grafana_response():
    """Mock für Grafana API-Antworten."""
    return b"fake-png-content"


@pytest.fixture
def mock_httpx_client():
    """Mock für httpx.AsyncClient."""
    client_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.content = b"fake-png-content"
    response_mock.raise_for_status = MagicMock()
    client_mock.get.return_value = response_mock
    return client_mock


@pytest.fixture
def mock_alert_dispatcher():
    """Mock für Alert-Dispatcher."""
    dispatcher_mock = AsyncMock()
    dispatcher_mock.send_email = AsyncMock()
    return dispatcher_mock


@pytest.fixture
def sample_report_config() -> dict[str, Any]:
    """Beispiel-Konfiguration für Reports."""
    return {
        "dashboard_uid": "test-dashboard",
        "panel_id": 1,
        "width": 1600,
        "height": 900,
        "subject": "Test Report",
        "body": "Test report body",
        "recipients": ["test@example.com"]
    }


@pytest.fixture
def sample_grafana_params() -> dict[str, Any]:
    """Beispiel-Parameter für Grafana-Requests."""
    return {
        "width": 1600,
        "height": 900,
        "from": "now-1h",
        "to": "now"
    }


@pytest.fixture(autouse=True)
def reset_mocks():
    """Automatisches Reset aller Mocks nach jedem Test."""
    return
    # Cleanup wird automatisch durch pytest durchgeführt
