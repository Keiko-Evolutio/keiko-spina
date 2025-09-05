"""Unit-Tests für services.webhooks.unified_alerting.

Testet die Migration von Slack- und Teams-Adaptern zu UnifiedHTTPClient.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from services.webhooks.alerting import WebhookAlertingException
from services.webhooks.unified_alerting import (
    UnifiedGenericWebhookAdapter,
    UnifiedSlackAdapter,
    UnifiedTeamsAdapter,
    create_generic_webhook_adapter,
    create_slack_adapter,
    create_teams_adapter,
)


class TestUnifiedSlackAdapter:
    """Tests für UnifiedSlackAdapter."""

    def test_adapter_initialization(self):
        """Testet Adapter-Initialisierung."""
        adapter = UnifiedSlackAdapter(
            webhook_url="https://hooks.slack.com/webhook/test",
            timeout_seconds=10.0
        )

        assert adapter.webhook_url == "https://hooks.slack.com/webhook/test"
        assert adapter.timeout_seconds == 10.0
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_ensure_client_creates_unified_client(self):
        """Testet _ensure_client erstellt UnifiedHTTPClient."""
        adapter = UnifiedSlackAdapter(
            webhook_url="https://hooks.slack.com/webhook/test"
        )

        with patch("services.webhooks.unified_alerting.ClientFactory") as mock_factory:
            mock_client = AsyncMock()
            mock_factory.create_alerting_client.return_value = mock_client

            client = await adapter._ensure_client()

            assert client == mock_client
            mock_factory.create_alerting_client.assert_called_once_with(
                webhook_url="https://hooks.slack.com/webhook/test",
                adapter_type="slack",
                timeout_seconds=5.0
            )

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Testet erfolgreiches Senden eines Slack-Alerts."""
        adapter = UnifiedSlackAdapter(
            webhook_url="https://hooks.slack.com/webhook/test"
        )

        # Mock UnifiedHTTPClient
        mock_client = AsyncMock()
        mock_client.post_json.return_value = {"ok": True}
        adapter._client = mock_client

        title = "Test Alert"
        message = {"error": "Something went wrong", "service": "test-service"}
        severity = "critical"

        # Mock redact_structure to return predictable result
        with patch("services.webhooks.unified_alerting.redact_structure") as mock_redact:
            mock_redact.return_value = message

            await adapter.send(title, message, severity)

            # Verify HTTP call
            expected_payload = {
                "text": "[CRITICAL] Test Alert",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{title}*\n```{json.dumps(message, ensure_ascii=False)}```",
                        },
                    }
                ],
            }

            mock_client.__aenter__.assert_called_once()
            mock_client.post_json.assert_called_once_with("", json_data=expected_payload)
            mock_redact.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_failure(self):
        """Testet fehlgeschlagenes Senden eines Slack-Alerts."""
        adapter = UnifiedSlackAdapter(
            webhook_url="https://hooks.slack.com/webhook/test"
        )

        # Mock UnifiedHTTPClient mit Fehler
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post_json.side_effect = Exception("Connection failed")
        adapter._client = mock_client

        with pytest.raises(WebhookAlertingException) as exc_info:
            await adapter.send("Test Alert", {"error": "test"}, "critical")

        assert "Slack send failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close(self):
        """Testet Client-Schließung."""
        adapter = UnifiedSlackAdapter(
            webhook_url="https://hooks.slack.com/webhook/test"
        )

        # Mock UnifiedHTTPClient
        mock_client = AsyncMock()
        adapter._client = mock_client

        await adapter.close()

        mock_client.close.assert_called_once()
        assert adapter._client is None


class TestUnifiedTeamsAdapter:
    """Tests für UnifiedTeamsAdapter."""

    def test_adapter_initialization(self):
        """Testet Adapter-Initialisierung."""
        adapter = UnifiedTeamsAdapter(
            webhook_url="https://outlook.office.com/webhook/test",
            timeout_seconds=10.0
        )

        assert adapter.webhook_url == "https://outlook.office.com/webhook/test"
        assert adapter.timeout_seconds == 10.0
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_ensure_client_creates_unified_client(self):
        """Testet _ensure_client erstellt UnifiedHTTPClient."""
        adapter = UnifiedTeamsAdapter(
            webhook_url="https://outlook.office.com/webhook/test"
        )

        with patch("services.webhooks.unified_alerting.ClientFactory") as mock_factory:
            mock_client = AsyncMock()
            mock_factory.create_alerting_client.return_value = mock_client

            client = await adapter._ensure_client()

            assert client == mock_client
            mock_factory.create_alerting_client.assert_called_once_with(
                webhook_url="https://outlook.office.com/webhook/test",
                adapter_type="teams",
                timeout_seconds=5.0
            )

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Testet erfolgreiches Senden eines Teams-Alerts."""
        adapter = UnifiedTeamsAdapter(
            webhook_url="https://outlook.office.com/webhook/test"
        )

        # Mock UnifiedHTTPClient
        mock_client = AsyncMock()
        mock_client.post_json.return_value = {"status": "success"}
        adapter._client = mock_client

        title = "Test Alert"
        message = {"error": "Something went wrong", "timestamp": "2024-01-01T00:00:00Z"}
        severity = "warning"

        await adapter.send(title, message, severity)

        # Verify HTTP call structure (Teams Adaptive Card)
        mock_client.__aenter__.assert_called_once()
        call_args = mock_client.post_json.call_args
        assert call_args[0] == ("",)  # URL argument

        payload = call_args[1]["json_data"]
        assert payload["type"] == "message"
        assert len(payload["attachments"]) == 1
        assert payload["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"

    @pytest.mark.asyncio
    async def test_send_failure(self):
        """Testet fehlgeschlagenes Senden eines Teams-Alerts."""
        adapter = UnifiedTeamsAdapter(
            webhook_url="https://outlook.office.com/webhook/test"
        )

        # Mock UnifiedHTTPClient mit Fehler
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post_json.side_effect = Exception("Connection failed")
        adapter._client = mock_client

        with pytest.raises(WebhookAlertingException) as exc_info:
            await adapter.send("Test Alert", {"error": "test"}, "warning")

        assert "Teams send failed" in str(exc_info.value)


class TestUnifiedGenericWebhookAdapter:
    """Tests für UnifiedGenericWebhookAdapter."""

    def test_adapter_initialization(self):
        """Testet Adapter-Initialisierung."""
        adapter = UnifiedGenericWebhookAdapter(
            webhook_url="https://api.example.com/webhook",
            adapter_name="custom",
            timeout_seconds=15.0,
            custom_headers={"X-API-Key": "secret"}
        )

        assert adapter.webhook_url == "https://api.example.com/webhook"
        assert adapter.adapter_name == "custom"
        assert adapter.timeout_seconds == 15.0
        assert adapter.custom_headers == {"X-API-Key": "secret"}

    @pytest.mark.asyncio
    async def test_ensure_client_creates_unified_client(self):
        """Testet _ensure_client erstellt UnifiedHTTPClient."""
        adapter = UnifiedGenericWebhookAdapter(
            webhook_url="https://api.example.com/webhook",
            custom_headers={"X-API-Key": "secret"}
        )

        with patch("services.webhooks.unified_alerting.ClientFactory") as mock_factory:
            mock_client = AsyncMock()
            mock_factory.create_webhook_client.return_value = mock_client

            client = await adapter._ensure_client()

            assert client == mock_client
            mock_factory.create_webhook_client.assert_called_once_with(
                webhook_url="https://api.example.com/webhook",
                timeout_seconds=5.0,
                custom_headers={"X-API-Key": "secret"}
            )

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Testet erfolgreiches Senden eines Generic-Webhook-Alerts."""
        adapter = UnifiedGenericWebhookAdapter(
            webhook_url="https://api.example.com/webhook",
            adapter_name="custom"
        )

        # Mock UnifiedHTTPClient
        mock_client = AsyncMock()
        mock_client.post_json.return_value = {"received": True}
        adapter._client = mock_client

        title = "Test Alert"
        message = {
            "error": "Something went wrong",
            "service": "test-service",
            "source": "test-service"
        }
        severity = "info"

        # Mock redact_structure to return predictable result
        with patch("services.webhooks.unified_alerting.redact_structure") as mock_redact:
            mock_redact.return_value = message

            await adapter.send(title, message, severity)

            # Verify HTTP call
            expected_payload = {
                "title": "Test Alert",
                "message": message,
                "severity": "info",
                "timestamp": None,  # No timestamp in message
                "source": "test-service"
            }

            mock_client.__aenter__.assert_called_once()
            mock_client.post_json.assert_called_once_with("", json_data=expected_payload)
            mock_redact.assert_called_once_with(message)


class TestFactoryFunctions:
    """Tests für Factory-Funktionen."""

    def test_create_slack_adapter(self):
        """Testet create_slack_adapter Factory-Funktion."""
        adapter = create_slack_adapter(
            webhook_url="https://hooks.slack.com/webhook/test",
            timeout_seconds=10.0
        )

        assert isinstance(adapter, UnifiedSlackAdapter)
        assert adapter.webhook_url == "https://hooks.slack.com/webhook/test"
        assert adapter.timeout_seconds == 10.0

    def test_create_teams_adapter(self):
        """Testet create_teams_adapter Factory-Funktion."""
        adapter = create_teams_adapter(
            webhook_url="https://outlook.office.com/webhook/test",
            timeout_seconds=15.0
        )

        assert isinstance(adapter, UnifiedTeamsAdapter)
        assert adapter.webhook_url == "https://outlook.office.com/webhook/test"
        assert adapter.timeout_seconds == 15.0

    def test_create_generic_webhook_adapter(self):
        """Testet create_generic_webhook_adapter Factory-Funktion."""
        adapter = create_generic_webhook_adapter(
            webhook_url="https://api.example.com/webhook",
            adapter_name="custom",
            timeout_seconds=20.0,
            custom_headers={"Authorization": "Bearer token"}
        )

        assert isinstance(adapter, UnifiedGenericWebhookAdapter)
        assert adapter.webhook_url == "https://api.example.com/webhook"
        assert adapter.adapter_name == "custom"
        assert adapter.timeout_seconds == 20.0
        assert adapter.custom_headers == {"Authorization": "Bearer token"}


class TestBackwardCompatibility:
    """Tests für Backward-Compatibility."""

    def test_slack_adapter_alias(self):
        """Testet SlackAdapter Alias."""
        from services.webhooks.unified_alerting import SlackAdapter

        adapter = SlackAdapter(webhook_url="https://hooks.slack.com/webhook/test")
        assert isinstance(adapter, UnifiedSlackAdapter)

    def test_teams_adapter_alias(self):
        """Testet TeamsAdapter Alias."""
        from services.webhooks.unified_alerting import TeamsAdapter

        adapter = TeamsAdapter(webhook_url="https://outlook.office.com/webhook/test")
        assert isinstance(adapter, UnifiedTeamsAdapter)
