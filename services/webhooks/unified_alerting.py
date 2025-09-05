"""Unified Alerting basierend auf UnifiedHTTPClient.

Migriert Slack- und Teams-Adapter zur neuen UnifiedHTTPClient-Architektur
während die bestehende API beibehalten wird.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from kei_logging.pii_redaction import redact_structure
from services.core.client_factory import ClientFactory

from .alerting import WebhookAlertingException

if TYPE_CHECKING:
    from services.core.unified_client import UnifiedHTTPClient

logger = get_logger(__name__)


@dataclass
class UnifiedSlackAdapter:
    """Unified Slack-Webhook Adapter basierend auf UnifiedHTTPClient.

    Drop-in Replacement für SlackAdapter mit verbesserter Architektur:
    - Verwendet UnifiedHTTPClient für HTTP-Kommunikation
    - Behält vollständige API-Kompatibilität bei
    - Verbesserte Error-Handling und Circuit Breaker Integration
    """

    webhook_url: str
    timeout_seconds: float = 5.0

    def __post_init__(self):
        """Initialisiert den Unified HTTP Client."""
        self._client: UnifiedHTTPClient | None = None

    async def _ensure_client(self) -> UnifiedHTTPClient:
        """Stellt sicher, dass der Unified HTTP Client initialisiert ist."""
        if self._client is None:
            self._client = ClientFactory.create_alerting_client(
                webhook_url=self.webhook_url,
                adapter_type="slack",
                timeout_seconds=self.timeout_seconds
            )
        return self._client

    async def send(self, title: str, message: dict[str, Any], severity: str) -> None:
        """Versendet einen Alert über Slack.

        Args:
            title: Kurztitel
            message: Strukturierte Nachricht (wird PII-bereinigt)
            severity: Schweregrad

        Raises:
            WebhookAlertingException: Bei Versand-Fehlern
        """
        payload = {
            "text": f"[{severity.upper()}] {title}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{title}*\n```{json.dumps(redact_structure(message), ensure_ascii=False)}```",
                    },
                }
            ],
        }

        try:
            client = await self._ensure_client()
            async with client:
                await client.post_json("", json_data=payload)
                logger.debug(f"Slack Alert erfolgreich versendet: {title}")

        except Exception as exc:
            logger.exception(f"Slack Alert fehlgeschlagen: {exc}")
            raise WebhookAlertingException(f"Slack send failed: {exc}") from exc

    async def close(self) -> None:
        """Schließt den HTTP Client."""
        if self._client:
            await self._client.close()
            self._client = None


@dataclass
class UnifiedTeamsAdapter:
    """Unified Microsoft Teams-Webhook Adapter basierend auf UnifiedHTTPClient.

    Drop-in Replacement für TeamsAdapter mit verbesserter Architektur:
    - Verwendet UnifiedHTTPClient für HTTP-Kommunikation
    - Behält vollständige API-Kompatibilität bei
    - Verbesserte Error-Handling und Circuit Breaker Integration
    """

    webhook_url: str
    timeout_seconds: float = 5.0

    def __post_init__(self):
        """Initialisiert den Unified HTTP Client."""
        self._client: UnifiedHTTPClient | None = None

    async def _ensure_client(self) -> UnifiedHTTPClient:
        """Stellt sicher, dass der Unified HTTP Client initialisiert ist."""
        if self._client is None:
            self._client = ClientFactory.create_alerting_client(
                webhook_url=self.webhook_url,
                adapter_type="teams",
                timeout_seconds=self.timeout_seconds
            )
        return self._client

    async def send(self, title: str, message: dict[str, Any], severity: str) -> None:
        """Versendet einen Alert über Microsoft Teams.

        Args:
            title: Kurztitel
            message: Strukturierte Nachricht (wird PII-bereinigt)
            severity: Schweregrad

        Raises:
            WebhookAlertingException: Bei Versand-Fehlern
        """
        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Large",
                                "weight": "Bolder",
                                "text": f"[{severity.upper()}] {title}"
                            },
                            {
                                "type": "TextBlock",
                                "wrap": True,
                                "text": f"```{json.dumps(redact_structure(message), ensure_ascii=False)}```"
                            },
                        ],
                    },
                }
            ],
        }

        try:
            client = await self._ensure_client()
            async with client:
                await client.post_json("", json_data=card)
                logger.debug(f"Teams Alert erfolgreich versendet: {title}")

        except Exception as exc:
            logger.exception(f"Teams Alert fehlgeschlagen: {exc}")
            raise WebhookAlertingException(f"Teams send failed: {exc}") from exc

    async def close(self) -> None:
        """Schließt den HTTP Client."""
        if self._client:
            await self._client.close()
            self._client = None


@dataclass
class UnifiedGenericWebhookAdapter:
    """Unified Generic Webhook Adapter für beliebige Webhook-Endpunkte.

    Ermöglicht die Verwendung von UnifiedHTTPClient für beliebige Webhook-Integrationen.
    """

    webhook_url: str
    adapter_name: str = "generic"
    timeout_seconds: float = 5.0
    custom_headers: dict[str, str] | None = None

    def __post_init__(self):
        """Initialisiert den Unified HTTP Client."""
        self._client: UnifiedHTTPClient | None = None

    async def _ensure_client(self) -> UnifiedHTTPClient:
        """Stellt sicher, dass der Unified HTTP Client initialisiert ist."""
        if self._client is None:
            self._client = ClientFactory.create_webhook_client(
                webhook_url=self.webhook_url,
                timeout_seconds=self.timeout_seconds,
                custom_headers=self.custom_headers or {}
            )
        return self._client

    async def send(self, title: str, message: dict[str, Any], severity: str) -> None:
        """Versendet einen Alert über generischen Webhook.

        Args:
            title: Kurztitel
            message: Strukturierte Nachricht (wird PII-bereinigt)
            severity: Schweregrad

        Raises:
            WebhookAlertingException: Bei Versand-Fehlern
        """
        payload = {
            "title": title,
            "message": redact_structure(message),
            "severity": severity,
            "timestamp": message.get("timestamp"),
            "source": message.get("source", "keiko")
        }

        try:
            client = await self._ensure_client()
            async with client:
                await client.post_json("", json_data=payload)
                logger.debug(f"{self.adapter_name} Alert erfolgreich versendet: {title}")

        except Exception as exc:
            logger.exception(f"{self.adapter_name} Alert fehlgeschlagen: {exc}")
            raise WebhookAlertingException(f"{self.adapter_name} send failed: {exc}") from exc

    async def close(self) -> None:
        """Schließt den HTTP Client."""
        if self._client:
            await self._client.close()
            self._client = None


# Backward-Compatibility Aliases
SlackAdapter = UnifiedSlackAdapter
TeamsAdapter = UnifiedTeamsAdapter


# Factory-Funktionen für einfache Erstellung
def create_slack_adapter(webhook_url: str, **kwargs) -> UnifiedSlackAdapter:
    """Erstellt einen Slack-Adapter mit UnifiedHTTPClient."""
    return UnifiedSlackAdapter(webhook_url=webhook_url, **kwargs)


def create_teams_adapter(webhook_url: str, **kwargs) -> UnifiedTeamsAdapter:
    """Erstellt einen Teams-Adapter mit UnifiedHTTPClient."""
    return UnifiedTeamsAdapter(webhook_url=webhook_url, **kwargs)


def create_generic_webhook_adapter(
    webhook_url: str,
    adapter_name: str = "generic",
    **kwargs
) -> UnifiedGenericWebhookAdapter:
    """Erstellt einen generischen Webhook-Adapter mit UnifiedHTTPClient."""
    return UnifiedGenericWebhookAdapter(
        webhook_url=webhook_url,
        adapter_name=adapter_name,
        **kwargs
    )
