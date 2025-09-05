"""Alerting-Service für Webhook-Subsystem.

Stellt webhook-basierte Adapter (Slack, Microsoft Teams) bereit und eine
asynchrone Dispatch-Logik mit Rate Limiting, Retry und PII-Redaktion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from config.settings import settings
from kei_logging import get_logger
from kei_logging.pii_redaction import redact_structure
from storage.cache.redis_cache import NoOpCache, get_cache_client

# Service-Imports)
try:
    from services.webhooks.notification_adapters import (
        SMTPEmailAdapter,
        TwilioConfig,
        TwilioSMSAdapter,
    )
    NOTIFICATION_ADAPTERS_AVAILABLE = True
except ImportError:
    SMTPEmailAdapter = None
    TwilioConfig = None
    TwilioSMSAdapter = None
    NOTIFICATION_ADAPTERS_AVAILABLE = False

logger = get_logger(__name__)


class WebhookAlertingException(Exception):
    """Spezifische Ausnahme für Alerting-Fehler."""


class AlertSeverity:
    """Schweregrade für Alerts."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertAdapter(Protocol):
    """Interface für Alert-Adapter."""

    async def send(self, title: str, message: dict[str, Any], severity: str) -> None:
        """Versendet einen Alert.

        Args:
            title: Kurztitel
            message: Strukturierte Nachricht (wird PII-bereinigt)
            severity: Schweregrad
        """


class EmailAdapter(Protocol):
    """Interface für E-Mail Versand."""

    async def send_email(self, subject: str, body: str, severity: str) -> None:
        ...


class SMSAdapter(Protocol):
    """Interface für SMS Versand."""

    async def send_sms(self, text: str, severity: str) -> None:
        ...


@dataclass
class SlackAdapter:
    """Slack-Webhook Adapter."""

    webhook_url: str

    async def send(self, title: str, message: dict[str, Any], severity: str) -> None:
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
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(self.webhook_url, json=payload)
            if resp.status_code >= 400:
                raise WebhookAlertingException(f"Slack send failed: {resp.status_code}")


@dataclass
class TeamsAdapter:
    """Microsoft Teams-Webhook Adapter (Adaptive Card im einfachsten Format)."""

    webhook_url: str

    async def send(self, title: str, message: dict[str, Any], severity: str) -> None:
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
                            {"type": "TextBlock", "size": "Large", "weight": "Bolder", "text": f"[{severity.upper()}] {title}"},
                            {"type": "TextBlock", "wrap": True, "text": f"```{json.dumps(redact_structure(message), ensure_ascii=False)}```"},
                        ],
                    },
                }
            ],
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(self.webhook_url, json=card)
            if resp.status_code >= 400:
                raise WebhookAlertingException(f"Teams send failed: {resp.status_code}")


class AlertDispatcher:
    """Zentraler Dispatcher mit Rate-Limiting und Retry."""

    def __init__(self, slack: AlertAdapter | None, teams: AlertAdapter | None, email: EmailAdapter | None = None, sms: SMSAdapter | None = None) -> None:
        # Adapter initialisieren
        self.slack = slack
        self.teams = teams
        self.email = email
        self.sms = sms
        # Einfaches Token-Bucket pro Minute
        self._tokens = settings.alert_rate_limit_per_minute
        self._refill_interval = 60.0
        self._lock = asyncio.Lock()
        self._refill_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Startet das periodische Token-Refill."""
        if self._refill_task is None:
            self._refill_task = asyncio.create_task(self._refill_loop())

    async def stop(self) -> None:
        """Stoppt das periodische Token-Refill."""
        if self._refill_task:
            self._refill_task.cancel()
            with contextlib.suppress(Exception):
                await self._refill_task
            self._refill_task = None

    async def _refill_loop(self) -> None:
        """Refill-Loop pro Minute."""
        try:
            while True:
                await asyncio.sleep(self._refill_interval)
                async with self._lock:
                    self._tokens = settings.alert_rate_limit_per_minute
        except asyncio.CancelledError:
            return

    async def send_alert(self, title: str, message: dict[str, Any], severity: str) -> None:
        """Versendet Alert über konfigurierte Adapter (non-blocking mit Retry)."""
        if not settings.alerting_enabled:
            return
        async with self._lock:
            if self._tokens <= 0:
                logger.debug("Alerting gedrosselt (Rate-Limit erreicht)")
                return
            self._tokens -= 1

        async def _attempt(adapter: AlertAdapter) -> None:
            delay = settings.alert_retry_backoff_seconds
            for attempt in range(1, settings.alert_retry_max_attempts + 1):
                try:
                    await adapter.send(title, message, severity)
                    return
                except (ConnectionError, TimeoutError) as exc:
                    if attempt >= settings.alert_retry_max_attempts:
                        logger.error(f"Alert-Zustellung fehlgeschlagen - Verbindungsproblem: {exc}")
                        # DLQ: fehlgeschlagene Benachrichtigung sichern
                        with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                            await _push_alert_dlq({
                                "title": title,
                                "message": redact_structure(message),
                                "severity": severity,
                                "channel": adapter.__class__.__name__,
                            })
                        return
                    await asyncio.sleep(delay)
                    delay *= 2
                except Exception as exc:
                    if attempt >= settings.alert_retry_max_attempts:
                        logger.exception(f"Alert-Zustellung fehlgeschlagen - Unerwarteter Fehler: {exc}")
                        # DLQ: fehlgeschlagene Benachrichtigung sichern
                        with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                            await _push_alert_dlq({
                                "title": title,
                                "message": redact_structure(message),
                                "severity": severity,
                                "channel": adapter.__class__.__name__,
                            })
                        return
                    await asyncio.sleep(delay)
                    delay *= 2

        tasks: list[asyncio.Task[None]] = []
        if self.slack:
            tasks.append(asyncio.create_task(_attempt(self.slack)))
        if self.teams:
            tasks.append(asyncio.create_task(_attempt(self.teams)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_email(self, subject: str, body: str, severity: str) -> None:
        """Versendet optional eine E-Mail (falls Adapter vorhanden) mit Retry."""
        if not settings.alerting_enabled or not self.email:
            return

        delay = settings.alert_retry_backoff_seconds
        for attempt in range(1, settings.alert_retry_max_attempts + 1):
            try:
                await self.email.send_email(subject=subject, body=body, severity=severity)
                return
            except (ConnectionError, TimeoutError) as exc:
                if attempt >= settings.alert_retry_max_attempts:
                    logger.error(f"E-Mail Zustellung fehlgeschlagen - Verbindungsproblem: {exc}")
                    return
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                if attempt >= settings.alert_retry_max_attempts:
                    logger.exception(f"E-Mail Zustellung fehlgeschlagen - Unerwarteter Fehler: {exc}")
                    return
                await asyncio.sleep(delay)
                delay *= 2

    async def send_sms(self, text: str, severity: str) -> None:
        """Versendet optional eine SMS (falls Adapter vorhanden) mit Retry."""
        if not settings.alerting_enabled or not self.sms:
            return

        delay = settings.alert_retry_backoff_seconds
        for attempt in range(1, settings.alert_retry_max_attempts + 1):
            try:
                await self.sms.send_sms(text=text, severity=severity)
                return
            except (ConnectionError, TimeoutError) as exc:
                if attempt >= settings.alert_retry_max_attempts:
                    logger.error(f"SMS Zustellung fehlgeschlagen - Verbindungsproblem: {exc}")
                    return
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                if attempt >= settings.alert_retry_max_attempts:
                    logger.exception(f"SMS Zustellung fehlgeschlagen - Unerwarteter Fehler: {exc}")
                    return
                await asyncio.sleep(delay)
                delay *= 2


_dispatcher: AlertDispatcher | None = None


def get_alert_dispatcher() -> AlertDispatcher:
    """Gibt Singleton-Dispatcher zurück und initialisiert Adapter aus Settings."""
    global _dispatcher
    if _dispatcher is None:
        slack = None
        teams = None
        email = None
        sms = None
        if settings.alert_slack_webhook_url.get_secret_value():
            slack = SlackAdapter(webhook_url=settings.alert_slack_webhook_url.get_secret_value())
        if settings.alert_teams_webhook_url.get_secret_value():
            teams = TeamsAdapter(webhook_url=settings.alert_teams_webhook_url.get_secret_value())
        # Optionale E-Mail/SMS Adapter lazy importieren (Konfiguration via ENV)
        try:
            if not NOTIFICATION_ADAPTERS_AVAILABLE:
                raise ImportError("Notification adapters not available")

            smtp_host = getattr(settings, "smtp_host", "")
            smtp_port = int(getattr(settings, "smtp_port", 0) or 0)
            smtp_user = getattr(settings, "smtp_user", "")
            smtp_password = getattr(settings, "smtp_password", "")
            smtp_from = getattr(settings, "smtp_from", "")
            if smtp_host and smtp_port and smtp_from:
                email = SMTPEmailAdapter(host=smtp_host, port=smtp_port, username=smtp_user or None, password=smtp_password or None, sender=smtp_from)
        except Exception:
            email = None

        try:
            if not NOTIFICATION_ADAPTERS_AVAILABLE:
                raise ImportError("Notification adapters not available")

            tw_sid = getattr(settings, "twilio_account_sid", "")
            tw_token = getattr(settings, "twilio_auth_token", "")
            tw_from = getattr(settings, "twilio_from_number", "")
            if tw_sid and tw_token and tw_from:
                sms = TwilioSMSAdapter(config=TwilioConfig(account_sid=tw_sid, auth_token=tw_token, from_number=tw_from))
        except Exception:
            sms = None

        _dispatcher = AlertDispatcher(slack=slack, teams=teams, email=email, sms=sms)
    return _dispatcher


async def _push_alert_dlq(item: dict[str, Any]) -> None:
    """Schreibt einen fehlgeschlagenen Alert in die DLQ (Redis List)."""
    try:
        client = await get_cache_client()
    except Exception:
        client = None
    if client is None or isinstance(client, NoOpCache):
        return
    try:
        await client.lpush("kei:alerting:dlq", json.dumps(item))
    except Exception:
        return


async def alert_dlq_size() -> int:
    """Gibt die Größe der Alert-DLQ zurück."""
    try:
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return 0
        val = await client.llen("kei:alerting:dlq")  # type: ignore[attr-defined]
        return int(val or 0)
    except Exception:
        return 0


async def alert_dlq_replay(max_items: int = 50) -> dict[str, int]:
    """Versucht DLQ-Einträge erneut zu senden (best effort)."""
    try:
        client = await get_cache_client()
    except Exception:
        client = None
    if client is None or isinstance(client, NoOpCache):
        return {"retried": 0, "success": 0, "failed": 0}

    retried = 0
    success = 0
    failed = 0
    dispatcher = get_alert_dispatcher()

    for _ in range(max(0, max_items)):
        data = None
        try:
            data = await client.rpop("kei:alerting:dlq")
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"DLQ-Abruf fehlgeschlagen - Verbindungsproblem: {e}")
            data = None
        except Exception as e:
            logger.warning(f"DLQ-Abruf fehlgeschlagen - Unerwarteter Fehler: {e}")
            data = None
        if not data:
            break
        retried += 1
        try:
            obj = json.loads(data)
            title = obj.get("title", "Keiko Alert")
            message = obj.get("message", {})
            severity = obj.get("severity", "warning")
            await dispatcher.send_alert(title=title, message=message, severity=severity)
            success += 1
        except (ValueError, TypeError) as e:
            logger.debug(f"DLQ-Retry fehlgeschlagen - JSON-/Validierungsfehler: {e}")
            failed += 1
            # Bei erneutem Fehler zurück in die DLQ legen (Ende der Liste)
            with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                await client.lpush("kei:alerting:dlq", data)
        except Exception as e:
            logger.warning(f"DLQ-Retry fehlgeschlagen - Unerwarteter Fehler: {e}")
            failed += 1
            # Bei erneutem Fehler zurück in die DLQ legen (Ende der Liste)
            with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                await client.lpush("kei:alerting:dlq", data)
    return {"retried": retried, "success": success, "failed": failed}


async def emit_critical(title: str, message: dict[str, Any]) -> None:
    """Helper zum Senden eines Critical-Alerts."""
    await get_alert_dispatcher().send_alert(title, message, AlertSeverity.CRITICAL)


async def emit_warning(title: str, message: dict[str, Any]) -> None:
    """Helper zum Senden eines Warning-Alerts."""
    await get_alert_dispatcher().send_alert(title, message, AlertSeverity.WARNING)


async def emit_info(title: str, message: dict[str, Any]) -> None:
    """Helper zum Senden eines Info-Alerts."""
    await get_alert_dispatcher().send_alert(title, message, AlertSeverity.INFO)


__all__ = [
    "AlertDispatcher",
    "AlertSeverity",
    "SlackAdapter",
    "TeamsAdapter",
    "WebhookAlertingException",
    "emit_critical",
    "emit_info",
    "emit_warning",
    "get_alert_dispatcher",
]
