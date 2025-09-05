"""Communication Settings für Keiko Personal Assistant.

Alle kommunikationsbezogenen Konfigurationen (E-Mail, SMS, etc.).
Folgt Single Responsibility Principle.
"""

from pydantic import Field
from pydantic_settings import BaseSettings

from .constants import DEFAULT_SMTP_PORT
from .env_utils import get_env_int, get_env_str


class CommunicationSettings(BaseSettings):
    """Kommunikations-spezifische Konfigurationen."""

    # E-Mail Configuration (SMTP)
    smtp_enabled: bool = Field(
        default=False,
        description="E-Mail-Versand aktivieren"
    )

    smtp_host: str = Field(
        default="",
        description="SMTP Server Host"
    )

    smtp_port: int = Field(
        default=DEFAULT_SMTP_PORT,
        description="SMTP Server Port"
    )

    smtp_user: str = Field(
        default="",
        description="SMTP Benutzername"
    )

    smtp_password: str = Field(
        default="",
        description="SMTP Passwort"
    )

    smtp_from: str = Field(
        default="",
        description="Standard-Absender E-Mail-Adresse"
    )

    smtp_use_tls: bool = Field(
        default=True,
        description="TLS für SMTP verwenden"
    )

    smtp_use_ssl: bool = Field(
        default=False,
        description="SSL für SMTP verwenden"
    )

    # SMS Configuration (Twilio)
    sms_enabled: bool = Field(
        default=False,
        description="SMS-Versand aktivieren"
    )

    twilio_account_sid: str = Field(
        default="",
        description="Twilio Account SID"
    )

    twilio_auth_token: str = Field(
        default="",
        description="Twilio Auth Token"
    )

    twilio_from_number: str = Field(
        default="",
        description="Twilio Absender-Telefonnummer"
    )

    # Push Notifications
    push_notifications_enabled: bool = Field(
        default=False,
        description="Push-Benachrichtigungen aktivieren"
    )

    firebase_credentials_path: str = Field(
        default="",
        description="Pfad zu Firebase Credentials JSON"
    )

    firebase_project_id: str = Field(
        default="",
        description="Firebase Project ID"
    )

    # Webhook Notifications
    webhook_notifications_enabled: bool = Field(
        default=True,
        description="Webhook-Benachrichtigungen aktivieren"
    )

    webhook_timeout_seconds: int = Field(
        default=30,
        description="Timeout für Webhook-Calls in Sekunden"
    )

    webhook_retry_attempts: int = Field(
        default=3,
        description="Anzahl Wiederholungsversuche für Webhooks"
    )

    # Slack Integration
    slack_enabled: bool = Field(
        default=False,
        description="Slack-Integration aktivieren"
    )

    slack_webhook_url: str = Field(
        default="",
        description="Slack Webhook URL"
    )

    slack_channel: str = Field(
        default="#general",
        description="Standard Slack-Channel"
    )

    # Microsoft Teams Integration
    teams_enabled: bool = Field(
        default=False,
        description="Microsoft Teams Integration aktivieren"
    )

    teams_webhook_url: str = Field(
        default="",
        description="Microsoft Teams Webhook URL"
    )

    # Discord Integration
    discord_enabled: bool = Field(
        default=False,
        description="Discord-Integration aktivieren"
    )

    discord_webhook_url: str = Field(
        default="",
        description="Discord Webhook URL"
    )

    class Config:
        """Pydantic-Konfiguration."""
        env_prefix = "COMMUNICATION_"
        case_sensitive = False


def load_communication_settings() -> CommunicationSettings:
    """Lädt Communication Settings aus Umgebungsvariablen.

    Returns:
        CommunicationSettings-Instanz
    """
    return CommunicationSettings(
        smtp_enabled=get_env_str("SMTP_ENABLED", "false").lower() == "true",
        smtp_host=get_env_str("SMTP_HOST"),
        smtp_port=get_env_int("SMTP_PORT", DEFAULT_SMTP_PORT),
        smtp_user=get_env_str("SMTP_USER"),
        smtp_password=get_env_str("SMTP_PASSWORD"),
        smtp_from=get_env_str("SMTP_FROM"),
        smtp_use_tls=get_env_str("SMTP_USE_TLS", "true").lower() == "true",
        smtp_use_ssl=get_env_str("SMTP_USE_SSL", "false").lower() == "true",
        sms_enabled=get_env_str("SMS_ENABLED", "false").lower() == "true",
        twilio_account_sid=get_env_str("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=get_env_str("TWILIO_AUTH_TOKEN"),
        twilio_from_number=get_env_str("TWILIO_FROM_NUMBER"),
        push_notifications_enabled=get_env_str("PUSH_NOTIFICATIONS_ENABLED", "false").lower() == "true",
        firebase_credentials_path=get_env_str("FIREBASE_CREDENTIALS_PATH"),
        firebase_project_id=get_env_str("FIREBASE_PROJECT_ID"),
        webhook_notifications_enabled=get_env_str("WEBHOOK_NOTIFICATIONS_ENABLED", "true").lower() == "true",
        webhook_timeout_seconds=get_env_int("WEBHOOK_TIMEOUT_SECONDS", 30),
        webhook_retry_attempts=get_env_int("WEBHOOK_RETRY_ATTEMPTS", 3),
        slack_enabled=get_env_str("SLACK_ENABLED", "false").lower() == "true",
        slack_webhook_url=get_env_str("SLACK_WEBHOOK_URL"),
        slack_channel=get_env_str("SLACK_CHANNEL", "#general"),
        teams_enabled=get_env_str("TEAMS_ENABLED", "false").lower() == "true",
        teams_webhook_url=get_env_str("TEAMS_WEBHOOK_URL"),
        discord_enabled=get_env_str("DISCORD_ENABLED", "false").lower() == "true",
        discord_webhook_url=get_env_str("DISCORD_WEBHOOK_URL")
    )


# Globale Communication Settings Instanz
communication_settings = load_communication_settings()


__all__ = [
    "CommunicationSettings",
    "communication_settings",
    "load_communication_settings"
]
