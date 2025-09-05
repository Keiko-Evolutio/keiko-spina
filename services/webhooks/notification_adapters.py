"""E-Mail und SMS Adapter für Alert-Dispatch (optionale Abhängigkeiten).

Adapter sind bewusst minimal gehalten und werden nur erstellt, wenn
Konfigurationen vorhanden sind. Fehler werden nach oben gereicht.
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass
class SMTPEmailAdapter:
    """Einfacher SMTP E-Mail Adapter.

    Attributes:
        host: SMTP Hostname
        port: SMTP Port
        username: Optionaler Benutzername
        password: Optionales Passwort
        sender: Absender-Adresse
        use_tls: Ob STARTTLS genutzt werden soll
    """

    host: str
    port: int
    sender: str
    username: str | None = None
    password: str | None = None
    use_tls: bool = True

    async def send_email(self, subject: str, body: str, severity: str) -> None:
        # Einfacher synchroner Versand in Thread-Executor wäre ideal; hier minimal synchron
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = self.sender
        msg["Subject"] = subject
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP(self.host, self.port, timeout=10) as server:
            if self.use_tls:
                server.starttls(context=context)
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(msg)


@dataclass
class TwilioConfig:
    """Konfiguration für Twilio SMS."""

    account_sid: str
    auth_token: str
    from_number: str


@dataclass
class TwilioSMSAdapter:
    """Minimaler Twilio SMS Adapter (HTTP API via twilio RestClient optional).

    Für die Implementierung ohne externe Abhängigkeit wird ein einfacher
    HTTP-Aufruf erwartet; im Produktionscode sollte der offizielle Client
    genutzt werden. Hier wird absichtlich ein no-op gezeigt, um Abhängigkeiten
    zu vermeiden. Ersetzen Sie dies durch Ihren bevorzugten SMS-Provider.
    """

    config: TwilioConfig

    async def send_sms(self, text: str, severity: str) -> None:
        # Platzhalter: Integrieren Sie hier die HTTP API Ihres SMS-Providers
        # Für Tests genügt, dass kein Fehler geworfen wird.
        return


__all__ = [
    "SMTPEmailAdapter",
    "TwilioConfig",
    "TwilioSMSAdapter",
]
