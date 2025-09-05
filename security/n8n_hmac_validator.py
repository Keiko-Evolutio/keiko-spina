"""HMAC-Validator für n8n Webhooks mit optionaler Secret-Auflösung via Azure Key Vault."""

from __future__ import annotations

import hmac
import time
from dataclasses import dataclass
from hashlib import sha256

from fastapi import HTTPException

from config.settings import settings
from kei_logging import get_logger

try:  # pragma: no cover - optional dependency
    from azure.identity.aio import DefaultAzureCredential
    from azure.keyvault.secrets.aio import SecretClient
    AZURE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AZURE_AVAILABLE = False


logger = get_logger(__name__)


@dataclass(slots=True)
class HmacValidationConfig:
    """Konfiguration für HMAC-Validierung."""

    timestamp_tolerance_seconds: int = 300
    key_vault_url: str | None = None
    key_vault_secret_name: str | None = None


class N8nHmacValidator:
    """Validiert Webhook-Signaturen und schützt vor Replay-Angriffen."""

    def __init__(self, config: HmacValidationConfig | None = None) -> None:
        # Konfiguration zusammenstellen
        self.config = config or HmacValidationConfig()

    async def _resolve_secret(self) -> str:
        """Lädt das HMAC-Secret aus Settings oder optional aus Azure Key Vault."""
        # Vorrang: Azure Key Vault, falls konfiguriert und verfügbar
        if self.config.key_vault_url and self.config.key_vault_secret_name and AZURE_AVAILABLE:
            try:
                credential = DefaultAzureCredential()
                client = SecretClient(vault_url=self.config.key_vault_url, credential=credential)
                secret = await client.get_secret(self.config.key_vault_secret_name)
                return secret.value
            except Exception as e:  # pragma: no cover - externe Umgebung
                logger.warning(f"KeyVault Secret-Auflösung fehlgeschlagen: {e}")

        # Fallback: Settings Secret
        return (
            settings.n8n_hmac_secret.get_secret_value()
            if hasattr(settings.n8n_hmac_secret, "get_secret_value")
            else str(settings.n8n_hmac_secret)
        )

    @staticmethod
    def _verify_signature(secret: str, payload: bytes, signature: str) -> bool:
        """Vergleicht Signaturen in konstanter Zeit."""
        expected = hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()
        return hmac.compare_digest(expected, signature or "")

    def _validate_timestamp(self, timestamp_header: str) -> None:
        """Validiert Zeitstempel."""
        try:
            ts = int(timestamp_header)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid timestamp header") from exc
        if abs(int(time.time()) - ts) > self.config.timestamp_tolerance_seconds:
            raise HTTPException(status_code=401, detail="Timestamp skew too large")

    async def validate(self, *, payload: bytes, signature: str, timestamp: str) -> None:
        """Führt vollständige HMAC-Validierung durch.

        Raises:
            HTTPException: bei fehlender/ungültiger Signatur oder Zeitabweichung
        """
        self._validate_timestamp(timestamp)
        secret = await self._resolve_secret()
        if not secret:
            raise HTTPException(status_code=500, detail="HMAC secret not configured")
        if not self._verify_signature(secret, payload, signature):
            raise HTTPException(status_code=401, detail="Invalid signature")


__all__ = ["HmacValidationConfig", "N8nHmacValidator"]
