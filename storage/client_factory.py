# backend/storage/client_factory.py
"""Client-Factory für Storage-System mit Singleton-Pattern und Lifecycle-Management."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

try:
    from azure.core.credentials_async import AsyncTokenCredential  # type: ignore
    from azure.identity.aio import ClientSecretCredential, DefaultAzureCredential  # type: ignore
    from azure.storage.blob.aio import BlobServiceClient  # type: ignore
    _AZURE_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncTokenCredential = Any  # type: ignore
    BlobServiceClient = Any  # type: ignore
    def default_azure_credential(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("azure-identity ist nicht installiert")
    def client_secret_credential(*args: Any, **kwargs: Any):  # type: ignore
        raise ImportError("azure-identity ist nicht installiert")
    # Aliases für Kompatibilität
    DefaultAzureCredential = default_azure_credential  # type: ignore
    ClientSecretCredential = client_secret_credential  # type: ignore
    _AZURE_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    aioredis = Any  # type: ignore
    _REDIS_AVAILABLE = False

from config.settings import settings
from kei_logging import get_logger
from services.core.ssl_config import get_ssl_config

from .constants import CacheConfig, ErrorMessages

logger = get_logger(__name__)


class ClientFactory:
    """Singleton-Factory für Storage-Clients mit Lifecycle-Management."""

    _instance: ClientFactory | None = None
    _azure_credential: AsyncTokenCredential | None = None
    _blob_client: BlobServiceClient | None = None
    _redis_client: aioredis.Redis | None = None

    def __new__(cls) -> ClientFactory:
        """Singleton-Pattern Implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_azure_clients(self) -> tuple[AsyncTokenCredential, BlobServiceClient]:
        """Gibt Azure-Clients zurück (Credential + BlobServiceClient)."""
        if not _AZURE_AVAILABLE:
            raise ImportError(ErrorMessages.AZURE_SDK_NOT_AVAILABLE)

        if self._azure_credential is None:
            # Verwende App Registration wenn verfügbar, sonst DefaultAzureCredential
            if (hasattr(settings, "azure_client_id") and
                hasattr(settings, "azure_client_secret") and
                hasattr(settings, "azure_tenant_id") and
                settings.azure_client_id and
                settings.azure_client_secret and
                settings.azure_tenant_id):

                logger.info("Verwende Azure App Registration für Storage Authentication")

                # SSL-Konfiguration für Entwicklungsumgebung
                credential_kwargs = {
                    "tenant_id": settings.azure_tenant_id,
                    "client_id": settings.azure_client_id,
                    "client_secret": settings.azure_client_secret
                }

                # SSL-Verifikation für Development deaktivieren falls konfiguriert
                if (settings.environment == "development" and
                    hasattr(settings.azure, "azure_ssl_verify") and
                    not settings.azure.azure_ssl_verify):

                    logger.debug("SSL-Verifikation für Azure deaktiviert (Development-Modus)")
                    # Für Azure Identity SDK: SSL-Verifikation deaktivieren
                    credential_kwargs["connection_verify"] = False

                self._azure_credential = ClientSecretCredential(**credential_kwargs)
            else:
                logger.info("Verwende DefaultAzureCredential für Storage Authentication")
                self._azure_credential = DefaultAzureCredential()

        if self._blob_client is None:
            ssl_config = get_ssl_config()
            self._blob_client = BlobServiceClient(
                account_url=str(settings.storage_account_url),
                credential=self._azure_credential,
                connection_verify=ssl_config.verify_ssl
            )
            logger.debug({
                "event": "blob_client_ready",
                "account_url": str(settings.storage_account_url)
            })

        return self._azure_credential, self._blob_client

    async def get_redis_client(self) -> aioredis.Redis:
        """Gibt Redis-Client zurück mit automatischem Fallback."""
        if not _REDIS_AVAILABLE:
            logger.warning("Redis SDK nicht installiert")
            return NoOpCache()  # type: ignore

        if self._redis_client is None:
            try:
                config = self._build_redis_config()
                self._redis_client = aioredis.Redis(**config)
                await self._redis_client.ping()
                logger.info("✅ Redis-Verbindung erfolgreich")
            except Exception as e:
                logger.warning(f"⚠️ {ErrorMessages.REDIS_NOT_AVAILABLE}: {e}")
                self._redis_client = NoOpCache()  # type: ignore

        return self._redis_client

    def _build_redis_config(self) -> dict[str, Any]:
        """Erstellt Redis-Konfiguration aus ENV/Settings."""
        url = os.environ.get("REDIS_URL", "").strip()
        if url:
            try:
                parsed = urlparse(url)
                return {
                    "host": parsed.hostname or "localhost",
                    "port": int(parsed.port or 6379),
                    "db": int(parsed.path.lstrip("/") or "0"),
                    **CacheConfig.REDIS_CONFIG_DEFAULTS
                }
            except Exception:
                pass  # Fallback auf Settings

        return {
            "host": getattr(settings, "redis_host", "localhost"),
            "port": getattr(settings, "redis_port", 6379),
            "db": getattr(settings, "redis_db", 0),
            **CacheConfig.REDIS_CONFIG_DEFAULTS
        }

    async def close_all_clients(self) -> None:
        """Schließt alle Clients und räumt Ressourcen auf."""
        # Redis-Client schließen
        if self._redis_client and not isinstance(self._redis_client, NoOpCache):
            try:
                if hasattr(self._redis_client, "aclose"):
                    await self._redis_client.aclose()  # type: ignore
                elif hasattr(self._redis_client, "close"):
                    await self._redis_client.close()  # type: ignore
            except Exception as e:
                logger.warning(f"Fehler beim Schließen der Redis-Verbindung: {e}")
            finally:
                self._redis_client = None
                logger.info("🔌 Redis-Verbindung geschlossen")

        # Azure-Clients schließen
        if self._blob_client is not None:
            await self._blob_client.close()
            self._blob_client = None

        if self._azure_credential is not None:
            await self._azure_credential.close()
            self._azure_credential = None

        logger.info("🔌 Alle Storage-Clients geschlossen")


class NoOpCache:
    """Fallback-Cache ohne Redis-Funktionalität."""

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        return True

    async def setex(self, key: str, time: int, value: Any) -> bool:
        return True

    async def delete(self, *keys: str) -> int:
        return len(keys)

    async def mget(self, keys: list[str]) -> list[None]:
        return [None] * len(keys)

    async def rpop(self, key: str) -> None:
        """Redis RPOP fallback - returns None (no data)."""
        return

    async def lpush(self, key: str, *values: Any) -> int:
        """Redis LPUSH fallback - returns count of values."""
        return len(values)

    async def llen(self, key: str) -> int:
        """Redis LLEN fallback - returns 0 (empty list)."""
        return 0

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass

    def pipeline(self):
        return self

    async def execute(self):
        return []


# Globale Factory-Instanz
client_factory = ClientFactory()
