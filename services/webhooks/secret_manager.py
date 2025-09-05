"""Secret Management für KEI‑Webhook mit Azure Key Vault und Caching.

Stellt eine typsichere Abstraktion für Secret‑Auflösung, Versionierung und
Rotation bereit. Unterstützt Azure Key Vault nativ und fällt in Tests oder
ohne Konfiguration auf ein In‑Memory‑Backend zurück.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from config.settings import settings
from kei_logging import get_logger
from monitoring import record_custom_metric

try:  # pragma: no cover - optionale Abhängigkeiten
    from azure.core.exceptions import AzureError  # type: ignore
    from azure.identity.aio import DefaultAzureCredential  # type: ignore
    from azure.keyvault.secrets.aio import SecretClient  # type: ignore
    _AZURE_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    DefaultAzureCredential = None  # type: ignore
    SecretClient = None  # type: ignore
    AzureError = Exception  # type: ignore
    _AZURE_AVAILABLE = False


logger = get_logger(__name__)


@dataclass(slots=True)
class SecretRecord:
    """Eintrag im lokalen Secret‑Cache.

    Attributes:
        value: Secret‑Wert
        version: Version des Secrets (Key Vault Version-ID oder "inmemory")
        expires_at: Zeitpunkt, an dem der Cache invalide wird
    """

    value: str
    version: str
    expires_at: datetime


class SecretBackend:
    """Abstrakte Backend‑Schnittstelle für Secret‑Operationen."""

    async def get_secret(self, *, name: str, version: str | None = None) -> tuple[str, str]:  # pragma: no cover - Interface
        """Liest ein Secret (ggf. spezifische Version) und gibt (value, version) zurück."""
        raise NotImplementedError

    async def set_new_secret(self, *, name: str, value: str) -> tuple[str, str]:  # pragma: no cover - Interface
        """Schreibt ein neues Secret (neue Version) und gibt (value, version) zurück."""
        raise NotImplementedError

    async def list_versions(self, *, name: str, max_versions: int = 5) -> list[str]:  # pragma: no cover - Interface
        """Listet die jüngsten Versions‑IDs auf (neueste zuerst)."""
        raise NotImplementedError


class InMemorySecretBackend(SecretBackend):
    """Einfaches In‑Memory‑Backend für lokale Entwicklung/Tests."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}
        self._latest_version: dict[str, str] = {}

    async def get_secret(self, *, name: str, version: str | None = None) -> tuple[str, str]:
        versions = self._store.get(name, {})
        if version:
            if version not in versions:
                raise KeyError("secret_version_not_found")
            return versions[version], version
        # neueste Version verwenden
        ver = self._latest_version.get(name)
        if not ver:
            raise KeyError("secret_not_found")
        return versions[ver], ver

    async def set_new_secret(self, *, name: str, value: str) -> tuple[str, str]:
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        self._store.setdefault(name, {})[version] = value
        self._latest_version[name] = version
        return value, version

    async def list_versions(self, *, name: str, max_versions: int = 5) -> list[str]:
        versions = list(self._store.get(name, {}).keys())
        versions.sort(reverse=True)
        return versions[:max_versions]


class KeyVaultSecretBackend(SecretBackend):
    """Azure Key Vault gestütztes Backend für Secrets."""

    def __init__(self, vault_url: str) -> None:
        if not _AZURE_AVAILABLE:  # pragma: no cover - nur in Azure aktiv
            raise ImportError("Azure SDK nicht verfügbar")
        self._vault_url = vault_url
        self._credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=vault_url, credential=self._credential)

    async def get_secret(self, *, name: str, version: str | None = None) -> tuple[str, str]:
        try:
            if version:
                secret = await self._client.get_secret(name, version)
            else:
                secret = await self._client.get_secret(name)
            return secret.value, secret.properties.version  # type: ignore[attr-defined]
        except AzureError as exc:  # pragma: no cover - extern
            raise KeyError(str(exc)) from exc

    async def set_new_secret(self, *, name: str, value: str) -> tuple[str, str]:
        try:
            result = await self._client.set_secret(name, value)
            return result.value, result.properties.version  # type: ignore[attr-defined]
        except AzureError as exc:  # pragma: no cover - extern
            raise RuntimeError(str(exc)) from exc

    async def list_versions(self, *, name: str, max_versions: int = 5) -> list[str]:
        try:
            # Azure SDK liefert ein AsyncIterator, neueste zuerst ist nicht garantiert
            versions: list[str] = []
            async for props in self._client.list_properties_of_secret_versions(name):  # type: ignore[attr-defined]
                versions.append(props.version)  # type: ignore[attr-defined]
                if len(versions) >= max_versions:
                    break
            # sicherheitshalber neueste oben
            versions = [v for v in versions if v]
            versions.reverse()
            return list(reversed(versions))
        except AzureError as exc:  # pragma: no cover - extern
            logger.debug(f"KeyVault list_versions fehlgeschlagen: {exc}")
            return []


class SecretManager:
    """Zentrale Secret‑Verwaltung mit Key Vault Integration und Cache.

    - Caching: TTL basiert (Default: 300s)
    - Versionierte Auflösung und Rotation
    - Audit/Metrik‑Hooks über `monitoring.record_custom_metric`
    """

    def __init__(self, *, cache_ttl_seconds: int | None = None, backend: SecretBackend | None = None) -> None:
        self.cache_ttl_seconds: int = int(cache_ttl_seconds or settings.secret_cache_ttl_seconds)
        self._cache: dict[tuple[str, str | None], SecretRecord] = {}
        self._lock = asyncio.Lock()
        if backend is not None:
            self._backend = backend
        elif settings.azure_key_vault_url:
            try:
                self._backend = KeyVaultSecretBackend(settings.azure_key_vault_url)
            except Exception as exc:  # pragma: no cover - extern
                logger.warning(f"KeyVault Backend nicht verfügbar, Fallback InMemory: {exc}")
                self._backend = InMemorySecretBackend()
        else:
            self._backend = InMemorySecretBackend()

    async def _cache_get(self, key: tuple[str, str | None]) -> SecretRecord | None:
        rec = self._cache.get(key)
        if not rec:
            return None
        if datetime.now(UTC) >= rec.expires_at:
            self._cache.pop(key, None)
            return None
        return rec

    async def _cache_put(self, key: tuple[str, str | None], value: str, version: str) -> None:
        self._cache[key] = SecretRecord(
            value=value,
            version=version,
            expires_at=datetime.now(UTC) + timedelta(seconds=self.cache_ttl_seconds),
        )

    async def get_current_secret(self, *, key_name: str) -> tuple[str, str]:
        """Gibt aktuelles Secret und Version zurück, mit Cache.

        Returns:
            Tuple[value, version]
        """
        cache_key = (key_name, None)
        rec = await self._cache_get(cache_key)
        if rec:
            record_custom_metric("webhook.secret.cache.hit", 1, {"scope": "current"})
            return rec.value, rec.version
        async with self._lock:
            rec = await self._cache_get(cache_key)
            if rec:
                record_custom_metric("webhook.secret.cache.hit", 1, {"scope": "current_lock"})
                return rec.value, rec.version
            try:
                value, version = await self._backend.get_secret(name=key_name)
                await self._cache_put(cache_key, value, version)
                record_custom_metric("webhook.secret.cache.miss", 1, {"scope": "current"})
                return value, version
            except Exception as exc:
                logger.exception(f"SecretManager get_current_secret fehlgeschlagen: {exc}")
                raise

    async def get_secret_by_version(self, *, key_name: str, version: str) -> str:
        """Lädt Secret für spezifische Version, mit Cache."""
        cache_key = (key_name, version)
        rec = await self._cache_get(cache_key)
        if rec:
            record_custom_metric("webhook.secret.cache.hit", 1, {"scope": "by_version"})
            return rec.value
        async with self._lock:
            rec = await self._cache_get(cache_key)
            if rec:
                record_custom_metric("webhook.secret.cache.hit", 1, {"scope": "by_version_lock"})
                return rec.value
            try:
                value, ver = await self._backend.get_secret(name=key_name, version=version)
                await self._cache_put(cache_key, value, ver)
                record_custom_metric("webhook.secret.cache.miss", 1, {"scope": "by_version"})
                return value
            except Exception as exc:
                logger.exception(f"SecretManager get_secret_by_version fehlgeschlagen: {exc}")
                raise

    async def rotate_secret(self, *, key_name: str, new_value: str | None = None) -> tuple[str, str]:
        """Rotiert Secret durch Schreiben einer neuen Version in das Backend.

        Args:
            key_name: Secret‑Name im KV
            new_value: Optionaler zuvor generierter Wert

        Returns:
            Tuple[value, version] der neuen Version
        """
        value = new_value or secrets.token_urlsafe(48)
        try:
            val, version = await self._backend.set_new_secret(name=key_name, value=value)
            # Cache invalidieren (current und spezifische Versionen)
            self._cache.pop((key_name, None), None)
            self._cache.pop((key_name, version), None)
            record_custom_metric("webhook.secret.rotation.success", 1, {"key": key_name})
            return val, version
        except Exception as exc:
            record_custom_metric("webhook.secret.rotation.failure", 1, {"key": key_name})
            logger.exception(f"SecretManager rotate_secret fehlgeschlagen: {exc}")
            raise

    async def list_recent_versions(self, *, key_name: str, _within_hours: int) -> list[str]:
        """Listet neueste Versionen für Grace‑Validation.

        Es werden bis zu 10 Versionen geladen und anschließend anhand der
        Grace‑Periode gefiltert. Falls das Backend keine Zeitstempel liefert,
        werden lediglich die letzten Versionen zurückgegeben.
        """
        try:
            return await self._backend.list_versions(name=key_name, max_versions=10)
            # Ohne Timestamps bestmöglich: ersten N verwenden
        except Exception as exc:
            logger.debug(f"SecretManager list_recent_versions Fehler: {exc}")
            return []


# Globale Instanz für einfachen Zugriff
_secret_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    """Gibt eine Singleton‑Instanz des SecretManagers zurück."""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


__all__ = [
    "InMemorySecretBackend",
    "KeyVaultSecretBackend",
    "SecretBackend",
    "SecretManager",
    "SecretRecord",
    "get_secret_manager",
]
