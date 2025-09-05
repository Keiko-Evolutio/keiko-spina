# backend/services/clients/clients.py
"""Vereinfachte Client-Implementierung mit Session-Management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from config.settings import get_settings
from kei_logging import get_logger

from .session_manager import cleanup_all_sessions, managed_session

try:
    import httpx
    _HTTPX_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _HTTPX_AVAILABLE = False

logger = get_logger(__name__)

# =====================================================================
# Feature Detection
# =====================================================================

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    aiohttp = None  # Fallback für aiohttp import
    _AIOHTTP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    _AZURE_AVAILABLE = True
except ImportError:  # pragma: no cover - optionale Abhängigkeit
    _AZURE_AVAILABLE = False
    DefaultAzureCredential = None

try:
    # Azure AI Foundry Deep Research (optional, mit Fallback)
    from azure.ai.projects import AIProjectClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore
    _AZURE_AI_AVAILABLE = True
except ImportError:  # pragma: no cover - optionale Abhängigkeit
    _AZURE_AI_AVAILABLE = False
    AIProjectClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore


# =====================================================================
# HTTP Client
# =====================================================================

class HTTPClient:
    """HTTP Client mit Session-Manager."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        cfg = get_settings()
        self._http2_enabled = bool(getattr(cfg, "webhook_http2_enabled", False))
        self._pool_limits = None
        if _HTTPX_AVAILABLE:
            self._pool_limits = httpx.Limits(
                max_connections=int(getattr(cfg, "webhook_http2_max_connections", 100)),
                max_keepalive_connections=int(getattr(cfg, "webhook_http2_max_keepalive", 20)),
            )

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Erstellt verwaltete HTTP-Session.

        Gibt einen asynchronen Kontextmanager zurück, der eine verwaltete
        HTTP-Session bereitstellt und nach der Nutzung wieder korrekt schließt.
        """
        # Bevorzugt httpx mit HTTP/2 (wenn aktiviert/verfügbar), sonst aiohttp
        if _HTTPX_AVAILABLE and self._http2_enabled:
            async with httpx.AsyncClient(timeout=self.timeout, http2=True, limits=self._pool_limits, headers={"User-Agent": "Keiko-Webhook/2"}) as client:  # type: ignore[arg-type]
                yield client
            return
        if _AIOHTTP_AVAILABLE:
            config = {"timeout": aiohttp.ClientTimeout(total=self.timeout), **kwargs}
            async with managed_session(**config) as session:
                yield session
            return
        raise RuntimeError("Weder httpx noch aiohttp verfügbar")

    async def get(self, url: str, **kwargs: Any) -> Any:
        """HTTP GET Request."""
        async with self.session() as session, session.get(url, **kwargs) as response:
            return await response.json()

    async def post(self, url: str, **kwargs: Any) -> Any:
        """HTTP POST Request."""
        async with self.session() as session:
            # httpx Client hat andere API als aiohttp
            if _HTTPX_AVAILABLE and isinstance(session, httpx.AsyncClient):
                resp = await session.post(url, **kwargs)
                return resp.json()
            # Mappe httpx-Parameter 'content' auf aiohttp 'data'
            if "content" in kwargs and "data" not in kwargs:
                kwargs = {**kwargs}
                kwargs["data"] = kwargs.pop("content")
            async with session.post(url, **kwargs) as response:
                return await response.json()


# =====================================================================
# Services Management
# =====================================================================

class Services:
    """Vereinfachte Services-Klasse."""

    def __init__(self) -> None:
        self._http_client = HTTPClient() if _AIOHTTP_AVAILABLE else None
        self._azure_credential = DefaultAzureCredential() if _AZURE_AVAILABLE else None
        # Azure AI Foundry Projekt-Client mit API-Key Fallback
        self._ai_project_client = self._create_ai_project_client()

    @property
    def http_session(self) -> HTTPClient | None:
        """HTTP Client zugriff."""
        return self._http_client

    @property
    def azure_credential(self) -> Any | None:
        """Azure Credential zugriff."""
        return self._azure_credential

    @property
    def ai_project_client(self) -> Any | None:
        """Azure AI Foundry Projekt-Client Zugriff (Fallback-fähig)."""
        return self._ai_project_client

    async def cleanup(self) -> None:
        """Bereinigt alle Service-Ressourcen."""
        await cleanup_all_sessions()

    # =====================================================================
    # Private Hilfsfunktionen
    # =====================================================================
    def _create_ai_project_client(self) -> Any | None:
        """Erstellt AIProjectClient basierend auf Konfiguration.

        Bevorzugt API-Key Credential, fällt auf DefaultAzureCredential zurück.
        """
        if not _AZURE_AI_AVAILABLE:
            return None

        try:
            # Konfiguration laden
            from config.settings import settings

            endpoint = getattr(settings, "project_keiko_services_endpoint", "")
            api_key = getattr(settings, "project_keiko_api_key", "")

            if not endpoint:
                return None

            # API-Key bevorzugen (Enterprise: Service Principals/Managed Identities via DefaultAzureCredential)
            if api_key:
                credential = AzureKeyCredential(api_key)  # type: ignore
            else:
                if not self._azure_credential:
                    return None
                credential = self._azure_credential

            return AIProjectClient(endpoint=endpoint, credential=credential)  # type: ignore
        except Exception as e:  # pragma: no cover - defensiv
            logger.warning(f"AIProjectClient konnte nicht erstellt werden: {e}")
            return None


# =====================================================================
# Global Service Access
# =====================================================================

def _get_services() -> Services | None:
    """Holt globale Services-Instanz."""
    try:
        return Services()
    except Exception as e:
        logger.debug(f"Services init fehlgeschlagen: {e}")
        return None


# Export-Funktionen
def http_client() -> HTTPClient | None:
    """HTTP Client Export."""
    svc = _get_services()
    return svc.http_session if svc else None


def foundry_credential() -> Any | None:
    """Azure Credential Export."""
    svc = _get_services()
    return svc.azure_credential if svc else None





# Auto-initialize
_services: Services | None = _get_services()
