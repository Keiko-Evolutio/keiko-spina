# backend/services/clients/common/http_config.py
"""HTTP Client Konfiguration für Client Services.

Bietet standardisierte HTTP Client Konfiguration und Session-Management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiohttp
import httpx

from kei_logging import get_logger
from services.core.ssl_config import get_ssl_config

from .constants import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_CONNECTION_LIMIT,
    DEFAULT_CONNECTION_LIMIT_PER_HOST,
    DEFAULT_KEEPALIVE_TIMEOUT,
    DEFAULT_TIMEOUT,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class HTTPClientConfig:
    """Konfiguration für HTTP Clients."""

    timeout: float = DEFAULT_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    connection_limit: int = DEFAULT_CONNECTION_LIMIT
    connection_limit_per_host: int = DEFAULT_CONNECTION_LIMIT_PER_HOST
    keepalive_timeout: int = DEFAULT_KEEPALIVE_TIMEOUT
    trust_env: bool = True
    verify_ssl: bool = True
    headers: dict[str, str] | None = None


def create_aiohttp_session_config(
    config: HTTPClientConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Erstellt Konfiguration für aiohttp.ClientSession.

    Args:
        config: HTTP Client Konfiguration (optional)
        **overrides: Überschreibungen für spezifische Parameter

    Returns:
        Dictionary mit aiohttp Session-Konfiguration
    """
    if config is None:
        config = HTTPClientConfig()

    # SSL-Konfiguration
    get_ssl_config()

    # Connector-Konfiguration als Dictionary (wird später zu TCPConnector)
    connector_config = {
        "limit": config.connection_limit,
        "limit_per_host": config.connection_limit_per_host,
        "keepalive_timeout": config.keepalive_timeout,
        "enable_cleanup_closed": True,
        "verify_ssl": config.verify_ssl,
    }

    session_config = {
        "timeout": aiohttp.ClientTimeout(
            total=config.timeout,
            connect=config.connect_timeout
        ),
        "connector_config": connector_config,  # Speichere Config statt Connector
        "trust_env": config.trust_env,
    }

    # Headers hinzufügen falls vorhanden
    if config.headers:
        session_config["headers"] = config.headers

    # Überschreibungen anwenden
    session_config.update(overrides)

    logger.debug({
        "event": "aiohttp_session_config_created",
        "timeout": config.timeout,
        "connect_timeout": config.connect_timeout,
        "connection_limit": config.connection_limit,
        "verify_ssl": config.verify_ssl,
    })

    return session_config


def create_aiohttp_connector(connector_config: dict[str, Any]) -> aiohttp.TCPConnector:
    """Erstellt aiohttp.TCPConnector aus Konfiguration.

    Diese Funktion ist separat, damit Tests die Connector-Erstellung vermeiden können.

    Args:
        connector_config: Connector-Konfiguration

    Returns:
        Konfigurierter TCPConnector
    """
    return aiohttp.TCPConnector(**connector_config)


def create_httpx_client_config(
    config: HTTPClientConfig | None = None,
    base_url: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Erstellt Konfiguration für httpx.AsyncClient.

    Args:
        config: HTTP Client Konfiguration (optional)
        base_url: Basis-URL für den Client (optional)
        **overrides: Überschreibungen für spezifische Parameter

    Returns:
        Dictionary mit httpx Client-Konfiguration
    """
    if config is None:
        config = HTTPClientConfig()

    # SSL-Konfiguration
    ssl_config = get_ssl_config()

    client_config = {
        "timeout": config.timeout,
        "verify": ssl_config.verify_ssl,
        "limits": httpx.Limits(
            max_connections=config.connection_limit,
            max_keepalive_connections=config.connection_limit_per_host,
        ),
        "trust_env": config.trust_env,
    }

    # Base URL hinzufügen falls vorhanden
    if base_url:
        client_config["base_url"] = base_url

    # Headers hinzufügen falls vorhanden
    if config.headers:
        client_config["headers"] = config.headers

    # Überschreibungen anwenden
    client_config.update(overrides)

    logger.debug({
        "event": "httpx_client_config_created",
        "base_url": base_url,
        "timeout": config.timeout,
        "verify_ssl": ssl_config.verify_ssl,
        "connection_limit": config.connection_limit,
    })

    return client_config


def create_azure_headers(
    api_key: str,
    content_type: str = "application/json",
    additional_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Erstellt Standard-Headers für Azure API-Aufrufe.

    Args:
        api_key: Azure API-Schlüssel
        content_type: Content-Type Header (default: application/json)
        additional_headers: Zusätzliche Headers (optional)

    Returns:
        Dictionary mit HTTP-Headers
    """
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": content_type,
        "Accept": "application/json",
    }

    if additional_headers:
        headers.update(additional_headers)

    return headers


def create_openai_headers(
    api_key: str,
    additional_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Erstellt Standard-Headers für OpenAI API-Aufrufe.

    Args:
        api_key: OpenAI API-Schlüssel
        additional_headers: Zusätzliche Headers (optional)

    Returns:
        Dictionary mit HTTP-Headers
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if additional_headers:
        headers.update(additional_headers)

    return headers


def create_kei_rpc_headers(
    api_token: str,
    tenant_id: str,
    additional_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Erstellt Standard-Headers für KEI-RPC API-Aufrufe.

    Args:
        api_token: KEI-RPC API-Token
        tenant_id: Tenant-ID
        additional_headers: Zusätzliche Headers (optional)

    Returns:
        Dictionary mit HTTP-Headers
    """
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Tenant-Id": tenant_id,
    }

    if additional_headers:
        headers.update(additional_headers)

    return headers


class StandardHTTPClientConfig:
    """Factory-Klasse für Standard HTTP Client Konfigurationen."""

    @staticmethod
    def content_safety() -> HTTPClientConfig:
        """Konfiguration für Content Safety Client."""
        return HTTPClientConfig(
            timeout=10.0,
            connect_timeout=5.0,
        )

    @staticmethod
    def image_generation() -> HTTPClientConfig:
        """Konfiguration für Image Generation Service."""
        return HTTPClientConfig(
            timeout=60.0,  # Längere Timeouts für Bildgenerierung
            connect_timeout=10.0,
        )

    @staticmethod
    def deep_research() -> HTTPClientConfig:
        """Konfiguration für Deep Research Service."""
        return HTTPClientConfig(
            timeout=120.0,  # Sehr lange Timeouts für Research
            connect_timeout=10.0,
        )

    @staticmethod
    def kei_rpc() -> HTTPClientConfig:
        """Konfiguration für KEI-RPC Client."""
        return HTTPClientConfig(
            timeout=15.0,
            connect_timeout=5.0,
        )
