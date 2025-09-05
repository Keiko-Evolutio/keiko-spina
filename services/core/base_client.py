"""Basis-Klassen für HTTP-Clients mit konsolidierter Funktionalität.

Konsolidiert duplizierte Patterns aus allen Client-Services.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from kei_logging import get_logger
from services.core.circuit_breaker import CircuitBreaker, CircuitPolicy
from services.core.constants import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OPEN_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT,
    HTTP_STATUS_OK,
    SERVICE_STATUS_AVAILABLE,
    SERVICE_STATUS_ERROR,
)

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class BaseClientConfig:
    """Basis-Konfiguration für alle HTTP-Clients."""

    base_url: str
    api_key: str | None = None
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = 3
    circuit_breaker_enabled: bool = True
    user_agent: str | None = None


class BaseHTTPClient(ABC):
    """Abstrakte Basis-Klasse für alle HTTP-Clients.

    Konsolidiert gemeinsame Funktionalität:
    - Circuit Breaker Integration
    - Standard HTTP-Client-Setup
    - Error-Handling-Patterns
    - Health-Check-Funktionalität
    """

    def __init__(self, config: BaseClientConfig, circuit_breaker_name: str) -> None:
        """Initialisiert den Basis-HTTP-Client.

        Args:
            config: Client-Konfiguration
            circuit_breaker_name: Name für Circuit Breaker
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None

        # Circuit Breaker Setup
        if config.circuit_breaker_enabled:
            self._circuit_breaker = CircuitBreaker(
                name=circuit_breaker_name,
                policy=CircuitPolicy(
                    failure_threshold=DEFAULT_FAILURE_THRESHOLD,
                    open_timeout_seconds=DEFAULT_OPEN_TIMEOUT_SECONDS,
                )
            )
        else:
            self._circuit_breaker = None

    async def __aenter__(self):
        """Async Context Manager Entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Stellt sicher, dass HTTP-Client initialisiert ist."""
        if self._client is not None:
            return

        headers = self._create_default_headers()

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
            http2=True,
        )

        logger.debug(f"HTTP-Client für {self.config.base_url} initialisiert")

    def _create_default_headers(self) -> dict[str, str]:
        """Erstellt Standard-Headers für HTTP-Requests."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.config.user_agent:
            headers["User-Agent"] = self.config.user_agent

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        return headers

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> httpx.Response:
        """Führt HTTP-Request mit Circuit Breaker aus.

        Args:
            method: HTTP-Methode
            url: Request-URL
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response

        Raises:
            httpx.HTTPError: Bei HTTP-Fehlern
            RuntimeError: Bei Circuit Breaker OPEN
        """
        await self._ensure_client()

        async def _request() -> httpx.Response:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        if self._circuit_breaker:
            return await self._circuit_breaker.call(_request)
        return await _request()

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET-Request."""
        return await self._make_request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST-Request."""
        return await self._make_request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """PUT-Request."""
        return await self._make_request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """DELETE-Request."""
        return await self._make_request("DELETE", url, **kwargs)

    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check durch.

        Returns:
            Health-Status-Dictionary
        """
        try:
            response = await self.get("/health")
            if response.status_code == HTTP_STATUS_OK:
                return {
                    "status": SERVICE_STATUS_AVAILABLE,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                }
        except Exception as e:
            logger.warning(f"Health-Check fehlgeschlagen für {self.config.base_url}: {e}")

        return {
            "status": SERVICE_STATUS_ERROR,
            "error": "Health-Check fehlgeschlagen",
        }

    async def close(self) -> None:
        """Schließt HTTP-Client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug(f"HTTP-Client für {self.config.base_url} geschlossen")

    @abstractmethod
    async def is_available(self) -> bool:
        """Prüft, ob der Service verfügbar ist.

        Returns:
            True wenn Service verfügbar ist
        """


class RetryableHTTPClient(BaseHTTPClient):
    """HTTP-Client mit Retry-Funktionalität.

    Erweitert BaseHTTPClient um automatische Retry-Logik.
    """

    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int | None = None,
        **kwargs: Any
    ) -> httpx.Response:
        """Führt HTTP-Request mit Retry-Logik aus.

        Args:
            method: HTTP-Methode
            url: Request-URL
            max_retries: Maximale Anzahl Wiederholungen
            **kwargs: Zusätzliche Request-Parameter

        Returns:
            HTTP-Response
        """
        retries = max_retries or self.config.max_retries
        last_exception = None

        for attempt in range(retries + 1):
            try:
                return await self._make_request(method, url, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < retries:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.debug(f"Request fehlgeschlagen (Versuch {attempt + 1}/{retries + 1}), "
                               f"Wiederholung in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.exception(f"Request nach {retries + 1} Versuchen fehlgeschlagen: {e}")

        raise last_exception

    async def get_with_retry(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET-Request mit Retry."""
        return await self._make_request_with_retry("GET", url, **kwargs)

    async def post_with_retry(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST-Request mit Retry."""
        return await self._make_request_with_retry("POST", url, **kwargs)

    async def is_available(self) -> bool:
        """Prüft, ob der Service verfügbar ist.

        Implementiert die abstrakte Methode der Basisklasse.
        Verwendet Health-Check mit Retry-Logik.

        Returns:
            True wenn Service verfügbar ist
        """
        try:
            health_status = await self.health_check()
            return health_status.get("status") == SERVICE_STATUS_AVAILABLE
        except Exception as e:
            logger.debug(f"Service-Verfügbarkeitsprüfung fehlgeschlagen: {e}")
            return False
