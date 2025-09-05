"""Konsolidierte HTTP-Client-Basis für KEI MCP.

Diese Klasse vereint die HTTP-Funktionalität aus kei_mcp_client.py und
unified_mcp_client.py und eliminiert Code-Duplikation.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from kei_logging import get_logger

from .constants import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_CONNECTION_POOL_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    HEADERS,
    SSL_CONFIG,
    STANDARD_HEADERS,
)
from .exceptions import (
    ErrorContext,
    KEIMCPConnectionError,
    KEIMCPError,
    KEIMCPTimeoutError,
    create_error_context,
    handle_http_error,
)
from .utils import (
    format_duration_ms,
    generate_correlation_id,
    generate_request_id,
    normalize_url,
    validate_url,
)

logger = get_logger(__name__)


@dataclass
class HTTPClientConfig:
    """Konfiguration für HTTP-Client."""

    base_url: str
    server_name: str
    api_key: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
    retry_backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR
    custom_headers: dict[str, str] | None = None
    verify_ssl: bool = True
    ssl_context: ssl.SSLContext | None = None
    http2_enabled: bool = True
    connection_pool_size: int = DEFAULT_CONNECTION_POOL_SIZE

    def __post_init__(self):
        """Validiert Konfiguration nach Initialisierung."""
        if not validate_url(self.base_url):
            raise ValueError(f"Ungültige Base-URL: {self.base_url}")

        if self.timeout_seconds <= 0:
            raise ValueError("Timeout muss positiv sein")

        if self.max_retries < 0:
            raise ValueError("Max-Retries darf nicht negativ sein")


@dataclass
class RequestMetrics:
    """Metriken für HTTP-Requests."""

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_seconds: float | None = None
    status_code: int | None = None
    response_size_bytes: int | None = None
    retry_count: int = 0
    error: Exception | None = None

    def finish(self, status_code: int | None = None, response_size: int | None = None):
        """Markiert Request als beendet und berechnet Metriken."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.status_code = status_code
        self.response_size_bytes = response_size


class BaseHTTPClient:
    """Konsolidierte HTTP-Client-Basis für KEI MCP.

    Vereint die HTTP-Funktionalität aus verschiedenen MCP-Clients und
    bietet einheitliche Schnittstelle für HTTP-Kommunikation.
    """

    def __init__(self, config: HTTPClientConfig):
        """Initialisiert HTTP-Client.

        Args:
            config: HTTP-Client-Konfiguration
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._session_headers: dict[str, str] = {}

        # Request-Tracking
        self._active_requests: dict[str, RequestMetrics] = {}

        logger.info(f"HTTP-Client initialisiert für {config.server_name}")

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

        # SSL-Konfiguration
        ssl_context = self._create_ssl_context()

        # Timeouts konfigurieren
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
            write=self.config.timeout_seconds,
            pool=self.config.timeout_seconds
        )

        # Connection Limits
        limits = httpx.Limits(
            max_connections=self.config.connection_pool_size,
            max_keepalive_connections=20,
            keepalive_expiry=5.0
        )

        # Session-Headers vorbereiten
        self._session_headers = self._prepare_session_headers()

        # HTTP-Client erstellen
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=timeout,
            limits=limits,
            verify=ssl_context if self.config.verify_ssl else False,
            http2=self.config.http2_enabled,
            headers=self._session_headers,
            follow_redirects=True
        )

        # Semaphore für Concurrency-Control
        self._semaphore = asyncio.Semaphore(self.config.connection_pool_size)

        logger.debug(f"HTTP-Client konfiguriert: HTTP/2={self.config.http2_enabled}, SSL={self.config.verify_ssl}")

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        """Erstellt SSL-Kontext basierend auf Konfiguration."""
        if self.config.ssl_context:
            return self.config.ssl_context

        if not self.config.verify_ssl:
            return None

        # Standard SSL-Kontext erstellen
        context = ssl.create_default_context()

        # SSL-Konfiguration anwenden
        context.check_hostname = SSL_CONFIG["CHECK_HOSTNAME"]
        context.verify_mode = ssl.CERT_REQUIRED if SSL_CONFIG["VERIFY_MODE"] else ssl.CERT_NONE

        # Minimum TLS-Version setzen
        if SSL_CONFIG["MINIMUM_VERSION"] == "TLSv1.2":
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        elif SSL_CONFIG["MINIMUM_VERSION"] == "TLSv1.3":
            context.minimum_version = ssl.TLSVersion.TLSv1_3

        # Cipher-Suite konfigurieren
        try:
            context.set_ciphers(SSL_CONFIG["CIPHERS"])
        except ssl.SSLError as e:
            logger.warning(f"Cipher-Konfiguration fehlgeschlagen: {e}")

        return context

    def _prepare_session_headers(self) -> dict[str, str]:
        """Bereitet Session-Headers vor."""
        headers = STANDARD_HEADERS.copy()

        # API-Key hinzufügen
        if self.config.api_key:
            headers[HEADERS["API_KEY"]] = self.config.api_key

        # Custom Headers hinzufügen
        if self.config.custom_headers:
            headers.update(self.config.custom_headers)

        return headers

    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | str | bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        context: ErrorContext | None = None
    ) -> httpx.Response:
        """Führt HTTP-Request mit Retry-Logik und Error-Handling durch.

        Args:
            method: HTTP-Methode
            endpoint: Endpoint-Pfad
            data: Request-Body
            params: Query-Parameter
            headers: Zusätzliche Headers
            timeout: Request-Timeout
            context: Error-Kontext

        Returns:
            HTTP-Response

        Raises:
            KEIMCPError: Bei Request-Fehlern
        """
        await self._ensure_client()

        # Request-ID generieren
        request_id = generate_request_id()
        correlation_id = generate_correlation_id()

        # Error-Kontext erweitern
        if context is None:
            context = create_error_context()
        context.request_id = request_id
        context.correlation_id = correlation_id
        context.server_name = self.config.server_name

        # URL normalisieren
        url = normalize_url(self.config.base_url, endpoint)

        # Request-Headers vorbereiten
        request_headers = self._prepare_request_headers(headers, correlation_id)

        # Request-Metriken initialisieren
        metrics = RequestMetrics()
        self._active_requests[request_id] = metrics

        try:
            # Request mit Retry-Logik ausführen
            response = await self._execute_request_with_retries(
                method=method,
                url=url,
                data=data,
                params=params,
                headers=request_headers,
                timeout=timeout or self.config.timeout_seconds,
                context=context,
                metrics=metrics
            )

            # Erfolgreiche Response verarbeiten
            metrics.finish(response.status_code, len(response.content))

            logger.debug(
                f"Request erfolgreich: {method} {endpoint} "
                f"({response.status_code}, {format_duration_ms(metrics.duration_seconds)})"
            )

            return response

        except Exception as e:
            metrics.error = e
            metrics.finish()

            logger.exception(
                f"Request fehlgeschlagen: {method} {endpoint} "
                f"({format_duration_ms(metrics.duration_seconds)}) - {e}"
            )

            # Exception zu KEI MCP Error konvertieren
            if isinstance(e, KEIMCPError):
                raise
            if isinstance(e, httpx.TimeoutException):
                raise KEIMCPTimeoutError(context=context, cause=e)
            if isinstance(e, httpx.ConnectError):
                raise KEIMCPConnectionError(context=context, cause=e)
            raise KEIMCPError(f"Unerwarteter Request-Fehler: {e}", context=context, cause=e)

        finally:
            # Request-Tracking bereinigen
            self._active_requests.pop(request_id, None)

    def _prepare_request_headers(
        self,
        additional_headers: dict[str, str] | None,
        correlation_id: str
    ) -> dict[str, str]:
        """Bereitet Request-Headers vor."""
        # Dictionary literal mit Correlation-ID initialisieren
        headers = {HEADERS["CORRELATION_ID"]: correlation_id}

        # Zusätzliche Headers hinzufügen
        if additional_headers:
            headers.update(additional_headers)

        return headers

    async def _execute_request_with_retries(
        self,
        method: str,
        url: str,
        data: dict[str, Any] | str | bytes | None,
        params: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
        context: ErrorContext,
        metrics: RequestMetrics
    ) -> httpx.Response:
        """Führt Request mit Retry-Logik aus."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Semaphore für Concurrency-Control
                async with self._semaphore:
                    response = await self._client.request(
                        method=method,
                        url=url,
                        json=data if isinstance(data, dict) else None,
                        content=data if isinstance(data, str | bytes) else None,
                        params=params,
                        headers=headers,
                        timeout=timeout
                    )

                # HTTP-Fehler-Status prüfen
                if response.status_code >= 400:
                    http_error = handle_http_error(
                        response.status_code,
                        response.text,
                        context
                    )

                    # Bei 5xx-Fehlern Retry versuchen
                    if response.status_code >= 500 and attempt < self.config.max_retries:
                        last_exception = http_error
                        metrics.retry_count += 1
                        await self._wait_for_retry(attempt)
                        continue

                    raise http_error

                return response

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    metrics.retry_count += 1
                    logger.warning(
                        f"Request-Fehler (Versuch {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    await self._wait_for_retry(attempt)
                    continue

                raise

        # Alle Retries fehlgeschlagen
        if last_exception:
            raise last_exception
        # Fallback für Type-Safety - sollte nie erreicht werden
        raise RuntimeError("Unerwarteter Zustand: Keine Exception und keine Response")

    async def _wait_for_retry(self, attempt: int) -> None:
        """Wartet vor Retry mit exponential backoff."""
        delay = self.config.retry_delay_seconds * (self.config.retry_backoff_factor ** attempt)
        await asyncio.sleep(delay)

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        context: ErrorContext | None = None
    ) -> httpx.Response:
        """GET-Request."""
        return await self.make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            headers=headers,
            timeout=timeout,
            context=context
        )

    async def post(
        self,
        endpoint: str,
        data: dict[str, Any] | str | bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        context: ErrorContext | None = None
    ) -> httpx.Response:
        """POST-Request."""
        return await self.make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
            context=context
        )

    async def put(
        self,
        endpoint: str,
        data: dict[str, Any] | str | bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        context: ErrorContext | None = None
    ) -> httpx.Response:
        """PUT-Request."""
        return await self.make_request(
            method="PUT",
            endpoint=endpoint,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
            context=context
        )

    async def delete(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        context: ErrorContext | None = None
    ) -> httpx.Response:
        """DELETE-Request."""
        return await self.make_request(
            method="DELETE",
            endpoint=endpoint,
            params=params,
            headers=headers,
            timeout=timeout,
            context=context
        )

    def get_request_metrics(self) -> dict[str, RequestMetrics]:
        """Gibt aktuelle Request-Metriken zurück."""
        return self._active_requests.copy()

    def get_client_stats(self) -> dict[str, Any]:
        """Gibt Client-Statistiken zurück."""
        return {
            "server_name": self.config.server_name,
            "base_url": self.config.base_url,
            "active_requests": len(self._active_requests),
            "http2_enabled": self.config.http2_enabled,
            "ssl_enabled": self.config.verify_ssl,
            "connection_pool_size": self.config.connection_pool_size,
        }

    async def close(self) -> None:
        """Schließt HTTP-Client und bereinigt Ressourcen."""
        if self._client:
            await self._client.aclose()
            self._client = None

        self._active_requests.clear()

        logger.debug(f"HTTP-Client für {self.config.server_name} geschlossen")
