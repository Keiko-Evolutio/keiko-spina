"""KEI-RPC REST Client.

Stellt einen einfachen, produktionsnahen Client für die KEI-RPC REST-API
bereit, inklusive Trace-Propagation, optionalem Retry mit Exponential Backoff
und bequemen Methoden für Standard-Operationen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import backoff
import httpx

from kei_logging import get_logger
from observability.traced_http_client import TracedHTTPXClient
from services.core.circuit_breaker import CircuitBreaker, CircuitPolicy
from services.core.constants import KEI_RPC_CIRCUIT_BREAKER_CONFIG

from .common import (
    DEFAULT_REQUEST_TIMEOUT,
    RetryableClient,
    StandardHTTPClientConfig,
    create_http_retry_config,
    create_kei_rpc_headers,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class KEIRPCClientConfig:
    """Konfiguration für KEI-RPC REST Client."""

    base_url: str
    api_token: str
    tenant_id: str
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT


class KEIRPCClient(RetryableClient):
    """REST-Client für KEI-RPC Ressourcen-Operationen."""

    def __init__(self, config: KEIRPCClientConfig) -> None:
        """Initialisiert Client mit Basis-Konfiguration.

        Args:
            config: Client-Konfiguration
        """
        # Retry-Konfiguration initialisieren
        super().__init__(create_http_retry_config())

        self._config = config

        # HTTP Client Konfiguration
        StandardHTTPClientConfig.kei_rpc()
        headers = create_kei_rpc_headers(config.api_token, config.tenant_id)

        self._client = TracedHTTPXClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            headers=headers,
        )

        # Circuit Breaker für RPC Upstream
        self._cb = CircuitBreaker(
            name="api-grpc-upstream",
            policy=CircuitPolicy(**KEI_RPC_CIRCUIT_BREAKER_CONFIG),
        )

    @backoff.on_exception(backoff.expo, (httpx.RequestError,), max_tries=3, jitter=backoff.full_jitter)
    async def list_resources(self, *, page: int = 1, per_page: int = 20, q: str | None = None, sort: str = "-updated_at") -> dict[str, Any]:
        """Listet Ressourcen mit Pagination/Filter/Sort.

        Returns:
            Antwort als Dictionary mit Feldern `items` und `pagination`
        """
        params: dict[str, Any] = {"page": page, "per_page": per_page, "sort": sort}
        if q:
            params["q"] = q
        async def _call():
            resp = await self._client.get("/api/v1/rpc/resources", params=params)
            resp.raise_for_status()
            return resp.json()

        return await self._cb.call(_call)

    @backoff.on_exception(backoff.expo, (httpx.RequestError,), max_tries=3, jitter=backoff.full_jitter)
    async def create_resource(self, name: str, *, idempotency_key: str | None = None) -> dict[str, Any]:
        """Erstellt eine Ressource mit optionaler Idempotenz.

        Args:
            name: Ressourcenname
            idempotency_key: Optionaler Idempotenzschlüssel
        """
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        async def _call():
            resp = await self._client.post("/api/v1/rpc/resources", json={"name": name}, headers=headers)
            resp.raise_for_status()
            return resp.json()

        return await self._cb.call(_call)

    @backoff.on_exception(backoff.expo, (httpx.RequestError,), max_tries=3, jitter=backoff.full_jitter)
    async def get_resource(self, resource_id: str, *, if_none_match: str | None = None) -> httpx.Response:
        """Liest Ressource. Gibt Response zurück (für 304-Handling)."""
        headers = {"If-None-Match": if_none_match} if if_none_match else None
        async def _call():
            return await self._client.get(f"/api/v1/rpc/resources/{resource_id}", headers=headers)

        return await self._cb.call(_call)

    @backoff.on_exception(backoff.expo, (httpx.RequestError,), max_tries=3, jitter=backoff.full_jitter)
    async def patch_resource(self, resource_id: str, *, name: str | None = None, if_match: str | None = None) -> dict[str, Any]:
        """Teil-Update einer Ressource mit If-Match."""
        headers = {"If-Match": if_match} if if_match else None
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        async def _call():
            resp = await self._client.patch(f"/api/v1/rpc/resources/{resource_id}", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

        return await self._cb.call(_call)

    @backoff.on_exception(backoff.expo, (httpx.RequestError,), max_tries=3, jitter=backoff.full_jitter)
    async def batch_create(self, items: list[str]) -> dict[str, Any]:
        """Batch-Erstellt Ressourcen aus einer Liste von Namen."""
        payload = [{"name": name} for name in items]
        async def _call():
            resp = await self._client.post("/api/v1/rpc/resources:batch", json=payload)
            resp.raise_for_status()
            return resp.json()

        return await self._cb.call(_call)


__all__ = ["KEIRPCClient", "KEIRPCClientConfig"]
