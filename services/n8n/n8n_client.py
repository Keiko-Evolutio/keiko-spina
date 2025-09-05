"""Asynchroner n8n-API-Client für Workflow-Triggering und Monitoring."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from config.settings import settings
from kei_logging import get_logger
from observability import add_span_attributes, trace_function, trace_span
from services.clients.common.retry_utils import RetryableClient, RetryConfig

from .models import ExecutionResult, ExecutionStatus, TriggerResult

logger = get_logger(__name__)

# Konstanten für n8n-Client-Konfiguration
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_POLL_INTERVAL = 0.5
DEFAULT_POLL_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 0.2


def _read_env(name: str, default: str = "") -> str:
    """Liest Umgebungsvariablen defensiv ein."""
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


class N8nClient(RetryableClient):
    """Leichtgewichtiger n8n-Client mit Retry- und Backoff-Logik.

    Der Client unterstützt zwei Trigger-Strategien:
    - REST-Ausführung über `POST /rest/workflows/{workflow_id}/run`
    - Webhook-Fallback über `POST /webhook/{webhook_path}`

    Der Fallback wird automatisch verwendet, wenn die REST-Route nicht verfügbar ist (z. B. 404).
    """

    # Standard-Header-Name für API-Key
    API_KEY_HEADER = "X-N8N-API-KEY"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        request_timeout: float | None = None,
        poll_interval_seconds: float | None = None,
        poll_timeout_seconds: float | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        # Initialisiere RetryableClient-Basis
        if retry_config is None:
            retry_config = RetryConfig(
                max_retries=DEFAULT_MAX_RETRIES,
                initial_delay=DEFAULT_RETRY_BACKOFF_BASE,
                backoff_multiplier=2.0,
                max_delay=10.0,
                exceptions=(httpx.HTTPError, httpx.HTTPStatusError)
            )
        super().__init__(retry_config)

        # Basis-Konfiguration aus ENV mit Fallbacks
        # Settings-Fallbacks nutzen, falls gesetzt
        settings_base = getattr(settings, "n8n_base_url", "")
        settings_key = (
            settings.n8n_api_key.get_secret_value() if getattr(settings, "n8n_api_key", None) else ""
        )

        self.base_url: str = (base_url or settings_base or _read_env("N8N_BASE_URL")).rstrip("/")
        self.api_key: str = api_key or settings_key or _read_env("N8N_API_KEY")

        # Zeitparameter mit Konstanten
        self.request_timeout: float = request_timeout or float(_read_env("N8N_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))
        self.poll_interval_seconds: float = poll_interval_seconds or float(
            _read_env("N8N_POLL_INTERVAL_SECONDS", str(DEFAULT_POLL_INTERVAL))
        )
        self.poll_timeout_seconds: float = poll_timeout_seconds or float(
            _read_env("N8N_POLL_TIMEOUT_SECONDS", str(DEFAULT_POLL_TIMEOUT))
        )

        # HTTP Client wird pro Anfrage erstellt (kurzlebig) für Einfachheit und Isolation
        self._client: httpx.AsyncClient | None = None

        # Validierung minimaler Anforderungen
        if not self.base_url:
            logger.warning("n8n base_url ist leer; Client wird als 'unavailable' behandelt")
        if not self.api_key:
            logger.warning("n8n api_key ist leer; REST-Trigger werden fehlschlagen")

    # ------------------------------------------------------------------
    # Low-Level HTTP
    # ------------------------------------------------------------------
    def _build_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Erstellt Standard-Header inklusive API-Key."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.API_KEY_HEADER] = self.api_key
        if extra:
            headers.update(extra)
        return headers

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Stellt einen Async HTTP-Client sicher."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout)
        return self._client

    async def _close_client(self) -> None:
        """Schließt den HTTP-Client sofern vorhanden."""
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    async def _make_request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Führt einen einzelnen HTTP-Request durch (ohne Retry-Logik)."""
        client = await self._ensure_client()
        response = await client.request(method, url, headers=headers, json=json)

        data: dict[str, Any] = {}
        try:
            data = response.json()
        except Exception:
            # n8n liefert in Fehlerfällen u. U. Text – tolerant bleiben
            data = {"raw": response.text}

        # Retry bei 5xx durch Exception
        if response.status_code >= 500:
            raise httpx.HTTPStatusError("Server Error", request=response.request, response=response)

        return response.status_code, data

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Führt HTTP-Request mit Retry-Logik durch (nutzt RetryableClient-Basis)."""
        return await self._execute_with_retry(
            self._make_request,
            method,
            url,
            headers=headers,
            json=json
        )

    # ------------------------------------------------------------------
    # High-Level API
    # ------------------------------------------------------------------
    @trace_function("n8n.trigger_workflow")
    async def trigger_workflow(
        self,
        workflow_id: str | int,
        payload: dict[str, Any],
        *,
        mode: str = "rest",
        webhook_path: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> TriggerResult:
        """Triggert einen Workflow und gibt die Execution-ID zurück (falls verfügbar).

        Args:
            workflow_id: n8n Workflow-ID (oder Name), wird für REST-Route genutzt
            payload: Eingabedaten für den Workflow
            mode: "rest" oder "webhook"; bei "rest" wird bei 404 auf "webhook" gewechselt
            webhook_path: Optionaler Webhook-Pfad für Fallback
            max_retries: Anzahl Wiederholungen bei transienten Fehlern
        """
        if not self.base_url:
            return TriggerResult(execution_id=None, started=False, raw={"error": "missing_base_url"})

        # Versuche REST-Trigger
        if mode == "rest":
            rest_url = f"{self.base_url}/rest/workflows/{workflow_id}/run"
            with trace_span("n8n.rest_trigger", {"url": rest_url, "workflow_id": str(workflow_id)}):
                status, data = await self._request_with_retry(
                    "POST", rest_url, headers=self._build_headers(), json=payload
                )
                add_span_attributes({"http.status_code": status})
                if status == 404 and webhook_path:
                    # Fallback auf Webhook-Trigger
                    return await self.trigger_workflow(
                        workflow_id, payload, mode="webhook", webhook_path=webhook_path, max_retries=max_retries
                    )

                # n8n liefert in der Regel eine executionId bei REST-Trigger
                execution_id_raw = (
                    data.get("data", {}).get("id")
                    or data.get("id")
                    or data.get("executionId")
                )
                execution_id = str(execution_id_raw) if execution_id_raw is not None else None
                return TriggerResult(execution_id=execution_id, started=status < 400, raw=data)

        # Webhook-Trigger
        hook_path = webhook_path or str(workflow_id)
        webhook_url = f"{self.base_url}/webhook/{hook_path.lstrip('/')}"
        with trace_span("n8n.webhook_trigger", {"url": webhook_url, "workflow_id": str(workflow_id)}):
            status, data = await self._request_with_retry(
                "POST", webhook_url, headers=self._build_headers(), json=payload
            )
            add_span_attributes({"http.status_code": status})
            # Webhooks liefern meist keine executionId – started flag nutzen
            return TriggerResult(execution_id=None, started=status < 400, raw=data)

    @trace_function("n8n.get_execution_status")
    async def get_execution_status(self, execution_id: str) -> ExecutionResult:
        """Liest den Status einer Execution."""
        if not self.base_url:
            return ExecutionResult(status=ExecutionStatus.unknown, finished=False, raw={"error": "missing_base_url"})

        url = f"{self.base_url}/rest/executions/{execution_id}"
        with trace_span("n8n.get_status", {"url": url, "execution_id": execution_id}):
            status_code, data = await self._request_with_retry("GET", url, headers=self._build_headers())
            add_span_attributes({"http.status_code": status_code})

        # Status ableiten; n8n liefert z. B. { finished: true, status: 'success' }
        raw_status = str(data.get("status") or data.get("data", {}).get("status") or "unknown").lower()
        finished = bool(data.get("finished") or data.get("data", {}).get("finished") or (raw_status in {"success", "error"}))
        try:
            status_enum = ExecutionStatus(raw_status)
        except Exception:
            status_enum = ExecutionStatus.unknown

        return ExecutionResult(status=status_enum, finished=finished, raw=data)

    @trace_function("n8n.poll_execution_status")
    async def poll_execution_status(
        self,
        execution_id: str,
        *,
        interval_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        """Pollt den Status bis Abschluss oder Timeout."""
        interval = interval_seconds or self.poll_interval_seconds
        timeout = timeout_seconds or self.poll_timeout_seconds

        start = asyncio.get_event_loop().time()
        while True:
            result = await self.get_execution_status(execution_id)
            if result.finished:
                return result
            if (asyncio.get_event_loop().time() - start) > timeout:
                return ExecutionResult(status=ExecutionStatus.unknown, finished=False, raw={"error": "timeout"})
            await asyncio.sleep(interval)

    async def aclose(self) -> None:
        """Schließt Ressourcen des Clients."""
        await self._close_client()


__all__ = [
    "ExecutionResult",
    "ExecutionStatus",
    "N8nClient",
    "TriggerResult",
]
