"""Webhook-Delivery-Executor für HTTP-Zustellungen.

Führt HTTP-Webhook-Deliveries durch mit SSL/mTLS-Unterstützung,
Circuit-Breaker-Integration und umfassendem Error-Handling.
"""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from monitoring import record_custom_metric
from services.clients.clients import HTTPClient

from ..circuit_breaker import CircuitConfig, WebhookCircuitBreaker
from ..constants import HTTP_REQUEST_TIMEOUT_SECONDS
from ..prometheus_metrics import (
    WEBHOOK_DELIVERIES_TOTAL,
    WEBHOOK_DELIVERY_DURATION,
    WEBHOOK_ERRORS_TOTAL,
)
from .signature_generator import WebhookSignatureGenerator
from .transform_engine import WebhookTransformEngine

if TYPE_CHECKING:
    import httpx

    from ..models import DeliveryRecord, WebhookEvent, WebhookTarget

logger = get_logger(__name__)


class DeliveryStatus(str, Enum):
    """Status einer Webhook-Delivery."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class DeliveryResult:
    """Ergebnis einer Webhook-Delivery."""
    status: DeliveryStatus
    http_status: int | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None
    response_headers: dict[str, str] | None = None


class WebhookDeliveryExecutor:
    """Führt HTTP-Webhook-Deliveries durch."""

    def __init__(
        self,
        *,
        http_client: HTTPClient | None = None,
        signature_generator: WebhookSignatureGenerator | None = None,
        transform_engine: WebhookTransformEngine | None = None,
        circuit_breaker: WebhookCircuitBreaker | None = None,
    ) -> None:
        """Initialisiert den Delivery-Executor.

        Args:
            http_client: HTTP-Client für Requests
            signature_generator: Signatur-Generator
            transform_engine: Event-Transform-Engine
            circuit_breaker: Circuit-Breaker für Resilience
        """
        self.http_client = http_client or HTTPClient(timeout=HTTP_REQUEST_TIMEOUT_SECONDS)
        self.signature_generator = signature_generator or WebhookSignatureGenerator()
        self.transform_engine = transform_engine or WebhookTransformEngine()
        self.circuit_breaker = circuit_breaker or WebhookCircuitBreaker(CircuitConfig())

    async def execute_delivery(
        self,
        target: WebhookTarget,
        event: WebhookEvent,
        record: DeliveryRecord,  # pylint: disable=unused-argument
    ) -> DeliveryResult:
        """Führt eine Webhook-Delivery durch.

        Args:
            target: Webhook-Target
            event: Webhook-Event
            record: Delivery-Record

        Returns:
            Delivery-Ergebnis
        """
        start_time = datetime.now(UTC)
        tenant_id = event.meta.tenant if event.meta else None
        circuit_policy = None  # Initialize to avoid unbound variable in exception handler

        try:
            # Circuit-Breaker prüfen
            circuit_policy = self._create_circuit_policy(target)
            if not self.circuit_breaker.allow_request(
                target_id=target.id,
                tenant_id=tenant_id,
                policy=circuit_policy
            ):
                return self._create_circuit_open_result()

            # Event transformieren
            transformed_event = self.transform_engine.transform_event(
                event, target.transform
            )
            payload = json.dumps(transformed_event).encode("utf-8")

            # Headers mit Signatur erstellen
            timestamp = int(event.occurred_at.timestamp())
            headers = await self.signature_generator.create_headers(
                target=target,
                payload=payload,
                timestamp=timestamp,
                event_type=event.event_type,
            )

            # SSL-Kontext für mTLS erstellen
            ssl_context = self._create_ssl_context(target)

            # HTTP-Request durchführen
            result = await self._execute_http_request(
                url=target.url,
                payload=payload,
                headers=headers,
                ssl_context=ssl_context,
            )

            # Circuit-Breaker-Status aktualisieren
            if result.status == DeliveryStatus.SUCCESS:
                self.circuit_breaker.on_success(
                    target_id=target.id,
                    tenant_id=tenant_id,
                    policy=circuit_policy
                )
            else:
                self.circuit_breaker.on_failure(
                    target_id=target.id,
                    tenant_id=tenant_id,
                    policy=circuit_policy
                )

            # Metriken aufzeichnen
            duration = (datetime.now(UTC) - start_time).total_seconds()
            result.duration_seconds = duration
            self._record_metrics(target, event, result, tenant_id)

            return result

        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Unerwartete Fehler behandeln
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.exception(
                "Unexpected error during delivery to %s: %s",
                target.url, exc
            )

            self.circuit_breaker.on_failure(
                target_id=target.id,
                tenant_id=tenant_id,
                policy=circuit_policy
            )

            return DeliveryResult(
                status=DeliveryStatus.FAILED,
                duration_seconds=duration,
                error_message=str(exc),
            )

    async def _execute_http_request(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
        ssl_context: ssl.SSLContext | None,
    ) -> DeliveryResult:
        """Führt HTTP-Request durch.

        Args:
            url: Ziel-URL
            payload: Request-Payload
            headers: Request-Headers
            ssl_context: SSL-Kontext für mTLS

        Returns:
            Delivery-Ergebnis
        """
        try:
            async with self.http_client.session() as session:
                # Unterscheide zwischen httpx und aiohttp
                if hasattr(session, "build_request"):
                    # httpx AsyncClient
                    result = await self._execute_httpx_request(
                        session, url, payload, headers, ssl_context
                    )
                else:
                    # aiohttp ClientSession
                    result = await self._execute_aiohttp_request(
                        session, url, payload, headers, ssl_context
                    )

                return result

        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._handle_request_exception(exc, url)

    async def _execute_httpx_request(
        self, session, url: str, payload: bytes,
        headers: dict[str, str], ssl_context: ssl.SSLContext | None
    ) -> DeliveryResult:
        """Führt httpx-Request durch."""
        client: httpx.AsyncClient = session
        kwargs: dict[str, Any] = {
            "content": payload,
            "headers": headers,
        }

        if ssl_context is not None:
            kwargs["verify"] = ssl_context

        response = await client.post(url, **kwargs)

        return DeliveryResult(
            status=(
                DeliveryStatus.SUCCESS
                if 200 <= response.status_code < 300
                else DeliveryStatus.FAILED
            ),
            http_status=response.status_code,
            response_headers=dict(response.headers),
        )

    async def _execute_aiohttp_request(
        self, session, url: str, payload: bytes,
        headers: dict[str, str], ssl_context: ssl.SSLContext | None
    ) -> DeliveryResult:
        """Führt aiohttp-Request durch."""
        async with session.post(
            url,
            data=payload,
            headers=headers,
            ssl=ssl_context
        ) as response:
            return DeliveryResult(
                status=(
                    DeliveryStatus.SUCCESS
                    if 200 <= response.status < 300
                    else DeliveryStatus.FAILED
                ),
                http_status=response.status,
                response_headers=dict(response.headers),
            )

    def _handle_request_exception(self, exc: Exception, url: str) -> DeliveryResult:
        """Behandelt Request-Exceptions."""
        import asyncio  # pylint: disable=import-outside-toplevel

        if isinstance(exc, asyncio.TimeoutError):
            return DeliveryResult(
                status=DeliveryStatus.TIMEOUT,
                error_message=f"Request timeout for {url}",
            )
        if isinstance(exc, OSError | ssl.SSLError):
            return DeliveryResult(
                status=DeliveryStatus.TRANSPORT_ERROR,
                error_message=f"Transport error for {url}: {exc}",
            )
        return DeliveryResult(
            status=DeliveryStatus.FAILED,
            error_message=f"Request failed for {url}: {exc}",
        )

    def _create_ssl_context(self, target: WebhookTarget) -> ssl.SSLContext | None:
        """Erstellt SSL-Kontext für mTLS."""
        if not (target.mtls_cert_pem and target.mtls_key_pem):
            return None

        try:
            ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_ctx.load_cert_chain(
                certfile=target.mtls_cert_pem,
                keyfile=target.mtls_key_pem
            )
            return ssl_ctx
        except (OSError, ssl.SSLError) as exc:
            logger.warning(
                "Failed to create SSL context for target %s: %s",
                target.id, exc
            )
            return None

    def _create_circuit_policy(self, target: WebhookTarget) -> CircuitConfig | None:
        """Erstellt Circuit-Breaker-Policy für Target."""
        try:
            return CircuitConfig(
                use_consecutive_failures=bool(
                    getattr(target, "cb_use_consecutive_failures", False)
                ),
                failure_threshold=int(
                    getattr(target, "cb_failure_threshold", 5) or 5
                ),
                recovery_timeout_seconds=float(
                    getattr(target, "cb_recovery_timeout_seconds", 60.0) or 60.0
                ),
                success_threshold=int(
                    getattr(target, "cb_success_threshold", 3) or 3
                ),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to create circuit policy for target %s: %s", target.id, exc)
            return None

    def _create_circuit_open_result(self) -> DeliveryResult:
        """Erstellt Ergebnis für offenen Circuit-Breaker."""
        return DeliveryResult(
            status=DeliveryStatus.CIRCUIT_OPEN,
            error_message="Circuit breaker is open",
        )

    def _record_metrics(
        self,
        target: WebhookTarget,
        event: WebhookEvent,
        result: DeliveryResult,
        tenant_id: str | None,
    ) -> None:
        """Zeichnet Metriken für Delivery auf."""
        try:
            # Prometheus Metriken
            status_label = result.status.value
            WEBHOOK_DELIVERIES_TOTAL.labels(
                target_id=target.id,
                event_type=event.event_type,
                tenant_id=tenant_id or "",
                status=status_label
            ).inc()

            WEBHOOK_DELIVERY_DURATION.labels(
                target_id=target.id,
                event_type=event.event_type,
                tenant_id=tenant_id or "",
                status=status_label
            ).observe(result.duration_seconds)

            # Error-Metriken
            if result.status != DeliveryStatus.SUCCESS:
                WEBHOOK_ERRORS_TOTAL.labels(
                    target_id=target.id,
                    event_type=event.event_type,
                    tenant_id=tenant_id or "",
                    error_type=result.status.value
                ).inc()

            # Custom Metriken
            if result.status == DeliveryStatus.SUCCESS:
                record_custom_metric(
                    "webhook.delivered",
                    1,
                    {"target": target.id, "event_type": event.event_type}
                )

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to record metrics: %s", exc)


__all__ = ["DeliveryResult", "DeliveryStatus", "WebhookDeliveryExecutor"]
