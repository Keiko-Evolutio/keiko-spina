"""Asynchroner Delivery‑Worker für Outbound Webhooks.

Verarbeitet DeliveryRecords mit Backoff/Retry, Transformation und optionaler mTLS.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import ssl
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from monitoring import record_custom_metric
from services.clients.clients import HTTPClient
from storage.cache.redis_cache import NoOpCache, get_cache_client

from .audit_logger import webhook_audit
from .circuit_breaker import CircuitConfig, WebhookCircuitBreaker
from .constants import (
    HTTP_CLIENT_TIMEOUT_SECONDS,
    MAX_BACKOFF_SECONDS,
    REDIS_SCAN_COUNT_DEFAULT,
    REDIS_SCAN_COUNT_WORK_STEALING,
)
from .exceptions import WebhookDeliveryException, WebhookTimeoutException
from .keys import dlq_key, outbox_key
from .models import DeliveryRecord, DeliveryStatus, TargetTransform, WebhookEvent, WebhookTarget
from .prometheus_metrics import (
    WEBHOOK_DELIVERIES_TOTAL,
    WEBHOOK_DELIVERY_DURATION,
    WEBHOOK_DLQ_TOTAL,
    WEBHOOK_ERRORS_TOTAL,
)
from .secret_manager import get_secret_manager
from .tracking import DeliveryTracker
from .workers.base_worker import BaseWorker, WorkerConfig

if TYPE_CHECKING:
    import httpx

logger = get_logger(__name__)


class DeliveryWorker(BaseWorker):
    """Hintergrund‑Worker für Outbound‑Zustellungen."""

    def __init__(self, queue_name: str = "default", *, poll_interval: float = 0.5) -> None:
        config = WorkerConfig(
            name=f"delivery-worker-{queue_name}",
            poll_interval_seconds=poll_interval,
        )
        super().__init__(config)

        self.queue_name = queue_name
        self._http = HTTPClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS)
        self._tracker = DeliveryTracker()
        # Circuit Breaker pro Worker‑Instanz (pro Target wird intern getrennt)
        self._breaker = WebhookCircuitBreaker(CircuitConfig())
        # Metriken/Zustand
        self.processed_count: int = 0
        self.failed_count: int = 0
        self.last_activity_at: datetime | None = None

    async def _run_cycle(self) -> None:
        """Führt einen Worker-Zyklus aus (BaseWorker-Interface)."""
        processed = await self._process_one()
        if processed:
            self.processed_count += 1
            self.last_activity_at = datetime.now(UTC)
        # Keine Arbeit gefunden wird von BaseWorker mit poll_interval gehandhabt

    async def _process_one(self) -> bool:
        """Verarbeitet einen Delivery‑Eintrag aus der Queue (falls vorhanden)."""
        data = await self._fetch_delivery_data()
        if not data:
            return False

        delivery_item = self._parse_delivery_data(data)
        if not delivery_item:
            return True  # Ungültiger Eintrag übersprungen

        record, target, event = delivery_item
        await self._deliver(record, target, event)
        return True

    async def _fetch_delivery_data(self) -> str | None:
        """Holt Delivery-Daten aus Redis-Queue."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return None

        try:
            # Erst public Queue prüfen
            data = await client.rpop(outbox_key(None, self.queue_name))
            if data:
                return data

            # Alle Tenant Outbox Keys scannen
            data = await self._scan_tenant_queues(client)
            if data:
                return data

            # Work-Stealing: andere Shards versuchen
            return await self._try_work_stealing(client)
        except Exception:
            return None

    async def _scan_tenant_queues(self, client) -> str | None:
        """Scannt Tenant-spezifische Queues."""
        pattern = f"kei:webhook:outbox:*:{self.queue_name}"
        async for key in client.scan_iter(match=pattern, count=REDIS_SCAN_COUNT_DEFAULT):  # type: ignore[attr-defined]
            data = await client.rpop(key)
            if data:
                return data
        return None

    async def _try_work_stealing(self, client) -> str | None:
        """Versucht Work-Stealing von anderen Shards."""
        if ":" not in self.queue_name:
            return None

        base, _ = self.queue_name.split(":", 1)
        pattern = f"kei:webhook:outbox:*:{base}:*"
        async for key in client.scan_iter(match=pattern, count=REDIS_SCAN_COUNT_WORK_STEALING):  # type: ignore[attr-defined]
            # Eigenen Shard überspringen
            if key.endswith((f":{self.queue_name.split(':', 1)[1]}", f":{self.queue_name}")):
                continue
            data = await client.rpop(key)
            if data:
                return data
        return None

    def _parse_delivery_data(self, data: str) -> tuple[DeliveryRecord, WebhookTarget, WebhookEvent] | None:
        """Parsed Delivery-Daten aus JSON."""
        try:
            payload = json.loads(data)
            record = DeliveryRecord(**payload["record"])  # type: ignore[index]
            target = WebhookTarget(**payload["target"])  # type: ignore[index]
            event = WebhookEvent(**payload["event"])  # type: ignore[index]
            return record, target, event
        except (ValueError, TypeError, KeyError):
            return None

    async def _deliver(self, record: DeliveryRecord, target: WebhookTarget, event: WebhookEvent) -> None:
        """Führt die eigentliche HTTP‑Zustellung durch mit Retries."""
        url = target.url
        effective_event = self._apply_transform(event, target.transform)
        body = json.dumps(effective_event).encode("utf-8")
        policy = None  # Initialize to avoid unbound variable in exception handlers
        # Secret-Auflösung: Key Vault bevorzugt, Fallback auf Legacy
        secret_value: str | None = None
        try:
            if target.secret_key_name:
                sm = get_secret_manager()
                if target.secret_version:
                    secret_value = await sm.get_secret_by_version(key_name=target.secret_key_name, version=target.secret_version)
                else:
                    secret_value, _ = await sm.get_current_secret(key_name=target.secret_key_name)
            elif target.legacy_secret:
                secret_value = target.legacy_secret
        except Exception:
            # Konservativer Fallback auf Legacy, wenn vorhanden
            if target.legacy_secret:
                secret_value = target.legacy_secret
        if not secret_value:
            raise WebhookDeliveryException(
                message="Kein Secret zur Signierung verfügbar",
                error_code="signing_secret_missing",
                context={"target_id": target.id},
            )
        signature = hmac.new(secret_value.encode("utf-8"), body, sha256).hexdigest()

        headers: dict[str, str] = {
            "content-type": "application/json",
            "x-kei-signature": signature,
            "x-kei-timestamp": str(int(event.occurred_at.timestamp())),
            "x-kei-event-type": event.event_type,
            **(target.headers or {}),
        }

        ssl_ctx = None
        if target.mtls_cert_pem and target.mtls_key_pem:
            try:
                ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                ssl_ctx.load_cert_chain(certfile=target.mtls_cert_pem, keyfile=target.mtls_key_pem)  # type: ignore[arg-type]
            except (OSError, ssl.SSLError):
                ssl_ctx = None

        try:
            # Circuit Breaker vor eigentlichem HTTP‑Call prüfen (Target‑Policy berücksichtigen)
            tenant = event.meta.tenant if event.meta else None
            try:
                from .circuit_breaker import CircuitConfig
                policy = CircuitConfig(
                    use_consecutive_failures=bool(getattr(target, "cb_use_consecutive_failures", False)),
                    failure_threshold=int(getattr(target, "cb_failure_threshold", 5) or 5),
                    recovery_timeout_seconds=float(getattr(target, "cb_recovery_timeout_seconds", 60.0) or 60.0),
                    success_threshold=int(getattr(target, "cb_success_threshold", 3) or 3),
                )
            except (ValueError, KeyError, AttributeError) as e:
                logger.debug(f"Fehler beim Laden der Circuit-Breaker-Policy: {e}")
                policy = None
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler beim Laden der Circuit-Breaker-Policy: {e}")
                policy = None
            if not self._breaker.allow_request(target_id=target.id, tenant_id=tenant, policy=policy):
                self.failed_count += 1
                record.status = DeliveryStatus.retrying
                record.last_error = "circuit_open"
                WEBHOOK_ERRORS_TOTAL.labels(target_id=target.id, event_type=event.event_type, tenant_id=tenant or "", error_type="circuit_open").inc()
                # Audit: Circuit open blockierte Call
                with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                    await webhook_audit.outbound_failed(
                        correlation_id=record.correlation_id,
                        delivery_id=record.delivery_id,
                        target_id=target.id,
                        event_type=event.event_type,
                        tenant_id=tenant,
                        error_details={"reason": "circuit_open"},
                        will_retry=True,
                    )
                await self._reschedule(record, target, event)
                return

            start_ts = datetime.now(UTC).timestamp()
            async with self._http.session() as session:
                # httpx vs aiohttp Unterschiede behandeln
                if hasattr(session, "post") and hasattr(session, "get") and hasattr(session, "build_request"):
                    # httpx AsyncClient
                    client: httpx.AsyncClient = session  # type: ignore[assignment]
                    req_kwargs: dict[str, Any] = {"content": body, "headers": headers}
                    if ssl_ctx is not None:
                        req_kwargs["verify"] = ssl_ctx
                    resp = await client.post(url, **req_kwargs)
                    status = resp.status_code
                else:
                    # aiohttp ClientSession
                    async with session.post(url, data=body, headers=headers, ssl=ssl_ctx) as resp:  # type: ignore[arg-type]
                        status = resp.status
                    ok = 200 <= status < 300
                    record.status = DeliveryStatus.success if ok else DeliveryStatus.retrying
                    record.attempt += 1
                    record.updated_at = event.occurred_at
                    if not ok:
                        record.last_error = f"HTTP {status}"
                        self._breaker.on_failure(target_id=target.id, tenant_id=tenant, policy=policy)
                        WEBHOOK_ERRORS_TOTAL.labels(target_id=target.id, event_type=event.event_type, tenant_id=tenant or "", error_type="http").inc()
                        await self._reschedule(record, target, event)
                        # Audit: fehlgeschlagen, mit Retry geplant
                        with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                            await webhook_audit.outbound_failed(
                                correlation_id=record.correlation_id,
                                delivery_id=record.delivery_id,
                                target_id=target.id,
                                event_type=event.event_type,
                                tenant_id=event.meta.tenant if event.meta else None,
                                error_details={"http_status": status},
                                will_retry=True,
                            )
                    else:
                        # Prometheus Metriken
                        WEBHOOK_DELIVERIES_TOTAL.labels(target_id=target.id, event_type=event.event_type, tenant_id=tenant or "", status="success").inc()
                        dur = max(0.0, datetime.now(UTC).timestamp() - start_ts)
                        WEBHOOK_DELIVERY_DURATION.labels(target_id=target.id, event_type=event.event_type, tenant_id=tenant or "", status="success").observe(dur)
                        self._breaker.on_success(target_id=target.id, tenant_id=tenant, policy=policy)
                        record_custom_metric("webhook.delivered", 1, {"target": target.id})
                        self.processed_count += 1
                        self.last_activity_at = datetime.now(UTC)
                        # Delivery als erfolgreich markieren
                        try:
                            obj = {
                                "record": {
                                    **record.model_dump(mode="json"),
                                    "status": DeliveryStatus.success,
                                    "delivered_at": datetime.now(UTC).isoformat(),
                                },
                                "target": target.model_dump(mode="json"),
                                "event": event.model_dump(mode="json"),
                            }
                            await self._tracker.update(obj)
                        except Exception:
                            pass
                        # Audit: erfolgreich zugestellt
                        with contextlib.suppress(Exception):
                            await webhook_audit.outbound_delivered(
                                correlation_id=record.correlation_id,
                                delivery_id=record.delivery_id,
                                target_id=target.id,
                                event_type=event.event_type,
                                tenant_id=event.meta.tenant if event.meta else None,
                            )
        except TimeoutError as exc:
            record.status = DeliveryStatus.retrying
            record.attempt += 1
            record.last_error = "timeout"
            self.failed_count += 1
            tenant = event.meta.tenant if event.meta else None
            self._breaker.on_failure(target_id=target.id, tenant_id=tenant, policy=policy)
            WEBHOOK_ERRORS_TOTAL.labels(target_id=target.id, event_type=event.event_type, tenant_id=tenant or "", error_type="timeout").inc()
            raise WebhookTimeoutException(
                message="Zustellung zeitüberschritten",
                error_code="delivery_timeout",
                context={"url": url, "target_id": target.id},
            ) from exc
        except (OSError, ssl.SSLError) as exc:
            record.status = DeliveryStatus.retrying
            record.attempt += 1
            record.last_error = "transport_error"
            self.failed_count += 1
            tenant = event.meta.tenant if event.meta else None
            self._breaker.on_failure(target_id=target.id, tenant_id=tenant, policy=policy)
            WEBHOOK_ERRORS_TOTAL.labels(target_id=target.id, event_type=event.event_type, tenant_id=tenant or "", error_type="transport").inc()
            raise WebhookDeliveryException(
                message="Transportfehler bei Zustellung",
                error_code="delivery_transport_error",
                context={"url": url, "target_id": target.id},
            ) from exc

    async def _reschedule(self, record: DeliveryRecord, target: WebhookTarget, event: WebhookEvent) -> None:
        """Plant erneuten Zustellversuch oder verschiebt in DLQ."""
        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            return
        if record.attempt >= (target.max_attempts or record.max_attempts):
            record.status = DeliveryStatus.dlq
            await client.lpush(
                dlq_key(event.meta.tenant if event.meta else None),
                json.dumps({"record": record.model_dump(), "target": target.model_dump(), "event": event.model_dump()}),
            )
            record_custom_metric("webhook.dlq", 1, {"target": target.id})
            with contextlib.suppress(AttributeError, ValueError):
                WEBHOOK_DLQ_TOTAL.labels(
                    target_id=target.id,
                    event_type=event.event_type,
                    tenant_id=event.meta.tenant if event.meta else "",
                ).inc()
            # Audit: Move to DLQ
            try:
                from .audit_logger import WebhookAuditEventType
                await webhook_audit.dlq_event(
                    event=WebhookAuditEventType.DLQ_MOVE,
                    delivery_id=record.delivery_id,
                    target_id=target.id,
                    tenant_id=event.meta.tenant if event.meta else None,
                    correlation_id=record.correlation_id,
                    details={"attempt": record.attempt},
                )
            except (ConnectionError, TimeoutError) as e:
                logger.debug(f"DLQ-Audit-Event fehlgeschlagen - Verbindungsproblem: {e}")
            except Exception as e:
                logger.warning(f"DLQ-Audit-Event fehlgeschlagen - Unerwarteter Fehler: {e}")
            return

        delay = (target.backoff_seconds or record.backoff_seconds) * (2 ** max(0, record.attempt - 1))
        await asyncio.sleep(min(delay, MAX_BACKOFF_SECONDS))
        await client.lpush(
            outbox_key(event.meta.tenant if event.meta else None, self.queue_name),
            json.dumps({"record": record.model_dump(), "target": target.model_dump(), "event": event.model_dump()}),
        )
        # Audit: Retried
        with contextlib.suppress(Exception):
            await webhook_audit.outbound_retried(
                correlation_id=record.correlation_id,
                delivery_id=record.delivery_id,
                target_id=target.id,
                event_type=event.event_type,
                tenant_id=event.meta.tenant if event.meta else None,
                details={"next_delay_s": min(delay, MAX_BACKOFF_SECONDS)},
            )

    def _apply_transform(self, event: WebhookEvent, transform: TargetTransform | None) -> dict[str, Any]:
        """Wendet optionale Transformationsregeln auf das Event an."""
        obj = event.model_dump(mode="json")
        if not transform:
            return obj
        data: dict[str, Any] = obj.get("data", {})
        if transform.include_fields:
            data = {k: v for k, v in data.items() if k in set(transform.include_fields)}
        if transform.exclude_fields:
            for k in list(data.keys()):
                if k in set(transform.exclude_fields):
                    data.pop(k, None)
        if transform.rename_map:
            renamed: dict[str, Any] = {}
            for k, v in data.items():
                renamed[transform.rename_map.get(k, k)] = v
            data = renamed
        if transform.add_fields:
            for k, v in transform.add_fields.items():
                data[k] = v
        if transform.drop_nulls:
            data = {k: v for k, v in data.items() if v is not None}
        obj["data"] = data
        return obj

    def get_status(self) -> dict[str, Any]:
        """Gibt aktuellen Worker‑Status und Metriken zurück."""
        now = datetime.now(UTC)
        started_at = self._started_at or now
        uptime_s = max(0, int((now - started_at).total_seconds()))
        rate_per_minute = (self.processed_count / max(1, uptime_s)) * 60.0
        return {
            "queue_name": self.queue_name,
            "active": self.is_running,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "last_activity": (self.last_activity_at or started_at).isoformat(),
            "uptime_seconds": uptime_s,
            "rate_per_minute": round(rate_per_minute, 3),
            "poll_interval": self.config.poll_interval_seconds,
        }


__all__ = ["DeliveryWorker"]
