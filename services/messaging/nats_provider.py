"""NATS/JetStream Provider – mit Outbox/Inbox, Chaos/Replay, Tracing, QoS, DLQ."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import ssl
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

try:
    from audit_system import action_logger
except ImportError:
    # Fallback wenn audit_system nicht verfügbar
    action_logger = None

from kei_logging import get_logger, structured_msg

from .chaos import apply_chaos, store_replay
from .config import bus_settings
from .dlq import ensure_dlq_stream
from .envelope import BusEnvelope
from .idempotency import is_duplicate, remember
from .metrics import BusMetrics, inject_trace
from .outbox import Inbox, Outbox
from .privacy import decrypt_fields, encrypt_fields, get_active_kms_key_id, redact_payload
from .security import authorize_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

try:  # pragma: no cover
    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext
    NATS_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Fallback-Typen für bessere IDE-Unterstützung
    class NATSClient:  # type: ignore
        def jetstream(self) -> Any: ...
        async def publish(self, subject: str, payload: bytes, **kwargs: Any) -> None: ...
        async def subscribe(self, subject: str, **kwargs: Any) -> Any: ...
        async def close(self) -> None: ...

    class JetStreamContext:  # type: ignore
        async def add_stream(self, **kwargs: Any) -> Any: ...
        async def publish(self, subject: str, payload: bytes, **kwargs: Any) -> None: ...
        async def subscribe(self, subject: str, **kwargs: Any) -> Any: ...
        async def pull_subscribe(self, subject: str, **kwargs: Any) -> Any: ...
        async def account_info(self) -> Any: ...

    NATS_AVAILABLE = False
except Exception as e:  # pragma: no cover
    logger.debug(f"Unerwarteter Fehler beim Import von NATS: {e}")
    # Fallback-Typen für bessere IDE-Unterstützung
    class NATSClient:  # type: ignore
        def jetstream(self) -> Any: ...
        async def publish(self, subject: str, payload: bytes, **kwargs: Any) -> None: ...
        async def subscribe(self, subject: str, **kwargs: Any) -> Any: ...
        async def close(self) -> None: ...

    class JetStreamContext:  # type: ignore
        async def add_stream(self, **kwargs: Any) -> Any: ...
        async def publish(self, subject: str, payload: bytes, **kwargs: Any) -> None: ...
        async def subscribe(self, subject: str, **kwargs: Any) -> Any: ...
        async def pull_subscribe(self, subject: str, **kwargs: Any) -> Any: ...
        async def account_info(self) -> Any: ...

    NATS_AVAILABLE = False

try:  # pragma: no cover
    from opentelemetry import propagate, trace
    from opentelemetry.trace import Link, SpanKind
    _OTEL = True
    _TRACER = trace.get_tracer(__name__)
except ImportError:  # pragma: no cover
    # Fallback-Definitionen für OpenTelemetry imports
    propagate = None
    trace = None
    Link = None
    SpanKind = None
    _OTEL = False
    _TRACER = None
except Exception as e:  # pragma: no cover
    logger.debug(f"Unerwarteter Fehler beim Import von OpenTelemetry: {e}")
    # Fallback-Definitionen für OpenTelemetry imports
    propagate = None
    trace = None
    Link = None
    SpanKind = None
    _OTEL = False
    _TRACER = None


class NATSProvider:
    def __init__(self) -> None:
        self.nc: NATSClient | None = None
        self.js: JetStreamContext | None = None
        self.metrics = BusMetrics()
        self._key_locks: dict[str, asyncio.Lock] = {}

    async def connect(self) -> None:
        if not bus_settings.enabled:
            logger.info("KEI-Bus ist deaktiviert")
            return
        try:
            tls: ssl.SSLContext | None = None
            if bus_settings.security.enable_mtls:
                tls = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
                if bus_settings.security.ca_cert_path:
                    tls.load_verify_locations(bus_settings.security.ca_cert_path)
                if bus_settings.security.client_cert_path and bus_settings.security.client_key_path:
                    tls.load_cert_chain(bus_settings.security.client_cert_path, bus_settings.security.client_key_path)
            kwargs: dict[str, Any] = {"servers": bus_settings.servers, "tls": tls}
            if bus_settings.security.nats_username and bus_settings.security.nats_password:
                kwargs["user"] = bus_settings.security.nats_username
                kwargs["password"] = bus_settings.security.nats_password
            self.nc = await NATSClient().connect(**kwargs)
            # JetStream kann in nats:alpine mit -js oder ohne existieren; robust prüfen
            try:
                # Prüfe ob jetstream() Methode verfügbar ist
                if hasattr(self.nc, "jetstream") and callable(getattr(self.nc, "jetstream", None)):
                    self.js = self.nc.jetstream()
                else:
                    # Fallback: Versuche JetStreamContext direkt zu erstellen
                    try:
                        self.js = JetStreamContext(self.nc)
                    except Exception:
                        logger.debug("JetStreamContext konnte nicht erstellt werden")
                        self.js = None

                # Einfacher Health-Check: Account Info abfragen
                if self.js and hasattr(self.js, "account_info"):
                    try:
                        await self.js.account_info()  # type: ignore[attr-defined]
                    except (AttributeError, ConnectionError, TimeoutError) as e:
                        logger.debug(f"JetStream Account-Info nicht verfügbar: {e}")
                        # JetStream eventuell nicht aktiv; auf None setzen
                        self.js = None
                    except Exception as e:
                        logger.warning(f"Unerwarteter Fehler bei JetStream Account-Info: {e}")
                        self.js = None
            except (AttributeError, ConnectionError) as e:
                logger.debug(f"JetStream-Initialisierung fehlgeschlagen: {e}")
                self.js = None
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler bei JetStream-Setup: {e}")
                self.js = None
            await self._setup_streams()
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"NATS-Verbindung fehlgeschlagen - Netzwerk-/System-Fehler: {e}")
            self.nc = None
            self.js = None
        except (ValueError, TypeError) as e:
            logger.error(f"NATS-Verbindung fehlgeschlagen - Konfigurationsfehler: {e}")
            self.nc = None
            self.js = None
        except Exception as e:
            logger.exception(f"NATS-Verbindung fehlgeschlagen - Unerwarteter Fehler: {e}")
            self.nc = None
            self.js = None

    async def _setup_streams(self) -> None:
        if not self.js:
            return
        for name, subjects in (
            ("AGENTS", ["kei.agents.>", "kei.agents.>.key.>"]),
            ("TASKS", ["kei.tasks.>", "kei.tasks.>.key.>"]),
            ("EVENTS", ["kei.events.>", "kei.events.>.key.>"]),
            ("A2A", ["kei.a2a.>", "kei.a2a.>.key.>"]),
        ):
            with contextlib.suppress(Exception):
                if self.js is not None:
                    await self.js.add_stream(name=name, subjects=subjects)
        with contextlib.suppress(Exception):
            if self.js is not None:
                await ensure_dlq_stream(self.js)

    async def publish(self, envelope: BusEnvelope) -> None:
        if not self.nc:
            from core.exceptions import KeikoServiceError
            raise KeikoServiceError("NATS nicht verbunden")

        # Outbox persist
        with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
            await Outbox("default").persist(envelope.id, envelope.model_dump())

        # Idempotenz (optional aktivierbar)
        if bool(getattr(bus_settings, "enable_publish_idempotency", False)):
            if await is_duplicate("publish", envelope.id):
                logger.debug(structured_msg("Duplicate publish ignoriert", correlation_id=envelope.corr_id, causation_id=envelope.causation_id, tenant=envelope.tenant, subject=envelope.subject, type=envelope.type, message_id=envelope.id))
                return

        # Redaction/Encryption
        if bus_settings.redact_payload_before_send:
            envelope.payload = redact_payload(envelope.payload)
        if bus_settings.enable_field_encryption and bus_settings.encryption_fields:
            envelope.payload = encrypt_fields(envelope.payload, bus_settings.encryption_fields, get_active_kms_key_id())

        # Tracing/Headers
        headers = inject_trace(envelope.headers)
        if _OTEL:
            with contextlib.suppress(Exception):
                propagate.inject(headers)
        envelope.headers = headers
        if envelope.corr_id:
            envelope.headers.setdefault("correlation_id", envelope.corr_id)
        if envelope.key:
            envelope.headers.setdefault("x-ordering-key", envelope.key)
        if "Nats-Msg-Id" not in envelope.headers:
            envelope.headers["Nats-Msg-Id"] = hashlib.sha1(f"{envelope.subject}:{envelope.key or ''}:{envelope.id}".encode()).hexdigest()

        # ACL
        authorize_message(envelope.headers, envelope.subject, action="publish")

        # Subject/Bytes
        subject_to_publish = envelope.subject
        if envelope.key:
            key_hash = hashlib.sha1(envelope.key.encode("utf-8")).hexdigest()[:12]
            subject_to_publish = f"{envelope.subject}.key.{key_hash}"
        data = json.dumps(envelope.model_dump()).encode("utf-8")
        started = asyncio.get_event_loop().time()

        # Chaos/Replay
        try:
            await store_replay(subject_to_publish, data)
            if not await apply_chaos(subject_to_publish, data):
                return
        except (ConnectionError, TimeoutError) as e:
            logger.debug(f"Chaos/Replay-Storage nicht verfügbar: {e}")
        except Exception as e:
            logger.warning(f"Unerwarteter Fehler bei Chaos/Replay: {e}")

        # Publish (mit optionalem Producer-Span)
        if _OTEL and _TRACER is not None:
            try:
                with _TRACER.start_as_current_span(f"nats.publish:{subject_to_publish}", kind=SpanKind.PRODUCER) as span:
                    span.set_attribute("messaging.system", "nats")
                    span.set_attribute("messaging.destination", subject_to_publish)
                    span.set_attribute("messaging.operation", "publish")
                    span.set_attribute("messaging.message_id", envelope.id)
                    if envelope.tenant:
                        span.set_attribute("messaging.tenant", envelope.tenant)
                    await self.nc.publish(subject_to_publish, data, headers=headers)  # type: ignore[arg-type]
            except Exception:
                await self.nc.publish(subject_to_publish, data, headers=headers)  # type: ignore[arg-type]
        else:
            await self.nc.publish(subject_to_publish, data, headers=headers)  # type: ignore[arg-type]

        # Metrics/Audit/Outbox cleanup
        self.metrics.mark_publish(envelope.subject, envelope.tenant)
        self.metrics.record_latency(envelope.subject, started, envelope.tenant)
        await remember("publish", envelope.id)
        with contextlib.suppress(Exception):
            await Outbox("default").remove(envelope.id)
        # Audit-Logging für Bus-Publish
        with contextlib.suppress(Exception):
            await action_logger.log_agent_action(
                action_type="bus_publish",
                correlation_id=envelope.corr_id or envelope.id,
                details={"subject": envelope.subject, "type": envelope.type, "tenant": envelope.tenant}
            )

    async def subscribe(
        self,
        subject: str,
        queue: str | None,
        handler: Callable[[BusEnvelope], Awaitable[None]],
        *,
        durable: str | None = None,
    ) -> None:
        if not self.js:
            # Fallback: Direktes Subscribe im Core NATS ohne JetStream
            if not self.nc:
                from core.exceptions import KeikoServiceError
                raise KeikoServiceError("NATS nicht verbunden")
            async def _raw_cb(msg):
                try:
                    payload = json.loads(msg.data.decode("utf-8"))
                    env = BusEnvelope(**payload)
                    await handler(env)
                finally:
                    with contextlib.suppress(Exception):
                        await msg.ack()
            try:
                await self.nc.subscribe(subject, cb=_raw_cb)  # type: ignore[arg-type]
                return
            except (ConnectionError, TimeoutError) as e:
                from core.exceptions import KeikoServiceError
                raise KeikoServiceError("NATS-Verbindung für Subscribe fehlgeschlagen") from e
            except Exception as e:
                from core.exceptions import KeikoServiceError
                raise KeikoServiceError("JetStream nicht verfügbar") from e

        async def _on_message(msg):
            span_cm = None
            span = None
            env = None  # Initialize to avoid unbound variable in exception handlers
            try:
                payload = json.loads(msg.data.decode("utf-8"))
                env = BusEnvelope(**payload)
                # Consumer Span
                if _OTEL and _TRACER is not None:
                    try:
                        headers_dict = dict(msg.headers or {})
                        remote_ctx = propagate.extract(headers_dict)
                        try:
                            prod_ctx = trace.get_current_span(remote_ctx).get_span_context()
                            links = [Link(prod_ctx)] if prod_ctx.is_valid else None
                        except Exception:
                            links = None
                        span_cm = _TRACER.start_as_current_span(f"nats.consume:{env.subject}", context=remote_ctx, kind=SpanKind.CONSUMER, links=links)
                        span = span_cm.__enter__()
                        span.set_attribute("messaging.system", "nats")
                        span.set_attribute("messaging.destination", env.subject)
                        span.set_attribute("messaging.operation", "process")
                        span.set_attribute("messaging.message_id", env.id)
                        if env.tenant:
                            span.set_attribute("messaging.tenant", env.tenant)
                    except (AttributeError, ImportError) as e:
                        logger.debug(f"OpenTelemetry-Span konnte nicht erstellt werden: {e}")
                        span_cm = None
                    except Exception as e:
                        logger.warning(f"Unerwarteter Fehler bei Tracing-Setup: {e}")
                        span_cm = None
                # ACL
                try:
                    authorize_message(dict(msg.headers or {}), env.subject, action="consume")
                except PermissionError:
                    if env is not None:
                        logger.warning(structured_msg("Consume abgelehnt (ACL)", correlation_id=env.corr_id, causation_id=env.causation_id, tenant=env.tenant, subject=env.subject, type=env.type, message_id=env.id))
                    await msg.term()
                    return
                # Inbox/Idempotenz
                try:
                    if env is not None and await is_duplicate("consume", env.id):
                        self.metrics.mark_redelivery(env.subject, env.tenant)
                        await msg.ack()
                        return
                except (ConnectionError, TimeoutError) as e:
                    logger.debug(f"Idempotenz-Prüfung fehlgeschlagen - Verbindungsproblem: {e}")
                except Exception as e:
                    logger.warning(f"Unerwarteter Fehler bei Idempotenz-Prüfung: {e}")
                with contextlib.suppress(ConnectionError, TimeoutError, ValueError):
                    if env is not None:
                        await Inbox("default").ack(env.id)
                # Decrypt
                with contextlib.suppress(ValueError, TypeError, KeyError):
                    env.payload = decrypt_fields(env.payload)
                # Ordering
                ok = env.key or dict(msg.headers or {}).get("x-ordering-key")
                if ok:
                    lk = self._key_locks.get(ok)
                    if lk is None:
                        lk = asyncio.Lock()
                        self._key_locks[ok] = lk
                    async with lk:
                        await handler(env)
                else:
                    await handler(env)
                # Ack + Metrics
                self.metrics.mark_consume(env.subject, env.tenant)
                await msg.ack()
                with contextlib.suppress(Exception):
                    await remember("consume", env.id)
                # Audit-Logging für Bus-Consume
                with contextlib.suppress(Exception):
                    await action_logger.log_agent_action(
                        action_type="bus_consume",
                        correlation_id=env.corr_id or env.id,
                        details={"subject": env.subject, "type": env.type, "tenant": env.tenant}
                    )
            except (ValueError, TypeError, KeyError) as e:
                logger.error(structured_msg("Handler-Fehler - Daten-/Validierungsfehler", error=str(e), correlation_id=getattr(env, "corr_id", None) if "env" in locals() else None, causation_id=getattr(env, "causation_id", None) if "env" in locals() else None, tenant=getattr(env, "tenant", None) if "env" in locals() else None, subject=subject, type=getattr(env, "type", None) if "env" in locals() else None, message_id=getattr(env, "id", None) if "env" in locals() else None))
            except Exception:
                logger.exception(structured_msg("Handler-Fehler - Unerwarteter Fehler", correlation_id=getattr(env, "corr_id", None) if "env" in locals() else None, causation_id=getattr(env, "causation_id", None) if "env" in locals() else None, tenant=getattr(env, "tenant", None) if "env" in locals() else None, subject=subject, type=getattr(env, "type", None) if "env" in locals() else None, message_id=getattr(env, "id", None) if "env" in locals() else None))
                # Retry mit Backoff/Jitter
                try:
                    cur = 0
                    try:
                        cur = int(dict(msg.headers or {}).get("x-retry-attempt") or 0)
                    except (ValueError, TypeError, KeyError):
                        cur = 0
                    except Exception as e:
                        logger.debug(f"Fehler beim Parsen des Retry-Attempt Headers: {e}")
                        cur = 0
                    nxt = cur + 1
                    if nxt > bus_settings.qos.max_redeliveries:
                        try:
                            dlq_subject = f"kei.dlq.{subject}"
                            if self.nc is not None:
                                await self.nc.publish(dlq_subject, msg.data, headers=msg.headers)
                            self.metrics.mark_dlq(env.subject if "env" in locals() else subject, getattr(env, "tenant", None))
                        except Exception:
                            pass
                        with contextlib.suppress(Exception):
                            await msg.term()
                        return
                    base = max(1, int(bus_settings.qos.retry_backoff_initial_ms))
                    cap = max(base, int(bus_settings.qos.retry_backoff_max_ms))
                    jitter = max(0, int(bus_settings.qos.retry_backoff_jitter_ms))
                    backoff = min(cap, base * (2 ** (nxt - 1)))
                    delay_ms = backoff + (random.randint(0, jitter) if jitter > 0 else 0)
                    with contextlib.suppress(Exception):
                        await msg.ack()
                    hdrs = dict(msg.headers or {})
                    hdrs["x-retry-attempt"] = str(nxt)
                    hdrs["x-retry-backoff-ms"] = str(delay_ms)
                    hdrs["x-retry-next-at"] = datetime.now(UTC).isoformat()
                    asyncio.create_task(self._republish_raw_with_delay(subject, msg.data, hdrs, delay_ms))
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(f"Retry-Republish fehlgeschlagen - Verbindungsproblem: {e}")
                    try:
                        dlq_subject = f"kei.dlq.{subject}"
                        if self.nc is not None:
                            await self.nc.publish(dlq_subject, msg.data, headers=msg.headers)
                        await msg.term()
                    except (ConnectionError, TimeoutError):
                        logger.error("DLQ-Publish fehlgeschlagen - Verbindung nicht verfügbar")
                    except Exception as dlq_err:
                        logger.error(f"DLQ-Publish fehlgeschlagen: {dlq_err}")
                except Exception as e:
                    logger.error(f"Unerwarteter Fehler bei Retry-Logik: {e}")
                    try:
                        dlq_subject = f"kei.dlq.{subject}"
                        if self.nc is not None:
                            await self.nc.publish(dlq_subject, msg.data, headers=msg.headers)
                        await msg.term()
                    except (ConnectionError, TimeoutError):
                        logger.error("DLQ-Publish fehlgeschlagen - Verbindung nicht verfügbar")
                    except Exception as dlq_err:
                        logger.error(f"DLQ-Publish fehlgeschlagen: {dlq_err}")
            finally:
                if span_cm is not None:
                    with contextlib.suppress(AttributeError, TypeError):
                        span_cm.__exit__(None, None, None)

        use_pull = bool(getattr(bus_settings.flow, "use_pull_subscriber", False)) and self.js is not None and hasattr(self.js, "pull_subscribe")
        key_wildcard = f"{subject}.key.>"
        if not use_pull:
            # Push-Subscriber mit Consumer-Optionen (ack_wait, max_ack_pending)
            try:
                if self.js is not None:
                    await self.js.subscribe(
                        subject,
                        queue=queue,
                        durable=durable,
                        cb=_on_message,
                        manual_ack=True,
                        ack_wait=bus_settings.flow.ack_wait_ms / 1000.0,  # Sekunden
                        max_ack_pending=bus_settings.flow.max_ack_pending,
                    )
            except Exception:
                # Fallback für einfache Fake/Legacy-Clients ohne zusätzliche Optionen
                if self.js is not None:
                    await self.js.subscribe(subject, queue=queue, durable=durable, cb=_on_message)
            try:
                if self.js is not None:
                    await self.js.subscribe(
                        key_wildcard,
                        queue=queue,
                        durable=durable,
                        cb=_on_message,
                        manual_ack=True,
                        ack_wait=bus_settings.flow.ack_wait_ms / 1000.0,
                        max_ack_pending=bus_settings.flow.max_ack_pending,
                    )
            except Exception:
                # Fallback-Subscribe ohne zusätzliche Optionen
                if self.js is not None:
                    await self.js.subscribe(key_wildcard, queue=queue, durable=durable, cb=_on_message)
        else:
            asyncio.create_task(self._pull_loop(subject, _on_message))
            asyncio.create_task(self._pull_loop(key_wildcard, _on_message))

    async def _republish_raw_with_delay(self, subject: str, data: bytes, headers: dict[str, Any], delay_ms: int) -> None:
        try:
            await asyncio.sleep(max(0, delay_ms) / 1000.0)
            if self.nc is not None:
                await self.nc.publish(subject, data, headers=headers)  # type: ignore[arg-type]
        except Exception as e:
            logger.exception(f"Raw Re-Publish nach Delay fehlgeschlagen: {e}")

    async def _pull_loop(self, subject: str, cb) -> None:
        try:
            if not self.js:
                return
            durable = f"durable_{hash(subject) & 0xFFFFFFFF:x}"
            sub = await self.js.pull_subscribe(subject=subject, durable=durable)
            batch = max(1, bus_settings.flow.pull_batch_size)
            tmo = max(0.1, bus_settings.flow.slow_consumer_timeout_ms / 1000.0)
            in_flight = 0
            max_in_flight = max(1, bus_settings.flow.max_in_flight)
            while True:
                try:
                    # Kreditorientierte Steuerung: nur fetch, wenn in_flight < max
                    if in_flight >= max_in_flight:
                        await asyncio.sleep(0.01)
                        continue
                    fetch_size = min(batch, max_in_flight - in_flight)
                    msgs = await sub.fetch(fetch_size, timeout=tmo)
                except Exception:
                    await asyncio.sleep(0.1)
                    continue
                for m in msgs:
                    try:
                        in_flight += 1
                        await cb(m)
                    except Exception:
                        pass
                    finally:
                        in_flight = max(0, in_flight - 1)
        except Exception as e:
            # Fängt unerwartete Fehler im Pull-Loop ab, um Endlosschleifen zu vermeiden
            logger.exception(f"Pull-Loop Fehler für Subject {subject}: {e}")
