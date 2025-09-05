"""KEI‑Webhook API Routen (Inbound/Outbound/Health).

Stellt Endpunkte für Inbound‑Verifikation, Outbound‑Enqueue, Target-
Management sowie Webhook‑Health bereit.
"""

from __future__ import annotations

import contextlib
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.middleware.scope_middleware import require_scopes
from config.settings import settings
from kei_logging import get_logger
from monitoring import record_custom_metric
from services.messaging import BusEnvelope, get_messaging_service
from services.messaging.naming import subject_for_event
from services.webhooks.audit_logger import (
    WebhookAuditOperation,
    webhook_audit,
)
from services.webhooks.exceptions import (
    WebhookAuthenticationException,
    WebhookValidationException,
)
from services.webhooks.health_prober import HealthProber
from services.webhooks.manager import get_webhook_manager
from services.webhooks.models import WebhookEventMeta, WebhookTarget
from services.webhooks.secret_manager import get_secret_manager
from services.webhooks.verification import InboundSignatureVerifier

from .base import create_router

logger = get_logger(__name__)

router = create_router("/webhooks", ["webhooks"])


class InboundAck(BaseModel):
    """Antwort auf einen Inbound‑Webhook."""

    status: str = "accepted"
    correlation_id: str | None = None


@router.post("/inbound/{topic}", response_model=InboundAck)
async def receive_inbound(
    topic: str,
    request: Request,
    x_signature: str = Header(default=""),
    x_timestamp: str = Header(default=""),
    x_nonce: str | None = Header(default=None),
    x_idempotency_key: str | None = Header(default=None),
) -> InboundAck:
    """Empfängt einen signierten Inbound‑Webhook und validiert diesen."""
    manager = get_webhook_manager()
    body = await request.body()

    # Frühes Audit: Inbound empfangen
    try:
        corr = request.headers.get("x-correlation-id") or None
        await webhook_audit.inbound_received(
            correlation_id=corr or "unknown",
            tenant_id=request.headers.get("x-tenant") or None,
            user_id=getattr(request.state, "user_id", None),
            topic=topic,
            details={"path": request.url.path},
        )
    except Exception:
        pass

    # Timestamp prüfen und validieren
    try:
        ts = int(x_timestamp)
    except (TypeError, ValueError) as exc:
        raise WebhookValidationException(
            message="Ungültiger Timestamp",
            error_code="invalid_timestamp",
            status_code=400,
            context={"header": "X-KEI-Timestamp"},
        ) from exc
    if abs(int(time.time()) - ts) > 600:
        raise HTTPException(status_code=401, detail="Timestamp skew too large")

    try:
        await manager.verify_inbound(payload=body, signature=x_signature, timestamp=str(ts), nonce=x_nonce, idempotency_key=x_idempotency_key)
    except WebhookAuthenticationException as exc:
        exc.log(logger, level=20)
        with contextlib.suppress(Exception):
            await webhook_audit.inbound_rejected(
                correlation_id=request.headers.get("x-correlation-id") or None,
                tenant_id=request.headers.get("x-tenant") or None,
                user_id=getattr(request.state, "user_id", None),
                topic=topic,
                error_details={"code": exc.error_code, "message": exc.message},
            )
        return JSONResponse(content=exc.to_dict(), status_code=exc.status_code)
    except WebhookValidationException as exc:
        exc.log(logger, level=20)
        with contextlib.suppress(Exception):
            await webhook_audit.inbound_rejected(
                correlation_id=request.headers.get("x-correlation-id") or None,
                tenant_id=request.headers.get("x-tenant") or None,
                user_id=getattr(request.state, "user_id", None),
                topic=topic,
                error_details={"code": exc.error_code, "message": exc.message},
            )
        return JSONResponse(content=exc.to_dict(), status_code=exc.status_code)

    # Payload lesen (tolerant bei Non‑JSON)
    try:
        payload_json: dict[str, Any] = await request.json()
    except Exception:
        payload_json = {"raw": body.decode("utf-8", errors="ignore")}

    # Schema-Validierung (wenn strukturierte JSON‑Payload vorliegt)
    if isinstance(payload_json, dict) and "raw" not in payload_json:
        # Bestimme Event‑Typ und Schema‑Version
        schema_version = request.headers.get("x-kei-schema-version") or (payload_json.get("meta", {}) or {}).get("schema_version")
        verifier = InboundSignatureVerifier()
        try:
            await verifier.validate_schema(event_type=topic, payload_obj=payload_json.get("data", {}), schema_version=schema_version)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - mapping zur HTTP Antwort
            return JSONResponse(content={"error": {"code": "schema_validation_failed", "message": str(exc)}}, status_code=422)

    # Auf KEI‑Bus publizieren (standardisiertes Subject)
    tenant = request.headers.get("x-tenant") or None
    subject = subject_for_event(tenant=tenant or "public", bounded_context="webhook", aggregate="inbound", event=topic, version=1)
    env = BusEnvelope(
        type=f"webhook.inbound.{topic}",
        subject=subject,
        tenant=tenant,
        payload={"topic": topic, "data": payload_json},
        headers={"x-source": "webhook", "x-idempotency-key": x_idempotency_key or ""},
    )
    try:
        bus = get_messaging_service()
        await bus.initialize()
        await bus.publish(env)
    except Exception as exc:  # pragma: no cover - Bus optional
        logger.debug(f"Inbound Bus Publish fehlgeschlagen: {exc}")

    # Audit‑Log (Best Effort)
    with contextlib.suppress(Exception):
        await webhook_audit.inbound_validated(
            correlation_id=payload_json.get("correlation_id") or env.corr_id or env.id,
            tenant_id=tenant,
            user_id=getattr(request.state, "user_id", None),
            topic=topic,
            details={"subject": subject},
        )

    return InboundAck(status="accepted", correlation_id=env.corr_id or env.id)


class TargetRequest(BaseModel):
    """Request für Target‑Anlage/Aktualisierung.

    Rückwärtskompatibilität: `secret` wird akzeptiert, aber wenn Key Vault
    konfiguriert ist, wird der Wert als neue Version in KV gespeichert und
    nicht im Target persistiert.
    """

    id: str
    url: str
    # BC: Klartext‑Secret optional (wird bei KV in Version überführt)
    secret: str | None = None
    # Key Vault Felder
    secret_key_name: str | None = None
    secret_version: str | None = None
    enabled: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    max_attempts: int = 5
    backoff_seconds: float = 1.0


class CreateTargetRequest(BaseModel):
    """Request für Target‑Erstellung mit optionaler ID (vom Frontend).

    Falls keine ID angegeben ist, wird eine ID aus `name` oder Host der URL
    abgeleitet und normalisiert (Kleinbuchstaben, Alphanumerik/Bindestrich).
    """

    id: str | None = None
    url: str
    name: str | None = None
    secret: str | None = None
    secret_key_name: str | None = None
    secret_version: str | None = None
    enabled: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    max_attempts: int = 5
    backoff_seconds: float = 1.0

@router.put("/targets/{target_id}")
async def upsert_target(target_id: str, request: TargetRequest, http_request: Request) -> dict[str, Any]:
    """Erstellt oder aktualisiert ein Ziel."""
    # RBAC: Target Verwaltung
    require_scopes(http_request, ["webhook:targets:manage"])
    if request.id != target_id:
        raise HTTPException(status_code=400, detail="ID mismatch")
    manager = get_webhook_manager()

    # Helper: kanonischen Secret‑Namen generieren (KV erlaubt nur a‑z, A‑Z, 0‑9, -)
    def canonical_secret_name(target_identifier: str) -> str:
        name = re.sub(r"[^0-9a-zA-Z-]", "-", target_identifier)
        return f"kei-webhook-{name}-hmac"

    # Key Vault: Optionales Einspielen eines Klartext‑Secrets in neue Version
    secret_key_name = request.secret_key_name
    secret_version = request.secret_version
    if settings.azure_key_vault_url:
        if not secret_key_name:
            secret_key_name = canonical_secret_name(request.id)
        if request.secret:
            try:
                sm = get_secret_manager()
                _, ver = await sm.rotate_secret(key_name=secret_key_name, new_value=request.secret)
                secret_version = ver
                # Metrik/Audit
                record_custom_metric("webhook.secret.rotation.success", 1, {"key": secret_key_name, "op": "upsert_seed"})
            except Exception as exc:
                record_custom_metric("webhook.secret.rotation.failure", 1, {"key": secret_key_name, "op": "upsert_seed"})
                raise HTTPException(status_code=500, detail=f"Secret write failed: {exc}")

    # Target Objekt ohne Klartext‑Secret erstellen
    target = WebhookTarget(
        id=request.id,
        url=request.url,
        secret_key_name=secret_key_name,
        secret_version=secret_version,
        enabled=request.enabled,
        headers=request.headers,
        max_attempts=request.max_attempts,
        backoff_seconds=request.backoff_seconds,
        tenant_id=http_request.headers.get("X-Tenant-Id") or http_request.headers.get("x-tenant"),
    )
    registry = manager.get_targets_registry(target.tenant_id)
    await registry.upsert(target)
    try:
        op = WebhookAuditOperation.CREATE if not await manager.targets.get(request.id) else WebhookAuditOperation.UPDATE
    except Exception:
        op = WebhookAuditOperation.UPDATE
    with contextlib.suppress(Exception):
        await webhook_audit.target_changed(
            operation=op,
            target_id=request.id,
            user_id=getattr(getattr(request, "state", object()), "user_id", None),
            tenant_id=None,
            correlation_id=request.headers.get("x-correlation-id") if hasattr(request, "headers") else None,
            details={"url": request.url},
        )
    return {"status": "ok"}


@router.post("/targets")
async def create_target(request: CreateTargetRequest, http_request: Request) -> dict[str, Any]:
    """Erstellt ein neues Target.

    - Erfordert Scope `webhook:targets:manage`.
    - Unterstützt optionales HMAC‑Secret‑Seeding über Azure Key Vault wie beim Upsert.
    """
    require_scopes(http_request, ["webhook:targets:manage"])  # type: ignore[name-defined]

    # ID aus Payload oder ableiten
    target_id = request.id
    if not target_id or not target_id.strip():
        # ID aus name oder URL ableiten
        raw = request.name or request.url
        try:
            if not request.name:
                from urllib.parse import urlparse
                host = urlparse(request.url).hostname or "target"
                raw = host
        except Exception:
            raw = raw or "target"
        # Normalisieren: Kleinbuchstaben, nicht alphanumerische Zeichen → '-'
        target_id = re.sub(r"[^0-9a-zA-Z-]", "-", (raw or "target").lower()).strip("-") or "target"

    manager = get_webhook_manager()

    # Helper: kanonischen Secret‑Namen generieren (KV erlaubt nur a‑z, A‑Z, 0‑9, -)
    def canonical_secret_name(target_identifier: str) -> str:
        name = re.sub(r"[^0-9a-zA-Z-]", "-", target_identifier)
        return f"kei-webhook-{name}-hmac"

    # Key Vault: Optionales Einspielen eines Klartext‑Secrets in neue Version
    secret_key_name = request.secret_key_name
    secret_version = request.secret_version
    if settings.azure_key_vault_url:
        if not secret_key_name:
            secret_key_name = canonical_secret_name(request.id)
        if request.secret:
            try:
                sm = get_secret_manager()
                _, ver = await sm.rotate_secret(key_name=secret_key_name, new_value=request.secret)
                secret_version = ver
                record_custom_metric("webhook.secret.rotation.success", 1, {"key": secret_key_name, "op": "create_seed"})
            except Exception as exc:
                record_custom_metric("webhook.secret.rotation.failure", 1, {"key": secret_key_name, "op": "create_seed"})
                raise HTTPException(status_code=500, detail=f"Secret write failed: {exc}")

    # Target Objekt ohne Klartext‑Secret erstellen
    target = WebhookTarget(
        id=target_id,
        url=request.url,
        secret_key_name=secret_key_name,
        secret_version=secret_version,
        enabled=request.enabled,
        headers=request.headers,
        max_attempts=request.max_attempts,
        backoff_seconds=request.backoff_seconds,
        tenant_id=http_request.headers.get("X-Tenant-Id") or http_request.headers.get("x-tenant"),
    )
    registry = manager.get_targets_registry(target.tenant_id)
    await registry.upsert(target)
    with contextlib.suppress(Exception):
        await webhook_audit.target_changed(
            operation=WebhookAuditOperation.CREATE,
            target_id=target_id,
            user_id=getattr(getattr(http_request, "state", object()), "user_id", None),
            tenant_id=None,
            correlation_id=http_request.headers.get("x-correlation-id") if hasattr(http_request, "headers") else None,
            details={"url": request.url},
        )
    return {"status": "ok", "target_id": target_id}


class RotateSecretRequest(BaseModel):
    """Request für manuelle Secret‑Rotation eines Targets."""

    force: bool = Field(default=False, description="Rotation erzwingen, auch wenn Intervall nicht erreicht ist")
    new_value: str | None = Field(default=None, description="Optionaler vorgegebener Secret‑Wert")


@router.post("/targets/{target_id}/rotate-secret")
async def rotate_secret(target_id: str, payload: RotateSecretRequest) -> dict[str, Any]:
    """Rotiert das HMAC‑Secret eines Targets über Azure Key Vault.

    - Setzt `previous_secret_version` und aktualisiert `secret_version`
    - Setzt Grace‑Periode gemäß Settings
    - Auditiert Operationen
    """
    manager = get_webhook_manager()
    target = await manager.targets.get(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    if not settings.azure_key_vault_url:
        raise HTTPException(status_code=400, detail="Key Vault not configured")
    key_name = target.secret_key_name or re.sub(r"[^0-9a-zA-Z-]", "-", f"kei-webhook-{target.id}-hmac")
    sm = get_secret_manager()

    old_version = target.secret_version
    try:
        _, new_version = await sm.rotate_secret(key_name=key_name, new_value=payload.new_value)
        now = datetime.now(UTC)
        target.previous_secret_version = old_version
        target.secret_version = new_version
        target.secret_last_rotated_at = now
        target.secret_grace_until = now + timedelta(hours=int(settings.secret_grace_period_hours))
        target.secret_key_name = key_name
        await manager.targets.upsert(target)
        with contextlib.suppress(Exception):
            await webhook_audit.target_changed(
                operation=WebhookAuditOperation.UPDATE,
                target_id=target_id,
                user_id=getattr(getattr(payload, "state", object()), "user_id", None) if hasattr(payload, "state") else None,
                tenant_id=None,
                correlation_id=None,
                details={"secret_rotated": True, "key": key_name},
            )
        return {"status": "rotated", "target_id": target_id, "version": new_version}
    except Exception as exc:
        # Rollback: nichts zu tun, da alte Version unverändert
        with contextlib.suppress(Exception):
            await webhook_audit.target_changed(
                operation=WebhookAuditOperation.UPDATE,
                target_id=target_id,
                user_id=None,
                tenant_id=None,
                correlation_id=None,
                details={"secret_rotated": False, "error": str(exc)},
            )
        raise HTTPException(status_code=500, detail=f"Rotation failed: {exc}")


class SecretStatusResponse(BaseModel):
    """Antwortmodell für Secret‑Status eines Targets."""

    secret_key_name: str | None
    secret_version: str | None
    previous_secret_version: str | None
    secret_last_rotated_at: str | None
    secret_grace_until: str | None
    uses_legacy_secret: bool


@router.get("/targets/{target_id}/secret-status", response_model=SecretStatusResponse)
async def secret_status(target_id: str) -> SecretStatusResponse:
    """Liefert Status der Secret‑Konfiguration für ein Target."""
    manager = get_webhook_manager()
    target = await manager.targets.get(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    return SecretStatusResponse(
        secret_key_name=target.secret_key_name,
        secret_version=target.secret_version,
        previous_secret_version=target.previous_secret_version,
        secret_last_rotated_at=target.secret_last_rotated_at.isoformat() if target.secret_last_rotated_at else None,
        secret_grace_until=target.secret_grace_until.isoformat() if target.secret_grace_until else None,
        uses_legacy_secret=bool(target.legacy_secret and not target.secret_key_name),
    )


class WebhookTargetsResponse(BaseModel):
    """Response für Webhook Targets Liste."""
    items: list[WebhookTarget]


@router.get("/targets", response_model=WebhookTargetsResponse)
async def list_targets(request: Request) -> WebhookTargetsResponse:
    """Listet alle registrierten Ziele."""
    require_scopes(request, ["webhook:targets:manage"])
    try:
        manager = get_webhook_manager()
        tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
        registry = manager.get_targets_registry(tenant_id)
        targets = await registry.list()
        # Sicherstellen, dass targets eine Liste ist
        if targets is None:
            targets = []
        return WebhookTargetsResponse(items=targets)
    except Exception as e:
        logger.exception(f"Fehler beim Laden der Webhook Targets: {e}")
        # Fallback: leere Liste zurückgeben
        return WebhookTargetsResponse(items=[])


class OutboundEnqueueRequest(BaseModel):
    """Anfrage zur Outbound‑Enqueue eines Events."""

    target_id: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    meta: WebhookEventMeta | None = None


class OutboundEnqueueResponse(BaseModel):
    """Antwort mit IDs für Nachverfolgung."""

    delivery_id: str
    event_id: str


@router.post("/outbound/enqueue", response_model=OutboundEnqueueResponse)
async def enqueue_outbound(http_request: Request, request: OutboundEnqueueRequest) -> OutboundEnqueueResponse:
    """Plant einen Outbound‑Webhook in der Outbox ein."""
    # RBAC: erfordert spezifischen Event‑Scope
    require_scopes(http_request, [f"webhook:outbound:send:{request.event_type}"])
    manager = get_webhook_manager()
    # Tenant in Meta sicherstellen (BC: Header → Meta)
    if request.meta is None:
        request.meta = WebhookEventMeta(tenant=http_request.headers.get("X-Tenant-Id") or http_request.headers.get("x-tenant"))
    elif request.meta.tenant is None:
        request.meta.tenant = http_request.headers.get("X-Tenant-Id") or http_request.headers.get("x-tenant")
    delivery_id, event_id = await manager.enqueue_outbound(
        target_id=request.target_id,
        event_type=request.event_type,
        data=request.data,
        meta=request.meta,
    )
    return OutboundEnqueueResponse(delivery_id=delivery_id, event_id=event_id)


class TargetHealthResponse(BaseModel):
    """Antwortmodell für Health‑Status eines Targets."""

    target_id: str
    health_status: str
    last_health_check: str | None


@router.get("/targets/{target_id}/health", response_model=TargetHealthResponse)
async def get_target_health(target_id: str, request: Request) -> TargetHealthResponse:
    """Liefert den aktuellen Health‑Status eines Targets."""
    require_scopes(request, ["webhook:targets:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
    registry = manager.get_targets_registry(tenant_id)
    target = await registry.get(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    return TargetHealthResponse(
        target_id=target.id,
        health_status=target.health_status or "unknown",
        last_health_check=target.last_health_check.isoformat() if target.last_health_check else None,
    )


class ProbeRequest(BaseModel):
    """Request zur aktiven Health‑Prüfung eines Targets."""

    timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)


@router.post("/targets/{target_id}/probe", response_model=TargetHealthResponse)
async def probe_target_health(target_id: str, request: Request, payload: ProbeRequest) -> TargetHealthResponse:
    """Führt einen aktiven Health‑Probe gegen ein Target aus."""
    require_scopes(request, ["webhook:targets:manage"])  # type: ignore[name-defined]
    tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
    prober = HealthProber(poll_seconds=3600.0, timeout_seconds=payload.timeout_seconds)
    # Einmaliger Probe für Ziel
    await prober._probe_one(target_id=target_id, tenant_id=tenant_id)  # noqa: SLF001
    # Ergebnis laden
    manager = get_webhook_manager()
    registry = manager.get_targets_registry(tenant_id)
    t = await registry.get(target_id)
    if not t:
        raise HTTPException(status_code=404, detail="Target not found after probe")
    return TargetHealthResponse(
        target_id=t.id,
        health_status=t.health_status or "unknown",
        last_health_check=t.last_health_check.isoformat() if t.last_health_check else None,
    )


class TargetPatchRequest(BaseModel):
    """Teilaktualisierung eines Targets.

    Ermöglicht einfache UI‑Aktionen wie Aktivieren/Deaktivieren ohne komplettes Upsert.
    """

    is_active: bool | None = Field(default=None, description="Aktivierungsstatus des Targets")


@router.patch("/targets/{target_id}")
async def patch_target(target_id: str, request: Request, payload: TargetPatchRequest) -> dict[str, Any]:
    """Aktualisiert Teilaspekte eines Targets (z. B. Aktivierungsstatus).

    - Erwartet `is_active` im Payload (UI‑Kompatibilität). Wird auf `enabled` gemappt.
    - Erfordert Scope `webhook:targets:manage`.
    """
    require_scopes(request, ["webhook:targets:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
    registry = manager.get_targets_registry(tenant_id)
    target = await registry.get(target_id)
    # Fallback: Falls im aktuellen Tenant kein Target gefunden wird, versuche Default‑Tenant
    if not target:
        fallback_registry = manager.get_targets_registry(None)
        fallback_target = await fallback_registry.get(target_id)
        if fallback_target:
            registry = fallback_registry
            target = fallback_target
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")

    # UI‑Feld `is_active` → Backend‑Feld `enabled`
    if payload.is_active is not None:
        target.enabled = bool(payload.is_active)

    await registry.upsert(target)
    with contextlib.suppress(Exception):
        await webhook_audit.target_changed(
            operation=WebhookAuditOperation.UPDATE,
            target_id=target_id,
            user_id=getattr(getattr(request, "state", object()), "user_id", None),
            tenant_id=None,
            correlation_id=request.headers.get("x-correlation-id") if hasattr(request, "headers") else None,
            details={"enabled": target.enabled},
        )
    return {"status": "ok", "target_id": target_id}


@router.delete("/targets/{target_id}")
async def delete_target(target_id: str, request: Request) -> dict[str, Any]:
    """Löscht ein vorhandenes Target.

    - Erfordert Scope `webhook:targets:manage`.
    - Nutzt Tenancy‑Header zur Auswahl des Registries.
    """
    require_scopes(request, ["webhook:targets:manage"])  # type: ignore[name-defined]
    manager = get_webhook_manager()
    tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
    registry = manager.get_targets_registry(tenant_id)

    # Optionaler Fallback auf Default‑Tenant
    existing = await registry.get(target_id)
    if not existing:
        fallback_registry = manager.get_targets_registry(None)
        fallback = await fallback_registry.get(target_id)
        if fallback:
            registry = fallback_registry
            existing = fallback
    if not existing:
        raise HTTPException(status_code=404, detail="Target not found")

    await registry.delete(target_id)
    with contextlib.suppress(Exception):
        await webhook_audit.target_changed(
            operation=WebhookAuditOperation.DELETE,
            target_id=target_id,
            user_id=getattr(getattr(request, "state", object()), "user_id", None),
            tenant_id=None,
            correlation_id=request.headers.get("x-correlation-id") if hasattr(request, "headers") else None,
            details={},
        )
    return {"status": "deleted", "target_id": target_id}


class HealthSummary(BaseModel):
    """Gesundheitsstatus des Webhook‑Subsystems."""

    status: str
    outbox_depth: int
    worker_pool: dict[str, Any]


@router.get("/health")
async def webhook_health() -> JSONResponse:
    """Einfacher Health‑Check für das Webhook‑Subsystem (200/503)."""
    manager = get_webhook_manager()
    result = await manager.health()
    status_code = 200 if result.get("status") == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)


@router.get("/health/detailed")
async def webhook_health_detailed() -> JSONResponse:
    """Detaillierter Health‑Check mit Outbox und Worker‑Metriken (200/503)."""
    manager = get_webhook_manager()
    base = await manager.health()
    # Zusätzliche Checks: Redis erreichbar?
    details: dict[str, Any] = {"components": []}
    try:
        from storage.cache.redis_cache import NoOpCache, get_cache_client

        client = await get_cache_client()
        if client is None or isinstance(client, NoOpCache):
            details["components"].append({"name": "redis", "status": "unavailable"})
        else:
            try:
                ok = await client.ping()
            except Exception:
                ok = False
            details["components"].append({"name": "redis", "status": "healthy" if ok else "unhealthy"})
    except Exception:
        details["components"].append({"name": "redis", "status": "unknown"})

    result = {**base, **details}
    status_code = 200 if result.get("status") == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)


__all__ = ["router"]
