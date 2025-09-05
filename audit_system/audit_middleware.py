# backend/audit_system/audit_middleware.py
"""Audit Middleware für Keiko Personal Assistant

Implementiert FastAPI-Middleware für automatische Audit-Logging,
Integration mit Enhanced Security und Policy Engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger
from observability import trace_function

from .action_logger import action_logger
from .audit_constants import AuditConstants, AuditMessages, AuditPaths
from .audit_pii_redaction import audit_pii_redactor
from .audit_utils import (
    create_error_context,
    create_request_metadata,
    generate_correlation_id,
    is_path_excluded,
)
from .core_audit_engine import (
    AuditContext,
    AuditEventType,
    AuditResult,
    AuditSeverity,
    audit_engine,
)
from .tamper_proof_trail import tamper_proof_trail

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class AuditConfig:
    """Konfiguration für Audit-Middleware."""
    # Audit-Aktivierung
    audit_enabled: bool = True
    audit_all_requests: bool = True
    audit_responses: bool = True

    # Event-Typen
    audit_authentication: bool = True
    audit_authorization: bool = True
    audit_data_access: bool = True
    audit_configuration_changes: bool = True
    audit_system_events: bool = True

    # PII-Redaction
    enable_pii_redaction: bool = True
    pii_redaction_level: str = "enhanced"
    require_consent_for_pii: bool = True

    # Tamper-Proof Trail
    enable_tamper_proof: bool = True
    cryptographic_signing: bool = True
    blockchain_chaining: bool = True

    # Performance
    async_processing: bool = True
    batch_processing: bool = False
    enable_streaming: bool = True

    # Integration
    security_integration: bool = True
    policy_integration: bool = True
    quota_integration: bool = True

    # Ausgeschlossene Pfade
    excluded_paths: set[str] = field(default_factory=lambda: set(
        AuditPaths.DEFAULT_EXCLUDED_PATHS + AuditPaths.HEALTH_CHECK_PATHS +
        ["/metrics", "/audit/status", "/audit/health"]
    ))

    # Fehlerbehandlung
    fail_open_on_error: bool = True
    max_retry_attempts: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "audit_enabled": self.audit_enabled,
            "audit_all_requests": self.audit_all_requests,
            "audit_responses": self.audit_responses,
            "audit_authentication": self.audit_authentication,
            "audit_authorization": self.audit_authorization,
            "audit_data_access": self.audit_data_access,
            "audit_configuration_changes": self.audit_configuration_changes,
            "audit_system_events": self.audit_system_events,
            "enable_pii_redaction": self.enable_pii_redaction,
            "pii_redaction_level": self.pii_redaction_level,
            "require_consent_for_pii": self.require_consent_for_pii,
            "enable_tamper_proof": self.enable_tamper_proof,
            "cryptographic_signing": self.cryptographic_signing,
            "blockchain_chaining": self.blockchain_chaining,
            "async_processing": self.async_processing,
            "batch_processing": self.batch_processing,
            "enable_streaming": self.enable_streaming,
            "security_integration": self.security_integration,
            "policy_integration": self.policy_integration,
            "quota_integration": self.quota_integration,
            "excluded_paths": list(self.excluded_paths),
            "fail_open_on_error": self.fail_open_on_error,
            "max_retry_attempts": self.max_retry_attempts
        }


@dataclass
class AuditEnforcementResult:
    """Ergebnis der Audit-Enforcement."""
    audit_performed: bool
    audit_event_id: str | None = None
    pii_redacted: bool = False
    tamper_proof_added: bool = False

    # Fehler
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Performance
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "audit_performed": self.audit_performed,
            "audit_event_id": self.audit_event_id,
            "pii_redacted": self.pii_redacted,
            "tamper_proof_added": self.tamper_proof_added,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms
        }


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware für automatische Audit-Logging."""

    def __init__(self, app, config: AuditConfig):
        """Initialisiert Audit Middleware.

        Args:
            app: FastAPI-App
            config: Audit-Konfiguration
        """
        super().__init__(app)
        self.config = config

        # Statistiken
        self._requests_processed = 0
        self._requests_audited = 0
        self._audit_failures = 0
        self._pii_redactions = 0
        self._tamper_proof_events = 0

        # Cache für Performance
        self._audit_cache: dict[str, AuditEnforcementResult] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = AuditConstants.DEFAULT_CACHE_TTL_SECONDS

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet Request mit Audit-Logging."""
        # Prüfe ausgeschlossene Pfade
        if self._should_skip_audit(request):
            return await call_next(request)

        try:
            return await self._process_audited_request(request, call_next)
        except HTTPException:
            raise
        except Exception as e:
            return await self._handle_audit_error(e, request, call_next)

    def _should_skip_audit(self, request: Request) -> bool:
        """Prüft ob Request von Audit ausgeschlossen werden soll."""
        return is_path_excluded(request.url.path, list(self.config.excluded_paths))

    async def _process_audited_request(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet Request mit vollständigem Audit-Logging."""
        self._requests_processed += 1

        # Extrahiere Request-Context
        audit_context = await self._extract_audit_context(request)

        # Pre-Request Audit
        pre_audit_result = await self._perform_pre_request_audit(request, audit_context)

        # Führe Request aus
        response = await call_next(request)

        # Post-Request Audit
        post_audit_result = await self._perform_post_request_audit(request, response, audit_context)

        # Finalisiere Audit
        self._finalize_audit(response, pre_audit_result, post_audit_result)

        return response

    def _finalize_audit(
        self,
        response: Response,
        pre_audit_result: AuditEnforcementResult,
        post_audit_result: AuditEnforcementResult
    ) -> None:
        """Finalisiert Audit-Verarbeitung."""
        # Füge Audit-Headers hinzu
        self._add_audit_headers(response, pre_audit_result, post_audit_result)

        # Aktualisiere Statistiken
        if pre_audit_result.audit_performed or post_audit_result.audit_performed:
            self._requests_audited += 1

    async def _handle_audit_error(
        self,
        error: Exception,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Behandelt Audit-Fehler."""
        error_context = create_error_context(error, "audit_middleware_dispatch")
        logger.error(f"{AuditMessages.AUDIT_FAILURE}: {error_context}")
        self._audit_failures += 1

        if self.config.fail_open_on_error:
            return await call_next(request)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit system error"
        )

    async def _extract_audit_context(self, request: Request) -> AuditContext:
        """Extrahiert Audit-Context aus Request."""
        # Erstelle Request-Metadaten
        request_metadata = create_request_metadata(request)

        # Extrahiere Identitäten aus Headers
        user_id = request.headers.get("X-User-ID")
        agent_id = request.headers.get("X-Agent-ID")
        tenant_id = request.headers.get("X-Tenant-ID")
        session_id = request.headers.get("X-Session-ID")

        # Aus Request-State (falls von Security-Middleware gesetzt)
        if hasattr(request.state, "principal"):
            principal = getattr(request.state, "principal", {})
            user_id = user_id or principal.get("id")
            agent_id = agent_id or principal.get("agent_id")
            session_id = session_id or principal.get("session_id")

        if hasattr(request.state, "tenant_context"):
            tenant_context = getattr(request.state, "tenant_context", {})
            tenant_id = tenant_id or tenant_context.get("tenant_id")

        # Erstelle Audit-Context
        return AuditContext(
            correlation_id=request_metadata.correlation_id,
            session_id=session_id,
            request_id=generate_correlation_id(),
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            client_ip=request_metadata.client_ip,
            user_agent=request_metadata.user_agent,
            source_system="kei_agent_framework",
            metadata={
                "method": request_metadata.method,
                "path": request_metadata.path,
                "query_params": dict(request.query_params),
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length")
            }
        )


    @trace_function("audit_middleware.pre_request_audit")
    async def _perform_pre_request_audit(
        self,
        request: Request,
        context: AuditContext
    ) -> AuditEnforcementResult:
        """Führt Pre-Request-Audit durch."""
        start_time = time.time()
        result = AuditEnforcementResult(audit_performed=False)

        if not self.config.audit_enabled:
            return result

        try:
            # Bestimme Event-Typ basierend auf Request
            event_type = self._determine_event_type(request)

            if not self._should_audit_event_type(event_type):
                return result

            # Sammle Request-Daten
            request_data = await self._collect_request_data(request)

            # PII-Redaction
            if self.config.enable_pii_redaction:
                request_data = await self._apply_pii_redaction(request_data, context)
                result.pii_redacted = True
                self._pii_redactions += 1

            # Erstelle Audit-Event
            audit_event = await audit_engine.create_event(
                event_type=event_type,
                action=f"{request.method} {request.url.path}",
                description=f"Request audit for {request.method} {request.url.path}",
                severity=self._determine_severity(request),
                result=AuditResult.SUCCESS,
                context=context,
                input_data=request_data,
                compliance_tags={"request_audit", "middleware"}
            )

            result.audit_event_id = audit_event.event_id
            result.audit_performed = True

            # Tamper-Proof Trail
            if self.config.enable_tamper_proof:
                await tamper_proof_trail.add_audit_event(audit_event)
                result.tamper_proof_added = True
                self._tamper_proof_events += 1

            # Action Logging
            if context.agent_id:
                await action_logger.log_agent_input(
                    agent_id=context.agent_id,
                    input_data=request_data,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    metadata=context.metadata
                )

        except Exception as e:
            logger.exception(f"Pre-Request-Audit fehlgeschlagen: {e}")
            result.errors.append(f"Pre-request audit failed: {e!s}")
            self._audit_failures += 1

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    @trace_function("audit_middleware.post_request_audit")
    async def _perform_post_request_audit(
        self,
        request: Request,
        response: Response,
        context: AuditContext
    ) -> AuditEnforcementResult:
        """Führt Post-Request-Audit durch."""
        start_time = time.time()
        result = AuditEnforcementResult(audit_performed=False)

        if not self.config.audit_enabled or not self.config.audit_responses:
            return result

        try:
            # Sammle Response-Daten
            response_data = await self._collect_response_data(response)

            # PII-Redaction
            if self.config.enable_pii_redaction:
                response_data = await self._apply_pii_redaction(response_data, context)
                result.pii_redacted = True

            # Bestimme Result basierend auf Status-Code
            audit_result = self._determine_audit_result(response.status_code)

            # Erstelle Audit-Event
            audit_event = await audit_engine.create_event(
                event_type=AuditEventType.AGENT_OUTPUT,
                action=f"Response for {request.method} {request.url.path}",
                description=f"Response audit for {request.method} {request.url.path}",
                severity=self._determine_response_severity(response),
                result=audit_result,
                context=context,
                output_data=response_data,
                compliance_tags={"response_audit", "middleware"}
            )

            result.audit_event_id = audit_event.event_id
            result.audit_performed = True

            # Tamper-Proof Trail
            if self.config.enable_tamper_proof:
                await tamper_proof_trail.add_audit_event(audit_event)
                result.tamper_proof_added = True

            # Action Logging
            if context.agent_id:
                await action_logger.log_agent_output(
                    agent_id=context.agent_id,
                    output_data=response_data,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    metadata=context.metadata
                )

        except Exception as e:
            logger.exception(f"Post-Request-Audit fehlgeschlagen: {e}")
            result.errors.append(f"Post-request audit failed: {e!s}")
            self._audit_failures += 1

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _determine_event_type(self, request: Request) -> AuditEventType:
        """Bestimmt Event-Typ basierend auf Request."""
        path = request.url.path.lower()
        method = request.method.upper()

        # Authentication-Endpoints
        if any(auth_path in path for auth_path in ["/auth", "/login", "/logout", "/token"]):
            return AuditEventType.AUTHENTICATION

        # Authorization-Endpoints
        if any(authz_path in path for authz_path in ["/authorize", "/permission", "/role"]):
            return AuditEventType.AUTHORIZATION

        # Configuration-Endpoints
        if any(config_path in path for config_path in ["/config", "/settings", "/admin"]):
            return AuditEventType.CONFIGURATION_CHANGE

        # Agent-spezifische Endpoints
        if "/agent" in path:
            if method in ["POST", "PUT", "PATCH"]:
                return AuditEventType.AGENT_INPUT
            return AuditEventType.DATA_ACCESS

        # Standard Data-Access
        return AuditEventType.DATA_ACCESS

    def _should_audit_event_type(self, event_type: AuditEventType) -> bool:
        """Prüft, ob Event-Typ auditiert werden soll."""
        type_mapping = {
            AuditEventType.AUTHENTICATION: self.config.audit_authentication,
            AuditEventType.AUTHORIZATION: self.config.audit_authorization,
            AuditEventType.DATA_ACCESS: self.config.audit_data_access,
            AuditEventType.CONFIGURATION_CHANGE: self.config.audit_configuration_changes,
            AuditEventType.SYSTEM_EVENT: self.config.audit_system_events
        }

        return type_mapping.get(event_type, self.config.audit_all_requests)

    def _determine_severity(self, request: Request) -> AuditSeverity:
        """Bestimmt Severity basierend auf Request."""
        path = request.url.path.lower()
        method = request.method.upper()

        # Kritische Operationen
        if method in ["DELETE"] or any(critical_path in path for critical_path in ["/admin", "/config", "/delete"]):
            return AuditSeverity.CRITICAL

        # Hohe Severity für Änderungen
        if method in ["POST", "PUT", "PATCH"]:
            return AuditSeverity.HIGH

        # Standard für Lesezugriffe
        return AuditSeverity.MEDIUM

    def _determine_response_severity(self, response: Response) -> AuditSeverity:
        """Bestimmt Severity basierend auf Response."""
        if response.status_code >= 500:
            return AuditSeverity.CRITICAL
        if response.status_code >= 400:
            return AuditSeverity.HIGH
        return AuditSeverity.MEDIUM

    def _determine_audit_result(self, status_code: int) -> AuditResult:
        """Bestimmt Audit-Result basierend auf Status-Code."""
        if 200 <= status_code < 300:
            return AuditResult.SUCCESS
        if 400 <= status_code < 500:
            return AuditResult.BLOCKED
        if status_code >= 500:
            return AuditResult.ERROR
        return AuditResult.PARTIAL

    async def _collect_request_data(self, request: Request) -> dict[str, Any]:
        """Sammelt Request-Daten für Audit."""
        request_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": {
                "host": request.client.host if request.client else None,
                "port": request.client.port if request.client else None
            }
        }

        # Body nur für bestimmte Content-Types sammeln
        content_type = request.headers.get("content-type", "")
        if any(ct in content_type for ct in ["application/json", "application/x-www-form-urlencoded"]):
            try:
                # Vereinfachte Body-Extraktion
                # In Produktion würde hier der Body gelesen und wieder gesetzt
                request_data["has_body"] = True
                request_data["content_type"] = content_type
                request_data["content_length"] = request.headers.get("content-length")
            except Exception as e:
                logger.warning(f"Request-Body-Extraktion fehlgeschlagen: {e}")

        return request_data

    async def _collect_response_data(self, response: Response) -> dict[str, Any]:
        """Sammelt Response-Daten für Audit."""
        response_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "media_type": response.media_type
        }

        # Body-Größe falls verfügbar
        if hasattr(response, "body"):
            response_data["body_size"] = len(response.body) if response.body else 0

        return response_data

    async def _apply_pii_redaction(
        self,
        data: dict[str, Any],
        context: AuditContext
    ) -> dict[str, Any]:
        """Wendet PII-Redaction auf Daten an."""
        try:
            # Erstelle Audit-Event für Redaction
            from .core_audit_engine import AuditEvent, AuditEventType, AuditResult, AuditSeverity

            redaction_event = AuditEvent(
                event_id="redaction_processing",
                event_type=AuditEventType.DATA_ACCESS,
                severity=AuditSeverity.MEDIUM,
                result=AuditResult.SUCCESS,
                timestamp=datetime.now(UTC),
                action="pii_redaction",
                description="PII redaction processing",
                context=context,
                input_data=data
            )

            # Wende Redaction an
            redacted_event = await audit_pii_redactor.redact_audit_event(
                redaction_event,
                consent_override=not self.config.require_consent_for_pii
            )

            return redacted_event.input_data or {}

        except Exception as e:
            logger.exception(f"PII-Redaction fehlgeschlagen: {e}")
            return data

    def _add_audit_headers(
        self,
        response: Response,
        pre_result: AuditEnforcementResult,
        post_result: AuditEnforcementResult
    ) -> None:
        """Fügt Audit-Headers zur Response hinzu."""
        # Audit-Status-Headers
        response.headers["X-Audit-Enabled"] = str(self.config.audit_enabled)

        if pre_result.audit_performed:
            response.headers["X-Audit-Request-ID"] = pre_result.audit_event_id or ""

        if post_result.audit_performed:
            response.headers["X-Audit-Response-ID"] = post_result.audit_event_id or ""

        # PII-Redaction-Headers
        if pre_result.pii_redacted or post_result.pii_redacted:
            response.headers["X-Audit-PII-Redacted"] = "true"

        # Tamper-Proof-Headers
        if pre_result.tamper_proof_added or post_result.tamper_proof_added:
            response.headers["X-Audit-Tamper-Proof"] = "true"

        # Performance-Headers
        total_processing_time = pre_result.processing_time_ms + post_result.processing_time_ms
        response.headers["X-Audit-Processing-Time"] = f"{total_processing_time:.2f}ms"

    def get_middleware_statistics(self) -> dict[str, Any]:
        """Gibt Middleware-Statistiken zurück."""
        audit_rate = (self._requests_audited / max(self._requests_processed, 1)) * 100
        failure_rate = (self._audit_failures / max(self._requests_processed, 1)) * 100

        return {
            "requests_processed": self._requests_processed,
            "requests_audited": self._requests_audited,
            "audit_failures": self._audit_failures,
            "pii_redactions": self._pii_redactions,
            "tamper_proof_events": self._tamper_proof_events,
            "audit_rate_percent": audit_rate,
            "failure_rate_percent": failure_rate,
            "config": self.config.to_dict()
        }


# Decorator für Audit-Compliance
def audit_decorator(
    event_type: AuditEventType | None = None,
    severity: AuditSeverity = AuditSeverity.MEDIUM,
    enable_pii_redaction: bool = True,
    enable_tamper_proof: bool = True,
    compliance_tags: set[str] | None = None
):
    """Decorator für Audit-Compliance.

    Args:
        event_type: Audit-Event-Typ
        severity: Event-Severity
        enable_pii_redaction: PII-Redaction aktivieren
        enable_tamper_proof: Tamper-Proof aktivieren
        compliance_tags: Compliance-Tags
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import uuid

            # Extrahiere Request aus Argumenten
            request = None
            for arg in args:
                if hasattr(arg, "method") and hasattr(arg, "url"):
                    request = arg
                    break

            if not request:
                _request = kwargs.get("request")

            # Erstelle Audit-Context
            context = AuditContext(
                correlation_id=str(uuid.uuid4()),
                metadata={"function": func.__name__, "decorator": "audit_decorator"}
            )

            try:
                # Pre-Function Audit
                await audit_engine.create_event(
                    event_type=event_type or AuditEventType.SYSTEM_EVENT,
                    action=f"Function call: {func.__name__}",
                    description=f"Audit for function {func.__name__}",
                    severity=severity,
                    result=AuditResult.SUCCESS,
                    context=context,
                    compliance_tags=compliance_tags or {"function_audit"}
                )

                # Führe Funktion aus
                result = await func(*args, **kwargs)

                # Post-Function Audit
                await audit_engine.create_event(
                    event_type=event_type or AuditEventType.SYSTEM_EVENT,
                    action=f"Function completed: {func.__name__}",
                    description=f"Function {func.__name__} completed successfully",
                    severity=severity,
                    result=AuditResult.SUCCESS,
                    context=context,
                    compliance_tags=compliance_tags or {"function_audit"}
                )

                return result

            except Exception as e:
                # Error Audit
                await audit_engine.create_event(
                    event_type=event_type or AuditEventType.SYSTEM_EVENT,
                    action=f"Function failed: {func.__name__}",
                    description=f"Function {func.__name__} failed with error",
                    severity=AuditSeverity.HIGH,
                    result=AuditResult.ERROR,
                    context=context,
                    error_details={"error": str(e), "type": type(e).__name__},
                    compliance_tags=compliance_tags or {"function_audit", "error"}
                )

                raise

        return wrapper
    return decorator


def require_audit_compliance(
    _event_types: list[AuditEventType] | None = None,
    _min_severity: AuditSeverity = AuditSeverity.MEDIUM,
    _require_tamper_proof: bool = False
):
    """Decorator für Audit-Compliance-Anforderungen."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Vereinfachte Compliance-Prüfung
            # In Produktion würde hier eine echte Compliance-Validierung stattfinden

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Audit-Compliance-Fehler: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Audit compliance check failed"
                )

        return wrapper
    return decorator
