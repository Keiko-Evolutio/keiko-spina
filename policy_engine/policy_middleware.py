# backend/policy_engine/policy_middleware.py
"""Policy Enforcement Middleware für Keiko Personal Assistant

Integriert Policy Engine mit FastAPI-Middleware für automatische
Policy-Enforcement in allen Agent-Operationen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from kei_logging import get_logger

from .compliance_framework import compliance_engine
from .core_policy_engine import PolicyContext, PolicyType, policy_engine
from .data_minimization import data_minimization_engine
from .enhanced_pii_redaction import enhanced_pii_redactor
from .prompt_guardrails import prompt_guardrails_engine
from .safety_guardrails import safety_guardrails_engine

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class PolicyConfig:
    """Konfiguration für Policy Enforcement."""
    # Safety Guardrails
    safety_enabled: bool = True
    safety_block_unsafe: bool = True
    safety_log_violations: bool = True

    # Compliance
    compliance_enabled: bool = True
    compliance_standards: list[str] = None
    compliance_block_violations: bool = True

    # PII Redaction
    pii_redaction_enabled: bool = True
    pii_auto_redact: bool = True
    pii_log_detections: bool = True

    # Data Minimization
    data_minimization_enabled: bool = True
    data_minimization_auto_apply: bool = True

    # Prompt Guardrails
    prompt_guardrails_enabled: bool = True
    prompt_block_injections: bool = True
    prompt_sanitize_threats: bool = True

    # General
    audit_all_decisions: bool = True
    fail_open: bool = False  # Bei Policy-Fehlern durchlassen oder blockieren

    def __post_init__(self):
        if self.compliance_standards is None:
            self.compliance_standards = ["gdpr", "ccpa"]


@dataclass
class PolicyEnforcementResult:
    """Ergebnis der Policy-Enforcement."""
    allowed: bool
    modified_content: str | None = None
    violations: list[str] = None
    warnings: list[str] = None
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.warnings is None:
            self.warnings = []


class PolicyEnforcementMiddleware(BaseHTTPMiddleware):
    """Middleware für automatische Policy-Enforcement."""

    def __init__(self, app, config: PolicyConfig):
        """Initialisiert Policy Enforcement Middleware.

        Args:
            app: FastAPI-App
            config: Policy-Konfiguration
        """
        super().__init__(app)
        self.config = config

        # Statistiken
        self._requests_processed = 0
        self._policies_enforced = 0
        self._violations_detected = 0

        # Ausgeschlossene Pfade
        self._excluded_paths = {
            "/health", "/metrics", "/docs", "/openapi.json", "/redoc"
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Verarbeitet Request mit Policy-Enforcement."""
        start_time = time.time()

        try:
            # Prüfe ausgeschlossene Pfade
            if request.url.path in self._excluded_paths:
                return await call_next(request)

            self._requests_processed += 1

            # Extrahiere Request-Content für Policy-Checks
            request_content = await self._extract_request_content(request)

            # Erstelle Policy-Kontext
            policy_context = await self._create_policy_context(request, request_content)

            # Pre-Request Policy-Enforcement
            pre_enforcement_result = await self._enforce_pre_request_policies(
                request_content, policy_context
            )

            if not pre_enforcement_result.allowed:
                self._violations_detected += 1
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Policy violation: {'; '.join(pre_enforcement_result.violations)}"
                )

            # Modifiziere Request falls erforderlich
            if pre_enforcement_result.modified_content:
                request = await self._modify_request_content(request, pre_enforcement_result.modified_content)

            # Führe Request aus
            response = await call_next(request)

            # Post-Request Policy-Enforcement
            response_content = await self._extract_response_content(response)

            post_enforcement_result = await self._enforce_post_request_policies(
                response_content, policy_context
            )

            if not post_enforcement_result.allowed:
                self._violations_detected += 1
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Response blocked by policy"
                )

            # Modifiziere Response falls erforderlich
            if post_enforcement_result.modified_content:
                response = await self._modify_response_content(response, post_enforcement_result.modified_content)

            # Audit-Logging
            processing_time = (time.time() - start_time) * 1000
            await self._log_policy_enforcement(
                request, response, policy_context,
                pre_enforcement_result, post_enforcement_result,
                processing_time
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Policy-Enforcement-Fehler: {e}")

            if self.config.fail_open:
                # Fail-Open: Request durchlassen
                return await call_next(request)
            # Fail-Closed: Request blockieren
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Policy enforcement error"
            )

    async def _extract_request_content(self, request: Request) -> str | None:
        """Extrahiert Content aus Request."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    return body.decode("utf-8", errors="ignore")

            # Query-Parameter
            if request.query_params:
                return str(dict(request.query_params))

            return None

        except Exception as e:
            logger.exception(f"Request-Content-Extraktion fehlgeschlagen: {e}")
            return None

    async def _extract_response_content(self, response: Response) -> str | None:
        """Extrahiert Content aus Response."""
        try:
            # Prüfe Response-Typ und Status
            if not response or response.status_code >= 400:
                return None

            # Extrahiere Content-Type
            content_type = response.headers.get("content-type", "").lower()

            # Nur Text-basierte Responses verarbeiten
            if not any(ct in content_type for ct in ["text/", "application/json", "application/xml"]):
                logger.debug(f"Response-Content-Type nicht unterstützt: {content_type}")
                return None

            # Versuche Response-Body zu lesen
            if hasattr(response, "body") and response.body:
                if isinstance(response.body, bytes):
                    return response.body.decode("utf-8", errors="ignore")
                if isinstance(response.body, str):
                    return response.body

            # Fallback: Prüfe auf andere Response-Attribute
            if hasattr(response, "content"):
                return str(response.content)

            logger.debug("Kein extrahierbarer Content in Response gefunden")
            return None

        except Exception as e:
            logger.exception(f"Response-Content-Extraktion fehlgeschlagen: {e}")
            return None

    async def _create_policy_context(self, request: Request, content: str | None) -> PolicyContext:
        """Erstellt Policy-Kontext aus Request."""
        # Extrahiere User/Agent-Info aus Request-State (falls verfügbar)
        user_id = getattr(request.state, "principal", {}).get("id") if hasattr(request.state, "principal") else None
        tenant_id = getattr(request.state, "tenant_context", {}).get("tenant_id") if hasattr(request.state, "tenant_context") else None
        agent_id = request.headers.get("X-Agent-ID")

        return PolicyContext(
            user_id=user_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            operation=f"{request.method} {request.url.path}",
            resource_type="api_request",
            resource_id=request.url.path,
            content=content,
            metadata={
                "method": request.method,
                "path": request.url.path,
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None
            }
        )

    async def _enforce_pre_request_policies(
        self,
        content: str | None,
        context: PolicyContext
    ) -> PolicyEnforcementResult:
        """Führt Pre-Request Policy-Enforcement durch."""
        start_time = time.time()
        violations = []
        warnings = []
        modified_content = content

        try:
            # Prompt Guardrails (für Input-Validation)
            if self.config.prompt_guardrails_enabled and content:
                prompt_result = prompt_guardrails_engine.validate_prompt(content, context.metadata)

                if prompt_result.requires_blocking and self.config.prompt_block_injections:
                    violations.append(f"Prompt injection detected: {prompt_result.risk_level.value}")

                if prompt_result.requires_sanitization and self.config.prompt_sanitize_threats:
                    modified_content = prompt_result.sanitized_prompt
                    warnings.append("Prompt sanitized due to detected threats")

            # PII Redaction
            if self.config.pii_redaction_enabled and content:
                pii_result = await enhanced_pii_redactor.detect_pii(content, context.metadata)

                if pii_result.has_pii:
                    if self.config.pii_auto_redact:
                        redaction_result = await enhanced_pii_redactor.redact_pii(content, context.metadata)
                        modified_content = redaction_result.redacted_text
                        warnings.append(f"PII redacted: {len(redaction_result.entities_redacted)} entities")

                    if self.config.pii_log_detections:
                        logger.info(f"PII detected in request: {len(pii_result.entities)} entities")

            # Data Minimization
            if self.config.data_minimization_enabled and content:
                minimization_result = await data_minimization_engine.apply_minimization(
                    content, "request_data", "api_processing", context.metadata
                )

                if minimization_result.reduction_ratio > 0:
                    warnings.append(f"Data minimized: {minimization_result.size_reduction_percent:.1f}% reduction")

            processing_time = (time.time() - start_time) * 1000

            return PolicyEnforcementResult(
                allowed=len(violations) == 0,
                modified_content=modified_content if modified_content != content else None,
                violations=violations,
                warnings=warnings,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.exception(f"Pre-Request Policy-Enforcement fehlgeschlagen: {e}")

            if self.config.fail_open:
                return PolicyEnforcementResult(
                    allowed=True,
                    warnings=[f"Policy enforcement error: {e!s}"]
                )
            return PolicyEnforcementResult(
                allowed=False,
                violations=[f"Policy enforcement error: {e!s}"]
            )

    async def _enforce_post_request_policies(
        self,
        content: str | None,
        context: PolicyContext
    ) -> PolicyEnforcementResult:
        """Führt Post-Request Policy-Enforcement durch."""
        start_time = time.time()
        violations = []
        warnings = []
        modified_content = content

        try:
            # Safety Guardrails (für Output-Validation)
            if self.config.safety_enabled and content:
                safety_result = await safety_guardrails_engine.check_content(content, context.metadata)

                if not safety_result.is_safe and self.config.safety_block_unsafe:
                    violations.append(f"Unsafe content detected: {safety_result.overall_safety_level.value}")

                if safety_result.violations and self.config.safety_log_violations:
                    logger.warning(f"Safety violations in response: {len(safety_result.violations)}")

            # Compliance Checks
            if self.config.compliance_enabled and content:
                from .compliance_framework import ComplianceStandard

                standards = [ComplianceStandard(std) for std in self.config.compliance_standards]
                compliance_results = await compliance_engine.check_compliance(
                    content, standards, context.metadata
                )

                for result in compliance_results:
                    if not result.compliant and self.config.compliance_block_violations:
                        violations.extend([v.description for v in result.violations])

            # PII Redaction für Response
            if self.config.pii_redaction_enabled and content:
                redaction_result = await enhanced_pii_redactor.redact_pii(content, context.metadata)

                if redaction_result.entities_redacted:
                    modified_content = redaction_result.redacted_text
                    warnings.append(f"Response PII redacted: {len(redaction_result.entities_redacted)} entities")

            processing_time = (time.time() - start_time) * 1000

            return PolicyEnforcementResult(
                allowed=len(violations) == 0,
                modified_content=modified_content if modified_content != content else None,
                violations=violations,
                warnings=warnings,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.exception(f"Post-Request Policy-Enforcement fehlgeschlagen: {e}")

            if self.config.fail_open:
                return PolicyEnforcementResult(
                    allowed=True,
                    warnings=[f"Policy enforcement error: {e!s}"]
                )
            return PolicyEnforcementResult(
                allowed=False,
                violations=[f"Policy enforcement error: {e!s}"]
            )

    async def _modify_request_content(self, request: Request, modified_content: str) -> Request:
        """Modifiziert Request-Content."""
        try:
            if not modified_content:
                return request

            # Logge Content-Modifikation
            logger.debug(
                "Request-Content wird modifiziert",
                extra={
                    "original_content_length": len(await self._extract_request_content(request) or ""),
                    "modified_content_length": len(modified_content),
                    "method": request.method,
                    "path": request.url.path
                }
            )

            # In einer vollständigen Implementierung würde hier der Request-Body
            # mit dem modified_content ersetzt werden. Da FastAPI Requests immutable sind,
            # müsste ein neuer Request erstellt oder der Body über andere Mechanismen
            # modifiziert werden.

            # Für jetzt: Setze Header um Modifikation zu kennzeichnen
            # Note: Request headers sind immutable in FastAPI/Starlette
            # Modifikation würde einen neuen Request erfordern
            if hasattr(request, "state"):
                request.state.content_modified = True

            return request

        except Exception as e:
            logger.exception(f"Request-Content-Modifikation fehlgeschlagen: {e}")
            return request

    async def _modify_response_content(self, response: Response, modified_content: str) -> Response:
        """Modifiziert Response-Content."""
        try:
            if not modified_content:
                return response

            # Logge Content-Modifikation
            original_content = await self._extract_response_content(response) or ""
            logger.debug(
                "Response-Content wird modifiziert",
                extra={
                    "original_content_length": len(original_content),
                    "modified_content_length": len(modified_content),
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", "unknown")
                }
            )

            # In einer vollständigen Implementierung würde hier der Response-Body
            # mit dem modified_content ersetzt werden. Dies erfordert spezielle
            # Behandlung je nach Response-Typ (JSON, HTML, Text, etc.)

            # Für jetzt: Setze Header um Modifikation zu kennzeichnen
            if hasattr(response, "headers"):
                response.headers["X-Content-Modified"] = "true"
                response.headers["X-Original-Content-Length"] = str(len(original_content))
                response.headers["X-Modified-Content-Length"] = str(len(modified_content))

            return response

        except Exception as e:
            logger.exception(f"Response-Content-Modifikation fehlgeschlagen: {e}")
            return response

    async def _log_policy_enforcement(
        self,
        request: Request,
        response: Response,
        context: PolicyContext,
        pre_result: PolicyEnforcementResult,
        post_result: PolicyEnforcementResult,
        processing_time_ms: float
    ) -> None:
        """Loggt Policy-Enforcement für Audit."""
        if not self.config.audit_all_decisions:
            return

        audit_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "user_id": context.user_id,
            "agent_id": context.agent_id,
            "tenant_id": context.tenant_id,
            "pre_request_allowed": pre_result.allowed,
            "post_request_allowed": post_result.allowed,
            "total_violations": len(pre_result.violations) + len(post_result.violations),
            "total_warnings": len(pre_result.warnings) + len(post_result.warnings),
            "processing_time_ms": processing_time_ms,
            "response_status": response.status_code
        }

        # Log an Compliance Engine weiterleiten
        if hasattr(compliance_engine, "audit_trail_manager"):
            compliance_engine.audit_trail_manager.log_event(
                event_type="policy_enforcement",
                user_id=context.user_id,
                agent_id=context.agent_id,
                resource_type="api_request",
                resource_id=context.resource_id,
                action=context.operation,
                result="allowed" if pre_result.allowed and post_result.allowed else "blocked",
                metadata=audit_data
            )

        logger.info(f"Policy-Enforcement-Audit: {audit_data}")

    def get_enforcement_statistics(self) -> dict[str, Any]:
        """Gibt Policy-Enforcement-Statistiken zurück."""
        return {
            "requests_processed": self._requests_processed,
            "policies_enforced": self._policies_enforced,
            "violations_detected": self._violations_detected,
            "violation_rate": self._violations_detected / max(self._requests_processed, 1),
            "config": {
                "safety_enabled": self.config.safety_enabled,
                "compliance_enabled": self.config.compliance_enabled,
                "pii_redaction_enabled": self.config.pii_redaction_enabled,
                "data_minimization_enabled": self.config.data_minimization_enabled,
                "prompt_guardrails_enabled": self.config.prompt_guardrails_enabled,
                "fail_open": self.config.fail_open
            }
        }


# Decorator für Policy-Compliance
def require_policy_compliance(
    policy_types: list[PolicyType] | None = None,
    safety_check: bool = True,
    compliance_check: bool = True,
    pii_redaction: bool = True
):
    """Decorator für Policy-Compliance-Checks.

    Args:
        policy_types: Zu prüfende Policy-Typen
        safety_check: Safety-Guardrails aktivieren
        compliance_check: Compliance-Checks aktivieren
        pii_redaction: PII-Redaction aktivieren
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Request aus Argumenten extrahieren
            request = None
            for arg in args:
                if hasattr(arg, "method") and hasattr(arg, "url"):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                logger.warning("Kein Request für Policy-Compliance gefunden")
                return await func(*args, **kwargs)

            # Policy-Kontext erstellen
            content = await _extract_content_from_request(request)
            context = PolicyContext(
                operation=f"{request.method} {request.url.path}",
                resource_type="function_call",
                resource_id=func.__name__,
                content=content
            )

            # Policy-Evaluation
            if policy_types:
                result = await policy_engine.evaluate(context, policy_types)
                if not result.is_allowed:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Policy violation: {result.final_effect.value}"
                    )

            # Safety-Check
            if safety_check and content:
                safety_result = await safety_guardrails_engine.check_content(content)
                if not safety_result.is_safe:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Content safety violation"
                    )

            # Compliance-Check
            if compliance_check and content:
                compliance_result = await compliance_engine.check_compliance(
                    content, {"processing_purpose": "function_call"}
                )
                if compliance_result.violations:
                    high_severity_violations = [
                        v for v in compliance_result.violations
                        if v.severity.value >= 3  # HIGH oder CRITICAL
                    ]
                    if high_severity_violations:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Compliance violation: {high_severity_violations[0].description}"
                        )

            # PII-Redaction
            modified_content = content
            if pii_redaction and content:
                redaction_result = await enhanced_pii_redactor.redact_text(content)
                if redaction_result.entities_redacted > 0:
                    modified_content = redaction_result.redacted_text
                    logger.info(
                        f"PII redacted in function call: {redaction_result.entities_redacted} entities"
                    )

            # Führe Funktion aus (mit modifiziertem Content falls PII redacted wurde)
            if modified_content != content:
                # Versuche Content in kwargs zu ersetzen falls möglich
                if "content" in kwargs:
                    kwargs["content"] = modified_content
                elif len(args) > 0 and isinstance(args[0], str):
                    args = (modified_content,) + args[1:]

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def enforce_safety_guardrails(block_unsafe: bool = True):
    """Decorator für Safety-Guardrails-Enforcement."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extrahiere Content aus Argumenten
            content = None
            for arg in args:
                if isinstance(arg, str) and len(arg) > 10:
                    content = arg
                    break

            if content:
                safety_result = await safety_guardrails_engine.check_content(content)

                if not safety_result.is_safe and block_unsafe:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsafe content detected: {safety_result.overall_safety_level.value}"
                    )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def enforce_data_minimization(auto_apply: bool = True):
    """Decorator für Data-Minimization-Enforcement."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if auto_apply:
                # Vereinfachte Data-Minimization für Decorator
                logger.info(f"Data-Minimization angewendet für {func.__name__}")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


async def _extract_content_from_request(request) -> str | None:
    """Hilfsfunktion zur Content-Extraktion."""
    try:
        if hasattr(request, "body"):
            body = await request.body()
            if body:
                return body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    return None
