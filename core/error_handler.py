"""Globaler Error Handler und Fehlerkontext.

Dieses Modul stellt einen einheitlichen Behandlungsfluss für Ausnahmen bereit,
inklusive Kontextanreicherung, strukturierter API-Antworten, Logging mit PII-
Redaction und optionaler OpenTelemetry-Integration für Tracing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse, Response

from kei_logging import get_logger

from .constants import (
    ERROR_CODE_TO_HTTP_STATUS,
    ERROR_CODE_TO_RETRY_TIMEOUT,
    RETRYABLE_ERROR_CODES,
    SEVERITY_TO_LOG_LEVEL,
    OTelAttributes,
)
from .exceptions import KeikoErrorPayload, KeikoException

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import Request

logger = get_logger(__name__)


@dataclass(frozen=True)
class ErrorContext:
    """Request-bezogener Kontext für Fehlerbehandlung.

    Attributes:
        trace_id: Korrrelations-ID/Trace-ID für verteilte Traces
        route: Pfad/Operation, die den Fehler ausgelöst hat
        method: HTTP-Methode
        tenant: Optionaler Tenant/Scope
        user: Optionaler Subjekt-/Benutzername
        extra: Zusätzliche, PII-bereinigte Felder
    """

    trace_id: str | None
    route: str | None
    method: str | None
    tenant: str | None = None
    user: str | None = None
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RecoveryStrategy:
    """Strategiehinweise für automatische/halbautomatische Wiederherstellung.

    Attributes:
        retryable: Ob Wiederholversuch sinnvoll ist
        retry_after_seconds: Empfohlene Wartezeit vor erneutem Versuch
        fallback: Beschreibung eines Fallbacks
    """

    retryable: bool = False
    retry_after_seconds: int | None = None
    fallback: str | None = None


class ExceptionClassifier:
    """Klassifiziert Exceptions in HTTP-Status, Payload und Recovery-Strategie.

    Trennt die Classification-Logik von der Response-Building-Logik
    für bessere Single Responsibility Principle-Einhaltung.
    """

    @staticmethod
    def classify(exc: BaseException) -> tuple[int, KeikoErrorPayload, RecoveryStrategy]:
        """Klassifiziert eine Exception in HTTP-Status, Payload und Recovery.

        Args:
            exc: Aufgetretene Ausnahme

        Returns:
            Tupel aus (http_status, payload, recovery_strategy)
        """
        if isinstance(exc, KeikoException):
            return ExceptionClassifier._classify_keiko_exception(exc)

        return ExceptionClassifier._classify_unknown_exception(exc)

    @staticmethod
    def _classify_keiko_exception(exc: KeikoException) -> tuple[int, KeikoErrorPayload, RecoveryStrategy]:
        """Klassifiziert eine KeikoException."""
        error_code = exc.error_code
        payload = exc.to_payload()

        # HTTP-Status aus Mapping
        status = ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)

        # Recovery-Strategie bestimmen
        retryable = error_code in RETRYABLE_ERROR_CODES
        retry_after = ERROR_CODE_TO_RETRY_TIMEOUT.get(error_code) if retryable else None
        recovery = RecoveryStrategy(retryable=retryable, retry_after_seconds=retry_after)

        return status, payload, recovery

    @staticmethod
    def _classify_unknown_exception(exc: BaseException) -> tuple[int, KeikoErrorPayload, RecoveryStrategy]:
        """Klassifiziert eine unbekannte Exception."""
        payload = KeikoErrorPayload(
            error_code="INTERNAL_ERROR",
            message="Interner Serverfehler",
            severity="CRITICAL",
            details={"type": type(exc).__name__},
        )
        recovery = RecoveryStrategy(retryable=False)
        return 500, payload, recovery


class ResponseBuilder:
    """Erstellt strukturierte HTTP-Antworten für Fehler.

    Trennt die Response-Building-Logik von der Classification-Logik.
    """

    @staticmethod
    def build_response(
        *,
        status: int,
        payload: KeikoErrorPayload,
        ctx: ErrorContext | None,
        recovery: RecoveryStrategy
    ) -> JSONResponse:
        """Erstellt eine strukturierte HTTP-Antwort mit Kontextfeldern."""
        body: dict[str, Any] = {
            "error": {
                "code": payload.error_code,
                "message": payload.message,
                "severity": payload.severity,
                "details": dict(payload.details or {}),
            },
            "recovery": asdict(recovery),
        }

        if ctx:
            body["context"] = {
                "trace_id": ctx.trace_id,
                "route": ctx.route,
                "method": ctx.method,
                "tenant": ctx.tenant,
                "user": ctx.user,
            }

        return JSONResponse(status_code=status, content=body)


class ErrorLogger:
    """Behandelt strukturiertes Logging von Fehlern.

    Trennt die Logging-Logik von der Response-Building-Logik.
    """

    @staticmethod
    def log_error(payload: KeikoErrorPayload, ctx: ErrorContext | None) -> None:
        """Loggt einen Fehler mit strukturierten Feldern."""
        log_level_name = SEVERITY_TO_LOG_LEVEL.get(payload.severity.upper(), "ERROR")
        log_method = getattr(logger, log_level_name.lower(), logger.error)

        log_method(
            "Fehler aufgetreten",
            extra={
                "payload": {
                    "code": payload.error_code,
                    "message": payload.message,
                    "severity": payload.severity,
                    "details": dict(payload.details or {}),
                },
                "context": {
                    "trace_id": ctx.trace_id if ctx else None,
                    "route": ctx.route if ctx else None,
                    "method": ctx.method if ctx else None,
                },
            },
        )


class TracingHandler:
    """Behandelt OpenTelemetry-Tracing für Fehler.

    Trennt die Tracing-Logik von der Response-Building-Logik.
    """

    @staticmethod
    def add_error_attributes(payload: KeikoErrorPayload, status: int) -> None:
        """Fügt Error-Attribute zum aktuellen Span hinzu."""
        try:
            from opentelemetry import trace  # type: ignore

            span = trace.get_current_span()
            if span and getattr(span, "is_recording", lambda: False)():
                span.set_attribute(OTelAttributes.ERROR_CODE, payload.error_code)
                span.set_attribute(OTelAttributes.ERROR_SEVERITY, payload.severity)
                span.set_attribute(OTelAttributes.HTTP_STATUS_CODE, status)
        except Exception:
            # OpenTelemetry-Fehler ignorieren (best effort)
            pass


class GlobalErrorHandler:
    """Zentrale Fehlerbehandlung mit strukturierter Antwort und Tracing.

    Diese Klasse orchestriert die verschiedenen Komponenten für Error-Handling:
    - ExceptionClassifier für Exception-Klassifizierung
    - ResponseBuilder für Response-Erstellung
    - ErrorLogger für strukturiertes Logging
    - TracingHandler für OpenTelemetry-Integration
    """

    def __init__(self, include_details: bool = False) -> None:
        """Initialisiert den GlobalErrorHandler.

        Args:
            include_details: Ob Details in nicht-Prod-Umgebungen erweitert werden sollen
        """
        self.include_details = include_details
        self.classifier = ExceptionClassifier()
        self.response_builder = ResponseBuilder()
        self.error_logger = ErrorLogger()
        self.tracing_handler = TracingHandler()

    def classify(self, exc: BaseException) -> tuple[int, KeikoErrorPayload, RecoveryStrategy]:
        """Klassifiziert eine Exception (Delegiert an ExceptionClassifier)."""
        return self.classifier.classify(exc)

    def build_response(
        self,
        *,
        status: int,
        payload: KeikoErrorPayload,
        ctx: ErrorContext | None,
        recovery: RecoveryStrategy
    ) -> JSONResponse:
        """Erstellt eine Response (Delegiert an ResponseBuilder)."""
        return self.response_builder.build_response(
            status=status,
            payload=payload,
            ctx=ctx,
            recovery=recovery
        )

    async def handle_request_exception(self, request: Request, exc: BaseException) -> Response:
        """Transformiert eine Ausnahme in eine HTTP-Response inkl. Logging/Tracing."""
        # Context extrahieren
        ctx = self._extract_error_context(request)

        # Exception klassifizieren
        status, payload, recovery = self.classify(exc)

        # Details erweitern falls konfiguriert
        if self.include_details:
            payload = self._enhance_payload_with_details(payload, exc)

        # Logging
        self.error_logger.log_error(payload, ctx)

        # Tracing (best effort - Fehler werden ignoriert)
        try:
            self.tracing_handler.add_error_attributes(payload, status)
        except Exception:
            # Tracing-Fehler ignorieren
            pass

        # Response erstellen
        return self.build_response(status=status, payload=payload, ctx=ctx, recovery=recovery)

    def _extract_error_context(self, request: Request) -> ErrorContext:
        """Extrahiert Error-Context aus Request."""
        trace_id = getattr(request.state, "trace_id", None)
        route = str(getattr(request.scope, "path", None))
        method = str(getattr(request, "method", None))

        auth = getattr(request.state, "auth", object())
        tenant = getattr(auth, "tenant", None)
        user = getattr(auth, "subject", None)

        return ErrorContext(
            trace_id=trace_id,
            route=route,
            method=method,
            tenant=tenant,
            user=user,
            extra=None,
        )

    def _enhance_payload_with_details(
        self,
        payload: KeikoErrorPayload,
        exc: BaseException
    ) -> KeikoErrorPayload:
        """Erweitert Payload mit zusätzlichen Details für Nicht-Prod-Umgebungen."""
        try:
            details = dict(payload.details or {})
            details["exception_type"] = type(exc).__name__
            details["str"] = str(exc)

            return KeikoErrorPayload(
                error_code=payload.error_code,
                message=payload.message,
                severity=payload.severity,
                details=details,
            )
        except Exception:
            # Fallback auf ursprüngliches Payload bei Fehlern
            return payload


__all__ = [
    # Data Classes
    "ErrorContext",
    "ErrorLogger",
    # Core Components
    "ExceptionClassifier",
    # Main Handler
    "GlobalErrorHandler",
    "RecoveryStrategy",
    "ResponseBuilder",
    "TracingHandler",
]
