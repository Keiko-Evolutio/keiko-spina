# backend/kei_logging/enhanced_error_handling.py
"""Erweiterte Error-Handling-Funktionalitäten mit automatischer Log-Link-Integration.

Implementiert automatische Log-Link-Injection in Exception-Klassen, Error-Responses
und System-Exceptions für verbesserte Debugging-Erfahrung.
"""

from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .clickable_logging_formatter import get_logger
from .log_links import create_log_link

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class ErrorContext:
    """Kontext-Informationen für Fehler."""
    error_id: str
    log_link: str
    timestamp: datetime
    correlation_id: str | None = None
    trace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    component: str | None = None
    operation: str | None = None
    additional_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "error_id": self.error_id,
            "log_link": self.log_link,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "component": self.component,
            "operation": self.operation,
            "additional_context": self.additional_context
        }


class LogLinkedError(Exception):
    """Basis-Exception-Klasse mit automatischer Log-Link-Integration."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        component: str | None = None,
        operation: str | None = None,
        additional_context: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        """Initialisiert LogLinkedError mit automatischer Log-Link-Generierung.

        Args:
            message: Fehler-Message
            error_code: Optionaler Error-Code
            correlation_id: Correlation-ID
            trace_id: Trace-ID
            user_id: User-ID
            agent_id: Agent-ID
            session_id: Session-ID
            component: Komponente wo Fehler auftrat
            operation: Operation die fehlschlug
            additional_context: Zusätzlicher Kontext
            cause: Ursprüngliche Exception
        """
        self.error_code = error_code
        self.cause = cause

        # Erstelle Error-Context
        self.error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            log_link="",  # Wird unten gesetzt
            timestamp=datetime.now(UTC),
            correlation_id=correlation_id,
            trace_id=trace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            component=component,
            operation=operation,
            additional_context=additional_context or {}
        )

        # Erstelle Log-Link
        try:
            self.error_context.log_link = create_log_link(
                level="ERROR",
                logger_name=self.__class__.__module__,
                message=message,
                correlation_id=correlation_id,
                trace_id=trace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                error_id=self.error_context.error_id,
                error_code=error_code,
                component=component,
                operation=operation,
                exception_type=self.__class__.__name__,
                cause=str(cause) if cause else None
            )
        except Exception as e:
            logger.warning(f"Log-Link-Erstellung fehlgeschlagen: {e}")
            self.error_context.log_link = f"[ERROR-ID: {self.error_context.error_id}]"

        # Erweitere Message um Log-Link
        enhanced_message = f"{message} {self.error_context.log_link}"

        super().__init__(enhanced_message)

        # Logge Fehler
        self._log_error(message)

    def _log_error(self, original_message: str) -> None:
        """Loggt Fehler mit vollständigem Kontext."""
        try:
            logger.error(
                f"Exception aufgetreten: {original_message}",
                extra={
                    "error_id": self.error_context.error_id,
                    "error_code": self.error_code,
                    "exception_type": self.__class__.__name__,
                    "correlation_id": self.error_context.correlation_id,
                    "trace_id": self.error_context.trace_id,
                    "user_id": self.error_context.user_id,
                    "agent_id": self.error_context.agent_id,
                    "session_id": self.error_context.session_id,
                    "component": self.error_context.component,
                    "operation": self.error_context.operation,
                    "additional_context": self.error_context.additional_context,
                    "cause": str(self.cause) if self.cause else None,
                    "stack_trace": traceback.format_exc()
                }
            )
        except Exception:
            # Logging sollte nie weitere Exceptions verursachen
            pass

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu Dictionary für API-Responses."""
        return {
            "error": {
                "message": str(self).replace(self.error_context.log_link, "").strip(),
                "error_code": self.error_code,
                "error_id": self.error_context.error_id,
                "log_link": self.error_context.log_link,
                "timestamp": self.error_context.timestamp.isoformat(),
                "type": self.__class__.__name__,
                "context": self.error_context.to_dict()
            }
        }


class ValidationError(LogLinkedError):
    """Validierungs-Fehler mit Log-Link."""

    def __init__(self, message: str, validation_field: str | None = None, value: Any | None = None, **kwargs):
        self.field = validation_field
        self.value = value

        # Erweitere Kontext
        additional_context = kwargs.get("additional_context", {})
        additional_context.update({
            "validation_field": validation_field,
            "validation_value": str(value) if value is not None else None
        })
        kwargs["additional_context"] = additional_context
        kwargs["error_code"] = kwargs.get("error_code", "VALIDATION_ERROR")

        super().__init__(message, **kwargs)


class AuthenticationError(LogLinkedError):
    """Authentifizierungs-Fehler mit Log-Link."""

    def __init__(self, message: str, auth_method: str | None = None, **kwargs):
        self.auth_method = auth_method

        # Erweitere Kontext
        additional_context = kwargs.get("additional_context", {})
        additional_context.update({
            "auth_method": auth_method
        })
        kwargs["additional_context"] = additional_context
        kwargs["error_code"] = kwargs.get("error_code", "AUTHENTICATION_ERROR")

        super().__init__(message, **kwargs)


class AuthorizationError(LogLinkedError):
    """Autorisierungs-Fehler mit Log-Link."""

    def __init__(self, message: str, resource: str | None = None, action: str | None = None, **kwargs):
        self.resource = resource
        self.action = action

        # Erweitere Kontext
        additional_context = kwargs.get("additional_context", {})
        additional_context.update({
            "resource": resource,
            "action": action
        })
        kwargs["additional_context"] = additional_context
        kwargs["error_code"] = kwargs.get("error_code", "AUTHORIZATION_ERROR")

        super().__init__(message, **kwargs)


class BusinessLogicError(LogLinkedError):
    """Business-Logic-Fehler mit Log-Link."""

    def __init__(self, message: str, business_rule: str | None = None, **kwargs):
        self.business_rule = business_rule

        # Erweitere Kontext
        additional_context = kwargs.get("additional_context", {})
        additional_context.update({
            "business_rule": business_rule
        })
        kwargs["additional_context"] = additional_context
        kwargs["error_code"] = kwargs.get("error_code", "BUSINESS_LOGIC_ERROR")

        super().__init__(message, **kwargs)


class ExternalServiceError(LogLinkedError):
    """External-Service-Fehler mit Log-Link."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        **kwargs
    ):
        self.service_name = service_name
        self.status_code = status_code
        self.response_body = response_body

        # Erweitere Kontext
        additional_context = kwargs.get("additional_context", {})
        additional_context.update({
            "service_name": service_name,
            "status_code": status_code,
            "response_body": response_body[:1000] if response_body else None  # Begrenze Größe
        })
        kwargs["additional_context"] = additional_context
        kwargs["error_code"] = kwargs.get("error_code", "EXTERNAL_SERVICE_ERROR")

        super().__init__(message, **kwargs)


class KeiSystemError(LogLinkedError):
    """System-Fehler mit Log-Link."""

    def __init__(self, message: str, system_component: str | None = None, **kwargs):
        self.system_component = system_component

        # Erweitere Kontext
        additional_context = kwargs.get("additional_context", {})
        additional_context.update({
            "system_component": system_component
        })
        kwargs["additional_context"] = additional_context
        kwargs["error_code"] = kwargs.get("error_code", "SYSTEM_ERROR")

        super().__init__(message, **kwargs)


def enhance_exception_with_log_link(
    exc: Exception,
    correlation_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    component: str | None = None,
    operation: str | None = None,
    additional_context: dict[str, Any] | None = None
) -> Exception:
    """Erweitert bestehende Exception um Log-Link.

    Args:
        exc: Ursprüngliche Exception
        correlation_id: Correlation-ID
        trace_id: Trace-ID
        user_id: User-ID
        agent_id: Agent-ID
        session_id: Session-ID
        component: Komponente
        operation: Operation
        additional_context: Zusätzlicher Kontext

    Returns:
        Exception mit Log-Link
    """
    # Wenn bereits LogLinkedError, nicht nochmal wrappen
    if isinstance(exc, LogLinkedError):
        return exc

    # Erstelle Log-Link
    try:
        log_link = create_log_link(
            level="ERROR",
            logger_name=exc.__class__.__module__,
            message=str(exc),
            correlation_id=correlation_id,
            trace_id=trace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            exception_type=exc.__class__.__name__,
            component=component,
            operation=operation,
            original_exception=str(exc)
        )

        # Erweitere Exception-Message
        original_message = str(exc)
        enhanced_message = f"{original_message} {log_link}"

        # Setze neue Message (falls möglich)
        if hasattr(exc, "args") and exc.args:
            exc.args = (enhanced_message, *exc.args[1:])

        # Füge Log-Link als Attribut hinzu
        exc.log_link = log_link

    except Exception as e:
        logger.warning(f"Exception-Enhancement fehlgeschlagen: {e}")

    return exc


def create_error_response(
    message: str,
    error_code: str | None = None,
    status_code: int = 500,
    correlation_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    component: str | None = None,
    operation: str | None = None,
    additional_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Erstellt standardisierte Error-Response mit Log-Link.

    Args:
        message: Fehler-Message
        error_code: Error-Code
        status_code: HTTP-Status-Code
        correlation_id: Correlation-ID
        trace_id: Trace-ID
        user_id: User-ID
        agent_id: Agent-ID
        session_id: Session-ID
        component: Komponente
        operation: Operation
        additional_context: Zusätzlicher Kontext

    Returns:
        Standardisierte Error-Response
    """
    error_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC)

    # Erstelle Log-Link
    try:
        log_link = create_log_link(
            level="ERROR",
            logger_name="error_response",
            message=message,
            correlation_id=correlation_id,
            trace_id=trace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            error_id=error_id,
            error_code=error_code,
            status_code=status_code,
            component=component,
            operation=operation
        )
    except Exception as e:
        logger.warning(f"Log-Link-Erstellung für Error-Response fehlgeschlagen: {e}")
        log_link = f"[ERROR-ID: {error_id}]"

    return {
        "error": {
            "message": message,
            "error_code": error_code,
            "error_id": error_id,
            "log_link": log_link,
            "timestamp": timestamp.isoformat(),
            "status_code": status_code,
            "correlation_id": correlation_id,
            "trace_id": trace_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "component": component,
            "operation": operation,
            "additional_context": additional_context or {}
        }
    }


def log_and_raise(
    exception_class: type[LogLinkedError],
    message: str,
    **kwargs
) -> None:
    """Loggt und wirft Exception mit Log-Link.

    Args:
        exception_class: Exception-Klasse
        message: Fehler-Message
        **kwargs: Zusätzliche Parameter für Exception
    """
    exc = exception_class(message, **kwargs)
    raise exc


# Exception-Handler-Decorator
def with_log_links(
    correlation_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    component: str | None = None,
    operation: str | None = None
):
    """Decorator der automatisch Log-Links zu Exceptions hinzufügt.

    Args:
        correlation_id: Correlation-ID
        trace_id: Trace-ID
        user_id: User-ID
        agent_id: Agent-ID
        session_id: Session-ID
        component: Komponente
        operation: Operation
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                enhanced_exc = enhance_exception_with_log_link(
                    e,
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    component=component or getattr(func, "__module__", "unknown"),
                    operation=operation or func.__name__
                )
                raise enhanced_exc

        return wrapper
    return decorator
