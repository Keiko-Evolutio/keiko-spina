# backend/kei_agents/logging_utils.py
"""Standardisierte Logging-Utilities für Keiko Personal Assistant

Enterprise-Grade Logging mit:
- Strukturierte Log-Nachrichten
- Kontext-basiertes Logging
- Performance-Metriken
- Error-Tracking
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from kei_logging import get_logger

from .exceptions import ErrorSeverity, KEIAgentError


@dataclass
class LogContext:
    """Kontext für strukturiertes Logging."""

    operation: str = ""
    component: str = ""
    agent_id: str | None = None
    user_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """Strukturierter Logger für Keiko Personal Assistant"""

    def __init__(self, component: str):
        """Initialisiert Structured Logger.

        Args:
            component: Name der Komponente
        """
        self.component = component
        self.logger = get_logger(component)
        self._context: LogContext | None = None

    def set_context(self, context: LogContext) -> None:
        """Setzt Logging-Kontext.

        Args:
            context: Logging-Kontext
        """
        self._context = context

    def clear_context(self) -> None:
        """Löscht Logging-Kontext."""
        self._context = None

    def _build_log_data(
        self,
        message: str,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None
    ) -> dict[str, Any]:
        """Erstellt strukturierte Log-Daten.

        Args:
            message: Log-Nachricht
            extra_data: Zusätzliche Daten
            context: Spezifischer Kontext

        Returns:
            Strukturierte Log-Daten
        """
        log_context = context or self._context

        data = {
            "message": message,
            "component": self.component,
            "timestamp": time.time(),
        }

        if log_context:
            if log_context.operation:
                data["operation"] = log_context.operation
            if log_context.agent_id:
                data["agent_id"] = log_context.agent_id
            if log_context.user_id:
                data["user_id"] = log_context.user_id
            if log_context.correlation_id:
                data["correlation_id"] = log_context.correlation_id
            if log_context.trace_id:
                data["trace_id"] = log_context.trace_id
            if log_context.session_id:
                data["session_id"] = log_context.session_id
            if log_context.metadata:
                data["metadata"] = log_context.metadata

        if extra_data:
            data.update(extra_data)

        return data

    def debug(
        self,
        message: str,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None
    ) -> None:
        """Debug-Logging."""
        data = self._build_log_data(message, extra_data, context)
        self.logger.debug(data)

    def info(
        self,
        message: str,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None
    ) -> None:
        """Info-Logging."""
        data = self._build_log_data(message, extra_data, context)
        self.logger.info(data)

    def warning(
        self,
        message: str,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None
    ) -> None:
        """Warning-Logging."""
        data = self._build_log_data(message, extra_data, context)
        self.logger.warning(data)

    def error(
        self,
        message: str,
        error: Exception | None = None,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None
    ) -> None:
        """Error-Logging."""
        data = self._build_log_data(message, extra_data, context)

        if error:
            data["error_type"] = type(error).__name__
            data["error_message"] = str(error)

            if isinstance(error, KEIAgentError):
                data["error_code"] = error.error_code
                data["error_severity"] = error.severity.value
                data["error_category"] = error.category.value
                data["error_details"] = error.details

        self.logger.error(data)

    def critical(
        self,
        message: str,
        error: Exception | None = None,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None
    ) -> None:
        """Critical-Logging."""
        data = self._build_log_data(message, extra_data, context)

        if error:
            data["error_type"] = type(error).__name__
            data["error_message"] = str(error)

            if isinstance(error, KEIAgentError):
                data["error_code"] = error.error_code
                data["error_severity"] = error.severity.value
                data["error_category"] = error.category.value
                data["error_details"] = error.details

        self.logger.critical(data)

    @contextmanager
    def operation_context(
        self,
        operation: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ):
        """Context Manager für Operation-Logging."""
        old_context = self._context

        new_context = LogContext(
            operation=operation,
            component=self.component,
            agent_id=agent_id,
            user_id=user_id,
            correlation_id=correlation_id or str(uuid.uuid4()),
            metadata=metadata or {}
        )

        self.set_context(new_context)

        start_time = time.time()
        self.info(f"Operation gestartet: {operation}")

        try:
            yield new_context

            duration = time.time() - start_time
            self.info(
                f"Operation erfolgreich abgeschlossen: {operation}",
                extra_data={"duration_seconds": duration}
            )

        except Exception as e:
            duration = time.time() - start_time
            self.error(
                f"Operation fehlgeschlagen: {operation}",
                error=e,
                extra_data={"duration_seconds": duration}
            )
            raise
        finally:
            self.set_context(old_context)


def log_performance(
    logger: StructuredLogger,
    operation: str,
    threshold_seconds: float = 1.0
):
    """Decorator für Performance-Logging.

    Args:
        logger: Structured Logger
        operation: Name der Operation
        threshold_seconds: Schwellwert für Warnung
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                if duration > threshold_seconds:
                    logger.warning(
                        f"Langsame Operation: {operation}",
                        extra_data={
                            "duration_seconds": duration,
                            "threshold_seconds": threshold_seconds
                        }
                    )
                else:
                    logger.debug(
                        f"Operation abgeschlossen: {operation}",
                        extra_data={"duration_seconds": duration}
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation fehlgeschlagen: {operation}",
                    error=e,
                    extra_data={"duration_seconds": duration}
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if duration > threshold_seconds:
                    logger.warning(
                        f"Langsame Operation: {operation}",
                        extra_data={
                            "duration_seconds": duration,
                            "threshold_seconds": threshold_seconds
                        }
                    )
                else:
                    logger.debug(
                        f"Operation abgeschlossen: {operation}",
                        extra_data={"duration_seconds": duration}
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation fehlgeschlagen: {operation}",
                    error=e,
                    extra_data={"duration_seconds": duration}
                )
                raise

        # Prüfen ob Funktion async ist
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper

    return decorator


def log_entry_exit(logger: StructuredLogger, operation: str):
    """Decorator für Entry/Exit-Logging.

    Args:
        logger: Structured Logger
        operation: Name der Operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Entering: {operation}")

            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting: {operation}")
                return result
            except Exception as e:
                logger.debug(f"Exiting with error: {operation} - {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Entering: {operation}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting: {operation}")
                return result
            except Exception as e:
                logger.debug(f"Exiting with error: {operation} - {e}")
                raise

        # Prüfen ob Funktion async ist
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper

    return decorator


# Convenience-Funktionen

def create_logger(component: str) -> StructuredLogger:
    """Erstellt Structured Logger für Komponente."""
    return StructuredLogger(component)


def log_agent_operation(
    logger: StructuredLogger,
    operation: str,
    agent_id: str,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None
):
    """Context Manager für Agent-Operation-Logging."""
    return logger.operation_context(
        operation=operation,
        agent_id=agent_id,
        user_id=user_id,
        metadata=metadata
    )


def log_error_with_context(
    logger: StructuredLogger,
    error: Exception,
    operation: str,
    agent_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None
) -> None:
    """Loggt Fehler mit Kontext.

    Args:
        logger: Structured Logger
        error: Exception
        operation: Operation die fehlgeschlagen ist
        agent_id: Agent-ID
        user_id: Benutzer-ID
        metadata: Zusätzliche Metadaten
    """
    context = LogContext(
        operation=operation,
        agent_id=agent_id,
        user_id=user_id,
        metadata=metadata or {}
    )

    severity = "error"
    if isinstance(error, KEIAgentError):
        if error.severity == ErrorSeverity.CRITICAL:
            severity = "critical"

    if severity == "critical":
        logger.critical(
            f"Kritischer Fehler in Operation: {operation}",
            error=error,
            context=context
        )
    else:
        logger.error(
            f"Fehler in Operation: {operation}",
            error=error,
            context=context
        )
