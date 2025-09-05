# backend/agents/factory/error_handlers.py
"""Standardisierte Error-Handling-Patterns für das Factory-Modul.

Konsolidiert alle Exception-Behandlungen zu einheitlichen, wiederverwendbaren
Patterns mit strukturiertem Logging und Retry-Mechanismen.
"""
from __future__ import annotations

import asyncio
import functools
import traceback
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, TypeVar

from kei_logging import get_logger

from .constants import (
    DEFAULT_RETRY_DELAY,
    EXPONENTIAL_BACKOFF_MULTIPLIER,
    MAX_RETRY_ATTEMPTS,
    ErrorSeverity,
)

# Type Variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)


# =============================================================================
# Custom Exception Classes
# =============================================================================

class FactoryError(Exception):
    """Basis-Exception für alle Factory-Fehler."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.error_message = message  # Renamed to avoid logging conflict
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Exception zu Dictionary für Logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_error": str(self.original_error) if self.original_error else None,
        }


class AgentCreationError(FactoryError):
    """Fehler bei der Agent-Erstellung."""


class MCPClientError(FactoryError):
    """Fehler bei MCP-Client-Operationen."""


class SessionError(FactoryError):
    """Fehler bei Session-Management."""


class FactoryInitializationError(FactoryError):
    """Fehler bei Factory-Initialisierung."""


class ValidationError(FactoryError):
    """Fehler bei Input-Validierung."""


# =============================================================================
# Error Handler Classes
# =============================================================================

class ErrorHandler:
    """Zentrale Error-Handler-Klasse für standardisierte Exception-Behandlung."""

    def __init__(self, logger_name: str | None = None) -> None:
        self.logger = get_logger(logger_name or __name__)
        self._error_counts: dict[str, int] = {}
        self._last_errors: dict[str, datetime] = {}

    def handle_error(
        self,
        error: Exception,
        context: str,
        *,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suppress: bool = False,
        additional_details: dict[str, Any] | None = None
    ) -> FactoryError | None:
        """Behandelt einen Fehler mit standardisiertem Logging und Tracking.

        Args:
            error: Die ursprüngliche Exception
            context: Kontext-Information wo der Fehler auftrat
            severity: Schweregrad des Fehlers
            suppress: Ob der Fehler unterdrückt werden soll
            additional_details: Zusätzliche Details für Logging

        Returns:
            FactoryError oder None wenn suppress=True
        """
        error_key = f"{context}:{type(error).__name__}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        self._last_errors[error_key] = datetime.now(UTC)

        # Details für Logging zusammenstellen
        details = {
            "context": context,
            "error_type": type(error).__name__,
            "error_count": self._error_counts[error_key],
            "traceback": traceback.format_exc(),
            **(additional_details or {})
        }

        # Factory-Error erstellen
        if isinstance(error, FactoryError):
            factory_error = error
            factory_error.details.update(details)
        else:
            factory_error = FactoryError(
                message=str(error),
                severity=severity,
                details=details,
                original_error=error
            )

        # Logging basierend auf Severity
        log_method = self._get_log_method(severity)
        log_method(
            f"Fehler in {context}: {factory_error.error_message}",
            extra=factory_error.to_dict()
        )

        if suppress:
            return None

        return factory_error

    def _get_log_method(self, severity: ErrorSeverity) -> Callable:
        """Gibt die passende Log-Methode für die Severity zurück."""
        severity_mapping = {
            ErrorSeverity.LOW: self.logger.debug,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical,
        }
        return severity_mapping.get(severity, self.logger.warning)

    def get_error_stats(self) -> dict[str, Any]:
        """Gibt Statistiken über behandelte Fehler zurück."""
        return {
            "total_errors": sum(self._error_counts.values()),
            "error_types": dict(self._error_counts),
            "last_errors": {
                key: time.isoformat()
                for key, time in self._last_errors.items()
            }
        }


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_on_error(
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff_multiplier: float = EXPONENTIAL_BACKOFF_MULTIPLIER,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    context: str | None = None
) -> Callable[[F], F]:
    """Decorator für automatische Retry-Logik bei Fehlern.

    Args:
        max_attempts: Maximale Anzahl Versuche
        delay: Initiale Verzögerung zwischen Versuchen
        backoff_multiplier: Multiplikator für exponential backoff
        exceptions: Tuple von Exception-Typen die Retry auslösen
        context: Kontext-Information für Logging
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                error_handler = ErrorHandler()
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts - 1:
                            # Letzter Versuch - Fehler nicht unterdrücken
                            error_handler.handle_error(
                                e,
                                context or func.__name__,
                                severity=ErrorSeverity.HIGH,
                                additional_details={
                                    "attempt": attempt + 1,
                                    "max_attempts": max_attempts,
                                    "final_attempt": True
                                }
                            )
                            raise

                        # Retry-Versuch loggen
                        error_handler.handle_error(
                            e,
                            context or func.__name__,
                            severity=ErrorSeverity.LOW,
                            suppress=True,
                            additional_details={
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "retry_delay": current_delay
                            }
                        )

                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_multiplier
                return None

            return async_wrapper
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        error_handler.handle_error(
                            e,
                            context or func.__name__,
                            severity=ErrorSeverity.HIGH,
                            additional_details={
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "final_attempt": True
                            }
                        )
                        raise

                    error_handler.handle_error(
                        e,
                        context or func.__name__,
                        severity=ErrorSeverity.LOW,
                        suppress=True,
                        additional_details={
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "retry_delay": current_delay
                        }
                    )

                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier
            return None

        return sync_wrapper

    return decorator


# =============================================================================
# Context Manager für Error Handling
# =============================================================================

@asynccontextmanager
async def error_context(
    context_name: str,
    *,
    suppress_errors: bool = False,
    _default_return: Any = None,
    error_handler: ErrorHandler | None = None
) -> AsyncGenerator[ErrorHandler, None]:
    """Context Manager für strukturiertes Error Handling.

    Args:
        context_name: Name des Kontexts für Logging
        suppress_errors: Ob Fehler unterdrückt werden sollen
        _default_return: Standard-Rückgabewert bei unterdrückten Fehlern
        error_handler: Optionaler Error Handler (wird erstellt falls None)
    """
    handler = error_handler or ErrorHandler()

    try:
        yield handler
    except Exception as e:
        factory_error = handler.handle_error(
            e,
            context_name,
            suppress=suppress_errors
        )

        if not suppress_errors and factory_error:
            raise factory_error


# =============================================================================
# Spezifische Error Handler für Factory-Komponenten
# =============================================================================

class AgentFactoryErrorHandler(ErrorHandler):
    """Spezialisierter Error Handler für Agent Factory Operationen."""

    async def handle_agent_creation_error(
        self,
        error: Exception,
        agent_id: str,
        framework: str
    ) -> AgentCreationError:
        """Behandelt Fehler bei Agent-Erstellung."""
        result = self.handle_error(
            error,
            "agent_creation",
            severity=ErrorSeverity.HIGH,
            additional_details={
                "agent_id": agent_id,
                "framework": framework
            }
        )
        # Konvertiere FactoryError zu AgentCreationError
        if isinstance(result, AgentCreationError):
            return result
        return AgentCreationError(
            message=result.error_message,
            error_code=result.error_code,
            severity=result.severity,
            details=result.details,
            original_error=result.original_error
        )

    async def handle_mcp_client_error(
        self,
        error: Exception,
        agent_id: str,
        server_ids: list[str] | None = None
    ) -> MCPClientError:
        """Behandelt Fehler bei MCP Client Operationen."""
        result = self.handle_error(
            error,
            "mcp_client_operation",
            severity=ErrorSeverity.MEDIUM,
            additional_details={
                "agent_id": agent_id,
                "server_ids": server_ids or []
            }
        )
        # Konvertiere FactoryError zu MCPClientError
        if isinstance(result, MCPClientError):
            return result
        return MCPClientError(
            message=result.error_message,
            error_code=result.error_code,
            severity=result.severity,
            details=result.details,
            original_error=result.original_error
        )


# =============================================================================
# Export für einfachen Import
# =============================================================================

__all__ = [
    "AgentCreationError",
    "AgentFactoryErrorHandler",
    # Handler Classes
    "ErrorHandler",
    # Exception Classes
    "FactoryError",
    "FactoryInitializationError",
    "MCPClientError",
    "SessionError",
    "ValidationError",
    # Context Managers
    "error_context",
    # Decorators
    "retry_on_error",
]
