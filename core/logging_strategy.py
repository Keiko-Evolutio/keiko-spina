"""Konsolidierte Logging-Strategien für Keiko.

Dieses Modul konsolidiert alle duplizierte Logging-Setup-Logik aus:
- backend/kei_agents/logging_utils.py::StructuredLogger
- backend/app/common/logger_utils.py::get_module_logger
- backend/kei_logging/__init__.py::setup_enhanced_logging

Eliminiert Code-Duplikation und stellt einheitliche Logging-API bereit.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from kei_logging import ClickableLoggingFormatter
from kei_logging import get_logger as _get_logger
from kei_logging.pii_redaction import PIIRedactionFilter


@dataclass
class LogContext:
    """Konsolidierter Logging-Kontext für alle Module."""
    operation: str
    component: str
    agent_id: str | None = None
    user_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] | None = None


def _ensure_handler(logger: logging.Logger, level: int = logging.INFO) -> None:
    """Stellt einen konsistenten Handler mit Formatter/Filter sicher."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ClickableLoggingFormatter())
    handler.addFilter(PIIRedactionFilter())
    logger.addHandler(handler)
    logger.propagate = False


class StructuredLogger:
    """Konsolidierter strukturierter Logger für alle Keiko-Module.

    Konsolidiert StructuredLogger aus kei_agents/logging_utils.py
    mit verbesserter Funktionalität und einheitlicher API.

    Methoden spiegeln Standard-Log-Level wider. Alle Nachrichten sind deutsch,
    Identifier bleiben englisch. PII wird durch den globalen Filter reduziert.
    """

    def __init__(self, name: str) -> None:
        self._logger = _get_logger(name)
        _ensure_handler(self._logger)
        self._context: LogContext | None = None

    def set_context(self, context: LogContext) -> None:
        """Setzt Logging-Kontext."""
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
        """Erstellt strukturierte Log-Daten."""
        data = {"message": message}

        # Kontext hinzufügen
        active_context = context or self._context
        if active_context:
            data.update({
                "operation": active_context.operation,
                "component": active_context.component,
                "agent_id": active_context.agent_id,
                "user_id": active_context.user_id,
                "correlation_id": active_context.correlation_id,
            })
            if active_context.metadata:
                data["metadata"] = active_context.metadata

        # Extra-Daten hinzufügen
        if extra_data:
            data.update(extra_data)

        return data

    def _log(self, level: int, message: str, *, fields: dict[str, Any] | None = None) -> None:
        try:
            payload = self._build_log_data(message, fields)
            self._logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            self._logger.log(level, message)

    def debug(self, message: str, *, fields: dict[str, Any] | None = None) -> None:
        self._log(logging.DEBUG, message, fields=fields)

    def info(
        self,
        message: str,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None,
        *,
        fields: dict[str, Any] | None = None
    ) -> None:
        """Info-Logging mit erweiterten Optionen."""
        combined_fields = {**(extra_data or {}), **(fields or {})}
        data = self._build_log_data(message, combined_fields, context)
        self._logger.info(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    def warning(self, message: str, *, fields: dict[str, Any] | None = None) -> None:
        self._log(logging.WARNING, message, fields=fields)

    def error(
        self,
        message: str,
        error: Exception | None = None,
        extra_data: dict[str, Any] | None = None,
        context: LogContext | None = None,
        *,
        fields: dict[str, Any] | None = None
    ) -> None:
        """Error-Logging mit Exception-Details."""
        combined_fields = {**(extra_data or {}), **(fields or {})}

        if error:
            combined_fields.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })

        data = self._build_log_data(message, combined_fields, context)
        self._logger.error(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    def critical(self, message: str, *, fields: dict[str, Any] | None = None) -> None:
        self._log(logging.CRITICAL, message, fields=fields)


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
            component=self._logger.name,
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


# ============================================================================
# CONVENIENCE-FUNKTIONEN (Konsolidiert aus verschiedenen Modulen)
# ============================================================================

def create_logger(component: str) -> StructuredLogger:
    """Erstellt Structured Logger für Komponente.

    Ersetzt create_logger() aus kei_agents/logging_utils.py
    """
    return StructuredLogger(component)


def get_module_logger(name: str) -> StructuredLogger:
    """Erstellt Module-Logger.

    Ersetzt get_module_logger() aus app/common/logger_utils.py
    """
    return StructuredLogger(name)


def get_error_logger(name: str) -> StructuredLogger:
    """Spezialisierter Fehler-Logger mit ERROR-Level."""
    lg = StructuredLogger(name)
    lg._logger.setLevel(logging.ERROR)
    return lg


def get_audit_logger() -> StructuredLogger:
    """Audit-Logger mit separatem Kanalnamen ``kei.audit``."""
    lg = StructuredLogger("kei.audit")
    lg._logger.setLevel(logging.INFO)
    return lg


def get_performance_logger() -> StructuredLogger:
    """Performance-Logger mit Kanal ``kei.perf``."""
    lg = StructuredLogger("kei.perf")
    lg._logger.setLevel(logging.INFO)
    return lg


def get_security_logger() -> StructuredLogger:
    """Security-Logger mit Kanal ``kei.sec``."""
    lg = StructuredLogger("kei.sec")
    lg._logger.setLevel(logging.WARNING)
    return lg


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Klassen
    "LogContext",
    "StructuredLogger",
    # Convenience-Funktionen
    "create_logger",
    "get_audit_logger",
    "get_error_logger",
    "get_module_logger",
    "get_performance_logger",
    "get_security_logger",
]



