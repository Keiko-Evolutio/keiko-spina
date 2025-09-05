# backend/agents/event_handler/utils.py
"""Utility-Funktionen für Event-Processing.

Wiederverwendbare Komponenten für Event-Handler-Implementierungen
zur Konsolidierung redundanter Patterns.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from kei_logging import get_logger

from .constants import (
    DEFAULT_DEDUPLICATION_CACHE_SIZE,
    DEFAULT_DEDUPLICATION_TTL_SECONDS,
    JSON_ENSURE_ASCII,
    MAX_EVENT_PAYLOAD_SIZE_BYTES,
    MAX_TOOL_OUTPUT_SIZE_BYTES,
    EventHandlerErrorCode,
)

logger = get_logger(__name__)

T = TypeVar("T")

# =============================================================================
# Event-Deduplication
# =============================================================================

@dataclass
class DeduplicationEntry:
    """Eintrag im Deduplication-Cache."""

    state_signature: tuple[str, str]
    timestamp: float
    access_count: int = 0


class EventDeduplicator:
    """Effiziente Event-Deduplication mit TTL und LRU-Cleanup."""

    def __init__(
        self,
        max_size: int = DEFAULT_DEDUPLICATION_CACHE_SIZE,
        ttl_seconds: int = DEFAULT_DEDUPLICATION_TTL_SECONDS
    ) -> None:
        """Initialisiert Event-Deduplicator.

        Args:
            max_size: Maximale Cache-Größe
            ttl_seconds: Time-to-Live für Cache-Einträge
        """
        self._cache: dict[str, DeduplicationEntry] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 Minuten

    def is_duplicate(self, event_id: str, state_signature: tuple[str, str]) -> bool:
        """Prüft ob Event ein Duplikat ist.

        Args:
            event_id: Eindeutige Event-ID
            state_signature: Status-Signatur für Vergleich

        Returns:
            True wenn Event bereits verarbeitet wurde
        """
        self._maybe_cleanup()

        entry = self._cache.get(event_id)
        if entry is None:
            # Neues Event - Cache-Eintrag erstellen
            self._cache[event_id] = DeduplicationEntry(
                state_signature=state_signature,
                timestamp=time.time(),
                access_count=1
            )
            self._enforce_size_limit()
            return False

        # Event existiert - Signatur vergleichen
        entry.access_count += 1
        if entry.state_signature == state_signature:
            return True  # Duplikat gefunden

        # Status hat sich geändert - Eintrag aktualisieren
        entry.state_signature = state_signature
        entry.timestamp = time.time()
        return False

    def _maybe_cleanup(self) -> None:
        """Führt periodische Cache-Bereinigung durch."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> None:
        """Entfernt abgelaufene Cache-Einträge."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now - entry.timestamp > self._ttl_seconds
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Deduplication-Cache: {len(expired_keys)} abgelaufene Einträge entfernt")

    def _enforce_size_limit(self) -> None:
        """Erzwingt Cache-Größenlimit durch LRU-Eviction."""
        if len(self._cache) <= self._max_size:
            return

        # Sortiere nach Zugriffshäufigkeit und Alter (LRU)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )

        # Entferne älteste/selten genutzte Einträge
        to_remove = len(self._cache) - self._max_size
        for key, _ in sorted_items[:to_remove]:
            del self._cache[key]

        logger.debug(f"Deduplication-Cache: {to_remove} LRU-Einträge entfernt")

    def get_stats(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        return {
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "last_cleanup": self._last_cleanup,
        }


# =============================================================================
# Event-Dispatcher
# =============================================================================

class EventDispatcher(Generic[T]):
    """Generischer Event-Dispatcher mit Deduplication und Error-Handling."""

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], Any],
        enable_deduplication: bool = True
    ) -> None:
        """Initialisiert Event-Dispatcher.

        Args:
            callback: Callback-Funktion für Event-Versendung
            enable_deduplication: Ob Deduplication aktiviert werden soll
        """
        self._callback = callback
        self._deduplicator = EventDeduplicator() if enable_deduplication else None

    async def dispatch(
        self,
        event_id: str,
        status: str,
        object_type: str,
        extra: Mapping[str, Any] | None = None
    ) -> bool:
        """Versendet Event mit optionaler Deduplication.

        Args:
            event_id: Eindeutige Event-ID
            status: Event-Status
            object_type: Objekt-Typ des Events
            extra: Zusätzliche Event-Daten

        Returns:
            True wenn Event versendet wurde, False bei Duplikat
        """
        # Deduplication-Prüfung
        if self._deduplicator:
            state_sig = (status, object_type)
            if self._deduplicator.is_duplicate(event_id, state_sig):
                return False

        # Payload erstellen
        payload = EventDispatcher._build_payload(event_id, status, extra)

        # Payload-Größe validieren
        if not EventDispatcher._validate_payload_size(payload):
            logger.warning(f"Event-Payload zu groß: {event_id}")
            return False

        # Event versenden
        try:
            await self._callback(payload)
            return True
        except Exception as e:
            logger.error(
                f"Event-Dispatch fehlgeschlagen für {event_id}: {e}",
                extra={
                    "event_id": event_id,
                    "status": status,
                    "object_type": object_type,
                    "error_type": type(e).__name__
                }
            )
            return False

    @staticmethod
    def _build_payload(
        event_id: str,
        status: str,
        extra: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Erstellt Event-Payload.

        Args:
            event_id: Eindeutige Event-ID
            status: Event-Status
            extra: Zusätzliche Event-Daten

        Returns:
            Event-Payload als Dictionary
        """
        payload: dict[str, Any] = {"id": event_id, "status": status}
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _validate_payload_size(payload: dict[str, Any]) -> bool:
        """Validiert Payload-Größe.

        Args:
            payload: Zu validierendes Payload

        Returns:
            True wenn Payload-Größe akzeptabel ist
        """
        try:
            payload_json = json.dumps(payload, ensure_ascii=JSON_ENSURE_ASCII)
            payload_size = len(payload_json.encode("utf-8"))
            return payload_size <= MAX_EVENT_PAYLOAD_SIZE_BYTES
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Gibt Dispatcher-Statistiken zurück."""
        stats = {"deduplication_enabled": self._deduplicator is not None}
        if self._deduplicator:
            stats.update(self._deduplicator.get_stats())
        return stats


# =============================================================================
# Tool-Execution-Utilities
# =============================================================================

@dataclass
class ToolExecutionResult:
    """Ergebnis einer Tool-Ausführung."""

    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    tool_name: str = ""


class ToolExecutor:
    """Utility für sichere Tool-Ausführung mit Timeout und Error-Handling."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        """Initialisiert Tool-Executor.

        Args:
            timeout_seconds: Timeout für Tool-Ausführung
        """
        self._timeout_seconds = timeout_seconds

    async def execute_with_timeout(
        self,
        func: Callable[..., Any],
        args: dict[str, Any],
        tool_name: str = "unknown"
    ) -> ToolExecutionResult:
        """Führt Tool mit Timeout aus.

        Args:
            func: Auszuführende Funktion
            args: Funktions-Argumente
            tool_name: Name des Tools für Logging

        Returns:
            Tool-Execution-Result
        """
        start_time = time.time()

        try:
            # Timeout-Wrapper für Tool-Ausführung
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**args),
                    timeout=self._timeout_seconds
                )
            else:
                # Synchrone Funktion in Thread-Pool ausführen
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**args)),
                    timeout=self._timeout_seconds
                )

            execution_time = (time.time() - start_time) * 1000
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                tool_name=tool_name
            )

        except TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Tool-Timeout nach {self._timeout_seconds}s"
            logger.warning(
                f"Tool '{tool_name}' Timeout: {execution_time:.1f}ms",
                extra={
                    "tool_name": tool_name,
                    "timeout_seconds": self._timeout_seconds,
                    "execution_time_ms": execution_time
                }
            )

            return ToolExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time,
                tool_name=tool_name
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(
                f"Tool '{tool_name}' Fehler: {error_msg}",
                extra={
                    "tool_name": tool_name,
                    "error_type": type(e).__name__,
                    "execution_time_ms": execution_time
                }
            )

            return ToolExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time,
                tool_name=tool_name
            )

    @staticmethod
    def serialize_result(result: Any) -> str:
        """Serialisiert Tool-Ergebnis zu String.

        Args:
            result: Zu serialisierendes Ergebnis

        Returns:
            Serialisierter String
        """
        try:
            if isinstance(result, (dict, list)):
                serialized = json.dumps(result, ensure_ascii=JSON_ENSURE_ASCII)
            else:
                serialized = str(result)

            # Größe validieren
            if len(serialized.encode("utf-8")) > MAX_TOOL_OUTPUT_SIZE_BYTES:
                return f"Tool-Output zu groß (>{MAX_TOOL_OUTPUT_SIZE_BYTES} Bytes)"

            return serialized

        except Exception as e:
            return f"Serialisierung fehlgeschlagen: {e}"


# =============================================================================
# Error-Handling-Utilities
# =============================================================================

@dataclass
class ErrorContext:
    """Kontext-Informationen für Error-Handling."""

    operation: str
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = None
    user_id: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)


class EventErrorHandler:
    """Erweiterte Error-Handling-Patterns für Event-Processing."""

    def __init__(self, component_name: str = "event_handler") -> None:
        """Initialisiert Error-Handler.

        Args:
            component_name: Name der Komponente für Logging
        """
        self._component_name = component_name
        self._error_counts: dict[str, int] = {}

    def handle_tool_error(
        self,
        tool_name: str,
        error: Exception,
        context: ErrorContext | None = None
    ) -> str:
        """Behandelt Tool-Execution-Fehler mit erweiterten Kontext.

        Args:
            tool_name: Name des fehlgeschlagenen Tools
            error: Aufgetretener Fehler
            context: Error-Kontext

        Returns:
            Formatierte Error-Message
        """
        error_code = EventHandlerErrorCode.TOOL_EXECUTION_FAILED
        self._increment_error_count(f"tool_error_{tool_name}")

        base_msg = f"[{error_code}] Tool '{tool_name}' fehlgeschlagen: {error}"

        if context:
            context_info = {
                "operation": context.operation,
                "component": context.component,
                "timestamp": context.timestamp.isoformat(),
                "correlation_id": context.correlation_id,
                "user_id": context.user_id,
                **context.additional_data
            }
            context_str = json.dumps(context_info, ensure_ascii=JSON_ENSURE_ASCII)
            return f"{base_msg} (Kontext: {context_str})"

        return base_msg

    def handle_dispatch_error(
        self,
        event_id: str,
        error: Exception,
        retry_count: int = 0,
        context: ErrorContext | None = None
    ) -> str:
        """Behandelt Event-Dispatch-Fehler mit Retry-Tracking.

        Args:
            event_id: ID des fehlgeschlagenen Events
            error: Aufgetretener Fehler
            retry_count: Anzahl der Retry-Versuche
            context: Error-Kontext

        Returns:
            Formatierte Error-Message
        """
        error_code = EventHandlerErrorCode.EVENT_DISPATCH_FAILED
        self._increment_error_count("dispatch_error")

        base_msg = f"[{error_code}] Event-Dispatch fehlgeschlagen für {event_id}: {error} (Retry: {retry_count})"

        if context and context.correlation_id:
            return f"{base_msg} (Correlation-ID: {context.correlation_id})"

        return base_msg

    def handle_timeout_error(
        self,
        operation: str,
        timeout_seconds: int,
        context: ErrorContext | None = None
    ) -> str:
        """Behandelt Timeout-Fehler.

        Args:
            operation: Name der Operation die timeout hatte
            timeout_seconds: Timeout-Wert in Sekunden
            context: Error-Kontext (optional, aktuell nicht verwendet)

        Returns:
            Formatierte Error-Message
        """
        error_code = EventHandlerErrorCode.TOOL_TIMEOUT
        self._increment_error_count("timeout_error")

        return f"[{error_code}] {operation} Timeout nach {timeout_seconds}s"

    def handle_validation_error(
        self,
        field_name: str,
        value: Any,
        expected: str,
        context: ErrorContext | None = None
    ) -> str:
        """Behandelt Validierungs-Fehler.

        Args:
            field_name: Name des validierten Feldes
            value: Ungültiger Wert
            expected: Erwartetes Format/Typ
            context: Error-Kontext

        Returns:
            Formatierte Error-Message
        """
        self._increment_error_count("validation_error")

        return f"Validierung fehlgeschlagen für '{field_name}': '{value}' (erwartet: {expected})"

    def _increment_error_count(self, error_type: str) -> None:
        """Erhöht Error-Counter für Monitoring."""
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    def get_error_stats(self) -> dict[str, Any]:
        """Gibt Error-Statistiken zurück."""
        return {
            "component": self._component_name,
            "error_counts": self._error_counts.copy(),
            "total_errors": sum(self._error_counts.values())
        }

    def reset_stats(self) -> None:
        """Setzt Error-Statistiken zurück."""
        self._error_counts.clear()


# =============================================================================
# Logging-Utilities
# =============================================================================

class StructuredLogger:
    """Strukturiertes Logging für Event-Handler."""

    def __init__(self, component_name: str) -> None:
        """Initialisiert Structured Logger.

        Args:
            component_name: Name der Komponente
        """
        self._component = component_name
        self._logger = get_logger(f"event_handler.{component_name}")

    def log_tool_execution(
        self,
        tool_name: str,
        execution_time_ms: float,
        success: bool,
        context: dict[str, Any] | None = None
    ) -> None:
        """Loggt Tool-Ausführung mit strukturierten Daten.

        Args:
            tool_name: Name des Tools
            execution_time_ms: Ausführungszeit in Millisekunden
            success: Ob Ausführung erfolgreich war
            context: Zusätzlicher Kontext
        """
        log_data = {
            "component": self._component,
            "operation": "tool_execution",
            "tool_name": tool_name,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "timestamp": datetime.now(UTC).isoformat()
        }

        if context:
            log_data.update(context)

        if success:
            self._logger.info(f"Tool '{tool_name}' erfolgreich ausgeführt ({execution_time_ms:.1f}ms)", extra=log_data)
        else:
            self._logger.error(f"Tool '{tool_name}' fehlgeschlagen ({execution_time_ms:.1f}ms)", extra=log_data)

    def log_event_dispatch(
        self,
        event_id: str,
        event_type: str,
        success: bool,
        deduplication_hit: bool = False,
        context: dict[str, Any] | None = None
    ) -> None:
        """Loggt Event-Dispatch mit strukturierten Daten.

        Args:
            event_id: Event-ID
            event_type: Event-Typ
            success: Ob Dispatch erfolgreich war
            deduplication_hit: Ob Event dedupliziert wurde
            context: Zusätzlicher Kontext
        """
        log_data = {
            "component": self._component,
            "operation": "event_dispatch",
            "event_id": event_id,
            "event_type": event_type,
            "success": success,
            "deduplication_hit": deduplication_hit,
            "timestamp": datetime.now(UTC).isoformat()
        }

        if context:
            log_data.update(context)

        if deduplication_hit:
            self._logger.debug(
                f"Event {event_id} dedupliziert", extra=log_data
            )
        elif success:
            self._logger.debug(
                f"Event {event_id} erfolgreich versendet", extra=log_data
            )
        else:
            self._logger.error(
                f"Event {event_id} Dispatch fehlgeschlagen", extra=log_data
            )


# =============================================================================
# Export-Liste
# =============================================================================

__all__ = [
    # Event-Deduplication
    "EventDeduplicator",
    "DeduplicationEntry",

    # Event-Dispatch
    "EventDispatcher",

    # Tool-Execution
    "ToolExecutor",
    "ToolExecutionResult",

    # Error-Handling
    "EventErrorHandler",
    "ErrorContext",

    # Logging
    "StructuredLogger",
]
