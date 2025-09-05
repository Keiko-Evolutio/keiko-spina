# backend/kei_logging/log_links.py
"""Log-Links-System für Keiko Personal Assistant

Implementiert automatische Log-Link-Generierung mit eindeutigen Log-IDs (UUIDs)
für jeden Log-Eintrag und Integration in Error-Messages und Exception-Handling.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .clickable_logging_formatter import get_logger

logger = get_logger(__name__)


class LogLinkFormat(str, Enum):
    """Formate für Log-Links."""
    STANDARD = "[LOG-ID: {log_id}]"
    COMPACT = "[{log_id}]"
    VERBOSE = "[LOG-ID: {log_id} | {timestamp}]"
    URL = "log://{log_id}"
    MARKDOWN = "[Log-ID: {log_id}](log://{log_id})"


@dataclass
class LogEntry:
    """Repräsentiert einen Log-Eintrag mit eindeutiger ID."""
    log_id: str
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    correlation_id: str | None = None
    trace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "extra_fields": self.extra_fields
        }


@dataclass
class LogLinkConfig:
    """Konfiguration für Log-Links-System."""
    # Format-Konfiguration
    link_format: LogLinkFormat = LogLinkFormat.STANDARD
    include_timestamp: bool = False
    include_level: bool = False

    # Storage-Konfiguration
    max_entries: int = 10000
    retention_hours: int = 24

    # Integration-Konfiguration
    auto_inject_in_errors: bool = True
    auto_inject_in_exceptions: bool = True
    auto_inject_in_responses: bool = True

    # Correlation-Konfiguration
    link_with_tracing: bool = True
    link_with_observability: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "link_format": self.link_format.value,
            "include_timestamp": self.include_timestamp,
            "include_level": self.include_level,
            "max_entries": self.max_entries,
            "retention_hours": self.retention_hours,
            "auto_inject_in_errors": self.auto_inject_in_errors,
            "auto_inject_in_exceptions": self.auto_inject_in_exceptions,
            "auto_inject_in_responses": self.auto_inject_in_responses,
            "link_with_tracing": self.link_with_tracing,
            "link_with_observability": self.link_with_observability
        }


class LogLinkRegistry:
    """Thread-safe Registry für Log-Einträge mit eindeutigen IDs."""

    def __init__(self, config: LogLinkConfig | None = None):
        """Initialisiert Log-Link-Registry.

        Args:
            config: Log-Link-Konfiguration
        """
        self.config = config or LogLinkConfig()
        self._entries: dict[str, LogEntry] = {}
        self._lock = threading.RLock()
        self._cleanup_counter = 0

        # Statistiken
        self._entries_created = 0
        self._entries_retrieved = 0
        self._cleanup_operations = 0

    def create_log_entry(
        self,
        level: str,
        logger_name: str,
        message: str,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        extra_fields: dict[str, Any] | None = None
    ) -> LogEntry:
        """Erstellt neuen Log-Eintrag mit eindeutiger ID.

        Args:
            level: Log-Level
            logger_name: Name des Loggers
            message: Log-Message
            correlation_id: Correlation-ID
            trace_id: Trace-ID
            user_id: User-ID
            agent_id: Agent-ID
            session_id: Session-ID
            extra_fields: Zusätzliche Felder

        Returns:
            Log-Eintrag mit eindeutiger ID
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC)

        entry = LogEntry(
            log_id=log_id,
            timestamp=timestamp,
            level=level,
            logger_name=logger_name,
            message=message,
            correlation_id=correlation_id,
            trace_id=trace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            extra_fields=extra_fields or {}
        )

        with self._lock:
            self._entries[log_id] = entry
            self._entries_created += 1

            # Periodisches Cleanup
            self._cleanup_counter += 1
            if self._cleanup_counter % 100 == 0:
                self._cleanup_old_entries()

        return entry

    def get_log_entry(self, log_id: str) -> LogEntry | None:
        """Holt Log-Eintrag anhand der ID.

        Args:
            log_id: Log-ID

        Returns:
            Log-Eintrag oder None
        """
        with self._lock:
            entry = self._entries.get(log_id)
            if entry:
                self._entries_retrieved += 1
            return entry

    def generate_log_link(
        self,
        log_id: str,
        format_override: LogLinkFormat | None = None
    ) -> str:
        """Generiert Log-Link für gegebene Log-ID.

        Args:
            log_id: Log-ID
            format_override: Optionales Format-Override

        Returns:
            Formatierter Log-Link
        """
        link_format = format_override or self.config.link_format

        # Hole Entry für zusätzliche Informationen
        entry = self.get_log_entry(log_id)

        if link_format == LogLinkFormat.STANDARD:
            return f"[LOG-ID: {log_id}]"
        if link_format == LogLinkFormat.COMPACT:
            return f"[{log_id}]"
        if link_format == LogLinkFormat.VERBOSE:
            timestamp = entry.timestamp.strftime("%H:%M:%S") if entry else "unknown"
            return f"[LOG-ID: {log_id} | {timestamp}]"
        if link_format == LogLinkFormat.URL:
            return f"log://{log_id}"
        if link_format == LogLinkFormat.MARKDOWN:
            return f"[Log-ID: {log_id}](log://{log_id})"
        return f"[LOG-ID: {log_id}]"

    def _cleanup_old_entries(self) -> None:
        """Bereinigt alte Log-Einträge."""
        cutoff_time = datetime.now(UTC).timestamp() - (self.config.retention_hours * 3600)

        with self._lock:
            # Entferne alte Einträge
            old_entries = [
                log_id for log_id, entry in self._entries.items()
                if entry.timestamp.timestamp() < cutoff_time
            ]

            for log_id in old_entries:
                del self._entries[log_id]

            # Begrenze Anzahl Einträge
            if len(self._entries) > self.config.max_entries:
                # Entferne älteste Einträge
                sorted_entries = sorted(
                    self._entries.items(),
                    key=lambda x: x[1].timestamp
                )

                entries_to_remove = len(self._entries) - self.config.max_entries
                for log_id, _ in sorted_entries[:entries_to_remove]:
                    del self._entries[log_id]

            if old_entries or len(self._entries) > self.config.max_entries:
                self._cleanup_operations += 1
                logger.debug(f"Log-Link-Cleanup: {len(old_entries)} alte Einträge entfernt")

    def search_entries(
        self,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        level: str | None = None,
        limit: int = 100
    ) -> list[LogEntry]:
        """Sucht Log-Einträge nach Kriterien.

        Args:
            correlation_id: Correlation-ID
            trace_id: Trace-ID
            user_id: User-ID
            agent_id: Agent-ID
            level: Log-Level
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste von Log-Einträgen
        """
        with self._lock:
            results = []

            for entry in self._entries.values():
                # Filter anwenden
                if correlation_id and entry.correlation_id != correlation_id:
                    continue
                if trace_id and entry.trace_id != trace_id:
                    continue
                if user_id and entry.user_id != user_id:
                    continue
                if agent_id and entry.agent_id != agent_id:
                    continue
                if level and entry.level != level:
                    continue

                results.append(entry)

                if len(results) >= limit:
                    break

            # Sortiere nach Timestamp (neueste zuerst)
            results.sort(key=lambda x: x.timestamp, reverse=True)

            return results

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Registry-Statistiken zurück.

        Returns:
            Statistiken-Dictionary
        """
        with self._lock:
            return {
                "config": self.config.to_dict(),
                "total_entries": len(self._entries),
                "entries_created": self._entries_created,
                "entries_retrieved": self._entries_retrieved,
                "cleanup_operations": self._cleanup_operations,
                "oldest_entry": min(
                    (entry.timestamp for entry in self._entries.values()),
                    default=None
                ),
                "newest_entry": max(
                    (entry.timestamp for entry in self._entries.values()),
                    default=None
                )
            }


class LogLinkFilter(logging.Filter):
    """Logging-Filter der automatisch Log-Links zu Log-Records hinzufügt."""

    def __init__(self, registry: LogLinkRegistry):
        """Initialisiert Log-Link-Filter.

        Args:
            registry: Log-Link-Registry
        """
        super().__init__()
        self.registry = registry

    def filter(self, record: logging.LogRecord) -> bool:
        """Fügt Log-Link zu Log-Record hinzu.

        Args:
            record: Log-Record

        Returns:
            True (Record wird nicht gefiltert)
        """
        try:
            # Extrahiere Kontext-Informationen aus Record
            correlation_id = getattr(record, "correlation_id", None)
            trace_id = getattr(record, "trace_id", None)
            user_id = getattr(record, "user_id", None)
            agent_id = getattr(record, "agent_id", None)
            session_id = getattr(record, "session_id", None)

            # Sammle Extra-Felder
            extra_fields = {}
            for attr in ["payload", "headers", "context", "tags", "fields"]:
                if hasattr(record, attr):
                    extra_fields[attr] = getattr(record, attr)

            # Erstelle Log-Entry
            entry = self.registry.create_log_entry(
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                correlation_id=correlation_id,
                trace_id=trace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                extra_fields=extra_fields
            )

            # Füge Log-ID und Link zu Record hinzu
            record.log_id = entry.log_id
            record.log_link = self.registry.generate_log_link(entry.log_id)

            # Integration mit Observability
            if self.registry.config.link_with_observability:
                try:
                    from observability import add_span_attributes

                    # Füge Log-ID zu aktuellem Span hinzu
                    add_span_attributes({
                        "log.id": entry.log_id,
                        "log.level": record.levelname,
                        "log.logger": record.name
                    })

                except ImportError:
                    add_span_attributes = None
                    # Observability nicht verfügbar

        except Exception as e:
            # Filter sollte nie Logging blockieren
            logger.warning(f"Log-Link-Filter-Fehler: {e}")

        return True


# Globale Log-Link-Registry
_global_registry: LogLinkRegistry | None = None
_registry_lock = threading.Lock()


def get_log_link_registry() -> LogLinkRegistry:
    """Holt oder erstellt globale Log-Link-Registry.

    Returns:
        Globale Log-Link-Registry
    """
    global _global_registry

    with _registry_lock:
        if _global_registry is None:
            _global_registry = LogLinkRegistry()

        return _global_registry


def setup_log_links(config: LogLinkConfig | None = None) -> LogLinkRegistry:
    """Setzt Log-Links-System auf.

    Args:
        config: Log-Link-Konfiguration

    Returns:
        Konfigurierte Log-Link-Registry
    """
    global _global_registry

    with _registry_lock:
        _global_registry = LogLinkRegistry(config)

        # Füge Filter zu Root-Logger hinzu
        root_logger = logging.getLogger()
        log_link_filter = LogLinkFilter(_global_registry)

        # Entferne existierende Log-Link-Filter
        for handler in root_logger.handlers:
            handler.filters = [
                f for f in handler.filters
                if not isinstance(f, LogLinkFilter)
            ]
            handler.addFilter(log_link_filter)

        logger.info("Log-Links-System erfolgreich konfiguriert")

        return _global_registry


def create_log_link(
    level: str,
    logger_name: str,
    message: str,
    **context
) -> str:
    """Erstellt Log-Link für gegebene Parameter.

    Args:
        level: Log-Level
        logger_name: Logger-Name
        message: Log-Message
        **context: Zusätzlicher Kontext

    Returns:
        Log-Link
    """
    registry = get_log_link_registry()

    entry = registry.create_log_entry(
        level=level,
        logger_name=logger_name,
        message=message,
        correlation_id=context.get("correlation_id"),
        trace_id=context.get("trace_id"),
        user_id=context.get("user_id"),
        agent_id=context.get("agent_id"),
        session_id=context.get("session_id"),
        extra_fields=context
    )

    return registry.generate_log_link(entry.log_id)


def get_log_entry_by_id(log_id: str) -> LogEntry | None:
    """Holt Log-Eintrag anhand der ID.

    Args:
        log_id: Log-ID

    Returns:
        Log-Eintrag oder None
    """
    registry = get_log_link_registry()
    return registry.get_log_entry(log_id)


def search_log_entries(**criteria) -> list[LogEntry]:
    """Sucht Log-Einträge nach Kriterien.

    Args:
        **criteria: Such-Kriterien

    Returns:
        Liste von Log-Einträgen
    """
    registry = get_log_link_registry()
    return registry.search_entries(**criteria)
