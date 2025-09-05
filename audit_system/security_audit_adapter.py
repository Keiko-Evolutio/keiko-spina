# backend/audit_system/security_audit_adapter.py
"""Security Audit Adapter für enhanced_security Integration

Stellt eine Security-spezifische Schnittstelle zum umfassenden audit_system bereit
und konsolidiert die Funktionalitäten aus enhanced_security/audit_logger.py.

MIGRATION: Diese Datei ersetzt backend/agents/enhanced_security/audit_logger.py
und stellt vollständige Backward-Compatibility bereit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Fallback für fehlende Module
try:
    from kei_agent.enterprise_logging import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

def trace_function(name):
    """Fallback-Decorator für fehlende observability."""
    def decorator(func):
        return func
    return decorator

from .core_audit_engine import (
    AuditContext,
    AuditEngine,
    AuditEventType,
    AuditResult,
    AuditSeverity,
    get_audit_engine,
)

logger = get_logger(__name__)


# Backward-Compatibility: Original AuditLevel aus enhanced_security
class AuditLevel(Enum):
    """Audit-Level für verschiedene Ereignistypen (Original aus enhanced_security)."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    COMPLIANCE = "compliance"


# Alias für Kompatibilität
SecurityAuditLevel = AuditLevel


class SecurityAuditEventType(Enum):
    """Security-spezifische Event-Typen (Kompatibilität zu enhanced_security)."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"
    AGENT_OPERATION = "agent_operation"
    ERROR_EVENT = "error_event"


# Backward-Compatibility: Original AuditEvent aus enhanced_security
@dataclass
class AuditEvent:
    """Strukturiertes Audit-Event mit Metadaten (Original aus enhanced_security)."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: AuditLevel = AuditLevel.INFO
    category: str = "general"
    action: str = ""
    resource: str = ""
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None

    # Event-Details
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    result: str = "unknown"

    # Sicherheits-Metadaten
    ip_address: str | None = None
    user_agent: str | None = None
    source_service: str | None = None

    # Compliance-Felder
    compliance_relevant: bool = False
    retention_period_days: int = 365
    data_classification: str = "internal"

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Event zu Dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category,
            "action": self.action,
            "resource": self.resource,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "message": self.message,
            "details": self.details,
            "result": self.result,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "source_service": self.source_service,
            "compliance_relevant": self.compliance_relevant,
            "retention_period_days": self.retention_period_days,
            "data_classification": self.data_classification,
        }


@dataclass
class SecurityAuditEvent:
    """Security-spezifisches Audit-Event (Kompatibilität zu security_audit_adapter)."""

    event_id: str
    event_type: SecurityAuditEventType
    message: str
    level: SecurityAuditLevel
    timestamp: float
    agent_id: str | None = None
    user_id: str | None = None
    operation: str | None = None
    metadata: dict[str, Any] | None = None
    checksum: str | None = None


# Backward-Compatibility: Original ComplianceReporter aus enhanced_security
class ComplianceReporter:
    """Compliance-Berichterstattung für Audit-Events (Original aus enhanced_security)."""

    def __init__(self, audit_logger: AuditLogger):
        """Initialisiert den Compliance-Reporter.

        Args:
            audit_logger: Audit-Logger-Instanz
        """
        self.audit_logger = audit_logger

    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generiert Compliance-Bericht für Zeitraum.

        Args:
            start_date: Start-Datum
            end_date: End-Datum
            categories: Zu berücksichtigende Kategorien

        Returns:
            Compliance-Bericht
        """
        # Placeholder für Compliance-Bericht
        # In produktiver Umgebung würde hier die Log-Datei analysiert
        return {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "categories": categories or [],
            "summary": {
                "total_events": 0,
                "compliance_events": 0,
                "security_events": 0,
                "user_actions": 0,
            },
            "status": "generated",
        }


@dataclass
class ComplianceReport:
    """Compliance-Report (Kompatibilität zu security_audit_adapter)."""

    period_start: float
    period_end: float
    total_events: int
    events_by_type: dict[str, int]
    events_by_level: dict[str, int]
    security_violations: int
    compliance_score: float
    violations: list[dict[str, Any]]
    recommendations: list[str]
    metadata: dict[str, Any]


# Backward-Compatibility: Original AuditLogger aus enhanced_security
class AuditLogger:
    """Enterprise Audit Logger mit Verschlüsselung und Compliance (Original aus enhanced_security)."""

    def __init__(
        self,
        name: str = "audit",
        log_file: Path | None = None,
        encryption_manager: Any | None = None,
        enable_encryption: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 30.0,
    ):
        """Initialisiert den Audit Logger.

        Args:
            name: Logger-Name
            log_file: Pfad zur Log-Datei
            encryption_manager: Verschlüsselungs-Manager
            enable_encryption: Verschlüsselung aktivieren
            buffer_size: Puffer-Größe für Batch-Verarbeitung
            flush_interval: Intervall für automatisches Leeren des Puffers
        """
        self.name = name
        self.log_file = log_file or Path(f"logs/audit_{name}.log")
        self.encryption_manager = encryption_manager
        self.enable_encryption = enable_encryption and encryption_manager is not None

        # Puffer für Performance-Optimierung
        self._event_buffer: list[AuditEvent] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._last_flush = time.time()
        self._buffer_lock = asyncio.Lock()

        # Standard-Logger für Fallback
        self._logger = logging.getLogger(f"audit.{name}")
        self._logger.setLevel(logging.INFO)

        # Stelle sicher, dass Log-Verzeichnis existiert
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Integration mit neuem Audit-System
        self.audit_engine = get_audit_engine()

    async def log_event(
        self,
        level: AuditLevel,
        category: str,
        action: str,
        resource: str = "",
        message: str = "",
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        correlation_id: str | None = None,
        result: str = "unknown",
        compliance_relevant: bool = False,
        **kwargs,
    ) -> str:
        """Protokolliert ein Audit-Event.

        Args:
            level: Audit-Level
            category: Event-Kategorie
            action: Durchgeführte Aktion
            resource: Betroffene Ressource
            message: Event-Nachricht
            details: Zusätzliche Details
            user_id: Benutzer-ID
            agent_id: Agent-ID
            session_id: Session-ID
            correlation_id: Korrelations-ID
            result: Ergebnis der Aktion
            compliance_relevant: Compliance-relevant
            **kwargs: Zusätzliche Metadaten

        Returns:
            Event-ID
        """
        event = AuditEvent(
            level=level,
            category=category,
            action=action,
            resource=resource,
            message=message,
            details=details or {},
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            correlation_id=correlation_id,
            result=result,
            compliance_relevant=compliance_relevant,
            **kwargs,
        )

        # Event zu Puffer hinzufügen
        async with self._buffer_lock:
            self._event_buffer.append(event)

            # Puffer leeren wenn voll oder Intervall erreicht
            if (
                len(self._event_buffer) >= self._buffer_size
                or time.time() - self._last_flush >= self._flush_interval
            ):
                await self._flush_buffer()

        return event.event_id

    async def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Protokolliert Benutzer-Aktionen für Audit-Trail."""
        return await self.log_event(
            level=AuditLevel.INFO,
            category="user_action",
            action=action,
            resource=resource,
            message=f"User action: {action} on {resource}",
            details=details,
            user_id=user_id,
            correlation_id=correlation_id,
            result=result,
            compliance_relevant=True,
        )

    async def log_task_created(self, task: Any) -> str:
        """Protokolliert Task-Erstellung."""
        return await self.log_event(
            level=AuditLevel.INFO,
            category="task_lifecycle",
            action="task_created",
            resource=f"task_{getattr(task, 'id', 'unknown')}",
            message=f"Task created: {getattr(task, 'type', 'unknown')}",
            details={
                "task_type": getattr(task, "type", "unknown"),
                "priority": getattr(task, "priority", 0),
            },
            user_id=getattr(task, "user_id", None),
            agent_id=getattr(task, "agent_id", None),
            compliance_relevant=True,
        )

    async def log_task_completed(self, task: Any, result: Any) -> str:
        """Protokolliert Task-Abschluss."""
        return await self.log_event(
            level=AuditLevel.INFO,
            category="task_lifecycle",
            action="task_completed",
            resource=f"task_{getattr(task, 'id', 'unknown')}",
            message=f"Task completed: {getattr(task, 'type', 'unknown')}",
            details={
                "task_type": getattr(task, "type", "unknown"),
                "duration_seconds": getattr(result, "duration_seconds", 0),
                "result_size_bytes": getattr(result, "result_size_bytes", 0),
            },
            user_id=getattr(task, "user_id", None),
            agent_id=getattr(task, "agent_id", None),
            result=getattr(result, "status", "completed"),
            compliance_relevant=True,
        )

    async def _flush_buffer(self) -> None:
        """Leert den Event-Puffer und schreibt Events in Datei."""
        if not self._event_buffer:
            return

        events_to_write = self._event_buffer.copy()
        self._event_buffer.clear()
        self._last_flush = time.time()

        try:
            # Events serialisieren
            log_entries = []
            for event in events_to_write:
                entry = json.dumps(event.to_dict(), ensure_ascii=False)

                # Verschlüsseln falls aktiviert
                if self.enable_encryption and self.encryption_manager:
                    try:
                        entry = await self.encryption_manager.encrypt_data(entry)
                    except Exception as e:
                        logger.warning(f"Encryption failed, logging unencrypted: {e}")

                log_entries.append(entry)

            # In Datei schreiben
            with open(self.log_file, "a", encoding="utf-8") as f:
                for entry in log_entries:
                    f.write(f"{entry}\n")

        except Exception as e:
            # Fallback auf Standard-Logger
            self._logger.error(f"Failed to write audit events: {e}")
            for event in events_to_write:
                self._logger.info(f"AUDIT: {event.to_dict()}")

    async def flush(self) -> None:
        """Erzwingt das Leeren des Puffers."""
        async with self._buffer_lock:
            await self._flush_buffer()


class SecurityAuditAdapter:
    """Security Audit Adapter - Konsolidiert enhanced_security/audit_logger.py Funktionalität."""

    def __init__(self, audit_engine: AuditEngine | None = None):
        """Initialisiert Security Audit Adapter.

        Args:
            audit_engine: Optionale Audit-Engine (verwendet Standard wenn None)
        """
        self.audit_engine = audit_engine or get_audit_engine()
        self._stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_level": {},
            "last_cleanup": time.time()
        }

        logger.info("Security Audit Adapter initialisiert")

    def _map_security_level_to_severity(self, level: SecurityAuditLevel) -> AuditSeverity:
        """Mappt Security-Level zu Audit-Severity."""
        mapping = {
            SecurityAuditLevel.INFO: AuditSeverity.LOW,
            SecurityAuditLevel.WARNING: AuditSeverity.MEDIUM,
            SecurityAuditLevel.ERROR: AuditSeverity.HIGH,
            SecurityAuditLevel.CRITICAL: AuditSeverity.CRITICAL,
            SecurityAuditLevel.SECURITY: AuditSeverity.HIGH,
            SecurityAuditLevel.COMPLIANCE: AuditSeverity.MEDIUM,
        }
        return mapping.get(level, AuditSeverity.MEDIUM)

    def _map_security_event_type_to_audit_type(self, event_type: SecurityAuditEventType) -> AuditEventType:
        """Mappt Security-Event-Type zu Audit-Event-Type."""
        mapping = {
            SecurityAuditEventType.AUTHENTICATION: AuditEventType.AUTHENTICATION,
            SecurityAuditEventType.AUTHORIZATION: AuditEventType.AUTHORIZATION,
            SecurityAuditEventType.DATA_ACCESS: AuditEventType.DATA_ACCESS,
            SecurityAuditEventType.DATA_MODIFICATION: AuditEventType.DATA_ACCESS,
            SecurityAuditEventType.SYSTEM_ACCESS: AuditEventType.SYSTEM_EVENT,
            SecurityAuditEventType.CONFIGURATION_CHANGE: AuditEventType.CONFIGURATION_CHANGE,
            SecurityAuditEventType.SECURITY_VIOLATION: AuditEventType.AUTHENTICATION,  # Closest match
            SecurityAuditEventType.COMPLIANCE_CHECK: AuditEventType.SYSTEM_EVENT,
            SecurityAuditEventType.AGENT_OPERATION: AuditEventType.AGENT_INPUT,
            SecurityAuditEventType.ERROR_EVENT: AuditEventType.SYSTEM_EVENT,
        }
        return mapping.get(event_type, AuditEventType.SYSTEM_EVENT)

    @trace_function("security_audit.log_event")
    async def log_event(
        self,
        event_type: SecurityAuditEventType,
        message: str,
        level: SecurityAuditLevel = SecurityAuditLevel.INFO,
        agent_id: str | None = None,
        user_id: str | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Protokolliert Security-Event (Kompatibilität zu enhanced_security).

        Args:
            event_type: Typ des Events
            message: Event-Nachricht
            level: Audit-Level
            agent_id: Agent-ID
            user_id: Benutzer-ID
            operation: Operation
            metadata: Zusätzliche Metadaten

        Returns:
            Event-ID
        """
        try:
            # Erstelle AuditContext
            context = AuditContext(
                correlation_id=f"security_{int(time.time())}",
                agent_id=agent_id,
                user_id=user_id,
                metadata=metadata or {}
            )

            # Mappe zu Audit-System-Typen
            audit_event_type = self._map_security_event_type_to_audit_type(event_type)
            severity = self._map_security_level_to_severity(level)

            # Erstelle Event im Audit-System
            audit_event = await self.audit_engine.create_event(
                event_type=audit_event_type,
                action=operation or event_type.value,
                description=message,
                severity=severity,
                result=AuditResult.SUCCESS if level in [SecurityAuditLevel.INFO] else AuditResult.FAILURE,
                context=context,
                input_data=metadata,
                compliance_tags={"security", "enhanced_security_compat"}
            )

            # Statistiken aktualisieren
            self._update_stats(event_type, level)

            logger.debug(f"Security-Event protokolliert: {audit_event.event_id}")
            return audit_event.event_id

        except Exception as e:
            logger.error(f"Security-Audit-Logging fehlgeschlagen: {e}")
            raise

    @trace_function("security_audit.log_security_event")
    async def log_security_event(
        self,
        message: str,
        violation_type: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Protokolliert Sicherheits-Event (Kompatibilität zu enhanced_security).

        Args:
            message: Event-Nachricht
            violation_type: Typ der Verletzung
            agent_id: Agent-ID
            user_id: Benutzer-ID
            severity: Schweregrad
            metadata: Zusätzliche Metadaten

        Returns:
            Event-ID
        """
        security_metadata = {
            "violation_type": violation_type,
            "severity": severity,
            **(metadata or {})
        }

        # Mappe Severity-String zu SecurityAuditLevel
        level_mapping = {
            "low": SecurityAuditLevel.INFO,
            "medium": SecurityAuditLevel.WARNING,
            "high": SecurityAuditLevel.ERROR,
            "critical": SecurityAuditLevel.CRITICAL
        }
        level = level_mapping.get(severity.lower(), SecurityAuditLevel.WARNING)

        return await self.log_event(
            event_type=SecurityAuditEventType.SECURITY_VIOLATION,
            message=message,
            level=SecurityAuditLevel.SECURITY,
            agent_id=agent_id,
            user_id=user_id,
            metadata=security_metadata
        )

    @trace_function("security_audit.log_compliance_event")
    async def log_compliance_event(
        self,
        message: str,
        compliance_rule: str,
        status: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Protokolliert Compliance-Event (Kompatibilität zu enhanced_security).

        Args:
            message: Event-Nachricht
            compliance_rule: Compliance-Regel
            status: Status (passed, failed, warning)
            agent_id: Agent-ID
            metadata: Zusätzliche Metadaten

        Returns:
            Event-ID
        """
        compliance_metadata = {
            "compliance_rule": compliance_rule,
            "status": status,
            **(metadata or {})
        }

        # Mappe Status zu Level
        level = SecurityAuditLevel.COMPLIANCE
        if status == "failed":
            level = SecurityAuditLevel.ERROR
        elif status == "warning":
            level = SecurityAuditLevel.WARNING

        return await self.log_event(
            event_type=SecurityAuditEventType.COMPLIANCE_CHECK,
            message=message,
            level=level,
            agent_id=agent_id,
            metadata=compliance_metadata
        )

    def _update_stats(self, event_type: SecurityAuditEventType, level: SecurityAuditLevel) -> None:
        """Aktualisiert interne Statistiken."""
        self._stats["total_events"] += 1

        # Events by type
        type_key = event_type.value
        self._stats["events_by_type"][type_key] = self._stats["events_by_type"].get(type_key, 0) + 1

        # Events by level
        level_key = level.value
        self._stats["events_by_level"][level_key] = self._stats["events_by_level"].get(level_key, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Gibt Statistiken zurück (Kompatibilität zu enhanced_security)."""
        return self._stats.copy()


# Globale Instanz für Kompatibilität
_security_audit_adapter: SecurityAuditAdapter | None = None


def get_security_audit_adapter() -> SecurityAuditAdapter:
    """Gibt globale Security Audit Adapter Instanz zurück."""
    global _security_audit_adapter
    if _security_audit_adapter is None:
        _security_audit_adapter = SecurityAuditAdapter()
    return _security_audit_adapter


# Exports für Kompatibilität (enhanced_security + security_audit_adapter)
__all__ = [
    # Original enhanced_security exports
    "AuditEvent",
    "AuditLevel",
    "AuditLogger",
    "ComplianceReporter",
    # Security audit adapter exports
    "SecurityAuditAdapter",
    "SecurityAuditLevel",
    "SecurityAuditEventType",
    "SecurityAuditEvent",
    "ComplianceReport",
    "get_security_audit_adapter",
]
