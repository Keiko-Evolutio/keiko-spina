# backend/audit_system/core_audit_engine.py
"""Core Audit Engine für Keiko Personal Assistant

Implementiert zentrale Audit-Event-Verwaltung, Event-Klassifizierung
und Integration mit bestehenden Logging-Systemen.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
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

# Fallback für fehlende audit_constants
try:
    from .audit_constants import AuditMessages
except ImportError:
    class AuditMessages:
        """Fallback für fehlende AuditMessages."""

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Typen von Audit-Events."""
    AGENT_INPUT = "agent_input"
    AGENT_OUTPUT = "agent_output"
    TOOL_CALL = "tool_call"
    AGENT_COMMUNICATION = "agent_communication"
    POLICY_ENFORCEMENT = "policy_enforcement"
    QUOTA_USAGE = "quota_usage"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"


class AuditSeverity(str, Enum):
    """Schweregrad von Audit-Events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditResult(str, Enum):
    """Ergebnis von Audit-Events."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class AuditContext:
    """Kontext-Informationen für Audit-Events."""
    correlation_id: str
    session_id: str | None = None
    request_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    tenant_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None
    source_system: str | None = None
    trace_id: str | None = None
    span_id: str | None = None

    # Zusätzliche Kontext-Daten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "source_system": self.source_system,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata
        }


@dataclass
class AuditSignature:
    """Kryptographische Signatur für Audit-Events."""
    algorithm: str
    signature: str
    public_key_id: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "algorithm": self.algorithm,
            "signature": self.signature,
            "public_key_id": self.public_key_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AuditEvent:
    """Zentrales Audit-Event."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    result: AuditResult

    # Zeitstempel
    timestamp: datetime

    # Event-Details
    action: str
    description: str
    resource_type: str | None = None
    resource_id: str | None = None

    # Kontext
    context: AuditContext = field(default_factory=lambda: AuditContext(correlation_id=str(uuid.uuid4())))

    # Event-Daten
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None

    # Sicherheit
    signature: AuditSignature | None = None
    hash_value: str | None = None
    previous_hash: str | None = None

    # Compliance
    compliance_tags: set[str] = field(default_factory=set)
    retention_period_days: int | None = None

    # Performance
    processing_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Event zu Dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "result": self.result.value,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "description": self.description,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "context": self.context.to_dict(),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_details": self.error_details,
            "signature": self.signature.to_dict() if self.signature else None,
            "hash_value": self.hash_value,
            "previous_hash": self.previous_hash,
            "compliance_tags": list(self.compliance_tags),
            "retention_period_days": self.retention_period_days,
            "processing_time_ms": self.processing_time_ms
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Erstellt Event aus Dictionary."""
        context_data = data.get("context", {})
        context = AuditContext(
            correlation_id=context_data.get("correlation_id", str(uuid.uuid4())),
            session_id=context_data.get("session_id"),
            request_id=context_data.get("request_id"),
            user_id=context_data.get("user_id"),
            agent_id=context_data.get("agent_id"),
            tenant_id=context_data.get("tenant_id"),
            client_ip=context_data.get("client_ip"),
            user_agent=context_data.get("user_agent"),
            source_system=context_data.get("source_system"),
            trace_id=context_data.get("trace_id"),
            span_id=context_data.get("span_id"),
            metadata=context_data.get("metadata", {})
        )

        signature = None
        if data.get("signature"):
            sig_data = data["signature"]
            signature = AuditSignature(
                algorithm=sig_data["algorithm"],
                signature=sig_data["signature"],
                public_key_id=sig_data["public_key_id"],
                timestamp=datetime.fromisoformat(sig_data["timestamp"])
            )

        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data["severity"]),
            result=AuditResult(data["result"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=data["action"],
            description=data["description"],
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            context=context,
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            error_details=data.get("error_details"),
            signature=signature,
            hash_value=data.get("hash_value"),
            previous_hash=data.get("previous_hash"),
            compliance_tags=set(data.get("compliance_tags", [])),
            retention_period_days=data.get("retention_period_days"),
            processing_time_ms=data.get("processing_time_ms")
        )


@dataclass
class AuditBlock:
    """Block in der Audit-Chain."""
    block_id: str
    timestamp: datetime
    events: list[AuditEvent]
    previous_block_hash: str | None
    block_hash: str
    merkle_root: str
    signature: AuditSignature

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Block zu Dictionary."""
        return {
            "block_id": self.block_id,
            "timestamp": self.timestamp.isoformat(),
            "events": [event.to_dict() for event in self.events],
            "previous_block_hash": self.previous_block_hash,
            "block_hash": self.block_hash,
            "merkle_root": self.merkle_root,
            "signature": self.signature.to_dict()
        }


@dataclass
class AuditChain:
    """Blockchain-ähnliche Audit-Chain."""
    chain_id: str
    created_at: datetime
    blocks: list[AuditBlock] = field(default_factory=list)

    @property
    def latest_block_hash(self) -> str | None:
        """Gibt Hash des letzten Blocks zurück."""
        return self.blocks[-1].block_hash if self.blocks else None

    @property
    def total_events(self) -> int:
        """Gibt Gesamtanzahl Events zurück."""
        return sum(len(block.events) for block in self.blocks)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Chain zu Dictionary."""
        return {
            "chain_id": self.chain_id,
            "created_at": self.created_at.isoformat(),
            "blocks": [block.to_dict() for block in self.blocks],
            "total_events": self.total_events
        }


class AuditEventProcessor(ABC):
    """Basis-Klasse für Audit-Event-Prozessoren."""

    @abstractmethod
    async def process_event(self, event: AuditEvent) -> bool:
        """Verarbeitet Audit-Event."""

    @abstractmethod
    async def process_batch(self, events: list[AuditEvent]) -> list[bool]:
        """Verarbeitet Batch von Events."""


class DefaultAuditEventProcessor(AuditEventProcessor):
    """Standard-Audit-Event-Prozessor."""

    def __init__(self) -> None:
        """Initialisiert Default Processor."""
        self._processed_events = 0
        self._failed_events = 0

    async def process_event(self, event: AuditEvent) -> bool:
        """Verarbeitet einzelnes Event."""
        try:
            # Hier würde die tatsächliche Verarbeitung stattfinden
            # z.B. Speicherung in Datenbank, Versendung an externe Systeme

            self._processed_events += 1
            return True

        except Exception as e:
            logger.exception(f"Event-Verarbeitung fehlgeschlagen: {e}")
            self._failed_events += 1
            return False

    async def process_batch(self, events: list[AuditEvent]) -> list[bool]:
        """Verarbeitet Batch von Events."""
        results = []

        for event in events:
            result = await self.process_event(event)
            results.append(result)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Prozessor-Statistiken zurück."""
        total = self._processed_events + self._failed_events
        success_rate = (self._processed_events / total) if total > 0 else 0.0

        return {
            "processed_events": self._processed_events,
            "failed_events": self._failed_events,
            "total_events": total,
            "success_rate": success_rate
        }


class AuditEngine:
    """Zentrale Audit-Engine."""

    def __init__(self) -> None:
        """Initialisiert Audit Engine."""
        self._processors: list[AuditEventProcessor] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._is_running = False

        # Event-Statistiken
        self._events_created = 0
        self._events_processed = 0
        self._events_failed = 0

        # Event-Cache für Performance
        self._event_cache: dict[str, AuditEvent] = {}
        self._cache_max_size = 1000

        # Standard-Prozessor registrieren
        self.register_processor(DefaultAuditEventProcessor())

    def register_processor(self, processor: AuditEventProcessor) -> None:
        """Registriert Event-Prozessor."""
        self._processors.append(processor)
        logger.info(f"Audit-Event-Prozessor registriert: {processor.__class__.__name__}")

    async def start(self) -> None:
        """Startet Audit-Engine."""
        if self._is_running:
            return

        self._is_running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info(AuditMessages.ENGINE_STARTED)

    async def stop(self) -> None:
        """Stoppt Audit-Engine."""
        self._is_running = False

        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        logger.info(AuditMessages.ENGINE_STOPPED)

    @trace_function("audit.create_event")
    async def create_event(
        self,
        event_type: AuditEventType,
        action: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        result: AuditResult = AuditResult.SUCCESS,
        context: AuditContext | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        compliance_tags: set[str] | None = None,
        retention_period_days: int | None = None
    ) -> AuditEvent:
        """Erstellt neues Audit-Event."""
        event_id = str(uuid.uuid4())

        if context is None:
            context = AuditContext(correlation_id=str(uuid.uuid4()))

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            result=result,
            timestamp=datetime.now(UTC),
            action=action,
            description=description,
            resource_type=resource_type,
            resource_id=resource_id,
            context=context,
            input_data=input_data,
            output_data=output_data,
            error_details=error_details,
            compliance_tags=compliance_tags or set(),
            retention_period_days=retention_period_days
        )

        self._events_created += 1

        # Event zur Verarbeitung einreihen
        await self._event_queue.put(event)

        # Event cachen
        self._cache_event(event)

        return event

    async def _processing_loop(self) -> None:
        """Haupt-Verarbeitungsschleife."""
        while self._is_running:
            try:
                # Event aus Queue holen (mit Timeout)
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                # Event an alle Prozessoren weiterleiten
                await self._process_event_with_processors(event)

            except TimeoutError:
                # Timeout ist normal, weiter warten
                continue
            except Exception as e:
                logger.exception(f"Fehler in Audit-Processing-Loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_event_with_processors(self, event: AuditEvent) -> None:
        """Verarbeitet Event mit allen registrierten Prozessoren."""
        success_count = 0

        for processor in self._processors:
            try:
                success = await processor.process_event(event)
                if success:
                    success_count += 1
            except Exception as e:
                logger.exception(f"Prozessor-Fehler: {e}")

        if success_count > 0:
            self._events_processed += 1
        else:
            self._events_failed += 1

    def _cache_event(self, event: AuditEvent) -> None:
        """Cached Event für schnellen Zugriff."""
        if len(self._event_cache) >= self._cache_max_size:
            # Entferne ältestes Event
            oldest_key = next(iter(self._event_cache))
            del self._event_cache[oldest_key]

        self._event_cache[event.event_id] = event

    def get_event(self, event_id: str) -> AuditEvent | None:
        """Gibt Event aus Cache zurück."""
        return self._event_cache.get(event_id)

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Engine-Statistiken zurück."""
        return {
            "events_created": self._events_created,
            "events_processed": self._events_processed,
            "events_failed": self._events_failed,
            "events_pending": self._event_queue.qsize(),
            "cache_size": len(self._event_cache),
            "registered_processors": len(self._processors),
            "is_running": self._is_running
        }


# Globale Instanz für einfachen Zugriff
_global_audit_engine: AuditEngine | None = None


def get_audit_engine() -> AuditEngine:
    """Gibt globale AuditEngine-Instanz zurück."""
    global _global_audit_engine
    if _global_audit_engine is None:
        _global_audit_engine = AuditEngine()
    return _global_audit_engine


# Globale Audit Engine Instanz
audit_engine = AuditEngine()
