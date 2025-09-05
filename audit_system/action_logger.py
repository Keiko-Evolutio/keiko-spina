# backend/audit_system/action_logger.py
"""Comprehensive Action Logger für Keiko Personal Assistant

Zeichnet alle Agent-Inputs, Outputs, Tool-Calls, Kommunikation
und Policy-Enforcement-Entscheidungen auf.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .audit_constants import AuditConstants, AuditMessages
from .audit_utils import generate_correlation_id
from .core_audit_engine import AuditContext, AuditEventType, AuditResult, AuditSeverity

logger = get_logger(__name__)


class ActionCategory(str, Enum):
    """Kategorien von Agent-Aktionen."""
    INPUT_PROCESSING = "input_processing"
    OUTPUT_GENERATION = "output_generation"
    TOOL_EXECUTION = "tool_execution"
    AGENT_COMMUNICATION = "agent_communication"
    POLICY_ENFORCEMENT = "policy_enforcement"
    QUOTA_MANAGEMENT = "quota_management"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "configuration"
    SYSTEM_OPERATION = "system_operation"


class ToolCallStatus(str, Enum):
    """Status von Tool-Calls."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class CommunicationType(str, Enum):
    """Typen von Agent-Kommunikation."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    DELEGATION = "delegation"
    COLLABORATION = "collaboration"


@dataclass
class AgentActionEvent:
    """Event für Agent-Aktionen."""
    action_id: str
    category: ActionCategory
    agent_id: str

    # Input/Output-Daten
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None

    # Verarbeitungsdetails
    processing_time_ms: float | None = None
    tokens_consumed: int | None = None
    model_used: str | None = None

    # Kontext
    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "action_id": self.action_id,
            "category": self.category.value,
            "agent_id": self.agent_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "processing_time_ms": self.processing_time_ms,
            "tokens_consumed": self.tokens_consumed,
            "model_used": self.model_used,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata
        }


@dataclass
class ToolCallEvent:
    """Event für Tool-Calls."""
    call_id: str
    tool_name: str
    agent_id: str
    status: ToolCallStatus

    # Call-Details
    parameters: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None

    # Ressourcen-Verbrauch
    cpu_time_ms: float | None = None
    memory_usage_mb: float | None = None
    network_calls: int | None = None

    # Sicherheit
    permission_required: str | None = None
    permission_granted: bool | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "parameters": self.parameters,
            "result": self.result,
            "error_details": self.error_details,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "network_calls": self.network_calls,
            "permission_required": self.permission_required,
            "permission_granted": self.permission_granted,
            "metadata": self.metadata
        }


@dataclass
class CommunicationEvent:
    """Event für Agent-zu-Agent-Kommunikation."""
    communication_id: str
    communication_type: CommunicationType

    # Teilnehmer
    sender_agent_id: str
    receiver_agent_id: str | None = None  # None für Broadcasts

    # Nachricht
    message_type: str | None = None
    message_content: dict[str, Any] | None = None
    message_size_bytes: int | None = None

    # Protokoll
    protocol: str | None = None  # z.B. "kei_rpc", "http", "websocket"
    endpoint: str | None = None

    # Timing
    sent_at: datetime | None = None
    received_at: datetime | None = None
    response_time_ms: float | None = None

    # Status
    delivery_status: str | None = None
    error_details: dict[str, Any] | None = None

    # Sicherheit
    encrypted: bool | None = None
    authentication_method: str | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "communication_id": self.communication_id,
            "communication_type": self.communication_type.value,
            "sender_agent_id": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
            "message_type": self.message_type,
            "message_content": self.message_content,
            "message_size_bytes": self.message_size_bytes,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "response_time_ms": self.response_time_ms,
            "delivery_status": self.delivery_status,
            "error_details": self.error_details,
            "encrypted": self.encrypted,
            "authentication_method": self.authentication_method,
            "metadata": self.metadata
        }


@dataclass
class PolicyEnforcementEvent:
    """Event für Policy-Enforcement-Entscheidungen."""
    enforcement_id: str
    policy_id: str
    policy_type: str

    # Enforcement-Details
    decision: str  # "allow", "deny", "modify"
    reason: str
    confidence_score: float | None = None

    # Kontext
    agent_id: str | None = None
    user_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None

    # Input/Output
    original_request: dict[str, Any] | None = None
    modified_request: dict[str, Any] | None = None
    policy_violations: list[str] | None = None

    # Timing
    evaluation_time_ms: float | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "enforcement_id": self.enforcement_id,
            "policy_id": self.policy_id,
            "policy_type": self.policy_type,
            "decision": self.decision,
            "reason": self.reason,
            "confidence_score": self.confidence_score,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "original_request": self.original_request,
            "modified_request": self.modified_request,
            "policy_violations": self.policy_violations,
            "evaluation_time_ms": self.evaluation_time_ms,
            "metadata": self.metadata
        }


@dataclass
class QuotaUsageEvent:
    """Event für Quota-Verbrauch."""
    usage_id: str
    quota_type: str
    quota_id: str

    # Verbrauch
    amount_requested: float
    amount_consumed: float
    amount_remaining: float

    # Kontext
    agent_id: str | None = None
    user_id: str | None = None
    operation_type: str | None = None

    # Status
    quota_exceeded: bool = False
    enforcement_action: str | None = None

    # Timing
    window_start: datetime | None = None
    window_end: datetime | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "usage_id": self.usage_id,
            "quota_type": self.quota_type,
            "quota_id": self.quota_id,
            "amount_requested": self.amount_requested,
            "amount_consumed": self.amount_consumed,
            "amount_remaining": self.amount_remaining,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "operation_type": self.operation_type,
            "quota_exceeded": self.quota_exceeded,
            "enforcement_action": self.enforcement_action,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "metadata": self.metadata
        }


class ActionLogger:
    """Comprehensive Action Logger."""

    def __init__(self):
        """Initialisiert Action Logger."""
        # Event-Queues für verschiedene Kategorien
        self._agent_action_queue: asyncio.Queue = asyncio.Queue()
        self._tool_call_queue: asyncio.Queue = asyncio.Queue()
        self._communication_queue: asyncio.Queue = asyncio.Queue()
        self._policy_enforcement_queue: asyncio.Queue = asyncio.Queue()
        self._quota_usage_queue: asyncio.Queue = asyncio.Queue()

        # Processing-Tasks
        self._processing_tasks: list[asyncio.Task] = []
        self._is_running = False

        # Statistiken
        self._events_logged = {
            ActionCategory.INPUT_PROCESSING: 0,
            ActionCategory.OUTPUT_GENERATION: 0,
            ActionCategory.TOOL_EXECUTION: 0,
            ActionCategory.AGENT_COMMUNICATION: 0,
            ActionCategory.POLICY_ENFORCEMENT: 0,
            ActionCategory.QUOTA_MANAGEMENT: 0
        }

        # Event-Cache für Korrelation
        self._event_cache: dict[str, dict[str, Any]] = {}
        self._cache_max_size = AuditConstants.DEFAULT_CACHE_MAX_SIZE

    async def start(self) -> None:
        """Startet Action Logger."""
        if self._is_running:
            return

        self._is_running = True

        # Starte Processing-Tasks für verschiedene Event-Typen
        self._processing_tasks = [
            asyncio.create_task(self._process_agent_actions()),
            asyncio.create_task(self._process_tool_calls()),
            asyncio.create_task(self._process_communications()),
            asyncio.create_task(self._process_policy_enforcements()),
            asyncio.create_task(self._process_quota_usage())
        ]

        logger.info(AuditMessages.LOGGER_STARTED)

    async def stop(self) -> None:
        """Stoppt Action Logger."""
        self._is_running = False

        # Stoppe alle Processing-Tasks
        for task in self._processing_tasks:
            task.cancel()

        # Warte auf Completion
        await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        logger.info(AuditMessages.LOGGER_STOPPED)

    @trace_function("action_logger.log_agent_input")
    async def log_agent_input(
        self,
        agent_id: str,
        input_data: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Agent-Input.

        Args:
            agent_id: Agent-ID
            input_data: Input-Daten
            user_id: User-ID
            session_id: Session-ID
            conversation_id: Conversation-ID
            metadata: Zusätzliche Metadaten

        Returns:
            Action-ID
        """
        action_id = generate_correlation_id()

        event = self._create_agent_action_event(
            action_id=action_id,
            category=ActionCategory.INPUT_PROCESSING,
            agent_id=agent_id,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            metadata=metadata
        )

        await self._agent_action_queue.put(event)
        self._events_logged[ActionCategory.INPUT_PROCESSING] += 1

        return action_id

    @trace_function("action_logger.log_agent_output")
    async def log_agent_output(
        self,
        agent_id: str,
        output_data: dict[str, Any],
        processing_time_ms: float | None = None,
        tokens_consumed: int | None = None,
        model_used: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Agent-Output.

        Args:
            agent_id: Agent-ID
            output_data: Output-Daten
            processing_time_ms: Verarbeitungszeit
            tokens_consumed: Verbrauchte Tokens
            model_used: Verwendetes Modell
            user_id: User-ID
            session_id: Session-ID
            conversation_id: Conversation-ID
            metadata: Zusätzliche Metadaten

        Returns:
            Action-ID
        """
        action_id = str(uuid.uuid4())

        event = AgentActionEvent(
            action_id=action_id,
            category=ActionCategory.OUTPUT_GENERATION,
            agent_id=agent_id,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            tokens_consumed=tokens_consumed,
            model_used=model_used,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            metadata=metadata or {}
        )

        await self._agent_action_queue.put(event)
        self._events_logged[ActionCategory.OUTPUT_GENERATION] += 1

        return action_id

    @trace_function("action_logger.log_tool_call")
    async def log_tool_call(
        self,
        tool_name: str,
        agent_id: str,
        parameters: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        status: ToolCallStatus = ToolCallStatus.COMPLETED,
        error_details: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        permission_required: str | None = None,
        permission_granted: bool | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Tool-Call.

        Args:
            tool_name: Name des Tools
            agent_id: Agent-ID
            parameters: Tool-Parameter
            result: Tool-Ergebnis
            status: Call-Status
            error_details: Fehler-Details
            duration_ms: Ausführungsdauer
            permission_required: Benötigte Permission
            permission_granted: Permission gewährt
            metadata: Zusätzliche Metadaten

        Returns:
            Call-ID
        """
        call_id = str(uuid.uuid4())

        event = ToolCallEvent(
            call_id=call_id,
            tool_name=tool_name,
            agent_id=agent_id,
            status=status,
            parameters=parameters,
            result=result,
            error_details=error_details,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC) if status in [ToolCallStatus.COMPLETED, ToolCallStatus.FAILED] else None,
            duration_ms=duration_ms,
            permission_required=permission_required,
            permission_granted=permission_granted,
            metadata=metadata or {}
        )

        await self._tool_call_queue.put(event)
        self._events_logged[ActionCategory.TOOL_EXECUTION] += 1

        return call_id

    @trace_function("action_logger.log_agent_communication")
    async def log_agent_communication(
        self,
        sender_agent_id: str,
        receiver_agent_id: str | None,
        communication_type: CommunicationType,
        message_type: str | None = None,
        message_content: dict[str, Any] | None = None,
        protocol: str | None = None,
        endpoint: str | None = None,
        response_time_ms: float | None = None,
        delivery_status: str | None = None,
        error_details: dict[str, Any] | None = None,
        encrypted: bool | None = None,
        authentication_method: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Agent-Kommunikation.

        Args:
            sender_agent_id: Sender-Agent-ID
            receiver_agent_id: Empfänger-Agent-ID
            communication_type: Kommunikationstyp
            message_type: Nachrichtentyp
            message_content: Nachrichteninhalt
            protocol: Verwendetes Protokoll
            endpoint: Endpoint
            response_time_ms: Response-Zeit
            delivery_status: Zustellstatus
            error_details: Fehler-Details
            encrypted: Verschlüsselt
            authentication_method: Authentifizierungsmethode
            metadata: Zusätzliche Metadaten

        Returns:
            Communication-ID
        """
        communication_id = str(uuid.uuid4())

        event = CommunicationEvent(
            communication_id=communication_id,
            communication_type=communication_type,
            sender_agent_id=sender_agent_id,
            receiver_agent_id=receiver_agent_id,
            message_type=message_type,
            message_content=message_content,
            message_size_bytes=len(str(message_content)) if message_content else None,
            protocol=protocol,
            endpoint=endpoint,
            sent_at=datetime.now(UTC),
            response_time_ms=response_time_ms,
            delivery_status=delivery_status,
            error_details=error_details,
            encrypted=encrypted,
            authentication_method=authentication_method,
            metadata=metadata or {}
        )

        await self._communication_queue.put(event)
        self._events_logged[ActionCategory.AGENT_COMMUNICATION] += 1

        return communication_id

    @trace_function("action_logger.log_policy_enforcement")
    async def log_policy_enforcement(
        self,
        policy_id: str,
        policy_type: str,
        decision: str,
        reason: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        original_request: dict[str, Any] | None = None,
        modified_request: dict[str, Any] | None = None,
        policy_violations: list[str] | None = None,
        confidence_score: float | None = None,
        evaluation_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Policy-Enforcement-Entscheidung.

        Args:
            policy_id: Policy-ID
            policy_type: Policy-Typ
            decision: Enforcement-Entscheidung
            reason: Grund für Entscheidung
            agent_id: Agent-ID
            user_id: User-ID
            resource_type: Ressourcen-Typ
            resource_id: Ressourcen-ID
            original_request: Original-Request
            modified_request: Modifizierter Request
            policy_violations: Policy-Verletzungen
            confidence_score: Confidence-Score
            evaluation_time_ms: Evaluationszeit
            metadata: Zusätzliche Metadaten

        Returns:
            Enforcement-ID
        """
        enforcement_id = str(uuid.uuid4())

        event = PolicyEnforcementEvent(
            enforcement_id=enforcement_id,
            policy_id=policy_id,
            policy_type=policy_type,
            decision=decision,
            reason=reason,
            confidence_score=confidence_score,
            agent_id=agent_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            original_request=original_request,
            modified_request=modified_request,
            policy_violations=policy_violations,
            evaluation_time_ms=evaluation_time_ms,
            metadata=metadata or {}
        )

        await self._policy_enforcement_queue.put(event)
        self._events_logged[ActionCategory.POLICY_ENFORCEMENT] += 1

        return enforcement_id

    @trace_function("action_logger.log_quota_usage")
    async def log_quota_usage(
        self,
        quota_type: str,
        quota_id: str,
        amount_requested: float,
        amount_consumed: float,
        amount_remaining: float,
        agent_id: str | None = None,
        user_id: str | None = None,
        operation_type: str | None = None,
        quota_exceeded: bool = False,
        enforcement_action: str | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Quota-Verbrauch.

        Args:
            quota_type: Quota-Typ
            quota_id: Quota-ID
            amount_requested: Angeforderte Menge
            amount_consumed: Verbrauchte Menge
            amount_remaining: Verbleibende Menge
            agent_id: Agent-ID
            user_id: User-ID
            operation_type: Operations-Typ
            quota_exceeded: Quota überschritten
            enforcement_action: Enforcement-Aktion
            window_start: Fenster-Start
            window_end: Fenster-Ende
            metadata: Zusätzliche Metadaten

        Returns:
            Usage-ID
        """
        usage_id = str(uuid.uuid4())

        event = QuotaUsageEvent(
            usage_id=usage_id,
            quota_type=quota_type,
            quota_id=quota_id,
            amount_requested=amount_requested,
            amount_consumed=amount_consumed,
            amount_remaining=amount_remaining,
            agent_id=agent_id,
            user_id=user_id,
            operation_type=operation_type,
            quota_exceeded=quota_exceeded,
            enforcement_action=enforcement_action,
            window_start=window_start,
            window_end=window_end,
            metadata=metadata or {}
        )

        await self._quota_usage_queue.put(event)
        self._events_logged[ActionCategory.QUOTA_MANAGEMENT] += 1

        return usage_id

    async def _process_agent_actions(self) -> None:
        """Verarbeitet Agent-Action-Events."""
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._agent_action_queue.get(),
                    timeout=1.0
                )

                await self._convert_and_forward_event(
                    event,
                    AuditEventType.AGENT_INPUT if event.category == ActionCategory.INPUT_PROCESSING else AuditEventType.AGENT_OUTPUT
                )

            except TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Fehler bei Agent-Action-Processing: {e}")

    async def _process_tool_calls(self) -> None:
        """Verarbeitet Tool-Call-Events."""
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._tool_call_queue.get(),
                    timeout=1.0
                )

                await self._convert_and_forward_event(event, AuditEventType.TOOL_CALL)

            except TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Fehler bei Tool-Call-Processing: {e}")

    async def _process_communications(self) -> None:
        """Verarbeitet Communication-Events."""
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._communication_queue.get(),
                    timeout=1.0
                )

                await self._convert_and_forward_event(event, AuditEventType.AGENT_COMMUNICATION)

            except TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Fehler bei Communication-Processing: {e}")

    async def _process_policy_enforcements(self) -> None:
        """Verarbeitet Policy-Enforcement-Events."""
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._policy_enforcement_queue.get(),
                    timeout=1.0
                )

                await self._convert_and_forward_event(event, AuditEventType.POLICY_ENFORCEMENT)

            except TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Fehler bei Policy-Enforcement-Processing: {e}")

    async def _process_quota_usage(self) -> None:
        """Verarbeitet Quota-Usage-Events."""
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._quota_usage_queue.get(),
                    timeout=1.0
                )

                await self._convert_and_forward_event(event, AuditEventType.QUOTA_USAGE)

            except TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Fehler bei Quota-Usage-Processing: {e}")

    async def _convert_and_forward_event(
        self,
        event: AgentActionEvent | ToolCallEvent | CommunicationEvent | PolicyEnforcementEvent | QuotaUsageEvent,
        audit_event_type: AuditEventType
    ) -> None:
        """Konvertiert Event zu AuditEvent und leitet weiter."""
        from .core_audit_engine import audit_engine

        # Bestimme Severity basierend auf Event-Typ und -Inhalt
        severity = AuditSeverity.MEDIUM
        result = AuditResult.SUCCESS

        if isinstance(event, ToolCallEvent) and event.status == ToolCallStatus.FAILED:
            severity = AuditSeverity.HIGH
            result = AuditResult.FAILURE
        elif (isinstance(event, PolicyEnforcementEvent) and event.decision == "deny") or (isinstance(event, QuotaUsageEvent) and event.quota_exceeded):
            severity = AuditSeverity.HIGH
            result = AuditResult.BLOCKED

        # Erstelle AuditContext
        context = AuditContext(
            correlation_id=str(uuid.uuid4()),
            agent_id=getattr(event, "agent_id", None),
            user_id=getattr(event, "user_id", None),
            session_id=getattr(event, "session_id", None),
            metadata=getattr(event, "metadata", {})
        )

        # Erstelle AuditEvent
        await audit_engine.create_event(
            event_type=audit_event_type,
            action=f"{event.__class__.__name__}",
            description=f"Logged {event.__class__.__name__}",
            severity=severity,
            result=result,
            context=context,
            input_data=event.to_dict(),
            compliance_tags={"action_logging", "comprehensive_audit"}
        )

        # Cache Event für Korrelation
        self._cache_event_for_correlation(event)

    def _cache_event_for_correlation(self, event: Any) -> None:
        """Cached Event für Korrelation."""
        if len(self._event_cache) >= self._cache_max_size:
            # Entferne ältestes Event
            oldest_key = next(iter(self._event_cache))
            del self._event_cache[oldest_key]

        event_id = getattr(event, "action_id", None) or getattr(event, "call_id", None) or str(uuid.uuid4())
        self._event_cache[event_id] = {
            "event": event,
            "timestamp": datetime.now(UTC),
            "type": event.__class__.__name__
        }

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Action-Logger-Statistiken zurück."""
        return {
            "events_logged": dict(self._events_logged),
            "total_events": sum(self._events_logged.values()),
            "cache_size": len(self._event_cache),
            "is_running": self._is_running,
            "queue_sizes": {
                "agent_actions": self._agent_action_queue.qsize(),
                "tool_calls": self._tool_call_queue.qsize(),
                "communications": self._communication_queue.qsize(),
                "policy_enforcements": self._policy_enforcement_queue.qsize(),
                "quota_usage": self._quota_usage_queue.qsize()
            }
        }

    def _create_agent_action_event(
        self,
        action_id: str,
        category: ActionCategory,
        agent_id: str,
        input_data: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> AgentActionEvent:
        """Erstellt AgentActionEvent mit Standard-Metadaten.

        Args:
            action_id: Eindeutige Action-ID
            category: Action-Kategorie
            agent_id: Agent-ID
            input_data: Input-Daten
            user_id: User-ID
            session_id: Session-ID
            conversation_id: Conversation-ID
            metadata: Zusätzliche Metadaten

        Returns:
            Konfiguriertes AgentActionEvent
        """
        return AgentActionEvent(
            action_id=action_id,
            category=category,
            agent_id=agent_id,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            metadata=metadata or {}
        )


# Globale Action Logger Instanz
action_logger = ActionLogger()
