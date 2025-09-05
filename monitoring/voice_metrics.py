"""Voice Workflow Metrics Implementation.
Implementiert spezifisches Monitoring für Voice-to-Orchestrator Workflows.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kei_logging import get_logger

from .interfaces import IMetricsCollector, IVoiceWorkflowMonitor, VoiceWorkflowMetrics

logger = get_logger(__name__)


@dataclass
class WorkflowTrackingState:
    """Tracking-State für einen Voice-Workflow."""
    workflow_id: str
    user_id: str
    session_id: str
    start_time: datetime

    # Phase-Tracking
    speech_to_text_start: datetime | None = None
    speech_to_text_duration_ms: float | None = None
    speech_confidence: float | None = None

    orchestrator_start: datetime | None = None
    orchestrator_duration_ms: float | None = None

    agent_selection_start: datetime | None = None
    agent_selection_duration_ms: float | None = None
    selected_agents: list[str] = field(default_factory=list)

    agent_execution_start: datetime | None = None
    agent_execution_duration_ms: float | None = None
    tools_called: list[str] = field(default_factory=list)

    # Status
    completed: bool = False
    success: bool = False
    error_type: str | None = None
    error_message: str | None = None


class VoiceWorkflowMonitor(IVoiceWorkflowMonitor):
    """Voice Workflow Monitor Implementation.
    Trackt komplette Voice-to-Orchestrator Workflows mit detaillierten Metriken.
    """

    def __init__(self, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector
        self._active_workflows: dict[str, WorkflowTrackingState] = {}
        self._completed_workflows: list[VoiceWorkflowMetrics] = []
        self._workflow_stats = defaultdict(int)

        # Metrik-Namen
        self.WORKFLOW_TOTAL = "voice_workflow_total"
        self.WORKFLOW_DURATION = "voice_workflow_duration_seconds"
        self.WORKFLOW_SUCCESS_RATE = "voice_workflow_success_rate"

        self.STT_DURATION = "voice_stt_duration_seconds"
        self.STT_CONFIDENCE = "voice_stt_confidence"

        self.ORCHESTRATOR_DURATION = "voice_orchestrator_duration_seconds"
        self.AGENT_SELECTION_DURATION = "voice_agent_selection_duration_seconds"
        self.AGENT_EXECUTION_DURATION = "voice_agent_execution_duration_seconds"

        self.ACTIVE_WORKFLOWS = "voice_active_workflows"
        self.ERROR_RATE = "voice_workflow_error_rate"

    async def start_workflow_tracking(self, workflow_id: str, user_id: str, session_id: str) -> None:
        """Startet Workflow-Tracking."""
        start_time = datetime.utcnow()

        tracking_state = WorkflowTrackingState(
            workflow_id=workflow_id,
            user_id=user_id,
            session_id=session_id,
            start_time=start_time
        )

        self._active_workflows[workflow_id] = tracking_state

        # Metriken aktualisieren
        self.metrics_collector.increment_counter(
            self.WORKFLOW_TOTAL,
            labels={"user_id": user_id, "session_id": session_id}
        )

        self.metrics_collector.set_gauge(
            self.ACTIVE_WORKFLOWS,
            len(self._active_workflows)
        )

        logger.debug(f"Started tracking voice workflow {workflow_id} for user {user_id}")

    async def track_speech_to_text(self, workflow_id: str, duration_ms: float, confidence: float = None) -> None:
        """Trackt Speech-to-Text Phase."""
        if workflow_id not in self._active_workflows:
            logger.warning(f"Workflow {workflow_id} not found for STT tracking")
            return

        state = self._active_workflows[workflow_id]
        state.speech_to_text_duration_ms = duration_ms
        state.speech_confidence = confidence

        # Metriken
        self.metrics_collector.observe_histogram(
            self.STT_DURATION,
            duration_ms / 1000.0,  # Convert to seconds
            labels={"user_id": state.user_id}
        )

        if confidence is not None:
            self.metrics_collector.observe_histogram(
                self.STT_CONFIDENCE,
                confidence,
                labels={"user_id": state.user_id}
            )

        logger.debug(f"Tracked STT for workflow {workflow_id}: {duration_ms}ms, confidence: {confidence}")

    async def track_orchestrator(self, workflow_id: str, duration_ms: float) -> None:
        """Trackt Orchestrator Phase."""
        if workflow_id not in self._active_workflows:
            logger.warning(f"Workflow {workflow_id} not found for orchestrator tracking")
            return

        state = self._active_workflows[workflow_id]
        state.orchestrator_duration_ms = duration_ms

        # Metriken
        self.metrics_collector.observe_histogram(
            self.ORCHESTRATOR_DURATION,
            duration_ms / 1000.0,
            labels={"user_id": state.user_id}
        )

        logger.debug(f"Tracked orchestrator for workflow {workflow_id}: {duration_ms}ms")

    async def track_agent_selection(self, workflow_id: str, duration_ms: float, agents: list[str]) -> None:
        """Trackt Agent-Selection Phase."""
        if workflow_id not in self._active_workflows:
            logger.warning(f"Workflow {workflow_id} not found for agent selection tracking")
            return

        state = self._active_workflows[workflow_id]
        state.agent_selection_duration_ms = duration_ms
        state.selected_agents = agents or []

        # Metriken
        self.metrics_collector.observe_histogram(
            self.AGENT_SELECTION_DURATION,
            duration_ms / 1000.0,
            labels={
                "user_id": state.user_id,
                "agent_count": str(len(agents))
            }
        )

        # Agent-spezifische Metriken
        for agent in agents:
            self.metrics_collector.increment_counter(
                "voice_agent_selected_total",
                labels={"agent_name": agent, "user_id": state.user_id}
            )

        logger.debug(f"Tracked agent selection for workflow {workflow_id}: {duration_ms}ms, agents: {agents}")

    async def track_agent_execution(self, workflow_id: str, duration_ms: float, success: bool, tools: list[str] = None) -> None:
        """Trackt Agent-Execution Phase."""
        if workflow_id not in self._active_workflows:
            logger.warning(f"Workflow {workflow_id} not found for agent execution tracking")
            return

        state = self._active_workflows[workflow_id]
        state.agent_execution_duration_ms = duration_ms
        state.tools_called = tools or []

        # Metriken
        self.metrics_collector.observe_histogram(
            self.AGENT_EXECUTION_DURATION,
            duration_ms / 1000.0,
            labels={
                "user_id": state.user_id,
                "success": str(success),
                "tool_count": str(len(tools) if tools else 0)
            }
        )

        # Tool-spezifische Metriken
        if tools:
            for tool in tools:
                self.metrics_collector.increment_counter(
                    "voice_tool_called_total",
                    labels={"tool_name": tool, "user_id": state.user_id, "success": str(success)}
                )

        logger.debug(f"Tracked agent execution for workflow {workflow_id}: {duration_ms}ms, success: {success}, tools: {tools}")

    async def complete_workflow(self, workflow_id: str, success: bool, error: str = None) -> VoiceWorkflowMetrics:
        """Beendet Workflow-Tracking und gibt Metriken zurück."""
        if workflow_id not in self._active_workflows:
            logger.warning(f"Workflow {workflow_id} not found for completion")
            return None

        state = self._active_workflows[workflow_id]
        end_time = datetime.utcnow()
        total_duration_ms = (end_time - state.start_time).total_seconds() * 1000

        # State aktualisieren
        state.completed = True
        state.success = success
        if error:
            state.error_type = type(error).__name__ if hasattr(error, "__class__") else "UnknownError"
            state.error_message = str(error)

        # VoiceWorkflowMetrics erstellen
        metrics = VoiceWorkflowMetrics(
            workflow_id=workflow_id,
            user_id=state.user_id,
            session_id=state.session_id,
            total_duration_ms=total_duration_ms,
            speech_to_text_duration_ms=state.speech_to_text_duration_ms or 0.0,
            orchestrator_duration_ms=state.orchestrator_duration_ms or 0.0,
            agent_selection_duration_ms=state.agent_selection_duration_ms or 0.0,
            agent_execution_duration_ms=state.agent_execution_duration_ms or 0.0,
            success=success,
            error_type=state.error_type,
            error_message=state.error_message,
            speech_confidence=state.speech_confidence,
            agents_used=state.selected_agents,
            tools_called=state.tools_called,
            timestamp=end_time
        )

        # Workflow aus aktiven entfernen und zu completed hinzufügen
        del self._active_workflows[workflow_id]
        self._completed_workflows.append(metrics)

        # Cleanup alte completed workflows (behalte nur letzte 1000)
        if len(self._completed_workflows) > 1000:
            self._completed_workflows = self._completed_workflows[-1000:]

        # Finale Metriken
        self.metrics_collector.observe_histogram(
            self.WORKFLOW_DURATION,
            total_duration_ms / 1000.0,
            labels={
                "user_id": state.user_id,
                "success": str(success),
                "error_type": state.error_type or "none"
            }
        )

        self.metrics_collector.set_gauge(
            self.ACTIVE_WORKFLOWS,
            len(self._active_workflows)
        )

        # Success Rate berechnen
        self._update_success_rate()

        logger.info(f"Completed voice workflow {workflow_id}: {total_duration_ms}ms, success: {success}")

        return metrics

    def _update_success_rate(self) -> None:
        """Aktualisiert Success Rate Metrik."""
        if not self._completed_workflows:
            return

        # Berechne Success Rate der letzten 100 Workflows
        recent_workflows = self._completed_workflows[-100:]
        successful = sum(1 for w in recent_workflows if w.success)
        success_rate = successful / len(recent_workflows)

        self.metrics_collector.set_gauge(
            self.WORKFLOW_SUCCESS_RATE,
            success_rate
        )

        # Error Rate
        error_rate = 1.0 - success_rate
        self.metrics_collector.set_gauge(
            self.ERROR_RATE,
            error_rate
        )

    def get_workflow_statistics(self) -> dict[str, Any]:
        """Gibt Workflow-Statistiken zurück."""
        if not self._completed_workflows:
            return {}

        recent_workflows = self._completed_workflows[-100:]

        total_workflows = len(recent_workflows)
        successful_workflows = sum(1 for w in recent_workflows if w.success)

        avg_duration = sum(w.total_duration_ms for w in recent_workflows) / total_workflows
        avg_stt_duration = sum(w.speech_to_text_duration_ms for w in recent_workflows if w.speech_to_text_duration_ms) / total_workflows

        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "success_rate": successful_workflows / total_workflows,
            "average_duration_ms": avg_duration,
            "average_stt_duration_ms": avg_stt_duration,
            "active_workflows": len(self._active_workflows),
            "most_used_agents": self._get_most_used_agents(recent_workflows),
            "most_used_tools": self._get_most_used_tools(recent_workflows)
        }

    def _get_most_used_agents(self, workflows: list[VoiceWorkflowMetrics]) -> list[dict[str, Any]]:
        """Gibt die am häufigsten verwendeten Agents zurück."""
        agent_counts = defaultdict(int)
        for workflow in workflows:
            for agent in workflow.agents_used:
                agent_counts[agent] += 1

        return [
            {"agent": agent, "count": count}
            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _get_most_used_tools(self, workflows: list[VoiceWorkflowMetrics]) -> list[dict[str, Any]]:
        """Gibt die am häufigsten verwendeten Tools zurück."""
        tool_counts = defaultdict(int)
        for workflow in workflows:
            for tool in workflow.tools_called:
                tool_counts[tool] += 1

        return [
            {"tool": tool, "count": count}
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
