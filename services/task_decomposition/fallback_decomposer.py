# backend/services/task_decomposition/fallback_decomposer.py
"""Regelbasierte Fallback-Decomposition bei LLM-Ausfall.

Implementiert robuste regelbasierte Task-Zerlegung als Fallback
für LLM-basierte Decomposition.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger
from task_management.core_task_manager import TaskPriority, TaskType

from .data_models import (
    DecompositionRequest,
    DecompositionStrategy,
    FallbackRule,
    SubtaskDefinition,
)

logger = get_logger(__name__)


class FallbackDecomposer:
    """Regelbasierte Fallback-Decomposition."""

    def __init__(self):
        """Initialisiert Fallback Decomposer."""
        self.rules: list[FallbackRule] = []
        self._load_default_rules()

        logger.info(f"Fallback Decomposer initialisiert mit {len(self.rules)} Regeln")

    async def decompose_task(self, request: DecompositionRequest) -> list[SubtaskDefinition]:
        """Zerlegt Task mit regelbasierten Fallback-Strategien.

        Args:
            request: Decomposition-Request

        Returns:
            Liste von Subtask-Definitionen
        """
        try:
            # Finde passende Regel
            matching_rule = self._find_matching_rule(request)

            if not matching_rule:
                logger.warning(f"Keine passende Fallback-Regel für Task {request.task_id}")
                return self._create_default_subtasks(request)

            # Generiere Subtasks basierend auf Regel
            subtasks = await self._generate_subtasks_from_rule(request, matching_rule)

            # Update Regel-Statistiken
            matching_rule.usage_count += 1
            matching_rule.last_used = request.request_timestamp

            logger.info({
                "event": "fallback_decomposition_completed",
                "task_id": request.task_id,
                "rule_used": matching_rule.rule_id,
                "subtask_count": len(subtasks)
            })

            return subtasks

        except Exception as e:
            logger.error(f"Fallback-Decomposition fehlgeschlagen: {e}")
            return self._create_default_subtasks(request)

    def _find_matching_rule(self, request: DecompositionRequest) -> FallbackRule:
        """Findet passende Regel für Request."""
        # Sortiere Regeln nach Priorität
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            if self._rule_matches(rule, request):
                return rule

        return None

    def _rule_matches(self, rule: FallbackRule, request: DecompositionRequest) -> bool:
        """Prüft ob Regel auf Request passt."""
        # Task-Type-Pattern-Matching
        task_type = request.task_type.value.lower()

        for pattern in rule.task_type_patterns:
            if pattern.lower() in task_type or task_type in pattern.lower():
                return True

        # Keyword-Matching in Beschreibung
        description = request.task_description.lower()
        for keyword in rule.keyword_triggers:
            if keyword.lower() in description:
                return True

        return False

    async def _generate_subtasks_from_rule(
        self,
        request: DecompositionRequest,
        rule: FallbackRule
    ) -> list[SubtaskDefinition]:
        """Generiert Subtasks basierend auf Regel."""
        subtasks = []

        for i, template in enumerate(rule.subtask_templates):
            # Erstelle Subtask aus Template
            subtask = SubtaskDefinition(
                subtask_id=f"{request.task_id}_fallback_{i+1}",
                name=self._render_template_string(template.get("name", f"Subtask {i+1}"), request),
                description=self._render_template_string(template.get("description", ""), request),
                task_type=TaskType(template.get("task_type", "agent_execution")),
                priority=TaskPriority(template.get("priority", request.task_priority.value)),
                payload=self._create_subtask_payload(template, request),
                estimated_duration_minutes=float(template.get("estimated_duration_minutes", 10.0)),
                required_capabilities=template.get("required_capabilities", ["general_processing"]),
                preferred_agent_types=template.get("preferred_agent_types", []),
                depends_on=template.get("depends_on", []),
                can_run_parallel=bool(template.get("can_run_parallel", True)),
                parallel_group=template.get("parallel_group"),
                success_criteria=template.get("success_criteria", []),
                validation_rules=template.get("validation_rules", {})
            )

            subtasks.append(subtask)

        return subtasks

    def _render_template_string(self, template: str, request: DecompositionRequest) -> str:
        """Rendert Template-String mit Request-Daten."""
        try:
            # Einfache Template-Variable-Ersetzung
            rendered = template.replace("{task_name}", request.task_name)
            rendered = rendered.replace("{task_description}", request.task_description)
            rendered = rendered.replace("{task_type}", request.task_type.value)
            rendered = rendered.replace("{priority}", request.task_priority.value)

            return rendered
        except Exception:
            return template

    def _create_subtask_payload(
        self,
        template: dict[str, Any],
        request: DecompositionRequest
    ) -> dict[str, Any]:
        """Erstellt Subtask-Payload aus Template."""
        payload = template.get("payload", {}).copy()

        # Füge Original-Task-Daten hinzu falls erforderlich
        if template.get("inherit_payload", False):
            payload.update(request.task_payload)

        # Template-Variablen ersetzen
        payload = self._render_payload_variables(payload, request)

        return payload

    def _render_payload_variables(
        self,
        payload: dict[str, Any],
        request: DecompositionRequest
    ) -> dict[str, Any]:
        """Ersetzt Template-Variablen in Payload."""
        rendered_payload = {}

        for key, value in payload.items():
            if isinstance(value, str):
                rendered_payload[key] = self._render_template_string(value, request)
            elif isinstance(value, dict):
                rendered_payload[key] = self._render_payload_variables(value, request)
            else:
                rendered_payload[key] = value

        return rendered_payload

    def _create_default_subtasks(self, request: DecompositionRequest) -> list[SubtaskDefinition]:
        """Erstellt Default-Subtasks als letzter Fallback."""
        logger.info(f"Erstelle Default-Subtasks für {request.task_id}")

        # Einfache Default-Decomposition
        subtasks = [
            SubtaskDefinition(
                subtask_id=f"{request.task_id}_default_1",
                name=f"Analyze {request.task_name}",
                description=f"Analyze and prepare {request.task_description}",
                task_type=TaskType.AGENT_EXECUTION,
                priority=request.task_priority,
                payload={"action": "analyze", "original_payload": request.task_payload},
                estimated_duration_minutes=5.0,
                required_capabilities=["general_processing"],
                can_run_parallel=False
            ),
            SubtaskDefinition(
                subtask_id=f"{request.task_id}_default_2",
                name=f"Execute {request.task_name}",
                description=f"Execute main logic for {request.task_description}",
                task_type=TaskType.AGENT_EXECUTION,
                priority=request.task_priority,
                payload=request.task_payload,
                estimated_duration_minutes=15.0,
                required_capabilities=["general_processing"],
                depends_on=[f"{request.task_id}_default_1"],
                can_run_parallel=False
            ),
            SubtaskDefinition(
                subtask_id=f"{request.task_id}_default_3",
                name=f"Finalize {request.task_name}",
                description=f"Finalize and validate results for {request.task_description}",
                task_type=TaskType.AGENT_EXECUTION,
                priority=request.task_priority,
                payload={"action": "finalize", "original_payload": request.task_payload},
                estimated_duration_minutes=5.0,
                required_capabilities=["general_processing"],
                depends_on=[f"{request.task_id}_default_2"],
                can_run_parallel=False
            )
        ]

        return subtasks

    def _load_default_rules(self) -> None:
        """Lädt Standard-Fallback-Regeln."""
        # Regel für Data Processing Tasks
        data_processing_rule = FallbackRule(
            rule_id="data_processing_fallback",
            name="Data Processing Fallback",
            description="Fallback für Data Processing Tasks",
            task_type_patterns=["data_processing", "batch_job", "etl"],
            complexity_range=(1.0, 10.0),
            keyword_triggers=["process", "transform", "analyze", "data"],
            subtask_templates=[
                {
                    "name": "Data Validation",
                    "description": "Validate input data for {task_name}",
                    "task_type": "data_processing",
                    "estimated_duration_minutes": 5.0,
                    "required_capabilities": ["data_validation"],
                    "can_run_parallel": False,
                    "payload": {"action": "validate", "inherit_payload": True}
                },
                {
                    "name": "Data Processing",
                    "description": "Process data for {task_name}",
                    "task_type": "data_processing",
                    "estimated_duration_minutes": 20.0,
                    "required_capabilities": ["data_processing"],
                    "depends_on": ["{task_id}_fallback_1"],
                    "can_run_parallel": False,
                    "payload": {"action": "process", "inherit_payload": True}
                },
                {
                    "name": "Result Export",
                    "description": "Export processed results for {task_name}",
                    "task_type": "data_processing",
                    "estimated_duration_minutes": 3.0,
                    "required_capabilities": ["data_export"],
                    "depends_on": ["{task_id}_fallback_2"],
                    "can_run_parallel": False,
                    "payload": {"action": "export", "inherit_payload": True}
                }
            ],
            default_strategy=DecompositionStrategy.SEQUENTIAL,
            priority=10
        )

        # Regel für NLP Analysis Tasks
        nlp_analysis_rule = FallbackRule(
            rule_id="nlp_analysis_fallback",
            name="NLP Analysis Fallback",
            description="Fallback für NLP Analysis Tasks",
            task_type_patterns=["nlp_analysis", "text_processing"],
            complexity_range=(1.0, 10.0),
            keyword_triggers=["text", "nlp", "language", "sentiment", "extract"],
            subtask_templates=[
                {
                    "name": "Text Preprocessing",
                    "description": "Preprocess text for {task_name}",
                    "task_type": "nlp_analysis",
                    "estimated_duration_minutes": 3.0,
                    "required_capabilities": ["text_preprocessing"],
                    "can_run_parallel": True,
                    "parallel_group": "preprocessing",
                    "payload": {"action": "preprocess", "inherit_payload": True}
                },
                {
                    "name": "NLP Analysis",
                    "description": "Perform NLP analysis for {task_name}",
                    "task_type": "nlp_analysis",
                    "estimated_duration_minutes": 15.0,
                    "required_capabilities": ["nlp_analysis"],
                    "depends_on": ["{task_id}_fallback_1"],
                    "can_run_parallel": False,
                    "payload": {"action": "analyze", "inherit_payload": True}
                },
                {
                    "name": "Result Formatting",
                    "description": "Format analysis results for {task_name}",
                    "task_type": "nlp_analysis",
                    "estimated_duration_minutes": 2.0,
                    "required_capabilities": ["result_formatting"],
                    "depends_on": ["{task_id}_fallback_2"],
                    "can_run_parallel": False,
                    "payload": {"action": "format", "inherit_payload": True}
                }
            ],
            default_strategy=DecompositionStrategy.PIPELINE,
            priority=20
        )

        # Regel für Agent Execution Tasks
        agent_execution_rule = FallbackRule(
            rule_id="agent_execution_fallback",
            name="Agent Execution Fallback",
            description="Fallback für allgemeine Agent Execution Tasks",
            task_type_patterns=["agent_execution", "workflow", "automation"],
            complexity_range=(1.0, 10.0),
            keyword_triggers=["execute", "run", "perform", "automate"],
            subtask_templates=[
                {
                    "name": "Task Preparation",
                    "description": "Prepare execution environment for {task_name}",
                    "task_type": "agent_execution",
                    "estimated_duration_minutes": 2.0,
                    "required_capabilities": ["task_preparation"],
                    "can_run_parallel": False,
                    "payload": {"action": "prepare", "inherit_payload": True}
                },
                {
                    "name": "Main Execution",
                    "description": "Execute main task logic for {task_name}",
                    "task_type": "agent_execution",
                    "estimated_duration_minutes": 10.0,
                    "required_capabilities": ["general_processing"],
                    "depends_on": ["{task_id}_fallback_1"],
                    "can_run_parallel": False,
                    "payload": {"action": "execute", "inherit_payload": True}
                },
                {
                    "name": "Cleanup",
                    "description": "Cleanup and finalize for {task_name}",
                    "task_type": "agent_execution",
                    "estimated_duration_minutes": 1.0,
                    "required_capabilities": ["cleanup"],
                    "depends_on": ["{task_id}_fallback_2"],
                    "can_run_parallel": False,
                    "payload": {"action": "cleanup", "inherit_payload": True}
                }
            ],
            default_strategy=DecompositionStrategy.SEQUENTIAL,
            priority=100  # Niedrigste Priorität (Default)
        )

        self.rules = [data_processing_rule, nlp_analysis_rule, agent_execution_rule]

    def add_rule(self, rule: FallbackRule) -> None:
        """Fügt neue Fallback-Regel hinzu."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

        logger.info(f"Fallback-Regel hinzugefügt: {rule.rule_id}")

    def get_rule_statistics(self) -> dict[str, Any]:
        """Gibt Regel-Statistiken zurück."""
        return {
            "total_rules": len(self.rules),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "usage_count": rule.usage_count,
                    "last_used": rule.last_used.isoformat() if rule.last_used else None,
                    "priority": rule.priority
                }
                for rule in self.rules
            ]
        }
