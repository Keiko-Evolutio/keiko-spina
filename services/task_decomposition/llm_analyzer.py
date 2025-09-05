# backend/services/task_decomposition/llm_analyzer.py
"""LLM-basierte Task-Analyse für intelligente Decomposition.

Nutzt LLM Client aus TASK 1 für Task-Komplexitätsbewertung,
Decomposition-Strategien und Subtask-Generierung.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from kei_logging import get_logger, log_orchestrator_step, training_trace
from services.llm import get_llm_client, get_template_manager
from services.llm.llm_client import LLMRequest

from .data_models import (
    ComplexityLevel,
    DecompositionRequest,
    DecompositionStrategy,
    SubtaskDefinition,
    TaskAnalysis,
)

logger = get_logger(__name__)


class LLMTaskAnalyzer:
    """LLM-basierte Task-Analyse und Decomposition."""

    def __init__(self):
        """Initialisiert LLM Task Analyzer."""
        self.llm_client = get_llm_client()
        self.template_manager = get_template_manager()

        # Konfiguration (Issue #55 Performance Targets)
        self.analysis_model = os.getenv("KEI_ORCHESTRATOR_LLM_MODEL", "gpt-4")
        self.max_tokens = int(os.getenv("KEI_ORCHESTRATOR_LLM_MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("KEI_ORCHESTRATOR_LLM_TEMPERATURE", "0.1"))  # Niedrige Temperatur für konsistente Ergebnisse
        self.analysis_timeout_seconds = float(os.getenv("KEI_ORCHESTRATOR_TASK_ANALYSIS_TIMEOUT_SECONDS", "2.0"))

        # Performance-Tracking
        self._analysis_count = 0
        self._total_analysis_time_ms = 0.0

        logger.info("LLM Task Analyzer initialisiert")

    @training_trace(context={"component": "llm_analyzer", "phase": "task_analysis"})
    async def analyze_task(self, request: DecompositionRequest) -> TaskAnalysis:
        """Analysiert Task mit LLM für Komplexität und Decomposition-Potential.

        Args:
            request: Decomposition-Request

        Returns:
            Task-Analyse-Ergebnis

        Raises:
            Exception: Bei LLM-Analyse-Fehlern
        """
        start_time = time.time()

        try:
            # Schritt 1: LLM-Prompt für Task-Analyse erstellen
            log_orchestrator_step(
                "Creating LLM Analysis Prompt",
                "llm_call",
                task_id=request.task_id,
                task_type=request.task_type.value,
                task_description_length=len(request.task_description),
                model=self.analysis_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            analysis_prompt = await self._create_analysis_prompt(request)

            # LLM-Request
            llm_request = LLMRequest(
                model=self.analysis_model,
                messages=[
                    {"role": "system", "content": self._get_analysis_system_prompt()},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                user_id=request.user_id,
                session_id=request.session_id
            )

            # Schritt 2: LLM-Analyse ausführen
            log_orchestrator_step(
                "Executing LLM Task Analysis",
                "llm_call",
                prompt_length=len(analysis_prompt),
                max_tokens=self.max_tokens,
                timeout_seconds=self.analysis_timeout_seconds
            )

            llm_response = await self.llm_client.chat_completion(llm_request)

            # Schritt 3: LLM-Response parsen
            log_orchestrator_step(
                "Parsing LLM Analysis Response",
                "llm_call",
                response_length=len(llm_response.content),
                cost_usd=getattr(llm_response, "cost_usd", 0.0),
                tokens_used=getattr(llm_response, "tokens_used", 0),
                prompt_tokens=getattr(llm_response, "prompt_tokens", 0),
                completion_tokens=getattr(llm_response, "completion_tokens", 0),
                model_used=getattr(llm_response, "model", self.analysis_model)
            )

            analysis = await self._parse_analysis_response(llm_response.content, request)

            # Schritt 4: Analyse-Ergebnis loggen
            log_orchestrator_step(
                "Task Analysis Completed",
                "task_decomposition",
                complexity_score=analysis.complexity_score,
                complexity_level=analysis.complexity_level.value,
                is_decomposable=analysis.is_decomposable,
                recommended_strategy=analysis.recommended_strategy.value,
                required_capabilities=analysis.required_capabilities,
                estimated_duration_minutes=analysis.estimated_duration_minutes,
                decomposition_confidence=analysis.decomposition_confidence
            )

            # Performance-Tracking
            analysis_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(analysis_time_ms)

            logger.info({
                "event": "llm_task_analysis_completed",
                "task_id": request.task_id,
                "complexity_score": analysis.complexity_score,
                "decomposable": analysis.is_decomposable,
                "analysis_time_ms": analysis_time_ms,
                "model": self.analysis_model
            })

            return analysis

        except Exception as e:
            logger.error(f"LLM Task-Analyse fehlgeschlagen: {e}")
            raise

    @training_trace(context={"component": "llm_analyzer", "phase": "subtask_generation"})
    async def generate_subtasks(
        self,
        request: DecompositionRequest,
        analysis: TaskAnalysis
    ) -> list[SubtaskDefinition]:
        """Generiert Subtasks basierend auf LLM-Analyse.

        Args:
            request: Decomposition-Request
            analysis: Task-Analyse-Ergebnis

        Returns:
            Liste von Subtask-Definitionen
        """
        if not analysis.is_decomposable:
            logger.info(f"Task {request.task_id} ist nicht decomposable")
            return []

        try:
            # Schritt 1: LLM-Prompt für Subtask-Generierung erstellen
            log_orchestrator_step(
                "Creating Subtask Generation Prompt",
                "task_decomposition",
                complexity_score=analysis.complexity_score,
                recommended_strategy=analysis.recommended_strategy.value,
                max_parallel_subtasks=analysis.max_parallel_subtasks,
                required_capabilities=analysis.required_capabilities,
                estimated_duration_minutes=analysis.estimated_duration_minutes
            )

            subtask_prompt = await self._create_subtask_prompt(request, analysis)

            # LLM-Request
            llm_request = LLMRequest(
                model=self.analysis_model,
                messages=[
                    {"role": "system", "content": self._get_subtask_system_prompt()},
                    {"role": "user", "content": subtask_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                user_id=request.user_id,
                session_id=request.session_id
            )

            # Schritt 2: LLM-Subtask-Generierung ausführen
            log_orchestrator_step(
                "Executing LLM Subtask Generation",
                "llm_call",
                prompt_length=len(subtask_prompt),
                model=self.analysis_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            llm_response = await self.llm_client.chat_completion(llm_request)

            # Schritt 3: Response parsen
            log_orchestrator_step(
                "Parsing Subtask Generation Response",
                "llm_call",
                response_length=len(llm_response.content),
                cost_usd=getattr(llm_response, "cost_usd", 0.0),
                tokens_used=getattr(llm_response, "tokens_used", 0)
            )

            subtasks = await self._parse_subtasks_response(llm_response.content, request)

            # Schritt 4: Subtask-Generierung abgeschlossen
            log_orchestrator_step(
                "Subtask Generation Completed",
                "task_decomposition",
                subtask_count=len(subtasks),
                strategy_used=analysis.recommended_strategy.value,
                total_estimated_duration=sum(st.estimated_duration_minutes for st in subtasks)
            )

            logger.info({
                "event": "llm_subtasks_generated",
                "task_id": request.task_id,
                "subtask_count": len(subtasks),
                "strategy": analysis.recommended_strategy.value
            })

            return subtasks

        except Exception as e:
            logger.error(f"LLM Subtask-Generierung fehlgeschlagen: {e}")
            return []

    def _get_analysis_system_prompt(self) -> str:
        """Gibt System-Prompt für Task-Analyse zurück."""
        return """Du bist ein Experte für Task-Analyse und Workflow-Optimierung.

Analysiere die gegebene Task und bewerte:
1. Komplexität (1-10 Skala)
2. Decomposition-Potential
3. Erforderliche Capabilities
4. Resource-Anforderungen
5. Parallelisierungs-Möglichkeiten
6. Risk-Faktoren

Antworte IMMER im JSON-Format mit allen erforderlichen Feldern.
Sei präzise und konsistent in deinen Bewertungen."""

    def _get_subtask_system_prompt(self) -> str:
        """Gibt System-Prompt für Subtask-Generierung zurück."""
        return """Du bist ein Experte für Task-Decomposition und Workflow-Design.

Zerlege die gegebene Task in optimale Subtasks mit:
1. Klaren Abhängigkeiten
2. Parallelisierungs-Möglichkeiten
3. Spezifischen Capability-Anforderungen
4. Realistischen Zeitschätzungen
5. Validierungs-Kriterien

Antworte IMMER im JSON-Format mit strukturierten Subtask-Definitionen.
Optimiere für Performance und Zuverlässigkeit."""

    async def _create_analysis_prompt(self, request: DecompositionRequest) -> str:
        """Erstellt Prompt für Task-Analyse."""
        # Nutze Template Manager für strukturierte Prompts
        try:
            return self.template_manager.render_template(
                "task_analysis",
                {
                    "task_description": f"{request.task_name}: {request.task_description}",
                    "context": f"Task Type: {request.task_type.value}, Priority: {request.task_priority.value}, Payload: {json.dumps(request.task_payload, indent=2)}",
                    "constraints": json.dumps({
                        "max_subtasks": request.max_subtasks,
                        "max_parallel_subtasks": request.max_parallel_subtasks,
                        "deadline": request.deadline.isoformat() if request.deadline else "None",
                        "resource_constraints": request.resource_constraints,
                        "available_agents": request.available_agents
                    }, indent=2),
                    "sla_requirements": json.dumps({
                        "enable_llm_analysis": request.enable_llm_analysis,
                        "enable_performance_prediction": request.enable_performance_prediction,
                        "enable_agent_matching": request.enable_agent_matching,
                        "preferred_strategy": request.preferred_strategy.value if request.preferred_strategy else "auto"
                    }, indent=2)
                }
            )
        except Exception as e:
            logger.warning(f"Template-Rendering fehlgeschlagen, verwende Fallback: {e}")
            # Fallback auf manuellen Prompt
            return f"""Analysiere diese Task für intelligente Decomposition:

Task: {request.task_name}
Beschreibung: {request.task_description}
Typ: {request.task_type.value}
Priorität: {request.task_priority.value}
Payload: {json.dumps(request.task_payload, indent=2)}
Constraints: {json.dumps(request.resource_constraints, indent=2)}
Max Subtasks: {request.max_subtasks}
Max Parallel Subtasks: {request.max_parallel_subtasks}
Deadline: {request.deadline.isoformat() if request.deadline else "None"}
Verfügbare Agents: {request.available_agents}

Bitte analysiere diese Task und gib eine strukturierte JSON-Antwort mit:
{{
  "complexity_score": <1-10>,
  "complexity_level": "<trivial|simple|moderate|complex|critical>",
  "estimated_duration_minutes": <float>,
  "is_decomposable": <true|false>,
  "recommended_strategy": "<sequential|parallel|pipeline|hybrid>",
  "decomposition_confidence": <0-1>,
  "required_capabilities": [<list of strings>],
  "optional_capabilities": [<list of strings>],
  "specialized_skills": [<list of strings>],
  "estimated_cpu_usage": <0-1>,
  "estimated_memory_mb": <int>,
  "estimated_network_io": <true|false>,
  "estimated_disk_io": <true|false>,
  "parallel_potential": <0-1>,
  "max_parallel_subtasks": <int>,
  "bottleneck_factors": [<list of strings>],
  "risk_factors": [<list of strings>],
  "failure_probability": <0-1>,
  "rollback_complexity": <0-1>,
  "analysis_confidence": <0-1>
}}

Wichtig: Antworte nur mit gültigem JSON, keine zusätzlichen Erklärungen."""

    async def _create_subtask_prompt(
        self,
        request: DecompositionRequest,
        analysis: TaskAnalysis
    ) -> str:
        """Erstellt Prompt für Subtask-Generierung."""
        try:
            return self.template_manager.render_template(
                "task_decomposition",
                {
                    "task_description": f"{request.task_name}: {request.task_description}",
                    "complexity_score": analysis.complexity_score,
                    "required_capabilities": analysis.required_capabilities,
                    "recommended_strategy": analysis.recommended_strategy.value,
                    "max_parallel_subtasks": analysis.max_parallel_subtasks,
                    "task_payload": json.dumps(request.task_payload, indent=2)
                }
            )
        except Exception:
            # Fallback auf manuellen Prompt
            return f"""Zerlege diese Task in optimale Subtasks:

Original Task: {request.task_name}
Beschreibung: {request.task_description}
Komplexität: {analysis.complexity_score}/10
Strategie: {analysis.recommended_strategy.value}
Max Parallel: {analysis.max_parallel_subtasks}

Erstelle Subtasks mit:
1. Eindeutigen Namen und Beschreibungen
2. Klaren Dependencies
3. Spezifischen Capabilities
4. Realistischen Zeitschätzungen
5. Parallelisierungs-Informationen

Antworte im JSON-Format:
{{
    "subtasks": [
        {{
            "name": "Subtask 1",
            "description": "Beschreibung",
            "task_type": "agent_execution",
            "priority": "normal",
            "estimated_duration_minutes": 10.0,
            "required_capabilities": ["capability1"],
            "depends_on": [],
            "can_run_parallel": true,
            "success_criteria": ["criteria1"]
        }}
    ]
}}"""

    async def _parse_analysis_response(
        self,
        response_content: str,
        request: DecompositionRequest
    ) -> TaskAnalysis:
        """Parst LLM-Response für Task-Analyse."""
        try:
            # JSON aus Response extrahieren
            response_data = self._extract_json_from_response(response_content)

            # Komplexitäts-Level bestimmen
            complexity_score = float(response_data.get("complexity_score", 5.0))
            complexity_level = self._determine_complexity_level(complexity_score)

            # Decomposition-Strategie
            strategy_str = response_data.get("recommended_strategy", "sequential")
            strategy = DecompositionStrategy(strategy_str)

            return TaskAnalysis(
                complexity_score=complexity_score,
                complexity_level=complexity_level,
                estimated_duration_minutes=float(response_data.get("estimated_duration_minutes", 30.0)),
                is_decomposable=bool(response_data.get("is_decomposable", True)),
                recommended_strategy=strategy,
                decomposition_confidence=float(response_data.get("decomposition_confidence", 0.8)),
                required_capabilities=response_data.get("required_capabilities", []),
                optional_capabilities=response_data.get("optional_capabilities", []),
                specialized_skills=response_data.get("specialized_skills", []),
                estimated_cpu_usage=float(response_data.get("estimated_cpu_usage", 0.5)),
                estimated_memory_mb=int(response_data.get("estimated_memory_mb", 512)),
                estimated_network_io=bool(response_data.get("estimated_network_io", False)),
                estimated_disk_io=bool(response_data.get("estimated_disk_io", False)),
                parallel_potential=float(response_data.get("parallel_potential", 0.5)),
                max_parallel_subtasks=int(response_data.get("max_parallel_subtasks", 3)),
                bottleneck_factors=response_data.get("bottleneck_factors", []),
                risk_factors=response_data.get("risk_factors", []),
                failure_probability=float(response_data.get("failure_probability", 0.1)),
                rollback_complexity=float(response_data.get("rollback_complexity", 0.3)),
                analysis_model=self.analysis_model,
                analysis_confidence=float(response_data.get("analysis_confidence", 0.8))
            )

        except Exception as e:
            logger.error(f"Fehler beim Parsen der LLM-Analyse: {e}")
            # Fallback auf Default-Analyse
            return self._create_fallback_analysis(request)

    async def _parse_subtasks_response(
        self,
        response_content: str,
        request: DecompositionRequest
    ) -> list[SubtaskDefinition]:
        """Parst LLM-Response für Subtasks."""
        try:
            response_data = self._extract_json_from_response(response_content)
            subtasks_data = response_data.get("subtasks", [])

            subtasks = []
            for i, subtask_data in enumerate(subtasks_data):
                from task_management.core_task_manager import TaskPriority, TaskType

                subtask = SubtaskDefinition(
                    subtask_id=f"{request.task_id}_subtask_{i+1}",
                    name=subtask_data.get("name", f"Subtask {i+1}"),
                    description=subtask_data.get("description", ""),
                    task_type=TaskType(subtask_data.get("task_type", "agent_execution")),
                    priority=TaskPriority(subtask_data.get("priority", "normal")),
                    payload=subtask_data.get("payload", {}),
                    estimated_duration_minutes=float(subtask_data.get("estimated_duration_minutes", 10.0)),
                    required_capabilities=subtask_data.get("required_capabilities", []),
                    preferred_agent_types=subtask_data.get("preferred_agent_types", []),
                    depends_on=subtask_data.get("depends_on", []),
                    can_run_parallel=bool(subtask_data.get("can_run_parallel", True)),
                    parallel_group=subtask_data.get("parallel_group"),
                    success_criteria=subtask_data.get("success_criteria", []),
                    validation_rules=subtask_data.get("validation_rules", {})
                )

                subtasks.append(subtask)

            return subtasks

        except Exception as e:
            logger.error(f"Fehler beim Parsen der Subtasks: {e}")
            return []

    def _extract_json_from_response(self, response_content: str) -> dict[str, Any]:
        """Extrahiert JSON aus LLM-Response."""
        try:
            # Versuche direktes JSON-Parsing
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Suche nach JSON-Block in Response
            import re
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Fallback auf leeres Dict
            logger.warning("Konnte kein JSON aus LLM-Response extrahieren")
            return {}

    def _determine_complexity_level(self, complexity_score: float) -> ComplexityLevel:
        """Bestimmt Komplexitäts-Level aus Score."""
        if complexity_score <= 2:
            return ComplexityLevel.TRIVIAL
        if complexity_score <= 4:
            return ComplexityLevel.SIMPLE
        if complexity_score <= 6:
            return ComplexityLevel.MODERATE
        if complexity_score <= 8:
            return ComplexityLevel.COMPLEX
        return ComplexityLevel.CRITICAL

    def _create_fallback_analysis(self, request: DecompositionRequest) -> TaskAnalysis:
        """Erstellt Fallback-Analyse bei LLM-Fehlern."""
        return TaskAnalysis(
            complexity_score=5.0,
            complexity_level=ComplexityLevel.MODERATE,
            estimated_duration_minutes=30.0,
            is_decomposable=True,
            recommended_strategy=DecompositionStrategy.SEQUENTIAL,
            decomposition_confidence=0.5,
            required_capabilities=["general_processing"],
            optional_capabilities=[],
            specialized_skills=[],
            estimated_cpu_usage=0.5,
            estimated_memory_mb=512,
            estimated_network_io=False,
            estimated_disk_io=False,
            parallel_potential=0.3,
            max_parallel_subtasks=2,
            bottleneck_factors=["unknown"],
            risk_factors=["llm_analysis_failed"],
            failure_probability=0.2,
            rollback_complexity=0.5,
            analysis_model="fallback",
            analysis_confidence=0.5
        )

    def _update_performance_stats(self, analysis_time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken."""
        self._analysis_count += 1
        self._total_analysis_time_ms += analysis_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_time = (
            self._total_analysis_time_ms / self._analysis_count
            if self._analysis_count > 0 else 0.0
        )

        return {
            "total_analyses": self._analysis_count,
            "avg_analysis_time_ms": avg_time,
            "total_analysis_time_ms": self._total_analysis_time_ms,
            "model_used": self.analysis_model
        }
