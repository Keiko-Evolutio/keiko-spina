# backend/services/task_decomposition/plan_validator.py
"""Plan-Validation für Decomposition-Pläne.

Validiert generierte Decomposition-Pläne auf Vollständigkeit,
Konsistenz und Ausführbarkeit.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

from .data_models import DecompositionPlan, DecompositionRequest, ValidationResult

logger = get_logger(__name__)


class PlanValidator:
    """Validator für Decomposition-Pläne."""

    def __init__(self):
        """Initialisiert Plan Validator."""
        # Validation-Konfiguration
        self.max_subtasks = 50
        self.max_dependency_depth = 10
        self.min_subtask_duration_minutes = 0.5
        self.max_subtask_duration_minutes = 120.0

        logger.info("Plan Validator initialisiert")

    async def validate_plan(
        self,
        plan: DecompositionPlan,
        request: DecompositionRequest
    ) -> ValidationResult:
        """Validiert Decomposition-Plan.

        Args:
            plan: Zu validierender Plan
            request: Original-Request

        Returns:
            Validation-Ergebnis
        """
        try:
            validation_result = ValidationResult(is_valid=True, validation_score=1.0)

            # 1. Completeness Check
            completeness_result = await self._check_completeness(plan, request)
            validation_result.completeness_check = completeness_result["valid"]
            if not completeness_result["valid"]:
                validation_result.critical_issues.extend(completeness_result["issues"])

            # 2. Dependency Check
            dependency_result = await self._check_dependencies(plan)
            validation_result.dependency_check = dependency_result["valid"]
            if not dependency_result["valid"]:
                validation_result.critical_issues.extend(dependency_result["issues"])

            # 3. Capability Check
            capability_result = await self._check_capabilities(plan)
            validation_result.capability_check = capability_result["valid"]
            if not capability_result["valid"]:
                validation_result.warnings.extend(capability_result["issues"])

            # 4. Resource Check
            resource_result = await self._check_resources(plan, request)
            validation_result.resource_check = resource_result["valid"]
            if not resource_result["valid"]:
                validation_result.warnings.extend(resource_result["issues"])

            # 5. Timing Check
            timing_result = await self._check_timing(plan, request)
            validation_result.timing_check = timing_result["valid"]
            if not timing_result["valid"]:
                validation_result.warnings.extend(timing_result["issues"])

            # Gesamtvalidierung
            validation_result.is_valid = (
                validation_result.completeness_check and
                validation_result.dependency_check and
                len(validation_result.critical_issues) == 0
            )

            # Validation-Score berechnen
            validation_result.validation_score = self._calculate_validation_score(validation_result)

            # Auto-Fixes vorschlagen
            auto_fixes = await self._suggest_auto_fixes(plan, validation_result)
            validation_result.auto_fixes_applied = auto_fixes

            logger.info({
                "event": "plan_validation_completed",
                "plan_id": plan.plan_id,
                "is_valid": validation_result.is_valid,
                "validation_score": validation_result.validation_score,
                "critical_issues": len(validation_result.critical_issues),
                "warnings": len(validation_result.warnings)
            })

            return validation_result

        except Exception as e:
            logger.error(f"Plan-Validation fehlgeschlagen: {e}")
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                critical_issues=[f"Validation-Fehler: {e!s}"]
            )

    async def _check_completeness(
        self,
        plan: DecompositionPlan,
        request: DecompositionRequest
    ) -> dict[str, Any]:
        """Prüft Vollständigkeit des Plans."""
        issues = []

        # Subtasks vorhanden?
        if not plan.subtasks:
            issues.append("Keine Subtasks im Plan")

        # Execution-Order vorhanden?
        if not plan.execution_order:
            issues.append("Keine Execution-Order definiert")

        # Alle Subtasks in Execution-Order enthalten?
        subtask_ids = {st.subtask_id for st in plan.subtasks}
        execution_ids = set()
        for group in plan.execution_order:
            execution_ids.update(group)

        missing_in_execution = subtask_ids - execution_ids
        if missing_in_execution:
            issues.append(f"Subtasks nicht in Execution-Order: {missing_in_execution}")

        extra_in_execution = execution_ids - subtask_ids
        if extra_in_execution:
            issues.append(f"Unbekannte IDs in Execution-Order: {extra_in_execution}")

        # Plan-Metadaten vollständig?
        if not plan.plan_id:
            issues.append("Plan-ID fehlt")

        if not plan.original_task_id:
            issues.append("Original-Task-ID fehlt")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    async def _check_dependencies(self, plan: DecompositionPlan) -> dict[str, Any]:
        """Prüft Dependency-Konsistenz."""
        issues = []

        # Sammle alle Subtask-IDs
        subtask_ids = {st.subtask_id for st in plan.subtasks}

        # Prüfe Dependencies
        for subtask in plan.subtasks:
            for dep_id in subtask.depends_on:
                if dep_id not in subtask_ids:
                    issues.append(f"Subtask {subtask.subtask_id} hängt von unbekanntem Subtask {dep_id} ab")

        # Prüfe auf zirkuläre Dependencies
        circular_deps = self._detect_circular_dependencies(plan.subtasks)
        if circular_deps:
            issues.append(f"Zirkuläre Dependencies erkannt: {circular_deps}")

        # Prüfe Dependency-Tiefe
        max_depth = self._calculate_max_dependency_depth(plan.subtasks)
        if max_depth > self.max_dependency_depth:
            issues.append(f"Dependency-Tiefe zu hoch: {max_depth} > {self.max_dependency_depth}")

        # Prüfe Dependency-Graph-Konsistenz
        if plan.dependency_graph:
            graph_issues = self._validate_dependency_graph(plan.dependency_graph, subtask_ids)
            issues.extend(graph_issues)

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    async def _check_capabilities(self, plan: DecompositionPlan) -> dict[str, Any]:
        """Prüft Capability-Anforderungen."""
        issues = []

        for subtask in plan.subtasks:
            # Capabilities definiert?
            if not subtask.required_capabilities:
                issues.append(f"Subtask {subtask.subtask_id} hat keine Required-Capabilities")

            # Agent-Assignment vorhanden und kompatibel?
            if subtask.subtask_id in plan.agent_assignments:
                agent_match = plan.agent_assignments[subtask.subtask_id]

                # Prüfe Capability-Coverage
                if agent_match.capability_coverage < 0.7:  # 70% Mindest-Coverage
                    issues.append(
                        f"Agent {agent_match.agent_id} für Subtask {subtask.subtask_id} "
                        f"hat unzureichende Capability-Coverage: {agent_match.capability_coverage:.2f}"
                    )

                # Prüfe Missing-Capabilities
                if agent_match.missing_capabilities:
                    issues.append(
                        f"Agent {agent_match.agent_id} für Subtask {subtask.subtask_id} "
                        f"fehlen Capabilities: {agent_match.missing_capabilities}"
                    )

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    async def _check_resources(
        self,
        plan: DecompositionPlan,
        request: DecompositionRequest
    ) -> dict[str, Any]:
        """Prüft Resource-Anforderungen."""
        issues = []

        # Prüfe Anzahl Subtasks
        if len(plan.subtasks) > self.max_subtasks:
            issues.append(f"Zu viele Subtasks: {len(plan.subtasks)} > {self.max_subtasks}")

        # Prüfe Request-Constraints
        if request.max_subtasks and len(plan.subtasks) > request.max_subtasks:
            issues.append(f"Subtask-Limit überschritten: {len(plan.subtasks)} > {request.max_subtasks}")

        # Prüfe parallele Subtasks
        max_parallel = max(len(group) for group in plan.execution_order) if plan.execution_order else 0
        if request.max_parallel_subtasks and max_parallel > request.max_parallel_subtasks:
            issues.append(f"Parallel-Limit überschritten: {max_parallel} > {request.max_parallel_subtasks}")

        # Prüfe Resource-Constraints
        if request.resource_constraints:
            for constraint, limit in request.resource_constraints.items():
                # TODO: Implementiere spezifische Resource-Checks - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
                pass

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    async def _check_timing(
        self,
        plan: DecompositionPlan,
        request: DecompositionRequest
    ) -> dict[str, Any]:
        """Prüft Timing-Aspekte."""
        issues = []

        # Prüfe Subtask-Dauern
        for subtask in plan.subtasks:
            duration = subtask.estimated_duration_minutes

            if duration < self.min_subtask_duration_minutes:
                issues.append(
                    f"Subtask {subtask.subtask_id} zu kurz: "
                    f"{duration} < {self.min_subtask_duration_minutes} Minuten"
                )

            if duration > self.max_subtask_duration_minutes:
                issues.append(
                    f"Subtask {subtask.subtask_id} zu lang: "
                    f"{duration} > {self.max_subtask_duration_minutes} Minuten"
                )

        # Prüfe Deadline-Einhaltung
        if request.deadline:
            estimated_completion = request.request_timestamp
            # TODO: Berechne geschätzte Completion-Zeit basierend auf Plan - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115

            # Vereinfachte Prüfung
            if plan.estimated_parallel_duration_minutes > 60:  # > 1 Stunde
                issues.append("Plan-Dauer möglicherweise zu lang für Deadline")

        # Prüfe Parallelisierungs-Effizienz
        if plan.parallelization_efficiency < 0.1:  # < 10% Effizienz
            issues.append(f"Niedrige Parallelisierungs-Effizienz: {plan.parallelization_efficiency:.2f}")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _detect_circular_dependencies(self, subtasks) -> list[str]:
        """Erkennt zirkuläre Dependencies."""
        # Vereinfachte Zirkularitäts-Erkennung
        visited = set()
        rec_stack = set()

        def has_cycle(subtask_id: str, deps_map: dict[str, list[str]]) -> bool:
            visited.add(subtask_id)
            rec_stack.add(subtask_id)

            for dep_id in deps_map.get(subtask_id, []):
                if dep_id not in visited:
                    if has_cycle(dep_id, deps_map):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(subtask_id)
            return False

        # Erstelle Dependency-Map
        deps_map = {st.subtask_id: st.depends_on for st in subtasks}

        # Prüfe auf Zyklen
        cycles = []
        for subtask in subtasks:
            if subtask.subtask_id not in visited:
                if has_cycle(subtask.subtask_id, deps_map):
                    cycles.append(subtask.subtask_id)

        return cycles

    def _calculate_max_dependency_depth(self, subtasks) -> int:
        """Berechnet maximale Dependency-Tiefe."""
        # Vereinfachte Tiefenberechnung
        deps_map = {st.subtask_id: st.depends_on for st in subtasks}

        def get_depth(subtask_id: str, visited: set[str] = None) -> int:
            if visited is None:
                visited = set()

            if subtask_id in visited:
                return 0  # Zirkuläre Dependency

            visited.add(subtask_id)

            max_dep_depth = 0
            for dep_id in deps_map.get(subtask_id, []):
                dep_depth = get_depth(dep_id, visited.copy())
                max_dep_depth = max(max_dep_depth, dep_depth)

            return max_dep_depth + 1

        max_depth = 0
        for subtask in subtasks:
            depth = get_depth(subtask.subtask_id)
            max_depth = max(max_depth, depth)

        return max_depth

    def _validate_dependency_graph(
        self,
        dependency_graph: dict[str, list[str]],
        subtask_ids: set[str]
    ) -> list[str]:
        """Validiert Dependency-Graph."""
        issues = []

        for subtask_id, dependencies in dependency_graph.items():
            if subtask_id not in subtask_ids:
                issues.append(f"Dependency-Graph enthält unbekannte Subtask-ID: {subtask_id}")

            for dep_id in dependencies:
                if dep_id not in subtask_ids:
                    issues.append(f"Dependency-Graph referenziert unbekannte Dependency: {dep_id}")

        return issues

    def _calculate_validation_score(self, validation_result: ValidationResult) -> float:
        """Berechnet Validation-Score."""
        score = 1.0

        # Kritische Issues reduzieren Score stark
        score -= len(validation_result.critical_issues) * 0.3

        # Warnings reduzieren Score moderat
        score -= len(validation_result.warnings) * 0.1

        # Einzelne Checks gewichten
        checks = [
            validation_result.completeness_check,
            validation_result.dependency_check,
            validation_result.capability_check,
            validation_result.resource_check,
            validation_result.timing_check
        ]

        passed_checks = sum(1 for check in checks if check)
        check_score = passed_checks / len(checks)

        # Kombiniere Scores
        final_score = (score * 0.7) + (check_score * 0.3)

        return max(0.0, min(1.0, final_score))

    async def _suggest_auto_fixes(
        self,
        plan: DecompositionPlan,
        validation_result: ValidationResult
    ) -> list[str]:
        """Schlägt automatische Fixes vor."""
        auto_fixes = []

        # TODO: Implementiere Auto-Fix-Logic - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/115
        # Beispiele:
        # - Fehlende Dependencies hinzufügen
        # - Execution-Order korrigieren
        # - Resource-Limits anpassen

        return auto_fixes
