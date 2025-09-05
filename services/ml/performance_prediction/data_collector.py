# backend/services/ml/performance_prediction/data_collector.py
"""Performance-Datensammlung für ML-Pipeline.

Sammelt historische Performance-Daten aus verschiedenen Quellen
und bereitet sie für ML-Training vor.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from agents.monitoring.performance_monitor import PerformanceMonitor
from agents.registry.dynamic_registry import DynamicAgentRegistry
from kei_logging import get_logger
from task_management.core_task_manager import TaskManager

from .data_models import AgentCharacteristics, PerformanceDataPoint, TaskCharacteristics

logger = get_logger(__name__)


class PerformanceDataCollector:
    """Sammelt Performance-Daten für ML-Training."""

    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        agent_registry: DynamicAgentRegistry,
        task_manager: TaskManager
    ):
        """Initialisiert Data Collector.

        Args:
            performance_monitor: Performance Monitor für Metriken
            agent_registry: Agent Registry für Agent-Informationen
            task_manager: Task Manager für Task-Informationen
        """
        self.performance_monitor = performance_monitor
        self.agent_registry = agent_registry
        self.task_manager = task_manager

        # Daten-Cache
        self._data_points: list[PerformanceDataPoint] = []
        self._collection_start_time = datetime.utcnow()

        # Konfiguration
        self.min_data_points = 1000  # Minimum für Training
        self.max_data_points = 50000  # Maximum im Memory
        self.collection_interval_seconds = 60  # 1 Minute

        logger.info({
            "event": "data_collector_initialized",
            "min_data_points": self.min_data_points,
            "max_data_points": self.max_data_points
        })

    async def start_collection(self) -> None:
        """Startet kontinuierliche Datensammlung."""
        logger.info("Starte Performance-Datensammlung...")

        while True:
            try:
                await self._collect_current_data()
                await asyncio.sleep(self.collection_interval_seconds)
            except Exception as e:
                logger.error(f"Fehler bei Datensammlung: {e}")
                await asyncio.sleep(self.collection_interval_seconds)

    async def _collect_current_data(self) -> None:
        """Sammelt aktuelle Performance-Daten."""
        try:
            # Hole abgeschlossene Tasks der letzten Periode
            completed_tasks = await self._get_completed_tasks()

            for task_info in completed_tasks:
                data_point = await self._create_data_point(task_info)
                if data_point:
                    self._data_points.append(data_point)

            # Memory-Management
            if len(self._data_points) > self.max_data_points:
                # Entferne älteste Datenpunkte
                self._data_points = self._data_points[-self.max_data_points:]

            logger.debug({
                "event": "data_collection_completed",
                "new_data_points": len(completed_tasks),
                "total_data_points": len(self._data_points)
            })

        except Exception as e:
            logger.error(f"Fehler bei aktueller Datensammlung: {e}")

    async def _get_completed_tasks(self) -> list[dict[str, Any]]:
        """Holt abgeschlossene Tasks der letzten Periode."""
        try:
            # Zeitfenster für Datensammlung
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=self.collection_interval_seconds * 2)

            # Hole Tasks vom Task Manager (vereinfacht für Demo)
            # TODO: Implementiere echte Integration mit Task Manager - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
            all_tasks = []  # Placeholder - würde echte completed tasks holen

            # Simuliere completed tasks für Demo
            completed_tasks = [
                {
                    "task_id": f"demo_task_{i}",
                    "task_type": "simple_query",
                    "agent_id": f"demo_agent_{i % 3}",
                    "execution_time_ms": 1000 + (i * 100),
                    "success": True,
                    "error_type": None,
                    "completed_at": end_time - timedelta(minutes=i),
                    "payload": {"query": f"demo query {i}"},
                    "context": {"priority": "normal"}
                }
                for i in range(5)  # 5 Demo-Tasks
            ]

            return completed_tasks

        except Exception as e:
            logger.error(f"Fehler beim Holen abgeschlossener Tasks: {e}")
            return []

    async def _create_data_point(self, task_info: dict[str, Any]) -> PerformanceDataPoint | None:
        """Erstellt PerformanceDataPoint aus Task-Informationen."""
        try:
            # Task-Charakteristika extrahieren
            task_characteristics = await self._extract_task_characteristics(task_info)

            # Agent-Charakteristika extrahieren
            agent_characteristics = await self._extract_agent_characteristics(
                task_info["agent_id"],
                task_info["completed_at"]
            )

            if not agent_characteristics:
                return None

            # Umgebungs-Informationen
            system_load = await self._get_system_load(task_info["completed_at"])
            concurrent_executions = await self._get_concurrent_executions(task_info["completed_at"])

            # Zeitinformationen
            completed_at = task_info["completed_at"]
            time_of_day_hour = completed_at.hour
            day_of_week = completed_at.weekday()

            # PerformanceDataPoint erstellen
            data_point = PerformanceDataPoint(
                execution_id=f"{task_info['task_id']}_{int(time.time())}",
                task_id=task_info["task_id"],
                agent_id=task_info["agent_id"],
                timestamp=completed_at,
                task_characteristics=task_characteristics,
                agent_characteristics=agent_characteristics,
                system_load=system_load,
                concurrent_executions=concurrent_executions,
                time_of_day_hour=time_of_day_hour,
                day_of_week=day_of_week,
                actual_execution_time_ms=float(task_info["execution_time_ms"]),
                success=task_info["success"],
                error_type=task_info.get("error_type")
            )

            return data_point

        except Exception as e:
            logger.error(f"Fehler beim Erstellen von DataPoint: {e}")
            return None

    async def _extract_task_characteristics(self, task_info: dict[str, Any]) -> TaskCharacteristics:
        """Extrahiert Task-Charakteristika."""
        payload = task_info.get("payload", {})
        context = task_info.get("context", {})

        # Komplexitäts-Score schätzen (vereinfacht)
        complexity_score = self._estimate_task_complexity(task_info)

        # Token-Schätzung
        estimated_tokens = self._estimate_tokens(payload)

        # Required Capabilities (vereinfacht)
        required_capabilities = self._extract_required_capabilities(task_info["task_type"])

        return TaskCharacteristics(
            task_type=task_info["task_type"],
            complexity_score=complexity_score,
            estimated_tokens=estimated_tokens,
            required_capabilities=required_capabilities,
            user_priority=context.get("priority", "normal"),
            deadline_urgency=context.get("deadline_urgency", 0.0),
            dependency_count=len(context.get("dependencies", [])),
            parallel_execution_possible=context.get("parallel_execution", True),
            payload_size_bytes=len(json.dumps(payload).encode()),
            contains_files=any("file" in str(v).lower() for v in payload.values()),
            requires_external_api=any("api" in str(v).lower() for v in payload.values())
        )

    async def _extract_agent_characteristics(
        self,
        agent_id: str,
        timestamp: datetime
    ) -> AgentCharacteristics | None:
        """Extrahiert Agent-Charakteristika."""
        try:
            # Agent-Info aus Registry
            agent_info = await self.agent_registry.get_agent(agent_id)
            if not agent_info:
                return None

            # Performance-Metriken aus Monitor
            capability_metrics = self.performance_monitor._capability_metrics
            agent_metrics = {k: v for k, v in capability_metrics.items() if k.startswith(agent_id)}

            # Aggregiere Metriken
            total_requests = sum(m.get("total_requests", 0) for m in agent_metrics.values())
            successful_requests = sum(m.get("successful_requests", 0) for m in agent_metrics.values())
            total_response_time = sum(m.get("total_response_time", 0.0) for m in agent_metrics.values())

            # Berechne abgeleitete Metriken
            success_rate = successful_requests / total_requests if total_requests > 0 else 1.0
            error_rate = 1.0 - success_rate
            avg_response_time = total_response_time / total_requests if total_requests > 0 else 0.0

            # Load-Schätzung (vereinfacht)
            current_load = min(1.0, total_requests / 100.0)  # Vereinfachte Schätzung

            return AgentCharacteristics(
                agent_id=agent_id,
                agent_type=agent_info.agent_type.value,
                capabilities=agent_info.capabilities,
                current_load=current_load,
                avg_response_time_ms=avg_response_time * 1000,
                success_rate=success_rate,
                error_rate=error_rate,
                max_concurrent_tasks=getattr(agent_info, "max_concurrent_tasks", 10),
                current_active_tasks=int(current_load * 10),  # Vereinfacht
                queue_length=0,  # TODO: Aus Task Manager holen - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
                total_completed_tasks=successful_requests,
                avg_task_completion_time_ms=avg_response_time * 1000,
                specialization_score=0.8  # TODO: Berechnen basierend auf Task-Type-Verteilung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
            )

        except Exception as e:
            logger.error(f"Fehler beim Extrahieren von Agent-Charakteristika: {e}")
            return None

    def _estimate_task_complexity(self, task_info: dict[str, Any]) -> float:
        """Schätzt Task-Komplexität (1-10 Skala)."""
        payload = task_info.get("payload", {})

        # Basis-Komplexität nach Task-Type
        type_complexity = {
            "simple_query": 2.0,
            "data_processing": 4.0,
            "llm_generation": 6.0,
            "complex_analysis": 8.0,
            "multi_step_workflow": 9.0
        }

        base_complexity = type_complexity.get(task_info["task_type"], 5.0)

        # Modifikatoren
        complexity_modifiers = 0.0

        # Payload-Größe
        payload_size = len(json.dumps(payload).encode())
        if payload_size > 10000:
            complexity_modifiers += 1.0
        elif payload_size > 1000:
            complexity_modifiers += 0.5

        # Anzahl Parameter
        param_count = len(payload)
        if param_count > 10:
            complexity_modifiers += 1.0
        elif param_count > 5:
            complexity_modifiers += 0.5

        # Externe Dependencies
        if any("api" in str(v).lower() for v in payload.values()):
            complexity_modifiers += 1.0

        final_complexity = min(10.0, max(1.0, base_complexity + complexity_modifiers))
        return final_complexity

    def _estimate_tokens(self, payload: dict[str, Any]) -> int:
        """Schätzt Token-Anzahl für Task."""
        # Vereinfachte Token-Schätzung: 4 Zeichen = 1 Token
        total_chars = sum(len(str(v)) for v in payload.values())
        return max(1, total_chars // 4)

    def _extract_required_capabilities(self, task_type: str) -> list[str]:
        """Extrahiert erforderliche Capabilities für Task-Type."""
        capability_mapping = {
            "simple_query": ["query_processing"],
            "data_processing": ["data_analysis", "computation"],
            "llm_generation": ["llm_integration", "text_generation"],
            "complex_analysis": ["advanced_analytics", "machine_learning"],
            "multi_step_workflow": ["workflow_orchestration", "task_coordination"]
        }

        return capability_mapping.get(task_type, ["general_processing"])

    async def _get_system_load(self, timestamp: datetime) -> float:
        """Holt System-Load zum gegebenen Zeitpunkt."""
        # TODO: Implementiere echte System-Load-Abfrage - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
        # Für jetzt: Vereinfachte Schätzung basierend auf Tageszeit
        hour = timestamp.hour
        if 9 <= hour <= 17:  # Arbeitszeit
            return 0.7
        if 18 <= hour <= 22:  # Abend
            return 0.4
        # Nacht
        return 0.2

    async def _get_concurrent_executions(self, timestamp: datetime) -> int:
        """Holt Anzahl gleichzeitiger Executions zum gegebenen Zeitpunkt."""
        # TODO: Implementiere echte Concurrent-Execution-Zählung - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/113
        # Für jetzt: Vereinfachte Schätzung
        return 5

    def get_data_points(self, limit: int | None = None) -> list[PerformanceDataPoint]:
        """Gibt gesammelte Datenpunkte zurück."""
        if limit:
            return self._data_points[-limit:]
        return self._data_points.copy()

    def get_data_summary(self) -> dict[str, Any]:
        """Gibt Zusammenfassung der gesammelten Daten zurück."""
        if not self._data_points:
            return {"total_data_points": 0}

        # Basis-Statistiken
        execution_times = [dp.actual_execution_time_ms for dp in self._data_points]

        return {
            "total_data_points": len(self._data_points),
            "collection_start_time": self._collection_start_time.isoformat(),
            "data_ready_for_training": len(self._data_points) >= self.min_data_points,
            "execution_time_stats": {
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "avg_ms": sum(execution_times) / len(execution_times),
                "median_ms": sorted(execution_times)[len(execution_times) // 2]
            },
            "task_types": list(set(dp.task_characteristics.task_type for dp in self._data_points)),
            "agent_ids": list(set(dp.agent_id for dp in self._data_points)),
            "success_rate": sum(1 for dp in self._data_points if dp.success) / len(self._data_points)
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Exportiert Daten als Pandas DataFrame."""
        if not self._data_points:
            return pd.DataFrame()

        # Konvertiere zu Feature-Vectors
        feature_data = []
        for dp in self._data_points:
            features = dp.to_feature_vector()
            features["target_execution_time_ms"] = dp.actual_execution_time_ms
            features["success"] = dp.success
            features["timestamp"] = dp.timestamp.isoformat()
            features["agent_id"] = dp.agent_id
            features["task_id"] = dp.task_id
            feature_data.append(features)

        return pd.DataFrame(feature_data)
