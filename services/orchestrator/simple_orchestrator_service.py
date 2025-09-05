#!/usr/bin/env python3
# backend/services/orchestrator/simple_orchestrator_service.py
"""Einfacher LLM-powered Orchestrator Service für Voice-Integration.

Vereinfachte Version ohne ML-Dependencies für sofortige Integration.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

from agents.registry.dynamic_registry import DynamicAgentRegistry
from kei_logging import get_logger, log_orchestrator_step, training_trace

logger = get_logger(__name__)


class SimpleOrchestratorService:
    """Einfacher LLM-powered Orchestrator Service für Voice-Integration."""

    def __init__(self):
        """Initialisiert einfachen Orchestrator Service."""
        self.agent_registry = DynamicAgentRegistry()
        self._is_running = False
        logger.info("Simple LLM-powered Orchestrator Service initialisiert")

    async def start(self) -> None:
        """Startet Orchestrator Service."""
        if self._is_running:
            return

        log_orchestrator_step(
            "Starting Simple LLM-powered Orchestrator Service",
            "orchestration",
            components=["AgentRegistry", "SimpleTaskDecomposition"]
        )

        # Stelle sicher, dass Agent Registry läuft
        if not self.agent_registry.is_initialized():
            await self.agent_registry.start()

        self._is_running = True
        logger.info("Simple LLM-powered Orchestrator Service gestartet")

    async def stop(self) -> None:
        """Stoppt Orchestrator Service."""
        if not self._is_running:
            return

        log_orchestrator_step(
            "Stopping Simple LLM-powered Orchestrator Service",
            "orchestration"
        )

        self._is_running = False
        logger.info("Simple LLM-powered Orchestrator Service gestoppt")

    @training_trace(context={"component": "simple_orchestrator", "phase": "voice_request"})
    async def handle_voice_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Verarbeitet Voice-Request über einfachen LLM-powered Orchestrator.

        Args:
            request: Voice-Request mit 'text', 'user_id', etc.

        Returns:
            Orchestrator-Response mit success, response_text, etc.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        log_orchestrator_step(
            "Processing Voice Request",
            "orchestration",
            request_id=request_id,
            user_input=request.get("text", ""),
            user_id=request.get("user_id"),
            session_id=request.get("session_id")
        )

        try:
            # 1. Task-Type-Erkennung
            user_input = request.get("text", "")
            task_type = self._detect_task_type(user_input)

            log_orchestrator_step(
                "Task Type Detected",
                "task_decomposition",
                request_id=request_id,
                detected_type=task_type,
                user_input_length=len(user_input)
            )

            # 2. Agent-Registry-Suche
            log_orchestrator_step(
                "Searching Agent Registry",
                "agent_call",
                request_id=request_id,
                task_type=task_type
            )

            selected_agent = await self._find_suitable_agent(task_type, request_id)

            if not selected_agent:
                log_orchestrator_step(
                    "No Suitable Agent Found",
                    "agent_call",
                    request_id=request_id,
                    task_type=task_type
                )
                return self._create_error_response(
                    request_id,
                    "Kein passender Agent gefunden",
                    f"Für Task-Type '{task_type}' ist kein Agent verfügbar",
                    time.time() - start_time
                )

            # 3. Agent-Ausführung simulieren
            log_orchestrator_step(
                "Executing Task with Agent",
                "agent_call",
                request_id=request_id,
                selected_agent_id=selected_agent,
                task_type=task_type
            )

            # Simuliere Agent-Ausführung
            await asyncio.sleep(0.1)  # Kurze Simulation

            execution_time = time.time() - start_time

            log_orchestrator_step(
                "Voice Request Completed Successfully",
                "orchestration",
                request_id=request_id,
                execution_time_seconds=execution_time,
                selected_agent=selected_agent
            )

            return {
                "success": True,
                "response_text": f"✅ Aufgabe erfolgreich an Agent '{selected_agent}' weitergeleitet",
                "task_id": str(uuid.uuid4()),
                "request_id": request_id,
                "execution_time": execution_time,
                "agent_used": selected_agent,
                "data": {"task_type": task_type, "user_input": user_input}
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Simple Orchestrator Service Fehler: {e}")

            log_orchestrator_step(
                "Voice Request Exception",
                "orchestration",
                request_id=request_id,
                error=str(e),
                execution_time_seconds=execution_time
            )

            return self._create_error_response(
                request_id,
                "Unerwarteter Fehler",
                str(e),
                execution_time
            )

    async def _find_suitable_agent(self, task_type: str, request_id: str) -> str | None:
        """Findet passenden Agent für Task-Type."""
        try:
            # Hole verfügbare Agents
            available_agents_raw = await self.agent_registry.list_agents()

            if not available_agents_raw:
                log_orchestrator_step(
                    "No Agents Found in Registry",
                    "agent_call",
                    request_id=request_id,
                    agent_count=0
                )
                return None

            # Konvertiere zu Dictionary falls Liste zurückgegeben wird
            if isinstance(available_agents_raw, list):
                available_agents = {
                    getattr(agent, "id", f"agent_{i}"): agent
                    for i, agent in enumerate(available_agents_raw)
                }
            else:
                available_agents = available_agents_raw

            log_orchestrator_step(
                "Agents Found in Registry",
                "agent_call",
                request_id=request_id,
                agent_count=len(available_agents),
                agent_ids=list(available_agents.keys())[:5]  # Erste 5 für Übersicht
            )

            # Einfache Agent-Selection basierend auf Task-Type
            if task_type == "image_generation":
                # Suche nach Image Generator Agent
                image_agent_id = "agent_image_generator"
                if image_agent_id in available_agents:
                    log_orchestrator_step(
                        "Image Generator Agent Found",
                        "agent_call",
                        request_id=request_id,
                        selected_agent_id=image_agent_id,
                        match_reason="task_type_match"
                    )
                    return image_agent_id

            # Fallback: Ersten verfügbaren Agent
            fallback_agent = next(iter(available_agents.keys()))
            log_orchestrator_step(
                "Fallback Agent Selected",
                "agent_call",
                request_id=request_id,
                selected_agent_id=fallback_agent,
                match_reason="fallback_first_available"
            )
            return fallback_agent

        except Exception as e:
            logger.exception(f"Agent-Selection fehlgeschlagen: {e}")
            return None

    def _detect_task_type(self, user_input: str) -> str:
        """Erkennt Task-Type aus User-Input."""
        user_input_lower = user_input.lower()

        if any(keyword in user_input_lower for keyword in ["bild", "image", "foto", "erstelle", "generiere", "zeichne"]):
            return "image_generation"
        if any(keyword in user_input_lower for keyword in ["suche", "recherche", "finde", "web", "internet"]):
            return "web_search"
        return "conversation"

    def _create_error_response(
        self,
        request_id: str,
        message: str,
        error: str | None,
        execution_time: float
    ) -> dict[str, Any]:
        """Erstellt Error-Response."""
        return {
            "success": False,
            "response_text": f"❌ {message}: {error}" if error else f"❌ {message}",
            "request_id": request_id,
            "execution_time": execution_time,
            "error": error
        }

    async def get_health(self) -> dict[str, Any]:
        """Gibt Health-Status zurück."""
        return {
            "status": "healthy" if self._is_running else "stopped",
            "components": {
                "agent_registry": "initialized" if self.agent_registry.is_initialized() else "not_initialized"
            },
            "timestamp": datetime.now().isoformat()
        }


# Global Simple Orchestrator Service Instance
simple_orchestrator_service = SimpleOrchestratorService()
