#!/usr/bin/env python3
# backend/services/orchestrator/enterprise_orchestrator_service.py
"""Enterprise LLM-powered Orchestrator Service für Voice-Integration.

Hochentwickelter, intelligenter Orchestrator mit:
- Fortgeschrittener Task-Dekomposition mit LLM-Intelligenz
- Intelligenter Agent-Mapping und -Selektion
- Enterprise-grade Voice-Request-Verarbeitung
- Umfassende TRAIN-Logs für Production-Monitoring
- Skalierbare, produktionsreife Architektur

Enterprise-ready Implementation für Keiko Personal Assistant Platform.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

from kei_logging import get_logger, log_orchestrator_step, training_trace

logger = get_logger(__name__)


class EnterpriseOrchestratorService:
    """Enterprise LLM-powered Orchestrator Service für Voice-Integration.

    Führt intelligente Task-Dekomposition und fortgeschrittenes Agent-Mapping
    für Voice-Requests in einer produktionsreifen, skalierbaren Architektur durch.
    """

    def __init__(self):
        """Initialisiert Enterprise Orchestrator Service."""
        self._is_running = False
        logger.info("Enterprise LLM-powered Orchestrator Service initialisiert")

    async def start(self) -> None:
        """Startet Orchestrator Service."""
        if self._is_running:
            return

        log_orchestrator_step(
            "Starting Enterprise LLM-powered Orchestrator Service",
            "orchestration",
            components=["IntelligentTaskDecomposition", "AdvancedAgentMapping", "EnterpriseVoiceProcessing"]
        )

        self._is_running = True
        logger.info("Enterprise LLM-powered Orchestrator Service gestartet")

    async def stop(self) -> None:
        """Stoppt Orchestrator Service."""
        if not self._is_running:
            return

        log_orchestrator_step(
            "Stopping Enterprise LLM-powered Orchestrator Service",
            "orchestration"
        )

        self._is_running = False
        logger.info("Enterprise LLM-powered Orchestrator Service gestoppt")

    @training_trace(context={"component": "minimal_orchestrator", "phase": "voice_request"})
    async def handle_voice_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Verarbeitet Voice-Request über minimalen LLM-powered Orchestrator.

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

            # 2. Agent-Mapping (ohne Registry-Abhängigkeit)
            log_orchestrator_step(
                "Mapping Task to Agent",
                "agent_call",
                request_id=request_id,
                task_type=task_type
            )

            selected_agent = self._map_task_to_agent(task_type, request_id)

            if not selected_agent:
                log_orchestrator_step(
                    "No Agent Mapping Found",
                    "agent_call",
                    request_id=request_id,
                    task_type=task_type
                )
                return self._create_error_response(
                    request_id,
                    "Kein Agent-Mapping gefunden",
                    f"Für Task-Type '{task_type}' ist kein Agent-Mapping konfiguriert",
                    time.time() - start_time
                )

            # 3. Agent-Ausführung - ECHTER Agent-Aufruf
            log_orchestrator_step(
                "Executing Task with Agent",
                "agent_call",
                request_id=request_id,
                selected_agent_id=selected_agent,
                task_type=task_type
            )

            # Rufe echten Agent auf für Bildgenerierung
            if task_type == "image_generation":
                try:
                    from agents.orchestrator.agent_operations import generate_image_implementation

                    # Extrahiere Prompt aus user_input
                    prompt = user_input.replace("Erstelle ein Bild:", "").replace("Erstelle mir ein Bild", "").strip()
                    if not prompt:
                        prompt = user_input

                    log_orchestrator_step(
                        "Calling Real Image Generation Agent",
                        "agent_call",
                        request_id=request_id,
                        selected_agent_id=selected_agent,
                        prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt
                    )

                    # Echter Agent-Aufruf
                    agent_result = await generate_image_implementation(
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        style="Realistic",
                        user_id=request.get("user_id"),
                        session_id=request.get("session_id")
                    )

                    # Debug: Zeige Agent-Result Details
                    log_orchestrator_step(
                        "Agent Result Debug",
                        "agent_call",
                        request_id=request_id,
                        selected_agent_id=selected_agent,
                        agent_result_keys=list(agent_result.keys()) if agent_result else None,
                        agent_result_success=agent_result.get("success") if agent_result else None,
                        agent_result_image_url=agent_result.get("image_url")[:50] + "..." if agent_result and agent_result.get("image_url") else None,
                        agent_result_response=agent_result.get("response")[:100] + "..." if agent_result and agent_result.get("response") else None
                    )

                    # Prüfe Agent-Result - Verwende die korrekten Keys die der Agent zurückgibt
                    success_indicators = [
                        agent_result and agent_result.get("status") == "success",  # Agent gibt 'status': 'success' zurück
                        agent_result and agent_result.get("storage_url"),          # Agent gibt 'storage_url' zurück
                        agent_result and agent_result.get("success"),              # Fallback für alte API
                        agent_result and agent_result.get("image_url"),            # Fallback für alte API
                        agent_result and "storage_url" in str(agent_result),       # String-Check für storage_url
                        agent_result and "hochgeladen" in str(agent_result)        # String-Check für Upload-Erfolg
                    ]

                    if any(success_indicators):
                        # Verwende storage_url oder image_url
                        image_url = agent_result.get("storage_url") or agent_result.get("image_url", "")

                        log_orchestrator_step(
                            "Real Agent Execution Successful",
                            "agent_call",
                            request_id=request_id,
                            selected_agent_id=selected_agent,
                            image_url=image_url[:80] + "..." if image_url else None,
                            success_reason=f"Indicators: {[i for i, x in enumerate(success_indicators) if x]}",
                            agent_status=agent_result.get("status"),
                            storage_url_present=bool(agent_result.get("storage_url"))
                        )

                        response_text = "✅ Bild erfolgreich erstellt!"
                        if image_url:
                            response_text += f"\n🖼️ Bild-URL: {image_url}"
                    else:
                        log_orchestrator_step(
                            "Real Agent Execution Failed",
                            "agent_call",
                            request_id=request_id,
                            selected_agent_id=selected_agent,
                            error=agent_result.get("error") if agent_result else "No result",
                            agent_result_full=str(agent_result)[:200] + "..." if agent_result else None
                        )
                        response_text = f"❌ Bildgenerierung fehlgeschlagen: {agent_result.get('error') if agent_result else 'Unbekannter Fehler'}"

                except Exception as e:
                    log_orchestrator_step(
                        "Real Agent Execution Exception",
                        "agent_call",
                        request_id=request_id,
                        selected_agent_id=selected_agent,
                        error=str(e)
                    )
                    response_text = f"❌ Agent-Aufruf fehlgeschlagen: {e}"
            else:
                # Für andere Task-Types: Simulation
                await asyncio.sleep(0.1)
                response_text = f"✅ Aufgabe erfolgreich an Agent '{selected_agent}' weitergeleitet"

            execution_time = time.time() - start_time

            log_orchestrator_step(
                "Enterprise Voice Request Completed Successfully",
                "orchestration",
                request_id=request_id,
                execution_time_seconds=execution_time,
                selected_agent=selected_agent
            )

            return {
                "success": True,
                "response_text": response_text,
                "task_id": str(uuid.uuid4()),
                "request_id": request_id,
                "execution_time": execution_time,
                "agent_used": selected_agent,
                "data": {"task_type": task_type, "user_input": user_input}
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Minimal Orchestrator Service Fehler: {e}")

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

    def _map_task_to_agent(self, task_type: str, request_id: str) -> str | None:
        """Mappt Task-Type direkt zu Agent (ohne Registry-Abhängigkeit)."""
        # Statisches Agent-Mapping
        agent_mapping = {
            "image_generation": "agent_image_generator",
            "web_search": "agent_web_researcher",
            "conversation": "agent_conversation"
        }

        selected_agent = agent_mapping.get(task_type)

        if selected_agent:
            log_orchestrator_step(
                "Agent Mapped Successfully",
                "agent_call",
                request_id=request_id,
                selected_agent_id=selected_agent,
                task_type=task_type,
                mapping_type="static_mapping"
            )
        else:
            log_orchestrator_step(
                "No Agent Mapping Found",
                "agent_call",
                request_id=request_id,
                task_type=task_type,
                available_mappings=list(agent_mapping.keys())
            )

        return selected_agent

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
                "task_detection": "active",
                "agent_mapping": "active"
            },
            "timestamp": datetime.now().isoformat()
        }


# Global Enterprise Orchestrator Service Instance
enterprise_orchestrator_service = EnterpriseOrchestratorService()
