"""Voice Service Registry Integration.

Integriert den Voice Service mit dem Registry-basierten Orchestrator.
Ersetzt direkte Orchestrator-Imports durch saubere A2A-Communication.

Features:
- Registry-basierte Orchestrierung
- AGENT_ORCHESTRATOR_ID Integration
- Function Call → Orchestrator Mapping
- Clean Text → Orchestrator → Response Pipeline
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger
from services.orchestrator.enterprise_orchestrator_service import EnterpriseOrchestratorService

logger = get_logger(__name__)


@dataclass
class VoiceOrchestratorConfig:
    """Konfiguration für Voice-Orchestrator Integration."""
    orchestrator_id: str
    timeout: float = 30.0
    retry_attempts: int = 3
    fallback_enabled: bool = True


class VoiceOrchestratorIntegration:
    """Integration zwischen Voice Service und Registry Orchestrator."""

    def __init__(self):
        """Initialisiert Voice-Orchestrator Integration."""
        self.config = self._load_config()
        self.function_mappings = self._create_function_mappings()

        logger.info(f"Voice-Orchestrator Integration initialisiert: {self.config.orchestrator_id}")

    def _load_config(self) -> VoiceOrchestratorConfig:
        """Lädt Konfiguration aus Environment."""
        orchestrator_id = os.getenv("AGENT_ORCHESTRATOR_ID")

        if not orchestrator_id:
            raise ValueError("AGENT_ORCHESTRATOR_ID nicht konfiguriert")

        return VoiceOrchestratorConfig(
            orchestrator_id=orchestrator_id,
            timeout=float(os.getenv("VOICE_ORCHESTRATOR_TIMEOUT", "30.0")),
            retry_attempts=int(os.getenv("VOICE_ORCHESTRATOR_RETRIES", "3")),
            fallback_enabled=os.getenv("VOICE_ORCHESTRATOR_FALLBACK", "true").lower() == "true"
        )

    def _create_function_mappings(self) -> dict[str, str]:
        """Erstellt Mapping von Function Calls zu natürlichen Texten."""
        return {
            "generate_image": "Erstelle ein Bild: {prompt}",
            "perform_web_research": "Suche im Internet nach: {query}",
            "photo_request": "Mache ein Foto",
            "general_conversation": "{text}",
            "unknown_function": "Führe folgende Aufgabe aus: {text}"
        }

    async def handle_function_call(
        self,
        function_name: str,
        function_args: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None
    ) -> dict[str, Any]:
        """Behandelt Function Call über Registry Orchestrator."""
        try:
            # 1. Function Call zu natürlichem Text konvertieren
            natural_text = self._convert_function_to_text(function_name, function_args)

            # 2. Orchestrator Request erstellen
            orchestrator_request = {
                "text": natural_text,
                "user_id": user_id or "voice_user",
                "session_id": session_id or f"voice_session_{int(time.time())}",
                "context": {
                    "source": "voice_function_call",
                    "original_function": function_name,
                    "original_args": function_args,
                    "timestamp": time.time()
                }
            }

            # 3. An Registry Orchestrator senden
            response = await self._send_to_orchestrator(orchestrator_request)

            # 4. Response zu Function Call Format konvertieren
            return self._convert_response_to_function_result(response, function_name)

        except Exception as e:
            logger.exception(f"Function call handling failed: {e}")

            if self.config.fallback_enabled:
                return self._create_fallback_response(function_name, str(e))
            raise

    async def handle_direct_text(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None
    ) -> str:
        """Behandelt direkten Text-Input über Registry Orchestrator."""
        try:
            # Orchestrator Request erstellen
            orchestrator_request = {
                "text": text,
                "user_id": user_id or "voice_user",
                "session_id": session_id or f"voice_session_{int(time.time())}",
                "context": {
                    "source": "voice_direct_text",
                    "timestamp": time.time()
                }
            }

            # An Registry Orchestrator senden
            response = await self._send_to_orchestrator(orchestrator_request)

            return response.response_text if response.success else "Entschuldigung, ich konnte die Anfrage nicht verarbeiten."

        except Exception as e:
            logger.exception(f"Direct text handling failed: {e}")

            if self.config.fallback_enabled:
                return "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut."
            raise

    async def _send_to_orchestrator(self, request: dict[str, Any]):
        """Sendet Request an Registry Orchestrator mit Retry-Logic."""
        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"Orchestrator request (attempt {attempt + 1}): {request['text']}")

                # Timeout für Orchestrator-Call
                orchestrator = EnterpriseOrchestratorService()
                await orchestrator.start()

                response_dict = await asyncio.wait_for(
                    orchestrator.handle_voice_request(request),
                    timeout=self.config.timeout
                )

                await orchestrator.stop()

                # Konvertiere Dict-Response zu kompatiblem Format
                response = self._convert_response_format(response_dict)

                logger.debug(f"Orchestrator response: success={response.success}")
                return response

            except TimeoutError as e:
                last_exception = e
                logger.warning(f"Orchestrator timeout (attempt {attempt + 1})")

            except Exception as e:
                last_exception = e
                logger.exception(f"Orchestrator error (attempt {attempt + 1}): {e}")

            # Exponential backoff zwischen Versuchen
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)

        # Alle Versuche fehlgeschlagen
        raise Exception(f"Orchestrator communication failed after {self.config.retry_attempts} attempts: {last_exception}")

    def _convert_response_format(self, response_dict: dict[str, Any]):
        """Konvertiert neue Orchestrator-Response zu kompatiblem Format."""
        from dataclasses import dataclass

        @dataclass
        class CompatibleResponse:
            success: bool
            response_text: str
            task_id: str = None
            agent_used: str = None
            execution_time: float = 0.0
            data: dict = None
            error: str = None

        return CompatibleResponse(
            success=response_dict.get("success", False),
            response_text=response_dict.get("response_text", ""),
            task_id=response_dict.get("task_id"),
            agent_used=response_dict.get("agent_used"),
            execution_time=response_dict.get("execution_time", 0.0),
            data=response_dict.get("data"),
            error=response_dict.get("error")
        )

    def _convert_function_to_text(self, function_name: str, args: dict[str, Any]) -> str:
        """Konvertiert Function Call zu natürlichem Text."""
        # Template für Function Name finden
        template = self.function_mappings.get(function_name, self.function_mappings["unknown_function"])

        try:
            # Template mit Argumenten füllen
            return template.format(**args, text=args.get("prompt", args.get("query", function_name)))

        except KeyError as e:
            logger.warning(f"Missing argument for template: {e}")
            # Fallback: Einfache Beschreibung
            return f"Führe {function_name} aus mit Parametern: {args}"

    def _convert_response_to_function_result(
        self,
        orchestrator_response,
        original_function: str
    ) -> dict[str, Any]:
        """Konvertiert Orchestrator Response zu Function Call Result."""
        if orchestrator_response.success:
            # Für Bildgenerierung: Echte Bilddaten extrahieren
            if original_function == "generate_image" and orchestrator_response.data:
                return {
                    "success": True,
                    "result": orchestrator_response.response_text,
                    "function": original_function,
                    "agent_used": orchestrator_response.agent_used,
                    "execution_time": orchestrator_response.execution_time,
                    "task_id": orchestrator_response.task_id,
                    # Echte Bilddaten hinzufügen
                    "image_data": orchestrator_response.data
                }
            # Standard-Response für andere Functions
            return {
                "success": True,
                "result": orchestrator_response.response_text,
                "function": original_function,
                "agent_used": orchestrator_response.agent_used,
                "execution_time": orchestrator_response.execution_time,
                "task_id": orchestrator_response.task_id
            }
        return {
            "success": False,
            "error": orchestrator_response.error or "Unknown error",
            "result": orchestrator_response.response_text,
            "function": original_function,
            "task_id": orchestrator_response.task_id
        }

    def _create_fallback_response(self, function_name: str, error: str) -> dict[str, Any]:
        """Erstellt Fallback-Response bei Fehlern."""
        fallback_messages = {
            "generate_image": "Entschuldigung, die Bildgenerierung ist momentan nicht verfügbar.",
            "perform_web_research": "Entschuldigung, die Web-Suche ist momentan nicht verfügbar.",
            "photo_request": "Entschuldigung, die Foto-Funktion ist momentan nicht verfügbar."
        }

        message = fallback_messages.get(function_name, "Entschuldigung, diese Funktion ist momentan nicht verfügbar.")

        return {
            "success": False,
            "error": error,
            "result": message,
            "function": function_name,
            "fallback": True
        }


class VoiceOrchestratorMetrics:
    """Metriken für Voice-Orchestrator Integration."""

    def __init__(self):
        """Initialisiert Metriken."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.function_call_counts = {}
        self.error_counts = {}

    def record_request(
        self,
        function_name: str,
        success: bool,
        response_time: float,
        error: str | None = None
    ):
        """Zeichnet Request-Metriken auf."""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.error_counts[error] = self.error_counts.get(error, 0) + 1

        self.total_response_time += response_time
        self.function_call_counts[function_name] = self.function_call_counts.get(function_name, 0) + 1

    def get_metrics(self) -> dict[str, Any]:
        """Gibt aktuelle Metriken zurück."""
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0.0
        )

        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "function_call_counts": self.function_call_counts,
            "error_counts": self.error_counts
        }


# Global Instances (Lazy Loading)
_voice_orchestrator_integration = None
_voice_orchestrator_metrics = None


def get_voice_orchestrator_integration() -> VoiceOrchestratorIntegration:
    """Lazy Loading für Voice Orchestrator Integration."""
    global _voice_orchestrator_integration
    if _voice_orchestrator_integration is None:
        _voice_orchestrator_integration = VoiceOrchestratorIntegration()
    return _voice_orchestrator_integration


def get_voice_orchestrator_metrics() -> VoiceOrchestratorMetrics:
    """Lazy Loading für Voice Orchestrator Metrics."""
    global _voice_orchestrator_metrics
    if _voice_orchestrator_metrics is None:
        _voice_orchestrator_metrics = VoiceOrchestratorMetrics()
    return _voice_orchestrator_metrics
