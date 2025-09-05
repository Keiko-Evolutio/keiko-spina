"""Voice Routes Refactored.

Refactored Voice Service mit Registry-basierter Orchestrierung.
Ersetzt direkte Orchestrator-Imports durch saubere A2A-Communication.

Features:
- Registry-basierte Orchestrierung über AGENT_ORCHESTRATOR_ID
- Entfernung direkter Orchestrator-Imports
- Clean Text → Orchestrator → Response Pipeline
- Backward-Compatibility mit bestehender Voice API
"""

import time
from typing import Any

from kei_logging import get_logger

from .voice_registry_integration import (
    get_voice_orchestrator_integration,
    get_voice_orchestrator_metrics,
)

logger = get_logger(__name__)


class VoiceServiceRefactored:
    """Refactored Voice Service mit Registry-Integration."""

    def __init__(self):
        """Initialisiert refactored Voice Service."""
        self.integration = get_voice_orchestrator_integration()
        self.metrics = get_voice_orchestrator_metrics()

        logger.info("Voice Service Refactored initialisiert mit Registry-Integration")

    async def handle_function_call(
        self,
        function_name: str,
        function_args: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None
    ) -> dict[str, Any]:
        """Behandelt Function Calls über Registry Orchestrator.

        Ersetzt die direkten Orchestrator-Imports:
        - generate_image_implementation
        - perform_web_research_implementation
        - photo_request_implementation
        """
        start_time = time.time()

        try:
            logger.info(f"Function call via registry orchestrator: {function_name}")

            # Über Registry Orchestrator ausführen
            result = await self.integration.handle_function_call(
                function_name=function_name,
                function_args=function_args,
                user_id=user_id,
                session_id=session_id
            )

            # Metriken aufzeichnen
            execution_time = time.time() - start_time
            self.metrics.record_request(
                function_name=function_name,
                success=result.get("success", False),
                response_time=execution_time,
                error=result.get("error")
            )

            logger.info(f"Function call completed: {function_name} (success={result.get('success')})")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Function call failed: {function_name} - {e}")

            # Metriken für Fehler aufzeichnen
            self.metrics.record_request(
                function_name=function_name,
                success=False,
                response_time=execution_time,
                error=str(e)
            )

            # Fallback-Response
            return {
                "success": False,
                "error": str(e),
                "result": f"Entschuldigung, {function_name} ist momentan nicht verfügbar.",
                "function": function_name
            }

    async def handle_direct_text(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None
    ) -> str:
        """Behandelt direkten Text-Input über Registry Orchestrator.

        Für Fälle wo kein Function Call erkannt wird.
        """
        start_time = time.time()

        try:
            logger.debug(f"Direct text via registry orchestrator: {text[:50]}...")

            # Über Registry Orchestrator ausführen
            response = await self.integration.handle_direct_text(
                text=text,
                user_id=user_id,
                session_id=session_id
            )

            # Metriken aufzeichnen
            execution_time = time.time() - start_time
            self.metrics.record_request(
                function_name="direct_text",
                success=True,
                response_time=execution_time
            )

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Direct text handling failed: {e}")

            # Metriken für Fehler aufzeichnen
            self.metrics.record_request(
                function_name="direct_text",
                success=False,
                response_time=execution_time,
                error=str(e)
            )

            return "Entschuldigung, ich konnte die Anfrage nicht verarbeiten."

    def get_metrics(self) -> dict[str, Any]:
        """Gibt aktuelle Voice Service Metriken zurück."""
        return self.metrics.get_metrics()


class VoiceRoutesRefactored:
    """Refactored Voice Routes mit Registry-Integration."""

    def __init__(self):
        """Initialisiert refactored Voice Routes."""
        self.voice_service = VoiceServiceRefactored()

    async def handle_legacy_function_call(
        self,
        mapped_name: str,
        params: dict[str, Any],
        user_id: str | None = None
    ) -> dict[str, Any]:
        """Behandelt Legacy Function Calls über Registry Orchestrator.

        Ersetzt die direkten Imports in voice_routes.py:
        - from kei_agents.orchestrator.agent_operations import generate_image_implementation
        - from kei_agents.orchestrator.agent_operations import perform_web_research_implementation
        - from kei_agents.orchestrator.agent_operations import photo_request_implementation
        """
        # Legacy Function Name Mapping
        function_mapping = {
            "generate_image": {
                "function": "generate_image",
                "args": {
                    "prompt": params.get("prompt", ""),
                    "size": params.get("size", "1024x1024"),
                    "quality": params.get("quality", "standard"),
                    "style": params.get("style", "Realistic")
                }
            },
            "perform_web_research": {
                "function": "perform_web_research",
                "args": {
                    "query": params.get("query", ""),
                    "max_results": params.get("max_results", 5)
                }
            },
            "photo_request": {
                "function": "photo_request",
                "args": {}
            }
        }

        if mapped_name not in function_mapping:
            logger.warning(f"Unknown legacy function: {mapped_name}")
            return {
                "success": False,
                "error": f"Unknown function: {mapped_name}",
                "result": f"Funktion {mapped_name} ist nicht verfügbar."
            }

        # Function Call über Registry Orchestrator
        function_config = function_mapping[mapped_name]

        return await self.voice_service.handle_function_call(
            function_name=function_config["function"],
            function_args=function_config["args"],
            user_id=user_id,
            session_id=params.get("session_id")
        )

    async def handle_text_input(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None
    ) -> str:
        """Behandelt Text-Input über Registry Orchestrator."""
        return await self.voice_service.handle_direct_text(
            text=text,
            user_id=user_id,
            session_id=session_id
        )

    def get_service_metrics(self) -> dict[str, Any]:
        """Gibt Service-Metriken zurück."""
        return self.voice_service.get_metrics()


# Global Instance für Backward-Compatibility
voice_routes_refactored = VoiceRoutesRefactored()


# Replacement Functions für Legacy Code
async def refactored_generate_image_implementation(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "Realistic",
    user_id: str | None = None,
    session_id: str | None = None
) -> dict[str, Any]:
    """Refactored generate_image_implementation.

    Ersetzt:
    from agents.orchestrator.agent_operations import generate_image_implementation
    """
    return await voice_routes_refactored.voice_service.handle_function_call(
        function_name="generate_image",
        function_args={
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "style": style
        },
        user_id=user_id,
        session_id=session_id
    )


async def refactored_perform_web_research_implementation(
    query: str,
    max_results: int = 5,
    user_id: str | None = None,
    session_id: str | None = None
) -> dict[str, Any]:
    """Refactored perform_web_research_implementation.

    Ersetzt:
    from agents.orchestrator.agent_operations import perform_web_research_implementation
    """
    return await voice_routes_refactored.voice_service.handle_function_call(
        function_name="perform_web_research",
        function_args={
            "query": query,
            "max_results": max_results
        },
        user_id=user_id,
        session_id=session_id
    )


async def refactored_photo_request_implementation(
    user_id: str | None = None,
    session_id: str | None = None
) -> dict[str, Any]:
    """Refactored photo_request_implementation.

    Ersetzt:
    from agents.orchestrator.agent_operations import photo_request_implementation
    """
    return await voice_routes_refactored.voice_service.handle_function_call(
        function_name="photo_request",
        function_args={},
        user_id=user_id,
        session_id=session_id
    )
