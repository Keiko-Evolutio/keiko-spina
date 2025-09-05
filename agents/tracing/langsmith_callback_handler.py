"""LangSmith CallbackHandler für Agent-Operationen.

Erfasst Token-Nutzung, Latenz und Prompt/Response mit PII-Redaktion.
Robuste Fehlerbehandlung für Produktionsumgebungen.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger
from monitoring.langsmith_integration import get_langsmith_integration, redact_pii

from .base_callback_handler import BaseCallbackHandler
from .tracing_utils import (
    LatencyTracker,
    MetricsBuilder,
    create_safe_preview,
    safe_get_error_code,
)

logger = get_logger(__name__)

# Konstanten
AGENT_RUN_NAME = "agent.execute"
RESPONSE_PREVIEW_KEY = "response_preview"


class LangSmithAgentCallbackHandler(BaseCallbackHandler):
    """Callback-Handler für Agent-Operationen mit LangSmith-Integration.

    Framework-agnostischer Handler für verschiedene Agent-Systeme.
    Fasst wichtige Ereignisse in drei Methoden zusammen.
    """

    def __init__(self, agent_id: str) -> None:
        """Initialisiert den Callback-Handler.

        Args:
            agent_id: Agent-ID für Monitoring.
        """
        super().__init__(agent_id)
        self._run_id: str | None = None
        self._latency_tracker = LatencyTracker()

    async def on_start(
        self,
        instruction: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Startet Monitoring eines Agentenlaufs.

        Args:
            instruction: Eingabeaufforderung/Taskbeschreibung.
            metadata: Optionale Zusatzinformationen.
        """
        integration = get_langsmith_integration()
        self._latency_tracker.start()

        try:
            async with integration.agent_run(
                name=AGENT_RUN_NAME,
                inputs={
                    "agent_id": self.agent_id,
                    "instruction": redact_pii(instruction or ""),
                    **(metadata or {}),
                },
            ) as run_ctx:
                # Run-ID für Updates speichern
                self._run_id = run_ctx.get("run_id")
        except Exception as exc:
            await self._handle_connection_error(exc, "LangSmith on_start")

    async def on_end(
        self,
        response: str | dict[str, Any] | Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """Beendet Monitoring und aktualisiert Metriken.

        Args:
            response: Agentenantwort, wird PII-bereinigt gespeichert.
            prompt_tokens: Eingabetoken (sofern verfügbar).
            completion_tokens: Ausgabetoken (sofern verfügbar).
        """
        try:
            await self._update_successful_run(response, prompt_tokens, completion_tokens)
        except Exception as exc:
            await self._handle_connection_error(exc, "LangSmith on_end")
        finally:
            self._cleanup_run_state()

    async def _update_successful_run(
        self,
        response: str | dict[str, Any] | Any,
        prompt_tokens: int,
        completion_tokens: int
    ) -> None:
        """Aktualisiert LangSmith-Run mit Erfolgs-Metriken.

        Args:
            response: Agentenantwort für Preview-Erstellung.
            prompt_tokens: Anzahl Eingabetoken.
            completion_tokens: Anzahl Ausgabetoken.
        """
        integration = get_langsmith_integration()
        if not (integration.is_active and self._run_id):
            return

        # Preview und Latenz vorbereiten
        preview = create_safe_preview(response)
        latency_seconds = self._latency_tracker.get_latency_seconds()

        # Metriken erstellen und Run aktualisieren
        metrics_payload = (
            MetricsBuilder()
            .add_tokens(prompt_tokens, completion_tokens)
            .add_success_error(success=True)
            .add_latency(latency_seconds)
            .build()
        )

        integration._client.update_run(  # type: ignore[attr-defined]
            run_id=self._run_id,
            extra={"metrics": metrics_payload},
            outputs={RESPONSE_PREVIEW_KEY: redact_pii(preview)},
        )

    async def on_error(self, error: Exception) -> None:
        """Meldet Fehlerereignis an LangSmith.

        Args:
            error: Ausgelöste Exception.
        """
        try:
            await self._update_failed_run(error)
        except Exception as exc:
            await self._handle_connection_error(exc, "LangSmith on_error")
        finally:
            self._cleanup_run_state()

    async def _update_failed_run(self, error: Exception) -> None:
        """Aktualisiert LangSmith-Run mit Fehler-Informationen.

        Args:
            error: Exception die aufgetreten ist.
        """
        integration = get_langsmith_integration()
        if not (integration.is_active and self._run_id):
            return

        error_code = safe_get_error_code(error)
        payload = {
            "metrics": MetricsBuilder().add_success_error(success=False).build(),
            "error": {"code": str(error_code)},
        }

        integration._client.update_run(  # type: ignore[attr-defined]
            run_id=self._run_id,
            extra=payload,
            error=str(error),
        )

    def _cleanup_run_state(self) -> None:
        """Bereinigt den internen Run-Zustand."""
        self._run_id = None
        self._latency_tracker.reset()


__all__ = ["LangSmithAgentCallbackHandler"]
