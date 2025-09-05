"""Enterprise-Grade LLM-Tracking für Pydantic Logfire.

Umfassende Instrumentierung und Monitoring für alle LLM-Interaktionen
in der Keiko Personal Assistant Plattform.

Features:
- Automatische LLM-Call-Instrumentierung (OpenAI, Anthropic, etc.)
- Token-Usage-Tracking und Kosten-Monitoring
- Performance-Metriken und Latenz-Analyse
- Agent-spezifische Metriken und Conversation-Tracking
- Integration mit bestehender Agent-Metrics-Infrastruktur
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from typing import Any

from kei_logging import get_logger

from .logfire_integration import LogfireManager, get_logfire_manager

logger = get_logger(__name__)


@dataclass
class LLMCallMetrics:
    """Umfassende Metriken für LLM-Calls."""

    # Identifikation
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    provider: str = ""
    call_type: str = "completion"  # completion, chat, embedding, etc.

    # Context
    agent_id: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float | None = None

    # Token-Usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Kosten
    estimated_cost_usd: float | None = None
    cost_per_token: float | None = None

    # Performance
    tokens_per_second: float | None = None
    latency_ms: float | None = None

    # Qualität
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self, end_time: datetime | None = None) -> None:
        """Finalisiert die Metriken am Ende eines Calls."""
        self.end_time = end_time or datetime.now(UTC)

        # Berechne Duration
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.duration_ms = duration * 1000
            self.latency_ms = self.duration_ms

        # Berechne Tokens per Second
        if self.completion_tokens and self.duration_ms:
            self.tokens_per_second = (self.completion_tokens / self.duration_ms) * 1000

    def update_tokens(self, prompt_tokens: int, completion_tokens: int, total_tokens: int | None = None) -> None:
        """Aktualisiert Token-Usage-Informationen."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens or (prompt_tokens + completion_tokens)

    def update_cost(self, cost_usd: float, cost_per_token: float | None = None) -> None:
        """Aktualisiert Kosten-Informationen."""
        self.estimated_cost_usd = cost_usd
        self.cost_per_token = cost_per_token

    def add_metadata(self, **kwargs) -> None:
        """Fügt Metadaten hinzu."""
        self.metadata.update(kwargs)

    def mark_error(self, error_type: str, error_message: str) -> None:
        """Markiert den Call als fehlgeschlagen."""
        self.success = False
        self.error_type = error_type
        self.error_message = error_message


class LLMCallContext:
    """Context-Manager für LLM-Call-Tracking."""

    def __init__(self, metrics: LLMCallMetrics, tracker: LLMCallTracker):
        self.metrics = metrics
        self.tracker = tracker

    def __enter__(self) -> LLMCallContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.metrics.mark_error(
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )

        self.metrics.finalize()
        self.tracker._finalize_call(self.metrics)

    def update_tokens(self, prompt_tokens: int, completion_tokens: int, total_tokens: int | None = None) -> None:
        """Aktualisiert Token-Usage."""
        self.metrics.update_tokens(prompt_tokens, completion_tokens, total_tokens)

    def update_cost(self, cost_usd: float, cost_per_token: float | None = None) -> None:
        """Aktualisiert Kosten."""
        self.metrics.update_cost(cost_usd, cost_per_token)

    def add_metadata(self, **kwargs) -> None:
        """Fügt Metadaten hinzu."""
        self.metrics.add_metadata(**kwargs)


class LLMCallTracker:
    """Enterprise-Grade LLM-Call-Tracker für umfassende LLM-Observability.

    Integriert sich nahtlos in die bestehende Agent-Metrics-Infrastruktur
    und bietet detaillierte Einblicke in alle LLM-Interaktionen.
    """

    def __init__(self, logfire_manager: LogfireManager | None = None):
        self.logfire_manager = logfire_manager or get_logfire_manager()
        self._active_calls: dict[str, LLMCallMetrics] = {}
        self._call_history: list[LLMCallMetrics] = []
        self._max_history = 1000  # Begrenze History für Memory-Management

        # Token-Kosten-Mapping (approximativ)
        self._token_costs = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
        }

    @contextmanager
    def track_llm_call(
        self,
        model: str,
        provider: str,
        call_type: str = "completion",
        agent_id: str | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        **metadata
    ) -> LLMCallContext:
        """Erstellt einen LLM-Call-Tracking-Context.

        Args:
            model: LLM-Model-Name
            provider: Provider (openai, anthropic, etc.)
            call_type: Art des Calls (completion, chat, embedding)
            agent_id: ID des aufrufenden Agents
            conversation_id: ID der Conversation
            user_id: Benutzer-ID
            session_id: Session-ID
            **metadata: Zusätzliche Metadaten

        Yields:
            LLMCallContext: Context für den LLM-Call
        """
        metrics = LLMCallMetrics(
            model=model,
            provider=provider,
            call_type=call_type,
            agent_id=agent_id,
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )

        # Registriere aktiven Call
        self._active_calls[metrics.call_id] = metrics

        # Starte Logfire-Span
        span_name = f"llm_call_{provider}_{model}"
        span_attributes = {
            "llm_provider": provider,
            "llm_model": model,
            "llm_call_type": call_type,
            "llm_call_id": metrics.call_id,
        }

        if agent_id:
            span_attributes["agent_id"] = agent_id
        if conversation_id:
            span_attributes["conversation_id"] = conversation_id

        try:
            with self.logfire_manager.span(span_name, **span_attributes):
                context = LLMCallContext(metrics, self)
                yield context
        except Exception as e:
            # Fehlerbehandlung für Context-Exceptions
            metrics.mark_error(type(e).__name__, str(e))
            metrics.finalize()
            self._finalize_call(metrics)
            raise
        finally:
            # Cleanup
            self._active_calls.pop(metrics.call_id, None)

    def _finalize_call(self, metrics: LLMCallMetrics) -> None:
        """Finalisiert einen LLM-Call und sendet Metriken."""
        try:
            # Schätze Kosten falls nicht gesetzt
            if not metrics.estimated_cost_usd and metrics.total_tokens:
                metrics.estimated_cost_usd = self._estimate_cost(
                    metrics.model,
                    metrics.prompt_tokens or 0,
                    metrics.completion_tokens or 0
                )

            # Sende an Logfire
            self._send_to_logfire(metrics)

            # Integriere mit Agent-Metrics (falls verfügbar)
            self._integrate_with_agent_metrics(metrics)

            # Füge zur History hinzu
            self._add_to_history(metrics)

        except Exception as e:
            logger.warning(f"Fehler beim Finalisieren des LLM-Calls: {e}")

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Schätzt die Kosten für einen LLM-Call."""
        model_lower = model.lower()

        # Finde passende Kosten-Konfiguration
        for model_key, costs in self._token_costs.items():
            if model_key in model_lower:
                input_cost = prompt_tokens * costs["input"]
                output_cost = completion_tokens * costs["output"]
                return input_cost + output_cost

        # Fallback: Durchschnittliche Kosten
        return (prompt_tokens + completion_tokens) * 0.00001

    def _send_to_logfire(self, metrics: LLMCallMetrics) -> None:
        """Sendet LLM-Call-Metriken an Logfire."""
        try:
            log_data = {
                "call_id": metrics.call_id,
                "model": metrics.model,
                "provider": metrics.provider,
                "call_type": metrics.call_type,
                "duration_ms": metrics.duration_ms,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
                "estimated_cost_usd": metrics.estimated_cost_usd,
                "tokens_per_second": metrics.tokens_per_second,
                "success": metrics.success,
                "agent_id": metrics.agent_id,
                "conversation_id": metrics.conversation_id,
            }

            # Füge Metadaten hinzu
            log_data.update(metrics.metadata)

            # Sende Log
            if metrics.success:
                self.logfire_manager.log_info(
                    f"LLM Call completed: {metrics.provider}/{metrics.model}",
                    **log_data
                )
            else:
                self.logfire_manager.log_error(
                    f"LLM Call failed: {metrics.error_type}",
                    error_message=metrics.error_message,
                    **log_data
                )

        except Exception as e:
            logger.warning(f"Fehler beim Senden an Logfire: {e}")

    def _integrate_with_agent_metrics(self, metrics: LLMCallMetrics) -> None:
        """Integriert mit der bestehenden Agent-Metrics-Infrastruktur."""
        try:
            if not metrics.agent_id:
                return

            # Versuche Agent-Metrics-Integration
            from .agent_metrics import get_agent_metrics_collector

            collector = get_agent_metrics_collector(metrics.agent_id)
            collector.record_tool_call(
                tool_name=f"{metrics.provider}_{metrics.model}",
                success=metrics.success,
                duration_ms=metrics.duration_ms or 0
            )

            # Zusätzliche LLM-spezifische Metriken
            if hasattr(collector, "record_llm_call"):
                collector.record_llm_call(
                    model=metrics.model,
                    provider=metrics.provider,
                    tokens=metrics.total_tokens or 0,
                    cost=metrics.estimated_cost_usd or 0.0,
                    success=metrics.success
                )

        except ImportError:
            # Agent-Metrics nicht verfügbar
            pass
        except Exception as e:
            logger.warning(f"Agent-Metrics-Integration fehlgeschlagen: {e}")

    def _add_to_history(self, metrics: LLMCallMetrics) -> None:
        """Fügt Call zur History hinzu."""
        self._call_history.append(metrics)

        # Begrenze History-Größe
        if len(self._call_history) > self._max_history:
            self._call_history = self._call_history[-self._max_history:]

    def get_active_calls(self) -> list[LLMCallMetrics]:
        """Gibt aktuell aktive Calls zurück."""
        return list(self._active_calls.values())

    def get_call_history(self, limit: int = 100) -> list[LLMCallMetrics]:
        """Gibt Call-History zurück."""
        return self._call_history[-limit:]

    def get_metrics_summary(self) -> dict[str, Any]:
        """Gibt eine Zusammenfassung der LLM-Metriken zurück."""
        recent_calls = self._call_history[-100:]

        if not recent_calls:
            return {"total_calls": 0}

        total_calls = len(recent_calls)
        successful_calls = sum(1 for call in recent_calls if call.success)
        total_tokens = sum(call.total_tokens or 0 for call in recent_calls)
        total_cost = sum(call.estimated_cost_usd or 0 for call in recent_calls)
        avg_duration = sum(call.duration_ms or 0 for call in recent_calls) / total_calls

        # Provider/Model-Statistiken
        providers = {}
        models = {}

        for call in recent_calls:
            providers[call.provider] = providers.get(call.provider, 0) + 1
            models[call.model] = models.get(call.model, 0) + 1

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_duration_ms": avg_duration,
            "active_calls": len(self._active_calls),
            "providers": providers,
            "models": models,
        }


# Globaler Tracker (Singleton)
_llm_tracker: LLMCallTracker | None = None


def get_llm_tracker() -> LLMCallTracker:
    """Gibt den globalen LLM-Tracker zurück (Singleton)."""
    global _llm_tracker
    if _llm_tracker is None:
        _llm_tracker = LLMCallTracker()
    return _llm_tracker


def track_llm_call(
    model: str,
    provider: str,
    call_type: str = "completion",
    **kwargs
) -> Callable:
    """Decorator für automatisches LLM-Call-Tracking.

    Args:
        model: LLM-Model-Name
        provider: Provider (openai, anthropic, etc.)
        call_type: Art des Calls
        **kwargs: Zusätzliche Metadaten

    Returns:
        Decorator-Funktion
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **func_kwargs):
                tracker = get_llm_tracker()
                try:
                    with tracker.track_llm_call(model, provider, call_type, **kwargs) as context:
                        result = await func(*args, **func_kwargs)

                        # Extrahiere Token-Usage falls verfügbar
                        if hasattr(result, "usage"):
                            usage = result.usage
                            context.update_tokens(
                                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                                completion_tokens=getattr(usage, "completion_tokens", 0),
                                total_tokens=getattr(usage, "total_tokens", 0)
                            )

                        return result
                except Exception:
                    # Fehler wird bereits im Context-Manager behandelt
                    raise
            return async_wrapper
        @wraps(func)
        def sync_wrapper(*args, **func_kwargs):
            tracker = get_llm_tracker()
            try:
                with tracker.track_llm_call(model, provider, call_type, **kwargs) as context:
                    result = func(*args, **func_kwargs)

                    # Extrahiere Token-Usage falls verfügbar
                    if hasattr(result, "usage"):
                        usage = result.usage
                        context.update_tokens(
                            prompt_tokens=getattr(usage, "prompt_tokens", 0),
                            completion_tokens=getattr(usage, "completion_tokens", 0),
                            total_tokens=getattr(usage, "total_tokens", 0)
                        )

                    return result
            except Exception:
                # Fehler wird bereits im Context-Manager behandelt
                raise
        return sync_wrapper
    return decorator
