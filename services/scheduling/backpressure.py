"""Backpressure-Entscheidung für Scheduling (Push vs Pull).

Nutzt Agent-Heartbeat-Hinweise (queue_length, desired_concurrency, max_queue_length)
und lokale In-Flight-Metriken, um die geeignete Scheduling-Strategie zu wählen.
"""

from __future__ import annotations

from .config import DEFAULT_SCHEDULER_CONFIG, BackpressureConfig


def decide_mode(
    *,
    agent_hints: dict | None,
    inflight_for_agent: int,
    config: BackpressureConfig | None = None,
    default_mode: str | None = None,
) -> str:
    """Entscheidet zwischen "push" (direkt) und "pull" (queue).

    Args:
        agent_hints: Agent-Heartbeat-Hinweise mit queue_length, max_queue_length, desired_concurrency
        inflight_for_agent: Anzahl aktuell laufender Tasks für den Agent
        config: Backpressure-Konfiguration (Default: globale Konfiguration)
        default_mode: Standard-Modus (überschreibt config.default_mode)

    Returns:
        "push" für direkte Ausführung, "pull" für Queue-basierte Ausführung

    Heuristik:
        - Wenn max_queue_length gesetzt und queue_length/max_queue_length >= threshold → pull
        - Wenn desired_concurrency gesetzt und inflight >= desired_concurrency → pull
        - Sonst default_mode
    """
    bp_config = config or DEFAULT_SCHEDULER_CONFIG.backpressure
    effective_default = default_mode or bp_config.default_mode
    hints = agent_hints or {}

    # Queue-basierte Entscheidung
    queue_length = _safe_int_extract(hints, "queue_length")
    max_queue_length = _safe_int_extract(hints, "max_queue_length")

    if queue_length is not None and max_queue_length and max_queue_length > 0:
        utilization_ratio = queue_length / float(max_queue_length)
        if utilization_ratio >= bp_config.queue_utilization_threshold:
            return "pull"

    # Concurrency-basierte Entscheidung
    desired_concurrency = _safe_int_extract(hints, "desired_concurrency")
    if desired_concurrency is not None and inflight_for_agent >= desired_concurrency:
        return "pull"

    return effective_default


def _safe_int_extract(hints: dict, key: str) -> int | None:
    """Extrahiert sicher einen Integer-Wert aus Hints.

    Args:
        hints: Dictionary mit Hint-Werten
        key: Schlüssel für den zu extrahierenden Wert

    Returns:
        Integer-Wert oder None bei Fehlern
    """
    try:
        value = hints.get(key)
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


__all__ = ["decide_mode"]
