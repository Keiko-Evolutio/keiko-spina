"""Utility-Funktionen für das Agent-Routing-Modul."""

from __future__ import annotations

from typing import Any


def normalize_string_list(items: list[str] | None) -> set[str]:
    """Normalisiert eine Liste von Strings zu einem lowercase Set.

    Args:
        items: Liste von Strings oder None.

    Returns:
        Set von normalisierten Strings.
    """
    if not items:
        return set()
    return {item.lower() for item in items}


def calculate_intersection_score(
    required: set[str],
    available: set[str]
) -> int:
    """Berechnet den Intersection-Score zwischen zwei Sets.

    Args:
        required: Set von erforderlichen Items.
        available: Set von verfügbaren Items.

    Returns:
        Anzahl der Übereinstimmungen.
    """
    return len(required.intersection(available))


def validate_agent_structure(agent: dict[str, Any]) -> bool:
    """Validiert die Grundstruktur eines Agent-Deskriptors.

    Args:
        agent: Agent-Deskriptor Dictionary.

    Returns:
        True wenn die Struktur gültig ist.
    """
    if not isinstance(agent, dict):
        return False

    if "id" not in agent:
        return False

    if "capabilities" in agent and not isinstance(agent["capabilities"], list):
        return False

    return True


__all__ = [
    "calculate_intersection_score",
    "normalize_string_list",
    "validate_agent_structure",
]
