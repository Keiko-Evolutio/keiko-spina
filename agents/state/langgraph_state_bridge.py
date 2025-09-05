"""State-Bridge für LangGraph-Workflows.

Erlaubt die Konvertierung zwischen generischen Dict-Zuständen und einer klar
typisierten Python-Repräsentation mit konsolidierten Reducer-Funktionen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from .state_constants import DEFAULT_EMPTY_MESSAGE, DEFAULT_STEP_VALUE
from .state_utils import (
    SerializationMixin,
    create_max_reducer,
    create_replace_reducer,
    serialize_state_safely,
)

# Konsolidierte Reducer-Funktionen
_generic_replace_reducer = create_replace_reducer()

# Typisierte Aliases für Backward-Compatibility
replace_message = _generic_replace_reducer
replace_bool = _generic_replace_reducer
replace_int = _generic_replace_reducer
replace_dict = _generic_replace_reducer

max_step = create_max_reducer()





@dataclass
class WorkflowState(SerializationMixin):
    """Typisierter Workflow-Zustand für LangChain/LangGraph-Integration.

    Diese Klasse stellt eine typisierte Repräsentation des Workflow-Zustands
    bereit und unterstützt automatische Serialisierung/Deserialisierung.

    Attributes:
        message: Aktuelle Workflow-Nachricht mit Replace-Reducer
        step: Aktueller Workflow-Schritt mit Max-Reducer (monoton steigend)

    Examples:
        >>> state = WorkflowState(message="Hallo", step=1)
        >>> data = state.to_dict()
        >>> restored = WorkflowState.from_dict(data)
    """

    message: Annotated[str, replace_message]
    step: Annotated[int, max_step] = DEFAULT_STEP_VALUE

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Workflow-Zustand in ein Dict.

        Returns:
            Dict mit allen State-Feldern

        Raises:
            ValueError: Bei Serialisierungsfehlern
        """
        return serialize_state_safely(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        """Erstellt WorkflowState aus einem Dict mit verbesserter Validierung.

        Args:
            data: Dict mit State-Daten

        Returns:
            Neue WorkflowState-Instanz

        Raises:
            ValueError: Bei ungültigen Daten oder Deserialisierungsfehlern
        """
        # Validierung zuerst - muss ein Dict sein
        if not isinstance(data, dict):
            raise ValueError("Data muss ein Dict sein")

        # Fallback auf Standard-Werte bei fehlenden Feldern
        validated_data = {
            "message": str(data.get("message", DEFAULT_EMPTY_MESSAGE)),
            "step": int(data.get("step", DEFAULT_STEP_VALUE))
        }

        return cls(**validated_data)


__all__ = [
    "WorkflowState",
    "max_step",
    "replace_bool",
    "replace_dict",
    "replace_int",
    "replace_message",
]
