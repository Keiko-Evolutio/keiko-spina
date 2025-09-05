"""Utility-Funktionen für das State-Management-Modul."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from kei_logging import get_logger
from observability import record_exception_in_span, trace_span

from .state_constants import (
    INVALID_CONFIG_ERROR,
    INVALID_STATE_ERROR,
    TRACE_STATE_DESERIALIZATION,
    TRACE_STATE_SERIALIZATION,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

T = TypeVar("T")
StateType = TypeVar("StateType")


class SerializableState(Protocol):
    """Protokoll für serialisierbare State-Objekte."""

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den State zu einem Dict."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerializableState:
        """Erstellt State-Objekt aus Dict."""
        ...


def create_replace_reducer() -> Callable[[T, T], T]:
    """Erstellt einen generischen Replace-Reducer.

    Der Reducer ersetzt den linken Wert immer durch den rechten Wert.
    Verwendung für: Strings, Booleans, Integers, Listen, Dicts.

    Returns:
        Reducer-Funktion die den rechten Wert zurückgibt
    """

    def replace_reducer(_left: T, right: T) -> T:
        """Ersetzt linken Wert durch rechten Wert."""
        return right

    return replace_reducer


def create_optional_replace_reducer() -> Callable[[T | None, T | None], T | None]:
    """Erstellt einen Replace-Reducer für Optional-Werte.

    Der Reducer ersetzt nur bei nicht-None rechten Werten.

    Returns:
        Reducer-Funktion die None-sichere Ersetzung durchführt
    """

    def optional_replace_reducer(left: T | None, right: T | None) -> T | None:
        """Ersetzt nur bei nicht-None rechten Werten."""
        return right if right is not None else left

    return optional_replace_reducer


def create_max_reducer() -> Callable[[int | float, int | float], int | float]:
    """Erstellt einen Max-Reducer für numerische Werte.

    Der Reducer gibt den größeren der beiden Werte zurück.
    Verwendung für: Step-Counter, Timestamps, Scores.

    Returns:
        Reducer-Funktion die den maximalen Wert zurückgibt
    """

    def max_reducer(left: int | float, right: int | float) -> int | float:
        """Gibt den größeren Wert zurück."""
        return max(left, right)

    return max_reducer


# =============================================================================
# State-Serialisierung Utilities
# =============================================================================


def serialize_state_safely(state: SerializableState) -> dict[str, Any]:
    """Serialisiert State-Objekt sicher mit Tracing.

    Args:
        state: State-Objekt das serialisiert werden soll

    Returns:
        Serialisiertes Dict

    Raises:
        ValueError: Bei Serialisierungsfehlern
    """
    with trace_span(TRACE_STATE_SERIALIZATION):
        try:
            result = state.to_dict()
            logger.debug(f"State erfolgreich serialisiert: {len(result)} Felder")
            return result
        except Exception as e:
            error_msg = INVALID_STATE_ERROR.format(details=str(e))
            record_exception_in_span(e, attributes={"operation": "serialize"})
            logger.exception(error_msg)
            raise ValueError(error_msg) from e


from typing import TypeVar

StateType = TypeVar('StateType')

def deserialize_state_safely(
    state_class: type[StateType], data: dict[str, Any]
) -> StateType:
    """Deserialisiert State-Objekt sicher mit Tracing.

    Args:
        state_class: Klasse des zu erstellenden State-Objekts
        data: Dict mit State-Daten

    Returns:
        Deserialisiertes State-Objekt

    Raises:
        ValueError: Bei Deserialisierungsfehlern
    """
    with trace_span(TRACE_STATE_DESERIALIZATION):
        try:
            # Basis-Validierung hier für Konsistenz
            if not isinstance(data, dict):
                raise ValueError("Data muss ein Dict sein")

            result = state_class.from_dict(data)
            class_name = getattr(state_class, "__name__", str(state_class))
            logger.debug(f"State erfolgreich deserialisiert: {class_name}")
            return result
        except Exception as e:
            error_msg = INVALID_STATE_ERROR.format(details=str(e))
            record_exception_in_span(e, attributes={"operation": "deserialize"})
            logger.exception(error_msg)
            raise ValueError(error_msg) from e


# =============================================================================
# Exception-Handler Utilities
# =============================================================================


async def handle_workflow_operation[T](
    operation: Callable[[], Awaitable[T]],
    context: str,
    **trace_attributes: Any,
) -> T:
    """Führt Workflow-Operation mit einheitlichem Exception-Handling aus.

    Args:
        operation: Async-Operation die ausgeführt werden soll
        context: Kontext-String für Logging und Tracing
        **trace_attributes: Zusätzliche Trace-Attribute

    Returns:
        Ergebnis der Operation

    Raises:
        Exception: Ursprüngliche Exception mit verbessertem Logging
    """
    with trace_span(f"workflow.{context}", trace_attributes):
        try:
            result = await operation()
            logger.debug(f"Workflow-Operation erfolgreich: {context}")
            return result
        except Exception as e:
            error_msg = f"Workflow-Operation fehlgeschlagen ({context}): {e}"
            record_exception_in_span(e, attributes={"context": context})
            logger.exception(error_msg)
            raise


# =============================================================================
# Validierungs-Utilities
# =============================================================================


def validate_workflow_config(config: dict[str, Any]) -> None:
    """Validiert Workflow-Konfiguration.

    Args:
        config: Konfiguration die validiert werden soll

    Raises:
        ValueError: Bei ungültiger Konfiguration
    """
    if not isinstance(config, dict):
        raise ValueError(INVALID_CONFIG_ERROR.format(details="Config muss ein Dict sein"))

    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        raise ValueError(INVALID_CONFIG_ERROR.format(details="configurable muss ein Dict sein"))

    thread_id = configurable.get("thread_id")
    if not thread_id or not isinstance(thread_id, str):
        raise ValueError(INVALID_CONFIG_ERROR.format(details="thread_id ist erforderlich"))


def validate_workflow_name(name: str) -> None:
    """Validiert Workflow-Namen.

    Args:
        name: Workflow-Name der validiert werden soll

    Raises:
        ValueError: Bei ungültigem Namen
    """
    if not name or not isinstance(name, str):
        raise ValueError("Workflow-Name muss ein nicht-leerer String sein")

    if len(name.strip()) == 0:
        raise ValueError("Workflow-Name darf nicht leer sein")


# =============================================================================
# Mixin-Klassen für Code-Wiederverwendung
# =============================================================================


class SerializationMixin:
    """Mixin für Dict-Serialisierung von Dataclasses.

    Stellt to_dict() und from_dict() Methoden mit verbesserter
    Validierung und Error-Handling bereit.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert Dataclass zu Dict mit sicherer Konvertierung."""
        from dataclasses import asdict, is_dataclass

        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} ist keine Dataclass. "
                          "SerializationMixin kann nur mit @dataclass dekorierte Klassen verwendet werden.")
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Erstellt Dataclass-Instanz aus Dict mit Validierung.

        Args:
            data: Dict mit Daten für die Dataclass

        Returns:
            Neue Instanz der Dataclass

        Raises:
            ValueError: Bei ungültigen Daten
        """
        if not isinstance(data, dict):
            raise ValueError("Data muss ein Dict sein")

        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(f"Ungültige Parameter für {cls.__name__}: {e}") from e
