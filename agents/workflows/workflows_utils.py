"""Gemeinsame Utility-Funktionen für Workflow-Module.

Dieses Modul stellt wiederverwendbare Funktionen bereit, die von mehreren
Workflow-Komponenten genutzt werden, um Code-Duplikation zu vermeiden.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


def run_sync(awaitable: Any) -> Any:
    """Führt ein Awaitable synchron aus unter Berücksichtigung laufender Event-Loops.

    Diese Funktion behandelt sowohl den Fall, dass bereits ein Event-Loop läuft
    (z.B. in LangGraph-Nodes), als auch den Fall ohne laufenden Loop.

    Args:
        awaitable: Das auszuführende Awaitable

    Returns:
        Das Ergebnis des Awaitable

    Raises:
        ServiceError: Wenn ein Coroutine in einem laufenden Event Loop ausgeführt werden soll
        RuntimeError: Bei anderen Event-Loop-Problemen

    Examples:
        >>> async def example():
        ...     return "result"
        >>> result = run_sync(example())
        >>> assert result == "result"
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            from app.common.error_handlers import ServiceError
            raise ServiceError(
                "Kann Coroutine nicht im laufenden Event Loop ausführen"
            )
        return loop.run_until_complete(awaitable)
    except RuntimeError:
        return asyncio.run(awaitable)


def handle_langgraph_import() -> tuple[Any, Any]:
    """Behandelt LangGraph-Import mit Fallback für Test-/Development-Umgebungen.

    Returns:
        Tuple aus (StateGraph, END) - entweder echte LangGraph-Objekte oder Fallbacks

    Examples:
        >>> StateGraph, END = handle_langgraph_import()
        >>> # StateGraph ist entweder langgraph.StateGraph oder None
        >>> # END ist entweder langgraph.END oder ein object()
    """
    try:  # pragma: no cover - optional zur Laufzeit
        from langgraph.graph import END, StateGraph
        return StateGraph, END
    except ImportError:  # pragma: no cover - test/runtime ohne langgraph
        return None, object()  # type: ignore
    except Exception as e:  # pragma: no cover - unerwarteter Import-Fehler
        logger.debug(f"Unerwarteter Fehler beim LangGraph-Import: {e}")
        return None, object()  # type: ignore


def manipulate_state_extras(state: Any, key: str, value: Any) -> None:
    """Sichere State-Extras-Manipulation mit Exception-Handling.

    Diese Funktion versucht, einen Wert in den 'extras' Dict eines State-Objekts
    zu setzen. Falls das State-Objekt kein 'extras' Attribut hat oder es kein
    Dict ist, wird es initialisiert.

    Args:
        state: Das State-Objekt (z.B. OrchestrationState)
        key: Der Schlüssel für den Wert
        value: Der zu setzende Wert

    Examples:
        >>> class MockState:
        ...     def __init__(self):
        ...         self.extras = {}
        >>> state = MockState()
        >>> manipulate_state_extras(state, "test_key", "test_value")
        >>> assert state.extras["test_key"] == "test_value"
    """
    try:
        extras = getattr(state, "extras", {})
        if isinstance(extras, dict):
            extras[key] = value
            state.extras = extras
        else:
            # Extras ist kein Dict - initialisiere neu
            state.extras = {key: value}
    except Exception:
        # Fallback: Versuche direkte Zuweisung
        try:
            state.extras = {key: value}
        except Exception:
            # Letzter Fallback: Ignoriere den Fehler
            pass


def safe_exception_handler(
    operation: str,
    error: Exception,
    logger_instance: Any | None = None,
    fallback_value: Any = None
) -> Any:
    """Standardisiertes Exception-Handling für Workflow-Operationen.

    Args:
        operation: Beschreibung der fehlgeschlagenen Operation
        error: Die aufgetretene Exception
        logger_instance: Optionaler spezifischer Logger (Standard: module logger)
        fallback_value: Optionaler Fallback-Wert der zurückgegeben wird

    Returns:
        Der fallback_value (falls angegeben)

    Examples:
        >>> try:
        ...     raise ValueError("Test error")
        ... except Exception as e:
        ...     result = safe_exception_handler("test operation", e, fallback_value="fallback")
        >>> assert result == "fallback"
    """
    log = logger_instance or logger
    log.warning(f"{operation} fehlgeschlagen: {error}")
    return fallback_value


def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Sichere Attribut-Abfrage mit Fallback.

    Args:
        obj: Das Objekt
        attr: Der Attribut-Name
        default: Fallback-Wert

    Returns:
        Der Attribut-Wert oder default

    Examples:
        >>> class TestObj:
        ...     test_attr = "value"
        >>> obj = TestObj()
        >>> assert safe_getattr(obj, "test_attr") == "value"
        >>> assert safe_getattr(obj, "missing_attr", "default") == "default"
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def safe_setattr(obj: Any, attr: str, value: Any) -> bool:
    """Sichere Attribut-Zuweisung mit Exception-Handling.

    Args:
        obj: Das Objekt
        attr: Der Attribut-Name
        value: Der zu setzende Wert

    Returns:
        True wenn erfolgreich, False bei Fehler

    Examples:
        >>> class TestObj:
        ...     pass
        >>> obj = TestObj()
        >>> assert safe_setattr(obj, "test_attr", "value") == True
        >>> assert getattr(obj, "test_attr") == "value"  # Dynamic attribute
    """
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def ensure_dict(value: Any, default_key: str | None = None) -> dict:
    """Stellt sicher, dass ein Wert ein Dictionary ist.

    Args:
        value: Der zu prüfende Wert
        default_key: Optionaler Schlüssel falls value kein Dict ist

    Returns:
        Ein Dictionary

    Examples:
        >>> assert ensure_dict({}) == {}
        >>> assert ensure_dict("test", "key") == {"key": "test"}
        >>> assert ensure_dict(None) == {}
    """
    if isinstance(value, dict):
        return value
    if value is not None and default_key:
        return {default_key: value}
    return {}


def validate_workflow_name(name: str) -> bool:
    """Validiert einen Workflow-Namen.

    Args:
        name: Der zu validierende Name

    Returns:
        True wenn gültig, False sonst

    Examples:
        >>> assert validate_workflow_name("valid_workflow") == True
        >>> assert validate_workflow_name("") == False
        >>> assert validate_workflow_name("invalid-name!") == False
    """
    if not isinstance(name, str) or not name.strip():
        return False

    # Einfache Validierung: Alphanumerisch + Underscore, mindestens 1 Zeichen
    import re
    pattern = r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, name.strip()))


__all__ = [
    "ensure_dict",
    "handle_langgraph_import",
    "manipulate_state_extras",
    "run_sync",
    "safe_exception_handler",
    "safe_getattr",
    "safe_setattr",
    "validate_workflow_name",
]
