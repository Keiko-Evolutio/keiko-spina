"""Zentrale Kompatibilitäts-Helfer für das `agents`-Paket.

Stellt generische Fallback-Funktionen und -Klassen bereit, um
Import-Fehler abzufangen, ohne die öffentliche API zu brechen.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)


def unavailable_function_factory(description: str = "Funktion") -> Callable[..., Any]:
    """Erzeugt eine Fallback-Funktion, die bei Aufruf einen Fehler auslöst.

    Args:
        description: Bezeichnung des nicht verfügbaren Elements

    Returns:
        Callable, das bei Ausführung einen RuntimeError wirft
    """

    def _unavailable(*_: Any, **__: Any) -> Any:
        raise RuntimeError(f"{description} ist nicht verfügbar")

    # Kennzeichnet Fallbacks zur Verfügbarkeitsprüfung
    _unavailable._is_compat_fallback = True
    return _unavailable


class UnavailableClass:
    """Fallback-Klasse, deren Instanziierung sofort fehlschlägt.

    Eignet sich als Platzhalter für nicht verfügbare Klassen.
    """

    # Kennzeichnet Fallbacks zur Verfügbarkeitsprüfung
    _is_compat_fallback: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Angeforderte Klasse ist nicht verfügbar")


def safe_import(module_path: str, names: Iterable[str], *, label: str) -> tuple[Any, ...]:
    """Versucht, angegebene Symbole zu importieren und liefert Fallbacks bei Fehlern.

    Args:
        module_path: Modulpfad (z. B. "agents.common")
        names: Zu importierende Symbolnamen
        label: Menschlich lesbares Label für Log-Ausgaben

    Returns:
        Tupel mit importierten Symbolen in der Reihenfolge von `names`.
        Bei ImportError wird für jedes Symbol eine Fallback-Funktion geliefert,
        die bei Aufruf einen Fehler auslöst.
    """
    try:
        module = __import__(module_path, fromlist=list(names))
        return tuple(getattr(module, n) for n in names)
    except ImportError as e:
        logger.debug(f"{label} nicht verfügbar: {e}")
        return tuple(unavailable_function_factory(f"{label}: {n}") for n in names)
