"""Utility-Funktionen für Tracing und Monitoring.

Stellt Funktionen für Latenz-Messung, Response-Verarbeitung
und Metrics-Erstellung bereit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

# Konstanten
DEFAULT_PREVIEW_LENGTH = 2000  # Maximale Länge für Response-Previews (2KB)
DEFAULT_LATENCY_VALUE = 0.0    # Fallback-Wert für Latenz-Messungen in Sekunden


class LatencyTracker:
    """Klasse für Latenz-Messung."""

    def __init__(self) -> None:
        """Initialisiert den Tracker."""
        self._start_time: float | None = None

    def start(self) -> None:
        """Startet die Latenz-Messung."""
        self._start_time = time.time()

    def get_latency_seconds(self) -> float:
        """Berechnet die Latenz seit dem Start.

        Returns:
            Latenz in Sekunden oder 0.0 falls kein Start-Zeitpunkt verfügbar.
        """
        if self._start_time is not None:
            return max(DEFAULT_LATENCY_VALUE, time.time() - self._start_time)
        return DEFAULT_LATENCY_VALUE

    def get_latency_milliseconds(self) -> float:
        """Berechnet die Latenz in Millisekunden.

        Returns:
            Latenz in Millisekunden.
        """
        return self.get_latency_seconds() * 1000

    def reset(self) -> None:
        """Setzt den Tracker zurück."""
        self._start_time = None


def create_safe_preview(
    content: str | dict[str, Any] | Any,
    max_length: int = DEFAULT_PREVIEW_LENGTH
) -> str:
    """Erstellt eine sichere Vorschau von beliebigem Content.

    Args:
        content: Zu verarbeitender Content.
        max_length: Maximale Länge der Vorschau.

    Returns:
        Gekürzte und sichere Content-Vorschau.
    """
    try:
        if isinstance(content, str):
            preview = content
        else:
            preview = json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError, UnicodeDecodeError):
        preview = str(content)

    return preview[:max_length] if len(preview) > max_length else preview


def create_token_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    prefix: str = "token"
) -> dict[str, int]:
    """Erstellt standardisierte Token-Metrics für LLM-Aufrufe.

    Args:
        prompt_tokens: Anzahl Eingabetoken (≥0).
        completion_tokens: Anzahl Ausgabetoken (≥0).
        prefix: Prefix für Metric-Namen (Standard: "token").

    Returns:
        Dictionary mit Token-Metrics.

    Example:
        >>> create_token_metrics(100, 50)
        {'token.prompt': 100, 'token.completion': 50, 'token.total': 150}
    """
    return {
        f"{prefix}.prompt": int(prompt_tokens),
        f"{prefix}.completion": int(completion_tokens),
        f"{prefix}.total": int(prompt_tokens + completion_tokens),
    }


def create_success_error_metrics(
    success: bool,
    prefix: str = "agent"
) -> dict[str, int]:
    """Erstellt standardisierte Success/Error-Metrics.

    Args:
        success: Ob die Operation erfolgreich war.
        prefix: Prefix für Metric-Namen.

    Returns:
        Dictionary mit Success/Error-Metrics.
    """
    success_value = 1 if success else 0
    error_value = 0 if success else 1

    return {
        f"{prefix}.success": success_value,
        f"{prefix}.error": error_value,
        f"{prefix}.success_rate": success_value,
        f"{prefix}.error_rate": error_value,
    }


def safe_get_error_code(error: Exception) -> str:
    """Extrahiert sicher einen Error-Code aus einer Exception.

    Args:
        error: Exception-Objekt.

    Returns:
        Error-Code oder Exception-Klassenname.
    """
    return getattr(error, "code", error.__class__.__name__)


class MetricsBuilder:
    """Builder für Metrics-Erstellung.

    Fluent API für das Zusammenstellen von Metrics-Dictionaries.

    Example:
        >>> builder = MetricsBuilder()
        >>> metrics = (builder
        ...     .add_tokens(100, 50)
        ...     .add_latency(1.5)
        ...     .add_success_error(True)
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialisiert den Builder."""
        self._metrics: dict[str, Any] = {}

    def add_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        prefix: str = "token"
    ) -> MetricsBuilder:
        """Fügt Token-Metrics hinzu.

        Args:
            prompt_tokens: Anzahl Eingabetoken.
            completion_tokens: Anzahl Ausgabetoken.
            prefix: Prefix für Metric-Namen.

        Returns:
            Self für Method-Chaining.
        """
        self._metrics.update(create_token_metrics(prompt_tokens, completion_tokens, prefix))
        return self

    def add_success_error(
        self,
        success: bool,
        prefix: str = "agent"
    ) -> MetricsBuilder:
        """Fügt Success/Error-Metrics hinzu.

        Args:
            success: Ob die Operation erfolgreich war.
            prefix: Prefix für Metric-Namen.

        Returns:
            Self für Method-Chaining.
        """
        self._metrics.update(create_success_error_metrics(success, prefix))
        return self

    def add_latency(
        self,
        latency_seconds: float,
        key: str = "latency.seconds"
    ) -> MetricsBuilder:
        """Fügt Latenz-Metric hinzu.

        Args:
            latency_seconds: Latenz in Sekunden.
            key: Metric-Key.

        Returns:
            Self für Method-Chaining.
        """
        self._metrics[key] = latency_seconds
        return self

    def add_custom(self, key: str, value: Any) -> MetricsBuilder:
        """Fügt benutzerdefinierte Metric hinzu.

        Args:
            key: Metric-Key.
            value: Metric-Wert.

        Returns:
            Self für Method-Chaining.
        """
        self._metrics[key] = value
        return self

    def build(self) -> dict[str, Any]:
        """Erstellt das finale Metrics-Dictionary.

        Returns:
            Metrics-Dictionary.
        """
        return self._metrics.copy()


def safe_execute_with_fallback(
    operation_name: str,
    operation_func: Callable,
    fallback_value: Any = None,
    log_errors: bool = True
) -> Any:
    """Führt eine Operation sicher aus mit Fallback-Verhalten.

    Args:
        operation_name: Name der Operation für Logging.
        operation_func: Auszuführende Funktion.
        fallback_value: Rückgabewert bei Fehlern.
        log_errors: Ob Fehler geloggt werden sollen.

    Returns:
        Ergebnis der Operation oder Fallback-Wert.
    """
    try:
        return operation_func()
    except (ConnectionError, TimeoutError) as exc:
        if log_errors:
            logger.warning("Verbindungsfehler in %s: %s", operation_name, exc)
        return fallback_value
    except Exception as exc:
        if log_errors:
            logger.debug("%s ignoriert: %s", operation_name, exc)
        return fallback_value


__all__ = [
    "DEFAULT_LATENCY_VALUE",
    "DEFAULT_PREVIEW_LENGTH",
    "LatencyTracker",
    "MetricsBuilder",
    "create_safe_preview",
    "create_success_error_metrics",
    "create_token_metrics",
    "safe_execute_with_fallback",
    "safe_get_error_code",
]
