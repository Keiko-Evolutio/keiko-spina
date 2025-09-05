"""Voice-Processing mit Session-Management und Konfigurations-Utilities."""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

# Paket-Metadaten
__version__ = "0.0.1"
__author__ = "Keiko Development Team"


# =============================================================================
# Komponenten-Importe
# =============================================================================

try:
    from voice.session import RealtimeSession
    _SESSION_AVAILABLE = True
except ImportError as e:
    logger.exception(f"Voice Session nicht verfügbar: {e}")
    RealtimeSession = None  # type: ignore[assignment]
    _SESSION_AVAILABLE = False

try:
    from voice.common.common import (
        get_default_configuration,
        get_default_configuration_data,
        load_prompty_config,
        load_prompty_file,
    )
    _CONFIG_AVAILABLE = True
except ImportError as e:
    logger.exception(f"Voice Configuration nicht verfügbar: {e}")
    get_default_configuration_data = None  # type: ignore[assignment]
    get_default_configuration = None  # type: ignore[assignment]
    load_prompty_file = None  # type: ignore[assignment]
    load_prompty_config = None  # type: ignore[assignment]
    _CONFIG_AVAILABLE = False


# =============================================================================
# Öffentliche API
# =============================================================================

def get_voice_system_status() -> dict[str, Any]:
    """Gibt den Voice-System-Status zurück."""
    return {
        "session_available": _SESSION_AVAILABLE,
        "config_available": _CONFIG_AVAILABLE,
        "healthy": _SESSION_AVAILABLE and _CONFIG_AVAILABLE,
    }


def is_voice_system_healthy() -> bool:
    """Prüft, ob das Voice-System funktionsfähig ist."""
    return _SESSION_AVAILABLE and _CONFIG_AVAILABLE


__all__ = [
    "RealtimeSession",
    "get_default_configuration",
    "get_default_configuration_data",
    "get_voice_system_status",
    "is_voice_system_healthy",
    "load_prompty_config",
    "load_prompty_file",
]
