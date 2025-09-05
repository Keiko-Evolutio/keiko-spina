"""Core Settings für Keiko Personal Assistant.

Basis-Konfiguration für Environment, Debug-Modus und Logging.
"""

from pydantic import Field
from pydantic_settings import BaseSettings

from .constants import DEFAULT_ENVIRONMENT, DEFAULT_LOG_LEVEL
from .env_utils import get_env_bool, get_env_str


class CoreSettings(BaseSettings):
    """Kern-Einstellungen für die Anwendung."""

    # Environment Configuration
    environment: str = Field(
        default=DEFAULT_ENVIRONMENT,
        description="Deployment-Umgebung (development, staging, production)"
    )

    debug_mode: bool = Field(
        default=True,
        description="Debug-Modus"
    )

    log_level: str = Field(
        default=DEFAULT_LOG_LEVEL,
        description="Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    class Config:
        """Pydantic-Konfiguration."""
        env_prefix = "KEIKO_"
        case_sensitive = False


def load_core_settings() -> CoreSettings:
    """Lädt Core Settings aus Umgebungsvariablen.

    Returns:
        CoreSettings-Instanz
    """
    return CoreSettings(
        environment=get_env_str("KEIKO_ENVIRONMENT", DEFAULT_ENVIRONMENT),
        debug_mode=get_env_bool("KEIKO_DEBUG_MODE", True),
        log_level=get_env_str("KEIKO_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    )


# Globale Core Settings Instanz
core_settings = load_core_settings()


__all__ = [
    "CoreSettings",
    "core_settings",
    "load_core_settings"
]
