# backend/data_models/core/__init__.py
"""Core Models Package – Domain-Objekte und Update-System."""

from __future__ import annotations

# Paket-Metadaten
from ..constants import LOG_MSG_CORE_MODELS_INITIALIZED, PACKAGE_AUTHOR, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__author__ = PACKAGE_AUTHOR

# Logging-Setup
from kei_logging import get_logger

logger = get_logger(__name__)

# Core Models importieren
from .core import (
    # Domain-Objekte
    Agent,
    # Update-System
    AgentUpdate,
    # Event-System
    AgentUpdateEvent,
    AudioUpdate,
    Configuration,
    ConsoleUpdate,
    Content,
    DefaultConfiguration,
    ErrorUpdate,
    Function,
    FunctionCompletionUpdate,
    FunctionParameter,
    FunctionUpdate,
    MessageUpdate,
    Role,
    SettingsUpdate,
    Update,
    UpdateFactory,
    UpdateType,
)

CORE_MODELS_AVAILABLE = True
from ..constants import LOG_MSG_CORE_MODELS_LOADED

logger.info(LOG_MSG_CORE_MODELS_LOADED)

# Explizite Exports
__all__ = [
    # Domain-Objekte
    "Agent",
    # Update-System
    "AgentUpdate",
    # Event-System
    "AgentUpdateEvent",
    "AudioUpdate",
    "Configuration",
    "ConsoleUpdate",
    "Content",
    "DefaultConfiguration",
    "ErrorUpdate",
    "Function",
    "FunctionCompletionUpdate",
    "FunctionParameter",
    "FunctionUpdate",
    "MessageUpdate",
    "Role",
    "SettingsUpdate",
    "Update",
    "UpdateFactory",
    "UpdateType",
]

# Status-Logging
logger.info(f"{LOG_MSG_CORE_MODELS_INITIALIZED} - {len(__all__)} Models verfügbar")
