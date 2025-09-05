# backend/data_models/core/core.py
"""Kern-Datenstrukturen für Agent-System und WebSocket-Kommunikation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

# Logging-Setup
from kei_logging import get_logger

logger = get_logger(__name__)

# OpenAI SDK Import

# Imports von gemeinsamen Utilities und Konstanten
from ..constants import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_CATEGORY,
    DEFAULT_ERROR_CODE,
    DEFAULT_ERROR_MESSAGE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_SEVERITY,
    DEFAULT_STATUS,
    # Default Values
    DEFAULT_VERSION,
    # Field Names
    FIELD_NAME_CONFIGURATION_NAME,
    FIELD_NAME_DEFAULT_CONFIGURATION_INSTRUCTIONS,
    FIELD_NAME_FUNCTION_DESCRIPTION,
    FIELD_NAME_FUNCTION_NAME,
    FIELD_NAME_FUNCTION_PARAMETER_NAME,
    FIELD_NAME_FUNCTION_PARAMETER_TYPE,
    # Log Messages
    LOG_MSG_EMPTY_CONTENT_WARNING,
    LOG_MSG_EMPTY_FUNCTION_NAME_WARNING,
    ROLE_ASSISTANT,
    ROLE_ORCHESTRATOR,
    ROLE_SPECIALIST,
    ROLE_SYSTEM,
    # Role Values
    ROLE_USER,
    # Update Types
    UPDATE_TYPE_AGENT,
    UPDATE_TYPE_AUDIO,
    UPDATE_TYPE_CONSOLE,
    UPDATE_TYPE_ERROR,
    UPDATE_TYPE_FUNCTION,
    UPDATE_TYPE_FUNCTION_COMPLETION,
    UPDATE_TYPE_INTERRUPT,
    UPDATE_TYPE_MESSAGE,
    UPDATE_TYPE_SETTINGS,
)
from ..utils import (
    ValidationMixin,
    ensure_dict_default,
    ensure_list_default,
    generate_uuid,
    utc_now,
    validate_non_empty_string,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from openai.types.beta.realtime.session_update_event import SessionTool

# Domain-Objekte

@dataclass(slots=True)
class Configuration(ValidationMixin):
    """Konfiguration aus Prompty-Datei oder Azure AI Foundry."""
    id: str
    name: str
    category: str
    default: bool
    content: str
    tools: list[dict[str, Any]] = field(default_factory=ensure_list_default)

    def __post_init__(self) -> None:
        """Validierung der Konfiguration."""
        if not self.id:
            self.id = generate_uuid()
        validate_non_empty_string(self.name, FIELD_NAME_CONFIGURATION_NAME)


@dataclass(slots=True)
class DefaultConfiguration(ValidationMixin):
    """Finale, gerenderte Konfiguration für Session."""
    id: str
    instructions: str
    tools: list[SessionTool] = field(default_factory=ensure_list_default)

    def __post_init__(self) -> None:
        """Validierung der Default-Konfiguration."""
        if not self.id:
            self.id = generate_uuid()
        validate_non_empty_string(self.instructions, FIELD_NAME_DEFAULT_CONFIGURATION_INSTRUCTIONS)


@dataclass(slots=True)
class Agent:
    """Agent-Vertrag: Identität, Metadaten und Status.

    Enthält optionale Felder für Owner, Tenant und Tags zur Unterstützung
    von Governance und Mandantentrennung.
    """

    id: str
    name: str
    type: str
    description: str
    status: str
    capabilities: list[str] = field(default_factory=ensure_list_default)
    parameters: list[dict[str, Any]] = field(default_factory=ensure_list_default)
    version: str = DEFAULT_VERSION
    owner: str | None = None
    tenant: str | None = None
    tags: list[str] = field(default_factory=ensure_list_default)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Initialisierung der Agent-Metadaten."""
        if not self.id:
            self.id = generate_uuid()
        if not self.created_at:
            self.created_at = utc_now()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass(slots=True)
class Function(ValidationMixin):
    """Funktions-Definition mit Parametern für Azure AI Foundry Tools."""
    name: str
    description: str
    parameters: list[FunctionParameter] = field(default_factory=ensure_list_default)
    func: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        """Validierung der Funktions-Definition."""
        validate_non_empty_string(self.name, FIELD_NAME_FUNCTION_NAME)
        validate_non_empty_string(self.description, FIELD_NAME_FUNCTION_DESCRIPTION)


@dataclass(slots=True)
class FunctionParameter(ValidationMixin):
    """Parameter-Spezifikation für Funktionen."""
    name: str
    type: str
    description: str
    required: bool = True
    default_value: Any | None = None

    def __post_init__(self) -> None:
        """Validierung der Parameter-Spezifikation."""
        validate_non_empty_string(self.name, FIELD_NAME_FUNCTION_PARAMETER_NAME)
        validate_non_empty_string(self.type, FIELD_NAME_FUNCTION_PARAMETER_TYPE)


@dataclass(slots=True)
class Content:
    """Strukturierter Inhalt für Updates."""
    type: str
    data: dict[str, Any] = field(default_factory=ensure_dict_default)
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Initialisierung des Contents."""
        if not self.timestamp:
            self.timestamp = utc_now()


# Update-System

class UpdateType(enum.Enum):
    """Update-Kategorien für WebSocket-Kommunikation."""
    AGENT = UPDATE_TYPE_AGENT
    AUDIO = UPDATE_TYPE_AUDIO
    CONSOLE = UPDATE_TYPE_CONSOLE
    ERROR = UPDATE_TYPE_ERROR
    FUNCTION = UPDATE_TYPE_FUNCTION
    FUNCTION_COMPLETION = UPDATE_TYPE_FUNCTION_COMPLETION
    MESSAGE = UPDATE_TYPE_MESSAGE
    SETTINGS = UPDATE_TYPE_SETTINGS
    INTERRUPT = UPDATE_TYPE_INTERRUPT


class Role(enum.Enum):
    """Chat-Rollen für Multi-Agent-Systeme."""
    USER = ROLE_USER
    ASSISTANT = ROLE_ASSISTANT
    SYSTEM = ROLE_SYSTEM
    ORCHESTRATOR = ROLE_ORCHESTRATOR
    SPECIALIST = ROLE_SPECIALIST


@dataclass(slots=True, frozen=True)
class Update:
    """Basis-Klasse für WebSocket-Updates."""
    update_id: str
    type: UpdateType
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Generiere Update-ID falls nicht vorhanden."""
        if not self.update_id:
            object.__setattr__(self, "update_id", generate_uuid())


@dataclass(slots=True, frozen=True)
class ConsoleUpdate(Update):
    """Konsolen-Ausgaben und Debug-Informationen."""
    payload: dict[str, Any] = field(default_factory=ensure_dict_default)
    level: str = DEFAULT_LOG_LEVEL


@dataclass(slots=True, frozen=True)
class MessageUpdate(Update):
    """Chat-Nachrichten zwischen User und Assistant."""
    role: Role = Role.ASSISTANT  # Default auf ASSISTANT setzen
    content: str = ""  # Leerer String als Default
    agent_id: str | None = None

    def __post_init__(self) -> None:
        """Validierung der Message-Update-Daten."""
        Update.__post_init__(self)
        # Validierung: content darf nicht komplett leer sein für bestimmte Rollen
        if self.role in [Role.USER, Role.ASSISTANT] and not self.content.strip():
            logger.warning(LOG_MSG_EMPTY_CONTENT_WARNING.format(
                role=self.role.value,
                update_id=self.update_id
            ))


@dataclass(slots=True, frozen=True)
class FunctionUpdate(Update):
    """Funktionsaufruf-Informationen."""
    call_id: str = ""  # Leerer String als Default
    name: str = ""  # Leerer String als Default
    arguments: dict[str, Any] = field(default_factory=ensure_dict_default)
    agent_id: str | None = None

    def __post_init__(self) -> None:
        """Validierung der Function-Update-Daten."""
        Update.__post_init__(self)
        if not self.call_id:
            object.__setattr__(self, "call_id", generate_uuid())
        if not self.name:
            logger.warning(LOG_MSG_EMPTY_FUNCTION_NAME_WARNING.format(call_id=self.call_id))


@dataclass(slots=True, frozen=True)
class FunctionCompletionUpdate(Update):
    """Funktions-Ergebnisse."""
    call_id: str = ""  # Leerer String als Default
    name: str = ""  # Leerer String als Default
    result: Any = None
    success: bool = True
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validierung der Function-Completion-Update-Daten."""
        Update.__post_init__(self)
        if not self.call_id:
            object.__setattr__(self, "call_id", generate_uuid())


@dataclass(slots=True, frozen=True)
class AudioUpdate(Update):
    """Audio-bezogene Updates."""
    audio_data: bytes | None = None
    audio_format: str = DEFAULT_AUDIO_FORMAT
    duration_ms: int | None = None


@dataclass(slots=True, frozen=True)
class ErrorUpdate(Update):
    """Fehler-Updates."""
    error_code: str = DEFAULT_ERROR_CODE
    error_message: str = DEFAULT_ERROR_MESSAGE
    error_details: dict[str, Any] | None = None
    severity: str = DEFAULT_SEVERITY


@dataclass(slots=True, frozen=True)
class SettingsUpdate(Update):
    """Einstellungs-Updates."""
    settings: dict[str, Any] = field(default_factory=ensure_dict_default)
    category: str = DEFAULT_CATEGORY


@dataclass(slots=True, frozen=True)
class AgentUpdate(Update):
    """Agent-Status-Updates."""
    agent_id: str = ""  # Leerer String als Default
    status: str = DEFAULT_STATUS
    metadata: dict[str, Any] = field(default_factory=ensure_dict_default)

    def __post_init__(self) -> None:
        """Validierung der Agent-Update-Daten."""
        super().__post_init__()
        if not self.agent_id:
            object.__setattr__(self, "agent_id", generate_uuid())


# Update Factory

class UpdateFactory:
    """Factory für Update-Erstellung mit konsolidierten Methoden."""

    @staticmethod
    def _create_base_update(_update_type: UpdateType) -> str:
        """Erstelle Basis-Update-ID für alle Update-Typen."""
        return generate_uuid()

    @staticmethod
    def create_console_update(
            payload: dict[str, Any],
            level: str = DEFAULT_LOG_LEVEL
    ) -> ConsoleUpdate:
        """Erstelle Console-Update."""
        return ConsoleUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.CONSOLE),
            type=UpdateType.CONSOLE,
            payload=payload,
            level=level
        )

    @staticmethod
    def create_message_update(
            role: Role,
            content: str,
            agent_id: str | None = None
    ) -> MessageUpdate:
        """Erstelle Message-Update."""
        return MessageUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.MESSAGE),
            type=UpdateType.MESSAGE,
            role=role,
            content=content,
            agent_id=agent_id
        )

    @staticmethod
    def create_function_update(
            call_id: str,
            name: str,
            arguments: dict[str, Any] | None = None,
            agent_id: str | None = None
    ) -> FunctionUpdate:
        """Erstelle Function-Update."""
        if arguments is None:
            arguments = ensure_dict_default()
        return FunctionUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.FUNCTION),
            type=UpdateType.FUNCTION,
            call_id=call_id,
            name=name,
            arguments=arguments,
            agent_id=agent_id
        )

    @staticmethod
    def create_function_completion_update(
            call_id: str,
            name: str,
            result: Any = None,
            success: bool = True,
            error_message: str | None = None
    ) -> FunctionCompletionUpdate:
        """Erstelle Function-Completion-Update."""
        return FunctionCompletionUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.FUNCTION_COMPLETION),
            type=UpdateType.FUNCTION_COMPLETION,
            call_id=call_id,
            name=name,
            result=result,
            success=success,
            error_message=error_message
        )

    @staticmethod
    def create_error_update(
            error_code: str,
            error_message: str,
            error_details: dict[str, Any] | None = None,
            severity: str = DEFAULT_SEVERITY
    ) -> ErrorUpdate:
        """Erstelle Error-Update."""
        return ErrorUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.ERROR),
            type=UpdateType.ERROR,
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            severity=severity
        )

    @staticmethod
    def create_agent_update(
            agent_id: str,
            status: str,
            metadata: dict[str, Any] | None = None
    ) -> AgentUpdate:
        """Erstelle Agent-Update."""
        if metadata is None:
            metadata = ensure_dict_default()
        return AgentUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.AGENT),
            type=UpdateType.AGENT,
            agent_id=agent_id,
            status=status,
            metadata=metadata
        )

    @staticmethod
    def create_audio_update(
            audio_data: bytes | None = None,
            audio_format: str = DEFAULT_AUDIO_FORMAT,
            duration_ms: int | None = None
    ) -> AudioUpdate:
        """Erstelle Audio-Update."""
        return AudioUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.AUDIO),
            type=UpdateType.AUDIO,
            audio_data=audio_data,
            audio_format=audio_format,
            duration_ms=duration_ms
        )

    @staticmethod
    def create_settings_update(
            settings: dict[str, Any],
            category: str = DEFAULT_CATEGORY
    ) -> SettingsUpdate:
        """Erstelle Settings-Update."""
        return SettingsUpdate(
            update_id=UpdateFactory._create_base_update(UpdateType.SETTINGS),
            type=UpdateType.SETTINGS,
            settings=settings,
            category=category
        )


# Event-System für Multi-Agent-Kommunikation

@dataclass(slots=True)
class AgentUpdateEvent:
    """Event für Agent-Updates zwischen Agenten."""
    event_id: str
    source_agent_id: str
    target_agent_id: str | None  # None = Broadcast
    event_type: str
    payload: dict[str, Any] = field(default_factory=ensure_dict_default)
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Generiere Event-ID falls nicht vorhanden."""
        if not self.event_id:
            self.event_id = generate_uuid()


# Protokolle für Typisierung

class UpdateProtocol(Protocol):
    """Protokoll für Update-Objekte."""
    update_id: str
    type: UpdateType
    timestamp: datetime


class AgentProtocol(Protocol):
    """Protokoll für Agent-Objekte."""
    id: str
    name: str
    type: str
    status: str


# Logging des erfolgreichen Ladens
from ..constants import LOG_MSG_CORE_MODELS_LOADED, LOG_MSG_OPENAI_SDK_AVAILABLE

logger.info(
    f"{LOG_MSG_CORE_MODELS_LOADED} - {len(UpdateType)} Update-Typen, {len(Role)} Rollen verfügbar"
)
logger.info(LOG_MSG_OPENAI_SDK_AVAILABLE)
