"""Role-Utilities für Voice Session Management.

Zentrale Funktionen für Role-Mapping und Role-Validation
zur Eliminierung von dupliziertem Code.
"""

from __future__ import annotations

from data_models import Role

from .session_constants import (
    ASSISTANT_ROLE,
    DEFAULT_ROLE,
    SYSTEM_ROLE,
    USER_ROLE,
    VALID_ROLES,
)


class RoleMappingError(ValueError):
    """Fehler bei Role-Mapping-Operationen."""

    def __init__(self, message: str, role_value: str | None = None) -> None:
        super().__init__(message)
        self.role_value = role_value


def normalize_role_string(role: str | None) -> str:
    """Normalisiert Role-String zu lowercase und behandelt None-Werte.

    Args:
        role: Role-String oder None

    Returns:
        Normalisierter Role-String (lowercase)

    Examples:
        >>> normalize_role_string("ASSISTANT")
        "assistant"
        >>> normalize_role_string(None)
        "assistant"
        >>> normalize_role_string("")
        "assistant"
    """
    if not role or not role.strip():
        return DEFAULT_ROLE

    normalized = role.strip().lower()

    # Validierung gegen bekannte Roles
    if normalized not in VALID_ROLES:
        # Fallback auf Default-Role statt Exception für Robustheit
        return DEFAULT_ROLE

    return normalized


def map_role_to_enum(role: str | None) -> Role:
    """Mappt Role-String zu Role-Enum.

    Zentrale Funktion zur Eliminierung der duplizierten Role-Mapping-Logik
    aus _process_output_item() und handle_output_item_done().

    Args:
        role: Role-String (kann None, leer oder ungültig sein)

    Returns:
        Role-Enum (Role.ASSISTANT, Role.USER oder Role.SYSTEM)

    Raises:
        RoleMappingError: Bei kritischen Mapping-Fehlern (nur in strict mode)

    Examples:
        >>> map_role_to_enum("assistant")
        Role.ASSISTANT
        >>> map_role_to_enum("USER")
        Role.USER
        >>> map_role_to_enum(None)
        Role.ASSISTANT
        >>> map_role_to_enum("invalid")
        Role.ASSISTANT
    """
    normalized_role = normalize_role_string(role)

    # Mapping zu Enum-Werten
    role_mapping = {
        ASSISTANT_ROLE: Role.ASSISTANT,
        USER_ROLE: Role.USER,
        SYSTEM_ROLE: Role.SYSTEM,
    }

    return role_mapping.get(normalized_role, Role.ASSISTANT)


def map_role_to_enum_strict(role: str | None) -> Role:
    """Mappt Role-String zu Role-Enum mit strikter Validierung.

    Wie map_role_to_enum(), aber wirft Exception bei ungültigen Roles
    statt Fallback auf Default-Role.

    Args:
        role: Role-String

    Returns:
        Role-Enum

    Raises:
        RoleMappingError: Bei ungültigen oder unbekannten Roles
    """
    if not role or not role.strip():
        raise RoleMappingError("Role darf nicht None oder leer sein", role)

    normalized_role = role.strip().lower()

    if normalized_role not in VALID_ROLES:
        raise RoleMappingError(
            f"Ungültige Role: '{role}'. Gültige Roles: {VALID_ROLES}",
            role
        )

    role_mapping = {
        ASSISTANT_ROLE: Role.ASSISTANT,
        USER_ROLE: Role.USER,
        SYSTEM_ROLE: Role.SYSTEM,
    }

    return role_mapping[normalized_role]


def validate_role_string(role: str) -> bool:
    """Validiert ob Role-String gültig ist.

    Args:
        role: Zu validierender Role-String

    Returns:
        True wenn gültig, False sonst

    Examples:
        >>> validate_role_string("assistant")
        True
        >>> validate_role_string("INVALID")
        False
    """
    if not role or not isinstance(role, str):
        return False

    return role.strip().lower() in VALID_ROLES


def get_role_display_name(role: str | Role) -> str:
    """Liefert benutzerfreundlichen Display-Namen für Role.

    Args:
        role: Role-String oder Role-Enum

    Returns:
        Benutzerfreundlicher Display-Name

    Examples:
        >>> get_role_display_name("assistant")
        "Assistant"
        >>> get_role_display_name(Role.USER)
        "User"
    """
    if isinstance(role, Role):
        role_str = role.value.lower()
    else:
        # Für ungültige Roles wird normalize_role_string() DEFAULT_ROLE zurückgeben
        role_str = normalize_role_string(role)

    display_names = {
        ASSISTANT_ROLE: "Assistant",
        USER_ROLE: "User",
        SYSTEM_ROLE: "System",
    }

    return display_names.get(role_str, "Assistant")


def is_assistant_role(role: str | None) -> bool:
    """Prüft ob Role ein Assistant ist.

    Args:
        role: Zu prüfender Role-String

    Returns:
        True wenn Assistant-Role, False sonst
    """
    return normalize_role_string(role) == ASSISTANT_ROLE


def is_user_role(role: str | None) -> bool:
    """Prüft ob Role ein User ist.

    Args:
        role: Zu prüfender Role-String

    Returns:
        True wenn User-Role, False sonst
    """
    return normalize_role_string(role) == USER_ROLE


def is_system_role(role: str | None) -> bool:
    """Prüft ob Role ein System ist.

    Args:
        role: Zu prüfender Role-String

    Returns:
        True wenn System-Role, False sonst
    """
    return normalize_role_string(role) == SYSTEM_ROLE


# =============================================================================
# Legacy-Kompatibilität
# =============================================================================

def legacy_role_mapping(role_value: str | None) -> Role:
    """Legacy-Kompatibilität für die ursprüngliche Role-Mapping-Logik.

    Repliziert exakt die ursprüngliche Logik aus session.py:
    role_enum = Role.ASSISTANT if role_value not in {"user", "system"} else (
        Role.USER if role_value == "user" else Role.SYSTEM
    )

    Args:
        role_value: Role-String (bereits normalisiert)

    Returns:
        Role-Enum nach Legacy-Logik

    Note:
        Diese Funktion sollte nur für Backward-Compatibility verwendet werden.
        Neue Code sollte map_role_to_enum() verwenden.
    """
    if role_value not in {USER_ROLE, SYSTEM_ROLE}:
        return Role.ASSISTANT

    return Role.USER if role_value == USER_ROLE else Role.SYSTEM


def convert_legacy_role_mapping(role: str | None) -> Role:
    """Konvertiert Role mit Legacy-Logik für exakte Backward-Compatibility.

    Repliziert die ursprüngliche Logik:
    role_value = (role or "assistant").lower()
    role_enum = Role.ASSISTANT if role_value not in {"user", "system"} else (
        Role.USER if role_value == "user" else Role.SYSTEM
    )

    Args:
        role: Original Role-String (kann None sein)

    Returns:
        Role-Enum nach exakter Legacy-Logik
    """
    role_value = (role or ASSISTANT_ROLE).lower()
    return legacy_role_mapping(role_value)
