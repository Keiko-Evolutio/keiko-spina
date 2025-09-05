"""Gemeinsame ID-Generierungs-Utilities für API-Module.

Dieses Modul stellt wiederverwendbare Funktionen für die Generierung
eindeutiger Identifikatoren bereit.
"""

from __future__ import annotations

import secrets
import string
import time
from uuid import uuid4

from kei_logging import get_logger

logger = get_logger(__name__)


# Konstanten für ID-Generierung
class IDConstants:
    """Zentrale Konstanten für ID-Generierung."""

    # Standard-Längen für verschiedene ID-Typen
    SHORT_ID_LENGTH = 8
    MEDIUM_ID_LENGTH = 16
    LONG_ID_LENGTH = 32

    # Präfixe für verschiedene Ressourcentypen
    CONFIG_PREFIX = "cfg"
    AGENT_PREFIX = "agt"
    SESSION_PREFIX = "ses"
    WEBHOOK_PREFIX = "whk"
    FUNCTION_PREFIX = "fnc"
    TEMPLATE_PREFIX = "tpl"

    # Zeichen-Sets für ID-Generierung
    ALPHANUMERIC = string.ascii_lowercase + string.digits
    ALPHANUMERIC_UPPER = string.ascii_letters + string.digits
    HEX_CHARS = string.hexdigits.lower()


def generate_uuid() -> str:
    """Generiert eine Standard-UUID4.

    Returns:
        UUID4 als String ohne Bindestriche
    """
    return uuid4().hex


def generate_short_uuid(length: int = IDConstants.SHORT_ID_LENGTH) -> str:
    """Generiert eine verkürzte UUID.

    Args:
        length: Gewünschte Länge der ID (Standard: 8)

    Returns:
        Verkürzte UUID als String

    Raises:
        ValueError: Falls length > 32 oder < 4
    """
    if length > 32:
        raise ValueError("ID-Länge kann nicht größer als 32 sein")
    if length < 4:
        raise ValueError("ID-Länge muss mindestens 4 sein")

    return uuid4().hex[:length]


def generate_secure_id(
    length: int = IDConstants.MEDIUM_ID_LENGTH,
    charset: str = IDConstants.ALPHANUMERIC
) -> str:
    """Generiert eine kryptographisch sichere ID.

    Args:
        length: Gewünschte Länge der ID
        charset: Zeichen-Set für die ID-Generierung

    Returns:
        Sichere ID als String

    Raises:
        ValueError: Falls length < 4
    """
    if length < 4:
        raise ValueError("ID-Länge muss mindestens 4 sein")

    return "".join(secrets.choice(charset) for _ in range(length))


def generate_prefixed_id(
    prefix: str,
    length: int = IDConstants.SHORT_ID_LENGTH,
    separator: str = "_"
) -> str:
    """Generiert eine ID mit Präfix.

    Args:
        prefix: Präfix für die ID
        length: Länge des ID-Teils (ohne Präfix)
        separator: Trennzeichen zwischen Präfix und ID

    Returns:
        Präfixierte ID als String

    Example:
        >>> generate_prefixed_id("cfg", 8)
        "cfg_a1b2c3d4"
    """
    id_part = generate_short_uuid(length)
    return f"{prefix}{separator}{id_part}"


def generate_timestamped_id(
    prefix: str | None = None,
    include_microseconds: bool = False
) -> str:
    """Generiert eine zeitstempel-basierte ID.

    Args:
        prefix: Optionaler Präfix
        include_microseconds: Ob Mikrosekunden eingeschlossen werden sollen

    Returns:
        Zeitstempel-basierte ID

    Example:
        >>> generate_timestamped_id("ses")
        "ses_1703123456_a1b2c3d4"
    """
    timestamp = int(time.time())
    if include_microseconds:
        timestamp = int(time.time() * 1_000_000)

    random_part = generate_short_uuid(8)

    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    return f"{timestamp}_{random_part}"


def generate_human_readable_id(
    prefix: str | None = None,
    word_count: int = 3,
    separator: str = "-"
) -> str:
    """Generiert eine menschenlesbare ID mit zufälligen Wörtern.

    Args:
        prefix: Optionaler Präfix
        word_count: Anzahl der Wörter
        separator: Trennzeichen zwischen Wörtern

    Returns:
        Menschenlesbare ID

    Note:
        Verwendet eine begrenzte Wortliste für Einfachheit.
        Für Produktionsumgebungen sollte eine größere Wortliste verwendet werden.
    """
    # Einfache Wortliste für Demo-Zwecke
    adjectives = [
        "quick", "bright", "calm", "bold", "wise", "kind", "swift", "clear",
        "strong", "gentle", "smart", "brave", "cool", "warm", "fresh", "clean"
    ]

    nouns = [
        "fox", "eagle", "wolf", "bear", "lion", "tiger", "hawk", "owl",
        "star", "moon", "sun", "wave", "fire", "wind", "rock", "tree"
    ]

    words = []
    for i in range(word_count):
        if i % 2 == 0 and i < word_count - 1:
            words.append(secrets.choice(adjectives))
        else:
            words.append(secrets.choice(nouns))

    # Füge eine kurze Zufallszahl hinzu für Eindeutigkeit
    random_num = secrets.randbelow(1000)
    words.append(str(random_num))

    id_part = separator.join(words)

    if prefix:
        return f"{prefix}{separator}{id_part}"
    return id_part


class IDGenerator:
    """Konfigurierbare ID-Generator-Klasse.

    Ermöglicht die Erstellung von ID-Generatoren mit spezifischen
    Konfigurationen für verschiedene Anwendungsfälle.
    """

    def __init__(
        self,
        prefix: str | None = None,
        length: int = IDConstants.SHORT_ID_LENGTH,
        charset: str = IDConstants.ALPHANUMERIC,
        separator: str = "_",
        use_timestamp: bool = False,
        use_secure: bool = False
    ) -> None:
        """Initialisiert ID-Generator.

        Args:
            prefix: Präfix für alle generierten IDs
            length: Länge des ID-Teils
            charset: Zeichen-Set für die Generierung
            separator: Trennzeichen zwischen Präfix und ID
            use_timestamp: Ob Zeitstempel verwendet werden soll
            use_secure: Ob kryptographisch sichere Generierung verwendet werden soll
        """
        self.prefix = prefix
        self.length = length
        self.charset = charset
        self.separator = separator
        self.use_timestamp = use_timestamp
        self.use_secure = use_secure

    def generate(self) -> str:
        """Generiert eine ID basierend auf der Konfiguration.

        Returns:
            Generierte ID als String
        """
        if self.use_timestamp:
            return generate_timestamped_id(self.prefix)

        if self.use_secure:
            id_part = generate_secure_id(self.length, self.charset)
        else:
            id_part = generate_short_uuid(self.length)

        if self.prefix:
            return f"{self.prefix}{self.separator}{id_part}"
        return id_part

    def generate_batch(self, count: int) -> list[str]:
        """Generiert mehrere IDs auf einmal.

        Args:
            count: Anzahl der zu generierenden IDs

        Returns:
            Liste von generierten IDs
        """
        return [self.generate() for _ in range(count)]


# Vorkonfigurierte Generatoren für häufige Anwendungsfälle
configuration_id_generator = IDGenerator(
    prefix=IDConstants.CONFIG_PREFIX,
    length=IDConstants.SHORT_ID_LENGTH
)

agent_id_generator = IDGenerator(
    prefix=IDConstants.AGENT_PREFIX,
    length=IDConstants.MEDIUM_ID_LENGTH,
    use_secure=True
)

session_id_generator = IDGenerator(
    prefix=IDConstants.SESSION_PREFIX,
    length=IDConstants.LONG_ID_LENGTH,
    use_timestamp=True
)

webhook_id_generator = IDGenerator(
    prefix=IDConstants.WEBHOOK_PREFIX,
    length=IDConstants.SHORT_ID_LENGTH
)


# Convenience-Funktionen für häufige ID-Typen
def generate_config_id() -> str:
    """Generiert eine Konfigurations-ID."""
    return configuration_id_generator.generate()


def generate_agent_id() -> str:
    """Generiert eine Agent-ID."""
    return agent_id_generator.generate()


def generate_session_id() -> str:
    """Generiert eine Session-ID."""
    return session_id_generator.generate()


def generate_webhook_id() -> str:
    """Generiert eine Webhook-ID."""
    return webhook_id_generator.generate()


def validate_id_format(
    id_value: str,
    expected_prefix: str | None = None,
    min_length: int = 4,
    max_length: int = 64
) -> bool:
    """Validiert das Format einer ID.

    Args:
        id_value: Zu validierende ID
        expected_prefix: Erwarteter Präfix (optional)
        min_length: Minimale Länge
        max_length: Maximale Länge

    Returns:
        True falls ID gültig ist
    """
    if not id_value or not isinstance(id_value, str):
        return False

    if len(id_value) < min_length or len(id_value) > max_length:
        return False

    if expected_prefix and not id_value.startswith(f"{expected_prefix}_"):
        return False

    # Prüfe auf gültige Zeichen (alphanumerisch + Unterstriche/Bindestriche)
    allowed_chars = set(string.ascii_letters + string.digits + "_-")
    return all(c in allowed_chars for c in id_value)
