"""Content-Utilities für Voice Session Management.

Funktionen für Content-Extraction und Content-Validation
aus Response-Items und Events.
"""

from __future__ import annotations

import re
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

class ContentExtractionError(ValueError):
    """Fehler bei Content-Extraction-Operationen."""

    def __init__(self, message: str, item: Any = None) -> None:
        super().__init__(message)
        self.item = item

# ContentItem Protocol wurde entfernt (Dead Code)

# ResponseItem Protocol wurde entfernt (Dead Code)

def extract_content_from_item(item: Any) -> str | None:
    """Extrahiert Content aus Response-Item mit verbesserter Type-Safety.

    Verbesserte Version der ursprünglichen _extract_content_from_item() Funktion
    mit besserer Error-Handling und Type-Hints.

    Args:
        item: Response-Item mit content-Attribut

    Returns:
        Extrahierter Content-String oder None

    Raises:
        ContentExtractionError: Bei kritischen Extraction-Fehlern

    Examples:
        >>> # Example with object having content attribute
        >>> class ExampleItem: pass
        >>> class ExampleContent: pass
        >>> item = ExampleItem()
        >>> content = ExampleContent()
        >>> content.text = "Hello"
        >>> item.content = [content]
        >>> extract_content_from_item(item)
        "Hello"
        >>> extract_content_from_item(None)
        None
    """
    if item is None:
        return None

    try:
        # Prüfe ob item content-Attribut hat
        if not hasattr(item, "content"):
            logger.debug("Item hat kein content-Attribut: %s", type(item).__name__)
            return None

        content_list = item.content
        if not content_list:
            logger.debug("Content-Liste ist leer oder None")
            return None

        # Nimm erstes Content-Item
        if not isinstance(content_list, list | tuple) or len(content_list) == 0:
            logger.debug("Content ist keine Liste oder leer: %s", type(content_list).__name__)
            return None

        content_item = content_list[0]
        if content_item is None:
            logger.debug("Erstes Content-Item ist None")
            return None

        # Versuche text-Attribut zu extrahieren
        if hasattr(content_item, "text") and content_item.text:
            text_content = content_item.text
            if isinstance(text_content, str) and text_content.strip():
                return text_content.strip()

        # Versuche transcript-Attribut zu extrahieren
        if hasattr(content_item, "transcript") and content_item.transcript:
            transcript_content = content_item.transcript
            if isinstance(transcript_content, str) and transcript_content.strip():
                return transcript_content.strip()

        # Fallback: String-Representation
        if content_item:
            str_content = str(content_item)
            if str_content and str_content.strip() and str_content != "None":
                return str_content.strip()

        return None

    except Exception as e:
        logger.exception("Fehler bei Content-Extraction: %s", e)
        # In Production: Graceful degradation statt Exception
        return None

def should_send_content(content: str | None) -> bool:
    """Prüft ob Content gesendet werden sollte.

    Verbesserte Version der ursprünglichen _should_send_content() Funktion
    mit erweiterten Validierungsregeln.

    Args:
        content: Zu prüfender Content-String

    Returns:
        True wenn Content gesendet werden sollte, False sonst

    Examples:
        >>> should_send_content("Hello World")
        True
        >>> should_send_content("")
        False
        >>> should_send_content(None)
        False
        >>> should_send_content("   ")
        False
    """
    if content is None:
        return False

    if not isinstance(content, str):
        return False

    # Entferne Whitespace für Prüfung
    stripped_content = content.strip()

    if not stripped_content:
        return False

    # Prüfe auf reine Whitespace-Zeichen
    if not stripped_content or stripped_content.isspace():
        return False

    # Prüfe auf minimale Content-Länge (optional)
    return not len(stripped_content) < 1

def validate_content_quality(content: str) -> bool:
    """Validiert Content-Qualität für erweiterte Filterung.

    Args:
        content: Zu validierender Content

    Returns:
        True wenn Content qualitativ hochwertig ist, False sonst
    """
    if not should_send_content(content):
        return False

    stripped = content.strip()

    # Prüfe auf minimale Länge für sinnvollen Content
    if len(stripped) < 2:
        return False

    # Prüfe auf reine Sonderzeichen
    if re.match(r"^[^\w\s]+$", stripped):
        return False

    # Prüfe auf repetitive Zeichen (z.B. "aaaaaaa")
    return not (len(set(stripped.lower())) == 1 and len(stripped) > 3)

def sanitize_content(content: str) -> str:
    """Bereinigt Content von problematischen Zeichen.

    Args:
        content: Zu bereinigender Content

    Returns:
        Bereinigter Content-String
    """
    if not content:
        return ""

    # Entferne führende/nachfolgende Whitespace
    sanitized = content.strip()

    # Normalisiere Whitespace (mehrere Spaces zu einem)
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Entferne Steuerzeichen (außer Newlines und Tabs)
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)


def extract_and_validate_content(item: Any) -> str | None:
    """Kombiniert Content-Extraction und Validation in einer Funktion.

    Args:
        item: Response-Item

    Returns:
        Validierter Content oder None
    """
    try:
        # Extrahiere Content
        content = extract_content_from_item(item)

        if not should_send_content(content):
            return None

        # Bereinige Content
        sanitized = sanitize_content(content)

        # Finale Validation
        if not should_send_content(sanitized):
            return None

        return sanitized

    except Exception as e:
        logger.exception("Fehler bei Content-Extraction und Validation: %s", e)
        return None

def get_content_summary(content: str, max_length: int = 50) -> str:
    """Erstellt Zusammenfassung von Content für Logging.

    Args:
        content: Content-String
        max_length: Maximale Länge der Zusammenfassung

    Returns:
        Gekürzte Content-Zusammenfassung
    """
    if not content:
        return "<empty>"

    if len(content) <= max_length:
        return content

    return content[:max_length - 3] + "..."

def extract_content_metadata(item: Any) -> dict[str, Any]:
    """Extrahiert Metadaten aus Content-Item für Debugging.

    Args:
        item: Response-Item

    Returns:
        Dictionary mit Content-Metadaten
    """
    metadata = {
        "has_content": hasattr(item, "content"),
        "content_type": type(item).__name__,
        "content_length": 0,
        "has_text": False,
        "has_transcript": False,
        "content_items_count": 0,
    }

    try:
        if hasattr(item, "content") and item.content:
            metadata["content_items_count"] = len(item.content) if isinstance(item.content, list | tuple) else 1

            if isinstance(item.content, list | tuple) and len(item.content) > 0:
                first_item = item.content[0]
                metadata["has_text"] = hasattr(first_item, "text") and bool(first_item.text)
                metadata["has_transcript"] = hasattr(first_item, "transcript") and bool(first_item.transcript)

                content = extract_content_from_item(item)
                if content:
                    metadata["content_length"] = len(content)

    except Exception as e:
        metadata["extraction_error"] = str(e)

    return metadata

# =============================================================================
# Legacy-Kompatibilität
# =============================================================================

def legacy_extract_content_from_item(item: Any) -> str | None:
    """Legacy-Kompatibilität für ursprüngliche _extract_content_from_item().

    Repliziert exakt die ursprüngliche Logik für Backward-Compatibility.
    """
    if not hasattr(item, "content") or not item.content:
        return None

    content_item = item.content[0]
    if hasattr(content_item, "text") and content_item.text:
        return content_item.text
    if hasattr(content_item, "transcript") and content_item.transcript:
        return content_item.transcript

    return str(content_item) if content_item else None

def legacy_should_send_content(content: str | None) -> bool:
    """Legacy-Kompatibilität für ursprüngliche _should_send_content().

    Repliziert exakt die ursprüngliche Logik für Backward-Compatibility.
    """
    return bool(content and content.strip())
