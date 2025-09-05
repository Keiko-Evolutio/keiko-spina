"""Intent-Erkennung für Bildgenerierung und Foto-Aufnahme.

Erkennt aus Freitext, ob eine Bildgenerierung oder Foto-Aufnahme
gewünscht ist und extrahiert Parameter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SIZES = {"1024x1024", "1024x1792", "1792x1024"}
_QUALITIES = {"standard", "hd"}
_STYLES = {"Realistic", "Artistic", "Cartoon", "Photography", "Digital Art"}

_IMAGE_GENERATION_KEYWORDS = (
    r"\bgenerate\b|\bcreate\b|\bdraw\b|\bpaint\b|\bzeichne\b|\berstelle\b|\berzeuge\b|\bgeneriere\b|"
    r"\bmale\b|\bdesign\b|\bkünstlerisch\b|\bartistic\b"
)

_PHOTO_CAPTURE_KEYWORDS = (
    r"\bmach.*foto\b|\b(nehm|nimm).*foto\b|\btake.*photo\b|\bcapture\b|\baufnehmen\b|\bfotografier\w*\b|"
    r"\bfoto.*von.*mir\b|\bphoto.*of.*me\b|\bselfie\b|\bkamera\b|\bcamera\b|"
    r"\bschieß.*foto\b|\bknips\b"
)

_GENERAL_IMAGE_KEYWORDS = r"\bimage\b|\bpicture\b|\bphoto\b|\bbild\b|\bfoto\b|\bfotografie\b"

_STYLE_MAP = {
    # Deutsch → Kategorie
    "realistisch": "Realistic",
    "fotorealistisch": "Realistic",
    "künstlerisch": "Artistic",
    "gemälde": "Artistic",
    "cartoon": "Cartoon",
    "comic": "Cartoon",
    "fotografie": "Photography",
    "photo": "Photography",
    "digital": "Digital Art",
    "digital art": "Digital Art",
}


@dataclass(slots=True)
class ImageIntent:
    """Ergebnis der Intent-Erkennung für Bildgenerierung."""

    is_image: bool
    prompt: str
    size: str | None = None
    quality: str | None = None
    style: str | None = None

    def missing(self) -> list[str]:
        """Gibt Liste fehlender Parameter zurück."""
        missing: list[str] = []
        if self.size not in _SIZES:
            missing.append("Größe")
        if self.quality not in _QUALITIES:
            missing.append("Qualität")
        if self.style not in _STYLES:
            missing.append("Stil")
        return missing


@dataclass(slots=True)
class PhotoIntent:
    """Ergebnis der Intent-Erkennung für Foto-Aufnahme."""

    is_photo: bool
    user_request: str
    resolution: str | None = None  # "640x480", "1280x720", "1920x1080"

    def missing(self) -> list[str]:
        """Gibt Liste fehlender Parameter zurück."""
        missing: list[str] = []
        supported_resolutions = {"640x480", "1280x720", "1920x1080"}
        if self.resolution and self.resolution not in supported_resolutions:
            missing.append("Auflösung")
        return missing


def detect_image_intent(user_input: str) -> ImageIntent:
    """Erkennt, ob der Nutzer ein AI-Bild generieren möchte und extrahiert Parameter.

    Args:
        user_input: Freitext des Nutzers

    Returns:
        ImageIntent: Strukturierte Erkennungsergebnisse
    """
    text = (user_input or "").strip()
    if not text:
        return ImageIntent(is_image=False, prompt="")

    # Prüfe zuerst auf explizite Bildgenerierung
    is_generation = bool(re.search(_IMAGE_GENERATION_KEYWORDS, text, flags=re.IGNORECASE))

    # Foto-Aufnahme separat erkennen
    is_photo_capture = bool(re.search(_PHOTO_CAPTURE_KEYWORDS, text, flags=re.IGNORECASE))

    # Kein generischer Fallback auf allgemeine Bild-Keywords, um False-Positives zu vermeiden

    # Wenn Foto-Aufnahme erkannt wird, ist es KEINE Bildgenerierung
    is_image = is_generation and not is_photo_capture

    # Größe extrahieren
    size_match = re.search(r"\b(1024x1024|1024x1792|1792x1024)\b", text)
    size = size_match.group(1) if size_match else None

    # Qualität extrahieren
    qual_match = re.search(r"\b(hd|standard)\b", text, flags=re.IGNORECASE)
    quality = qual_match.group(1).lower() if qual_match else None

    # Stil aus bekannten Schlüsselwörtern ableiten
    detected_style: str | None = None
    for key, mapped in _STYLE_MAP.items():
        if key in text.lower():
            detected_style = mapped
            break

    # Prompt ist der Originaltext (ohne explizite Parameterentfernung, um Kontext zu behalten)
    return ImageIntent(
        is_image=is_image,
        prompt=text,
        size=size,
        quality=quality,
        style=detected_style,
    )


def detect_photo_intent(user_input: str) -> PhotoIntent:
    """Erkennt, ob der Nutzer ein Foto aufnehmen möchte und extrahiert Parameter.

    Args:
        user_input: Freitext des Nutzers

    Returns:
        PhotoIntent: Strukturierte Erkennungsergebnisse
    """
    text = (user_input or "").strip()
    if not text:
        return PhotoIntent(is_photo=False, user_request="")

    # Prüfe auf Foto-Aufnahme Keywords
    is_photo = bool(re.search(_PHOTO_CAPTURE_KEYWORDS, text, flags=re.IGNORECASE))

    # Auflösung extrahieren (falls angegeben)
    resolution_match = re.search(r"\b(640x480|1280x720|1920x1080)\b", text)
    resolution = resolution_match.group(1) if resolution_match else None

    return PhotoIntent(
        is_photo=is_photo,
        user_request=text,
        resolution=resolution,
    )


def detect_photo_ready_intent(user_input: str) -> bool:
    """Erkennt, ob der Nutzer bereit für die Foto-Aufnahme ist.

    Args:
        user_input: Freitext des Nutzers

    Returns:
        True wenn Bereitschaft erkannt wird, False sonst
    """
    text = (user_input or "").strip().lower()
    if not text:
        return False

    # Bereitschafts-Keywords (erweitert aus voice_routes.py)
    ready_patterns = [
        "ja", "yes", "bereit", "ready", "ich bin bereit", "jetzt bin ich bereit",
        "jetzt bereit", "mach das foto", "mach ein foto", "jetzt foto",
        "i am ready", "i'm ready", "take the photo", "take a photo",
        "ok", "okay", "los", "go", "jetzt", "now"
    ]

    return any(pattern in text for pattern in ready_patterns)


def build_followup_question(intent: ImageIntent) -> str | None:
    """Erzeugt eine Rückfrage, falls Parameter fehlen.

    Args:
        intent: Erkanntes Intent-Ergebnis

    Returns:
        Optionale Rückfrage in Deutsch
    """
    if not intent.is_image:
        return None

    missing = intent.missing()
    if not missing:
        return None

    # Benutzerfreundliche Rückfrage mit Beispielen
    parts: list[str] = []
    if "Stil" in missing:
        parts.append(
            "Welchen Stil bevorzugst du? (Realistic, Artistic, Cartoon, Photography, Digital Art)"
        )
    if "Größe" in missing:
        parts.append("Welche Größe? (1024x1024, 1024x1792, 1792x1024)")
    if "Qualität" in missing:
        parts.append("Welche Qualität? (standard, hd)")

    return " ".join(parts) if parts else None
