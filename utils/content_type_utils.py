# backend/utils/content_type_utils.py
"""Zentralisierte Content-Type-Erkennung und -Validierung.

Konsolidiert die Content-Type-Logik aus verschiedenen Storage-Modulen
in eine einheitliche, testbare Implementierung.
"""

from __future__ import annotations

from kei_logging import get_logger

logger = get_logger(__name__)

# Content-Type-Konstanten
CONTENT_TYPE_PNG = "image/png"
CONTENT_TYPE_JPEG = "image/jpeg"
CONTENT_TYPE_WEBP = "image/webp"

# Magic Number Signatures für Bilddateien
MAGIC_NUMBERS = {
    b"\x89PNG": CONTENT_TYPE_PNG,
    b"\xFF\xD8\xFF": CONTENT_TYPE_JPEG,
    b"RIFF": CONTENT_TYPE_WEBP,  # Spezielle Behandlung für WEBP
}

# Dateiendung zu Content-Type Mapping
EXTENSION_MAP = {
    "png": CONTENT_TYPE_PNG,
    "jpg": CONTENT_TYPE_JPEG,
    "jpeg": CONTENT_TYPE_JPEG,
    "webp": CONTENT_TYPE_WEBP,
}

# Unterstützte Content-Types
SUPPORTED_CONTENT_TYPES = {
    CONTENT_TYPE_PNG,
    CONTENT_TYPE_JPEG,
    CONTENT_TYPE_WEBP,
}


def detect_content_type_from_data(image_data: bytes) -> str:
    """Erkennt Content-Type basierend auf Bilddaten (Magic Numbers).

    Args:
        image_data: Bilddaten als Bytes

    Returns:
        MIME-Type basierend auf Magic Numbers
    """
    if not image_data:
        return CONTENT_TYPE_PNG  # Default

    # PNG-Erkennung
    if image_data.startswith(b"\x89PNG"):
        return CONTENT_TYPE_PNG

    # JPEG-Erkennung
    if image_data.startswith(b"\xFF\xD8\xFF"):
        return CONTENT_TYPE_JPEG

    # WEBP-Erkennung (komplexer)
    if image_data.startswith(b"RIFF") and len(image_data) >= 12 and b"WEBP" in image_data[:12]:
        return CONTENT_TYPE_WEBP

    # Default zu PNG
    logger.debug("Unbekanntes Bildformat, verwende PNG als Default")
    return CONTENT_TYPE_PNG


def detect_content_type_from_filename(filename: str) -> str:
    """Erkennt Content-Type basierend auf Dateiendung.

    Args:
        filename: Dateiname mit Endung

    Returns:
        MIME-Type basierend auf Dateiendung
    """
    if not filename or "." not in filename:
        return CONTENT_TYPE_PNG  # Default

    extension = filename.lower().split(".")[-1]
    return EXTENSION_MAP.get(extension, CONTENT_TYPE_PNG)


def validate_content_type(content_type: str) -> bool:
    """Validiert ob Content-Type unterstützt wird.

    Args:
        content_type: Zu validierender Content-Type

    Returns:
        True wenn unterstützt, False sonst
    """
    return content_type in SUPPORTED_CONTENT_TYPES


def get_file_extension_for_content_type(content_type: str) -> str:
    """Gibt die Standard-Dateiendung für einen Content-Type zurück.

    Args:
        content_type: MIME-Type

    Returns:
        Dateiendung ohne Punkt
    """
    content_type_to_extension = {
        CONTENT_TYPE_PNG: "png",
        CONTENT_TYPE_JPEG: "jpg",
        CONTENT_TYPE_WEBP: "webp",
    }
    return content_type_to_extension.get(content_type, "png")


def normalize_content_type(
    content_type: str | None,
    filename: str | None = None,
    image_data: bytes | None = None
) -> str:
    """Normalisiert und validiert Content-Type.

    Args:
        content_type: Optionaler Content-Type
        filename: Optionaler Dateiname für Fallback
        image_data: Optionale Bilddaten für Magic-Number-Erkennung

    Returns:
        Validierter und normalisierter Content-Type
    """
    # Wenn Content-Type gegeben und gültig, verwende ihn
    if content_type and validate_content_type(content_type):
        return content_type

    # Fallback auf Magic-Number-Erkennung (höchste Priorität)
    if image_data:
        detected = detect_content_type_from_data(image_data)
        if validate_content_type(detected):
            return detected

    # Fallback auf Dateiname-basierte Erkennung
    if filename:
        detected = detect_content_type_from_filename(filename)
        if validate_content_type(detected):
            return detected

    # Default zu PNG
    return CONTENT_TYPE_PNG


class ContentTypeDetector:
    """Erweiterte Content-Type-Erkennung mit Caching und Statistiken."""

    def __init__(self) -> None:
        """Initialisiert Content-Type-Detector."""
        self._detection_cache: dict[bytes, str] = {}
        self._detection_stats = {
            "total_detections": 0,
            "cache_hits": 0,
            "magic_number_detections": 0,
            "filename_detections": 0,
            "default_fallbacks": 0,
        }

    def detect(
        self,
        image_data: bytes | None = None,
        filename: str | None = None,
        content_type: str | None = None
    ) -> str:
        """Umfassende Content-Type-Erkennung.

        Args:
            image_data: Optionale Bilddaten
            filename: Optionaler Dateiname
            content_type: Optionaler vorgegebener Content-Type

        Returns:
            Erkannter oder validierter Content-Type
        """
        self._detection_stats["total_detections"] += 1

        # 1. Validiere vorgegebenen Content-Type
        if content_type and validate_content_type(content_type):
            return content_type

        # 2. Magic Number Detection (höchste Priorität)
        if image_data:
            # Cache-Lookup für Performance
            cache_key = image_data[:12] if len(image_data) >= 12 else image_data
            if cache_key in self._detection_cache:
                self._detection_stats["cache_hits"] += 1
                return self._detection_cache[cache_key]

            detected = detect_content_type_from_data(image_data)
            self._detection_cache[cache_key] = detected
            self._detection_stats["magic_number_detections"] += 1
            return detected

        # 3. Filename-basierte Erkennung
        if filename:
            detected = detect_content_type_from_filename(filename)
            self._detection_stats["filename_detections"] += 1
            return detected

        # 4. Default Fallback
        self._detection_stats["default_fallbacks"] += 1
        return CONTENT_TYPE_PNG

    def get_stats(self) -> dict[str, int]:
        """Gibt Erkennungsstatistiken zurück."""
        return self._detection_stats.copy()

    def clear_cache(self) -> None:
        """Leert den Detection-Cache."""
        self._detection_cache.clear()


# Globale Instanz für einfache Verwendung
content_type_detector = ContentTypeDetector()


__all__ = [
    "CONTENT_TYPE_JPEG",
    "CONTENT_TYPE_PNG",
    "CONTENT_TYPE_WEBP",
    "SUPPORTED_CONTENT_TYPES",
    "ContentTypeDetector",
    "content_type_detector",
    "detect_content_type_from_data",
    "detect_content_type_from_filename",
    "get_file_extension_for_content_type",
    "normalize_content_type",
    "validate_content_type",
]
