"""Konstanten und Konfiguration für den Camera Service.

Zentrale Definition aller Magic Numbers, Error Messages und Konfigurationswerte
für eine bessere Wartbarkeit und Konsistenz.
"""

from __future__ import annotations


# Kamera-Konfiguration
class CameraConfig:
    """Konfigurationskonstanten für Kamera-Operationen."""

    # Timeout-Einstellungen
    DEFAULT_INIT_TIMEOUT_SECONDS: float = 10.0

    # Warm-up Einstellungen
    WARMUP_FRAME_COUNT: int = 8
    WARMUP_FRAME_DELAY_SECONDS: float = 0.03

    # Bildqualität
    JPEG_QUALITY: int = 85
    PNG_COMPRESSION_LEVEL: int = 6

    # Dateigrößen-Limits
    MAX_IMAGE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5MB

    # Helligkeits-Normalisierung
    BRIGHTNESS_TARGET: float = 95.0
    BRIGHTNESS_THRESHOLD: float = 60.0
    MIN_GAMMA: float = 1.1
    MAX_GAMMA: float = 3.0

    # OpenCV Eigenschaften (für bessere Lesbarkeit)
    CV_PROP_FRAME_WIDTH: int = 3
    CV_PROP_FRAME_HEIGHT: int = 4
    CV_AUTO_EXPOSURE_VALUE: float = 0.25  # V4L2: 0.25 = Auto


# Unterstützte Auflösungen
SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "640x480": (640, 480),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
}


# Error Messages
class ErrorMessages:
    """Standardisierte Fehlermeldungen für den Camera Service."""

    # OpenCV Verfügbarkeit
    OPENCV_NOT_AVAILABLE = "OpenCV ist nicht verfügbar. Bitte 'opencv-python' installieren."

    # Kamera-Initialisierung
    CAMERA_NOT_OPENED = "Kamera {index} konnte nicht geöffnet werden"
    NO_CAMERA_AVAILABLE = "Keine Kamera verfügbar"
    CAMERA_ACCESS_DENIED = "Zugriff auf Kamera verweigert"

    # Frame-Capture
    CAMERA_NOT_INITIALIZED = "Kamera ist nicht initialisiert"
    FRAME_READ_FAILED = "Konnte kein Frame von der Kamera lesen"

    # Bildkodierung
    JPEG_ENCODING_FAILED = "JPEG Encoding fehlgeschlagen"
    PNG_ENCODING_FAILED = "PNG Encoding fehlgeschlagen"
    IMAGE_SIZE_EXCEEDED = "Bildgröße überschreitet {limit}MB Limit"

    # Logging Messages
    CAMERA_OPENED = "Kamera geöffnet"


# MIME Types
class ImageFormats:
    """MIME Types für unterstützte Bildformate."""

    JPEG = "image/jpeg"
    PNG = "image/png"


# Kamera-Indizes für Fallback-Suche
DEFAULT_CAMERA_INDICES = [0, 1, 2]
