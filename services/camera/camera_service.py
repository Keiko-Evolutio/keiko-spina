"""Kamera-Service für plattformübergreifenden Foto-Capture mittels OpenCV.

Dieser Service kapselt die Kamera-Initialisierung, Auflösungseinstellung,
Frame-Capture und Bildkodierung (JPEG/PNG) inklusive robuster Fehlerbehandlung
und Ressourcenfreigabe über das Context-Manager-Pattern.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - optional import fallbacks in Tests
    import cv2
    _CV_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    _CV_AVAILABLE = False

from kei_logging import get_logger

from .camera_constants import (
    DEFAULT_CAMERA_INDICES,
    SUPPORTED_RESOLUTIONS,
    CameraConfig,
    ErrorMessages,
    ImageFormats,
)

logger = get_logger(__name__)


class CameraInitializationError(RuntimeError):
    """Fehler bei der Kamera-Initialisierung."""


class CameraAccessDeniedError(PermissionError):
    """Fehler wenn der Zugriff auf die Kamera verweigert wurde."""


class CameraService:
    """Kapselt Zugriff auf eine lokale Kamera mittels OpenCV.

    Der Service nutzt ein Context-Manager-Pattern, um sicherzustellen, dass
    Ressourcen (z. B. `cv2.VideoCapture`) korrekt freigegeben werden.

    Attributes:
        camera_index: Bevorzugter Kamera-Index.
        resolution: Gewünschte Auflösung (Breite, Höhe).
        init_timeout_s: Timeout für die Kamera-Initialisierung in Sekunden.
    """

    def __init__(
        self,
        camera_index: int = 0,
        resolution: tuple[int, int] = SUPPORTED_RESOLUTIONS["640x480"],
        *,
        init_timeout_s: float = CameraConfig.DEFAULT_INIT_TIMEOUT_SECONDS,
    ) -> None:
        # Parameter speichern
        self.camera_index: int = camera_index
        self.resolution: tuple[int, int] = resolution
        self.init_timeout_s: float = init_timeout_s

        # Interner Capture-Handle
        self._cap: Any | None = None

    def __enter__(self) -> CameraService:
        """Initialisiert die Kamera synchron innerhalb des Enter-Blocks.

        Raises:
            CameraInitializationError: Wenn die Kamera nicht geöffnet werden kann.
            CameraAccessDeniedError: Wenn der Zugriff verweigert wird.
            RuntimeError: Wenn OpenCV nicht verfügbar ist.
        """
        self._check_opencv_availability()
        self._initialize_camera()
        return self

    def _check_opencv_availability(self) -> None:
        """Prüft OpenCV-Verfügbarkeit und ermöglicht Test-Mocks.

        Raises:
            RuntimeError: Wenn OpenCV nicht verfügbar ist.
        """
        if not _CV_AVAILABLE or cv2 is None:
            try:
                import builtins as _bi  # type: ignore
                cv_mod = getattr(_bi, "cv2", None)
                if cv_mod is None:
                    raise RuntimeError(ErrorMessages.OPENCV_NOT_AVAILABLE)
                # Nutzung des bereitgestellten Mocks
                globals()["cv2"] = cv_mod
            except Exception as _e:
                raise RuntimeError(ErrorMessages.OPENCV_NOT_AVAILABLE)

    def _initialize_camera(self) -> None:
        """Initialisiert die Kamera mit Fallback auf andere Indizes.

        Raises:
            CameraInitializationError: Wenn keine Kamera geöffnet werden kann.
            CameraAccessDeniedError: Wenn der Zugriff verweigert wird.
        """
        start_time: float = time.time()
        candidate_indices: list[int] = [self.camera_index] + [
            i for i in DEFAULT_CAMERA_INDICES if i != self.camera_index
        ]
        last_error: str | None = None

        for camera_index in candidate_indices:
            if time.time() - start_time > self.init_timeout_s:
                break

            try:
                cap: Any | None = self._try_open_camera(camera_index)
                if cap is not None:
                    self._configure_camera_settings(cap)
                    self._perform_camera_warmup(cap)
                    self._cap = cap
                    logger.info(
                        ErrorMessages.CAMERA_OPENED,
                        extra={"index": camera_index, "resolution": self.resolution}
                    )
                    return
            except Exception as e:  # pragma: no cover - hardware-spezifisch
                last_error = str(e)

        # Fehlerklassifikation und Exception werfen
        self._handle_initialization_error(last_error)

    def _try_open_camera(self, camera_index: int) -> Any | None:
        """Versucht eine Kamera zu öffnen.

        Args:
            camera_index: Index der zu öffnenden Kamera.

        Returns:
            VideoCapture-Objekt oder None wenn fehlgeschlagen.
        """
        cap: Any = cv2.VideoCapture(camera_index)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None
        return cap

    def _configure_camera_settings(self, cap: Any) -> None:
        """Konfiguriert Kamera-Einstellungen wie Auflösung und Belichtung.

        Args:
            cap: VideoCapture-Objekt der Kamera.
        """
        # Auflösung setzen (best effort)
        width: int
        height: int
        width, height = self.resolution
        cap.set(CameraConfig.CV_PROP_FRAME_WIDTH, float(width))
        cap.set(CameraConfig.CV_PROP_FRAME_HEIGHT, float(height))

        # Automatische Belichtung aktivieren (best effort, plattformabhängig)
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CameraConfig.CV_AUTO_EXPOSURE_VALUE)
        except Exception:
            pass  # Ignoriere Fehler bei plattformspezifischen Einstellungen

    def _perform_camera_warmup(self, cap: Any) -> None:
        """Führt Kamera-Warm-up für Auto-Exposure/Weißabgleich durch.

        Args:
            cap: VideoCapture-Objekt der Kamera.
        """
        try:
            for _ in range(CameraConfig.WARMUP_FRAME_COUNT):
                _ = cap.read()
                time.sleep(CameraConfig.WARMUP_FRAME_DELAY_SECONDS)
        except Exception:
            pass  # Ignoriere Warm-up Fehler

    def _handle_initialization_error(self, last_error: str | None) -> None:
        """Behandelt Initialisierungsfehler und wirft entsprechende Exceptions.

        Args:
            last_error: Letzter aufgetretener Fehler.

        Raises:
            CameraAccessDeniedError: Bei Zugriffsverweigerung.
            CameraInitializationError: Bei anderen Initialisierungsfehlern.
        """
        if last_error and "denied" in last_error.lower():
            raise CameraAccessDeniedError(last_error)
        raise CameraInitializationError(last_error or ErrorMessages.NO_CAMERA_AVAILABLE)

    def __exit__(
        self,
        exc_type: type | None,
        exc: BaseException | None,
        _tb: Any | None
    ) -> None:
        """Gibt die Kameraressourcen zuverlässig frei."""
        try:
            if self._cap is not None:
                self._cap.release()
        finally:
            self._cap = None

    def _normalize_brightness(self, frame: Any) -> Any:
        """Hebt zu dunkle Bilder vorsichtig an (Gamma-Korrektur).

        Args:
            frame: Eingabe-Frame von der Kamera.

        Returns:
            Helligkeits-korrigiertes Frame oder Original bei Fehlern.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean())

            if mean_brightness <= 0.0 or mean_brightness >= CameraConfig.BRIGHTNESS_THRESHOLD:
                return frame

            return self._apply_gamma_correction(frame, mean_brightness)
        except Exception:
            return frame

    def _apply_gamma_correction(self, frame: Any, current_brightness: float) -> Any:
        """Wendet Gamma-Korrektur auf unterbelichtete Frames an.

        Args:
            frame: Eingabe-Frame.
            current_brightness: Aktuelle durchschnittliche Helligkeit.

        Returns:
            Gamma-korrigiertes Frame.
        """
        gamma = self._calculate_gamma_value(current_brightness)

        # LUT für Gamma-Korrektur erstellen
        inv_gamma = 1.0 / gamma
        table = ((i / 255.0) ** inv_gamma * 255.0 for i in range(256))

        import numpy as np
        lut = np.array(list(table), dtype="uint8")
        return cv2.LUT(frame, lut)

    def _calculate_gamma_value(self, current_brightness: float) -> float:
        """Berechnet optimalen Gamma-Wert für Helligkeitskorrektur.

        Args:
            current_brightness: Aktuelle durchschnittliche Helligkeit.

        Returns:
            Gamma-Wert zwischen MIN_GAMMA und MAX_GAMMA.
        """
        # Verhindere Division durch Null
        safe_brightness = max(current_brightness, 0.1)
        safe_target = max(CameraConfig.BRIGHTNESS_TARGET, 1.0)

        gamma = math.log(safe_target) / math.log(safe_brightness)
        return min(CameraConfig.MAX_GAMMA, max(CameraConfig.MIN_GAMMA, gamma))

    def capture_frame(self) -> tuple[bytes, dict[str, Any]]:
        """Erfasst ein einzelnes Frame und kodiert es als JPEG (Fallback PNG).

        Returns:
            Tuple aus Bilddaten (Bytes) und Metadaten.

        Raises:
            RuntimeError: Wenn kein Frame gelesen werden kann oder Kodierung fehlschlägt.
        """
        if self._cap is None:
            raise RuntimeError(ErrorMessages.CAMERA_NOT_INITIALIZED)

        # Frame von Kamera lesen
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(ErrorMessages.FRAME_READ_FAILED)

        # Helligkeitskorrektur anwenden
        frame = self._normalize_brightness(frame)

        # Frame kodieren (JPEG mit PNG Fallback)
        image_data, image_format = self._encode_frame(frame)

        # Metadaten erstellen
        metadata = self._create_frame_metadata(image_data, image_format)

        return image_data, metadata

    def _encode_frame(self, frame: Any) -> tuple[bytes, str]:
        """Kodiert Frame als JPEG oder PNG (Fallback).

        Args:
            frame: Zu kodierendes Frame.

        Returns:
            Tuple aus Bilddaten und Format-String.

        Raises:
            RuntimeError: Wenn beide Kodierungsversuche fehlschlagen.
        """
        # JPEG zuerst versuchen
        try:
            return self._encode_frame_as_jpeg(frame)
        except Exception:
            # PNG Fallback
            return self._encode_frame_as_png(frame)

    def _encode_frame_as_jpeg(self, frame: Any) -> tuple[bytes, str]:
        """Kodiert Frame als JPEG.

        Args:
            frame: Zu kodierendes Frame.

        Returns:
            Tuple aus JPEG-Daten und Format-String.

        Raises:
            ValueError: Wenn JPEG-Kodierung fehlschlägt.
        """
        encode_ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), CameraConfig.JPEG_QUALITY]
        )
        if not encode_ok:
            raise ValueError(ErrorMessages.JPEG_ENCODING_FAILED)
        return buf.tobytes(), ImageFormats.JPEG

    def _encode_frame_as_png(self, frame: Any) -> tuple[bytes, str]:
        """Kodiert Frame als PNG.

        Args:
            frame: Zu kodierendes Frame.

        Returns:
            Tuple aus PNG-Daten und Format-String.

        Raises:
            RuntimeError: Wenn PNG-Kodierung fehlschlägt.
        """
        encode_ok, buf = cv2.imencode(
            ".png",
            frame,
            [int(cv2.IMWRITE_PNG_COMPRESSION), CameraConfig.PNG_COMPRESSION_LEVEL]
        )
        if not encode_ok:
            raise RuntimeError(ErrorMessages.PNG_ENCODING_FAILED)
        return buf.tobytes(), ImageFormats.PNG

    def _create_frame_metadata(self, image_data: bytes, image_format: str) -> dict[str, Any]:
        """Erstellt Metadaten für erfasstes Frame.

        Args:
            image_data: Kodierte Bilddaten.
            image_format: MIME-Type des Bildformats.

        Returns:
            Dictionary mit Frame-Metadaten.

        Raises:
            RuntimeError: Wenn Bildgröße das Limit überschreitet.
        """
        # Größenlimit prüfen
        if len(image_data) > CameraConfig.MAX_IMAGE_SIZE_BYTES:
            limit_mb = CameraConfig.MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
            raise RuntimeError(ErrorMessages.IMAGE_SIZE_EXCEEDED.format(limit=limit_mb))

        # Aktuelle Auflösung abrufen
        if self._cap is not None:
            width = int(self._cap.get(CameraConfig.CV_PROP_FRAME_WIDTH))
            height = int(self._cap.get(CameraConfig.CV_PROP_FRAME_HEIGHT))
        else:
            width, height = self.resolution

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "resolution": {"width": width, "height": height},
            "format": image_format,
            "file_size": len(image_data),
        }


__all__ = [
    "SUPPORTED_RESOLUTIONS",
    "CameraAccessDeniedError",
    "CameraInitializationError",
    "CameraService",
]
