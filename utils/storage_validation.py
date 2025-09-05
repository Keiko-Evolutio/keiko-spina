# backend/utils/storage_validation.py
"""Zentralisierte Storage-Validierung.

Konsolidiert Validierungs-Logik für Storage-Operationen
aus verschiedenen Modulen in eine einheitliche Implementierung.
"""

from __future__ import annotations

from agents.constants import (
    MAX_IMAGE_FILE_SIZE,
    SUPPORTED_IMAGE_CONTENT_TYPES,
)

from .content_type_utils import validate_content_type


class StorageValidationError(Exception):
    """Basis-Exception für Storage-Validierungsfehler."""


class FileSizeError(StorageValidationError):
    """Exception für Dateigröße-Probleme."""


class ContentTypeError(StorageValidationError):
    """Exception für Content-Type-Probleme."""


class FilenameError(StorageValidationError):
    """Exception für Dateiname-Probleme."""


def validate_image_data(image_data: bytes) -> None:
    """Validiert Bilddaten.

    Args:
        image_data: Zu validierende Bilddaten

    Raises:
        FileSizeError: Bei ungültiger Dateigröße
    """
    if len(image_data) == 0:
        raise FileSizeError("Bilddaten sind leer")

    if len(image_data) > MAX_IMAGE_FILE_SIZE:
        raise FileSizeError(
            f"Datei zu groß: {len(image_data)} Bytes "
            f"(Maximum: {MAX_IMAGE_FILE_SIZE} Bytes)"
        )


def validate_content_type_for_upload(content_type: str) -> None:
    """Validiert Content-Type für Upload.

    Args:
        content_type: Zu validierender Content-Type

    Raises:
        ContentTypeError: Bei ungültigem Content-Type
    """
    if not validate_content_type(content_type):
        raise ContentTypeError(
            f"Nicht unterstützter Content-Type: {content_type}. "
            f"Unterstützt: {', '.join(SUPPORTED_IMAGE_CONTENT_TYPES)}"
        )


def validate_filename(filename: str | None) -> None:
    """Validiert Dateiname.

    Args:
        filename: Zu validierender Dateiname

    Raises:
        FilenameError: Bei ungültigem Dateiname
    """
    if not filename or not filename.strip():
        raise FilenameError("Dateiname erforderlich")

    # Zusätzliche Validierungen können hier hinzugefügt werden
    # z.B. ungültige Zeichen, Pfad-Traversal, etc.


def validate_container_name(container_name: str) -> None:
    """Validiert Container-Name.

    Args:
        container_name: Zu validierender Container-Name

    Raises:
        StorageValidationError: Bei ungültigem Container-Name
    """
    if not container_name or not container_name.strip():
        raise StorageValidationError("Container-Name erforderlich")

    # Azure Blob Storage Container-Name Regeln
    if len(container_name) < 3 or len(container_name) > 63:
        raise StorageValidationError(
            f"Container-Name muss zwischen 3 und 63 Zeichen lang sein: {container_name}"
        )

    if not container_name.islower():
        raise StorageValidationError(
            f"Container-Name muss kleingeschrieben sein: {container_name}"
        )


def validate_blob_name(blob_name: str) -> None:
    """Validiert Blob-Name.

    Args:
        blob_name: Zu validierender Blob-Name

    Raises:
        StorageValidationError: Bei ungültigem Blob-Name
    """
    if not blob_name or not blob_name.strip():
        raise StorageValidationError("Blob-Name erforderlich")

    # Azure Blob Storage Blob-Name Regeln
    if len(blob_name) > 1024:
        raise StorageValidationError(
            f"Blob-Name zu lang: {len(blob_name)} Zeichen (Maximum: 1024)"
        )


def validate_upload_parameters(
    image_data: bytes,
    filename: str,
    content_type: str,
    container: str | None = None,
) -> None:
    """Validiert alle Upload-Parameter in einem Aufruf.

    Args:
        image_data: Bilddaten
        filename: Dateiname
        content_type: Content-Type
        container: Optionaler Container-Name

    Raises:
        StorageValidationError: Bei Validierungsfehlern
    """
    validate_image_data(image_data)
    validate_filename(filename)
    validate_content_type_for_upload(content_type)

    if container:
        validate_container_name(container)

    validate_blob_name(filename)


class StorageValidator:
    """Erweiterte Storage-Validierung mit konfigurierbaren Regeln."""

    def __init__(
        self,
        *,
        max_file_size: int = MAX_IMAGE_FILE_SIZE,
        allowed_content_types: set[str] | None = None,
        strict_filename_validation: bool = False,
    ) -> None:
        """Initialisiert Storage-Validator.

        Args:
            max_file_size: Maximale Dateigröße
            allowed_content_types: Erlaubte Content-Types
            strict_filename_validation: Strenge Dateiname-Validierung
        """
        self.max_file_size = max_file_size
        self.allowed_content_types = allowed_content_types or set(SUPPORTED_IMAGE_CONTENT_TYPES)
        self.strict_filename_validation = strict_filename_validation

    def validate_upload(
        self,
        image_data: bytes,
        filename: str,
        content_type: str,
        container: str | None = None,
    ) -> None:
        """Validiert Upload mit konfigurierbaren Regeln.

        Args:
            image_data: Bilddaten
            filename: Dateiname
            content_type: Content-Type
            container: Optionaler Container-Name

        Raises:
            StorageValidationError: Bei Validierungsfehlern
        """
        # Dateigröße prüfen
        if len(image_data) == 0:
            raise FileSizeError("Bilddaten sind leer")

        if len(image_data) > self.max_file_size:
            raise FileSizeError(
                f"Datei zu groß: {len(image_data)} Bytes "
                f"(Maximum: {self.max_file_size} Bytes)"
            )

        # Content-Type prüfen
        if content_type not in self.allowed_content_types:
            raise ContentTypeError(
                f"Nicht unterstützter Content-Type: {content_type}. "
                f"Unterstützt: {', '.join(self.allowed_content_types)}"
            )

        # Dateiname prüfen
        validate_filename(filename)

        if self.strict_filename_validation:
            self._validate_filename_strict(filename)

        # Container prüfen
        if container:
            validate_container_name(container)

        # Blob-Name prüfen
        validate_blob_name(filename)

    def _validate_filename_strict(self, filename: str) -> None:
        """Strenge Dateiname-Validierung."""
        import re

        # Keine gefährlichen Zeichen
        if re.search(r'[<>:"|?*]', filename):
            raise FilenameError(f"Dateiname enthält ungültige Zeichen: {filename}")

        # Keine Pfad-Traversal
        if ".." in filename or filename.startswith(("/", "\\")):
            raise FilenameError(f"Unsicherer Dateiname: {filename}")


__all__ = [
    "ContentTypeError",
    "FileSizeError",
    "FilenameError",
    "StorageValidationError",
    "StorageValidator",
    "validate_blob_name",
    "validate_container_name",
    "validate_content_type_for_upload",
    "validate_filename",
    "validate_image_data",
    "validate_upload_parameters",
]
