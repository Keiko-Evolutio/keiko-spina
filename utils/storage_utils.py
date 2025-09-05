# backend/utils/storage_utils.py
"""Konsolidierte Storage-Utilities für Azure Blob Storage.

Zentralisiert Upload-Funktionen und URL-Generierung für bessere
Wartbarkeit und Testbarkeit.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from agents.constants import (
    DEFAULT_IMAGE_CONTAINER,
    DEFAULT_SAS_EXPIRY_MINUTES,
    ErrorMessagesImageGenerator,
)
from agents.constants import (
    LogEventsImageGenerator as LogEvents,
)
from kei_logging import get_logger

from .content_type_utils import (
    normalize_content_type,
)
from .sanitization_constants import (
    MILLISECONDS_CONVERSION_FACTOR,
    UNIQUE_ID_LENGTH,
)
from .storage_validation import (
    StorageValidator,
)

logger = get_logger(__name__)

# Lazy import für Azure Storage (optional dependency)
try:
    from storage.azure_blob_storage import azure_blob_storage
except ImportError:
    azure_blob_storage = None


class StorageError(Exception):
    """Basis-Exception für Storage-Operationen."""


class StorageNotConfiguredError(StorageError):
    """Exception wenn Azure Storage nicht konfiguriert ist."""


class StorageUploadError(StorageError):
    """Exception bei Upload-Fehlern."""


class StorageUtils:
    """Zentrale Utility-Klasse für Azure Storage-Operationen.

    Bietet einheitliche Schnittstelle für Upload und URL-Generierung
    mit verbessertem Error-Handling und Logging.
    """

    def __init__(
        self,
        *,
        default_container: str = DEFAULT_IMAGE_CONTAINER,
        default_expiry_minutes: int = DEFAULT_SAS_EXPIRY_MINUTES,
        storage_client: Any | None = None,
        validator: StorageValidator | None = None,
    ) -> None:
        """Initialisiert Storage Utils.

        Args:
            default_container: Standard-Container für Uploads
            default_expiry_minutes: Standard-Gültigkeitsdauer für SAS-URLs
            storage_client: Optionaler Storage-Client (für Tests)
            validator: Optionaler Storage-Validator
        """
        self.default_container = default_container
        self.default_expiry_minutes = default_expiry_minutes
        self._storage_client = storage_client or azure_blob_storage
        self._validator = validator or StorageValidator()

        if self._storage_client is None:
            logger.warning("Azure Storage Client nicht verfügbar")

        logger.debug(
            {
                "event": "storage_utils_init",
                "default_container": default_container,
                "default_expiry_minutes": default_expiry_minutes,
                "storage_available": self._storage_client is not None,
            }
        )

    @property
    def is_available(self) -> bool:
        """Prüft ob Storage verfügbar ist."""
        return self._storage_client is not None

    async def upload_image(
        self,
        image_data: bytes,
        *,
        filename: str | None = None,
        container: str | None = None,
        content_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Lädt ein Bild in Azure Storage hoch.

        Args:
            image_data: Bilddaten als Bytes
            filename: Optionaler Dateiname (wird generiert falls None)
            container: Container-Name (verwendet default falls None)
            content_type: MIME-Type (wird erkannt falls None)
            user_id: Benutzer-ID für Pfad-Generierung
            session_id: Session-ID für Pfad-Generierung
            metadata: Zusätzliche Metadaten

        Returns:
            Upload-Ergebnis mit URLs und Metadaten

        Raises:
            StorageNotConfiguredError: Wenn Storage nicht konfiguriert
            StorageUploadError: Bei Upload-Fehlern
        """
        if not self.is_available:
            raise StorageNotConfiguredError(ErrorMessagesImageGenerator.AZURE_STORAGE_NOT_CONFIGURED)

        # Parameter vorbereiten und validieren
        upload_params = self._prepare_upload_parameters(
            image_data, filename, container, content_type, user_id, session_id
        )

        # Upload durchführen
        return await self._execute_upload(upload_params, metadata or {})

    def _prepare_upload_parameters(
        self,
        image_data: bytes,
        filename: str | None,
        container: str | None,
        content_type: str | None,
        user_id: str | None,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Bereitet Upload-Parameter vor und validiert sie."""
        container = container or self.default_container
        filename = filename or self._generate_filename(user_id, session_id)
        content_type = normalize_content_type(
            content_type,
            filename,
            image_data=image_data
        )

        # Validierung mit zentralem Validator
        self._validator.validate_upload(image_data, filename, content_type, container)

        return {
            "image_data": image_data,
            "container": container,
            "filename": filename,
            "content_type": content_type,
        }

    async def _execute_upload(
        self,
        upload_params: dict[str, Any],
        metadata: dict[str, str]
    ) -> dict[str, Any]:
        """Führt den eigentlichen Upload durch."""
        upload_start = time.perf_counter()

        self._log_upload_start(upload_params)

        try:
            # Upload und SAS-URL-Generierung
            blob_url, sas_url = await self._perform_storage_operations(
                upload_params, metadata
            )

            # Ergebnis zusammenstellen
            result = self._build_upload_result(
                upload_params, blob_url, sas_url, upload_start, metadata
            )

            self._log_upload_success(result)
            return result

        except Exception as e:
            self._log_upload_error(upload_params, upload_start, e)
            raise StorageUploadError(f"Upload fehlgeschlagen: {e}") from e

    async def generate_sas_url(
        self,
        container: str,
        filename: str,
        *,
        expiry_minutes: int | None = None,
    ) -> str:
        """Generiert eine SAS-URL für ein Blob.

        Args:
            container: Container-Name
            filename: Dateiname/Blob-Name
            expiry_minutes: Gültigkeitsdauer (verwendet default falls None)

        Returns:
            SAS-URL

        Raises:
            StorageNotConfiguredError: Wenn Storage nicht konfiguriert
        """
        if not self.is_available:
            raise StorageNotConfiguredError(ErrorMessagesImageGenerator.AZURE_STORAGE_NOT_CONFIGURED)

        expiry_minutes = expiry_minutes or self.default_expiry_minutes

        try:
            sas_url = await self._storage_client.generate_sas_url(
                container=container,
                name=filename,
                expiry_minutes=expiry_minutes,
            )

            logger.debug(
                {
                    "event": "sas_url_generated",
                    "container": container,
                    "filename": filename,
                    "expiry_minutes": expiry_minutes,
                }
            )

            return sas_url

        except Exception as e:
            logger.exception(
                {
                    "event": "sas_url_generation_failed",
                    "container": container,
                    "filename": filename,
                    "error": str(e),
                }
            )
            raise StorageUploadError(f"SAS-URL-Generierung fehlgeschlagen: {e}") from e

    async def _perform_storage_operations(
        self,
        upload_params: dict[str, Any],
        metadata: dict[str, str]
    ) -> tuple[str, str]:
        """Führt Upload und SAS-URL-Generierung durch."""
        # Upload durchführen
        blob_url = await self._storage_client.upload_image_bytes(
            container_name=upload_params["container"],
            blob_name=upload_params["filename"],
            data=upload_params["image_data"],
            content_type=upload_params["content_type"],
            metadata=metadata,
        )

        # SAS-URL generieren
        sas_url = await self._storage_client.generate_sas_url(
            container_name=upload_params["container"],
            blob_name=upload_params["filename"],
            expiry_minutes=self.default_expiry_minutes,
        )

        return blob_url, sas_url

    def _build_upload_result(
        self,
        upload_params: dict[str, Any],
        blob_url: str,
        sas_url: str,
        upload_start: float,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        """Erstellt Upload-Ergebnis-Dictionary."""
        upload_time_ms = (time.perf_counter() - upload_start) * MILLISECONDS_CONVERSION_FACTOR

        return {
            "success": True,
            "blob_url": blob_url,
            "sas_url": sas_url,
            "container": upload_params["container"],
            "filename": upload_params["filename"],
            "content_type": upload_params["content_type"],
            "size_bytes": len(upload_params["image_data"]),
            "upload_time_ms": upload_time_ms,
            "metadata": metadata,
        }

    def _log_upload_start(self, upload_params: dict[str, Any]) -> None:
        """Loggt Upload-Start."""
        logger.debug(
            {
                "event": LogEvents.STORAGE_UPLOAD_START,
                "container": upload_params["container"],
                "filename": upload_params["filename"],
                "content_type": upload_params["content_type"],
                "size_bytes": len(upload_params["image_data"]),
            }
        )

    def _log_upload_success(self, result: dict[str, Any]) -> None:
        """Loggt erfolgreichen Upload."""
        logger.info(
            {
                "event": LogEvents.STORAGE_UPLOAD_COMPLETE,
                "container": result["container"],
                "filename": result["filename"],
                "upload_time_ms": result["upload_time_ms"],
                "size_bytes": result["size_bytes"],
            }
        )

    def _log_upload_error(
        self,
        upload_params: dict[str, Any],
        upload_start: float,
        error: Exception
    ) -> None:
        """Loggt Upload-Fehler."""
        upload_time_ms = (time.perf_counter() - upload_start) * MILLISECONDS_CONVERSION_FACTOR

        logger.error(
            {
                "event": "storage_upload_failed",
                "container": upload_params["container"],
                "filename": upload_params["filename"],
                "upload_time_ms": upload_time_ms,
                "error": str(error),
            }
        )

    def _generate_filename(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        extension: str = "png",
    ) -> str:
        """Generiert einen eindeutigen Dateinamen.

        Args:
            user_id: Benutzer-ID
            session_id: Session-ID
            extension: Datei-Erweiterung

        Returns:
            Generierter Dateiname mit Pfad
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        unique_id = uuid4().hex[:UNIQUE_ID_LENGTH]

        # Pfad-Präfix basierend auf verfügbaren IDs
        path_prefix = session_id or user_id or "anonymous"

        return f"{path_prefix}/{timestamp}_{unique_id}.{extension}"


# Globale Instanz für einfache Verwendung
storage_utils = StorageUtils()


# Convenience-Funktionen für Backward-Compatibility
async def upload_image_bytes(
    container: str,
    name: str,
    data: bytes,
    *,
    content_type: str | None = None,
    **kwargs
) -> str:
    """Convenience-Funktion für Image-Upload.

    Args:
        container: Container-Name
        name: Dateiname
        data: Bilddaten
        content_type: Content-Type
        **kwargs: Zusätzliche Parameter

    Returns:
        Blob-URL
    """
    result = await storage_utils.upload_image(
        image_data=data,
        filename=name,
        container=container,
        content_type=content_type,
        **kwargs
    )
    return result["blob_url"]


async def generate_sas_url(
    container: str,
    name: str,
    *,
    expiry_minutes: int = DEFAULT_SAS_EXPIRY_MINUTES,
    **kwargs
) -> str:
    """Convenience-Funktion für SAS-URL-Generierung.

    Args:
        container: Container-Name
        name: Dateiname
        expiry_minutes: Gültigkeitsdauer
        **kwargs: Zusätzliche Parameter

    Returns:
        SAS-URL
    """
    return await storage_utils.generate_sas_url(
        container=container,
        filename=name,
        expiry_minutes=expiry_minutes,
    )


__all__ = [
    "StorageError",
    "StorageNotConfiguredError",
    "StorageUploadError",
    "StorageUtils",
    "generate_sas_url",
    "storage_utils",
    "upload_image_bytes",
]
