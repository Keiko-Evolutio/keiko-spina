# backend/storage/azure_blob_storage/azure_blob_storage.py
"""Azure Blob Storage Management."""

from __future__ import annotations

import contextlib

try:
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError  # type: ignore
    from azure.storage.blob.aio import ContainerClient  # type: ignore
    _AZURE_AVAILABLE = True
except ImportError:  # pragma: no cover
    class ResourceExistsError(Exception):
        """Fallback für Azure ResourceExistsError."""

    class ResourceNotFoundError(Exception):
        """Fallback für Azure ResourceNotFoundError."""

    ContainerClient = None  # type: ignore
    _AZURE_AVAILABLE = False

from datetime import UTC
from typing import TYPE_CHECKING

from kei_logging import get_logger

from ..client_factory import client_factory
from ..constants import ErrorMessages
from ..utils import handle_storage_errors, validate_container_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


@contextlib.asynccontextmanager
@handle_storage_errors("get_storage_client")
async def get_storage_client(
    container_name: str,
    *,
    create_if_missing: bool = True,
) -> AsyncGenerator[ContainerClient, None]:
    """Context Manager für Azure Blob Storage Container-Clients.

    Args:
        container_name: Name des Containers
        create_if_missing: Container automatisch erstellen falls nicht vorhanden

    Yields:
        ContainerClient: Konfigurierter Container-Client

    Raises:
        FileNotFoundError: Container nicht gefunden und create_if_missing=False
        ResourceNotFoundError: Azure-Zugriffsprobleme
    """
    validate_container_name(container_name)
    _, blob_client = await client_factory.get_azure_clients()
    container = blob_client.get_container_client(container_name)

    # Container erstellen falls gewünscht
    if create_if_missing:
        await _create_container_if_needed(container, container_name)

    try:
        # Existenz prüfen falls create_if_missing=False
        if not create_if_missing:
            await container.get_container_properties()

        yield container

    except ResourceNotFoundError as exc:
        error_msg = ErrorMessages.CONTAINER_NOT_FOUND.format(container_name=container_name)
        logger.exception(error_msg)
        raise FileNotFoundError(error_msg) from exc
    finally:
        await container.close()


async def _create_container_if_needed(container: ContainerClient, container_name: str) -> None:
    """Erstellt Container falls er nicht existiert."""
    try:
        await container.create_container()
        logger.info(f"Container '{container_name}' erstellt")
    except ResourceExistsError:
        pass  # Container existiert bereits


async def close_storage_clients() -> None:
    """Schließt globale Storage-Clients und räumt Ressourcen auf.

    Sollte beim Application-Shutdown aufgerufen werden.
    """
    await client_factory.close_all_clients()


@handle_storage_errors("upload_image_bytes")
async def upload_image_bytes(
    container_name: str,
    blob_name: str,
    data: bytes,
    *,
    content_type: str = "image/png",
    overwrite: bool = True,
    metadata: dict = None,
) -> str:
    """Lädt Bild-Bytes in Azure Blob Storage hoch und gibt die Blob-URL zurück.

    Args:
        container_name: Ziel-Containername
        blob_name: Dateiname/Blob-Pfad
        data: Bilddaten als Bytes
        content_type: MIME-Typ (z. B. image/png)
        overwrite: Überschreiben vorhandener Blobs erlauben
        metadata: Optionale Metadaten für den Blob

    Returns:
        Öffentliche Blob-URL (ohne SAS). Für sicheren Zugriff SAS generieren.
    """
    from ..utils import create_operation_context, validate_blob_name

    validate_container_name(container_name)
    validate_blob_name(blob_name)

    _, blob_client = await client_factory.get_azure_clients()
    container = blob_client.get_container_client(container_name)
    blob_client_async = container.get_blob_client(blob_name)

    # Upload-Kontext für Logging
    context = create_operation_context(
        "blob_upload_start",
        container=container_name,
        blob=blob_name,
        size=len(data),
        content_type=content_type
    )
    logger.debug(context)

    # Upload durchführen
    await _perform_blob_upload(blob_client_async, data, content_type, overwrite, metadata)

    url = str(blob_client_async.url)
    logger.info("Bild hochgeladen: %s", url)
    return url


async def _perform_blob_upload(
    blob_client_async,
    data: bytes,
    content_type: str,
    overwrite: bool,
    metadata: dict = None
) -> None:
    """Führt den eigentlichen Blob-Upload durch."""
    from azure.storage.blob import ContentSettings  # type: ignore

    content_settings = ContentSettings(content_type=content_type)

    # Upload mit Fallback für ältere SDK-Versionen
    try:
        await blob_client_async.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=content_settings,
            metadata=metadata or {},
        )
    except TypeError:
        # Fallback: Ältere Signatur mit Timeout
        from ..constants import StorageConstants
        await blob_client_async.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=content_settings,
            metadata=metadata or {},
            timeout=StorageConstants.DEFAULT_TIMEOUT,
        )


@handle_storage_errors("generate_sas_url")
async def generate_sas_url(
    container_name: str,
    blob_name: str,
    *,
    expiry_minutes: int = 30,
    permissions: str = "r",
) -> str:
    """Erzeugt eine zeitlich begrenzte SAS-URL für ein Blob.

    Args:
        container_name: Containername
        blob_name: Blobname
        expiry_minutes: Gültigkeitsdauer in Minuten
        permissions: Zugriffsrechte (z. B. "r" für Read)

    Returns:
        SAS-URL als String
    """
    from ..constants import StorageConstants
    from ..utils import sanitize_log_url, validate_blob_name

    validate_container_name(container_name)
    validate_blob_name(blob_name)

    # Mindest-Gültigkeitsdauer sicherstellen
    effective_expiry = max(expiry_minutes, StorageConstants.MIN_EXPIRY_MINUTES)

    try:
        sas_url = await _generate_user_delegation_sas(
            container_name, blob_name, effective_expiry, permissions
        )
        logger.debug({
            "event": "sas_generated",
            "url_prefix": sanitize_log_url(sas_url)
        })
        return sas_url
    except Exception as e:  # pragma: no cover
        logger.warning(f"{ErrorMessages.SAS_GENERATION_FAILED}: {e}")
        return await _get_fallback_url(container_name, blob_name)


async def _generate_user_delegation_sas(
    container_name: str,
    blob_name: str,
    expiry_minutes: int,
    permissions: str
) -> str:
    """Generiert User Delegation SAS über Managed Identity."""
    from datetime import datetime, timedelta

    from azure.storage.blob import BlobSasPermissions, generate_blob_sas  # type: ignore

    from config.settings import settings

    _, blob_client = await client_factory.get_azure_clients()

    # User Delegation Key abrufen
    key_times = _calculate_key_times(expiry_minutes)
    user_delegation_key = await blob_client.get_user_delegation_key(**key_times)

    # SAS-Token generieren
    expiry = datetime.now(UTC) + timedelta(minutes=expiry_minutes)
    perms = BlobSasPermissions(read="r" in permissions)

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,  # type: ignore
        container_name=container_name,
        blob_name=blob_name,
        user_delegation_key=user_delegation_key,
        permission=perms,
        expiry=expiry,
    )

    account_url = str(settings.storage_account_url).rstrip("/")
    return f"{account_url}/{container_name}/{blob_name}?{sas_token}"


def _calculate_key_times(expiry_minutes: int) -> dict:
    """Berechnet Start- und End-Zeit für User Delegation Key."""
    from datetime import datetime, timedelta

    from ..constants import StorageConstants

    key_start = datetime.now(UTC)
    key_expiry = key_start + timedelta(minutes=max(expiry_minutes + 5, StorageConstants.MIN_EXPIRY_MINUTES))

    return {
        "key_start_time": key_start,
        "key_expiry_time": key_expiry,
    }


async def _get_fallback_url(container_name: str, blob_name: str) -> str:
    """Gibt unsignierte URL als Fallback zurück."""
    _, blob_client = await client_factory.get_azure_clients()
    return str(blob_client.get_blob_client(container_name, blob_name).url)
