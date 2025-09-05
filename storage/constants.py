# backend/storage/constants.py
"""Storage-System Konstanten und Konfiguration."""

from __future__ import annotations

from typing import Any


class StorageConstants:
    """Zentrale Konstanten für das Storage-System."""

    # Timeouts (Sekunden)
    DEFAULT_TIMEOUT = 30
    CONNECTION_TIMEOUT = 5
    REDIS_SOCKET_TIMEOUT = 5

    # Cache TTL (Sekunden)
    CACHE_TTL_CONFIGURATION = 3600
    CACHE_TTL_AGENT = 1800
    CACHE_TTL_SESSION = 300
    CACHE_TTL_USER_PROFILE = 900
    CACHE_TTL_TEMPORARY = 60
    CACHE_TTL_DEFAULT = 300

    # Query-Konstanten
    DEFAULT_BATCH_SIZE = 25
    PRIORITY_NORMAL = 1
    PRIORITY_HIGH = 2

    # Cache-Namespace
    CACHE_NAMESPACE = "keiko"

    # Content-Types
    CONTENT_TYPE_PNG = "image/png"
    CONTENT_TYPE_JPEG = "image/jpeg"

    # SAS-Permissions
    SAS_PERMISSION_READ = "r"
    SAS_PERMISSION_WRITE = "w"

    # Default-Werte
    DEFAULT_EXPIRY_MINUTES = 30
    MIN_EXPIRY_MINUTES = 10


class CacheConfig:
    """Cache-Konfiguration."""

    TTL_MAP: dict[str, int] = {
        "configuration": StorageConstants.CACHE_TTL_CONFIGURATION,
        "agent": StorageConstants.CACHE_TTL_AGENT,
        "session": StorageConstants.CACHE_TTL_SESSION,
        "user_profile": StorageConstants.CACHE_TTL_USER_PROFILE,
        "temporary": StorageConstants.CACHE_TTL_TEMPORARY,
        "default": StorageConstants.CACHE_TTL_DEFAULT,
    }

    REDIS_CONFIG_DEFAULTS: dict[str, Any] = {
        "decode_responses": True,
        "socket_connect_timeout": StorageConstants.REDIS_SOCKET_TIMEOUT,
    }


class ErrorMessages:
    """Standardisierte Fehlermeldungen."""

    AZURE_SDK_NOT_AVAILABLE = "Azure SDK nicht verfügbar. Bitte azure-identity und azure-storage-blob installieren."
    REDIS_NOT_AVAILABLE = "Redis nicht verfügbar"
    CONTAINER_NOT_FOUND = "Container '{container_name}' nicht gefunden"
    CACHE_OPERATION_FAILED = "Cache-Operation fehlgeschlagen"
    COSMOS_READ_FAILED = "Cosmos DB read fehlgeschlagen"
    BLOB_UPLOAD_FAILED = "Fehler beim Hochladen des Bildes '{blob_name}'"
    SAS_GENERATION_FAILED = "SAS-Generierung fehlgeschlagen, fallback auf unsignierte URL"
