# backend/storage/utils.py
"""Utility-Funktionen für das Storage-System."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel

from kei_logging import get_logger

from .constants import StorageConstants

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_storage_errors(operation_name: str) -> Callable[[F], F]:
    """Decorator für einheitliches Error-Handling in Storage-Operationen."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                logger.error("%s fehlgeschlagen - Verbindungsproblem: %s", operation_name, e)
                raise
            except (ValueError, TypeError) as e:
                logger.error("%s fehlgeschlagen - Validierungsfehler: %s", operation_name, e)
                raise
            except Exception as e:
                logger.exception("%s fehlgeschlagen - Unerwarteter Fehler: %s", operation_name, e)
                raise
        return wrapper  # type: ignore
    return decorator


class JsonEncoder(json.JSONEncoder):
    """JSON-Encoder für Dataclasses, Pydantic und DateTime."""

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_value(value: Any) -> str:
    """Serialisiert Wert zu JSON-String."""
    return json.dumps(value, cls=JsonEncoder, ensure_ascii=False)


def deserialize_value(value: str) -> Any:
    """Deserialisiert JSON-String zu Python-Objekt."""
    return json.loads(value)


def build_cache_key(prefix: str, key: str, namespace: str = StorageConstants.CACHE_NAMESPACE) -> str:
    """Erstellt standardisierten Cache-Key."""
    return f"{namespace}:{prefix}:{key}"


def validate_container_name(container_name: str) -> None:
    """Validiert Azure Container-Namen."""
    if not container_name:
        raise ValueError("Container-Name darf nicht leer sein")

    if not container_name.islower():
        raise ValueError("Container-Name muss kleingeschrieben sein")

    if len(container_name) < 3 or len(container_name) > 63:
        raise ValueError("Container-Name muss zwischen 3 und 63 Zeichen lang sein")


def validate_blob_name(blob_name: str) -> None:
    """Validiert Azure Blob-Namen."""
    if not blob_name:
        raise ValueError("Blob-Name darf nicht leer sein")

    if blob_name.startswith("/") or blob_name.endswith("/"):
        raise ValueError("Blob-Name darf nicht mit '/' beginnen oder enden")


def sanitize_log_url(url: str) -> str:
    """Entfernt sensitive Daten aus URLs für Logging."""
    if "?" in url:
        base_url, _ = url.split("?", 1)
        return f"{base_url}?[SAS-TOKEN]"
    return url


def get_content_type_from_extension(filename: str) -> str:
    """Ermittelt Content-Type basierend auf Dateiendung."""
    extension = filename.lower().split(".")[-1] if "." in filename else ""

    content_type_map = {
        "png": StorageConstants.CONTENT_TYPE_PNG,
        "jpg": StorageConstants.CONTENT_TYPE_JPEG,
        "jpeg": StorageConstants.CONTENT_TYPE_JPEG,
    }

    return content_type_map.get(extension, StorageConstants.CONTENT_TYPE_PNG)


def calculate_cache_ttl(cache_type: str, custom_ttl: int | None = None) -> int:
    """Berechnet effektive Cache-TTL."""
    if custom_ttl is not None:
        return custom_ttl

    from .constants import CacheConfig
    return CacheConfig.TTL_MAP.get(cache_type, CacheConfig.TTL_MAP["default"])


class PerformanceTimer:
    """Einfacher Performance-Timer für Operationen."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: float | None = None

    def __enter__(self) -> PerformanceTimer:
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            import time
            duration = time.time() - self.start_time
            logger.debug("%s dauerte %.3fs", self.operation_name, duration)


def create_operation_context(operation_name: str, **context_data) -> dict[str, Any]:
    """Erstellt Kontext-Dictionary für Logging."""
    return {
        "operation": operation_name,
        "timestamp": datetime.now().isoformat(),
        **context_data
    }
