"""Gemeinsame Redis-Operationen für das Webhook-System.

Konsolidiert häufig verwendete Redis-Patterns um Duplicate Code zu eliminieren.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from kei_logging import get_logger
from storage.cache.redis_cache import NoOpCache, get_cache_client

logger = get_logger(__name__)

T = TypeVar("T")


class RedisOperationError(Exception):
    """Fehler bei Redis-Operationen."""


async def safe_redis_operation(operation_name: str, operation_func, default_value: Any = None) -> Any:
    """Führt Redis-Operation sicher aus mit standardisiertem Error-Handling.
    
    Args:
        operation_name: Name der Operation für Logging
        operation_func: Async-Funktion die die Redis-Operation ausführt
        default_value: Rückgabewert bei Fehlern
        
    Returns:
        Ergebnis der Operation oder default_value bei Fehlern
    """
    client = await get_cache_client()
    if client is None or isinstance(client, NoOpCache):
        logger.debug(f"Redis nicht verfügbar für {operation_name}")
        return default_value

    try:
        return await operation_func(client)
    except Exception as exc:
        logger.debug(f"Redis-Operation '{operation_name}' fehlgeschlagen: {exc}")
        return default_value


async def redis_get_json(key: str, model_class: type[T] | None = None) -> T | dict[str, Any] | None:
    """Lädt JSON-Daten aus Redis und parsed sie optional zu einem Model.
    
    Args:
        key: Redis-Key
        model_class: Optional Pydantic-Model-Klasse für Parsing
        
    Returns:
        Geparste Daten oder None bei Fehlern
    """
    async def _operation(client):
        data = await client.get(key)
        if not data:
            return None

        json_data = json.loads(data)
        if model_class:
            return model_class(**json_data)
        return json_data

    return await safe_redis_operation(f"get_json:{key}", _operation)


async def redis_set_json(key: str, data: Any, ttl_seconds: int | None = None) -> bool:
    """Speichert Daten als JSON in Redis.
    
    Args:
        key: Redis-Key
        data: Zu speichernde Daten (wird zu JSON serialisiert)
        ttl_seconds: Optional TTL in Sekunden
        
    Returns:
        True bei Erfolg, False bei Fehlern
    """
    async def _operation(client):
        if hasattr(data, "model_dump"):
            # Pydantic Model
            json_str = json.dumps(data.model_dump())
        else:
            json_str = json.dumps(data)

        await client.set(key, json_str)
        if ttl_seconds:
            await client.expire(key, ttl_seconds)
        return True

    result = await safe_redis_operation(f"set_json:{key}", _operation, False)
    return bool(result)


async def redis_hash_get_all(key: str, model_class: type[T] | None = None) -> dict[str, T] | dict[str, dict[str, Any]]:
    """Lädt alle Hash-Felder aus Redis.
    
    Args:
        key: Redis-Hash-Key
        model_class: Optional Model-Klasse für Parsing der Werte
        
    Returns:
        Dictionary mit allen Hash-Feldern
    """
    async def _operation(client):
        hash_data = await client.hgetall(key)
        if not hash_data:
            return {}

        result = {}
        for field, value in hash_data.items():
            try:
                json_data = json.loads(value)
                if model_class:
                    result[field] = model_class(**json_data)
                else:
                    result[field] = json_data
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.debug(f"Fehler beim Parsen von Hash-Feld {field}")
                continue
        return result

    return await safe_redis_operation(f"hash_get_all:{key}", _operation, {})


async def redis_hash_set(key: str, field: str, data: Any) -> bool:
    """Setzt ein Hash-Feld in Redis.
    
    Args:
        key: Redis-Hash-Key
        field: Hash-Feld-Name
        data: Zu speichernde Daten
        
    Returns:
        True bei Erfolg, False bei Fehlern
    """
    async def _operation(client):
        if hasattr(data, "model_dump"):
            json_str = json.dumps(data.model_dump())
        else:
            json_str = json.dumps(data)
        await client.hset(key, field, json_str)
        return True

    result = await safe_redis_operation(f"hash_set:{key}:{field}", _operation, False)
    return bool(result)


async def redis_hash_delete(key: str, field: str) -> bool:
    """Löscht ein Hash-Feld aus Redis.
    
    Args:
        key: Redis-Hash-Key
        field: Hash-Feld-Name
        
    Returns:
        True bei Erfolg, False bei Fehlern
    """
    async def _operation(client):
        await client.hdel(key, field)
        return True

    result = await safe_redis_operation(f"hash_delete:{key}:{field}", _operation, False)
    return bool(result)


async def redis_list_pop(key: str, from_right: bool = True) -> str | None:
    """Holt Element aus Redis-Liste.
    
    Args:
        key: Redis-List-Key
        from_right: True für RPOP, False für LPOP
        
    Returns:
        Element oder None wenn Liste leer
    """
    async def _operation(client):
        if from_right:
            return await client.rpop(key)
        return await client.lpop(key)

    return await safe_redis_operation(f"list_pop:{key}", _operation)


async def redis_list_push(key: str, data: Any, to_left: bool = True) -> bool:
    """Fügt Element zu Redis-Liste hinzu.
    
    Args:
        key: Redis-List-Key
        data: Hinzuzufügende Daten
        to_left: True für LPUSH, False für RPUSH
        
    Returns:
        True bei Erfolg, False bei Fehlern
    """
    async def _operation(client):
        if isinstance(data, str):
            value = data
        else:
            value = json.dumps(data.model_dump() if hasattr(data, "model_dump") else data)

        if to_left:
            await client.lpush(key, value)
        else:
            await client.rpush(key, value)
        return True

    result = await safe_redis_operation(f"list_push:{key}", _operation, False)
    return bool(result)


async def redis_scan_keys(pattern: str, count: int = 100) -> list[str]:
    """Scannt Redis-Keys nach Pattern.
    
    Args:
        pattern: Redis-Pattern (mit * Wildcards)
        count: Anzahl Keys pro Scan-Iteration
        
    Returns:
        Liste gefundener Keys
    """
    async def _operation(client):
        keys = []
        async for key in client.scan_iter(match=pattern, count=count):  # type: ignore[attr-defined]
            keys.append(key)
        return keys

    return await safe_redis_operation(f"scan_keys:{pattern}", _operation, [])


__all__ = [
    "RedisOperationError",
    "redis_get_json",
    "redis_hash_delete",
    "redis_hash_get_all",
    "redis_hash_set",
    "redis_list_pop",
    "redis_list_push",
    "redis_scan_keys",
    "redis_set_json",
    "safe_redis_operation",
]
