"""Utility-Komponenten für das Mesh-Subsystem.

Stellt wiederverwendbare, thread-sichere Komponenten bereit:

## Hash-Generierung
- **HashGenerator**: Zentrale Hash-Funktionen für Idempotenz und Caching
- Konsistente Hash-Generierung für verschiedene Datentypen

## Cache-Management
- **ThreadSafeCache**: LRU-Cache mit TTL
- **IdempotencyManager**: Cache für Duplikatserkennung
- Thread-sichere Operationen mit RLock

## Thread-Safety-Utilities
- Thread-sichere Counter-Implementierung
- Atomic Increment/Decrement-Operationen
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, TypeVar

from .mesh_constants import DEFAULT_ENCODING, HASH_ALGORITHM

T = TypeVar("T")


class HashGenerator:
    """Hash-Generator für konsistente Hash-Erzeugung.

    Bietet statische Methoden für verschiedene Hash-Anwendungsfälle:
    - Content-Hashing für beliebige Datentypen
    - Cache-Key-Generierung aus Komponenten
    - Idempotenz-Hashes für Event-Duplikatserkennung
    """

    @staticmethod
    def generate_content_hash(
        content: str | dict[str, Any],
        algorithm: str = HASH_ALGORITHM,
        encoding: str = DEFAULT_ENCODING
    ) -> str:
        """Generiert deterministischen Hash für beliebigen Content.

        Args:
            content: Zu hashender Content (String oder Dictionary)
            algorithm: Hash-Algorithmus (Standard: sha256)
            encoding: Text-Encoding (Standard: utf-8)

        Returns:
            Hexadezimaler Hash-String

        Raises:
            ValueError: Bei ungültigem Hash-Algorithmus
        """
        try:
            if isinstance(content, dict):
                # Sortierte Dict-Items für deterministische Hashes
                content_str = str(sorted(content.items()))
            else:
                content_str = str(content)

            return hashlib.new(algorithm, content_str.encode(encoding)).hexdigest()
        except ValueError as e:
            raise ValueError(f"Ungültiger Hash-Algorithmus '{algorithm}': {e}") from e

    @staticmethod
    def generate_cache_key(
        components: list[str | None],
        separator: str = "|",
        algorithm: str = HASH_ALGORITHM
    ) -> str:
        """Generiert deterministischen Cache-Key aus Komponenten.

        Args:
            components: Liste von Key-Komponenten (None-Werte werden gefiltert)
            separator: Trennzeichen zwischen Komponenten (Standard: |)
            algorithm: Hash-Algorithmus (Standard: sha256)

        Returns:
            Hexadezimaler Hash-String als Cache-Key

        Raises:
            ValueError: Bei ungültigem Hash-Algorithmus
        """
        try:
            # Filtere None-Werte und konvertiere zu Strings
            clean_components = [str(comp) for comp in components if comp is not None]
            raw_key = separator.join(clean_components)
            return hashlib.new(algorithm, raw_key.encode(DEFAULT_ENCODING)).hexdigest()
        except ValueError as e:
            raise ValueError(f"Ungültiger Hash-Algorithmus '{algorithm}': {e}") from e

    @staticmethod
    def generate_idempotency_hash(
        event_type: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        agent_id: str | None = None
    ) -> str:
        """Generiert Idempotenz-Hash für Event-Duplikatserkennung.

        Args:
            event_type: Event-Typ (erforderlich)
            payload: Event-Payload als Dictionary
            idempotency_key: Optionaler expliziter Idempotenz-Key
            agent_id: Optionale Agent-ID für Scoping

        Returns:
            Deterministischer Idempotenz-Hash

        Note:
            Der Hash berücksichtigt alle Parameter für maximale Eindeutigkeit.
            Gleiche Events mit gleichen Parametern erzeugen identische Hashes.
        """
        components = [
            idempotency_key or "",
            event_type,
            agent_id or "",
            str(sorted(payload.items())),
        ]
        return HashGenerator.generate_cache_key(components)


class ThreadSafeCache:
    """Thread-sicherer LRU-Cache mit TTL-Support.

    Features:
    - LRU (Least Recently Used) Eviction-Policy
    - TTL (Time To Live) für automatische Expiration
    - Thread-sichere Operationen mit RLock
    - Automatische Größenbegrenzung
    """

    def __init__(self, max_size: int = 10_000, default_ttl: int = 300) -> None:
        """Initialisiert Thread-sicheren Cache.

        Args:
            max_size: Maximale Cache-Größe (Standard: 10.000)
            default_ttl: Standard-TTL in Sekunden (Standard: 300 = 5min)

        Raises:
            ValueError: Bei ungültigen Parametern
        """
        if max_size <= 0:
            raise ValueError("max_size muss positiv sein")
        if default_ttl <= 0:
            raise ValueError("default_ttl muss positiv sein")
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._access_times: dict[str, float] = {}

    def get(self, key: str, default: T = None) -> T | None:
        """Holt Wert aus Cache mit automatischer TTL-Prüfung.

        Args:
            key: Cache-Key
            default: Rückgabewert bei Cache-Miss

        Returns:
            Cached Value oder default bei Miss/Expiration
        """
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return default

            # TTL-Prüfung
            if time.time() - entry["timestamp"] > entry.get("ttl", self._default_ttl):
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
                return default

            # Access-Time aktualisieren für LRU
            self._access_times[key] = time.time()
            return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Setzt Wert im Cache mit optionaler TTL.

        Args:
            key: Cache-Key
            value: Zu cachender Wert
            ttl: Optionale TTL in Sekunden (überschreibt default_ttl)
        """
        with self._lock:
            # Cache-Größe begrenzen (LRU-Eviction)
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl or self._default_ttl
            }
            self._access_times[key] = time.time()

    def remove(self, key: str) -> bool:
        """Entfernt Eintrag aus Cache."""
        with self._lock:
            removed = key in self._cache
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            return removed

    def clear(self) -> int:
        """Löscht alle Cache-Einträge."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            return count

    def size(self) -> int:
        """Gibt aktuelle Cache-Größe zurück."""
        with self._lock:
            return len(self._cache)

    def _evict_lru(self) -> None:
        """Entfernt den am längsten nicht verwendeten Eintrag."""
        if not self._access_times:
            return

        # Finde Key mit ältester Access-Time
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._cache.pop(lru_key, None)
        self._access_times.pop(lru_key, None)


class IdempotencyManager:
    """Manager für Idempotenz-Prüfungen mit Cache."""

    def __init__(self, cache_size: int = 10_000, ttl: int = 300):
        """Initialisiert Idempotency-Manager.

        Args:
            cache_size: Maximale Cache-Größe
            ttl: TTL für Idempotenz-Einträge in Sekunden
        """
        self._seen_hashes = ThreadSafeCache(max_size=cache_size, default_ttl=ttl)
        self._duplicate_count = 0
        self._check_count = 0

    def is_duplicate(self, hash_value: str) -> bool:
        """Prüft ob Hash bereits gesehen wurde.

        Args:
            hash_value: Zu prüfender Hash

        Returns:
            True wenn Duplikat, False wenn neu
        """
        self._check_count += 1

        if self._seen_hashes.get(hash_value):
            self._duplicate_count += 1
            return True

        # Als gesehen markieren
        self._seen_hashes.set(hash_value, True)
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Idempotenz-Statistiken zurück."""
        return {
            "total_checks": self._check_count,
            "duplicates_found": self._duplicate_count,
            "duplicate_rate": self._duplicate_count / max(self._check_count, 1),
            "cache_size": self._seen_hashes.size()
        }

    def clear(self) -> int:
        """Löscht Idempotenz-Cache."""
        return self._seen_hashes.clear()


def create_thread_safe_counter() -> dict[str, Any]:
    """Erstellt thread-sicheren Counter mit Lock."""
    return {
        "value": 0,
        "lock": threading.RLock()
    }


def increment_counter(counter: dict[str, Any], amount: int = 1) -> int:
    """Inkrementiert thread-sicheren Counter."""
    with counter["lock"]:
        counter["value"] += amount
        return counter["value"]


def get_counter_value(counter: dict[str, Any]) -> int:
    """Holt thread-sicheren Counter-Wert."""
    with counter["lock"]:
        return counter["value"]


__all__ = [
    "HashGenerator",
    "IdempotencyManager",
    "ThreadSafeCache",
    "create_thread_safe_counter",
    "get_counter_value",
    "increment_counter",
]
