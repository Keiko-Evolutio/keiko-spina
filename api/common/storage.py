"""Gemeinsame Storage-Utilities für API-Module.

Dieses Modul stellt wiederverwendbare In-Memory Storage-Klassen bereit,
die in verschiedenen API-Modulen verwendet werden können.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

T = TypeVar("T")


class BaseStorage(ABC, Generic[T]):
    """Abstrakte Basis-Klasse für Storage-Implementierungen.

    Definiert die gemeinsame Schnittstelle für alle Storage-Backends.
    """

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Ruft Element anhand des Schlüssels ab.

        Args:
            key: Eindeutiger Schlüssel

        Returns:
            Element oder None falls nicht gefunden
        """

    @abstractmethod
    async def set(self, key: str, value: T) -> None:
        """Speichert Element unter dem Schlüssel.

        Args:
            key: Eindeutiger Schlüssel
            value: Zu speicherndes Element
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Löscht Element anhand des Schlüssels.

        Args:
            key: Eindeutiger Schlüssel

        Returns:
            True falls Element gelöscht wurde, False falls nicht gefunden
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Prüft ob Element existiert.

        Args:
            key: Eindeutiger Schlüssel

        Returns:
            True falls Element existiert
        """

    @abstractmethod
    async def list_all(self) -> dict[str, T]:
        """Ruft alle gespeicherten Elemente ab.

        Returns:
            Dictionary mit allen Schlüssel-Wert-Paaren
        """

    @abstractmethod
    async def clear(self) -> None:
        """Löscht alle gespeicherten Elemente."""


class InMemoryStorage(BaseStorage[T]):
    """Thread-sichere In-Memory Storage-Implementierung.

    Bietet grundlegende CRUD-Operationen mit optionaler TTL-Unterstützung.
    Geeignet für Entwicklung, Tests und kleine Deployments.

    Attributes:
        _data: Internes Dictionary für Datenspeicherung
        _lock: AsyncIO-Lock für Thread-Sicherheit
        _ttl_data: TTL-Informationen für Elemente
    """

    def __init__(self, enable_ttl: bool = False) -> None:
        """Initialisiert In-Memory Storage.

        Args:
            enable_ttl: Aktiviert TTL-Unterstützung für automatische Bereinigung
        """
        self._data: dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._enable_ttl = enable_ttl
        self._ttl_data: dict[str, float] | None = {} if enable_ttl else None

    async def get(self, key: str) -> T | None:
        """Ruft Element ab und prüft TTL falls aktiviert."""
        async with self._lock:
            if self._enable_ttl and await self._is_expired(key):
                await self._remove_expired(key)
                return None
            return self._data.get(key)

    async def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Speichert Element mit optionaler TTL.

        Args:
            key: Eindeutiger Schlüssel
            value: Zu speicherndes Element
            ttl_seconds: TTL in Sekunden (nur wenn TTL aktiviert)
        """
        async with self._lock:
            self._data[key] = value
            if self._enable_ttl and ttl_seconds is not None:
                self._ttl_data[key] = time.time() + ttl_seconds

    async def delete(self, key: str) -> bool:
        """Löscht Element und TTL-Informationen."""
        async with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            if self._enable_ttl:
                self._ttl_data.pop(key, None)
            return existed

    async def exists(self, key: str) -> bool:
        """Prüft Existenz unter Berücksichtigung der TTL."""
        async with self._lock:
            if key not in self._data:
                return False
            if self._enable_ttl and await self._is_expired(key):
                await self._remove_expired(key)
                return False
            return True

    async def list_all(self) -> dict[str, T]:
        """Ruft alle nicht-abgelaufenen Elemente ab."""
        async with self._lock:
            if self._enable_ttl:
                await self._cleanup_expired()
            return self._data.copy()

    async def clear(self) -> None:
        """Löscht alle Daten und TTL-Informationen."""
        async with self._lock:
            self._data.clear()
            if self._enable_ttl:
                self._ttl_data.clear()

    async def find_by_condition(self, condition: Callable[[T], bool]) -> list[tuple[str, T]]:
        """Findet Elemente basierend auf einer Bedingung.

        Args:
            condition: Funktion die True für passende Elemente zurückgibt

        Returns:
            Liste von (key, value) Tupeln für passende Elemente
        """
        async with self._lock:
            if self._enable_ttl:
                await self._cleanup_expired()
            return [(k, v) for k, v in self._data.items() if condition(v)]

    async def count(self) -> int:
        """Zählt aktuelle Anzahl der Elemente."""
        async with self._lock:
            if self._enable_ttl:
                await self._cleanup_expired()
            return len(self._data)

    async def _is_expired(self, key: str) -> bool:
        """Prüft ob Element abgelaufen ist."""
        if not self._enable_ttl or key not in self._ttl_data:
            return False
        return time.time() > self._ttl_data[key]

    async def _remove_expired(self, key: str) -> None:
        """Entfernt abgelaufenes Element."""
        self._data.pop(key, None)
        self._ttl_data.pop(key, None)

    async def _cleanup_expired(self) -> None:
        """Bereinigt alle abgelaufenen Elemente."""
        if not self._enable_ttl:
            return

        current_time = time.time()
        expired_keys = [
            key for key, expiry in self._ttl_data.items()
            if current_time > expiry
        ]

        for key in expired_keys:
            self._data.pop(key, None)
            self._ttl_data.pop(key, None)


class NamedInMemoryStorage(InMemoryStorage[T]):
    """Erweiterte In-Memory Storage mit Name-basierter Suche.

    Speziell für Objekte mit 'name'-Attribut oder 'name'-Schlüssel.
    Bietet zusätzliche Methoden für Name-basierte Operationen.
    """

    def __init__(self, enable_ttl: bool = False, name_extractor: Callable[[T], str] | None = None) -> None:
        """Initialisiert Named Storage.

        Args:
            enable_ttl: Aktiviert TTL-Unterstützung
            name_extractor: Funktion zum Extrahieren des Namens aus dem Objekt
        """
        super().__init__(enable_ttl)
        self._name_extractor = name_extractor or self._default_name_extractor

    def _default_name_extractor(self, obj: T) -> str:
        """Standard Name-Extraktor für Objekte mit 'name'-Attribut oder -Schlüssel."""
        if hasattr(obj, "name"):
            return obj.name
        if isinstance(obj, dict) and "name" in obj:
            return obj["name"]  # type: ignore
        if hasattr(obj, "configuration") and isinstance(obj.configuration, dict):
            return obj.configuration.get("name", "")  # type: ignore
        raise ValueError(f"Kann Name nicht aus Objekt extrahieren: {type(obj)}")

    async def exists_by_name(self, name: str) -> bool:
        """Prüft ob Element mit dem Namen existiert.

        Args:
            name: Name des Elements

        Returns:
            True falls Element mit dem Namen existiert
        """
        async with self._lock:
            if self._enable_ttl:
                await self._cleanup_expired()

            for obj in self._data.values():
                try:
                    if self._name_extractor(obj) == name:
                        return True
                except (ValueError, AttributeError, KeyError):
                    continue
            return False

    async def find_by_name(self, name: str) -> tuple[str, T] | None:
        """Findet Element anhand des Namens.

        Args:
            name: Name des Elements

        Returns:
            (key, value) Tupel oder None falls nicht gefunden
        """
        async with self._lock:
            if self._enable_ttl:
                await self._cleanup_expired()

            for key, obj in self._data.items():
                try:
                    if self._name_extractor(obj) == name:
                        return key, obj
                except (ValueError, AttributeError, KeyError):
                    continue
            return None


# Globale Storage-Instanzen für häufig verwendete Patterns
_global_storages: dict[str, BaseStorage[Any]] = {}


def get_storage(name: str, storage_type: type = InMemoryStorage, **kwargs: Any) -> BaseStorage[Any]:
    """Ruft globale Storage-Instanz ab oder erstellt neue.

    Args:
        name: Name der Storage-Instanz
        storage_type: Typ der Storage-Implementierung
        **kwargs: Zusätzliche Parameter für Storage-Konstruktor

    Returns:
        Storage-Instanz
    """
    if name not in _global_storages:
        _global_storages[name] = storage_type(**kwargs)
        logger.debug(f"Neue Storage-Instanz erstellt: {name} ({storage_type.__name__})")

    return _global_storages[name]


def clear_all_storages() -> None:
    """Löscht alle globalen Storage-Instanzen.

    Nützlich für Tests und Cleanup-Operationen.
    """
    _global_storages.clear()
    logger.debug("Alle globalen Storage-Instanzen gelöscht")
