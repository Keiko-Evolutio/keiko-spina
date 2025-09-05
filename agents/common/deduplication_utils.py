# backend/agents/common/deduplication_utils.py
"""Konsolidierte Deduplication-Utilities.

Zentrale Implementierung aller Deduplication-Patterns die in verschiedenen
Modulen dupliziert waren. Eliminiert Code-Duplikation und bietet einheitliche API.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from kei_logging import get_logger

from ..constants import (
    CACHE_CLEANUP_INTERVAL,
    CACHE_MAX_SIZE,
    CACHE_TTL,
)

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class DeduplicationEntry:
    """Eintrag für Deduplication-Cache."""

    fingerprint: str
    timestamp: float
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class DeduplicationStrategy(ABC):
    """Abstrakte Basis für Deduplication-Strategien."""

    @abstractmethod
    def generate_fingerprint(self, item: Any) -> str:
        """Generiert eindeutigen Fingerprint für Item."""

    @abstractmethod
    def is_duplicate(self, fingerprint: str) -> bool:
        """Prüft ob Fingerprint bereits existiert."""


class HashBasedDeduplication(DeduplicationStrategy):
    """Hash-basierte Deduplication-Strategie."""

    def __init__(self, cache_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL):
        """Initialisiert Hash-basierte Deduplication.

        Args:
            cache_size: Maximale Cache-Größe
            ttl: Time-to-Live für Einträge in Sekunden
        """
        self._cache: dict[str, DeduplicationEntry] = {}
        self._cache_size = cache_size
        self._ttl = ttl
        self._last_cleanup = time.time()

    def generate_fingerprint(self, item: Any) -> str:
        """Generiert SHA-256 Hash als Fingerprint.

        Args:
            item: Zu hashender Gegenstand

        Returns:
            SHA-256 Hash als Hex-String
        """
        if isinstance(item, str):
            content = item.encode("utf-8")
        elif isinstance(item, dict):
            # Sortierte JSON-Repräsentation für konsistente Hashes
            import json
            content = json.dumps(item, sort_keys=True).encode("utf-8")
        else:
            content = str(item).encode("utf-8")

        return hashlib.sha256(content).hexdigest()

    def is_duplicate(self, fingerprint: str) -> bool:
        """Prüft ob Fingerprint bereits im Cache existiert.

        Args:
            fingerprint: Zu prüfender Fingerprint

        Returns:
            True wenn Duplikat, False wenn neu
        """
        self._maybe_cleanup()

        # Prüfe nach Cleanup nochmal ob Eintrag noch existiert
        if fingerprint in self._cache:
            entry = self._cache[fingerprint]
            entry.access_count += 1
            return True

        # Neuen Eintrag hinzufügen
        self._cache[fingerprint] = DeduplicationEntry(
            fingerprint=fingerprint,
            timestamp=time.time()
        )

        self._enforce_size_limit()
        return False

    def _maybe_cleanup(self) -> None:
        """Bereinigt abgelaufene Einträge falls nötig."""
        current_time = time.time()
        if current_time - self._last_cleanup < CACHE_CLEANUP_INTERVAL:
            return

        expired_keys = []
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self._ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        self._last_cleanup = current_time

        if expired_keys:
            logger.debug(f"Bereinigt {len(expired_keys)} abgelaufene Deduplication-Einträge")

    def _enforce_size_limit(self) -> None:
        """Stellt sicher dass Cache-Größe nicht überschritten wird."""
        if len(self._cache) <= self._cache_size:
            return

        # Entferne älteste Einträge (LRU-ähnlich)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )

        entries_to_remove = len(self._cache) - self._cache_size
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self._cache[key]


class SignatureBasedDeduplication(DeduplicationStrategy):
    """Signatur-basierte Deduplication für strukturierte Daten."""

    def __init__(self, cache_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL):
        """Initialisiert Signatur-basierte Deduplication.

        Args:
            cache_size: Maximale Cache-Größe
            ttl: Time-to-Live für Einträge in Sekunden
        """
        self._cache: dict[str, DeduplicationEntry] = {}
        self._cache_size = cache_size
        self._ttl = ttl

    def generate_fingerprint(self, item: Any) -> str:
        """Generiert strukturelle Signatur als Fingerprint.

        Args:
            item: Strukturiertes Objekt

        Returns:
            Strukturelle Signatur als String
        """
        if hasattr(item, "__dict__"):
            # Objekt mit Attributen
            attrs = sorted(item.__dict__.items())
            return "|".join(f"{k}:{v}" for k, v in attrs)
        if isinstance(item, dict):
            # Dictionary
            sorted_items = sorted(item.items())
            return "|".join(f"{k}:{v}" for k, v in sorted_items)
        if isinstance(item, (list, tuple)):
            # Sequenz
            return "|".join(str(x) for x in item)
        return str(item)

    def is_duplicate(self, fingerprint: str) -> bool:
        """Prüft ob Signatur bereits existiert."""
        current_time = time.time()

        # Bereinige abgelaufene Einträge
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self._ttl
        ]
        for key in expired_keys:
            del self._cache[key]

        if fingerprint in self._cache:
            self._cache[fingerprint].access_count += 1
            return True

        self._cache[fingerprint] = DeduplicationEntry(
            fingerprint=fingerprint,
            timestamp=current_time
        )
        return False


class UnifiedDeduplicationManager(Generic[T]):
    """Einheitlicher Deduplication-Manager für alle Use Cases."""

    def __init__(
        self,
        strategy: DeduplicationStrategy | None = None,
        cache_size: int = CACHE_MAX_SIZE,
        ttl: int = CACHE_TTL
    ):
        """Initialisiert Unified Deduplication Manager.

        Args:
            strategy: Deduplication-Strategie (Standard: HashBasedDeduplication)
            cache_size: Maximale Cache-Größe
            ttl: Time-to-Live für Einträge
        """
        self._strategy = strategy or HashBasedDeduplication(cache_size, ttl)
        self._duplicate_count = 0
        self._check_count = 0

    def is_duplicate(self, item: T) -> bool:
        """Prüft ob Item ein Duplikat ist.

        Args:
            item: Zu prüfendes Item

        Returns:
            True wenn Duplikat, False wenn neu
        """
        self._check_count += 1
        fingerprint = self._strategy.generate_fingerprint(item)

        if self._strategy.is_duplicate(fingerprint):
            self._duplicate_count += 1
            return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Deduplication-Statistiken zurück.

        Returns:
            Dictionary mit Statistiken
        """
        return {
            "total_checks": self._check_count,
            "duplicates_found": self._duplicate_count,
            "duplicate_rate": (
                self._duplicate_count / self._check_count
                if self._check_count > 0 else 0.0
            ),
            "cache_size": len(getattr(self._strategy, "_cache", {})),
        }


# Convenience-Funktionen für häufige Use Cases

def create_alert_deduplicator(
    cache_size: int = CACHE_MAX_SIZE,
    ttl: int = CACHE_TTL
) -> UnifiedDeduplicationManager[dict[str, Any]]:
    """Erstellt Deduplicator für Alerts."""
    return UnifiedDeduplicationManager(
        HashBasedDeduplication(cache_size, ttl)
    )


def create_event_deduplicator(
    cache_size: int = CACHE_MAX_SIZE,
    ttl: int = CACHE_TTL
) -> UnifiedDeduplicationManager[Any]:
    """Erstellt Deduplicator für Events."""
    return UnifiedDeduplicationManager(
        SignatureBasedDeduplication(cache_size, ttl)
    )


def create_idempotency_manager(
    cache_size: int = CACHE_MAX_SIZE,
    ttl: int = CACHE_TTL
) -> UnifiedDeduplicationManager[str]:
    """Erstellt Idempotency-Manager für Request-Deduplication."""
    return UnifiedDeduplicationManager(
        HashBasedDeduplication(cache_size, ttl)
    )


__all__ = [
    "DeduplicationEntry",
    "DeduplicationStrategy",
    "HashBasedDeduplication",
    "SignatureBasedDeduplication",
    "UnifiedDeduplicationManager",
    "create_alert_deduplicator",
    "create_event_deduplicator",
    "create_idempotency_manager",
]
