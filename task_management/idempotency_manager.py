# backend/task_management/idempotency_manager.py
"""Idempotency Manager für KEI-Agent-Framework Task Management.

Implementiert Request-ID-basierte Idempotenz, Correlation-IDs für Task-Tracking
und Duplicate-Detection mit konfigurierbaren Time-Windows.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .utils import (
    generate_uuid,
    get_current_utc_datetime,
)

logger = get_logger(__name__)


class IdempotencyKeyType(str, Enum):
    """Typen von Idempotency-Keys."""
    REQUEST_ID = "request_id"
    CORRELATION_ID = "correlation_id"
    CUSTOM = "custom"
    HASH_BASED = "hash_based"


class DuplicateDetectionStrategy(str, Enum):
    """Strategien für Duplicate-Detection."""
    EXACT_MATCH = "exact_match"
    CONTENT_HASH = "content_hash"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TIME_WINDOW = "time_window"


@dataclass
class IdempotencyKey:
    """Idempotency-Key-Definition."""
    key: str
    key_type: IdempotencyKeyType
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    # Metadata
    source: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Key abgelaufen ist."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "key": self.key,
            "key_type": self.key_type.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source": self.source,
            "context": self.context,
            "is_expired": self.is_expired
        }


@dataclass
class CorrelationID:
    """Correlation-ID für Task-Tracking."""
    correlation_id: str
    parent_correlation_id: str | None = None
    root_correlation_id: str | None = None

    # Tracking-Information
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = None

    # Hierarchie
    depth: int = 0
    children: set[str] = field(default_factory=set)

    # Kontext
    context: dict[str, Any] = field(default_factory=dict)

    def add_child(self, child_correlation_id: str) -> None:
        """Fügt Child-Correlation-ID hinzu."""
        self.children.add(child_correlation_id)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "parent_correlation_id": self.parent_correlation_id,
            "root_correlation_id": self.root_correlation_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "depth": self.depth,
            "children": list(self.children),
            "context": self.context
        }


@dataclass
class DuplicateDetectionResult:
    """Ergebnis der Duplicate-Detection."""
    is_duplicate: bool
    original_request_id: str | None = None
    original_response: dict[str, Any] | None = None
    detection_strategy: DuplicateDetectionStrategy | None = None
    confidence_score: float = 0.0

    # Detection-Details
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    time_since_original_ms: float | None = None
    similarity_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "is_duplicate": self.is_duplicate,
            "original_request_id": self.original_request_id,
            "original_response": self.original_response,
            "detection_strategy": self.detection_strategy.value if self.detection_strategy else None,
            "confidence_score": self.confidence_score,
            "detected_at": self.detected_at.isoformat(),
            "time_since_original_ms": self.time_since_original_ms,
            "similarity_metrics": self.similarity_metrics
        }


@dataclass
class RequestCache:
    """Cache für Request-Response-Paare."""
    request_id: str
    idempotency_key: str

    # Request-Details
    request_data: dict[str, Any]
    request_hash: str

    # Response-Details
    response_data: dict[str, Any] | None = None
    response_status: str | None = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    expires_at: datetime | None = None

    # Metadata
    correlation_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None

    @property
    def is_completed(self) -> bool:
        """Prüft, ob Request abgeschlossen ist."""
        return self.response_data is not None

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Cache-Entry abgelaufen ist."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "request_data": self.request_data,
            "request_hash": self.request_hash,
            "response_data": self.response_data,
            "response_status": self.response_status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "is_completed": self.is_completed,
            "is_expired": self.is_expired
        }


@dataclass
class IdempotencyConfig:
    """Konfiguration für Idempotency Manager."""
    # Cache-Konfiguration
    default_cache_ttl_seconds: int = 3600  # 1 Stunde
    max_cache_size: int = 10000
    cleanup_interval_seconds: int = 300  # 5 Minuten

    # Duplicate-Detection
    duplicate_detection_window_seconds: int = 300  # 5 Minuten
    content_hash_algorithm: str = "sha256"
    similarity_threshold: float = 0.95

    # Correlation-Tracking
    max_correlation_depth: int = 10
    correlation_ttl_seconds: int = 86400  # 24 Stunden

    # Performance
    enable_async_cleanup: bool = True
    enable_compression: bool = True
    enable_metrics: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "default_cache_ttl_seconds": self.default_cache_ttl_seconds,
            "max_cache_size": self.max_cache_size,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "duplicate_detection_window_seconds": self.duplicate_detection_window_seconds,
            "content_hash_algorithm": self.content_hash_algorithm,
            "similarity_threshold": self.similarity_threshold,
            "max_correlation_depth": self.max_correlation_depth,
            "correlation_ttl_seconds": self.correlation_ttl_seconds,
            "enable_async_cleanup": self.enable_async_cleanup,
            "enable_compression": self.enable_compression,
            "enable_metrics": self.enable_metrics
        }


class IdempotencyManager:
    """Manager für Idempotenz und Korrelation."""

    def __init__(self, config: IdempotencyConfig | None = None):
        """Initialisiert Idempotency Manager.

        Args:
            config: Idempotency-Konfiguration
        """
        self.config = config or IdempotencyConfig()

        # Cache-Storage
        self._request_cache: dict[str, RequestCache] = {}
        self._idempotency_keys: dict[str, IdempotencyKey] = {}
        self._correlation_ids: dict[str, CorrelationID] = {}

        # Indizes für Performance
        self._cache_by_hash: dict[str, str] = {}  # content_hash -> request_id
        self._cache_by_user: dict[str, set[str]] = {}  # user_id -> request_ids
        self._cache_by_correlation: dict[str, set[str]] = {}  # correlation_id -> request_ids

        # Locks für Thread-Safety
        self._cache_lock = asyncio.Lock()
        self._correlation_lock = asyncio.Lock()

        # Background-Tasks
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        # Statistiken
        self._cache_hits = 0
        self._cache_misses = 0
        self._duplicates_detected = 0
        self._correlations_created = 0

    async def start(self) -> None:
        """Startet Idempotency Manager."""
        if self._is_running:
            return

        self._is_running = True

        if self.config.enable_async_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Idempotency Manager gestartet")

    async def stop(self) -> None:
        """Stoppt Idempotency Manager."""
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        logger.info("Idempotency Manager gestoppt")

    @trace_function("idempotency.generate_idempotency_key")
    def generate_idempotency_key(
        self,
        request_data: dict[str, Any],
        key_type: IdempotencyKeyType = IdempotencyKeyType.HASH_BASED,
        custom_key: str | None = None,
        ttl_seconds: int | None = None
    ) -> IdempotencyKey:
        """Generiert Idempotency-Key.

        Args:
            request_data: Request-Daten
            key_type: Key-Typ
            custom_key: Custom Key (für CUSTOM-Typ)
            ttl_seconds: TTL in Sekunden

        Returns:
            Idempotency-Key
        """
        if key_type == IdempotencyKeyType.CUSTOM and custom_key:
            key = custom_key
        elif key_type == IdempotencyKeyType.HASH_BASED:
            key = self._generate_content_hash(request_data)
        else:
            key = generate_uuid()

        expires_at = None
        if ttl_seconds:
            expires_at = get_current_utc_datetime() + timedelta(seconds=ttl_seconds)

        idempotency_key = IdempotencyKey(
            key=key,
            key_type=key_type,
            expires_at=expires_at,
            context={"request_data_size": len(str(request_data))}
        )

        self._idempotency_keys[key] = idempotency_key

        return idempotency_key

    @trace_function("idempotency.create_correlation_id")
    async def create_correlation_id(
        self,
        parent_correlation_id: str | None = None,
        created_by: str | None = None,
        context: dict[str, Any] | None = None
    ) -> CorrelationID:
        """Erstellt Correlation-ID.

        Args:
            parent_correlation_id: Parent-Correlation-ID
            created_by: Ersteller
            context: Kontext-Daten

        Returns:
            Correlation-ID
        """
        correlation_id = generate_uuid()

        async with self._correlation_lock:
            # Bestimme Root und Depth
            root_correlation_id = correlation_id
            depth = 0

            if parent_correlation_id:
                parent_corr = self._correlation_ids.get(parent_correlation_id)
                if parent_corr:
                    root_correlation_id = parent_corr.root_correlation_id or parent_correlation_id
                    depth = parent_corr.depth + 1

                    # Füge zu Parent hinzu
                    parent_corr.add_child(correlation_id)

            # Prüfe Max-Depth
            if depth > self.config.max_correlation_depth:
                logger.warning(f"Correlation-Depth-Limit erreicht: {depth}")
                depth = self.config.max_correlation_depth

            # Erstelle Correlation-ID
            corr_id = CorrelationID(
                correlation_id=correlation_id,
                parent_correlation_id=parent_correlation_id,
                root_correlation_id=root_correlation_id,
                created_by=created_by,
                depth=depth,
                context=context or {}
            )

            self._correlation_ids[correlation_id] = corr_id
            self._correlations_created += 1

        logger.debug(f"Correlation-ID erstellt: {correlation_id} (Depth: {depth})")

        return corr_id

    @trace_function("idempotency.check_duplicate")
    async def check_duplicate(
        self,
        request_data: dict[str, Any],
        idempotency_key: str | None = None,
        strategy: DuplicateDetectionStrategy = DuplicateDetectionStrategy.CONTENT_HASH
    ) -> DuplicateDetectionResult:
        """Prüft auf Duplicate-Request.

        Args:
            request_data: Request-Daten
            idempotency_key: Idempotency-Key
            strategy: Detection-Strategie

        Returns:
            Duplicate-Detection-Ergebnis
        """
        async with self._cache_lock:
            if strategy == DuplicateDetectionStrategy.EXACT_MATCH and idempotency_key:
                return await self._check_exact_match(idempotency_key)

            if strategy == DuplicateDetectionStrategy.CONTENT_HASH:
                return await self._check_content_hash(request_data)

            if strategy == DuplicateDetectionStrategy.TIME_WINDOW:
                return await self._check_time_window(request_data)

            # Keine Duplicate-Detection
            return DuplicateDetectionResult(
                is_duplicate=False,
                detection_strategy=strategy
            )

    async def _check_exact_match(self, idempotency_key: str) -> DuplicateDetectionResult:
        """Prüft auf exakte Übereinstimmung."""
        for request_id, cache_entry in self._request_cache.items():
            if cache_entry.idempotency_key == idempotency_key and cache_entry.is_completed:
                self._cache_hits += 1

                time_since_original = None
                if cache_entry.completed_at:
                    time_since_original = (datetime.now(UTC) - cache_entry.completed_at).total_seconds() * 1000

                return DuplicateDetectionResult(
                    is_duplicate=True,
                    original_request_id=request_id,
                    original_response=cache_entry.response_data,
                    detection_strategy=DuplicateDetectionStrategy.EXACT_MATCH,
                    confidence_score=1.0,
                    time_since_original_ms=time_since_original
                )

        self._cache_misses += 1
        return DuplicateDetectionResult(
            is_duplicate=False,
            detection_strategy=DuplicateDetectionStrategy.EXACT_MATCH
        )

    async def _check_content_hash(self, request_data: dict[str, Any]) -> DuplicateDetectionResult:
        """Prüft auf Content-Hash-Übereinstimmung."""
        content_hash = self._generate_content_hash(request_data)

        request_id = self._cache_by_hash.get(content_hash)
        if request_id:
            cache_entry = self._request_cache.get(request_id)
            if cache_entry and cache_entry.is_completed:
                self._cache_hits += 1
                self._duplicates_detected += 1

                time_since_original = None
                if cache_entry.completed_at:
                    time_since_original = (datetime.now(UTC) - cache_entry.completed_at).total_seconds() * 1000

                return DuplicateDetectionResult(
                    is_duplicate=True,
                    original_request_id=request_id,
                    original_response=cache_entry.response_data,
                    detection_strategy=DuplicateDetectionStrategy.CONTENT_HASH,
                    confidence_score=1.0,
                    time_since_original_ms=time_since_original
                )

        self._cache_misses += 1
        return DuplicateDetectionResult(
            is_duplicate=False,
            detection_strategy=DuplicateDetectionStrategy.CONTENT_HASH
        )

    async def _check_time_window(self, request_data: dict[str, Any]) -> DuplicateDetectionResult:
        """Prüft auf Duplicates im Time-Window."""
        cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.duplicate_detection_window_seconds)
        content_hash = self._generate_content_hash(request_data)

        for cache_entry in self._request_cache.values():
            if (cache_entry.created_at >= cutoff_time and
                cache_entry.request_hash == content_hash and
                cache_entry.is_completed):

                self._cache_hits += 1
                self._duplicates_detected += 1

                time_since_original = (datetime.now(UTC) - cache_entry.completed_at).total_seconds() * 1000

                return DuplicateDetectionResult(
                    is_duplicate=True,
                    original_request_id=cache_entry.request_id,
                    original_response=cache_entry.response_data,
                    detection_strategy=DuplicateDetectionStrategy.TIME_WINDOW,
                    confidence_score=0.9,
                    time_since_original_ms=time_since_original
                )

        self._cache_misses += 1
        return DuplicateDetectionResult(
            is_duplicate=False,
            detection_strategy=DuplicateDetectionStrategy.TIME_WINDOW
        )

    @trace_function("idempotency.cache_request")
    async def cache_request(
        self,
        request_id: str,
        idempotency_key: str,
        request_data: dict[str, Any],
        correlation_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ttl_seconds: int | None = None
    ) -> RequestCache:
        """Cached Request für Idempotenz.

        Args:
            request_id: Request-ID
            idempotency_key: Idempotency-Key
            request_data: Request-Daten
            correlation_id: Correlation-ID
            user_id: User-ID
            agent_id: Agent-ID
            ttl_seconds: TTL in Sekunden

        Returns:
            Request-Cache-Entry
        """
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        elif self.config.default_cache_ttl_seconds:
            expires_at = datetime.now(UTC) + timedelta(seconds=self.config.default_cache_ttl_seconds)

        request_hash = self._generate_content_hash(request_data)

        cache_entry = RequestCache(
            request_id=request_id,
            idempotency_key=idempotency_key,
            request_data=request_data,
            request_hash=request_hash,
            expires_at=expires_at,
            correlation_id=correlation_id,
            user_id=user_id,
            agent_id=agent_id
        )

        async with self._cache_lock:
            # Cache-Size-Limit prüfen
            if len(self._request_cache) >= self.config.max_cache_size:
                await self._evict_oldest_entries()

            # Cache-Entry speichern
            self._request_cache[request_id] = cache_entry

            # Indizes aktualisieren
            self._cache_by_hash[request_hash] = request_id

            if user_id:
                if user_id not in self._cache_by_user:
                    self._cache_by_user[user_id] = set()
                self._cache_by_user[user_id].add(request_id)

            if correlation_id:
                if correlation_id not in self._cache_by_correlation:
                    self._cache_by_correlation[correlation_id] = set()
                self._cache_by_correlation[correlation_id].add(request_id)

        logger.debug(f"Request gecacht: {request_id}")

        return cache_entry

    @trace_function("idempotency.update_response")
    async def update_response(
        self,
        request_id: str,
        response_data: dict[str, Any],
        response_status: str = "success"
    ) -> bool:
        """Aktualisiert Response im Cache.

        Args:
            request_id: Request-ID
            response_data: Response-Daten
            response_status: Response-Status

        Returns:
            True wenn erfolgreich
        """
        async with self._cache_lock:
            cache_entry = self._request_cache.get(request_id)
            if not cache_entry:
                return False

            cache_entry.response_data = response_data
            cache_entry.response_status = response_status
            cache_entry.completed_at = datetime.now(UTC)

            logger.debug(f"Response aktualisiert: {request_id}")

            return True

    def _generate_content_hash(self, data: dict[str, Any]) -> str:
        """Generiert Content-Hash für Daten."""
        # Sortiere Keys für konsistente Hashes
        sorted_data = self._sort_dict_recursively(data)
        content_str = str(sorted_data)

        if self.config.content_hash_algorithm == "sha256":
            return hashlib.sha256(content_str.encode("utf-8")).hexdigest()
        if self.config.content_hash_algorithm == "md5":
            return hashlib.md5(content_str.encode("utf-8")).hexdigest()
        return hashlib.sha1(content_str.encode("utf-8")).hexdigest()

    def _sort_dict_recursively(self, obj: Any) -> Any:
        """Sortiert Dictionary rekursiv für konsistente Hashes."""
        if isinstance(obj, dict):
            return {k: self._sort_dict_recursively(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [self._sort_dict_recursively(item) for item in obj]
        return obj

    async def _evict_oldest_entries(self) -> None:
        """Entfernt älteste Cache-Entries."""
        # Sortiere nach created_at
        sorted_entries = sorted(
            self._request_cache.items(),
            key=lambda x: x[1].created_at
        )

        # Entferne älteste 10%
        entries_to_remove = int(len(sorted_entries) * 0.1)

        for i in range(entries_to_remove):
            request_id, cache_entry = sorted_entries[i]
            await self._remove_cache_entry(request_id)

    async def _remove_cache_entry(self, request_id: str) -> None:
        """Entfernt Cache-Entry."""
        cache_entry = self._request_cache.get(request_id)
        if not cache_entry:
            return

        # Aus Hauptcache entfernen
        del self._request_cache[request_id]

        # Aus Indizes entfernen
        self._cache_by_hash.pop(cache_entry.request_hash, None)

        if cache_entry.user_id:
            self._cache_by_user.get(cache_entry.user_id, set()).discard(request_id)

        if cache_entry.correlation_id:
            self._cache_by_correlation.get(cache_entry.correlation_id, set()).discard(request_id)

    async def _cleanup_loop(self) -> None:
        """Background-Loop für Cache-Cleanup."""
        while self._is_running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(self.config.cleanup_interval_seconds)
            except Exception as e:
                logger.exception(f"Cleanup-Loop-Fehler: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_entries(self) -> None:
        """Bereinigt abgelaufene Cache-Entries."""
        now = datetime.now(UTC)
        expired_request_ids = []
        expired_correlation_ids = []
        expired_idempotency_keys = []

        async with self._cache_lock:
            # Finde abgelaufene Request-Cache-Entries
            for request_id, cache_entry in self._request_cache.items():
                if cache_entry.is_expired:
                    expired_request_ids.append(request_id)

            # Entferne abgelaufene Request-Cache-Entries
            for request_id in expired_request_ids:
                await self._remove_cache_entry(request_id)

        async with self._correlation_lock:
            # Finde abgelaufene Correlation-IDs
            correlation_ttl = timedelta(seconds=self.config.correlation_ttl_seconds)
            for correlation_id, corr_obj in self._correlation_ids.items():
                if now - corr_obj.created_at > correlation_ttl:
                    expired_correlation_ids.append(correlation_id)

            # Entferne abgelaufene Correlation-IDs
            for correlation_id in expired_correlation_ids:
                del self._correlation_ids[correlation_id]

        # Finde abgelaufene Idempotency-Keys
        for key, idempotency_key in self._idempotency_keys.items():
            if idempotency_key.is_expired:
                expired_idempotency_keys.append(key)

        # Entferne abgelaufene Idempotency-Keys
        for key in expired_idempotency_keys:
            del self._idempotency_keys[key]

        if expired_request_ids or expired_correlation_ids or expired_idempotency_keys:
            logger.info(f"Cleanup: {len(expired_request_ids)} Requests, {len(expired_correlation_ids)} Correlations, {len(expired_idempotency_keys)} Keys entfernt")

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Idempotency-Manager-Statistiken zurück."""
        return {
            "cache_size": len(self._request_cache),
            "correlation_ids": len(self._correlation_ids),
            "idempotency_keys": len(self._idempotency_keys),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio": self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            "duplicates_detected": self._duplicates_detected,
            "correlations_created": self._correlations_created,
            "is_running": self._is_running,
            "config": self.config.to_dict()
        }


# Globale Idempotency Manager Instanz
idempotency_manager = IdempotencyManager()
