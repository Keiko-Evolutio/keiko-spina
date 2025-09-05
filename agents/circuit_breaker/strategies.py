"""Fallback und Cache Strategien für Agent Circuit Breaker.
Implementiert verschiedene Strategien für Fallback-Agents und Response-Caching.
"""

import asyncio
import hashlib
import json
import time
from typing import Any

from kei_logging import get_logger

from ..constants import (
    DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_FALLBACK_EXECUTION_TIME_MS,
    FALLBACK_AGENTS,
)
from .interfaces import (
    AgentCallContext,
    AgentExecutionResult,
    AgentType,
    ICacheStrategy,
    IFallbackStrategy,
)

logger = get_logger(__name__)


class SimpleFallbackStrategy(IFallbackStrategy):
    """Einfache Fallback-Strategie.
    Verwendet vordefinierte Fallback-Mappings für verschiedene Agent-Typen.
    """

    def __init__(self):
        """Initialisiert Fallback-Strategie mit konfigurierten Mappings."""
        # Fallback-Mappings aus Konstanten laden
        self._fallback_mappings: dict[AgentType, list[str]] = {
            AgentType.VOICE_AGENT: FALLBACK_AGENTS["voice_agent"],
            AgentType.TOOL_AGENT: FALLBACK_AGENTS["tool_agent"],
            AgentType.WORKFLOW_AGENT: FALLBACK_AGENTS["workflow_agent"],
            AgentType.ORCHESTRATOR_AGENT: [],  # Orchestrator hat keine Fallbacks
            AgentType.CUSTOM_AGENT: FALLBACK_AGENTS["custom_agent"]
        }

        # Agent-spezifische Fallback-Mappings
        self._agent_specific_fallbacks: dict[str, list[str]] = {
            "complex_voice_agent": ["simple_voice_agent", "echo_agent"],
            "advanced_tool_agent": ["basic_tool_agent", "generic_tool_agent"],
            "multi_step_workflow": ["single_step_workflow", "simple_workflow_agent"],
            "ai_orchestrator": ["rule_based_orchestrator", "simple_orchestrator"]
        }

        logger.info("Simple fallback strategy initialized")

    async def get_fallback_agent(
        self,
        original_agent_id: str,
        agent_type: AgentType,
        context: AgentCallContext | None = None
    ) -> str | None:
        """Bestimmt Fallback-Agent."""
        # Prüfe agent-spezifische Fallbacks zuerst
        if original_agent_id in self._agent_specific_fallbacks:
            fallbacks = self._agent_specific_fallbacks[original_agent_id]
            if fallbacks:
                # Gib ersten verfügbaren Fallback zurück
                # In einer vollständigen Implementation würde man hier
                # die Verfügbarkeit der Agents prüfen
                return fallbacks[0]

        # Verwende Agent-Type-basierte Fallbacks
        fallbacks = self._fallback_mappings.get(agent_type, [])
        if fallbacks:
            # Filtere den ursprünglichen Agent aus den Fallbacks
            available_fallbacks = [fb for fb in fallbacks if fb != original_agent_id]
            if available_fallbacks:
                return available_fallbacks[0]

        logger.warning(f"No fallback agent found for {original_agent_id} ({agent_type.value})")
        return None

    async def execute_fallback(
        self,
        context: AgentCallContext,
        original_error: Exception
    ) -> AgentExecutionResult:
        """Führt Fallback-Execution aus."""
        fallback_agent_id = await self.get_fallback_agent(
            context.agent_id, context.agent_type, context
        )

        if not fallback_agent_id:
            return AgentExecutionResult(
                success=False,
                error="No fallback agent available",
                fallback_used=False
            )

        # Vereinfachte Fallback-Execution
        # In einer vollständigen Implementation würde man hier
        # den tatsächlichen Fallback-Agent aufrufen

        if context.agent_type == AgentType.VOICE_AGENT:
            # Voice-Agent Fallback: Einfache Text-Response
            fallback_result = f"I apologize, but I'm experiencing technical difficulties. Here's a simple response to your request: {context.task}"
        elif context.agent_type == AgentType.TOOL_AGENT:
            # Tool-Agent Fallback: Basis-Funktionalität
            fallback_result = {"status": "fallback", "message": "Tool executed with basic functionality"}
        elif context.agent_type == AgentType.WORKFLOW_AGENT:
            # Workflow-Agent Fallback: Vereinfachter Workflow
            fallback_result = {"workflow_status": "completed_simplified", "steps": ["basic_step"]}
        else:
            # Generic Fallback
            fallback_result = {"status": "fallback", "original_task": context.task}

        return AgentExecutionResult(
            success=True,
            result=fallback_result,
            execution_time_ms=DEFAULT_FALLBACK_EXECUTION_TIME_MS,
            fallback_used=True,
            fallback_agent_id=fallback_agent_id
        )


class MemoryCacheStrategy(ICacheStrategy):
    """In-Memory Cache-Strategie.
    Cached Agent-Responses im Speicher mit TTL-Support.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_MAX_SIZE,
        cleanup_interval_seconds: int = DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS
    ):
        self.max_size = max_size
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Cache Storage: cache_key -> (response, expiry_time)
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_times: dict[str, float] = {}  # LRU tracking

        # Thread Safety
        self._lock = asyncio.Lock()

        # Cleanup Task
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        logger.info(f"Memory cache strategy initialized (max_size: {max_size})")

    async def start_cleanup_task(self) -> None:
        """Startet periodische Cleanup-Task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug("Cache cleanup task started")

    async def stop_cleanup_task(self) -> None:
        """Stoppt Cleanup-Task."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.debug("Cache cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Periodische Cleanup-Schleife."""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(self.cleanup_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_entries(self) -> None:
        """Entfernt abgelaufene Cache-Einträge."""
        current_time = time.time()

        async with self._lock:
            expired_keys = []

            for cache_key, (response, expiry_time) in self._cache.items():
                if current_time > expiry_time:
                    expired_keys.append(cache_key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _evict_lru_entries(self) -> None:
        """Entfernt LRU-Einträge wenn Cache voll ist."""
        if len(self._cache) <= self.max_size:
            return

        # Sortiere nach Access-Zeit (älteste zuerst)
        sorted_keys = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )

        # Entferne älteste Einträge
        entries_to_remove = len(self._cache) - self.max_size + 1
        for i in range(entries_to_remove):
            if i < len(sorted_keys):
                key_to_remove = sorted_keys[i][0]
                if key_to_remove in self._cache:
                    del self._cache[key_to_remove]
                if key_to_remove in self._access_times:
                    del self._access_times[key_to_remove]

        logger.debug(f"Evicted {entries_to_remove} LRU cache entries")

    async def get_cache_key(self, context: AgentCallContext) -> str:
        """Generiert Cache-Key für Kontext."""
        # Erstelle deterministischen Cache-Key basierend auf Kontext
        key_data = {
            "agent_id": context.agent_id,
            "agent_type": context.agent_type.value,
            "framework": context.framework,
            "task": context.task,
            "tool_name": context.tool_name,
            "tool_parameters": context.tool_parameters
        }

        # Serialisiere zu JSON und hashe
        key_json = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_json.encode()).hexdigest()

        return f"agent_call:{cache_key}"

    async def get_cached_response(self, cache_key: str) -> Any | None:
        """Holt gecachte Response."""
        current_time = time.time()

        async with self._lock:
            if cache_key in self._cache:
                response, expiry_time = self._cache[cache_key]

                if current_time <= expiry_time:
                    # Cache Hit - aktualisiere Access-Zeit
                    self._access_times[cache_key] = current_time
                    logger.debug(f"Cache hit: {cache_key}")
                    return response
                # Abgelaufen - entferne
                del self._cache[cache_key]
                if cache_key in self._access_times:
                    del self._access_times[cache_key]
                logger.debug(f"Cache expired: {cache_key}")

            return None

    async def cache_response(
        self,
        cache_key: str,
        response: Any,
        ttl_seconds: int = 300
    ) -> None:
        """Cached Response."""
        current_time = time.time()
        expiry_time = current_time + ttl_seconds

        async with self._lock:
            # Prüfe Cache-Größe und evict falls nötig
            await self._evict_lru_entries()

            # Cache Response
            self._cache[cache_key] = (response, expiry_time)
            self._access_times[cache_key] = current_time

            logger.debug(f"Cached response: {cache_key} (TTL: {ttl_seconds}s)")

    async def invalidate_cache(self, cache_key: str) -> None:
        """Invalidiert Cache-Eintrag."""
        async with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
            if cache_key in self._access_times:
                del self._access_times[cache_key]

            logger.debug(f"Cache invalidated: {cache_key}")

    async def get_cache_statistics(self) -> dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        async with self._lock:
            current_time = time.time()

            # Zähle aktive Einträge
            active_entries = 0
            expired_entries = 0

            for cache_key, (response, expiry_time) in self._cache.items():
                if current_time <= expiry_time:
                    active_entries += 1
                else:
                    expired_entries += 1

            return {
                "total_entries": len(self._cache),
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "max_size": self.max_size,
                "usage_percentage": (len(self._cache) / self.max_size) * 100,
                "cleanup_interval_seconds": self.cleanup_interval_seconds
            }
