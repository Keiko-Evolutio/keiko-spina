"""Custom Agent Loader für Prompty-basierte Agents.

Lädt und verwaltet Prompty-basierte Custom Agents mit Enterprise-Caching,
Singleton-Pattern und robustem Error-Handling für optimale Performance.
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Final

import prompty

from agents.factory.singleton_mixin import SingletonMixin
from data_models import Agent
from kei_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from prompty.core import Prompty

logger = get_logger(__name__)

# Konstanten
DEFAULT_AGENTS_SUBPATH: Final[str] = "voice/prompty"
PROMPTY_FILE_EXTENSION: Final[str] = "*.prompty"
AGENT_TYPE_PROMPTY: Final[str] = "prompty"

DEFAULT_CACHE_TTL: Final[int] = 3600
MAX_CACHE_SIZE: Final[int] = 1000
MAX_PARALLEL_LOADS: Final[int] = 10
LOAD_TIMEOUT_SECONDS: Final[int] = 30
MAX_RETRY_ATTEMPTS: Final[int] = 3
RETRY_DELAY_SECONDS: Final[float] = 0.5


class CustomAgentLoader(SingletonMixin):
    """Lädt und verwaltet Prompty-basierte Custom Agents.

    Implementiert Singleton-Pattern für konsistente Agent-Verwaltung
    und Enterprise-Caching für optimale Performance.
    """

    def _initialize_singleton(self, agents_dir: Path | None = None) -> None:
        """Initialisiert den Loader mit konfigurierbarem Agent-Verzeichnis."""
        self._agents_dir = agents_dir or CustomAgentLoader._get_default_agents_dir()
        self._cache: dict[str, Prompty] = {}
        self._cache_stats: dict[str, int | str | None] = {
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
            "last_refresh": None,
        }

        logger.debug(f"CustomAgentLoader initialisiert: {self._agents_dir}")

    def _singleton_cleanup(self) -> None:
        """Cleanup-Hook für Singleton-Reset."""
        super()._singleton_cleanup()
        self._cache.clear()
        self._cache_stats.update({
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
            "last_refresh": None,
        })
        logger.debug("CustomAgentLoader Cache geleert")

    @staticmethod
    def _get_default_agents_dir() -> Path:
        """Ermittelt Standard-Verzeichnis für Agents.

        Returns:
            Path zum Standard-Verzeichnis für Agents
        """
        return Path(__file__).parents[2] / DEFAULT_AGENTS_SUBPATH

    async def load_agents(self, refresh: bool = False) -> Mapping[str, Prompty]:
        """Lädt alle .prompty-Dateien aus dem Agent-Verzeichnis.

        Args:
            refresh: Erzwingt Neuladen auch bei vorhandenem Cache

        Returns:
            Mapping von Agent-IDs zu Prompty-Objekten

        Raises:
            RuntimeError: Wenn Agent-Verzeichnis nicht zugänglich ist
        """
        try:
            # Cache-Hit prüfen
            if self._cache and not refresh:
                cache_hits = self._cache_stats.get("cache_hits", 0)
                if isinstance(cache_hits, int):
                    self._cache_stats["cache_hits"] = cache_hits + 1
                logger.debug(f"Cache-Hit: {len(self._cache)} Agents aus Cache geladen")
                return self._cache

            # Cache leeren und neu laden
            self._cache.clear()
            total_loads = self._cache_stats.get("total_loads", 0)
            if isinstance(total_loads, int):
                self._cache_stats["total_loads"] = total_loads + 1

            result = await self._load_from_directory()
            self._cache_stats["last_refresh"] = CustomAgentLoader._get_current_timestamp()

            return result

        except Exception as e:
            failed_loads = self._cache_stats.get("failed_loads", 0)
            if isinstance(failed_loads, int):
                self._cache_stats["failed_loads"] = failed_loads + 1
            logger.exception(f"Fehler beim Laden der Agents: {e}")
            return {}

    async def _load_from_directory(self) -> Mapping[str, Prompty]:
        """Lädt alle Prompty-Dateien aus dem konfigurierten Verzeichnis.

        Returns:
            Mapping von Agent-IDs zu Prompty-Objekten
        """
        prompty_files = self._discover_prompty_files()
        if not prompty_files:
            return self._cache

        logger.debug(f"Lade {len(prompty_files)} .prompty-Dateien parallel")

        results = await self._load_files_parallel(prompty_files)
        self._process_loading_results(results)

        return self._cache

    def _discover_prompty_files(self) -> list[Path]:
        """Entdeckt alle .prompty-Dateien im konfigurierten Verzeichnis.

        Returns:
            Liste der gefundenen .prompty-Dateien
        """
        if not self._agents_dir.exists():
            logger.warning(f"Agent-Verzeichnis {self._agents_dir} existiert nicht")
            return []

        prompty_files = list(self._agents_dir.glob(PROMPTY_FILE_EXTENSION))
        if not prompty_files:
            logger.info(f"Keine .prompty-Dateien in {self._agents_dir} gefunden")

        return prompty_files

    async def _load_files_parallel(self, files: list[Path]) -> list:
        """Lädt Prompty-Dateien parallel mit Batch-Processing und Error-Handling.

        Args:
            files: Liste der zu ladenden Dateien

        Returns:
            Liste der Ergebnisse (Tuples oder Exceptions)
        """
        # Batch-Processing für bessere Performance bei vielen Dateien
        if len(files) <= MAX_PARALLEL_LOADS:
            return await asyncio.gather(
                *[self._load_single_file(file) for file in files],
                return_exceptions=True
            )

        # Große Dateimengen in Batches aufteilen
        results = []
        for i in range(0, len(files), MAX_PARALLEL_LOADS):
            batch = files[i:i + MAX_PARALLEL_LOADS]
            batch_results = await asyncio.gather(
                *[CustomAgentLoader._load_single_file(file) for file in batch],
                return_exceptions=True
            )
            results.extend(batch_results)

            # Pause zwischen Batches
            if i + MAX_PARALLEL_LOADS < len(files):
                await asyncio.sleep(0.1)

        return results

    def _process_loading_results(self, results: list) -> None:
        """Verarbeitet die Ergebnisse des parallelen Ladens.

        Args:
            results: Liste der Lade-Ergebnisse
        """
        loaded_count = 0
        failed_count = 0

        for result in results:
            if isinstance(result, tuple):
                agent_id, prompty_obj = result
                self._cache[agent_id] = prompty_obj
                loaded_count += 1
            elif isinstance(result, Exception):
                failed_count += 1
                logger.warning(f"Agent-Loading fehlgeschlagen: {result}")

        # Cache-Statistiken aktualisieren
        successful_loads = self._cache_stats.get("successful_loads", 0)
        failed_loads = self._cache_stats.get("failed_loads", 0)
        if isinstance(successful_loads, int):
            self._cache_stats["successful_loads"] = successful_loads + loaded_count
        if isinstance(failed_loads, int):
            self._cache_stats["failed_loads"] = failed_loads + failed_count

        logger.info(
            f"Agent-Loading abgeschlossen: {loaded_count} erfolgreich, "
            f"{failed_count} fehlgeschlagen, {len(self._cache)} im Cache"
        )

    @staticmethod
    async def _load_single_file(file_path: Path) -> tuple[str, Prompty]:
        """Lädt einzelne Prompty-Datei mit robustem Error Handling und Retry-Mechanismus.

        Args:
            file_path: Pfad zur .prompty-Datei

        Returns:
            Tuple aus Agent-ID und Prompty-Objekt

        Raises:
            Exception: Bei kritischen Fehlern beim Laden
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                logger.debug(f"Lade Prompty-Datei: {file_path.name} (Versuch {attempt + 1})")

                # Timeout-Schutz
                prompty_obj = await asyncio.wait_for(
                    prompty.load_async(str(file_path)),
                    timeout=LOAD_TIMEOUT_SECONDS
                )
                agent_id = file_path.stem

                logger.debug(f"Prompty-Datei erfolgreich geladen: {agent_id}")
                return agent_id, prompty_obj

            except FileNotFoundError:
                error_msg = f"Prompty-Datei nicht gefunden: {file_path.name}"
                logger.exception(error_msg)
                raise RuntimeError(error_msg) from None
            except TimeoutError:
                error_msg = f"Timeout beim Laden von {file_path.name} nach {LOAD_TIMEOUT_SECONDS}s"
                logger.warning(error_msg)
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    raise RuntimeError(error_msg) from None
            except Exception as e:
                error_msg = f"Fehler beim Laden von {file_path.name}: {e}"
                logger.warning(f"{error_msg} (Versuch {attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.exception(error_msg)
                    raise RuntimeError(error_msg) from e

                # Pause vor Wiederholung
                await asyncio.sleep(RETRY_DELAY_SECONDS)

        # Fallback für Type-Safety
        raise RuntimeError(f"Unerwarteter Fehler beim Laden von {file_path.name}")

    def get_client_agents(self) -> dict[str, Agent]:
        """Konvertiert gecachte Prompty-Objekte zu client-kompatiblen Agents.

        Returns:
            Dictionary mit Agent-IDs als Keys und Agent-Objekten als Values
        """
        if not self._cache:
            logger.warning("Keine Agents im Cache - load_agents() zuerst aufrufen")
            return {}

        return {
            agent_id: self._prompty_to_agent(agent_id, prompty_obj)
            for agent_id, prompty_obj in self._cache.items()
        }

    def _prompty_to_agent(self, agent_id: str, prompty_instance: Prompty) -> Agent:
        """Konvertiert Prompty-Objekt in Agent-Dataclass.

        Args:
            agent_id: Eindeutige Agent-ID
            prompty_instance: Prompty-Objekt

        Returns:
            Agent-Dataclass für Client-Kompatibilität
        """
        metadata = getattr(prompty_instance, "metadata", {})

        return Agent(
            id=agent_id,
            name=metadata.get("name", CustomAgentLoader._generate_display_name(agent_id)),
            type=AGENT_TYPE_PROMPTY,
            description=metadata.get("description", f"Prompty-basierter Agent: {agent_id}"),
            status="available",
            parameters=[],
        )

    def prompty_to_agent(self, agent_id: str, prompty_instance: Prompty) -> Agent:
        """Public API für Prompty-zu-Agent-Konvertierung.

        Args:
            agent_id: Eindeutige Agent-ID
            prompty_instance: Prompty-Objekt

        Returns:
            Agent-Dataclass für Client-Kompatibilität
        """
        return self._prompty_to_agent(agent_id, prompty_instance)

    @staticmethod
    def _generate_display_name(agent_id: str) -> str:
        """Generiert benutzerfreundlichen Display-Namen aus Agent-ID.

        Args:
            agent_id: Agent-ID zur Konvertierung

        Returns:
            Benutzerfreundlicher Display-Name
        """
        return agent_id.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _get_current_timestamp() -> str:
        """Gibt aktuellen Timestamp als ISO-String zurück.

        Returns:
            Aktueller Timestamp im ISO-Format
        """
        from datetime import datetime
        return datetime.now(UTC).isoformat()

    def _calculate_cache_hit_ratio(self) -> float:
        """Berechnet Cache-Hit-Ratio als Prozentsatz."""
        cache_hits = self._cache_stats.get("cache_hits", 0)
        total_loads = self._cache_stats.get("total_loads", 0)

        if not isinstance(cache_hits, int) or not isinstance(total_loads, int):
            return 0.0

        total_requests = cache_hits + total_loads
        if total_requests == 0:
            return 0.0
        return round((cache_hits / total_requests) * 100, 2)

    def clear_cache(self) -> None:
        """Leert den Agent-Cache."""
        self._cache.clear()
        logger.debug("Agent-Cache geleert")

    def validate_cache(self) -> dict[str, bool]:
        """Validiert die Integrität des Caches.

        Returns:
            Dictionary mit Validierungsergebnissen
        """
        validation_results = {
            "cache_size_valid": len(self._cache) <= MAX_CACHE_SIZE,
            "all_agents_valid": True,
            "agents_dir_exists": self._agents_dir.exists(),
        }

        # Validiere jeden Agent im Cache
        for agent_id, prompty_obj in self._cache.items():
            if not hasattr(prompty_obj, "metadata"):
                validation_results["all_agents_valid"] = False
                logger.warning(f"Agent {agent_id} hat keine Metadaten")
                break

        return validation_results

    def get_cache_stats(self) -> dict[str, int | str | float | None]:
        """Gibt erweiterte Cache-Statistiken zurück."""
        return {
            "cached_agents": len(self._cache),
            "max_cache_size": MAX_CACHE_SIZE,
            "total_loads": self._cache_stats["total_loads"],
            "successful_loads": self._cache_stats["successful_loads"],
            "failed_loads": self._cache_stats["failed_loads"],
            "cache_hits": self._cache_stats["cache_hits"],
            "last_refresh": self._cache_stats["last_refresh"],
            "cache_hit_ratio": self._calculate_cache_hit_ratio(),
        }

    @classmethod
    def reset_singleton(cls) -> None:
        """Setzt Singleton-Instanz zurück (hauptsächlich für Tests)."""
        from agents.factory.singleton_mixin import SingletonMeta
        SingletonMeta.reset_instance(cls)


# Singleton-Instanz und Legacy-API

def get_custom_agent_loader() -> CustomAgentLoader:
    """Gibt Singleton-Instanz des CustomAgentLoader zurück."""
    return CustomAgentLoader()


# Legacy API-Funktionen
async def load_custom_agents(*, refresh: bool = False) -> Mapping[str, Prompty]:
    """Lädt alle .prompty-Dateien aus dem Agent-Verzeichnis.

    Args:
        refresh: Erzwingt Neuladen auch bei vorhandenem Cache

    Returns:
        Mapping von Agent-IDs zu Prompty-Objekten
    """
    loader = get_custom_agent_loader()
    return await loader.load_agents(refresh=refresh)


def get_client_agents() -> dict[str, Agent]:
    """Gibt client-kompatible Agent-Objekte zurück.

    Returns:
        Dictionary mit Agent-IDs als Keys und Agent-Objekten als Values
    """
    loader = get_custom_agent_loader()
    return loader.get_client_agents()


def prompty_to_agent(agent_id: str, prompty_instance: Prompty) -> Agent:
    """Konvertiert Prompty-Objekt in Agent-Dataclass.

    Args:
        agent_id: Eindeutige Agent-ID
        prompty_instance: Prompty-Objekt

    Returns:
        Agent-Dataclass für Client-Kompatibilität
    """
    loader = get_custom_agent_loader()
    return loader.prompty_to_agent(agent_id, prompty_instance)
