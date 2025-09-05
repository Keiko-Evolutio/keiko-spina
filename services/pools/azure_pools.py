# backend/services/pools/azure_pools.py
"""Azure Resource Pool System."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar

from kei_logging import get_logger

logger = get_logger(__name__)

# Konstanten für Pool-Konfiguration
DEFAULT_POOL_SIZE = 10
DEFAULT_CONNECTION_TIMEOUT = 30.0
MAX_POOL_SIZE = 100

# Type Variables
T = TypeVar("T")


class PoolHealth(Enum):
    """Pool-Status-Enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class PoolConfig:
    """Konfiguration für Resource-Pools."""
    max_size: int = DEFAULT_POOL_SIZE
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT

    def __post_init__(self) -> None:
        """Validiert Pool-Konfiguration."""
        if self.max_size <= 0 or self.max_size > MAX_POOL_SIZE:
            raise ValueError(f"Pool-Größe muss zwischen 1 und {MAX_POOL_SIZE} liegen")
        if self.connection_timeout <= 0:
            raise ValueError("Connection-Timeout muss positiv sein")


class PoolMetrics:
    """Pool-Metriken für Monitoring."""

    def __init__(self, pool_name: str) -> None:
        self.pool_name = pool_name
        self.active_connections = 0
        self.total_connections_created = 0
        self.total_connections_closed = 0
        self.health_status = PoolHealth.HEALTHY

    def connection_acquired(self) -> None:
        """Registriert eine erworbene Verbindung."""
        self.active_connections += 1

    def connection_released(self) -> None:
        """Registriert eine freigegebene Verbindung."""
        self.active_connections = max(0, self.active_connections - 1)

    def connection_created(self) -> None:
        """Registriert eine neu erstellte Verbindung."""
        self.total_connections_created += 1

    def connection_closed(self) -> None:
        """Registriert eine geschlossene Verbindung."""
        self.total_connections_closed += 1


class ResourceProtocol(Protocol):
    """Protokoll für Pool-Ressourcen."""

    async def close(self) -> None:
        """Schließt die Ressource."""
        ...


class BaseResourcePool:
    """Basis-Resource-Pool für wiederverwendbare Ressourcen.

    Implementiert ein generisches Pool-Pattern mit Connection-Management,
    Health-Monitoring und automatischer Bereinigung.
    """

    def __init__(
        self,
        pool_name: str,
        config: PoolConfig | None = None
    ) -> None:
        """Initialisiert den Resource-Pool.

        Args:
            pool_name: Name des Pools für Logging und Monitoring
            config: Pool-Konfiguration (optional, verwendet Defaults)
        """
        self.pool_name = pool_name
        self.config = config or PoolConfig()
        self._pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.config.max_size)
        self._in_use: set[Any] = set()
        self.metrics = PoolMetrics(pool_name)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialisiert den Pool.

        Raises:
            RuntimeError: Wenn Pool bereits initialisiert ist
        """
        if self._initialized:
            logger.warning(f"Pool {self.pool_name} bereits initialisiert")
            return

        self._initialized = True
        logger.info(f"Pool {self.pool_name} initialisiert (max_size={self.config.max_size})")

    async def _acquire_connection(self) -> Any:
        """Erwirbt eine Verbindung aus dem Pool oder erstellt eine neue.

        Returns:
            Verbindung aus dem Pool oder neu erstellte Verbindung

        Raises:
            RuntimeError: Wenn Pool nicht initialisiert ist
        """
        if not self._initialized:
            raise RuntimeError(f"Pool {self.pool_name} nicht initialisiert")

        try:
            # Versuche bestehende Verbindung aus Pool zu holen
            connection = self._pool.get_nowait()
            logger.debug(f"Verbindung aus Pool {self.pool_name} wiederverwendet")
            return connection
        except asyncio.QueueEmpty:
            # Erstelle neue Verbindung wenn Pool leer
            connection = await self._create_connection()
            self.metrics.connection_created()
            logger.debug(f"Neue Verbindung für Pool {self.pool_name} erstellt")
            return connection

    async def _release_connection(self, connection: Any) -> None:
        """Gibt eine Verbindung zurück an den Pool oder schließt sie.

        Args:
            connection: Freizugebende Verbindung
        """
        if connection is None:
            return

        self._in_use.discard(connection)

        try:
            # Versuche Verbindung zurück in Pool zu legen
            self._pool.put_nowait(connection)
            logger.debug(f"Verbindung an Pool {self.pool_name} zurückgegeben")
        except asyncio.QueueFull:
            # Pool voll - Verbindung schließen
            await self._close_connection(connection)
            self.metrics.connection_closed()
            logger.debug(f"Verbindung geschlossen (Pool {self.pool_name} voll)")

        self.metrics.connection_released()

    def _handle_connection_error(self, connection: Any, error: Exception) -> None:
        """Behandelt Verbindungsfehler und aktualisiert Pool-Status.

        Args:
            connection: Fehlerhafte Verbindung
            error: Aufgetretener Fehler
        """
        logger.warning(f"Verbindungsfehler in Pool {self.pool_name}: {error}")
        self.metrics.health_status = PoolHealth.DEGRADED

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[Any]:
        """Context Manager für Pool-Verbindungen.

        Yields:
            Verbindung aus dem Pool

        Raises:
            RuntimeError: Wenn Pool nicht initialisiert ist
            Exception: Bei Verbindungsfehlern
        """
        connection = None
        try:
            connection = await self._acquire_connection()
            self._in_use.add(connection)
            self.metrics.connection_acquired()

            yield connection

        except Exception as e:
            self._handle_connection_error(connection, e)
            if connection:
                await self._close_connection(connection)
                self.metrics.connection_closed()
                connection = None
            raise
        finally:
            if connection:
                await self._release_connection(connection)

    async def _create_connection(self) -> Any:
        """Erstellt eine neue Verbindung.

        Diese Methode muss von Subklassen implementiert werden.

        Returns:
            Neue Verbindung/Ressource

        Raises:
            NotImplementedError: Wenn nicht von Subklasse implementiert
        """
        raise NotImplementedError(
            f"_create_connection() muss von {self.__class__.__name__} implementiert werden"
        )

    async def _close_connection(self, connection: Any) -> None:
        """Schließt eine Verbindung.

        Standard-Implementierung versucht verschiedene Close-Methoden.
        Kann von Subklassen überschrieben werden.

        Args:
            connection: Zu schließende Verbindung
        """
        if connection is None:
            return

        try:
            # Versuche verschiedene Close-Methoden
            if hasattr(connection, "aclose"):
                await connection.aclose()
            elif hasattr(connection, "close"):
                close_method = connection.close
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
        except Exception as e:
            logger.debug(f"Fehler beim Schließen der Verbindung: {e}")

    async def _cleanup_in_use_connections(self) -> None:
        """Schließt alle aktiven Verbindungen."""
        connections_to_close = list(self._in_use)
        for conn in connections_to_close:
            try:
                await self._close_connection(conn)
                self.metrics.connection_closed()
            except Exception as e:
                logger.debug(f"Fehler beim Schließen aktiver Verbindung: {e}")

        self._in_use.clear()
        logger.debug(f"Pool {self.pool_name}: {len(connections_to_close)} aktive Verbindungen geschlossen")

    async def _cleanup_pooled_connections(self) -> None:
        """Schließt alle Verbindungen im Pool."""
        closed_count = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await self._close_connection(conn)
                self.metrics.connection_closed()
                closed_count += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.debug(f"Fehler beim Schließen Pool-Verbindung: {e}")

        logger.debug(f"Pool {self.pool_name}: {closed_count} Pool-Verbindungen geschlossen")

    async def cleanup(self) -> None:
        """Bereinigt den Pool und schließt alle Verbindungen.

        Schließt sowohl aktive als auch Pool-Verbindungen und
        setzt den Pool-Status zurück.
        """
        if not self._initialized:
            logger.debug(f"Pool {self.pool_name} bereits bereinigt")
            return

        logger.info(f"Bereinige Pool {self.pool_name}")

        # Schließe alle Verbindungen
        await self._cleanup_in_use_connections()
        await self._cleanup_pooled_connections()

        # Reset Pool-Status
        self._initialized = False
        self.metrics.health_status = PoolHealth.HEALTHY

        logger.info(f"Pool {self.pool_name} bereinigt")


class HTTPClientPool(BaseResourcePool):
    """HTTP Client Pool mit Integration zu services.clients.HTTPClient.

    Verwaltet wiederverwendbare HTTPClient-Instanzen für optimierte
    HTTP-Requests mit Session-Pooling und Konfiguration.
    """

    def __init__(
        self,
        pool_name: str,
        config: PoolConfig | None = None,
        http_timeout: float = DEFAULT_CONNECTION_TIMEOUT
    ) -> None:
        """Initialisiert HTTP Client Pool.

        Args:
            pool_name: Name des Pools
            config: Pool-Konfiguration (optional)
            http_timeout: HTTP-Timeout für Clients
        """
        super().__init__(pool_name, config)
        self.http_timeout = http_timeout

    async def _create_connection(self) -> Any:
        """Erstellt neue HTTPClient-Instanz.

        Verwendet services.clients.HTTPClient für konsistente
        HTTP-Client-Konfiguration mit Session-Management.

        Returns:
            HTTPClient-Instanz

        Raises:
            ImportError: Wenn services.clients nicht verfügbar
            RuntimeError: Wenn HTTPClient-Erstellung fehlschlägt
        """
        try:
            # Import services.clients.HTTPClient
            from services.clients.clients import HTTPClient

            # Erstelle HTTPClient mit konfiguriertem Timeout
            http_client = HTTPClient(timeout=self.http_timeout)

            logger.debug(f"HTTPClient für Pool {self.pool_name} erstellt (timeout={self.http_timeout})")
            return http_client

        except ImportError as e:
            logger.exception(f"services.clients.HTTPClient nicht verfügbar: {e}")
            raise RuntimeError(
                "services.clients.HTTPClient erforderlich für HTTPClientPool"
            ) from e
        except Exception as e:
            logger.exception(f"HTTPClient-Erstellung fehlgeschlagen: {e}")
            raise RuntimeError(f"HTTPClient-Erstellung fehlgeschlagen: {e}") from e

    async def _close_connection(self, connection: Any) -> None:
        """Schließt HTTPClient-Verbindung.

        HTTPClient-Instanzen haben keine explizite close()-Methode,
        da sie Session-Manager verwenden. Cleanup erfolgt automatisch.

        Args:
            connection: HTTPClient-Instanz
        """
        if connection is None:
            return

        try:
            # HTTPClient verwendet managed_session - kein explizites Close nötig
            # Session-Cleanup erfolgt automatisch durch Context Manager
            logger.debug(f"HTTPClient für Pool {self.pool_name} freigegeben")

        except Exception as e:
            logger.debug(f"HTTPClient-Cleanup-Fehler: {e}")

    @asynccontextmanager
    async def get_http_client(self) -> AsyncIterator[Any]:
        """Convenience-Methode für HTTP-Client-Zugriff.

        Yields:
            HTTPClient-Instanz aus dem Pool
        """
        async with self.get_connection() as client:
            yield client


@dataclass
class PoolManagerConfig:
    """Konfiguration für Pool Manager."""
    http_pool_size: int = DEFAULT_POOL_SIZE
    http_timeout: float = DEFAULT_CONNECTION_TIMEOUT

    def __post_init__(self) -> None:
        """Validiert Pool Manager-Konfiguration."""
        if self.http_pool_size <= 0:
            raise ValueError("HTTP Pool-Größe muss positiv sein")
        if self.http_timeout <= 0:
            raise ValueError("HTTP Timeout muss positiv sein")


class PoolManager:
    """Zentraler Pool Manager für alle Resource-Pools.

    Verwaltet verschiedene Pool-Typen und bietet einheitliche
    Initialisierung, Cleanup und Health-Monitoring.
    """

    def __init__(self, config: PoolManagerConfig | None = None) -> None:
        """Initialisiert Pool Manager.

        Args:
            config: Pool Manager-Konfiguration (optional)
        """
        self.config = config or PoolManagerConfig()
        self.pools: dict[str, BaseResourcePool] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialisiert alle Standard-Pools.

        Erstellt und initialisiert HTTP Client Pool mit
        konfigurierter Größe und Timeout.

        Raises:
            RuntimeError: Wenn Pool-Initialisierung fehlschlägt
        """
        if self._initialized:
            logger.warning("Pool Manager bereits initialisiert")
            return

        try:
            # HTTP Client Pool erstellen
            http_pool_config = PoolConfig(
                max_size=self.config.http_pool_size,
                connection_timeout=self.config.http_timeout
            )

            http_pool = HTTPClientPool(
                "http_client",
                config=http_pool_config,
                http_timeout=self.config.http_timeout
            )

            await http_pool.initialize()
            self.pools["http_client"] = http_pool

            self._initialized = True
            logger.info(
                f"Pool Manager initialisiert mit {len(self.pools)} Pools "
                f"(HTTP: {self.config.http_pool_size} Connections)"
            )

        except Exception as e:
            logger.exception(f"Pool Manager-Initialisierung fehlgeschlagen: {e}")
            # Cleanup bei Fehler
            await self._cleanup_partial_initialization()
            raise RuntimeError(f"Pool Manager-Initialisierung fehlgeschlagen: {e}") from e

    async def _cleanup_partial_initialization(self) -> None:
        """Bereinigt teilweise initialisierte Pools bei Fehlern."""
        for pool in list(self.pools.values()):
            try:
                await pool.cleanup()
            except Exception as e:
                logger.debug(f"Fehler beim Cleanup von Pool {pool.pool_name}: {e}")

        self.pools.clear()
        self._initialized = False

    async def cleanup(self) -> None:
        """Bereinigt alle Pools und setzt Manager zurück.

        Schließt alle Pool-Verbindungen und setzt den
        Manager-Status zurück.
        """
        if not self._initialized:
            logger.debug("Pool Manager bereits bereinigt")
            return

        logger.info("Bereinige Pool Manager")

        cleanup_errors = []
        for pool_name, pool in self.pools.items():
            try:
                await pool.cleanup()
                logger.debug(f"Pool {pool_name} bereinigt")
            except Exception as e:
                cleanup_errors.append(f"{pool_name}: {e}")
                logger.exception(f"Fehler beim Bereinigen von Pool {pool_name}: {e}")

        self.pools.clear()
        self._initialized = False

        if cleanup_errors:
            logger.warning(f"Pool-Cleanup-Fehler: {cleanup_errors}")
        else:
            logger.info("Pool Manager bereinigt")

    def get_pool(self, pool_name: str) -> BaseResourcePool | None:
        """Ruft Pool nach Name ab.

        Args:
            pool_name: Name des gewünschten Pools

        Returns:
            Pool-Instanz oder None wenn nicht gefunden
        """
        if not self._initialized:
            logger.warning(f"Pool Manager nicht initialisiert - Pool {pool_name} nicht verfügbar")
            return None

        return self.pools.get(pool_name)

    async def get_health_status(self) -> dict[str, Any]:
        """Ruft Health-Status aller Pools ab.

        Returns:
            Dictionary mit Pool-Health-Informationen
        """
        pool_details = {}
        for name, pool in self.pools.items():
            pool_details[name] = {
                "health_status": pool.metrics.health_status.value,
                "active_connections": pool.metrics.active_connections,
                "total_created": pool.metrics.total_connections_created,
                "total_closed": pool.metrics.total_connections_closed,
                "max_size": pool.config.max_size,
                "pool_size": pool._pool.qsize() if hasattr(pool, "_pool") else 0
            }

        return {
            "initialized": self._initialized,
            "pool_count": len(self.pools),
            "pools": pool_details
        }


# Globaler Pool Manager für einfache API
_pool_manager: PoolManager | None = None


def _get_pool_manager() -> PoolManager:
    """Ruft globalen Pool Manager ab oder erstellt ihn.

    Returns:
        Globaler PoolManager
    """
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = PoolManager()
    return _pool_manager


# Public API für einfache Pool-Verwaltung
async def initialize_pools(config: PoolManagerConfig | None = None) -> None:
    """Initialisiert das Pool-System.

    Args:
        config: Optional Pool Manager-Konfiguration

    Raises:
        RuntimeError: Wenn Initialisierung fehlschlägt
    """
    global _pool_manager
    if config is not None:
        _pool_manager = PoolManager(config)

    manager = _get_pool_manager()
    await manager.initialize()


async def cleanup_pools() -> None:
    """Bereinigt das Pool-System.

    Schließt alle Pools und setzt den globalen Manager zurück.
    """
    global _pool_manager
    if _pool_manager is not None:
        await _pool_manager.cleanup()
        _pool_manager = None


def get_http_client() -> HTTPClientPool | None:
    """Ruft HTTP Client Pool ab.

    Returns:
        HTTPClientPool-Instanz oder None wenn nicht verfügbar
    """
    manager = _get_pool_manager()
    pool = manager.get_pool("http_client")
    return pool if isinstance(pool, HTTPClientPool) else None


async def get_health_status() -> dict[str, Any]:
    """Ruft Pool-Health-Status ab.

    Returns:
        Dictionary mit Health-Informationen aller Pools
    """
    manager = _get_pool_manager()
    return await manager.get_health_status()


__all__ = [
    "DEFAULT_CONNECTION_TIMEOUT",
    "DEFAULT_POOL_SIZE",
    "MAX_POOL_SIZE",
    "BaseResourcePool",
    "HTTPClientPool",
    "PoolConfig",
    "PoolHealth",
    "PoolManager",
    "PoolManagerConfig",
    "PoolMetrics",
    "cleanup_pools",
    "get_health_status",
    "get_http_client",
    "initialize_pools",
]
