# backend/services/core/manager.py
"""Service Manager für zentrale Service-Verwaltung.

Verwaltet Lifecycle und Health-Status aller verfügbaren Services.
"""

import inspect
from dataclasses import dataclass
from typing import Any

from kei_logging import get_logger

from .features import features

# Service-Imports
try:
    from services.clients.clients import HTTPClient, Services
    HTTP_CLIENT_AVAILABLE = True
    AZURE_SERVICES_AVAILABLE = True
except ImportError:
    HTTPClient = None
    Services = None
    HTTP_CLIENT_AVAILABLE = False
    AZURE_SERVICES_AVAILABLE = False

try:
    from services.clients.deep_research import create_deep_research_service
    DEEP_RESEARCH_AVAILABLE = True
except ImportError:
    create_deep_research_service = None
    DEEP_RESEARCH_AVAILABLE = False

try:
    from services.pools.azure_pools import cleanup_pools, initialize_pools
    AZURE_POOLS_AVAILABLE = True
except ImportError:
    initialize_pools = None
    cleanup_pools = None
    AZURE_POOLS_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class ServiceSetupResult:
    """Ergebnis eines Service-Setup-Vorgangs."""

    name: str
    success: bool
    service: Any | None = None
    error: str | None = None


@dataclass
class CleanupTask:
    """Cleanup-Task für Service-Bereinigung."""

    task_type: str
    service_name: str
    coroutine: Any


class ServiceManager:
    """Zentraler Service Manager.

    Verwaltet den Lifecycle aller verfügbaren Services und bietet
    Health-Monitoring und Cleanup-Funktionalität.

    Implementiert ServiceManagerService Interface für DI-Container-Kompatibilität.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialisiert verfügbare Services."""
        if self._initialized:
            logger.debug("Service Manager bereits initialisiert")
            return

        setup_results = await self._setup_available_services()
        self._initialized = True

        successful_services = [r.name for r in setup_results if r.success]
        failed_services = [r.name for r in setup_results if not r.success]

        logger.info(f"Service Manager initialisiert: {len(successful_services)} erfolgreich")
        if failed_services:
            logger.warning(f"Services fehlgeschlagen: {', '.join(failed_services)}")

    async def _setup_available_services(self) -> list[ServiceSetupResult]:
        """Setup für alle verfügbaren Services.

        Returns:
            Liste der Setup-Ergebnisse für alle Services
        """
        setup_tasks = []

        if features.is_available("http_clients"):
            setup_tasks.append(("http_client", self._setup_http_client))

        if features.is_available("azure_core"):
            setup_tasks.append(("azure_services", self._setup_azure_services))

        if features.is_available("pools"):
            setup_tasks.append(("pools", self._setup_pools))

        results = []
        for _service_name, setup_func in setup_tasks:
            result = await setup_func()
            results.append(result)

        return results

    async def _setup_http_client(self) -> ServiceSetupResult:
        """HTTP-Client Setup."""
        try:
            if not HTTP_CLIENT_AVAILABLE:
                raise ImportError("HTTP client not available")
            client = HTTPClient()
            self._services["http_client"] = client
            return ServiceSetupResult("http_client", True, client)
        except Exception as e:
            error_msg = f"HTTP-Client Setup fehlgeschlagen: {e}"
            logger.warning(error_msg)
            return ServiceSetupResult("http_client", False, error=error_msg)

    async def _setup_azure_services(self) -> ServiceSetupResult:
        """Azure Services Setup."""
        try:
            services = await self._create_azure_services()
            await self._setup_deep_research_service()
            return ServiceSetupResult("azure_services", True, services)
        except Exception as e:
            error_msg = f"Azure Services Setup fehlgeschlagen: {e}"
            logger.warning(error_msg)
            return ServiceSetupResult("azure_services", False, error=error_msg)

    async def _create_azure_services(self) -> Any:
        """Erstellt Azure Services Instanz."""
        if not AZURE_SERVICES_AVAILABLE:
            raise ImportError("Azure services not available")
        services = Services()
        self._services["azure_services"] = services
        return services

    async def _setup_deep_research_service(self) -> None:
        """Setup für Deep Research Service (optional)."""
        try:
            if not DEEP_RESEARCH_AVAILABLE:
                raise ImportError("Deep research service not available")
            dr_service = create_deep_research_service()
            if dr_service is not None:
                self._services["deep_research"] = dr_service
                logger.debug("Deep Research Service erfolgreich registriert")
        except Exception as e:
            logger.debug(f"Deep Research Service Setup übersprungen: {e}")

    async def _setup_pools(self) -> ServiceSetupResult:
        """Pool-System Setup."""
        try:
            if not AZURE_POOLS_AVAILABLE:
                raise ImportError("Azure pools not available")
            await initialize_pools()
            self._services["pools"] = True
            return ServiceSetupResult("pools", True, True)
        except Exception as e:
            error_msg = f"Pool Setup fehlgeschlagen: {e}"
            logger.warning(error_msg)
            return ServiceSetupResult("pools", False, error=error_msg)

    async def cleanup(self) -> None:
        """Bereinigt alle Services."""
        if not self._initialized:
            logger.debug("Service Manager nicht initialisiert - Cleanup übersprungen")
            return

        cleanup_tasks = self._collect_cleanup_tasks()
        await self._execute_cleanup_tasks(cleanup_tasks)

        self._services.clear()
        self._initialized = False
        logger.info("Service Manager Cleanup abgeschlossen")

    def _collect_cleanup_tasks(self) -> list[CleanupTask]:
        """Sammelt alle Cleanup-Tasks für verfügbare Services."""
        cleanup_tasks = []

        for name, service in self._services.items():
            if service is None:
                continue

            task = self._create_service_cleanup_task(name, service)
            if task:
                cleanup_tasks.append(task)

        # Spezielle Pool-Bereinigung hinzufügen
        pool_task = self._create_pool_cleanup_task()
        if pool_task:
            cleanup_tasks.append(pool_task)

        return cleanup_tasks

    def _create_service_cleanup_task(self, name: str, service: Any) -> CleanupTask | None:
        """Erstellt Cleanup-Task für einen Service."""
        # Prüfe auf close-Methode
        if hasattr(service, "close") and callable(service.close):
            return self._create_close_task(name, service)

        # Fallback: cleanup-Methode
        if hasattr(service, "cleanup") and callable(service.cleanup):
            return CleanupTask("cleanup", name, service.cleanup())

        return None

    def _create_close_task(self, name: str, service: Any) -> CleanupTask | None:
        """Erstellt Close-Task für einen Service."""
        try:
            close_method = service.close
            sig = inspect.signature(close_method)

            if len(sig.parameters) == 0:
                return CleanupTask("close", name, close_method())
        except Exception:
            # Fallback: Versuche cleanup stattdessen
            if hasattr(service, "cleanup") and callable(service.cleanup):
                return CleanupTask("cleanup", name, service.cleanup())

        return None

    def _create_pool_cleanup_task(self) -> CleanupTask | None:
        """Erstellt Cleanup-Task für Pool-System."""
        if "pools" not in self._services:
            return None

        try:
            if not AZURE_POOLS_AVAILABLE:
                raise ImportError("Azure pools not available")
            return CleanupTask("pool_cleanup", "pools", cleanup_pools())
        except ImportError:
            return None

    async def _execute_cleanup_tasks(self, cleanup_tasks: list[CleanupTask]) -> None:
        """Führt alle Cleanup-Tasks aus."""
        for task in cleanup_tasks:
            try:
                if task.coroutine is not None:
                    await task.coroutine
                    logger.debug(f"{task.task_type} für {task.service_name} erfolgreich")
            except Exception as e:
                logger.warning(f"{task.task_type} für {task.service_name} fehlgeschlagen: {e}")

    async def get_health_status(self) -> dict[str, Any]:
        """Service-Health-Status."""
        status = {
            "initialized": self._initialized,
            "service_count": len(self._services),
            "features": features.all_features,
            "services": {}
        }

        for name, service in self._services.items():
            try:
                if hasattr(service, "is_healthy"):
                    status["services"][name] = await service.is_healthy()
                elif hasattr(service, "health_status"):
                    status["services"][name] = await service.health_status()
                # AI Project Client schlicht vorhanden?
                elif name == "ai_project_client":
                    status["services"][name] = "available" if service is not None else "unavailable"
                else:
                    status["services"][name] = "available"
            except Exception:
                status["services"][name] = "error"

        return status

    def get_service(self, name: str) -> Any | None:
        """Service abrufen."""
        return self._services.get(name)

    def is_healthy(self) -> bool:
        """Prüft grundlegende Service-Verfügbarkeit."""
        return (self._initialized and
                len(self._services) > 0 and
                features.is_available("http_clients"))

    # ServiceManagerService Interface-Implementierung
    async def register_service(self, service_id: str, service: Any) -> None:
        """Registriert einen Service beim Manager."""
        self._services[service_id] = service
        logger.debug(f"Service registriert: {service_id}")

    async def unregister_service(self, service_id: str) -> None:
        """Entfernt einen Service vom Manager."""
        if service_id in self._services:
            del self._services[service_id]
            logger.debug(f"Service deregistriert: {service_id}")
        else:
            raise RuntimeError(f"Service nicht gefunden: {service_id}")



    async def get_service_list(self) -> list[str]:
        """Liefert Liste aller registrierten Services."""
        return list(self._services.keys())

    async def get_service_status(self, service_id: str) -> dict[str, Any]:
        """Liefert Status eines spezifischen Services."""
        if service_id not in self._services:
            raise RuntimeError(f"Service nicht gefunden: {service_id}")

        service = self._services[service_id]
        return {
            "service_id": service_id,
            "status": "active",
            "type": type(service).__name__,
            "available": True
        }


# Globaler Service Manager
service_manager = ServiceManager()
