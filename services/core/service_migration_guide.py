"""Service Migration Guide für neue Basis-Klassen.

Dokumentiert die Migration bestehender Services zu den neuen Basis-Klassen
und bietet Adapter für Backward-Compatibility.
"""

from __future__ import annotations

from typing import Any

from kei_logging import get_logger
from services.interfaces._base import CoreService, InfrastructureService, LifecycleService

from .base_service import BaseService, MonitoringService, PeriodicService

logger = get_logger(__name__)


class ServiceMigrationAdapter:
    """Adapter für Migration bestehender Services zu neuen Basis-Klassen.

    Bietet Backward-Compatibility und schrittweise Migration.
    """

    @staticmethod
    def create_base_service_adapter(
        legacy_service: Any,
        service_name: str
    ) -> LifecycleService:
        """Erstellt BaseService-Adapter für Legacy-Service.

        Args:
            legacy_service: Bestehender Service
            service_name: Name des Services

        Returns:
            BaseService-Adapter
        """

        class LegacyServiceAdapter(LifecycleService):
            """Adapter für Legacy-Service zu LifecycleService.

            BEISPIEL-CODE: Zeigt Migration von Legacy-Services.
            """

            def __init__(self):
                self.service_name = service_name
                self.legacy_service = legacy_service

            async def initialize(self) -> None:
                """Initialisiert Legacy-Service."""
                if hasattr(self.legacy_service, "start"):
                    await self.legacy_service.start()
                elif hasattr(self.legacy_service, "initialize"):
                    await self.legacy_service.initialize()

            async def shutdown(self) -> None:
                """Stoppt Legacy-Service."""
                if hasattr(self.legacy_service, "stop"):
                    await self.legacy_service.stop()
                elif hasattr(self.legacy_service, "shutdown"):
                    await self.legacy_service.shutdown()

            async def health_check(self) -> dict[str, Any]:
                """Führt Health-Check für Legacy-Service durch."""
                if hasattr(self.legacy_service, "health_check"):
                    result = await self.legacy_service.health_check()
                    if isinstance(result, dict):
                        return result
                    return {"status": "healthy" if result else "unhealthy"}
                if hasattr(self.legacy_service, "get_status"):
                    status = self.legacy_service.get_status()
                    return {"status": "healthy" if status in ["healthy", "running", "available"] else "unhealthy"}
                return {"status": "healthy" if hasattr(self.legacy_service, "running") and self.legacy_service.running else "unknown"}

        # BEISPIEL: Instanziierung des Adapters
        # In der Praxis sollte dies über Dependency Injection erfolgen
        return LegacyServiceAdapter()

    @staticmethod
    def create_periodic_service_adapter(
        legacy_service: Any,
        service_name: str,
        interval_seconds: float = 30.0
    ) -> CoreService:
        """Erstellt PeriodicService-Adapter für Legacy-Service.

        Args:
            legacy_service: Bestehender Service mit periodischen Tasks
            service_name: Name des Services
            interval_seconds: Intervall zwischen Tasks

        Returns:
            PeriodicService-Adapter
        """

        class LegacyPeriodicServiceAdapter(CoreService):
            """Adapter für Legacy-Service zu CoreService mit periodischen Tasks.

            BEISPIEL-CODE: Zeigt Migration von Legacy-Services mit periodischen Tasks.
            """

            def __init__(self):
                self.service_name = service_name
                self.interval_seconds = interval_seconds
                self.legacy_service = legacy_service

            async def initialize(self) -> None:
                """Initialisiert Legacy-Service."""
                if hasattr(self.legacy_service, "start"):
                    await self.legacy_service.start()
                elif hasattr(self.legacy_service, "initialize"):
                    await self.legacy_service.initialize()

            async def shutdown(self) -> None:
                """Stoppt Legacy-Service."""
                if hasattr(self.legacy_service, "stop"):
                    await self.legacy_service.stop()
                elif hasattr(self.legacy_service, "shutdown"):
                    await self.legacy_service.shutdown()

            async def perform_periodic_task(self) -> None:
                """Führt periodische Task für Legacy-Service durch."""
                if hasattr(self.legacy_service, "_perform_periodic_task"):
                    await self.legacy_service._perform_periodic_task()
                elif hasattr(self.legacy_service, "_check_all_agents"):
                    # Spezifisch für AgentHeartbeatService
                    await self.legacy_service._check_all_agents()
                elif hasattr(self.legacy_service, "_revalidation_loop"):
                    # Spezifisch für DomainRevalidationService
                    await self.legacy_service._perform_revalidation()

            async def _perform_health_check(self) -> bool:
                """Führt Health-Check für Legacy-Service durch."""
                if hasattr(self.legacy_service, "health_check"):
                    result = await self.legacy_service.health_check()
                    if isinstance(result, dict):
                        return result.get("status") == "healthy"
                    return bool(result)
                return hasattr(self.legacy_service, "running") and self.legacy_service.running

        # BEISPIEL: Instanziierung des Adapters
        # In der Praxis sollte dies über Dependency Injection erfolgen
        return LegacyPeriodicServiceAdapter()

    @staticmethod
    def create_monitoring_service_adapter(
        legacy_service: Any,
        service_name: str,
        interval_seconds: float = 30.0,
        max_failures: int = 3
    ) -> InfrastructureService:
        """Erstellt MonitoringService-Adapter für Legacy-Service.

        Args:
            legacy_service: Bestehender Service mit Monitoring
            service_name: Name des Services
            interval_seconds: Intervall zwischen Checks
            max_failures: Maximale Fehler vor Eskalation

        Returns:
            MonitoringService-Adapter
        """

        class LegacyMonitoringServiceAdapter(InfrastructureService):
            """Adapter für Legacy-Service zu InfrastructureService mit Monitoring.

            BEISPIEL-CODE: Zeigt Migration von Legacy-Services mit Monitoring-Funktionalität.
            """

            def __init__(self):
                self.service_name = service_name
                self.interval_seconds = interval_seconds
                self.max_failures = max_failures
                self.legacy_service = legacy_service

            async def initialize(self) -> None:
                """Initialisiert Legacy-Service."""
                if hasattr(self.legacy_service, "start"):
                    await self.legacy_service.start()
                elif hasattr(self.legacy_service, "initialize"):
                    await self.legacy_service.initialize()

            async def shutdown(self) -> None:
                """Stoppt Legacy-Service."""
                if hasattr(self.legacy_service, "stop"):
                    await self.legacy_service.stop()
                elif hasattr(self.legacy_service, "shutdown"):
                    await self.legacy_service.shutdown()

            async def perform_monitoring_task(self) -> None:
                """Führt Monitoring-Task für Legacy-Service durch."""
                if hasattr(self.legacy_service, "_check_all_agents"):
                    # Spezifisch für AgentHeartbeatService
                    await self.legacy_service._check_all_agents()
                elif hasattr(self.legacy_service, "_perform_monitoring_task"):
                    await self.legacy_service._perform_monitoring_task()

            async def escalate_failure(self, target: str) -> None:
                """Eskaliert Fehler für Legacy-Service."""
                if hasattr(self.legacy_service, "_escalate_failure"):
                    await self.legacy_service._escalate_failure(target)
                elif hasattr(self.legacy_service, "_remove_agent"):
                    # Spezifisch für AgentHeartbeatService
                    await self.legacy_service._remove_agent(target)

            async def _perform_health_check(self) -> bool:
                """Führt Health-Check für Legacy-Service durch."""
                if hasattr(self.legacy_service, "health_check"):
                    result = await self.legacy_service.health_check()
                    if isinstance(result, dict):
                        return result.get("status") == "healthy"
                    return bool(result)
                return hasattr(self.legacy_service, "running") and self.legacy_service.running

        # BEISPIEL: Instanziierung des Adapters
        # In der Praxis sollte dies über Dependency Injection erfolgen
        return LegacyMonitoringServiceAdapter()


class ServiceMigrationRegistry:
    """Registry für Service-Migrationen.

    Verwaltet die schrittweise Migration bestehender Services.
    """

    def __init__(self):
        self._migrated_services: dict[str, BaseService] = {}
        self._migration_status: dict[str, str] = {}

    def register_migration(
        self,
        service_name: str,
        legacy_service: Any,
        migration_type: str = "base"
    ) -> BaseService:
        """Registriert Service-Migration.

        Args:
            service_name: Name des Services
            legacy_service: Legacy-Service-Instanz
            migration_type: Typ der Migration (base, periodic, monitoring)

        Returns:
            Migrierter Service
        """
        if migration_type == "periodic":
            migrated_service = ServiceMigrationAdapter.create_periodic_service_adapter(
                legacy_service, service_name
            )
        elif migration_type == "monitoring":
            migrated_service = ServiceMigrationAdapter.create_monitoring_service_adapter(
                legacy_service, service_name
            )
        else:
            migrated_service = ServiceMigrationAdapter.create_base_service_adapter(
                legacy_service, service_name
            )

        self._migrated_services[service_name] = migrated_service
        self._migration_status[service_name] = f"migrated_to_{migration_type}"

        logger.info(f"Service {service_name} zu {migration_type} migriert")
        return migrated_service

    def get_migrated_service(self, service_name: str) -> BaseService | None:
        """Gibt migrierten Service zurück."""
        return self._migrated_services.get(service_name)

    def get_migration_status(self) -> dict[str, str]:
        """Gibt Migration-Status aller Services zurück."""
        return self._migration_status.copy()

    async def start_all_migrated_services(self) -> None:
        """Startet alle migrierten Services."""
        for service_name, service in self._migrated_services.items():
            try:
                await service.start()
                logger.info(f"Migrierter Service {service_name} gestartet")
            except Exception as e:
                logger.exception(f"Fehler beim Starten von {service_name}: {e}")

    async def stop_all_migrated_services(self) -> None:
        """Stoppt alle migrierten Services."""
        for service_name, service in self._migrated_services.items():
            try:
                await service.stop()
                logger.info(f"Migrierter Service {service_name} gestoppt")
            except Exception as e:
                logger.exception(f"Fehler beim Stoppen von {service_name}: {e}")


# Globale Migration-Registry
migration_registry = ServiceMigrationRegistry()


# Convenience-Funktionen für häufige Migrationen
def migrate_agent_heartbeat_service(heartbeat_service: Any) -> MonitoringService:
    """Migriert AgentHeartbeatService zu MonitoringService."""
    return migration_registry.register_migration(
        "agent_heartbeat",
        heartbeat_service,
        "monitoring"
    )


def migrate_domain_revalidation_service(revalidation_service: Any) -> PeriodicService:
    """Migriert DomainRevalidationService zu PeriodicService."""
    return migration_registry.register_migration(
        "domain_revalidation",
        revalidation_service,
        "periodic"
    )


def migrate_rate_limiter_service(rate_limiter_service: Any) -> BaseService:
    """Migriert RateLimiterService zu BaseService."""
    return migration_registry.register_migration(
        "rate_limiter",
        rate_limiter_service,
        "base"
    )
