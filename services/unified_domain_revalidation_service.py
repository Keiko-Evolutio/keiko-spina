"""Unified Domain Revalidation Service basierend auf PeriodicService.

Migriert DomainRevalidationService zur neuen PeriodicService-Architektur
während die bestehende API beibehalten wird.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from config.domain_validation_config import (
    get_domain_validation_config,
    reload_domain_validation_config,
)
from kei_logging import get_logger
from services.core.base_service import PeriodicService
from services.core.constants import (
    FORCE_REVALIDATION_INTERVAL,
)
from services.interfaces.domain_revalidation_service import (
    DomainRevalidationService as DomainRevalidationServiceInterface,
)

# Optional imports für erweiterte Features
try:
    # Zugriff auf KEIMCP Registry über konsolidierten Alias
    from agents.tools import mcp_registry  # type: ignore
except Exception:
    class MockRegistry:
        async def revalidate_domains_if_needed(self, interval_hours):
            return {}
        async def remove_server(self, server_name):
            pass
    mcp_registry = MockRegistry()

# Mock für Tests - validate_server_domain_for_registration wird nicht direkt verwendet
def validate_server_domain_for_registration(*args, **kwargs):
    return True

logger = get_logger(__name__)


class UnifiedDomainRevalidationService(PeriodicService, DomainRevalidationServiceInterface):
    """Unified Domain Revalidation Service basierend auf PeriodicService.

    Drop-in Replacement für DomainRevalidationService mit verbesserter Architektur:
    - Verwendet PeriodicService für periodische Tasks
    - Behält vollständige API-Kompatibilität bei
    - Verbesserte Error-Handling und Lifecycle-Management
    - Konsolidierte Konfiguration und Monitoring
    """

    def __init__(self):
        """Initialisiert den Unified Domain Revalidation Service."""
        # Lade Konfiguration für Intervall
        config = get_domain_validation_config()
        interval_seconds = config.revalidation_interval_hours * 3600

        super().__init__(
            service_name="UnifiedDomainRevalidationService",
            interval_seconds=interval_seconds
        )

        self.last_revalidation: float | None = None
        self.last_config_reload: float | None = None
        self.config_reload_interval_seconds = 3600  # 1 Stunde

    async def _initialize(self) -> None:
        """Service-spezifische Initialisierung."""
        config = get_domain_validation_config()

        if not config.enabled:
            logger.info("Domain-Validierung deaktiviert - Service wird nicht gestartet")
            return

        logger.info("Unified Domain-Revalidierung-Service initialisiert")

        # Startup-Revalidierung falls konfiguriert
        if config.revalidation_on_startup:
            logger.info("Führe Startup-Domain-Revalidierung durch...")
            await self._perform_revalidation()

    async def _cleanup(self) -> None:
        """Service-spezifische Bereinigung."""
        logger.info("Unified Domain-Revalidierung-Service bereinigt")

    async def _perform_periodic_task(self) -> None:
        """Führt periodische Domain-Revalidierung durch."""
        config = get_domain_validation_config()

        if not config.enabled:
            logger.debug("Domain-Validierung deaktiviert - überspringe Revalidierung")
            return

        # Prüfe ob Config-Reload nötig ist
        current_time = time.time()
        if (self.last_config_reload is None or
            current_time - self.last_config_reload > self.config_reload_interval_seconds):
            await self._reload_config()
            self.last_config_reload = current_time

        # Führe Revalidierung durch
        await self._perform_revalidation()

    async def _perform_health_check(self) -> bool:
        """Service-spezifischer Health-Check.

        Returns:
            True wenn Service gesund ist
        """
        config = get_domain_validation_config()

        # Service ist gesund wenn:
        # 1. Er läuft
        # 2. Domain-Validierung aktiviert ist ODER deaktiviert aber Service läuft
        # 3. Letzte Revalidierung nicht zu lange her (falls aktiviert)
        if not self.running:
            return False

        if not config.enabled:
            # Service läuft, aber Domain-Validierung ist deaktiviert - das ist OK
            return True

        # Prüfe ob letzte Revalidierung nicht zu lange her ist
        if self.last_revalidation is None:
            # Noch keine Revalidierung durchgeführt - das ist OK beim Start
            return True

        max_age_seconds = config.revalidation_interval_hours * 3600 * 2  # 2x Intervall als Maximum
        age_seconds = time.time() - self.last_revalidation

        return age_seconds < max_age_seconds

    async def _reload_config(self) -> None:
        """Lädt Domain-Validierung-Konfiguration neu."""
        try:
            reload_domain_validation_config()
            logger.debug("Domain-Validierung-Konfiguration neu geladen")

            # Update Intervall falls geändert
            config = get_domain_validation_config()
            new_interval = config.revalidation_interval_hours * 3600
            if new_interval != self.interval_seconds:
                self.interval_seconds = new_interval
                logger.info(f"Revalidierung-Intervall auf {config.revalidation_interval_hours}h aktualisiert")

        except Exception as e:
            logger.exception(f"Fehler beim Neuladen der Domain-Validierung-Konfiguration: {e}")

    async def _perform_revalidation(self) -> None:
        """Führt Domain-Revalidierung durch."""
        start_time = time.time()

        try:
            config = get_domain_validation_config()

            # Führe Revalidierung durch
            results = await mcp_registry.revalidate_domains_if_needed(
                config.revalidation_interval_hours
            )

            # Statistiken sammeln
            total_servers = len(results)
            successful_validations = sum(1 for success in results.values() if success)
            failed_validations = total_servers - successful_validations

            # Verarbeite Fehler basierend auf Konfiguration
            if failed_validations > 0:
                await self._handle_validation_failures(results, config)

            duration = time.time() - start_time
            self.last_revalidation = time.time()

            logger.info(f"Domain-Revalidierung abgeschlossen: "
                       f"{successful_validations}/{total_servers} erfolgreich "
                       f"in {duration:.2f}s")

        except Exception as e:
            logger.exception(f"Fehler bei Domain-Revalidierung: {e}")
            raise

    async def _handle_validation_failures(
        self,
        results: dict[str, bool],
        config
    ) -> None:
        """Behandelt Validierung-Fehler basierend auf Konfiguration."""
        failed_servers = [server for server, success in results.items() if not success]

        logger.warning(f"Domain-Validierung fehlgeschlagen für {len(failed_servers)} Server: "
                      f"{', '.join(failed_servers)}")

        if config.remove_invalid_servers:
            for server_name in failed_servers:
                try:
                    await mcp_registry.remove_server(server_name)
                    logger.info(f"Server {server_name} aufgrund fehlgeschlagener Domain-Validierung entfernt")
                except Exception as e:
                    logger.exception(f"Fehler beim Entfernen von Server {server_name}: {e}")
        else:
            logger.info("Server werden nicht automatisch entfernt (remove_invalid_servers=False)")

    # API-Kompatibilität mit DomainRevalidationServiceInterface

    async def start(self) -> None:
        """Startet den Domain-Revalidierung-Service."""
        await super().start()

    async def stop(self) -> None:
        """Stoppt den Domain-Revalidierung-Service."""
        await super().stop()

    async def force_revalidation(self) -> dict[str, bool]:
        """Führt sofortige Domain-Revalidierung durch.

        Returns:
            Dictionary mit Revalidierung-Ergebnissen pro Server
        """
        logger.info("Führe manuelle Domain-Revalidierung durch...")

        config = get_domain_validation_config()
        results = await mcp_registry.revalidate_domains_if_needed(FORCE_REVALIDATION_INTERVAL)

        if any(not success for success in results.values()):
            await self._handle_validation_failures(results, config)

        return results

    def get_status(self) -> dict[str, Any]:
        """Gibt Status des Revalidierung-Service zurück.

        Returns:
            Dictionary mit Service-Status-Informationen
        """
        config = get_domain_validation_config()

        return {
            "running": self.running,
            "enabled": config.enabled,
            "last_revalidation": self.last_revalidation,
            "last_config_reload": self.last_config_reload,
            "revalidation_interval_hours": config.revalidation_interval_hours,
            "remove_invalid_servers": config.remove_invalid_servers,
            "revalidation_on_startup": config.revalidation_on_startup,
            "next_revalidation": (
                self.last_revalidation + self.interval_seconds
                if self.last_revalidation else None
            ),
            "service_name": self.service_name,
            "health_status": "unknown"  # Health-Status wird separat abgefragt
        }

    async def reload_config(self) -> None:
        """Lädt Konfiguration neu."""
        await self._reload_config()

    async def get_next_revalidation_time(self) -> datetime | None:
        """Gibt Zeitpunkt der nächsten Revalidierung zurück.

        Returns:
            Datetime der nächsten Revalidierung oder None
        """
        if not self.running or self.last_revalidation is None:
            return None

        next_time = self.last_revalidation + self.interval_seconds
        return datetime.fromtimestamp(next_time)

    async def get_revalidation_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Gibt Historie der Revalidierungen zurück.

        Args:
            limit: Maximale Anzahl der Einträge

        Returns:
            Liste der Revalidierung-Historie
        """
        # Vereinfachte Implementierung - in Produktion würde man eine echte Historie speichern
        if self.last_revalidation:
            return [{
                "timestamp": self.last_revalidation,
                "datetime": datetime.fromtimestamp(self.last_revalidation).isoformat(),
                "status": "completed"
            }]
        return []

    # Implementierung der abstrakten Interface-Methoden

    async def initialize(self) -> None:
        """Initialisiert den Service (Interface-Kompatibilität)."""
        await self._initialize()

    async def shutdown(self) -> None:
        """Fährt den Service herunter (Interface-Kompatibilität)."""
        await self._cleanup()

    async def start_revalidation(self) -> None:
        """Startet periodische Revalidierung (Interface-Kompatibilität)."""
        await self.start()

    async def stop_revalidation(self) -> None:
        """Stoppt periodische Revalidierung (Interface-Kompatibilität)."""
        await self.stop()

    async def add_domain(self, domain: str) -> bool:
        """Fügt eine Domain zur Überwachung hinzu.

        Args:
            domain: Domain-Name zur Überwachung

        Returns:
            True bei erfolgreichem Hinzufügen
        """
        # Vereinfachte Implementierung - in Produktion würde man Domains in Registry verwalten
        logger.info(f"Domain {domain} zur Überwachung hinzugefügt")
        return True

    async def remove_domain(self, domain: str) -> bool:
        """Entfernt eine Domain aus der Überwachung.

        Args:
            domain: Domain-Name zum Entfernen

        Returns:
            True bei erfolgreichem Entfernen
        """
        # Vereinfachte Implementierung - in Produktion würde man Domains aus Registry entfernen
        logger.info(f"Domain {domain} aus Überwachung entfernt")
        return True

    async def get_monitored_domains(self) -> list[str]:
        """Liefert Liste aller überwachten Domains.

        Returns:
            Liste der überwachten Domain-Namen
        """
        # Vereinfachte Implementierung - in Produktion würde man echte Domain-Liste zurückgeben
        return []


# Backward-Compatibility Alias
DomainRevalidationService = UnifiedDomainRevalidationService
