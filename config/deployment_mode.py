"""Deployment Mode Detection und Management für Keiko Personal Assistant.

Dieses Modul erkennt automatisch den Deployment-Modus beim Start und stellt
diese Information für alle anderen Services zur Verfügung.

Deployment-Modi:
- STANDALONE: Nur Backend ohne Container (einfacher Start)
- CONTAINER_DEV: Backend + Development Container (make dev)
- PRODUCTION: Production Deployment
"""

import logging
import os
import socket
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Verfügbare Deployment-Modi."""
    STANDALONE = "standalone"
    CONTAINER_DEV = "container_dev"
    PRODUCTION = "production"


class DeploymentModeManager:
    """Manager für Deployment-Modus-Erkennung und -Verwaltung."""

    _instance: Optional["DeploymentModeManager"] = None
    _mode: DeploymentMode | None = None
    _mode_file = Path(".deployment_mode")

    def __new__(cls) -> "DeploymentModeManager":
        """Singleton-Pattern für globalen Zugriff."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def detect_and_set_mode(self) -> DeploymentMode:
        """Erkennt automatisch den Deployment-Modus und setzt ihn.

        Returns:
            DeploymentMode: Der erkannte Deployment-Modus
        """
        if self._mode is not None:
            return self._mode

        # Prüfe zuerst Environment-Variable (höchste Priorität)
        env_mode = os.getenv("KEIKO_DEPLOYMENT_MODE")
        if env_mode:
            try:
                self._mode = DeploymentMode(env_mode.lower())
                logger.info(f"Deployment-Modus aus Environment-Variable: {self._mode.value}")
                self._save_mode_to_file()
                return self._mode
            except ValueError:
                logger.warning(f"Ungültiger KEIKO_DEPLOYMENT_MODE: {env_mode}")

        # Prüfe gespeicherten Modus aus vorherigem Start
        saved_mode = self._load_mode_from_file()
        if saved_mode and self._validate_mode(saved_mode):
            self._mode = saved_mode
            logger.info(f"Deployment-Modus aus gespeicherter Datei: {self._mode.value}")
            return self._mode

        # Automatische Erkennung
        self._mode = self._auto_detect_mode()
        logger.info(f"Deployment-Modus automatisch erkannt: {self._mode.value}")
        self._save_mode_to_file()
        return self._mode

    def _auto_detect_mode(self) -> DeploymentMode:
        """Automatische Erkennung des Deployment-Modus.

        Returns:
            DeploymentMode: Der erkannte Modus
        """
        # Prüfe Environment-Variable für Production
        environment = os.getenv("ENVIRONMENT", "development").lower()
        if environment == "production":
            logger.info("Production-Umgebung erkannt")
            return DeploymentMode.PRODUCTION

        # Prüfe Container-Verfügbarkeit für Development
        if self._check_development_containers():
            logger.info("Development-Container erkannt")
            return DeploymentMode.CONTAINER_DEV

        logger.info("Standalone-Modus erkannt (keine Container)")
        return DeploymentMode.STANDALONE

    def _check_development_containers(self) -> bool:
        """Prüft, ob Development-Container verfügbar sind.

        Returns:
            bool: True wenn Container verfügbar sind
        """
        # Prüfe essenzielle Services (PostgreSQL, Redis, NATS, OTEL-Collector)
        essential_services = [
            ("localhost", 5432, "PostgreSQL"),
            ("localhost", 6379, "Redis"),
            ("localhost", 4222, "NATS"),
            ("localhost", 4318, "OTEL-Collector")
        ]

        available_services = 0
        for host, port, name in essential_services:
            if self._check_port_available(host, port):
                logger.debug(f"{name} auf Port {port} verfügbar")
                available_services += 1

        # Wenn mindestens 3 der 4 essentiellen Services laufen, sind Container verfügbar
        if available_services >= 3:
            logger.debug(f"{available_services}/4 essentielle Services verfügbar")
            return True

        # Prüfe Docker-Container direkt als Fallback
        if self._check_docker_containers():
            logger.debug("Docker-Container direkt erkannt")
            return True

        return False

    def has_otel_collector(self) -> bool:
        """Prüft spezifisch, ob der OTEL-Collector verfügbar ist.

        Returns:
            bool: True wenn OTEL-Collector auf Port 4318 verfügbar ist
        """
        return self._check_port_available("localhost", 4318)

    def _check_port_available(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Prüft, ob ein Port erreichbar ist.

        Args:
            host: Hostname
            port: Port-Nummer
            timeout: Timeout in Sekunden

        Returns:
            bool: True wenn Port erreichbar ist
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _check_docker_containers(self) -> bool:
        """Prüft Docker-Container direkt.

        Returns:
            bool: True wenn Keiko-Container laufen
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=keiko-", "--format", "{{.Names}}"],
                check=False, capture_output=True, text=True, timeout=3
            )
            return "keiko-" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def _validate_mode(self, mode: DeploymentMode) -> bool:
        """Validiert, ob ein gespeicherter Modus noch gültig ist.

        Args:
            mode: Zu validierender Modus

        Returns:
            bool: True wenn Modus noch gültig ist
        """
        if mode == DeploymentMode.PRODUCTION:
            return os.getenv("ENVIRONMENT", "development").lower() == "production"
        if mode == DeploymentMode.CONTAINER_DEV:
            return self._check_development_containers()
        # STANDALONE
        return not self._check_development_containers()

    def _save_mode_to_file(self) -> None:
        """Speichert den aktuellen Modus in eine Datei."""
        try:
            if self._mode:
                self._mode_file.write_text(self._mode.value)
                logger.debug(f"Deployment-Modus gespeichert: {self._mode.value}")
        except Exception as e:
            logger.warning(f"Fehler beim Speichern des Deployment-Modus: {e}")

    def _load_mode_from_file(self) -> DeploymentMode | None:
        """Lädt den Modus aus einer gespeicherten Datei.

        Returns:
            Optional[DeploymentMode]: Gespeicherter Modus oder None
        """
        try:
            if self._mode_file.exists():
                mode_str = self._mode_file.read_text().strip()
                return DeploymentMode(mode_str)
        except Exception as e:
            logger.debug(f"Fehler beim Laden des gespeicherten Modus: {e}")
        return None

    def get_mode(self) -> DeploymentMode:
        """Gibt den aktuellen Deployment-Modus zurück.

        Returns:
            DeploymentMode: Aktueller Modus
        """
        if self._mode is None:
            return self.detect_and_set_mode()
        return self._mode

    def is_standalone(self) -> bool:
        """Prüft, ob im Standalone-Modus."""
        return self.get_mode() == DeploymentMode.STANDALONE

    def is_container_dev(self) -> bool:
        """Prüft, ob im Container-Development-Modus."""
        return self.get_mode() == DeploymentMode.CONTAINER_DEV

    def is_production(self) -> bool:
        """Prüft, ob im Production-Modus."""
        return self.get_mode() == DeploymentMode.PRODUCTION

    def has_containers(self) -> bool:
        """Prüft, ob Container verfügbar sind."""
        return self.get_mode() in [DeploymentMode.CONTAINER_DEV, DeploymentMode.PRODUCTION]

    def force_mode(self, mode: DeploymentMode) -> None:
        """Erzwingt einen bestimmten Modus (für Tests oder manuelle Überschreibung).

        Args:
            mode: Zu setzender Modus
        """
        self._mode = mode
        self._save_mode_to_file()
        logger.info(f"Deployment-Modus manuell gesetzt: {mode.value}")

    def reset(self) -> None:
        """Setzt den Modus zurück und löscht gespeicherte Daten."""
        self._mode = None
        try:
            if self._mode_file.exists():
                self._mode_file.unlink()
        except Exception as e:
            logger.warning(f"Fehler beim Löschen der Modus-Datei: {e}")


# Globale Instanz für einfachen Zugriff
deployment_mode = DeploymentModeManager()


def get_deployment_mode() -> DeploymentMode:
    """Convenience-Funktion für den Zugriff auf den Deployment-Modus.

    Returns:
        DeploymentMode: Aktueller Deployment-Modus
    """
    return deployment_mode.get_mode()


def is_standalone() -> bool:
    """Prüft, ob im Standalone-Modus."""
    return deployment_mode.is_standalone()


def is_container_dev() -> bool:
    """Prüft, ob im Container-Development-Modus."""
    return deployment_mode.is_container_dev()


def is_production() -> bool:
    """Prüft, ob im Production-Modus."""
    return deployment_mode.is_production()


def has_containers() -> bool:
    """Prüft, ob Container verfügbar sind."""
    return deployment_mode.has_containers()
