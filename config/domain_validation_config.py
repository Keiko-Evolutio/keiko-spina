"""Konfiguration für Domain-Validierung und periodische Revalidierung.

Dieses Modul definiert Konfigurationsoptionen für die Domain-Whitelist-Validierung
und periodische Revalidierung von registrierten MCP-Servern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from kei_logging import get_logger

logger = get_logger(__name__)


@dataclass
class DomainValidationConfig:
    """Konfiguration für Domain-Validierung."""

    # Basis-Konfiguration
    enabled: bool = True
    allowed_domains: list[str] = None

    # Periodische Revalidierung
    enable_periodic_revalidation: bool = False
    revalidation_interval_hours: int = 24
    revalidation_on_startup: bool = False

    # Verhalten bei Revalidierung-Fehlern
    remove_invalid_servers: bool = False
    mark_invalid_as_unhealthy: bool = True

    # Whitelist-Reload
    enable_config_reload: bool = False
    config_reload_interval_minutes: int = 60

    def __post_init__(self):
        """Post-Initialisierung für Validierung."""
        if self.allowed_domains is None:
            self.allowed_domains = []

        # Validiere Konfiguration
        if self.revalidation_interval_hours < 1:
            logger.warning("Revalidierung-Intervall zu klein, setze auf 1 Stunde")
            self.revalidation_interval_hours = 1

        if self.config_reload_interval_minutes < 5:
            logger.warning("Config-Reload-Intervall zu klein, setze auf 5 Minuten")
            self.config_reload_interval_minutes = 5


def load_domain_validation_config() -> DomainValidationConfig:
    """Lädt Domain-Validierung-Konfiguration aus Umgebungsvariablen.

    Returns:
        DomainValidationConfig-Instanz
    """
    # Basis-Konfiguration
    enabled = os.getenv("KEI_MCP_DOMAIN_VALIDATION_ENABLED", "true").lower() == "true"

    # Allowed Domains aus Umgebungsvariable laden
    allowed_domains_str = os.getenv("KEI_MCP_ALLOWED_DOMAINS", "")
    allowed_domains = []
    if allowed_domains_str:
        allowed_domains = [
            domain.strip()
            for domain in allowed_domains_str.split(",")
            if domain.strip()
        ]

    # Periodische Revalidierung
    enable_periodic_revalidation = os.getenv(
        "KEI_MCP_ENABLE_PERIODIC_REVALIDATION",
        "false"
    ).lower() == "true"

    revalidation_interval_hours = int(os.getenv(
        "KEI_MCP_REVALIDATION_INTERVAL_HOURS",
        "24"
    ))

    revalidation_on_startup = os.getenv(
        "KEI_MCP_REVALIDATION_ON_STARTUP",
        "false"
    ).lower() == "true"

    # Verhalten bei Fehlern
    remove_invalid_servers = os.getenv(
        "KEI_MCP_REMOVE_INVALID_SERVERS",
        "false"
    ).lower() == "true"

    mark_invalid_as_unhealthy = os.getenv(
        "KEI_MCP_MARK_INVALID_AS_UNHEALTHY",
        "true"
    ).lower() == "true"

    # Config-Reload
    enable_config_reload = os.getenv(
        "KEI_MCP_ENABLE_CONFIG_RELOAD",
        "false"
    ).lower() == "true"

    config_reload_interval_minutes = int(os.getenv(
        "KEI_MCP_CONFIG_RELOAD_INTERVAL_MINUTES",
        "60"
    ))

    domain_config = DomainValidationConfig(
        enabled=enabled,
        allowed_domains=allowed_domains,
        enable_periodic_revalidation=enable_periodic_revalidation,
        revalidation_interval_hours=revalidation_interval_hours,
        revalidation_on_startup=revalidation_on_startup,
        remove_invalid_servers=remove_invalid_servers,
        mark_invalid_as_unhealthy=mark_invalid_as_unhealthy,
        enable_config_reload=enable_config_reload,
        config_reload_interval_minutes=config_reload_interval_minutes
    )

    logger.info(f"Domain-Validierung-Konfiguration geladen: "
               f"enabled={domain_config.enabled}, "
               f"domains={len(domain_config.allowed_domains)}, "
               f"periodic_revalidation={domain_config.enable_periodic_revalidation}")

    return domain_config


# Globale Konfiguration-Instanz
DOMAIN_VALIDATION_CONFIG = load_domain_validation_config()


def reload_domain_validation_config() -> DomainValidationConfig:
    """Lädt Domain-Validierung-Konfiguration neu.

    Returns:
        Neue DomainValidationConfig-Instanz
    """
    global DOMAIN_VALIDATION_CONFIG

    logger.info("Lade Domain-Validierung-Konfiguration neu...")
    new_config = load_domain_validation_config()

    # Prüfe auf Änderungen
    if new_config.allowed_domains != DOMAIN_VALIDATION_CONFIG.allowed_domains:
        logger.info(f"Domain-Whitelist geändert: "
                   f"{len(DOMAIN_VALIDATION_CONFIG.allowed_domains)} -> "
                   f"{len(new_config.allowed_domains)} Domains")

    DOMAIN_VALIDATION_CONFIG = new_config
    return new_config


def get_domain_validation_config() -> DomainValidationConfig:
    """Gibt aktuelle Domain-Validierung-Konfiguration zurück.

    Returns:
        Aktuelle DomainValidationConfig-Instanz
    """
    return DOMAIN_VALIDATION_CONFIG


# Beispiel-Konfiguration für .env-Datei
EXAMPLE_ENV_CONFIG = """
# Domain-Validierung Konfiguration
KEI_MCP_DOMAIN_VALIDATION_ENABLED=true
KEI_MCP_ALLOWED_DOMAINS=example.com,api.trusted-service.com,localhost

# Periodische Revalidierung
KEI_MCP_ENABLE_PERIODIC_REVALIDATION=true
KEI_MCP_REVALIDATION_INTERVAL_HOURS=24
KEI_MCP_REVALIDATION_ON_STARTUP=false

# Verhalten bei Revalidierung-Fehlern
KEI_MCP_REMOVE_INVALID_SERVERS=false
KEI_MCP_MARK_INVALID_AS_UNHEALTHY=true

# Config-Reload (für Whitelist-Updates ohne Neustart)
KEI_MCP_ENABLE_CONFIG_RELOAD=true
KEI_MCP_CONFIG_RELOAD_INTERVAL_MINUTES=60
"""


def print_example_config():
    """Gibt Beispiel-Konfiguration aus."""


if __name__ == "__main__":
    # Zeige aktuelle Konfiguration und Beispiel
    config = get_domain_validation_config()


    print_example_config()
