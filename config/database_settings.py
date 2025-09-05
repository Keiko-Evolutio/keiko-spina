"""Database Settings für Keiko Personal Assistant.

Alle datenbankbezogenen Konfigurationen in einer fokussierten Klasse.
Folgt Single Responsibility Principle.
"""

from pydantic import Field
from pydantic_settings import BaseSettings

from .env_utils import get_env_str


class DatabaseSettings(BaseSettings):
    """Datenbank-spezifische Konfigurationen."""

    # Primary Database
    database_url: str = Field(
        default="sqlite:///./app.db",
        description="Haupt-Datenbank URL"
    )

    database_pool_size: int = Field(
        default=10,
        description="Datenbank Connection Pool Größe"
    )

    database_max_overflow: int = Field(
        default=20,
        description="Maximale zusätzliche Verbindungen"
    )

    database_pool_timeout: int = Field(
        default=30,
        description="Connection Pool Timeout in Sekunden"
    )

    database_pool_recycle: int = Field(
        default=3600,
        description="Connection Recycle Zeit in Sekunden"
    )

    # Database Migration
    auto_migrate: bool = Field(
        default=True,
        description="Automatische Datenbankmigrationen"
    )

    migration_timeout: int = Field(
        default=300,
        description="Migration Timeout in Sekunden"
    )

    # Database Backup
    backup_enabled: bool = Field(
        default=True,
        description="Automatische Backups aktivieren"
    )

    backup_interval_hours: int = Field(
        default=24,
        description="Backup-Intervall in Stunden"
    )

    backup_retention_days: int = Field(
        default=30,
        description="Backup-Aufbewahrung in Tagen"
    )

    backup_location: str = Field(
        default="./backups",
        description="Backup-Speicherort"
    )

    # Query Performance
    query_timeout: int = Field(
        default=30,
        description="Standard Query Timeout in Sekunden"
    )

    slow_query_threshold: float = Field(
        default=1.0,
        description="Schwellwert für langsame Queries in Sekunden"
    )

    log_slow_queries: bool = Field(
        default=True,
        description="Langsame Queries loggen"
    )

    # Database Monitoring
    enable_query_logging: bool = Field(
        default=False,
        description="Query-Logging aktivieren (nur Development)"
    )

    enable_performance_monitoring: bool = Field(
        default=True,
        description="Performance-Monitoring aktivieren"
    )

    class Config:
        """Pydantic-Konfiguration."""
        env_prefix = "DATABASE_"
        case_sensitive = False


def load_database_settings() -> DatabaseSettings:
    """Lädt Database Settings aus Umgebungsvariablen.

    Returns:
        DatabaseSettings-Instanz
    """
    return DatabaseSettings(
        database_url=get_env_str("DATABASE_URL", "sqlite:///./app.db"),
        database_pool_size=get_env_str("DATABASE_POOL_SIZE", "10"),
        database_max_overflow=get_env_str("DATABASE_MAX_OVERFLOW", "20"),
        database_pool_timeout=get_env_str("DATABASE_POOL_TIMEOUT", "30"),
        database_pool_recycle=get_env_str("DATABASE_POOL_RECYCLE", "3600"),
        auto_migrate=get_env_str("AUTO_MIGRATE", "true").lower() == "true",
        migration_timeout=get_env_str("MIGRATION_TIMEOUT", "300"),
        backup_enabled=get_env_str("BACKUP_ENABLED", "true").lower() == "true",
        backup_interval_hours=get_env_str("BACKUP_INTERVAL_HOURS", "24"),
        backup_retention_days=get_env_str("BACKUP_RETENTION_DAYS", "30"),
        backup_location=get_env_str("BACKUP_LOCATION", "./backups"),
        query_timeout=get_env_str("QUERY_TIMEOUT", "30"),
        slow_query_threshold=float(get_env_str("SLOW_QUERY_THRESHOLD", "1.0")),
        log_slow_queries=get_env_str("LOG_SLOW_QUERIES", "true").lower() == "true",
        enable_query_logging=get_env_str("ENABLE_QUERY_LOGGING", "false").lower() == "true",
        enable_performance_monitoring=get_env_str("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    )


# Globale Database Settings Instanz
database_settings = load_database_settings()


__all__ = [
    "DatabaseSettings",
    "database_settings",
    "load_database_settings"
]
