"""Produktionsreife Pydantic Logfire-Konfiguration für Keiko Personal Assistant.

Enterprise-Grade-Konfiguration mit vollständiger Integration in die bestehende
Observability-Infrastruktur. Unterstützt alle Logfire-Features und Enterprise-Anforderungen.

Features:
- Vollständige Logfire-Integration mit allen verfügbaren Instrumentierungen
- Enterprise-Sicherheit mit PII-Redaction und Compliance
- Performance-Optimierung und Sampling-Strategien
- Multi-Environment-Support (Development, Staging, Production)
- Nahtlose Integration mit OpenTelemetry/Jaeger/Prometheus
- Monitoring und Alerting für die Logfire-Integration selbst
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

from kei_logging import get_logger

logger = get_logger(__name__)


class LogfireEnvironment(str, Enum):
    """Deployment-Umgebungen für Logfire."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogfireMode(str, Enum):
    """Logfire-Betriebsmodi für verschiedene Deployment-Szenarien."""
    DISABLED = "disabled"           # Logfire komplett deaktiviert
    LOCAL_ONLY = "local_only"       # Nur an lokalen OTEL-Collector
    CLOUD_ONLY = "cloud_only"       # Nur an Logfire Cloud
    DUAL_EXPORT = "dual_export"     # An beide Ziele (empfohlen für Staging)


@dataclass
class LogfirePIIRedactionConfig:
    """Konfiguration für PII-Redaction und Datenschutz-Compliance."""

    # Felder die komplett entfernt werden sollen
    delete_fields: set[str] = field(default_factory=lambda: {
        "password", "secret", "token", "key", "auth", "credential",
        "api_key", "access_token", "refresh_token", "session_token",
        "private_key", "secret_key", "auth_token", "bearer_token"
    })

    # Felder die gehashed werden sollen
    hash_fields: set[str] = field(default_factory=lambda: {
        "user_id", "session_id", "device_id", "client_id",
        "request_id", "trace_id", "correlation_id"
    })

    # Felder die maskiert werden sollen
    mask_fields: set[str] = field(default_factory=lambda: {
        "username", "email", "phone", "name", "address",
        "ip_address", "user_agent", "referrer"
    })

    # Regex-Pattern für String-Redaction
    string_patterns: dict[str, str] = field(default_factory=lambda: {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    })

    # Ersetzungsstrings
    replacements: dict[str, str] = field(default_factory=lambda: {
        "email": "***@***.***",
        "phone": "***-***-****",
        "ssn": "***-**-****",
        "credit_card": "****-****-****-****",
        "ip_address": "***.***.***.***"
    })


class LogfireSettings(BaseSettings):
    """Produktionsreife Logfire-Konfiguration mit Enterprise-Features.

    Unterstützt alle Logfire-Instrumentierungen und Enterprise-Anforderungen.
    Vollständig kompatibel mit der bestehenden OpenTelemetry-Infrastruktur.
    """

    model_config = ConfigDict(
        env_prefix="LOGFIRE_",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore"  # Ignoriere unbekannte Umgebungsvariablen
    )

    # === Grundkonfiguration ===
    mode: LogfireMode = Field(
        default=LogfireMode.LOCAL_ONLY,
        description="Logfire-Betriebsmodus"
    )

    environment: LogfireEnvironment = Field(
        default_factory=lambda: LogfireEnvironment(
            os.getenv("LOGFIRE_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")).lower()
        ),
        description="Deployment-Umgebung (aus LOGFIRE_ENVIRONMENT oder ENVIRONMENT-Variable)"
    )

    service_name: str = Field(
        default="keiko-personal-assistant",
        description="Service-Name für Logfire"
    )

    service_version: str = Field(
        default="1.0.0",
        description="Service-Version"
    )

    # === Logfire Cloud-Konfiguration ===
    token: str | None = Field(
        default=None,
        description="Logfire API Token (erforderlich für Cloud-Modi)",
        alias="LOGFIRE_TOKEN"
    )

    project_name: str | None = Field(
        default=None,
        description="Logfire Projekt-Name",
        alias="LOGFIRE_PROJECT_NAME"
    )

    # === Performance und Sampling ===
    trace_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace-Sampling-Rate (0.0-1.0)"
    )

    batch_timeout_ms: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Batch-Timeout in Millisekunden"
    )

    max_batch_size: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximale Batch-Größe"
    )

    # === Sicherheit und Compliance ===
    enable_pii_redaction: bool = Field(
        default=True,
        description="PII-Redaction aktivieren"
    )

    pii_config: LogfirePIIRedactionConfig = Field(
        default_factory=LogfirePIIRedactionConfig,
        description="PII-Redaction-Konfiguration"
    )

    # === Instrumentierung ===
    enable_auto_instrumentation: bool = Field(
        default=True,
        description="Automatische Instrumentierung aktivieren"
    )

    instrument_openai: bool = Field(
        default=True,
        description="OpenAI-Instrumentierung"
    )

    instrument_anthropic: bool = Field(
        default=True,
        description="Anthropic-Instrumentierung"
    )

    instrument_httpx: bool = Field(
        default=True,
        description="HTTPX-Instrumentierung"
    )

    instrument_requests: bool = Field(
        default=True,
        description="Requests-Instrumentierung"
    )

    instrument_sqlalchemy: bool = Field(
        default=True,
        description="SQLAlchemy-Instrumentierung"
    )

    instrument_fastapi: bool = Field(
        default=True,
        description="FastAPI-Instrumentierung"
    )

    instrument_pydantic: bool = Field(
        default=True,
        description="Pydantic-Instrumentierung"
    )

    instrument_system_metrics: bool = Field(
        default=True,
        description="System-Metrics-Instrumentierung"
    )

    # === Fallback und Fehlerbehandlung ===
    enable_fallback: bool = Field(
        default=True,
        description="Fallback auf OpenTelemetry bei Logfire-Fehlern"
    )

    fallback_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Timeout für Logfire-Requests"
    )

    # === Debug und Monitoring ===
    debug_mode: bool = Field(
        default=False,
        description="Debug-Modus aktivieren"
    )

    log_level: str = Field(
        default="INFO",
        description="Log-Level für Logfire-Integration"
    )

    # === Console-Konfiguration ===
    console_enabled: bool = Field(
        default=True,
        description="Console-Output aktivieren"
    )

    console_colors: str = Field(
        default="auto",
        description="Farben in Console-Output ('auto', 'always', 'never')"
    )

    @field_validator("trace_sample_rate")
    def validate_sample_rate(cls, v: float) -> float:
        """Validiert die Sampling-Rate."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("trace_sample_rate muss zwischen 0.0 und 1.0 liegen")
        return v

    @field_validator("token")
    def validate_token_format(cls, v: str | None) -> str | None:
        """Validiert das Logfire-Token-Format."""
        if v and not v.startswith("pylf_"):
            logger.warning("Logfire-Token hat unerwartetes Format")
        return v

    @field_validator("console_colors")
    def validate_console_colors(cls, v: str) -> str:
        """Validiert console_colors Werte."""
        valid_values = {"auto", "always", "never"}
        if v not in valid_values:
            logger.warning(f"Ungültiger console_colors Wert '{v}', verwende 'auto'")
            return "auto"
        return v


def _configure_logfire_based_on_deployment_mode() -> None:
    """Konfiguriert Logfire basierend auf dem erkannten Deployment-Modus.
    Respektiert explizite 'disabled' Einstellungen aus Konfigurationsdateien.
    """
    try:
        from config.deployment_mode import DeploymentMode, get_deployment_mode

        # Prüfe, ob Logfire explizit deaktiviert wurde
        current_mode = os.getenv("LOGFIRE_MODE", "").lower()
        if current_mode == "disabled":
            logger.info("Logfire explizit deaktiviert - überspringe Deployment-Modus-Konfiguration")
            return

        mode = get_deployment_mode()
        logger.info(f"Konfiguriere Logfire für Deployment-Modus: {mode.value}")

        if mode == DeploymentMode.STANDALONE:
            # Standalone: Nur Cloud, keine lokalen Exporte
            logger.info("Standalone-Modus: Verwende cloud_only")
            os.environ["LOGFIRE_MODE"] = "cloud_only"
            os.environ["LOGFIRE_ENABLE_FALLBACK"] = "false"
            os.environ["LOGFIRE_FALLBACK_TIMEOUT_SECONDS"] = "1.0"  # Mindestens 1.0 für Validierung
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
            os.environ["LOGFIRE_INSTRUMENT_SYSTEM_METRICS"] = "false"
            os.environ["LOGFIRE_CONSOLE_COLORS"] = "auto"  # Korrekte Werte für Logfire

        elif mode == DeploymentMode.CONTAINER_DEV:
            # Container Dev: Prüfe OTEL-Collector-Verfügbarkeit
            from config.deployment_mode import deployment_mode

            if deployment_mode.has_otel_collector():
                logger.info("Container-Dev-Modus: OTEL-Collector verfügbar - verwende dual_export")
                os.environ["LOGFIRE_MODE"] = "dual_export"
                os.environ["LOGFIRE_ENABLE_FALLBACK"] = "true"
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
            else:
                logger.info("Container-Dev-Modus: OTEL-Collector nicht verfügbar - verwende cloud_only")
                os.environ["LOGFIRE_MODE"] = "cloud_only"
                os.environ["LOGFIRE_ENABLE_FALLBACK"] = "false"
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""

            os.environ["LOGFIRE_INSTRUMENT_SYSTEM_METRICS"] = "true"
            os.environ["LOGFIRE_CONSOLE_COLORS"] = "auto"

        elif mode == DeploymentMode.PRODUCTION:
            # Production: Cloud only mit optimierten Einstellungen
            logger.info("Production-Modus: Verwende cloud_only mit Production-Optimierungen")
            os.environ["LOGFIRE_MODE"] = "cloud_only"
            os.environ["LOGFIRE_ENABLE_FALLBACK"] = "true"
            os.environ["LOGFIRE_TRACE_SAMPLE_RATE"] = "0.1"
            os.environ["LOGFIRE_INSTRUMENT_SYSTEM_METRICS"] = "true"
            os.environ["LOGFIRE_CONSOLE_COLORS"] = "never"

    except ImportError as e:
        logger.warning(f"Deployment-Mode-Modul nicht verfügbar: {e}")
        # Fallback auf alte Logik
        _legacy_container_check()
    except Exception as e:
        logger.error(f"Fehler bei Deployment-Modus-Konfiguration: {e}")
        # Fallback auf sichere Einstellungen
        os.environ["LOGFIRE_MODE"] = "cloud_only"
        os.environ["LOGFIRE_ENABLE_FALLBACK"] = "false"


def _legacy_container_check() -> None:
    """Legacy-Fallback für Container-Prüfung."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 4318))
        sock.close()

        if result == 0:
            logger.info("Legacy: Container erkannt")
            os.environ["LOGFIRE_MODE"] = "dual_export"
            os.environ["LOGFIRE_ENABLE_FALLBACK"] = "true"
        else:
            logger.info("Legacy: Standalone erkannt")
            os.environ["LOGFIRE_MODE"] = "cloud_only"
            os.environ["LOGFIRE_ENABLE_FALLBACK"] = "false"
    except Exception:
        os.environ["LOGFIRE_MODE"] = "cloud_only"
        os.environ["LOGFIRE_ENABLE_FALLBACK"] = "false"


def get_logfire_settings() -> LogfireSettings:
    """Lädt und validiert umgebungsspezifische Logfire-Konfiguration.

    Automatische Auswahl basierend auf:
    1. Deployment-Modus (standalone/container_dev/production)
    2. ENVIRONMENT-Variable (development/production)
    3. Lokale Override-Dateien

    Returns:
        LogfireSettings: Konfigurierte Logfire-Settings
    """
    try:
        # Lade umgebungsspezifische Konfigurationsdatei
        _load_env_file()

        # Konfiguriere basierend auf Deployment-Modus
        _configure_logfire_based_on_deployment_mode()

        settings = LogfireSettings()

        # Bestimme aktuelle Umgebung für Logging
        current_environment = os.getenv("ENVIRONMENT", "development").lower()

        # Validiere Cloud-Modi
        if settings.mode in [LogfireMode.CLOUD_ONLY, LogfireMode.DUAL_EXPORT]:
            if not settings.token:
                logger.error(f"Logfire-Token fehlt für Cloud-Modus in {current_environment}-Umgebung")
                if current_environment == "production":
                    logger.critical("KRITISCH: Production-Deployment ohne Logfire-Token!")
                settings.mode = LogfireMode.DISABLED
                return settings

        # Umgebungsspezifische Validierungen
        if settings.environment == LogfireEnvironment.PRODUCTION:
            # Production-spezifische Warnungen
            if settings.trace_sample_rate > 0.5:
                logger.warning("Hohe Sampling-Rate in Produktion - Performance-Impact und Kosten möglich")
            if settings.debug_mode:
                logger.warning("Debug-Modus in Produktion aktiviert - sollte deaktiviert werden")
            if settings.console_enabled:
                logger.warning("Console-Output in Produktion aktiviert - sollte deaktiviert werden")

            # Production-Sicherheitsprüfungen
            if not settings.enable_pii_redaction:
                logger.error("PII-Redaction in Produktion deaktiviert - Compliance-Risiko!")

        elif settings.environment == LogfireEnvironment.DEVELOPMENT:
            # Development-spezifische Optimierungen
            if settings.trace_sample_rate < 1.0:
                logger.info(f"Development-Modus mit reduzierter Sampling-Rate: {settings.trace_sample_rate}")
            if not settings.debug_mode:
                logger.info("Debug-Modus in Development deaktiviert - kann für Troubleshooting aktiviert werden")

        # Erfolgreiche Konfiguration loggen
        config_source = "lokale Override-Datei" if os.path.exists(".env.logfire_local") else f"{current_environment}-spezifische Konfiguration"
        logger.info(
            f"Logfire-Konfiguration erfolgreich geladen: "
            f"mode={settings.mode.value}, "
            f"env={settings.environment.value}, "
            f"source={config_source}"
        )

        return settings

    except Exception as e:
        logger.error(f"Fehler beim Laden der Logfire-Konfiguration: {e}")
        # Fallback auf sichere Defaults
        fallback_settings = LogfireSettings(mode=LogfireMode.DISABLED)
        logger.warning("Verwende Fallback-Konfiguration mit deaktiviertem Logfire")
        return fallback_settings


def _load_env_file() -> None:
    """Lädt umgebungsspezifische Logfire-Konfigurationsdatei basierend auf ENVIRONMENT-Variable.

    Priorität:
    1. config/features/logfire.dev.env (wenn ENVIRONMENT=development)
    2. config/features/logfire.prod.env (wenn ENVIRONMENT=production)
    3. Fallback auf Backend-Root (.env.logfire_dev/.env.logfire_prod)
    4. Fallback auf Default-Werte
    """
    try:
        # Bestimme umgebungsspezifische Konfigurationsdatei
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            # Prüfe zuerst config/features/, dann Backend-Root
            env_file = Path("config/features/logfire.prod.env")
            if not env_file.exists():
                env_file = Path(".env.logfire_prod")
            config_name = "Production"
        elif environment == "development":
            # Prüfe zuerst config/features/, dann Backend-Root
            env_file = Path("config/features/logfire.dev.env")
            if not env_file.exists():
                env_file = Path(".env.logfire_dev")
            config_name = "Development"
        else:
            # Fallback für andere Umgebungen (staging, testing, etc.)
            env_file = Path("config/features/logfire.dev.env")
            if not env_file.exists():
                env_file = Path(".env.logfire_dev")
            config_name = f"Development (Fallback für {environment})"
            logger.warning(f"Unbekannte Umgebung '{environment}' - verwende Development-Konfiguration")

        # 3. Lade umgebungsspezifische Konfiguration
        if env_file.exists():
            logger.info(f"Lade {config_name} Logfire-Konfiguration: {env_file}")
            _load_single_env_file(env_file)
        else:
            logger.warning(f"Logfire-Konfigurationsdatei nicht gefunden: {env_file}")
            logger.info("Verwende Default-Konfiguration aus Environment-Variablen")

    except Exception as e:
        logger.error(f"Fehler beim Laden der Logfire-Konfiguration: {e}")


def _load_single_env_file(env_file: Path) -> None:
    """Lädt eine einzelne .env-Datei und setzt die Environment-Variablen."""
    try:
        with open(env_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    try:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # Entferne Anführungszeichen falls vorhanden
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]

                        os.environ[key] = value

                    except ValueError:
                        logger.warning(f"Ungültige Zeile in {env_file}:{line_num}: {line}")

    except Exception as e:
        logger.error(f"Fehler beim Lesen der Datei {env_file}: {e}")


def validate_logfire_config(settings: LogfireSettings) -> bool:
    """Validiert eine Logfire-Konfiguration.

    Args:
        settings: Zu validierende Logfire-Settings

    Returns:
        bool: True wenn Konfiguration gültig ist
    """
    try:
        # Disabled-Modus ist immer gültig
        if settings.mode == LogfireMode.DISABLED:
            return True

        # Cloud-Modi benötigen Token
        if settings.mode in [LogfireMode.CLOUD_ONLY, LogfireMode.DUAL_EXPORT]:
            if not settings.token:
                logger.error("Logfire-Token fehlt für Cloud-Modus")
                return False

        # Validiere Sampling-Rate
        if not 0.0 <= settings.trace_sample_rate <= 1.0:
            logger.error("Ungültige Sampling-Rate")
            return False

        logger.info("Logfire-Konfiguration erfolgreich validiert")
        return True

    except Exception as e:
        logger.error(f"Fehler bei Konfigurationsvalidierung: {e}")
        return False
