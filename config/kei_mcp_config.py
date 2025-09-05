"""Konfiguration für KEI-MCP Server.

Diese Datei enthält alle Konfigurationsoptionen für die Integration
externer MCP Server in die Keiko Personal Assistant Plattform.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


class KEIMCPConfig(BaseModel):
    """Konfiguration für KEI-MCP Server."""

    server_name: str = Field(..., description="Eindeutiger Name des MCP Servers")
    base_url: str = Field(..., description="Basis-URL des MCP Servers")
    api_key: str | None = Field(None, description="API-Key für Authentifizierung")
    timeout_seconds: float = Field(default=30.0, description="Timeout für HTTP-Requests")
    max_retries: int = Field(default=3, description="Maximale Anzahl von Wiederholungsversuchen")
    custom_headers: dict[str, str] | None = Field(None, description="Zusätzliche HTTP-Headers")


class KEIMCPSettings(BaseModel):
    """Settings für KEI-MCP Server Integration.

    Diese Klasse definiert alle Konfigurationsoptionen für die KEI-MCP
    Server Integration, einschließlich automatischer Registrierung,
    Health Monitoring und Performance-Tuning.
    """

    # Automatisch zu registrierende Server
    auto_register_servers: list[KEIMCPConfig] = Field(
        default_factory=list,
        description="Liste der Server die beim Startup automatisch registriert werden"
    )

    # Health Check Konfiguration
    health_check_interval_seconds: float = Field(
        default=60.0,
        description="Intervall für automatische Health Checks in Sekunden",
        ge=10.0,
        le=3600.0
    )

    # Standard-Timeouts
    default_timeout_seconds: float = Field(
        default=30.0,
        description="Standard-Timeout für HTTP-Requests in Sekunden",
        ge=1.0,
        le=300.0
    )

    default_max_retries: int = Field(
        default=3,
        description="Standard-Anzahl von Wiederholungsversuchen",
        ge=0,
        le=10
    )

    # Cache-Konfiguration
    tool_cache_ttl_seconds: float = Field(
        default=300.0,
        description="Time-to-Live für Tool-Cache in Sekunden",
        ge=60.0,
        le=3600.0
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=100,
        description="Rate Limit für Tool-Aufrufe pro Minute pro Server",
        ge=1,
        le=10000
    )

    # Sicherheit
    require_api_key: bool = Field(
        default=True,
        description="Ob API-Keys für externe Server erforderlich sind"
    )

    allowed_domains: list[str] | None = Field(
        default=None,
        description="Whitelist erlaubter Domains für externe Server"
    )

    # Monitoring
    enable_detailed_logging: bool = Field(
        default=True,
        description="Ob detailliertes Logging aktiviert werden soll"
    )

    enable_performance_metrics: bool = Field(
        default=True,
        description="Ob Performance-Metriken erfasst werden sollen"
    )

    # Erweiterte Optionen
    max_concurrent_requests: int = Field(
        default=50,
        description="Maximale Anzahl gleichzeitiger Requests pro Server",
        ge=1,
        le=1000
    )

    connection_pool_size: int = Field(
        default=10,
        description="Größe des HTTP-Connection-Pools",
        ge=1,
        le=100
    )

    @field_validator("allowed_domains")
    def validate_allowed_domains(cls, v):
        """Validiert die erlaubten Domains."""
        if v is not None:
            for domain in v:
                if not domain or not isinstance(domain, str):
                    raise ValueError("Alle Domains müssen gültige Strings sein")
        return v


def load_kei_mcp_settings() -> KEIMCPSettings:
    """Lädt die externe MCP Konfiguration aus Umgebungsvariablen und Defaults.

    Returns:
        Konfigurierte ExternalMCPSettings-Instanz
    """
    # Basis-Konfiguration
    settings = KEIMCPSettings()

    # Umgebungsvariablen überschreiben Defaults
    if os.getenv("EXTERNAL_MCP_HEALTH_CHECK_INTERVAL"):
        settings.health_check_interval_seconds = float(
            os.getenv("EXTERNAL_MCP_HEALTH_CHECK_INTERVAL", "60.0")
        )

    if os.getenv("EXTERNAL_MCP_DEFAULT_TIMEOUT"):
        settings.default_timeout_seconds = float(
            os.getenv("EXTERNAL_MCP_DEFAULT_TIMEOUT", "30.0")
        )

    if os.getenv("EXTERNAL_MCP_RATE_LIMIT"):
        settings.rate_limit_requests_per_minute = int(
            os.getenv("EXTERNAL_MCP_RATE_LIMIT", "100")
        )

    if os.getenv("EXTERNAL_MCP_REQUIRE_API_KEY"):
        settings.require_api_key = os.getenv("EXTERNAL_MCP_REQUIRE_API_KEY", "true").lower() == "true"

    # Erlaubte Domains aus Umgebungsvariable
    if os.getenv("EXTERNAL_MCP_ALLOWED_DOMAINS"):
        domains = os.getenv("EXTERNAL_MCP_ALLOWED_DOMAINS", "").split(",")
        settings.allowed_domains = [domain.strip() for domain in domains if domain.strip()]

    return settings


def create_example_server_configs() -> list[KEIMCPConfig]:
    """Erstellt Beispiel-Konfigurationen für externe MCP Server.

    Diese Funktion zeigt, wie externe MCP Server konfiguriert werden können.
    In einer Produktionsumgebung sollten diese Konfigurationen aus sicheren
    Quellen wie Azure Key Vault geladen werden.

    Returns:
        Liste von Beispiel-Server-Konfigurationen
    """
    example_configs = []

    # Beispiel: Weather Service
    if os.getenv("WEATHER_MCP_URL") and os.getenv("WEATHER_MCP_API_KEY"):
        weather_config = KEIMCPConfig(
            server_name="weather-service",
            base_url=os.getenv("WEATHER_MCP_URL"),
            api_key=os.getenv("WEATHER_MCP_API_KEY"),
            timeout_seconds=15.0,
            max_retries=2,
            custom_headers={
                "User-Agent": "Keiko-Personal-Assistant/1.0",
                "X-Service-Type": "weather"
            }
        )
        example_configs.append(weather_config)

    # Beispiel: Database Service (intern)
    if os.getenv("DATABASE_MCP_URL"):
        database_config = KEIMCPConfig(
            server_name="database-service",
            base_url=os.getenv("DATABASE_MCP_URL"),
            api_key=os.getenv("DATABASE_MCP_API_KEY"),
            timeout_seconds=45.0,
            max_retries=3,
            custom_headers={
                "X-Service-Type": "database",
                "X-Internal-Service": "true"
            }
        )
        example_configs.append(database_config)

    # Beispiel: Document Processing Service
    if os.getenv("DOCUMENT_MCP_URL"):
        document_config = KEIMCPConfig(
            server_name="document-processor",
            base_url=os.getenv("DOCUMENT_MCP_URL"),
            api_key=os.getenv("DOCUMENT_MCP_API_KEY"),
            timeout_seconds=120.0,  # Längerer Timeout für Dokumentenverarbeitung
            max_retries=1,
            custom_headers={
                "X-Service-Type": "document",
                "Accept": "application/json"
            }
        )
        example_configs.append(document_config)

    # Beispiel: AI/ML Service
    if os.getenv("AI_MCP_URL"):
        ai_config = KEIMCPConfig(
            server_name="ai-ml-service",
            base_url=os.getenv("AI_MCP_URL"),
            api_key=os.getenv("AI_MCP_API_KEY"),
            timeout_seconds=60.0,
            max_retries=2,
            custom_headers={
                "X-Service-Type": "ai-ml",
                "X-Model-Version": "v2.1"
            }
        )
        example_configs.append(ai_config)

    return example_configs


# Globale Konfiguration laden
KEI_MCP_SETTINGS = load_kei_mcp_settings()

# Auto-Register Server aus Umgebungsvariablen
KEI_MCP_SETTINGS.auto_register_servers = create_example_server_configs()


# Vordefinierte Server-Templates für häufige Use Cases
SERVER_TEMPLATES = {
    "weather": {
        "description": "Weather and forecast service",
        "typical_tools": ["get_weather", "get_forecast", "get_alerts"],
        "required_capabilities": ["weather", "location"],
        "recommended_timeout": 15.0
    },

    "database": {
        "description": "Database query and management service",
        "typical_tools": ["query", "insert", "update", "delete"],
        "required_capabilities": ["database", "sql"],
        "recommended_timeout": 45.0
    },

    "document": {
        "description": "Document processing and analysis service",
        "typical_tools": ["parse_pdf", "extract_text", "analyze_document"],
        "required_capabilities": ["document", "text", "analysis"],
        "recommended_timeout": 120.0
    },

    "ai_ml": {
        "description": "AI/ML inference and processing service",
        "typical_tools": ["predict", "classify", "generate", "analyze"],
        "required_capabilities": ["ai", "ml", "inference"],
        "recommended_timeout": 60.0
    },

    "web": {
        "description": "Web scraping and API integration service",
        "typical_tools": ["scrape", "api_call", "fetch_data"],
        "required_capabilities": ["web", "http", "api"],
        "recommended_timeout": 30.0
    },

    "file": {
        "description": "File storage and management service",
        "typical_tools": ["upload", "download", "list", "delete"],
        "required_capabilities": ["file", "storage"],
        "recommended_timeout": 60.0
    }
}


def get_server_template(template_name: str) -> dict[str, Any] | None:
    """Gibt ein Server-Template zurück.

    Args:
        template_name: Name des Templates

    Returns:
        Template-Dictionary oder None
    """
    return SERVER_TEMPLATES.get(template_name)


def create_server_config_from_template(
    template_name: str,
    server_name: str,
    base_url: str,
    api_key: str | None = None,
    **kwargs
) -> KEIMCPConfig | None:
    """Erstellt eine Server-Konfiguration basierend auf einem Template.

    Args:
        template_name: Name des Templates
        server_name: Name des Servers
        base_url: Basis-URL des Servers
        api_key: Optionaler API-Key
        **kwargs: Zusätzliche Konfigurationsoptionen

    Returns:
        Konfigurierte KEIMCPConfig oder None
    """
    template = get_server_template(template_name)
    if not template:
        return None

    # Standard-Werte aus Template
    timeout_seconds = kwargs.get("timeout_seconds", template.get("recommended_timeout", 30.0))
    max_retries = kwargs.get("max_retries", KEI_MCP_SETTINGS.default_max_retries)

    # Custom Headers basierend auf Template
    custom_headers = kwargs.get("custom_headers", {})
    custom_headers.update({
        "User-Agent": "Keiko-Personal-Assistant/1.0",
        "X-Service-Type": template_name,
        "X-Template-Version": "1.0"
    })

    return KEIMCPConfig(
        server_name=server_name,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        custom_headers=custom_headers
    )


__all__ = [
    "KEI_MCP_SETTINGS",
    "SERVER_TEMPLATES",
    "KEIMCPConfig",
    "KEIMCPSettings",
    "create_example_server_configs",
    "create_server_config_from_template",
    "get_server_template",
    "load_kei_mcp_settings"
]


# Globale Settings-Instanz (Duplikat entfernt - bereits in Zeile 240 definiert)
