"""Security Settings für Keiko Personal Assistant.

Alle sicherheitsbezogenen Konfigurationen in einer fokussierten Klasse.
Folgt Single Responsibility Principle.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

from .constants import (
    DEFAULT_DDOS_ANOMALY_REQUESTS_PER_MINUTE,
    DEFAULT_DDOS_BURST_THRESHOLD,
    DEFAULT_DDOS_IP_BLOCK_DURATION_SECONDS,
    DEFAULT_N8N_HMAC_SECRET,
    DEFAULT_SECRET_CACHE_TTL_SECONDS,
    DEFAULT_SECRET_GRACE_PERIOD_HOURS,
    DEFAULT_SECRET_ROTATION_INTERVAL_DAYS,
)
from .env_utils import get_env_bool, get_env_int, get_env_str


class SecuritySettings(BaseSettings):
    """Sicherheits-spezifische Konfigurationen."""

    # Secret Management
    secret_rotation_enabled: bool = Field(
        default=True,
        description="Automatische Secret-Rotation aktivieren"
    )

    secret_rotation_interval_days: int = Field(
        default=DEFAULT_SECRET_ROTATION_INTERVAL_DAYS,
        description="Intervall für Secret-Rotation in Tagen"
    )

    secret_grace_period_hours: int = Field(
        default=DEFAULT_SECRET_GRACE_PERIOD_HOURS,
        description="Gnadenfrist für alte Secrets in Stunden"
    )

    secret_cache_ttl_seconds: int = Field(
        default=DEFAULT_SECRET_CACHE_TTL_SECONDS,
        description="Cache TTL für Secrets in Sekunden"
    )

    inbound_hmac_secret_key_name: str = Field(
        default="",
        description="Name des HMAC Secret Keys für eingehende Requests"
    )

    # n8n Webhook Security
    n8n_hmac_secret: SecretStr = Field(
        default=SecretStr(DEFAULT_N8N_HMAC_SECRET),
        description="HMAC Secret für n8n Webhook-Validierung"
    )

    n8n_base_url: str = Field(
        default="",
        description="Basis-URL für n8n Integration"
    )

    n8n_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API Key für n8n Integration"
    )

    # DDoS Protection
    ddos_enabled: bool = Field(
        default=True,
        description="DDoS-Schutz aktivieren"
    )

    ddos_ip_block_duration_seconds: int = Field(
        default=DEFAULT_DDOS_IP_BLOCK_DURATION_SECONDS,
        ge=0,
        description="IP-Block-Dauer bei DDoS in Sekunden"
    )

    ddos_anomaly_requests_per_minute: int = Field(
        default=DEFAULT_DDOS_ANOMALY_REQUESTS_PER_MINUTE,
        ge=1,
        description="Anomalie-Schwellwert für Requests pro Minute"
    )

    ddos_burst_threshold: int = Field(
        default=DEFAULT_DDOS_BURST_THRESHOLD,
        ge=1,
        description="Burst-Schwellwert für DDoS-Erkennung"
    )

    ddos_geo_filter_enabled: bool = Field(
        default=False,
        description="Geografische Filterung aktivieren"
    )

    ddos_geo_allowed_countries: str = Field(
        default="",
        description="Erlaubte Länder (CSV ISO Codes, z.B. 'DE,AT,CH')"
    )

    ddos_trust_proxy_headers: bool = Field(
        default=True,
        description="Proxy-Headers (X-Forwarded-For, X-Real-IP) vertrauen"
    )

    # Authentication Configuration
    auth_enabled: bool = Field(
        default=True,
        description="Authentifizierung aktivieren"
    )

    auth_mode: str = Field(
        default="hybrid",
        description="Authentifizierungsmodus (oidc, static, hybrid)"
    )

    jwt_secret_key: SecretStr = Field(
        default=SecretStr(""),
        description="JWT Secret Key für Token-Signierung"
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT-Algorithmus"
    )

    jwt_expiration_hours: int = Field(
        default=24,
        description="JWT Token Gültigkeitsdauer in Stunden"
    )

    # API Security
    api_key_required: bool = Field(
        default=True,
        description="API-Key für alle Requests erforderlich"
    )

    cors_enabled: bool = Field(
        default=True,
        description="CORS-Unterstützung aktivieren"
    )

    cors_allowed_origins: str = Field(
        default="*",
        description="Erlaubte CORS Origins (CSV)"
    )

    class Config:
        """Pydantic-Konfiguration."""
        env_prefix = "SECURITY_"
        case_sensitive = False


def load_security_settings() -> SecuritySettings:
    """Lädt Security Settings aus Umgebungsvariablen.

    Returns:
        SecuritySettings-Instanz
    """
    return SecuritySettings(
        secret_rotation_enabled=get_env_bool("SECRET_ROTATION_ENABLED", True),
        secret_rotation_interval_days=get_env_int("SECRET_ROTATION_INTERVAL_DAYS", DEFAULT_SECRET_ROTATION_INTERVAL_DAYS),
        secret_grace_period_hours=get_env_int("SECRET_GRACE_PERIOD_HOURS", DEFAULT_SECRET_GRACE_PERIOD_HOURS),
        secret_cache_ttl_seconds=get_env_int("SECRET_CACHE_TTL_SECONDS", DEFAULT_SECRET_CACHE_TTL_SECONDS),
        inbound_hmac_secret_key_name=get_env_str("INBOUND_HMAC_SECRET_KEY_NAME"),
        n8n_hmac_secret=SecretStr(get_env_str("N8N_HMAC_SECRET", DEFAULT_N8N_HMAC_SECRET)),
        n8n_base_url=get_env_str("N8N_BASE_URL"),
        n8n_api_key=SecretStr(get_env_str("N8N_API_KEY")),
        ddos_enabled=get_env_bool("DDOS_ENABLED", True),
        ddos_ip_block_duration_seconds=get_env_int("DDOS_IP_BLOCK_DURATION_SECONDS", DEFAULT_DDOS_IP_BLOCK_DURATION_SECONDS),
        ddos_anomaly_requests_per_minute=get_env_int("DDOS_ANOMALY_REQUESTS_PER_MINUTE", DEFAULT_DDOS_ANOMALY_REQUESTS_PER_MINUTE),
        ddos_burst_threshold=get_env_int("DDOS_BURST_THRESHOLD", DEFAULT_DDOS_BURST_THRESHOLD),
        ddos_geo_filter_enabled=get_env_bool("DDOS_GEO_FILTER_ENABLED", False),
        ddos_geo_allowed_countries=get_env_str("DDOS_GEO_ALLOWED_COUNTRIES"),
        ddos_trust_proxy_headers=get_env_bool("DDOS_TRUST_PROXY_HEADERS", True),
        auth_enabled=get_env_bool("AUTH_ENABLED", False),
        auth_mode=get_env_str("AUTH_MODE", "hybrid"),
        jwt_secret_key=SecretStr(get_env_str("JWT_SECRET_KEY")),
        jwt_algorithm=get_env_str("JWT_ALGORITHM", "HS256"),
        jwt_expiration_hours=get_env_int("JWT_EXPIRATION_HOURS", 24),
        api_key_required=get_env_bool("API_KEY_REQUIRED", True),
        cors_enabled=get_env_bool("CORS_ENABLED", True),
        cors_allowed_origins=get_env_str("CORS_ALLOWED_ORIGINS", "*")
    )


# Globale Security Settings Instanz
security_settings = load_security_settings()


__all__ = [
    "SecuritySettings",
    "load_security_settings",
    "security_settings"
]
