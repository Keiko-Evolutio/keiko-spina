"""Einheitliche Rate-Limiting-Konfiguration für Keiko Personal Assistant.

Konsolidiert die duplizierten Rate-Limiting-Implementierungen zu einer
einheitlichen, enterprise-grade Lösung mit Clean Code Prinzipien.

Ersetzt:
- rate_limit_config.py
- rate_limiting_config.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from kei_logging import get_logger

from .constants import (
    ADMIN_TIER_BURST_SIZE,
    ADMIN_TIER_REQUESTS_PER_DAY,
    ADMIN_TIER_REQUESTS_PER_HOUR,
    ADMIN_TIER_REQUESTS_PER_MINUTE,
    BASIC_TIER_BURST_SIZE,
    BASIC_TIER_FRAMES_PER_SECOND,
    BASIC_TIER_MAX_CONCURRENT_STREAMS,
    BASIC_TIER_MAX_STREAM_DURATION_SECONDS,
    BASIC_TIER_REQUESTS_PER_DAY,
    BASIC_TIER_REQUESTS_PER_HOUR,
    # Rate Limiting Constants
    BASIC_TIER_REQUESTS_PER_MINUTE,
    BASIC_TIER_REQUESTS_PER_SECOND,
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_GRACE_PERIOD_SECONDS,
    DEFAULT_IP_REQUESTS_PER_MINUTE,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_RATE_LIMIT_BURST_REFILL_RATE,
    DEFAULT_RATE_LIMIT_SOFT_LIMIT_FACTOR,
    DEFAULT_RATE_LIMIT_WINDOW_SIZE_SECONDS,
    DEFAULT_REDIS_DB,
    DEFAULT_REDIS_HOST,
    DEFAULT_REDIS_PORT,
    DEFAULT_REDIS_TIMEOUT_SECONDS,
    ENTERPRISE_TIER_BURST_SIZE,
    ENTERPRISE_TIER_FRAMES_PER_SECOND,
    ENTERPRISE_TIER_MAX_CONCURRENT_STREAMS,
    ENTERPRISE_TIER_MAX_STREAM_DURATION_SECONDS,
    ENTERPRISE_TIER_REQUESTS_PER_DAY,
    ENTERPRISE_TIER_REQUESTS_PER_HOUR,
    ENTERPRISE_TIER_REQUESTS_PER_MINUTE,
    ENTERPRISE_TIER_REQUESTS_PER_SECOND,
    FREE_TIER_FRAMES_PER_SECOND,
    FREE_TIER_MAX_CONCURRENT_STREAMS,
    FREE_TIER_MAX_STREAM_DURATION_SECONDS,
    # Streaming Constants
    FREE_TIER_REQUESTS_PER_SECOND,
    PREMIUM_TIER_BURST_SIZE,
    PREMIUM_TIER_FRAMES_PER_SECOND,
    PREMIUM_TIER_MAX_CONCURRENT_STREAMS,
    PREMIUM_TIER_MAX_STREAM_DURATION_SECONDS,
    PREMIUM_TIER_REQUESTS_PER_DAY,
    PREMIUM_TIER_REQUESTS_PER_HOUR,
    PREMIUM_TIER_REQUESTS_PER_MINUTE,
    PREMIUM_TIER_REQUESTS_PER_SECOND,
    UNLIMITED_TIER_BURST_CAPACITY,
    UNLIMITED_TIER_FRAMES_PER_SECOND,
    UNLIMITED_TIER_MAX_CONCURRENT_STREAMS,
    UNLIMITED_TIER_MAX_STREAM_DURATION_SECONDS,
    UNLIMITED_TIER_REQUESTS_PER_SECOND,
)
from .env_utils import (
    get_env_bool,
    get_env_enum,
    get_env_int,
    get_env_str,
    get_redis_config,
)

logger = get_logger(__name__)


class RateLimitBackend(str, Enum):
    """Rate Limiting Backend-Typen."""
    REDIS = "redis"
    MEMORY = "memory"
    HYBRID = "hybrid"  # Redis mit Memory-Fallback


class RateLimitTier(str, Enum):
    """Rate Limit Tiers für verschiedene API-Key-Typen."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"
    UNLIMITED = "unlimited"


class RateLimitAlgorithm(str, Enum):
    """Rate Limiting Algorithmen."""
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"


class EndpointType(str, Enum):
    """Endpoint-Typen für spezifische Rate-Limiting-Strategien."""
    REST = "rest"
    WEBSOCKET = "websocket"
    SSE = "sse"
    GRAPHQL = "graphql"


class IdentificationStrategy(str, Enum):
    """Strategien zur Benutzer-Identifikation."""
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    TENANT_ID = "tenant_id"


@dataclass
class RateLimitPolicy:
    """Rate Limit Policy für spezifische Operation."""

    # Standard Rate Limits
    requests_per_minute: int = BASIC_TIER_REQUESTS_PER_MINUTE
    requests_per_hour: int = BASIC_TIER_REQUESTS_PER_HOUR
    requests_per_day: int = BASIC_TIER_REQUESTS_PER_DAY

    # Burst-Handling
    burst_size: int = BASIC_TIER_BURST_SIZE
    burst_refill_rate: float = DEFAULT_RATE_LIMIT_BURST_REFILL_RATE

    # Algorithmus
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW

    # Zeitfenster
    window_size_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SIZE_SECONDS

    # Graceful Degradation
    soft_limit_factor: float = DEFAULT_RATE_LIMIT_SOFT_LIMIT_FACTOR

    # Streaming-spezifische Limits
    requests_per_second: float | None = None
    frames_per_second: float | None = None
    max_concurrent_streams: int | None = None
    max_stream_duration_seconds: int | None = None

    # Erweiterte Konfiguration
    grace_period_seconds: int = DEFAULT_GRACE_PERIOD_SECONDS
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    max_backoff_seconds: int = DEFAULT_MAX_BACKOFF_SECONDS

    # Endpoint-spezifische Konfiguration
    endpoint_type: EndpointType = EndpointType.REST
    identification_strategy: IdentificationStrategy = IdentificationStrategy.API_KEY
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validierung der Policy-Parameter."""
        if self.burst_size > self.requests_per_minute:
            logger.warning(f"Burst-Size ({self.burst_size}) größer als Minuten-Limit ({self.requests_per_minute})")

        if self.soft_limit_factor >= 1.0:
            self.soft_limit_factor = DEFAULT_RATE_LIMIT_SOFT_LIMIT_FACTOR
            logger.warning(f"Soft-Limit-Factor auf {DEFAULT_RATE_LIMIT_SOFT_LIMIT_FACTOR} korrigiert")

    @classmethod
    def from_tier(cls, tier: RateLimitTier) -> RateLimitPolicy:
        """Erstellt Policy basierend auf Tier."""
        if tier == RateLimitTier.FREE:
            return cls(
                requests_per_minute=BASIC_TIER_REQUESTS_PER_MINUTE // 2,
                requests_per_hour=BASIC_TIER_REQUESTS_PER_HOUR // 2,
                requests_per_day=BASIC_TIER_REQUESTS_PER_DAY // 2,
                burst_size=BASIC_TIER_BURST_SIZE // 2,
                requests_per_second=FREE_TIER_REQUESTS_PER_SECOND,
                frames_per_second=FREE_TIER_FRAMES_PER_SECOND,
                max_concurrent_streams=FREE_TIER_MAX_CONCURRENT_STREAMS,
                max_stream_duration_seconds=FREE_TIER_MAX_STREAM_DURATION_SECONDS,
                identification_strategy=IdentificationStrategy.IP_ADDRESS
            )
        if tier == RateLimitTier.BASIC:
            return cls(
                requests_per_minute=BASIC_TIER_REQUESTS_PER_MINUTE,
                requests_per_hour=BASIC_TIER_REQUESTS_PER_HOUR,
                requests_per_day=BASIC_TIER_REQUESTS_PER_DAY,
                burst_size=BASIC_TIER_BURST_SIZE,
                requests_per_second=BASIC_TIER_REQUESTS_PER_SECOND,
                frames_per_second=BASIC_TIER_FRAMES_PER_SECOND,
                max_concurrent_streams=BASIC_TIER_MAX_CONCURRENT_STREAMS,
                max_stream_duration_seconds=BASIC_TIER_MAX_STREAM_DURATION_SECONDS,
                identification_strategy=IdentificationStrategy.USER_ID
            )
        if tier == RateLimitTier.PREMIUM:
            return cls(
                requests_per_minute=PREMIUM_TIER_REQUESTS_PER_MINUTE,
                requests_per_hour=PREMIUM_TIER_REQUESTS_PER_HOUR,
                requests_per_day=PREMIUM_TIER_REQUESTS_PER_DAY,
                burst_size=PREMIUM_TIER_BURST_SIZE,
                requests_per_second=PREMIUM_TIER_REQUESTS_PER_SECOND,
                frames_per_second=PREMIUM_TIER_FRAMES_PER_SECOND,
                max_concurrent_streams=PREMIUM_TIER_MAX_CONCURRENT_STREAMS,
                max_stream_duration_seconds=PREMIUM_TIER_MAX_STREAM_DURATION_SECONDS,
                identification_strategy=IdentificationStrategy.API_KEY
            )
        if tier == RateLimitTier.ENTERPRISE:
            return cls(
                requests_per_minute=ENTERPRISE_TIER_REQUESTS_PER_MINUTE,
                requests_per_hour=ENTERPRISE_TIER_REQUESTS_PER_HOUR,
                requests_per_day=ENTERPRISE_TIER_REQUESTS_PER_DAY,
                burst_size=ENTERPRISE_TIER_BURST_SIZE,
                requests_per_second=ENTERPRISE_TIER_REQUESTS_PER_SECOND,
                frames_per_second=ENTERPRISE_TIER_FRAMES_PER_SECOND,
                max_concurrent_streams=ENTERPRISE_TIER_MAX_CONCURRENT_STREAMS,
                max_stream_duration_seconds=ENTERPRISE_TIER_MAX_STREAM_DURATION_SECONDS,
                identification_strategy=IdentificationStrategy.TENANT_ID
            )
        if tier == RateLimitTier.ADMIN:
            return cls(
                requests_per_minute=ADMIN_TIER_REQUESTS_PER_MINUTE,
                requests_per_hour=ADMIN_TIER_REQUESTS_PER_HOUR,
                requests_per_day=ADMIN_TIER_REQUESTS_PER_DAY,
                burst_size=ADMIN_TIER_BURST_SIZE,
                requests_per_second=ENTERPRISE_TIER_REQUESTS_PER_SECOND * 2,
                frames_per_second=ENTERPRISE_TIER_FRAMES_PER_SECOND * 2,
                max_concurrent_streams=ENTERPRISE_TIER_MAX_CONCURRENT_STREAMS * 2,
                max_stream_duration_seconds=ENTERPRISE_TIER_MAX_STREAM_DURATION_SECONDS * 2,
                identification_strategy=IdentificationStrategy.TENANT_ID
            )
        # UNLIMITED
        return cls(
            requests_per_minute=UNLIMITED_TIER_REQUESTS_PER_SECOND * 60,
            requests_per_hour=UNLIMITED_TIER_REQUESTS_PER_SECOND * 3600,
            requests_per_day=UNLIMITED_TIER_REQUESTS_PER_SECOND * 86400,
            burst_size=UNLIMITED_TIER_BURST_CAPACITY,
            requests_per_second=UNLIMITED_TIER_REQUESTS_PER_SECOND,
            frames_per_second=UNLIMITED_TIER_FRAMES_PER_SECOND,
            max_concurrent_streams=UNLIMITED_TIER_MAX_CONCURRENT_STREAMS,
            max_stream_duration_seconds=UNLIMITED_TIER_MAX_STREAM_DURATION_SECONDS,
            identification_strategy=IdentificationStrategy.TENANT_ID,
            enabled=False  # Praktisch unbegrenzt
        )


@dataclass
class TierConfiguration:
    """Konfiguration für Rate Limit Tier."""

    tier: RateLimitTier
    policy: RateLimitPolicy = field(default_factory=lambda: RateLimitPolicy())

    # Operation-spezifische Policies
    operation_policies: dict[str, RateLimitPolicy] = field(default_factory=dict)

    # Whitelist-Features
    ip_whitelist: list[str] = field(default_factory=list)
    bypass_rate_limits: bool = False

    # Tenant-spezifische Konfiguration
    api_keys: list[str] = field(default_factory=list)
    enabled: bool = True
    max_users: int | None = None
    features: list[str] = field(default_factory=list)
    quotas: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-Initialisierung für Default-Policy."""
        if not hasattr(self.policy, "tier") or self.policy == RateLimitPolicy():
            self.policy = RateLimitPolicy.from_tier(self.tier)

    def get_policy(self, operation: str = "default") -> RateLimitPolicy:
        """Gibt Policy für Operation zurück."""
        return self.operation_policies.get(operation, self.policy)


class UnifiedRateLimitConfig(BaseModel):
    """Einheitliche Rate-Limiting-Konfiguration."""

    # Backend-Konfiguration
    backend: RateLimitBackend = Field(
        default=RateLimitBackend.HYBRID,
        description="Rate Limiting Backend"
    )

    # Redis-Konfiguration
    redis_host: str = Field(default=DEFAULT_REDIS_HOST, description="Redis Host")
    redis_port: int = Field(default=DEFAULT_REDIS_PORT, description="Redis Port")
    redis_db: int = Field(default=DEFAULT_REDIS_DB, description="Redis Database")
    redis_password: str | None = Field(default=None, description="Redis Password")
    redis_ssl: bool = Field(default=False, description="Redis SSL")
    redis_timeout: float = Field(default=DEFAULT_REDIS_TIMEOUT_SECONDS, description="Redis Timeout")

    # Fallback-Konfiguration
    memory_fallback_enabled: bool = Field(
        default=True,
        description="Memory-Fallback bei Redis-Ausfall"
    )

    # Tier-Konfigurationen
    tiers: dict[RateLimitTier, TierConfiguration] = Field(
        default_factory=dict,
        description="Tier-spezifische Konfigurationen"
    )

    # Globale Einstellungen
    default_tier: RateLimitTier = Field(
        default=RateLimitTier.BASIC,
        description="Standard-Tier für unbekannte API-Keys"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Prometheus-Metriken aktivieren")
    enable_logging: bool = Field(default=True, description="Rate Limit Logging aktivieren")

    # Headers
    include_rate_limit_headers: bool = Field(
        default=True,
        description="Rate Limit Headers in Responses"
    )

    # Cleanup
    cleanup_interval_seconds: int = Field(
        default=DEFAULT_CLEANUP_INTERVAL_SECONDS,
        description="Cleanup-Intervall für abgelaufene Einträge"
    )

    # IP-basierte Limits (Fallback)
    ip_based_fallback: bool = Field(
        default=True,
        description="IP-basierte Limits wenn kein API-Key"
    )

    ip_requests_per_minute: int = Field(
        default=DEFAULT_IP_REQUESTS_PER_MINUTE,
        description="IP-basierte Requests pro Minute"
    )

    # Konfigurationsdateien
    config_dir: str = Field(
        default="backend/config/rate_limits",
        description="Verzeichnis für Konfigurationsdateien"
    )

    @field_validator("tiers", mode="before")
    def setup_default_tiers(cls, v: dict) -> dict:
        """Setzt Standard-Tier-Konfigurationen."""
        if not v:
            v = {}

        # Standard-Tiers erstellen falls nicht vorhanden
        for tier in RateLimitTier:
            if tier not in v:
                v[tier] = TierConfiguration(tier=tier)

        return v

    def get_tier_config(self, tier: RateLimitTier) -> TierConfiguration:
        """Gibt Tier-Konfiguration zurück."""
        return self.tiers.get(tier, self.tiers[self.default_tier])

    def get_api_key_tier(self, api_key: str) -> RateLimitTier | None:
        """Ermittelt Tier für API-Key."""
        for tier, config in self.tiers.items():
            if api_key in config.api_keys:
                return tier
        return None


def load_unified_rate_limit_config() -> UnifiedRateLimitConfig:
    """Lädt einheitliche Rate Limit Konfiguration aus Umgebungsvariablen.

    Returns:
        UnifiedRateLimitConfig-Instanz
    """
    # Redis-Konfiguration laden
    redis_config = get_redis_config("KEI_MCP_REDIS_")

    # Basis-Konfiguration
    config_data = {
        "backend": get_env_enum("KEI_MCP_RATE_LIMIT_BACKEND", RateLimitBackend, RateLimitBackend.HYBRID),
        "redis_host": redis_config["host"],
        "redis_port": redis_config["port"],
        "redis_db": redis_config["db"],
        "redis_password": redis_config["password"],
        "redis_ssl": redis_config["ssl"],
        "redis_timeout": redis_config["timeout"],
        "memory_fallback_enabled": get_env_bool("KEI_MCP_MEMORY_FALLBACK", True),
        "default_tier": get_env_enum("KEI_MCP_DEFAULT_TIER", RateLimitTier, RateLimitTier.BASIC),
        "enable_metrics": get_env_bool("KEI_MCP_RATE_LIMIT_METRICS", True),
        "enable_logging": get_env_bool("KEI_MCP_RATE_LIMIT_LOGGING", True),
        "include_rate_limit_headers": get_env_bool("KEI_MCP_RATE_LIMIT_HEADERS", True),
        "cleanup_interval_seconds": get_env_int("KEI_MCP_CLEANUP_INTERVAL", DEFAULT_CLEANUP_INTERVAL_SECONDS),
        "ip_based_fallback": get_env_bool("KEI_MCP_IP_FALLBACK", True),
        "ip_requests_per_minute": get_env_int("KEI_MCP_IP_REQUESTS_PER_MINUTE", DEFAULT_IP_REQUESTS_PER_MINUTE),
        "config_dir": get_env_str("KEI_MCP_CONFIG_DIR", "backend/config/rate_limits")
    }

    config = UnifiedRateLimitConfig(**config_data)

    # Tier-spezifische Konfiguration aus Umgebungsvariablen laden
    _load_tier_configs_from_env(config)

    # Konfigurationsdateien laden falls vorhanden
    _load_config_files(config)

    logger.info(f"Einheitliche Rate Limit Konfiguration geladen: "
               f"backend={config.backend}, "
               f"redis={config.redis_host}:{config.redis_port}, "
               f"default_tier={config.default_tier}, "
               f"tiers={len(config.tiers)}")

    return config


def _load_tier_configs_from_env(config: UnifiedRateLimitConfig) -> None:
    """Lädt Tier-spezifische Konfigurationen aus Umgebungsvariablen."""
    for tier in RateLimitTier:
        tier_prefix = f"KEI_MCP_TIER_{tier.value.upper()}"

        # Tier-Konfiguration erstellen falls nicht vorhanden
        if tier not in config.tiers:
            config.tiers[tier] = TierConfiguration(tier=tier)

        tier_config = config.tiers[tier]

        # Globale Tier-Limits
        if env_val := os.getenv(f"{tier_prefix}_REQUESTS_PER_MINUTE"):
            tier_config.policy.requests_per_minute = int(env_val)

        if env_val := os.getenv(f"{tier_prefix}_REQUESTS_PER_HOUR"):
            tier_config.policy.requests_per_hour = int(env_val)

        if env_val := os.getenv(f"{tier_prefix}_REQUESTS_PER_DAY"):
            tier_config.policy.requests_per_day = int(env_val)

        # Streaming-Limits
        if env_val := os.getenv(f"{tier_prefix}_REQUESTS_PER_SECOND"):
            tier_config.policy.requests_per_second = float(env_val)

        if env_val := os.getenv(f"{tier_prefix}_FRAMES_PER_SECOND"):
            tier_config.policy.frames_per_second = float(env_val)

        # Bypass-Einstellungen
        if env_val := os.getenv(f"{tier_prefix}_BYPASS_RATE_LIMITS"):
            tier_config.bypass_rate_limits = env_val.lower() == "true"

        # IP-Whitelist
        if env_val := os.getenv(f"{tier_prefix}_IP_WHITELIST"):
            tier_config.ip_whitelist = [ip.strip() for ip in env_val.split(",") if ip.strip()]

        # API-Keys
        if env_val := os.getenv(f"{tier_prefix}_API_KEYS"):
            tier_config.api_keys = [key.strip() for key in env_val.split(",") if key.strip()]


def _load_config_files(config: UnifiedRateLimitConfig) -> None:
    """Lädt Konfiguration aus YAML-Dateien."""
    config_dir = Path(config.config_dir)
    if not config_dir.exists():
        logger.debug(f"Konfigurationsverzeichnis nicht gefunden: {config_dir}")
        return

    # Lade Tenant-Konfigurationen
    tenant_file = config_dir / "tenants.yaml"
    if tenant_file.exists():
        _load_tenant_config_file(config, tenant_file)

    # Lade Endpoint-Konfigurationen
    endpoint_file = config_dir / "endpoints.yaml"
    if endpoint_file.exists():
        _load_endpoint_config_file(config, endpoint_file)


def _load_tenant_config_file(config: UnifiedRateLimitConfig, config_file: Path) -> None:
    """Lädt Tenant-Konfigurationen aus YAML-Datei."""
    try:
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for tenant_data in data.get("tenants", []):
            tenant_id = tenant_data.get("tenant_id")
            tier_str = tenant_data.get("tier", "basic")

            try:
                tier = RateLimitTier(tier_str)
            except ValueError:
                logger.warning(f"Ungültiges Tier für Tenant {tenant_id}: {tier_str}")
                continue

            if tier not in config.tiers:
                config.tiers[tier] = TierConfiguration(tier=tier)

            tier_config = config.tiers[tier]

            # API-Keys hinzufügen
            api_keys = tenant_data.get("api_keys", [])
            tier_config.api_keys.extend(api_keys)

            # Weitere Konfiguration
            tier_config.enabled = tenant_data.get("enabled", True)
            tier_config.max_users = tenant_data.get("max_users")
            tier_config.features = tenant_data.get("features", [])
            tier_config.quotas = tenant_data.get("quotas", {})

            # Custom Limits
            if custom_limits := tenant_data.get("custom_limits"):
                _apply_custom_limits(tier_config.policy, custom_limits)

    except Exception as e:
        logger.exception(f"Fehler beim Laden der Tenant-Konfiguration: {e}")


def _load_endpoint_config_file(config: UnifiedRateLimitConfig, config_file: Path) -> None:
    """Lädt Endpoint-Konfigurationen aus YAML-Datei."""
    try:
        with open(config_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for endpoint_data in data.get("endpoints", []):
            pattern = endpoint_data.get("pattern")
            endpoint_config = endpoint_data.get("config", {})

            # Erstelle Policy für Endpoint
            policy = RateLimitPolicy()
            _apply_custom_limits(policy, endpoint_config)

            # Füge zu allen relevanten Tiers hinzu
            for tier_config in config.tiers.values():
                tier_config.operation_policies[pattern] = policy

    except Exception as e:
        logger.exception(f"Fehler beim Laden der Endpoint-Konfiguration: {e}")


def _apply_custom_limits(policy: RateLimitPolicy, limits: dict[str, Any]) -> None:
    """Wendet Custom Limits auf Policy an."""
    for key, value in limits.items():
        if hasattr(policy, key):
            setattr(policy, key, value)


# Globale Konfiguration-Instanz
_unified_rate_limit_config: UnifiedRateLimitConfig | None = None


def get_unified_rate_limit_config() -> UnifiedRateLimitConfig:
    """Gibt aktuelle einheitliche Rate Limit Konfiguration zurück.

    Returns:
        Aktuelle UnifiedRateLimitConfig-Instanz
    """
    global _unified_rate_limit_config
    if _unified_rate_limit_config is None:
        _unified_rate_limit_config = load_unified_rate_limit_config()
    return _unified_rate_limit_config


def reload_unified_rate_limit_config() -> UnifiedRateLimitConfig:
    """Lädt einheitliche Rate Limit Konfiguration neu.

    Returns:
        Neue UnifiedRateLimitConfig-Instanz
    """
    global _unified_rate_limit_config

    logger.info("Lade einheitliche Rate Limit Konfiguration neu...")
    new_config = load_unified_rate_limit_config()

    _unified_rate_limit_config = new_config
    return new_config


# Backward Compatibility Alias
RateLimitConfig = UnifiedRateLimitConfig


def get_rate_limit_config() -> UnifiedRateLimitConfig:
    """Backward compatibility function für get_rate_limit_config.

    Returns:
        Aktuelle UnifiedRateLimitConfig-Instanz
    """
    return get_unified_rate_limit_config()


__all__ = [
    "EndpointType",
    "IdentificationStrategy",
    "RateLimitAlgorithm",
    "RateLimitBackend",
    "RateLimitConfig",  # Backward compatibility
    "RateLimitPolicy",
    "RateLimitTier",
    "TierConfiguration",
    "UnifiedRateLimitConfig",
    "get_rate_limit_config",  # Backward compatibility
    "get_unified_rate_limit_config",
    "load_unified_rate_limit_config",
    "reload_unified_rate_limit_config",
]
