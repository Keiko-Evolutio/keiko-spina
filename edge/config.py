"""Edge Computing Configuration für Keiko Personal Assistant.

Dieses Modul implementiert die Konfigurationsverwaltung für das Edge Computing-System.
"""

import os
from dataclasses import dataclass, field
from typing import Any

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
from .edge_types import LoadBalancingStrategy, LoadBalancingWeights

logger = get_logger(__name__)


def get_env_str(key: str, default: str = "") -> str:
    """Holt String-Wert aus Umgebungsvariablen."""
    return os.getenv(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    """Holt Integer-Wert aus Umgebungsvariablen."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Holt Float-Wert aus Umgebungsvariablen."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """Holt Boolean-Wert aus Umgebungsvariablen."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_list(key: str, default: list[str] | None = None) -> list[str]:
    """Holt Liste aus Umgebungsvariablen (komma-separiert)."""
    if default is None:
        default = []

    value = os.getenv(key, "")
    if not value:
        return default

    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class EdgeNodeConfig:
    """Konfiguration für Edge-Nodes."""

    # Node-Identifikation
    node_id: str = ""
    node_type: str = "audio-processor"
    region: str = "default"

    # Server-Konfiguration
    host: str = "0.0.0.0"
    port: int = 8080
    health_port: int = 8081
    workers: int = 2
    max_connections: int = 100

    # Registry-Verbindung
    registry_url: str = "http://localhost:8080"
    heartbeat_interval_seconds: int = 30
    registration_retry_attempts: int = 3

    # Capabilities
    supported_capabilities: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 10


@dataclass
class EdgeProcessingConfig:
    """Konfiguration für Edge-Processing."""

    # Task-Management
    max_concurrent_tasks_per_node: int = 10
    task_timeout_seconds: int = 30
    task_retry_attempts: int = 3
    task_queue_size: int = 1000

    # Audio-Processing
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    audio_buffer_size: int = 1024
    supported_audio_formats: list[str] = field(
        default_factory=lambda: ["float32", "int16", "int32"]
    )

    # Model-Management
    models_path: str = "/models"
    model_cache_size_mb: int = 512
    lazy_loading_enabled: bool = True
    preload_on_startup: bool = False


@dataclass
class EdgeLoadBalancingConfig:
    """Konfiguration für Load Balancing."""

    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    weights: LoadBalancingWeights = field(default_factory=LoadBalancingWeights)
    routing_cache_ttl_seconds: int = 60
    health_check_interval_seconds: int = 30
    failover_timeout_seconds: int = 5


@dataclass
class EdgeMonitoringConfig:
    """Konfiguration für Monitoring."""

    # Metriken-Sammlung
    metrics_collection_interval_seconds: int = 10
    metrics_retention_days: int = 7
    enable_detailed_metrics: bool = False

    # Alerting
    alert_evaluation_interval_seconds: int = 30
    enable_alerting: bool = True
    alert_thresholds: dict[str, float] = field(default_factory=lambda: {
        "cpu_usage_percent": 80.0,
        "memory_usage_percent": 85.0,
        "disk_usage_percent": 90.0,
        "error_rate_percent": 5.0
    })


@dataclass
class EdgeScalingConfig:
    """Konfiguration für Auto-Scaling."""

    # Auto-Scaling
    auto_scaling_enabled: bool = True
    scaling_evaluation_interval_seconds: int = 60
    scaling_cooldown_period_seconds: int = 300

    # Node-Limits
    min_nodes_global: int = 2
    max_nodes_global: int = 100
    min_nodes_per_region: int = 1
    max_nodes_per_region: int = 50

    # Scaling-Faktoren
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7

    # Thresholds
    scale_up_cpu_threshold: float = 80.0
    scale_down_cpu_threshold: float = 30.0
    scale_up_memory_threshold: float = 85.0
    scale_down_memory_threshold: float = 40.0


@dataclass
class EdgeSecurityConfig:
    """Konfiguration für Security."""

    # Authentication
    enable_authentication: bool = True
    api_key_required: bool = True
    api_key: str = ""

    # Encryption
    enable_encryption: bool = True
    tls_enabled: bool = True
    tls_cert_path: str = ""
    tls_key_path: str = ""

    # Input Validation
    enable_input_validation: bool = True
    max_input_size_mb: int = 10
    allowed_mime_types: list[str] = field(default_factory=lambda: [
        "audio/wav", "audio/mp3", "audio/flac", "application/octet-stream"
    ])


@dataclass
class EdgeCacheConfig:
    """Konfiguration für Caching."""

    # Cache-Einstellungen
    cache_enabled: bool = True
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 3600
    eviction_policy: str = "lru"

    # Cache-Typen
    enable_model_cache: bool = True
    enable_result_cache: bool = True
    enable_routing_cache: bool = True


@dataclass
class EdgeConfig:
    """Hauptkonfiguration für Edge Computing-System."""

    # Sub-Konfigurationen
    node: EdgeNodeConfig = field(default_factory=EdgeNodeConfig)
    processing: EdgeProcessingConfig = field(default_factory=EdgeProcessingConfig)
    load_balancing: EdgeLoadBalancingConfig = field(default_factory=EdgeLoadBalancingConfig)
    monitoring: EdgeMonitoringConfig = field(default_factory=EdgeMonitoringConfig)
    scaling: EdgeScalingConfig = field(default_factory=EdgeScalingConfig)
    security: EdgeSecurityConfig = field(default_factory=EdgeSecurityConfig)
    cache: EdgeCacheConfig = field(default_factory=EdgeCacheConfig)

    # Globale Einstellungen
    environment: str = "development"
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    @classmethod
    def from_env(cls) -> "EdgeConfig":
        """Erstellt EdgeConfig aus Umgebungsvariablen."""
        # Node-Konfiguration
        node_config = EdgeNodeConfig(
            node_id=get_env_str("EDGE_NODE_ID", "edge-node-1"),
            node_type=get_env_str("EDGE_NODE_TYPE", "audio-processor"),
            region=get_env_str("EDGE_NODE_REGION", "default"),
            host=get_env_str("EDGE_HOST", "0.0.0.0"),
            port=get_env_int("EDGE_PORT", 8080),
            health_port=get_env_int("EDGE_HEALTH_PORT", 8081),
            workers=get_env_int("EDGE_WORKERS", 2),
            max_connections=get_env_int("EDGE_MAX_CONNECTIONS", 100),
            registry_url=get_env_str("EDGE_REGISTRY_URL", "http://localhost:8080"),
            heartbeat_interval_seconds=get_env_int("EDGE_HEARTBEAT_INTERVAL", 30),
            registration_retry_attempts=get_env_int("EDGE_REGISTRATION_RETRY_ATTEMPTS", 3),
            supported_capabilities=get_env_list("EDGE_SUPPORTED_CAPABILITIES", ["audio-processing"]),
            max_concurrent_tasks=get_env_int("EDGE_MAX_CONCURRENT_TASKS", 10)
        )

        # Processing-Konfiguration
        processing_config = EdgeProcessingConfig(
            max_concurrent_tasks_per_node=get_env_int("EDGE_MAX_CONCURRENT_TASKS_PER_NODE", 10),
            task_timeout_seconds=get_env_int("EDGE_TASK_TIMEOUT_SECONDS", 30),
            task_retry_attempts=get_env_int("EDGE_TASK_RETRY_ATTEMPTS", 3),
            task_queue_size=get_env_int("EDGE_TASK_QUEUE_SIZE", 1000),
            audio_sample_rate=get_env_int("EDGE_AUDIO_SAMPLE_RATE", 48000),
            audio_channels=get_env_int("EDGE_AUDIO_CHANNELS", 1),
            audio_buffer_size=get_env_int("EDGE_AUDIO_BUFFER_SIZE", 1024),
            supported_audio_formats=get_env_list("EDGE_SUPPORTED_AUDIO_FORMATS", ["float32", "int16"]),
            models_path=get_env_str("EDGE_MODELS_PATH", "/models"),
            model_cache_size_mb=get_env_int("EDGE_MODEL_CACHE_SIZE_MB", 512),
            lazy_loading_enabled=get_env_bool("EDGE_LAZY_LOADING_ENABLED", True),
            preload_on_startup=get_env_bool("EDGE_PRELOAD_ON_STARTUP", False)
        )

        # Load Balancing-Konfiguration
        load_balancing_config = EdgeLoadBalancingConfig(
            strategy=LoadBalancingStrategy(get_env_str("EDGE_LOAD_BALANCING_STRATEGY", "adaptive")),
            routing_cache_ttl_seconds=get_env_int("EDGE_ROUTING_CACHE_TTL_SECONDS", 60),
            health_check_interval_seconds=get_env_int("EDGE_HEALTH_CHECK_INTERVAL_SECONDS", 30),
            failover_timeout_seconds=get_env_int("EDGE_FAILOVER_TIMEOUT_SECONDS", 5)
        )

        # Monitoring-Konfiguration
        monitoring_config = EdgeMonitoringConfig(
            metrics_collection_interval_seconds=get_env_int("EDGE_METRICS_COLLECTION_INTERVAL", 10),
            metrics_retention_days=get_env_int("EDGE_METRICS_RETENTION_DAYS", 7),
            enable_detailed_metrics=get_env_bool("EDGE_ENABLE_DETAILED_METRICS", False),
            alert_evaluation_interval_seconds=get_env_int("EDGE_ALERT_EVALUATION_INTERVAL", 30),
            enable_alerting=get_env_bool("EDGE_ENABLE_ALERTING", True)
        )

        # Scaling-Konfiguration
        scaling_config = EdgeScalingConfig(
            auto_scaling_enabled=get_env_bool("EDGE_AUTO_SCALING_ENABLED", True),
            scaling_evaluation_interval_seconds=get_env_int("EDGE_SCALING_EVALUATION_INTERVAL", 60),
            scaling_cooldown_period_seconds=get_env_int("EDGE_SCALING_COOLDOWN_PERIOD", 300),
            min_nodes_global=get_env_int("EDGE_MIN_NODES_GLOBAL", 2),
            max_nodes_global=get_env_int("EDGE_MAX_NODES_GLOBAL", 100),
            min_nodes_per_region=get_env_int("EDGE_MIN_NODES_PER_REGION", 1),
            max_nodes_per_region=get_env_int("EDGE_MAX_NODES_PER_REGION", 50),
            scale_up_factor=get_env_float("EDGE_SCALE_UP_FACTOR", 1.5),
            scale_down_factor=get_env_float("EDGE_SCALE_DOWN_FACTOR", 0.7),
            scale_up_cpu_threshold=get_env_float("EDGE_SCALE_UP_CPU_THRESHOLD", 80.0),
            scale_down_cpu_threshold=get_env_float("EDGE_SCALE_DOWN_CPU_THRESHOLD", 30.0),
            scale_up_memory_threshold=get_env_float("EDGE_SCALE_UP_MEMORY_THRESHOLD", 85.0),
            scale_down_memory_threshold=get_env_float("EDGE_SCALE_DOWN_MEMORY_THRESHOLD", 40.0)
        )

        # Security-Konfiguration
        security_config = EdgeSecurityConfig(
            enable_authentication=get_env_bool("EDGE_ENABLE_AUTHENTICATION", True),
            api_key_required=get_env_bool("EDGE_API_KEY_REQUIRED", True),
            api_key=get_env_str("EDGE_API_KEY", ""),
            enable_encryption=get_env_bool("EDGE_ENABLE_ENCRYPTION", True),
            tls_enabled=get_env_bool("EDGE_TLS_ENABLED", True),
            tls_cert_path=get_env_str("EDGE_TLS_CERT_PATH", ""),
            tls_key_path=get_env_str("EDGE_TLS_KEY_PATH", ""),
            enable_input_validation=get_env_bool("EDGE_ENABLE_INPUT_VALIDATION", True),
            max_input_size_mb=get_env_int("EDGE_MAX_INPUT_SIZE_MB", 10),
            allowed_mime_types=get_env_list("EDGE_ALLOWED_MIME_TYPES", [
                "audio/wav", "audio/mp3", "audio/flac", "application/octet-stream"
            ])
        )

        # Cache-Konfiguration
        cache_config = EdgeCacheConfig(
            cache_enabled=get_env_bool("EDGE_CACHE_ENABLED", True),
            cache_size_mb=get_env_int("EDGE_CACHE_SIZE_MB", 1024),
            cache_ttl_seconds=get_env_int("EDGE_CACHE_TTL_SECONDS", 3600),
            eviction_policy=get_env_str("EDGE_CACHE_EVICTION_POLICY", "lru"),
            enable_model_cache=get_env_bool("EDGE_ENABLE_MODEL_CACHE", True),
            enable_result_cache=get_env_bool("EDGE_ENABLE_RESULT_CACHE", True),
            enable_routing_cache=get_env_bool("EDGE_ENABLE_ROUTING_CACHE", True)
        )

        return cls(
            node=node_config,
            processing=processing_config,
            load_balancing=load_balancing_config,
            monitoring=monitoring_config,
            scaling=scaling_config,
            security=security_config,
            cache=cache_config,
            environment=get_env_str("EDGE_ENVIRONMENT", "development"),
            debug_mode=get_env_bool("EDGE_DEBUG_MODE", False),
            log_level=get_env_str("EDGE_LOG_LEVEL", "INFO"),
            enable_detailed_logging=get_env_bool("EDGE_ENABLE_DETAILED_LOGGING", False)
        )

    @classmethod
    def default(cls) -> "EdgeConfig":
        """Erstellt Standard-EdgeConfig."""
        return cls()

    def validate(self) -> list[str]:
        """Validiert die Konfiguration und gibt Fehler zurück."""
        errors = []

        # Node-Validierung
        if not self.node.node_id:
            errors.append("Node-ID ist erforderlich")

        if self.node.port <= 0 or self.node.port > 65535:
            errors.append("Ungültiger Port")

        # Processing-Validierung
        if self.processing.max_concurrent_tasks_per_node <= 0:
            errors.append("Max concurrent tasks muss größer als 0 sein")

        if self.processing.task_timeout_seconds <= 0:
            errors.append("Task timeout muss größer als 0 sein")

        # Scaling-Validierung
        if self.scaling.min_nodes_global > self.scaling.max_nodes_global:
            errors.append("Min nodes kann nicht größer als max nodes sein")

        return errors


def get_edge_config() -> EdgeConfig:
    """Holt Edge-Konfiguration aus Umgebungsvariablen oder gibt Standard zurück."""
    try:
        config = EdgeConfig.from_env()
        errors = config.validate()

        if errors:
            logger.warning(f"Konfigurationsfehler gefunden: {errors}")
            logger.info("Verwende Standard-Konfiguration")
            return EdgeConfig.default()

        logger.info("Edge-Konfiguration aus Umgebungsvariablen geladen")
        return config

    except Exception as e:
        logger.error(f"Fehler beim Laden der Edge-Konfiguration: {e}")
        logger.info("Verwende Standard-Konfiguration")
        return EdgeConfig.default()


def create_edge_config(**overrides: Any) -> EdgeConfig:
    """Erstellt Edge-Konfiguration mit Overrides.

    Args:
        **overrides: Konfiguration-Overrides

    Returns:
        EdgeConfig-Instanz
    """
    config = get_edge_config()

    # Overrides anwenden
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unbekannter Konfigurationsparameter: {key}")

    return config


def create_development_edge_config() -> EdgeConfig:
    """Erstellt Development-optimierte Edge-Konfiguration."""
    config = EdgeConfig.default()

    # Development-spezifische Overrides
    config.environment = "development"
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.enable_detailed_logging = True
    config.security.enable_authentication = False
    config.security.api_key_required = False
    config.scaling.auto_scaling_enabled = False

    logger.info("Development Edge-Konfiguration erstellt")
    return config


def create_production_edge_config() -> EdgeConfig:
    """Erstellt Production-optimierte Edge-Konfiguration."""
    config = get_edge_config()

    # Production-spezifische Overrides
    config.environment = "production"
    config.debug_mode = False
    config.log_level = "INFO"
    config.enable_detailed_logging = False
    config.security.enable_authentication = True
    config.security.api_key_required = True
    config.security.enable_encryption = True
    config.scaling.auto_scaling_enabled = True

    logger.info("Production Edge-Konfiguration erstellt")
    return config
