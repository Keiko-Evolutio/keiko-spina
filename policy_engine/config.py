# backend/policy_engine/config.py
"""Konfigurationsklassen für das Policy-Engine-Modul.

Stellt konfigurierbare Einstellungen für verschiedene Policy-Engine-Komponenten
zur Verfügung und ermöglicht Runtime-Konfiguration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from .constants import *
except ImportError:
    # Fallback für direkten Import
    from .constants import *


@dataclass
class CacheConfig:
    """Konfiguration für Caching-Verhalten."""

    enabled: bool = ENABLE_CACHING
    ttl_seconds: int = DEFAULT_CACHE_TTL
    max_size: int = DEFAULT_CACHE_SIZE

    # Spezifische Cache-TTLs
    policy_cache_ttl: int = POLICY_CACHE_TTL
    tool_access_cache_ttl: int = TOOL_ACCESS_CACHE_TTL
    redaction_cache_ttl: int = REDACTION_CACHE_TTL

    @classmethod
    def from_env(cls) -> CacheConfig:
        """Erstellt CacheConfig aus Umgebungsvariablen."""
        return cls(
            enabled=os.getenv("POLICY_CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("POLICY_CACHE_TTL", str(DEFAULT_CACHE_TTL))),
            max_size=int(os.getenv("POLICY_CACHE_SIZE", str(DEFAULT_CACHE_SIZE)))
        )


@dataclass
class DetectionConfig:
    """Konfiguration für Detection-Verhalten."""

    # Confidence-Schwellwerte
    confidence_low: float = CONFIDENCE_LOW
    confidence_medium: float = CONFIDENCE_MEDIUM
    confidence_high: float = CONFIDENCE_HIGH
    confidence_very_high: float = CONFIDENCE_VERY_HIGH
    confidence_certain: float = CONFIDENCE_CERTAIN

    # Risk-Schwellwerte
    risk_threshold_low: float = RISK_THRESHOLD_LOW
    risk_threshold_medium: float = RISK_THRESHOLD_MEDIUM
    risk_threshold_high: float = RISK_THRESHOLD_HIGH
    risk_threshold_critical: float = RISK_THRESHOLD_CRITICAL

    # Toxicity-spezifische Schwellwerte
    toxicity_threshold_low: float = TOXICITY_THRESHOLD_LOW
    toxicity_threshold_medium: float = TOXICITY_THRESHOLD_MEDIUM
    toxicity_threshold_high: float = TOXICITY_THRESHOLD_HIGH
    toxicity_threshold_unsafe: float = TOXICITY_THRESHOLD_UNSAFE

    @classmethod
    def from_env(cls) -> DetectionConfig:
        """Erstellt DetectionConfig aus Umgebungsvariablen."""
        return cls(
            confidence_low=float(os.getenv("DETECTION_CONFIDENCE_LOW", str(CONFIDENCE_LOW))),
            confidence_medium=float(os.getenv("DETECTION_CONFIDENCE_MEDIUM", str(CONFIDENCE_MEDIUM))),
            confidence_high=float(os.getenv("DETECTION_CONFIDENCE_HIGH", str(CONFIDENCE_HIGH))),
            risk_threshold_low=float(os.getenv("DETECTION_RISK_LOW", str(RISK_THRESHOLD_LOW))),
            risk_threshold_medium=float(os.getenv("DETECTION_RISK_MEDIUM", str(RISK_THRESHOLD_MEDIUM))),
            risk_threshold_high=float(os.getenv("DETECTION_RISK_HIGH", str(RISK_THRESHOLD_HIGH)))
        )


@dataclass
class RedactionConfig:
    """Konfiguration für Redaction-Verhalten."""

    # Standard-Masks
    default_mask: str = DEFAULT_MASK
    hash_prefix: str = HASH_PREFIX
    encrypt_prefix: str = ENCRYPT_PREFIX
    token_prefix: str = TOKEN_PREFIX

    # Hash-Konfiguration
    hash_length: int = DEFAULT_HASH_LENGTH
    short_hash_length: int = SHORT_HASH_LENGTH
    long_hash_length: int = LONG_HASH_LENGTH

    # Token-Konfiguration
    token_modulo: int = DEFAULT_TOKEN_MODULO
    large_token_modulo: int = LARGE_TOKEN_MODULO
    token_format_width: int = TOKEN_FORMAT_WIDTH

    # Encrypt-Konfiguration
    encrypt_modulo: int = DEFAULT_ENCRYPT_MODULO
    encrypt_format_width: int = ENCRYPT_FORMAT_WIDTH

    @classmethod
    def from_env(cls) -> RedactionConfig:
        """Erstellt RedactionConfig aus Umgebungsvariablen."""
        return cls(
            default_mask=os.getenv("REDACTION_DEFAULT_MASK", DEFAULT_MASK),
            hash_length=int(os.getenv("REDACTION_HASH_LENGTH", str(DEFAULT_HASH_LENGTH))),
            token_modulo=int(os.getenv("REDACTION_TOKEN_MODULO", str(DEFAULT_TOKEN_MODULO)))
        )


@dataclass
class PerformanceConfig:
    """Konfiguration für Performance-Monitoring."""

    enabled: bool = ENABLE_PERFORMANCE_MONITORING
    warning_threshold_ms: float = PERFORMANCE_WARNING_THRESHOLD
    error_threshold_ms: float = PERFORMANCE_ERROR_THRESHOLD

    # Timeouts
    default_timeout: int = DEFAULT_PROCESSING_TIMEOUT
    fast_timeout: int = FAST_PROCESSING_TIMEOUT
    slow_timeout: int = SLOW_PROCESSING_TIMEOUT

    # Batch-Verarbeitung
    default_batch_size: int = DEFAULT_BATCH_SIZE
    min_batch_size: int = MIN_BATCH_SIZE
    max_batch_size: int = MAX_BATCH_SIZE

    @classmethod
    def from_env(cls) -> PerformanceConfig:
        """Erstellt PerformanceConfig aus Umgebungsvariablen."""
        return cls(
            enabled=os.getenv("PERFORMANCE_MONITORING_ENABLED", "true").lower() == "true",
            warning_threshold_ms=float(os.getenv("PERFORMANCE_WARNING_THRESHOLD", str(PERFORMANCE_WARNING_THRESHOLD))),
            error_threshold_ms=float(os.getenv("PERFORMANCE_ERROR_THRESHOLD", str(PERFORMANCE_ERROR_THRESHOLD))),
            default_timeout=int(os.getenv("PROCESSING_TIMEOUT", str(DEFAULT_PROCESSING_TIMEOUT)))
        )


@dataclass
class ContentConfig:
    """Konfiguration für Content-Verarbeitung."""

    max_content_length: int = MAX_CONTENT_LENGTH
    max_snippet_length: int = MAX_SNIPPET_LENGTH
    max_description_length: int = MAX_DESCRIPTION_LENGTH

    # Pattern-Limits
    max_patterns_per_category: int = MAX_PATTERNS_PER_CATEGORY
    max_keywords_per_category: int = MAX_KEYWORDS_PER_CATEGORY

    @classmethod
    def from_env(cls) -> ContentConfig:
        """Erstellt ContentConfig aus Umgebungsvariablen."""
        return cls(
            max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", str(MAX_CONTENT_LENGTH))),
            max_snippet_length=int(os.getenv("MAX_SNIPPET_LENGTH", str(MAX_SNIPPET_LENGTH))),
            max_description_length=int(os.getenv("MAX_DESCRIPTION_LENGTH", str(MAX_DESCRIPTION_LENGTH)))
        )


@dataclass
class DataMinimizationConfig:
    """Konfiguration für Data-Minimization."""

    # Sampling-Konfiguration
    default_strata_count: int = DEFAULT_STRATA_COUNT
    min_strata_count: int = MIN_STRATA_COUNT
    max_strata_count: int = MAX_STRATA_COUNT

    # Sampling-Raten
    default_sampling_rate: float = DEFAULT_SAMPLING_RATE
    min_sampling_rate: float = MIN_SAMPLING_RATE
    max_sampling_rate: float = MAX_SAMPLING_RATE

    @classmethod
    def from_env(cls) -> DataMinimizationConfig:
        """Erstellt DataMinimizationConfig aus Umgebungsvariablen."""
        return cls(
            default_strata_count=int(os.getenv("DATA_MIN_STRATA_COUNT", str(DEFAULT_STRATA_COUNT))),
            default_sampling_rate=float(os.getenv("DATA_MIN_SAMPLING_RATE", str(DEFAULT_SAMPLING_RATE)))
        )


@dataclass
class FeatureConfig:
    """Konfiguration für Feature-Toggles."""

    enable_caching: bool = ENABLE_CACHING
    enable_circuit_breaker: bool = ENABLE_CIRCUIT_BREAKER
    enable_performance_monitoring: bool = ENABLE_PERFORMANCE_MONITORING
    enable_detailed_logging: bool = ENABLE_DETAILED_LOGGING
    enable_async_processing: bool = ENABLE_ASYNC_PROCESSING

    @classmethod
    def from_env(cls) -> FeatureConfig:
        """Erstellt FeatureConfig aus Umgebungsvariablen."""
        return cls(
            enable_caching=os.getenv("FEATURE_CACHING", "true").lower() == "true",
            enable_circuit_breaker=os.getenv("FEATURE_CIRCUIT_BREAKER", "true").lower() == "true",
            enable_performance_monitoring=os.getenv("FEATURE_PERFORMANCE_MONITORING", "true").lower() == "true",
            enable_detailed_logging=os.getenv("FEATURE_DETAILED_LOGGING", "false").lower() == "true",
            enable_async_processing=os.getenv("FEATURE_ASYNC_PROCESSING", "true").lower() == "true"
        )


@dataclass
class PolicyEngineConfig:
    """Hauptkonfiguration für Policy-Engine."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    data_minimization: DataMinimizationConfig = field(default_factory=DataMinimizationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @classmethod
    def from_env(cls) -> PolicyEngineConfig:
        """Erstellt PolicyEngineConfig aus Umgebungsvariablen."""
        return cls(
            cache=CacheConfig.from_env(),
            detection=DetectionConfig.from_env(),
            redaction=RedactionConfig.from_env(),
            performance=PerformanceConfig.from_env(),
            content=ContentConfig.from_env(),
            data_minimization=DataMinimizationConfig.from_env(),
            features=FeatureConfig.from_env()
        )

    @classmethod
    def default(cls) -> PolicyEngineConfig:
        """Erstellt Standard-PolicyEngineConfig."""
        return cls()

    def validate(self) -> list[str]:
        """Validiert Konfiguration und gibt Liste von Fehlern zurück."""
        errors = []

        # Validiere Detection-Schwellwerte
        if not (0.0 <= self.detection.confidence_low <= 1.0):
            errors.append("confidence_low muss zwischen 0.0 und 1.0 liegen")

        if not (0.0 <= self.detection.risk_threshold_low <= 1.0):
            errors.append("risk_threshold_low muss zwischen 0.0 und 1.0 liegen")

        # Validiere Cache-Konfiguration
        if self.cache.ttl_seconds <= 0:
            errors.append("cache_ttl_seconds muss positiv sein")

        if self.cache.max_size <= 0:
            errors.append("cache_max_size muss positiv sein")

        # Validiere Performance-Konfiguration
        if self.performance.warning_threshold_ms <= 0:
            errors.append("performance_warning_threshold muss positiv sein")

        if self.performance.default_timeout <= 0:
            errors.append("default_timeout muss positiv sein")

        # Validiere Content-Limits
        if self.content.max_content_length <= 0:
            errors.append("max_content_length muss positiv sein")

        # Validiere Data-Minimization
        if not (0.0 < self.data_minimization.default_sampling_rate <= 1.0):
            errors.append("default_sampling_rate muss zwischen 0.0 und 1.0 liegen")

        return errors


# Globale Standard-Konfiguration
default_config = PolicyEngineConfig.default()

# Funktion zum Laden der Konfiguration
def load_config() -> PolicyEngineConfig:
    """Lädt Konfiguration aus Umgebungsvariablen oder gibt Standard zurück."""
    try:
        policy_config = PolicyEngineConfig.from_env()
        errors = policy_config.validate()

        if errors:
            return PolicyEngineConfig.default()

        return policy_config
    except Exception:
        return PolicyEngineConfig.default()


# Globale Konfigurationsinstanz
config = load_config()
