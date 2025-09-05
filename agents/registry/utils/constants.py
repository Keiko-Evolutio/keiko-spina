# backend/agents/registry/utils/constants.py
"""Konstanten für das Registry-System.

Konsolidiert alle Magic Numbers und Hard-coded Strings in semantisch benannte Konstanten.
"""

from datetime import timedelta
from typing import Final


class MatchingConstants:
    """Konstanten für Agent-Matching-Algorithmen."""

    # Match-Score-Gewichtungen
    TEXT_MATCH_WEIGHT: Final[float] = 0.8
    CAPABILITY_MATCH_WEIGHT: Final[float] = 0.6
    CATEGORY_MATCH_WEIGHT: Final[float] = 0.4

    # Score-Grenzen
    MIN_MATCH_SCORE: Final[float] = 0.0
    MAX_MATCH_SCORE: Final[float] = 1.0

    # Default-Werte
    DEFAULT_MATCH_LIMIT: Final[int] = 10
    MIN_VIABLE_SCORE: Final[float] = 0.1


class HealthThresholds:
    """Konstanten für Health-Monitoring."""

    # Health-Score-Grenzen
    MIN_HEALTH_SCORE: Final[float] = 0.7
    CRITICAL_HEALTH_SCORE: Final[float] = 0.3
    EXCELLENT_HEALTH_SCORE: Final[float] = 0.95

    # Failure-Thresholds
    MAX_FAILURE_COUNT: Final[int] = 3
    UNHEALTHY_THRESHOLD: Final[int] = 3

    # Zeitintervalle
    DEFAULT_HEALTH_CHECK_INTERVAL: Final[timedelta] = timedelta(seconds=30)
    RECOVERY_TIMEOUT: Final[timedelta] = timedelta(minutes=5)
    HEALTH_HISTORY_RETENTION: Final[timedelta] = timedelta(hours=24)


class LoadBalancingConstants:
    """Konstanten für Load-Balancing."""

    # Load-Factor-Grenzen
    MAX_LOAD_FACTOR: Final[float] = 0.9
    SCALE_UP_THRESHOLD: Final[float] = 0.8  # 80%
    SCALE_DOWN_THRESHOLD: Final[float] = 0.3  # 30%

    # Scaling-Faktoren
    SCALE_UP_MULTIPLIER: Final[float] = 1.2
    SCALE_DOWN_MULTIPLIER: Final[float] = 0.8

    # Default-Werte
    MIN_INSTANCES: Final[int] = 1
    DEFAULT_LOAD_FACTOR: Final[float] = 0.05
    DEFAULT_RESPONSE_TIME: Final[float] = 5.0
    RESPONSE_TIME_REDUCTION_FACTOR: Final[float] = 4.0


class CacheConstants:
    """Konstanten für Caching."""

    # Cache-Zeiten
    DEFAULT_CACHE_AGE: Final[timedelta] = timedelta(hours=1)
    AGENT_CACHE_TTL: Final[timedelta] = timedelta(minutes=30)
    DISCOVERY_CACHE_TTL: Final[timedelta] = timedelta(minutes=15)

    # Cache-Größen
    MAX_CACHE_SIZE: Final[int] = 1000
    DEFAULT_CACHE_SIZE: Final[int] = 100


class RolloutConstants:
    """Konstanten für Rollout-Management."""

    # Rollout-Zeitintervalle
    DEFAULT_MONITORING_INTERVAL: Final[int] = 30  # Sekunden
    CANARY_DURATION_MINUTES: Final[int] = 60
    BLUE_GREEN_SWITCH_DELAY: Final[int] = 5  # Minuten
    ROLLING_DELAY_SECONDS: Final[int] = 30

    # Rollout-Prozentsätze
    DEFAULT_CANARY_PERCENTAGE: Final[float] = 10.0
    SUCCESS_THRESHOLD_PERCENTAGE: Final[float] = 95.0

    # Traffic-Split-Schritte
    DEFAULT_TRAFFIC_SPLIT_STEPS: Final[list[int]] = [5, 10, 25, 50, 100]

    # Timeouts
    MAX_ROLLOUT_DURATION_MINUTES: Final[int] = 120


class AgentConstants:
    """Konstanten für Agent-Konfiguration."""

    # Agent-Kategorien
    VALID_CATEGORIES: Final[set[str]] = {
        "assistant", "specialist", "automation", "analysis",
        "communication", "integration", "orchestration", "custom"
    }

    # Capability-Keywords für automatische Erkennung
    CAPABILITY_KEYWORDS: Final[dict[str, list[str]]] = {
        "code_interpreter": ["code", "python", "script", "program"],
        "file_search": ["search", "file", "document", "find"],
        "function_calling": ["function", "api", "call", "tool"],
        "web_research": ["research", "web", "internet", "external", "sources", "links"],
        "image_generation": ["image", "dalle", "visual", "picture", "generate"],
        "conversation": ["chat", "conversation", "talk", "assistant"],
    }

    # Category-Keywords für automatische Kategorisierung
    CATEGORY_KEYWORDS: Final[dict[str, list[str]]] = {
        "assistant": ["assistant", "helper", "support"],
        "analysis": ["analyst", "research", "data", "analyze"],
        "automation": ["automation", "workflow", "process"],
        "specialist": ["specialist", "expert", "domain"],
    }


class ValidationConstants:
    """Konstanten für Validierung."""

    # String-Längen
    MAX_AGENT_NAME_LENGTH: Final[int] = 100
    MAX_DESCRIPTION_LENGTH: Final[int] = 500
    MAX_CAPABILITY_NAME_LENGTH: Final[int] = 50

    # Regex-Patterns
    AGENT_ID_PATTERN: Final[str] = r"^[a-zA-Z0-9_-]+$"
    VERSION_PATTERN: Final[str] = r"^\d+\.\d+\.\d+$"

    # Default-Werte
    DEFAULT_AGENT_NAME: Final[str] = "Unknown Agent"
    DEFAULT_AGENT_TYPE: Final[str] = "custom"


class ErrorConstants:
    """Konstanten für Fehlerbehandlung."""

    # Error-Codes
    AGENT_NOT_FOUND: Final[str] = "AGENT_NOT_FOUND"
    DUPLICATE_REGISTRATION: Final[str] = "DUPLICATE_REGISTRATION"
    REGISTRY_UNAVAILABLE: Final[str] = "REGISTRY_UNAVAILABLE"
    VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"

    # Retry-Konfiguration
    MAX_RETRY_ATTEMPTS: Final[int] = 3
    RETRY_DELAY_SECONDS: Final[float] = 1.0
    EXPONENTIAL_BACKOFF_FACTOR: Final[float] = 2.0
