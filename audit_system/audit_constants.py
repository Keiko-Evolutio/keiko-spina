# backend/audit_system/audit_constants.py
"""Zentrale Konstanten für das Audit-System.

Eliminiert Magic Numbers und Hard-coded Strings für bessere Wartbarkeit
und Konfigurierbarkeit des Audit-Systems.
"""

from __future__ import annotations

from typing import Final


class AuditConstants:
    """Zentrale Konstanten für das Audit-System."""

    # Cache-Konfiguration
    DEFAULT_CACHE_MAX_SIZE: Final[int] = 10_000
    DEFAULT_CACHE_TTL_SECONDS: Final[int] = 300  # 5 Minuten
    MIN_CACHE_SIZE: Final[int] = 100
    MAX_CACHE_SIZE: Final[int] = 100_000

    # Timing-Konfiguration
    DEFAULT_MONITORING_INTERVAL_SECONDS: Final[float] = 60.0  # 1 Minute
    DEFAULT_FLUSH_INTERVAL_SECONDS: Final[float] = 5.0
    DEFAULT_PROCESSING_TIMEOUT_SECONDS: Final[float] = 30.0

    # Performance-Schwellenwerte
    MAX_ERROR_RATE_PERCENT: Final[float] = 10.0
    MAX_PROCESSING_TIME_MS: Final[float] = 1000.0
    DEFAULT_BATCH_SIZE: Final[int] = 100
    MAX_BATCH_SIZE: Final[int] = 10_000

    # Retention-Konfiguration
    DEFAULT_RETENTION_DAYS: Final[int] = 90
    MIN_RETENTION_DAYS: Final[int] = 1
    MAX_RETENTION_DAYS: Final[int] = 2555  # ~7 Jahre

    # Dateigrößen (in Bytes)
    DEFAULT_MAX_FILE_SIZE: Final[int] = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE: Final[int] = 1 * 1024 * 1024  # 1MB
    MAX_FILE_SIZE: Final[int] = 1024 * 1024 * 1024  # 1GB

    # Backup-Konfiguration
    DEFAULT_BACKUP_COUNT: Final[int] = 10
    MIN_BACKUP_COUNT: Final[int] = 1
    MAX_BACKUP_COUNT: Final[int] = 100

    # String-Konstanten
    DEFAULT_CORRELATION_ID: Final[str] = "unknown"
    DEFAULT_ACTOR: Final[str] = "system"
    DEFAULT_CLIENT_IP: Final[str] = "127.0.0.1"

    # Compliance-Tags
    MIDDLEWARE_COMPLIANCE_TAG: Final[str] = "middleware"
    REQUEST_AUDIT_TAG: Final[str] = "request_audit"
    ACTION_LOGGING_TAG: Final[str] = "action_logging"
    COMPREHENSIVE_AUDIT_TAG: Final[str] = "comprehensive_audit"


class AuditPaths:
    """Standard-Pfade für das Audit-System."""

    # Ausgeschlossene Pfade für Middleware
    DEFAULT_EXCLUDED_PATHS: Final[list[str]] = [
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico"
    ]

    # Health-Check-Pfade
    HEALTH_CHECK_PATHS: Final[list[str]] = [
        "/health",
        "/healthz",
        "/ready"
    ]

    # Standard-Log-Verzeichnisse
    DEFAULT_LOG_DIR: Final[str] = "/var/log/kei-audit"
    DEFAULT_LOG_FILE: Final[str] = "audit.log"


class AuditAlertTypes:
    """Standard-Alert-Typen für das Audit-System."""

    HIGH_ERROR_RATE: Final[str] = "high_error_rate"
    SLOW_PROCESSING: Final[str] = "slow_processing"
    TAMPER_DETECTION: Final[str] = "tamper_detection"
    COMPLIANCE_VIOLATION: Final[str] = "compliance_violation"
    SECURITY_INCIDENT: Final[str] = "security_incident"
    SYSTEM_FAILURE: Final[str] = "system_failure"


class AuditMessages:
    """Standard-Nachrichten für das Audit-System."""

    # Erfolgs-Nachrichten
    ENGINE_STARTED: Final[str] = "Audit-Engine gestartet"
    ENGINE_STOPPED: Final[str] = "Audit-Engine gestoppt"
    LOGGER_STARTED: Final[str] = "Action Logger gestartet"
    LOGGER_STOPPED: Final[str] = "Action Logger gestoppt"
    MONITORING_STARTED: Final[str] = "Audit-Monitoring gestartet"
    MONITORING_STOPPED: Final[str] = "Audit-Monitoring gestoppt"

    # Fehler-Nachrichten
    PROCESSING_ERROR: Final[str] = "Fehler bei Event-Verarbeitung"
    MONITORING_ERROR: Final[str] = "Analytics-Monitoring-Fehler"
    AUDIT_FAILURE: Final[str] = "Audit-Compliance-Fehler"

    # Alert-Nachrichten
    HIGH_ERROR_RATE_TITLE: Final[str] = "High Audit Error Rate"
    SLOW_PROCESSING_TITLE: Final[str] = "Slow Audit Processing"

    # Beschreibungs-Templates
    ERROR_RATE_DESCRIPTION: Final[str] = "Error rate is {rate:.1f}%"
    PROCESSING_TIME_DESCRIPTION: Final[str] = "Average processing time is {time:.1f}ms"


class AuditEnvironmentVariables:
    """Umgebungsvariablen für das Audit-System."""

    # Log-Konfiguration
    LOG_DIR: Final[str] = "KEI_AUDIT_LOG_DIR"
    LOG_FILE: Final[str] = "KEI_AUDIT_LOG_FILE"
    LOG_LEVEL: Final[str] = "KEI_AUDIT_LOG_LEVEL"
    CONSOLE_LOGGING: Final[str] = "KEI_AUDIT_CONSOLE"

    # Performance-Konfiguration
    CACHE_SIZE: Final[str] = "KEI_AUDIT_CACHE_SIZE"
    BATCH_SIZE: Final[str] = "KEI_AUDIT_BATCH_SIZE"
    FLUSH_INTERVAL: Final[str] = "KEI_AUDIT_FLUSH_INTERVAL"

    # Feature-Flags
    ENABLE_TAMPER_PROOF: Final[str] = "KEI_AUDIT_TAMPER_PROOF"
    ENABLE_PII_REDACTION: Final[str] = "KEI_AUDIT_PII_REDACTION"
    ENABLE_MONITORING: Final[str] = "KEI_AUDIT_MONITORING"


__all__ = [
    "AuditAlertTypes",
    "AuditConstants",
    "AuditEnvironmentVariables",
    "AuditMessages",
    "AuditPaths"
]
