# backend/config/webhook_audit_config.py
"""Webhook Audit Konfiguration für Keiko Personal Assistant.

Stellt eine typsichere Konfiguration für Webhook-Audit-Funktionalitäten bereit.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class WebhookAuditConfig(BaseModel):
    """Konfiguration für Webhook-Audit-Funktionalitäten."""

    # Audit Aktivierung
    enabled: bool = Field(default=False)

    # Audit Level (DEBUG, INFO, WARNING, ERROR)
    level: str = Field(default="INFO")

    # Log-Verzeichnis
    log_dir: str = Field(default="logs")

    # Log-Datei Name
    log_file: str = Field(default="webhook_audit.log")

    # Retention in Tagen
    retention_days: int = Field(default=30, ge=1, le=365)

    # Maximale Log-Dateigröße in MB
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)

    # Anzahl der Backup-Dateien
    backup_count: int = Field(default=5, ge=1, le=50)


def get_webhook_audit_config() -> WebhookAuditConfig:
    """Lädt Webhook-Audit-Konfiguration aus Environment-Variablen.

    Returns:
        WebhookAuditConfig: Konfigurationsobjekt
    """
    return WebhookAuditConfig(
        enabled=os.getenv("KEI_WEBHOOK_AUDIT_ENABLED", "false").lower() == "true",
        level=os.getenv("KEI_WEBHOOK_AUDIT_LEVEL", "INFO").upper(),
        log_dir=os.getenv("KEI_WEBHOOK_AUDIT_LOG_DIR", "logs"),
        log_file=os.getenv("KEI_WEBHOOK_AUDIT_LOG_FILE", "webhook_audit.log"),
        retention_days=int(os.getenv("KEI_WEBHOOK_AUDIT_RETENTION_DAYS", "30")),
        max_file_size_mb=int(os.getenv("KEI_WEBHOOK_AUDIT_MAX_FILE_SIZE_MB", "100")),
        backup_count=int(os.getenv("KEI_WEBHOOK_AUDIT_BACKUP_COUNT", "5"))
    )


# Globale Instanz
webhook_audit_config: WebhookAuditConfig = get_webhook_audit_config()


__all__ = [
    "WebhookAuditConfig",
    "get_webhook_audit_config",
    "webhook_audit_config",
]
