"""Webhook-spezifisches Audit-Logging.

Dieses Modul stellt einen dedizierten, asynchronen Audit-Logger für das
KEI-Webhook Subsystem bereit. Es erzeugt strukturierte JSON-Logs mit
Korrelations- und Domänenfeldern und integriert PII-/DLP-Redaktion.

Konfiguration (ENV):
  - KEI_WEBHOOK_AUDIT_ENABLED: "true|false" (Default: true)
  - KEI_WEBHOOK_AUDIT_LEVEL: "info|warning|error|debug" (Default: info)
  - KEI_WEBHOOK_AUDIT_LOG_DIR: Pfad (Default: "/var/log/kei")
  - KEI_WEBHOOK_AUDIT_LOG_FILE: Dateiname (Default: "webhook_audit.log")
  - KEI_WEBHOOK_AUDIT_MAX_FILE_SIZE: Bytes (Default: 52428800 = 50MB)
  - KEI_WEBHOOK_AUDIT_BACKUP_COUNT: Anzahl Rotationsdateien (Default: 10)
  - KEI_WEBHOOK_AUDIT_RETENTION_DAYS: Tage im Event-Feld (Default: 90)
  - KEI_WEBHOOK_AUDIT_ASYNC: "true|false" (Default: true)

Hinweis: Die tatsächliche Dateilöschung richtet sich nach Rotation (backup_count).
Das Feld `retention_days` wird als Hinweis für externe Systeme mitgeschrieben.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import logging.handlers
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from kei_logging import get_logger
from kei_logging.pii_redaction import PIIRedactionConfig, load_pii_config, redact_structure

from .secret_manager import get_secret_manager

logger = get_logger(__name__)


def redact_details(details: dict[str, Any], path_hint: str | None = None) -> dict[str, Any]:
    """Redaktiert PII in Details-Dictionary.

    Args:
        details: Details-Dictionary zum Redaktieren
        path_hint: Optionaler Pfad-Hinweis (wird derzeit nicht verwendet)

    Returns:
        Redaktiertes Details-Dictionary
    """
    return redact_structure(details)


class WebhookAuditEventType(str, Enum):
    """Webhook-Audit-Eventtypen für Nachvollziehbarkeit."""

    INBOUND_RECEIVED = "inbound_received"
    INBOUND_VALIDATED = "inbound_validated"
    INBOUND_REJECTED = "inbound_rejected"

    OUTBOUND_ENQUEUED = "outbound_enqueued"
    OUTBOUND_DELIVERED = "outbound_delivered"
    OUTBOUND_FAILED = "outbound_failed"
    OUTBOUND_RETRIED = "outbound_retried"

    TARGET_CREATED = "target_created"
    TARGET_UPDATED = "target_updated"
    TARGET_DELETED = "target_deleted"

    DLQ_MOVE = "dlq_move"
    DLQ_REQUEUE = "dlq_requeue"
    DLQ_PURGE = "dlq_purge"

    SECURITY_INVALID_SIGNATURE = "security_invalid_signature"
    SECURITY_REPLAY_ATTACK = "security_replay_attack"
    SECURITY_RATE_LIMIT_EXCEEDED = "security_rate_limit_exceeded"


class WebhookAuditOperation(str, Enum):
    """Operationen, die im Feld `operation` erfasst werden."""

    RECEIVE = "receive"
    VALIDATE = "validate"
    REJECT = "reject"
    ENQUEUE = "enqueue"
    DELIVER = "deliver"
    RETRY = "retry"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MOVE_TO_DLQ = "move_to_dlq"
    REQUEUE_DLQ = "requeue_dlq"
    PURGE_DLQ = "purge_dlq"


class WebhookAuditResult(str, Enum):
    """Ergebnisstatus des Audit-Events."""

    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"


@dataclass
class WebhookAuditRecord:
    """Strukturiertes Audit-Record-Modell für Webhook-Events.

    Attributes:
        event_type: Typ des Audit-Events (z. B. inbound_received)
        correlation_id: Korrelation zur Anfragennachverfolgung
        timestamp: ISO-8601 Zeitstempel in UTC
        operation: Fachliche Operation (create, update, deliver, ...)
        result: Ergebnis (success/failure/retry)
        user_id: Authentifizierter Benutzer (falls verfügbar)
        tenant_id: Mandant/Namensraum (falls verfügbar)
        delivery_id: Zustell-ID (Outbound)
        target_id: Ziel-ID
        error_details: Strukturierte Fehlerinformationen
        details: Zusätzliche kontextspezifische Daten (werden redaktiert)
        sensitive_data_redacted: Kennzeichen, ob Redaction angewandt wurde
        retention_days: Empfohlene Aufbewahrungsdauer
    """

    event_type: WebhookAuditEventType
    correlation_id: str
    timestamp: str
    operation: WebhookAuditOperation
    result: WebhookAuditResult
    user_id: str | None = None
    tenant_id: str | None = None
    delivery_id: str | None = None
    target_id: str | None = None
    error_details: dict[str, Any] | None = None
    details: dict[str, Any] = None  # type: ignore[assignment]
    sensitive_data_redacted: bool = False
    retention_days: int = 90

    def to_json(self, pii_cfg: PIIRedactionConfig | None = None, path_hint: str | None = None) -> str:
        """Serialisiert das Record zu JSON und redaktiert Details.

        Args:
            pii_cfg: PII-Redaktionskonfiguration
            path_hint: Optionaler Pfad für feldbasierte Redaction

        Returns:
            JSON-Zeile als str
        """
        # Details redaktieren
        safe_details = redact_details(self.details or {}, path_hint)
        masked = _contains_mask(safe_details, pii_cfg or load_pii_config())

        obj = asdict(self)
        obj["details"] = safe_details
        obj["sensitive_data_redacted"] = masked or self.sensitive_data_redacted
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def compute_signature(self, key: str) -> str:
        """Berechnet eine HMAC‑Signatur über das JSON‑Payload.

        Args:
            key: Geheimschlüssel

        Returns:
            hexadezimale HMAC Signatur
        """
        payload = json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hmac.new(key.encode("utf-8"), payload, sha256).hexdigest()


def _contains_mask(data: Any, pii_cfg: PIIRedactionConfig) -> bool:
    """Prüft rekursiv, ob ein Maskenwert in der Struktur vorkommt.

    Args:
        data: Beliebige strukturierte Daten
        pii_cfg: PII-Config mit Maskenwert

    Returns:
        True, falls Maskenwert entdeckt wurde
    """
    try:
        if data is None:
            return False
        if isinstance(data, str):
            return pii_cfg.mask in data
        if isinstance(data, dict):
            return any(_contains_mask(v, pii_cfg) for v in data.values())
        if isinstance(data, list | tuple):
            return any(_contains_mask(v, pii_cfg) for v in data)
        return False
    except Exception:
        return False


@dataclass
class WebhookAuditConfig:
    """Konfiguration für Webhook-Audit-Logging."""

    enabled: bool
    level: int
    log_dir: str
    log_file: str
    max_file_size: int
    backup_count: int
    retention_days: int
    async_processing: bool

    @staticmethod
    def load() -> WebhookAuditConfig:
        """Lädt Konfiguration aus ENV mit Defaults."""
        enabled = os.getenv("KEI_WEBHOOK_AUDIT_ENABLED", "true").lower() == "true"
        level_str = os.getenv("KEI_WEBHOOK_AUDIT_LEVEL", "info").lower()
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        level = level_map.get(level_str, logging.INFO)
        log_dir = os.getenv("KEI_WEBHOOK_AUDIT_LOG_DIR", "/var/log/kei")
        log_file = os.getenv("KEI_WEBHOOK_AUDIT_LOG_FILE", "webhook_audit.log")
        try:
            max_file_size = int(os.getenv("KEI_WEBHOOK_AUDIT_MAX_FILE_SIZE", str(50 * 1024 * 1024)))
        except Exception:
            max_file_size = 50 * 1024 * 1024
        try:
            backup_count = int(os.getenv("KEI_WEBHOOK_AUDIT_BACKUP_COUNT", "10"))
        except Exception:
            backup_count = 10
        try:
            retention_days = int(os.getenv("KEI_WEBHOOK_AUDIT_RETENTION_DAYS", "90"))
        except Exception:
            retention_days = 90
        async_processing = os.getenv("KEI_WEBHOOK_AUDIT_ASYNC", "true").lower() == "true"
        return WebhookAuditConfig(
            enabled=enabled,
            level=level,
            log_dir=log_dir,
            log_file=log_file,
            max_file_size=max_file_size,
            backup_count=backup_count,
            retention_days=retention_days,
            async_processing=async_processing,
        )


class WebhookAuditLogger:
    """Asynchroner, strukturierter Audit-Logger für Webhook-Events.

    Nutzt eine Event-Queue, Rotationsdateien und PII-/DLP-Redaktion.
    """

    def __init__(self, config: WebhookAuditConfig | None = None) -> None:
        self.config = config or WebhookAuditConfig.load()
        self._pii_cfg = load_pii_config()
        self._event_queue: asyncio.Queue[WebhookAuditRecord] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Konfiguriert einen dedizierten Logger mit JSON-Ausgabe."""
        lg = logging.getLogger("services_webhooks_audit")
        lg.setLevel(self.config.level)
        if not lg.handlers:
            # Datei-Handler mit Rotation
            log_dir = Path(self.config.log_dir)
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / self.config.log_file
            except (OSError, PermissionError) as e:
                logger.warning(f"Audit-Log-Verzeichnis konnte nicht erstellt werden: {e}. Verwende Fallback.")
                log_path = Path.cwd() / "logs" / self.config.log_file
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Audit-Log-Setup: {e}. Verwende Fallback.")
                log_path = Path.cwd() / "logs" / self.config.log_file
                log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(handler)
        return lg

    async def start(self) -> None:
        """Startet die asynchrone Eventverarbeitung."""
        if not self.config.enabled:
            return
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._run())
            logger.info("WebhookAuditLogger Verarbeitung gestartet")

    async def stop(self) -> None:
        """Stoppt die Eventverarbeitung (best effort)."""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task
            logger.info("WebhookAuditLogger Verarbeitung gestoppt")

    async def _run(self) -> None:
        """Verarbeitet Events aus der Queue bis zum Abbruch."""
        while True:
            try:
                record = await self._event_queue.get()
                self._emit(record)
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except (ValueError, TypeError) as exc:  # pragma: no cover
                logger.error(f"WebhookAuditLogger Daten-/Validierungsfehler: {exc}")
            except Exception as exc:  # pragma: no cover
                logger.exception(f"WebhookAuditLogger Unerwarteter Fehler: {exc}")

    def _emit(self, record: WebhookAuditRecord) -> None:
        """Schreibt ein einzelnes Event in das Log (synchron).

        Sicherheits- und Fehler-Events werden als WARNING geloggt, erfolgreiche
        Events als INFO. Dies ermöglicht einfache Level-Filterung über ENV.
        """
        try:
            payload = record.to_json(self._pii_cfg)
            # Signatur berechnen (best effort) und als Prefix schreiben
            sig = self._safe_compute_signature(record)
            if sig:
                payload = json.dumps({"_sig": sig, "event": json.loads(payload)}, ensure_ascii=False)
            level = self._level_for(record)
            self._logger.log(level, payload)
        except (ValueError, TypeError) as exc:  # pragma: no cover
            logger.error(f"Audit-Emit fehlgeschlagen - JSON-/Serialisierungsfehler: {exc}")
        except (OSError, PermissionError) as exc:  # pragma: no cover
            logger.error(f"Audit-Emit fehlgeschlagen - Datei-/Berechtigungsfehler: {exc}")
        except Exception as exc:  # pragma: no cover
            logger.exception(f"Audit-Emit fehlgeschlagen - Unerwarteter Fehler: {exc}")

    def _level_for(self, record: WebhookAuditRecord) -> int:
        """Bestimmt den Log-Level für ein Audit-Event."""
        try:
            # Security und Failure -> WARNING
            if record.event_type in {
                WebhookAuditEventType.SECURITY_INVALID_SIGNATURE,
                WebhookAuditEventType.SECURITY_REPLAY_ATTACK,
                WebhookAuditEventType.SECURITY_RATE_LIMIT_EXCEEDED,
            }:
                return logging.WARNING
            if record.result == WebhookAuditResult.FAILURE:
                return logging.WARNING
            # Retry -> INFO (sichtbar aber nicht alarmierend)
            return logging.INFO
        except Exception:
            return logging.INFO

    async def log(self, record: WebhookAuditRecord) -> None:
        """Loggt ein Event asynchron oder synchron gemäß Konfiguration."""
        if not self.config.enabled:
            return
        if self.config.async_processing:
            try:
                # Lazy-Start der Verarbeitung
                await self.start()
                await self._event_queue.put(record)
                return
            except (asyncio.QueueFull, asyncio.CancelledError) as e:  # pragma: no cover - Fallback
                logger.debug(f"Async-Audit-Logging fehlgeschlagen - Queue-/Cancel-Problem: {e}")
            except Exception as e:  # pragma: no cover - Fallback
                logger.warning(f"Async-Audit-Logging fehlgeschlagen - Unerwarteter Fehler: {e}")
        # Synchroner Fallback
        self._emit(record)

    def _safe_compute_signature(self, record: WebhookAuditRecord) -> str | None:
        """Berechnet HMAC-Signatur mit Key-Vault Integration.

        Returns:
            Signatur oder None bei Fehlern/fehlender Konfiguration
        """
        try:
            key_name = os.getenv("KEI_WEBHOOK_AUDIT_KV_KEY_NAME", "kei-webhook-audit-hmac")
            sm = get_secret_manager()
            key, _ = sm.get_current_secret_sync(key_name=key_name)  # type: ignore[attr-defined]
            if not key:
                return None
            return record.compute_signature(key)
        except Exception:
            return None

    def verify_payload(self, payload: str) -> bool:
        """Verifiziert eine Audit-Log Zeile mit eingebetteter Signatur.

        Args:
            payload: JSON-Zeile wie geloggt

        Returns:
            True bei gültiger Signatur
        """
        try:
            obj = json.loads(payload)
            sig = obj.get("_sig")
            event = obj.get("event")
            if not sig or not event:
                return False
            rec = WebhookAuditRecord(**event)
            expect = self._safe_compute_signature(rec)
            return bool(expect and hmac.compare_digest(sig, expect))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Convenience-APIs für spezifische Events
    # ------------------------------------------------------------------
    async def inbound_received(
        self,
        *,
        correlation_id: str,
        tenant_id: str | None,
        user_id: str | None,
        topic: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt Empfang eines Inbound-Webhooks."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.INBOUND_RECEIVED,
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.RECEIVE,
                result=WebhookAuditResult.SUCCESS,
                user_id=user_id,
                tenant_id=tenant_id,
                details={"event_type": topic, **(details or {})},
                retention_days=self.config.retention_days,
            )
        )

    async def inbound_validated(
        self,
        *,
        correlation_id: str,
        tenant_id: str | None,
        user_id: str | None,
        topic: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt erfolgreiche Inbound-Validierung."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.INBOUND_VALIDATED,
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.VALIDATE,
                result=WebhookAuditResult.SUCCESS,
                user_id=user_id,
                tenant_id=tenant_id,
                details={"event_type": topic, **(details or {})},
                retention_days=self.config.retention_days,
            )
        )

    async def inbound_rejected(
        self,
        *,
        correlation_id: str,
        tenant_id: str | None,
        user_id: str | None,
        topic: str,
        error_details: dict[str, Any],
    ) -> None:
        """Loggt abgelehnten Inbound-Webhooks mit Fehlerdetails."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.INBOUND_REJECTED,
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.REJECT,
                result=WebhookAuditResult.FAILURE,
                user_id=user_id,
                tenant_id=tenant_id,
                error_details=error_details,
                details={"event_type": topic},
                retention_days=self.config.retention_days,
            )
        )

    async def outbound_enqueued(
        self,
        *,
        correlation_id: str,
        delivery_id: str,
        target_id: str,
        event_type: str,
        tenant_id: str | None,
        user_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt Enqueue eines Outbound-Webhooks."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.OUTBOUND_ENQUEUED,
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.ENQUEUE,
                result=WebhookAuditResult.SUCCESS,
                user_id=user_id,
                tenant_id=tenant_id,
                delivery_id=delivery_id,
                target_id=target_id,
                details={"event_type": event_type, **(details or {})},
                retention_days=self.config.retention_days,
            )
        )

    async def outbound_delivered(
        self,
        *,
        correlation_id: str | None,
        delivery_id: str,
        target_id: str,
        event_type: str,
        tenant_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt erfolgreiche Zustellung eines Outbound-Webhooks."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.OUTBOUND_DELIVERED,
                correlation_id=correlation_id or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.DELIVER,
                result=WebhookAuditResult.SUCCESS,
                user_id=None,
                tenant_id=tenant_id,
                delivery_id=delivery_id,
                target_id=target_id,
                details={"event_type": event_type, **(details or {})},
                retention_days=self.config.retention_days,
            )
        )

    async def outbound_failed(
        self,
        *,
        correlation_id: str | None,
        delivery_id: str,
        target_id: str,
        event_type: str,
        tenant_id: str | None,
        error_details: dict[str, Any],
        will_retry: bool,
    ) -> None:
        """Loggt fehlgeschlagene Zustellung inkl. Retry-Indikator."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.OUTBOUND_FAILED,
                correlation_id=correlation_id or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.DELIVER,
                result=WebhookAuditResult.RETRY if will_retry else WebhookAuditResult.FAILURE,
                user_id=None,
                tenant_id=tenant_id,
                delivery_id=delivery_id,
                target_id=target_id,
                error_details=error_details,
                details={"event_type": event_type, "will_retry": will_retry},
                retention_days=self.config.retention_days,
            )
        )

    async def outbound_retried(
        self,
        *,
        correlation_id: str | None,
        delivery_id: str,
        target_id: str,
        event_type: str,
        tenant_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt einen erneuten Zustellversuch."""
        await self.log(
            WebhookAuditRecord(
                event_type=WebhookAuditEventType.OUTBOUND_RETRIED,
                correlation_id=correlation_id or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.RETRY,
                result=WebhookAuditResult.RETRY,
                user_id=None,
                tenant_id=tenant_id,
                delivery_id=delivery_id,
                target_id=target_id,
                details={"event_type": event_type, **(details or {})},
                retention_days=self.config.retention_days,
            )
        )

    async def target_changed(
        self,
        *,
        operation: WebhookAuditOperation,
        target_id: str,
        user_id: str | None,
        tenant_id: str | None,
        correlation_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt Target-Operationen (create/update/delete)."""
        event_map: dict[WebhookAuditOperation, WebhookAuditEventType] = {
            WebhookAuditOperation.CREATE: WebhookAuditEventType.TARGET_CREATED,
            WebhookAuditOperation.UPDATE: WebhookAuditEventType.TARGET_UPDATED,
            WebhookAuditOperation.DELETE: WebhookAuditEventType.TARGET_DELETED,
        }
        await self.log(
            WebhookAuditRecord(
                event_type=event_map.get(operation, WebhookAuditEventType.TARGET_UPDATED),
                correlation_id=correlation_id or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                operation=operation,
                result=WebhookAuditResult.SUCCESS,
                user_id=user_id,
                tenant_id=tenant_id,
                target_id=target_id,
                details=details or {},
                retention_days=self.config.retention_days,
            )
        )

    async def dlq_event(
        self,
        *,
        event: WebhookAuditEventType,
        delivery_id: str,
        target_id: str,
        tenant_id: str | None,
        correlation_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt DLQ-Ereignisse (move/requeue/purge)."""
        await self.log(
            WebhookAuditRecord(
                event_type=event,
                correlation_id=correlation_id or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                operation=(
                    WebhookAuditOperation.MOVE_TO_DLQ if event == WebhookAuditEventType.DLQ_MOVE
                    else WebhookAuditOperation.REQUEUE_DLQ if event == WebhookAuditEventType.DLQ_REQUEUE
                    else WebhookAuditOperation.PURGE_DLQ
                ),
                result=WebhookAuditResult.SUCCESS,
                user_id=None,
                tenant_id=tenant_id,
                delivery_id=delivery_id,
                target_id=target_id,
                details=details or {},
                retention_days=self.config.retention_days,
            )
        )

    async def security_event(
        self,
        *,
        event: WebhookAuditEventType,
        correlation_id: str | None,
        tenant_id: str | None,
        user_id: str | None,
        error_details: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Loggt Security-relevante Ereignisse (invalid signature/replay/rate-limit)."""
        await self.log(
            WebhookAuditRecord(
                event_type=event,
                correlation_id=correlation_id or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                operation=WebhookAuditOperation.VALIDATE,
                result=WebhookAuditResult.FAILURE,
                user_id=user_id,
                tenant_id=tenant_id,
                error_details=error_details,
                details=details or {},
                retention_days=self.config.retention_days,
            )
        )


# Globale, wiederverwendbare Instanz
_global_audit_logger: WebhookAuditLogger | None = None


def get_webhook_audit_logger() -> WebhookAuditLogger:
    """Gibt Singleton-Instanz des Webhook-Audit-Loggers zurück."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = WebhookAuditLogger()
    return _global_audit_logger


# Bequemer Alias für Importe
webhook_audit: WebhookAuditLogger = get_webhook_audit_logger()


__all__ = [
    "WebhookAuditConfig",
    "WebhookAuditEventType",
    "WebhookAuditLogger",
    "WebhookAuditOperation",
    "WebhookAuditRecord",
    "WebhookAuditResult",
    "get_webhook_audit_logger",
    "webhook_audit",
]
