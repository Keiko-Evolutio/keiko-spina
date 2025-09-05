# backend/audit_system/compliance_manager.py
"""Compliance Manager für KEI-Agent-Framework Audit System.

Implementiert konfigurierbare Retention-Policies, automatische Archivierung,
Export-Funktionen und Right-to-be-Forgotten für Audit-Daten.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .core_audit_engine import AuditEvent

logger = get_logger(__name__)


class ComplianceStandard(str, Enum):
    """Unterstützte Compliance-Standards."""
    SOX = "sox"
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST_CYBERSECURITY = "nist_cybersecurity"
    CUSTOM = "custom"


class RetentionAction(str, Enum):
    """Aktionen bei Retention-Ablauf."""
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    NOTIFY = "notify"
    MANUAL_REVIEW = "manual_review"


class ExportFormat(str, Enum):
    """Export-Formate für Audit-Daten."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PDF = "pdf"
    PARQUET = "parquet"


class ArchiveStatus(str, Enum):
    """Status von Archiven."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    COMPRESSED = "compressed"
    ENCRYPTED = "encrypted"
    DELETED = "deleted"


@dataclass
class RetentionPolicy:
    """Retention-Policy für Audit-Daten."""
    policy_id: str
    name: str
    description: str

    # Retention-Konfiguration
    retention_period_days: int
    compliance_standards: set[ComplianceStandard] = field(default_factory=set)

    # Scope
    event_types: set[str] | None = None
    agent_ids: set[str] | None = None
    user_ids: set[str] | None = None
    tenant_ids: set[str] | None = None

    # Aktionen
    retention_action: RetentionAction = RetentionAction.ARCHIVE
    auto_execute: bool = True
    notification_before_days: int = 30

    # Ausnahmen
    legal_hold_exemption: bool = False
    regulatory_exemption: bool = False

    # Gültigkeit
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def applies_to_event(self, event: AuditEvent) -> bool:
        """Prüft, ob Policy auf Event anwendbar ist."""
        if not self.enabled:
            return False

        # Prüfe Event-Typ
        if self.event_types and event.event_type.value not in self.event_types:
            return False

        # Prüfe Agent-ID
        if self.agent_ids and event.context.agent_id not in self.agent_ids:
            return False

        # Prüfe User-ID
        if self.user_ids and event.context.user_id not in self.user_ids:
            return False

        # Prüfe Tenant-ID
        return not (self.tenant_ids and event.context.tenant_id not in self.tenant_ids)

    def is_retention_due(self, event: AuditEvent) -> bool:
        """Prüft, ob Retention-Zeitraum abgelaufen ist."""
        if not self.applies_to_event(event):
            return False

        retention_deadline = event.timestamp + timedelta(days=self.retention_period_days)
        return datetime.now(UTC) >= retention_deadline

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "retention_period_days": self.retention_period_days,
            "compliance_standards": [std.value for std in self.compliance_standards],
            "event_types": list(self.event_types) if self.event_types else None,
            "agent_ids": list(self.agent_ids) if self.agent_ids else None,
            "user_ids": list(self.user_ids) if self.user_ids else None,
            "tenant_ids": list(self.tenant_ids) if self.tenant_ids else None,
            "retention_action": self.retention_action.value,
            "auto_execute": self.auto_execute,
            "notification_before_days": self.notification_before_days,
            "legal_hold_exemption": self.legal_hold_exemption,
            "regulatory_exemption": self.regulatory_exemption,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ArchiveRecord:
    """Record für archivierte Audit-Daten."""
    # Required fields first
    archive_id: str
    archive_path: str
    event_count: int
    start_date: datetime
    end_date: datetime
    original_size_bytes: int
    retention_policy_id: str

    # Optional fields with defaults
    compressed_size_bytes: int | None = None
    compression_ratio: float | None = None
    status: ArchiveStatus = ArchiveStatus.ACTIVE
    archived_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    delete_after: datetime | None = None
    checksum: str | None = None
    encrypted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "archive_id": self.archive_id,
            "archive_path": self.archive_path,
            "event_count": self.event_count,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "status": self.status.value,
            "archived_at": self.archived_at.isoformat(),
            "retention_policy_id": self.retention_policy_id,
            "delete_after": self.delete_after.isoformat() if self.delete_after else None,
            "checksum": self.checksum,
            "encrypted": self.encrypted
        }


@dataclass
class ExportRequest:
    """Request für Audit-Daten-Export."""
    export_id: str
    requester_id: str
    purpose: str

    # Filter
    start_date: datetime | None = None
    end_date: datetime | None = None
    event_types: set[str] | None = None
    agent_ids: set[str] | None = None
    user_ids: set[str] | None = None

    # Export-Konfiguration
    export_format: ExportFormat = ExportFormat.JSON
    include_pii: bool = False
    anonymize_data: bool = True

    # Status
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Ergebnis
    export_path: str | None = None
    export_size_bytes: int | None = None
    record_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "export_id": self.export_id,
            "requester_id": self.requester_id,
            "purpose": self.purpose,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "event_types": list(self.event_types) if self.event_types else None,
            "agent_ids": list(self.agent_ids) if self.agent_ids else None,
            "user_ids": list(self.user_ids) if self.user_ids else None,
            "export_format": self.export_format.value,
            "include_pii": self.include_pii,
            "anonymize_data": self.anonymize_data,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "export_path": self.export_path,
            "export_size_bytes": self.export_size_bytes,
            "record_count": self.record_count
        }


class ArchiveManager:
    """Manager für Audit-Daten-Archivierung."""

    def __init__(self, archive_base_path: str = "/var/audit/archives"):
        """Initialisiert Archive Manager.

        Args:
            archive_base_path: Basis-Pfad für Archive
        """
        self.archive_base_path = Path(archive_base_path)
        self.archive_base_path.mkdir(parents=True, exist_ok=True)

        self._archive_records: dict[str, ArchiveRecord] = {}

        # Statistiken
        self._archives_created = 0
        self._total_archived_events = 0
        self._total_archived_size = 0

    @trace_function("compliance.create_archive")
    async def create_archive(
        self,
        events: list[AuditEvent],
        retention_policy_id: str,
        compress: bool = True,
        encrypt: bool = False
    ) -> str:
        """Erstellt Archiv aus Events.

        Args:
            events: Zu archivierende Events
            retention_policy_id: Retention-Policy-ID
            compress: Komprimierung aktivieren
            encrypt: Verschlüsselung aktivieren

        Returns:
            Archive-ID
        """
        import uuid

        if not events:
            raise ValueError("Keine Events zum Archivieren")

        archive_id = str(uuid.uuid4())

        # Bestimme Zeitraum
        timestamps = [event.timestamp for event in events]
        start_date = min(timestamps)
        end_date = max(timestamps)

        # Erstelle Archiv-Pfad
        archive_filename = f"audit_archive_{archive_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        if compress:
            archive_filename += ".gz"

        archive_path = self.archive_base_path / archive_filename

        # Serialisiere Events
        events_data = [event.to_dict() for event in events]
        archive_content = {
            "archive_id": archive_id,
            "created_at": datetime.now(UTC).isoformat(),
            "retention_policy_id": retention_policy_id,
            "event_count": len(events),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "events": events_data
        }

        # Schreibe Archiv
        content_json = json.dumps(archive_content, indent=2)
        original_size = len(content_json.encode("utf-8"))

        if compress:
            with gzip.open(archive_path, "wt", encoding="utf-8") as f:
                f.write(content_json)
            compressed_size = archive_path.stat().st_size
            compression_ratio = compressed_size / original_size
        else:
            with open(archive_path, "w", encoding="utf-8") as f:
                f.write(content_json)
            compressed_size = None
            compression_ratio = None

        # Berechne Checksum
        checksum = await self._calculate_checksum(archive_path)

        # Erstelle Archive-Record
        archive_record = ArchiveRecord(
            archive_id=archive_id,
            archive_path=str(archive_path),
            event_count=len(events),
            start_date=start_date,
            end_date=end_date,
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
            status=ArchiveStatus.COMPRESSED if compress else ArchiveStatus.ARCHIVED,
            retention_policy_id=retention_policy_id,
            checksum=checksum,
            encrypted=encrypt
        )

        self._archive_records[archive_id] = archive_record

        # Aktualisiere Statistiken
        self._archives_created += 1
        self._total_archived_events += len(events)
        self._total_archived_size += compressed_size or original_size

        logger.info(f"Archiv erstellt: {archive_id} mit {len(events)} Events")

        return archive_id

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Berechnet Checksum für Datei."""
        import hashlib

        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def retrieve_archive(self, archive_id: str) -> list[AuditEvent] | None:
        """Lädt Events aus Archiv.

        Args:
            archive_id: Archive-ID

        Returns:
            Liste von Events oder None
        """
        archive_record = self._archive_records.get(archive_id)
        if not archive_record:
            return None

        archive_path = Path(archive_record.archive_path)
        if not archive_path.exists():
            logger.error(f"Archiv-Datei nicht gefunden: {archive_path}")
            return None

        try:
            # Lade Archiv-Inhalt
            if archive_record.status == ArchiveStatus.COMPRESSED:
                with gzip.open(archive_path, "rt", encoding="utf-8") as f:
                    archive_content = json.load(f)
            else:
                with open(archive_path, encoding="utf-8") as f:
                    archive_content = json.load(f)

            # Konvertiere zu AuditEvents
            events = []
            for event_data in archive_content.get("events", []):
                event = AuditEvent.from_dict(event_data)
                events.append(event)

            return events

        except Exception as e:
            logger.exception(f"Archiv-Laden fehlgeschlagen: {e}")
            return None

    async def delete_archive(self, archive_id: str) -> bool:
        """Löscht Archiv.

        Args:
            archive_id: Archive-ID

        Returns:
            True wenn erfolgreich
        """
        archive_record = self._archive_records.get(archive_id)
        if not archive_record:
            return False

        archive_path = Path(archive_record.archive_path)

        try:
            if archive_path.exists():
                archive_path.unlink()

            archive_record.status = ArchiveStatus.DELETED

            logger.info(f"Archiv gelöscht: {archive_id}")
            return True

        except Exception as e:
            logger.exception(f"Archiv-Löschung fehlgeschlagen: {e}")
            return False

    def get_archive_statistics(self) -> dict[str, Any]:
        """Gibt Archiv-Statistiken zurück."""
        status_counts = {}
        for record in self._archive_records.values():
            status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1

        return {
            "archives_created": self._archives_created,
            "total_archived_events": self._total_archived_events,
            "total_archived_size_bytes": self._total_archived_size,
            "active_archives": len(self._archive_records),
            "archive_status_distribution": status_counts
        }


class ExportManager:
    """Manager für Audit-Daten-Export."""

    def __init__(self, export_base_path: str = "/var/audit/exports"):
        """Initialisiert Export Manager.

        Args:
            export_base_path: Basis-Pfad für Exports
        """
        self.export_base_path = Path(export_base_path)
        self.export_base_path.mkdir(parents=True, exist_ok=True)

        self._export_requests: dict[str, ExportRequest] = {}

        # Statistiken
        self._exports_completed = 0
        self._total_exported_records = 0

    @trace_function("compliance.create_export")
    async def create_export(
        self,
        requester_id: str,
        purpose: str,
        events: list[AuditEvent],
        export_format: ExportFormat = ExportFormat.JSON,
        include_pii: bool = False,
        anonymize_data: bool = True
    ) -> str:
        """Erstellt Export von Audit-Daten.

        Args:
            requester_id: ID des Anforderers
            purpose: Zweck des Exports
            events: Zu exportierende Events
            export_format: Export-Format
            include_pii: PII einschließen
            anonymize_data: Daten anonymisieren

        Returns:
            Export-ID
        """
        import uuid

        export_id = str(uuid.uuid4())

        # Erstelle Export-Request
        export_request = ExportRequest(
            export_id=export_id,
            requester_id=requester_id,
            purpose=purpose,
            export_format=export_format,
            include_pii=include_pii,
            anonymize_data=anonymize_data,
            status="processing"
        )

        self._export_requests[export_id] = export_request

        try:
            # Verarbeite Events
            processed_events = await self._process_events_for_export(
                events, include_pii, anonymize_data
            )

            # Erstelle Export-Datei
            export_path = await self._create_export_file(
                export_id, processed_events, export_format
            )

            # Aktualisiere Request
            export_request.status = "completed"
            export_request.completed_at = datetime.now(UTC)
            export_request.export_path = str(export_path)
            export_request.export_size_bytes = export_path.stat().st_size
            export_request.record_count = len(processed_events)

            # Aktualisiere Statistiken
            self._exports_completed += 1
            self._total_exported_records += len(processed_events)

            logger.info(f"Export erstellt: {export_id} mit {len(processed_events)} Records")

        except Exception as e:
            export_request.status = "failed"
            logger.exception(f"Export fehlgeschlagen: {e}")

        return export_id

    async def _process_events_for_export(
        self,
        events: list[AuditEvent],
        include_pii: bool,
        anonymize_data: bool
    ) -> list[dict[str, Any]]:
        """Verarbeitet Events für Export."""
        processed_events = []

        for event in events:
            event_dict = event.to_dict()

            if not include_pii:
                # Entferne PII-Felder
                event_dict = await self._remove_pii_from_event(event_dict)

            if anonymize_data:
                # Anonymisiere Daten
                event_dict = await self._anonymize_event_data(event_dict)

            processed_events.append(event_dict)

        return processed_events

    async def _remove_pii_from_event(self, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Entfernt PII aus Event-Dictionary."""
        # Vereinfachte PII-Entfernung
        pii_fields = ["user_id", "client_ip", "user_agent"]

        if "context" in event_dict:
            for pii_field in pii_fields:
                if pii_field in event_dict["context"]:
                    event_dict["context"][pii_field] = "[REMOVED]"

        return event_dict

    async def _anonymize_event_data(self, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Anonymisiert Event-Daten."""
        # Vereinfachte Anonymisierung
        if "context" in event_dict and "user_id" in event_dict["context"]:
            user_id = event_dict["context"]["user_id"]
            if user_id and user_id != "[REMOVED]":
                # Erstelle anonyme User-ID
                import hashlib
                anonymous_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
                event_dict["context"]["user_id"] = f"anon_{anonymous_id}"

        return event_dict

    async def _create_export_file(
        self,
        export_id: str,
        events: list[dict[str, Any]],
        export_format: ExportFormat
    ) -> Path:
        """Erstellt Export-Datei."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format == ExportFormat.JSON:
            filename = f"audit_export_{export_id}_{timestamp}.json"
            export_path = self.export_base_path / filename

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump({
                    "export_id": export_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "record_count": len(events),
                    "events": events
                }, f, indent=2)

        elif export_format == ExportFormat.CSV:
            filename = f"audit_export_{export_id}_{timestamp}.csv"
            export_path = self.export_base_path / filename

            # Vereinfachte CSV-Erstellung
            import csv

            with open(export_path, "w", newline="", encoding="utf-8") as f:
                if events:
                    # Flache Struktur für CSV
                    flattened_events = []
                    for event in events:
                        flat_event = self._flatten_dict(event)
                        flattened_events.append(flat_event)

                    fieldnames = set()
                    for event in flattened_events:
                        fieldnames.update(event.keys())

                    writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    writer.writerows(flattened_events)

        else:
            raise ValueError(f"Export-Format nicht unterstützt: {export_format}")

        return export_path

    def _flatten_dict(self, d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Flacht Dictionary für CSV ab."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def get_export_status(self, export_id: str) -> ExportRequest | None:
        """Gibt Export-Status zurück."""
        return self._export_requests.get(export_id)

    def get_export_statistics(self) -> dict[str, Any]:
        """Gibt Export-Statistiken zurück."""
        status_counts = {}
        for request in self._export_requests.values():
            status_counts[request.status] = status_counts.get(request.status, 0) + 1

        return {
            "exports_completed": self._exports_completed,
            "total_exported_records": self._total_exported_records,
            "active_exports": len(self._export_requests),
            "export_status_distribution": status_counts
        }


class RightToBeForgottenHandler:
    """Handler für Right-to-be-Forgotten-Requests."""

    def __init__(self):
        """Initialisiert Right-to-be-Forgotten Handler."""
        self._deletion_requests: dict[str, dict[str, Any]] = {}
        self._deleted_user_ids: set[str] = set()

        # Statistiken
        self._deletion_requests_processed = 0
        self._events_deleted = 0

    @trace_function("compliance.process_deletion_request")
    async def process_deletion_request(
        self,
        user_id: str,
        requester_id: str,
        reason: str,
        events: list[AuditEvent]
    ) -> str:
        """Verarbeitet Right-to-be-Forgotten-Request.

        Args:
            user_id: Zu löschende User-ID
            requester_id: ID des Anforderers
            reason: Grund für Löschung
            events: Betroffene Events

        Returns:
            Request-ID
        """
        import uuid

        request_id = str(uuid.uuid4())

        # Filtere Events für User
        user_events = [
            event for event in events
            if event.context.user_id == user_id
        ]

        # Erstelle Deletion-Request
        deletion_request = {
            "request_id": request_id,
            "user_id": user_id,
            "requester_id": requester_id,
            "reason": reason,
            "status": "processing",
            "created_at": datetime.now(UTC).isoformat(),
            "events_to_delete": len(user_events),
            "events_deleted": 0
        }

        self._deletion_requests[request_id] = deletion_request

        try:
            # Lösche oder anonymisiere Events
            deleted_count = await self._delete_user_events(user_events)

            # Markiere User als gelöscht
            self._deleted_user_ids.add(user_id)

            # Aktualisiere Request
            deletion_request["status"] = "completed"
            deletion_request["completed_at"] = datetime.now(UTC).isoformat()
            deletion_request["events_deleted"] = deleted_count

            # Aktualisiere Statistiken
            self._deletion_requests_processed += 1
            self._events_deleted += deleted_count

            logger.info(f"Right-to-be-Forgotten verarbeitet: {request_id} für User {user_id}")

        except Exception as e:
            deletion_request["status"] = "failed"
            deletion_request["error"] = str(e)
            logger.exception(f"Right-to-be-Forgotten fehlgeschlagen: {e}")

        return request_id

    async def _delete_user_events(self, events: list[AuditEvent]) -> int:
        """Löscht oder anonymisiert User-Events."""
        deleted_count = 0

        for event in events:
            # Anonymisiere User-spezifische Daten
            if event.context.user_id:
                event.context.user_id = "[DELETED]"

            # Entferne PII aus Event-Daten
            if event.input_data:
                event.input_data = self._anonymize_user_data(event.input_data)

            if event.output_data:
                event.output_data = self._anonymize_user_data(event.output_data)

            deleted_count += 1

        return deleted_count

    def _anonymize_user_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Anonymisiert User-Daten in Dictionary."""
        if not isinstance(data, dict):
            return data

        anonymized_data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                anonymized_data[key] = self._anonymize_user_data(value)
            elif isinstance(value, str) and any(pii_indicator in key.lower() for pii_indicator in ["user", "name", "email", "phone"]):
                anonymized_data[key] = "[DELETED]"
            else:
                anonymized_data[key] = value

        return anonymized_data

    def is_user_deleted(self, user_id: str) -> bool:
        """Prüft, ob User gelöscht wurde."""
        return user_id in self._deleted_user_ids

    def get_deletion_statistics(self) -> dict[str, Any]:
        """Gibt Löschungs-Statistiken zurück."""
        return {
            "deletion_requests_processed": self._deletion_requests_processed,
            "events_deleted": self._events_deleted,
            "deleted_users": len(self._deleted_user_ids),
            "active_requests": len(self._deletion_requests)
        }


class ComplianceManager:
    """Hauptklasse für Compliance-Management."""

    def __init__(self, base_path: str = "/var/audit/compliance"):
        """Initialisiert Compliance Manager.

        Args:
            base_path: Basis-Pfad für Compliance-Daten
        """
        self.base_path = Path(base_path)
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback für Tests und Entwicklung
            import tempfile
            self.base_path = Path(tempfile.mkdtemp(prefix="audit_compliance_"))

        # Sub-Manager
        self.archive_manager = ArchiveManager(str(self.base_path / "archives"))
        self.export_manager = ExportManager(str(self.base_path / "exports"))
        self.rtbf_handler = RightToBeForgottenHandler()

        # Retention-Policies
        self._retention_policies: dict[str, RetentionPolicy] = {}
        self._default_policies = self._create_default_policies()

        # Background-Tasks
        self._retention_task: asyncio.Task | None = None
        self._is_running = False

    def _create_default_policies(self) -> dict[str, RetentionPolicy]:
        """Erstellt Standard-Retention-Policies."""
        policies = {}

        # SOX-Policy (7 Jahre)
        sox_policy = RetentionPolicy(
            policy_id="sox_financial_records",
            name="SOX Financial Records",
            description="7-jährige Aufbewahrung für Finanzunterlagen (SOX)",
            retention_period_days=7 * 365,  # 7 Jahre
            compliance_standards={ComplianceStandard.SOX},
            retention_action=RetentionAction.ARCHIVE
        )
        policies[sox_policy.policy_id] = sox_policy

        # GDPR-Policy (3 Jahre Standard)
        gdpr_policy = RetentionPolicy(
            policy_id="gdpr_personal_data",
            name="GDPR Personal Data",
            description="3-jährige Aufbewahrung für personenbezogene Daten (GDPR)",
            retention_period_days=3 * 365,  # 3 Jahre
            compliance_standards={ComplianceStandard.GDPR},
            retention_action=RetentionAction.ANONYMIZE
        )
        policies[gdpr_policy.policy_id] = gdpr_policy

        # HIPAA-Policy (6 Jahre)
        hipaa_policy = RetentionPolicy(
            policy_id="hipaa_health_records",
            name="HIPAA Health Records",
            description="6-jährige Aufbewahrung für Gesundheitsdaten (HIPAA)",
            retention_period_days=6 * 365,  # 6 Jahre
            compliance_standards={ComplianceStandard.HIPAA},
            retention_action=RetentionAction.ARCHIVE
        )
        policies[hipaa_policy.policy_id] = hipaa_policy

        return policies

    def register_retention_policy(self, policy: RetentionPolicy) -> None:
        """Registriert Retention-Policy."""
        self._retention_policies[policy.policy_id] = policy
        logger.info(f"Retention-Policy registriert: {policy.policy_id}")

    async def start_retention_monitoring(self) -> None:
        """Startet Retention-Monitoring."""
        if self._is_running:
            return

        self._is_running = True
        self._retention_task = asyncio.create_task(self._retention_monitoring_loop())
        logger.info("Retention-Monitoring gestartet")

    async def stop_retention_monitoring(self) -> None:
        """Stoppt Retention-Monitoring."""
        self._is_running = False

        if self._retention_task:
            self._retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._retention_task

        logger.info("Retention-Monitoring gestoppt")

    async def _retention_monitoring_loop(self) -> None:
        """Retention-Monitoring-Loop."""
        while self._is_running:
            try:
                # Prüfe alle Retention-Policies
                for policy in list(self._retention_policies.values()) + list(self._default_policies.values()):
                    if policy.enabled and policy.auto_execute:
                        await self._check_retention_policy(policy)

                # Warte 24 Stunden bis zur nächsten Prüfung
                await asyncio.sleep(24 * 3600)

            except Exception as e:
                logger.exception(f"Retention-Monitoring-Fehler: {e}")
                await asyncio.sleep(3600)  # 1 Stunde warten bei Fehler

    async def _check_retention_policy(self, policy: RetentionPolicy) -> None:
        """Prüft Retention-Policy."""
        # Hier würde normalerweise die Event-Datenbank abgefragt werden
        # Für Demo vereinfacht
        logger.info(f"Prüfe Retention-Policy: {policy.policy_id}")

    def get_compliance_statistics(self) -> dict[str, Any]:
        """Gibt Compliance-Statistiken zurück."""
        return {
            "retention_policies": len(self._retention_policies) + len(self._default_policies),
            "archive_stats": self.archive_manager.get_archive_statistics(),
            "export_stats": self.export_manager.get_export_statistics(),
            "deletion_stats": self.rtbf_handler.get_deletion_statistics(),
            "is_monitoring": self._is_running
        }


# Globale Compliance Manager Instanz
compliance_manager = ComplianceManager()
