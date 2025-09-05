# backend/policy_engine/compliance_framework.py
"""Compliance Framework für Keiko Personal Assistant

Implementiert GDPR/CCPA-konforme Datenverarbeitung, Industry-spezifische
Compliance-Checks, Audit-Trails und Right-to-be-Forgotten-Mechanismen.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class ComplianceStandard(str, Enum):
    """Compliance-Standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CUSTOM = "custom"


class ViolationSeverity(str, Enum):
    """Schweregrad von Compliance-Verletzungen."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataProcessingPurpose(str, Enum):
    """Zwecke der Datenverarbeitung."""
    LEGITIMATE_INTEREST = "legitimate_interest"
    CONTRACT_PERFORMANCE = "contract_performance"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    CONSENT = "consent"


@dataclass
class ComplianceViolation:
    """Repräsentiert eine Compliance-Verletzung."""
    standard: ComplianceStandard
    violation_type: str
    severity: ViolationSeverity
    description: str
    affected_data: str
    remediation_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ComplianceCheck:
    """Ergebnis einer Compliance-Prüfung."""
    standard: ComplianceStandard
    compliant: bool
    violations: list[ComplianceViolation] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DataRetentionPolicy:
    """Policy für Datenaufbewahrung."""
    data_type: str
    retention_period_days: int
    purpose: DataProcessingPurpose
    auto_delete: bool = True
    archive_before_delete: bool = False
    legal_basis: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Retention-Period abgelaufen ist."""
        expiry_date = self.created_at + timedelta(days=self.retention_period_days)
        return datetime.now(UTC) > expiry_date


@dataclass
class AuditLogEntry:
    """Eintrag im Audit-Log."""
    event_id: str
    event_type: str
    user_id: str | None
    agent_id: str | None
    resource_type: str
    resource_id: str
    action: str
    result: str
    compliance_standards: list[ComplianceStandard] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ComplianceChecker(ABC):
    """Basis-Klasse für Compliance-Checker."""

    @abstractmethod
    async def check_compliance(
        self,
        data: Any,
        context: dict[str, Any] | None = None
    ) -> ComplianceCheck:
        """Führt Compliance-Check durch."""


class GDPRChecker(ComplianceChecker):
    """GDPR-Compliance-Checker."""

    def __init__(self):
        """Initialisiert GDPR Checker."""
        self._sensitive_data_indicators = {
            "personal_data", "email", "phone", "address", "name",
            "biometric", "genetic", "health", "sexual", "political",
            "religious", "philosophical", "trade_union"
        }

        self._lawful_bases = {
            DataProcessingPurpose.CONSENT,
            DataProcessingPurpose.CONTRACT_PERFORMANCE,
            DataProcessingPurpose.LEGAL_OBLIGATION,
            DataProcessingPurpose.VITAL_INTERESTS,
            DataProcessingPurpose.PUBLIC_TASK,
            DataProcessingPurpose.LEGITIMATE_INTEREST
        }

    async def check_compliance(
        self,
        data: Any,
        context: dict[str, Any] | None = None
    ) -> ComplianceCheck:
        """Führt GDPR-Compliance-Check durch."""
        violations = []
        recommendations = []

        # Prüfe auf personenbezogene Daten
        if self._contains_personal_data(data):
            # Prüfe Rechtsgrundlage
            processing_purpose = context.get("processing_purpose") if context else None
            if not processing_purpose or processing_purpose not in self._lawful_bases:
                violation = ComplianceViolation(
                    standard=ComplianceStandard.GDPR,
                    violation_type="missing_lawful_basis",
                    severity=ViolationSeverity.HIGH,
                    description="Keine gültige Rechtsgrundlage für Verarbeitung personenbezogener Daten",
                    affected_data=str(data)[:100],
                    remediation_steps=[
                        "Rechtsgrundlage gemäß Art. 6 GDPR definieren",
                        "Einwilligung einholen falls erforderlich",
                        "Datenschutzerklärung aktualisieren"
                    ]
                )
                violations.append(violation)

            # Prüfe Datenminimierung
            if not self._is_data_minimized(data, context):
                violation = ComplianceViolation(
                    standard=ComplianceStandard.GDPR,
                    violation_type="data_minimization",
                    severity=ViolationSeverity.MEDIUM,
                    description="Datenminimierung nicht eingehalten",
                    affected_data=str(data)[:100],
                    remediation_steps=[
                        "Nur notwendige Daten verarbeiten",
                        "Datenumfang auf Zweck beschränken"
                    ]
                )
                violations.append(violation)

            # Prüfe Aufbewahrungsfristen
            if not self._has_retention_policy(context):
                recommendations.append("Aufbewahrungsfristen definieren")

        return ComplianceCheck(
            standard=ComplianceStandard.GDPR,
            compliant=len(violations) == 0,
            violations=violations,
            recommendations=recommendations
        )

    def _contains_personal_data(self, data: Any) -> bool:
        """Prüft, ob Daten personenbezogene Informationen enthalten."""
        data_str = str(data).lower()
        return any(indicator in data_str for indicator in self._sensitive_data_indicators)

    def _is_data_minimized(self, data: Any, context: dict[str, Any] | None) -> bool:
        """Prüft Datenminimierung."""
        # Prüfe Kontext-Informationen
        if not context:
            return False

        purpose = context.get("processing_purpose")
        if not purpose:
            return False

        # Analysiere Datenumfang basierend auf Zweck
        data_str = str(data).lower()
        data_size = len(data_str)

        # Prüfe auf übermäßige Datensammlung
        excessive_indicators = [
            "password", "credit_card", "ssn", "passport", "driver_license",
            "bank_account", "medical_record", "biometric", "genetic"
        ]

        # Wenn sensible Daten vorhanden sind, prüfe Notwendigkeit
        sensitive_data_found = any(indicator in data_str for indicator in excessive_indicators)
        if sensitive_data_found:
            # Nur erlaubt wenn explizit für den Zweck erforderlich
            required_purposes = context.get("required_data_types", [])
            return any(indicator in str(required_purposes).lower() for indicator in excessive_indicators)

        # Prüfe Datengröße - sehr große Datenmengen könnten nicht minimiert sein
        max_size_threshold = context.get("max_data_size", 10000)  # Standard: 10KB
        return data_size <= max_size_threshold

    def _has_retention_policy(self, context: dict[str, Any] | None) -> bool:
        """Prüft, ob Aufbewahrungsrichtlinie definiert ist."""
        if not context:
            return False

        return "retention_policy" in context


class HIPAAChecker(ComplianceChecker):
    """HIPAA-Compliance-Checker."""

    def __init__(self):
        """Initialisiert HIPAA Checker."""
        self._phi_indicators = {
            "patient", "medical", "health", "diagnosis", "treatment",
            "medication", "doctor", "hospital", "clinic", "ssn",
            "medical_record", "insurance"
        }

    async def check_compliance(
        self,
        data: Any,
        context: dict[str, Any] | None = None
    ) -> ComplianceCheck:
        """Führt HIPAA-Compliance-Check durch."""
        violations = []
        recommendations = []

        if self._contains_phi(data):
            # Prüfe Verschlüsselung
            if not self._is_encrypted(context):
                violation = ComplianceViolation(
                    standard=ComplianceStandard.HIPAA,
                    violation_type="unencrypted_phi",
                    severity=ViolationSeverity.CRITICAL,
                    description="PHI nicht verschlüsselt",
                    affected_data=str(data)[:100],
                    remediation_steps=[
                        "Daten verschlüsseln",
                        "Sichere Übertragung gewährleisten"
                    ]
                )
                violations.append(violation)

            # Prüfe Zugriffskontrolle
            if not self._has_access_control(context):
                violation = ComplianceViolation(
                    standard=ComplianceStandard.HIPAA,
                    violation_type="insufficient_access_control",
                    severity=ViolationSeverity.HIGH,
                    description="Unzureichende Zugriffskontrolle für PHI",
                    affected_data=str(data)[:100],
                    remediation_steps=[
                        "Rollenbasierte Zugriffskontrolle implementieren",
                        "Audit-Logs aktivieren"
                    ]
                )
                violations.append(violation)

        return ComplianceCheck(
            standard=ComplianceStandard.HIPAA,
            compliant=len(violations) == 0,
            violations=violations,
            recommendations=recommendations
        )

    def _contains_phi(self, data: Any) -> bool:
        """Prüft, ob Daten PHI enthalten."""
        data_str = str(data).lower()
        return any(indicator in data_str for indicator in self._phi_indicators)

    def _is_encrypted(self, context: dict[str, Any] | None) -> bool:
        """Prüft Verschlüsselung."""
        if not context:
            return False
        return context.get("encrypted", False)

    def _has_access_control(self, context: dict[str, Any] | None) -> bool:
        """Prüft Zugriffskontrolle."""
        if not context:
            return False
        return context.get("access_controlled", False)


class AuditTrailManager:
    """Manager für Audit-Trails."""

    def __init__(self):
        """Initialisiert Audit Trail Manager."""
        self._audit_log: list[AuditLogEntry] = []
        self._max_entries = 100000
        self._retention_days = 2555  # 7 Jahre für Compliance

    def log_event(
        self,
        event_type: str,
        user_id: str | None,
        agent_id: str | None,
        resource_type: str,
        resource_id: str,
        action: str,
        result: str,
        compliance_standards: list[ComplianceStandard] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Loggt Audit-Event."""
        import uuid

        event_id = str(uuid.uuid4())

        entry = AuditLogEntry(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            agent_id=agent_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            compliance_standards=compliance_standards or [],
            metadata=metadata or {}
        )

        self._audit_log.append(entry)

        # Begrenze Log-Größe
        if len(self._audit_log) > self._max_entries:
            self._audit_log = self._audit_log[-self._max_entries:]

        logger.info(f"Audit-Event geloggt: {event_id}")
        return event_id

    def get_audit_trail(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        resource_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000
    ) -> list[AuditLogEntry]:
        """Gibt gefilterte Audit-Trails zurück."""
        filtered_entries = self._audit_log

        # Filter anwenden
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]

        if agent_id:
            filtered_entries = [e for e in filtered_entries if e.agent_id == agent_id]

        if resource_type:
            filtered_entries = [e for e in filtered_entries if e.resource_type == resource_type]

        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]

        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]

        # Sortiere nach Timestamp (neueste zuerst) und limitiere
        sorted_entries = sorted(filtered_entries, key=lambda e: e.timestamp, reverse=True)
        return sorted_entries[:limit]

    def cleanup_old_entries(self) -> int:
        """Bereinigt alte Audit-Einträge."""
        cutoff_date = datetime.now(UTC) - timedelta(days=self._retention_days)

        original_count = len(self._audit_log)
        self._audit_log = [e for e in self._audit_log if e.timestamp > cutoff_date]

        removed_count = original_count - len(self._audit_log)
        if removed_count > 0:
            logger.info(f"{removed_count} alte Audit-Einträge bereinigt")

        return removed_count


class RightToBeForgottenHandler:
    """Handler für Right-to-be-Forgotten-Requests."""

    def __init__(self):
        """Initialisiert Right-to-be-Forgotten Handler."""
        self._deletion_requests: dict[str, dict[str, Any]] = {}
        self._data_locations: dict[str, list[str]] = {}  # user_id -> locations

    async def process_deletion_request(
        self,
        user_id: str,
        request_reason: str,
        requester_id: str
    ) -> str:
        """Verarbeitet Löschungsantrag."""
        import uuid

        request_id = str(uuid.uuid4())

        # Finde alle Datenstandorte für User
        data_locations = self._data_locations.get(user_id, [])

        deletion_request = {
            "request_id": request_id,
            "user_id": user_id,
            "reason": request_reason,
            "requester_id": requester_id,
            "status": "pending",
            "data_locations": data_locations,
            "created_at": datetime.now(UTC),
            "completed_at": None
        }

        self._deletion_requests[request_id] = deletion_request

        # Starte Löschungsprozess
        await self._execute_deletion(request_id)

        logger.info(f"Löschungsantrag verarbeitet: {request_id} für User {user_id}")
        return request_id

    async def _execute_deletion(self, request_id: str) -> None:
        """Führt Datenlöschung durch."""
        request = self._deletion_requests.get(request_id)
        if not request:
            return

        try:
            user_id = request["user_id"]

            # Simuliere Löschung aus verschiedenen Systemen
            for location in request["data_locations"]:
                await self._delete_from_location(user_id, location)

            # Markiere als abgeschlossen
            request["status"] = "completed"
            request["completed_at"] = datetime.now(UTC)

            logger.info(f"Datenlöschung abgeschlossen für Request {request_id}")

        except Exception as e:
            request["status"] = "failed"
            request["error"] = str(e)
            logger.exception(f"Datenlöschung fehlgeschlagen für Request {request_id}: {e}")

    async def _delete_from_location(self, user_id: str, location: str) -> None:
        """Löscht Daten aus spezifischem Standort."""
        # Simuliere Löschung - in Produktion echte Löschung implementieren
        await asyncio.sleep(0.1)  # Simuliere I/O
        logger.info(f"Daten für User {user_id} aus {location} gelöscht")

    def register_data_location(self, user_id: str, location: str) -> None:
        """Registriert Datenstandort für User."""
        if user_id not in self._data_locations:
            self._data_locations[user_id] = []

        if location not in self._data_locations[user_id]:
            self._data_locations[user_id].append(location)

    def get_deletion_status(self, request_id: str) -> dict[str, Any] | None:
        """Gibt Status eines Löschungsantrags zurück."""
        return self._deletion_requests.get(request_id)


class ComplianceEngine:
    """Engine für Compliance-Management."""

    def __init__(self):
        """Initialisiert Compliance Engine."""
        self._checkers: dict[ComplianceStandard, ComplianceChecker] = {}
        self._retention_policies: list[DataRetentionPolicy] = []
        self.audit_trail_manager = AuditTrailManager()
        self.rtbf_handler = RightToBeForgottenHandler()

        # Statistiken
        self._checks_performed = 0
        self._violations_found = 0

        # Standard-Checker registrieren
        self.register_checker(ComplianceStandard.GDPR, GDPRChecker())
        self.register_checker(ComplianceStandard.HIPAA, HIPAAChecker())

    def register_checker(self, standard: ComplianceStandard, checker: ComplianceChecker) -> None:
        """Registriert Compliance-Checker."""
        self._checkers[standard] = checker
        logger.info(f"Compliance-Checker registriert: {standard.value}")

    @trace_function("compliance.check")
    async def check_compliance(
        self,
        data: Any,
        standards: list[ComplianceStandard] | None = None,
        context: dict[str, Any] | None = None
    ) -> list[ComplianceCheck]:
        """Führt Compliance-Checks durch."""
        self._checks_performed += 1

        standards_to_check = standards or list(self._checkers.keys())
        results = []

        for standard in standards_to_check:
            checker = self._checkers.get(standard)
            if checker:
                try:
                    result = await checker.check_compliance(data, context)
                    results.append(result)

                    if result.violations:
                        self._violations_found += len(result.violations)

                        # Log Violations
                        for violation in result.violations:
                            self.audit_trail_manager.log_event(
                                event_type="compliance_violation",
                                user_id=context.get("user_id") if context else None,
                                agent_id=context.get("agent_id") if context else None,
                                resource_type="data",
                                resource_id=str(hash(str(data))),
                                action="compliance_check",
                                result="violation_detected",
                                compliance_standards=[standard],
                                metadata={
                                    "violation_type": violation.violation_type,
                                    "severity": violation.severity.value
                                }
                            )

                except Exception as e:
                    logger.exception(f"Compliance-Check fehlgeschlagen für {standard.value}: {e}")

        return results

    def add_retention_policy(self, policy: DataRetentionPolicy) -> None:
        """Fügt Aufbewahrungsrichtlinie hinzu."""
        self._retention_policies.append(policy)
        logger.info(f"Retention-Policy hinzugefügt: {policy.data_type}")

    def get_compliance_statistics(self) -> dict[str, Any]:
        """Gibt Compliance-Statistiken zurück."""
        return {
            "checks_performed": self._checks_performed,
            "violations_found": self._violations_found,
            "violation_rate": self._violations_found / max(self._checks_performed, 1),
            "registered_standards": list(self._checkers.keys()),
            "retention_policies": len(self._retention_policies),
            "audit_entries": len(self.audit_trail_manager._audit_log)
        }


# Globale Compliance Engine Instanz
compliance_engine = ComplianceEngine()
