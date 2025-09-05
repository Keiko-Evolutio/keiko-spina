# backend/audit_system/audit_pii_redaction.py
"""Enhanced PII Redaction für Audit-Trails im Keiko Personal Assistant

Implementiert reversible PII-Redaction, GDPR/CCPA-konforme Anonymisierung
und Consent-Management für Audit-Daten.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from cryptography.fernet import Fernet

from kei_logging import get_logger
from observability import trace_function

from .core_audit_engine import AuditContext

if TYPE_CHECKING:
    from .core_audit_engine import AuditEvent

logger = get_logger(__name__)


class RedactionLevel(str, Enum):
    """Level der PII-Redaction."""
    NONE = "none"
    BASIC = "basic"
    ENHANCED = "enhanced"
    FULL = "full"
    IRREVERSIBLE = "irreversible"


class PIICategory(str, Enum):
    """Kategorien von PII-Daten."""
    PERSONAL_IDENTIFIER = "personal_identifier"
    CONTACT_INFO = "contact_info"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    BEHAVIORAL_DATA = "behavioral_data"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ConsentStatus(str, Enum):
    """Status des Consent für Audit-Daten."""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


@dataclass
class PIIAuditPolicy:
    """Policy für PII-Behandlung in Audit-Trails."""
    policy_id: str
    name: str
    description: str

    # Redaction-Konfiguration
    default_redaction_level: RedactionLevel
    category_redaction_levels: dict[PIICategory, RedactionLevel] = field(default_factory=dict)

    # Retention und Consent
    require_consent: bool = True
    consent_expiry_days: int | None = 365
    auto_redact_after_days: int | None = 90

    # Compliance
    gdpr_compliant: bool = True
    ccpa_compliant: bool = True
    hipaa_compliant: bool = False

    # Ausnahmen
    exempted_users: set[str] = field(default_factory=set)
    exempted_agents: set[str] = field(default_factory=set)

    # Gültigkeit
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_redaction_level_for_category(self, category: PIICategory) -> RedactionLevel:
        """Gibt Redaction-Level für Kategorie zurück."""
        return self.category_redaction_levels.get(category, self.default_redaction_level)

    def is_user_exempted(self, user_id: str) -> bool:
        """Prüft, ob User von Redaction ausgenommen ist."""
        return user_id in self.exempted_users

    def is_agent_exempted(self, agent_id: str) -> bool:
        """Prüft, ob Agent von Redaction ausgenommen ist."""
        return agent_id in self.exempted_agents


@dataclass
class ReversibleRedaction:
    """Reversible PII-Redaction mit Verschlüsselung."""
    redaction_id: str
    original_value: str
    redacted_value: str
    pii_category: PIICategory

    # Verschlüsselung
    encrypted_original: bytes
    encryption_key_id: str

    # Metadaten
    redacted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    redacted_by: str | None = None

    # Consent
    consent_id: str | None = None
    consent_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "redaction_id": self.redaction_id,
            "redacted_value": self.redacted_value,
            "pii_category": self.pii_category.value,
            "redacted_at": self.redacted_at.isoformat(),
            "redacted_by": self.redacted_by,
            "consent_id": self.consent_id,
            "consent_required": self.consent_required,
            "encryption_key_id": self.encryption_key_id
        }


@dataclass
class ConsentRecord:
    """Consent-Record für Audit-Daten."""
    consent_id: str
    user_id: str
    purpose: str
    status: ConsentStatus

    # Gültigkeit
    granted_at: datetime | None = None
    expires_at: datetime | None = None
    withdrawn_at: datetime | None = None

    # Scope
    data_categories: set[PIICategory] = field(default_factory=set)
    processing_purposes: set[str] = field(default_factory=set)

    # Metadaten
    consent_method: str | None = None  # "explicit", "implicit", "opt_in"
    consent_evidence: dict[str, Any] | None = None

    # Compliance
    gdpr_lawful_basis: str | None = None
    ccpa_category: str | None = None

    def is_valid(self) -> bool:
        """Prüft, ob Consent gültig ist."""
        if self.status != ConsentStatus.GRANTED:
            return False

        now = datetime.now(UTC)

        if self.expires_at and now > self.expires_at:
            return False

        return not (self.withdrawn_at and now > self.withdrawn_at)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "consent_id": self.consent_id,
            "user_id": self.user_id,
            "purpose": self.purpose,
            "status": self.status.value,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "data_categories": [cat.value for cat in self.data_categories],
            "processing_purposes": list(self.processing_purposes),
            "consent_method": self.consent_method,
            "consent_evidence": self.consent_evidence,
            "gdpr_lawful_basis": self.gdpr_lawful_basis,
            "ccpa_category": self.ccpa_category
        }


class ConsentManager:
    """Manager für Consent-Verwaltung."""

    def __init__(self):
        """Initialisiert Consent Manager."""
        self._consent_records: dict[str, ConsentRecord] = {}
        self._user_consents: dict[str, list[str]] = {}  # user_id -> consent_ids

        # Statistiken
        self._consents_granted = 0
        self._consents_withdrawn = 0
        self._consents_expired = 0

    def grant_consent(
        self,
        user_id: str,
        purpose: str,
        data_categories: set[PIICategory],
        processing_purposes: set[str],
        consent_method: str = "explicit",
        expires_in_days: int | None = 365,
        gdpr_lawful_basis: str | None = None,
        consent_evidence: dict[str, Any] | None = None
    ) -> str:
        """Gewährt Consent.

        Args:
            user_id: User-ID
            purpose: Zweck der Datenverarbeitung
            data_categories: Betroffene Datenkategorien
            processing_purposes: Verarbeitungszwecke
            consent_method: Consent-Methode
            expires_in_days: Gültigkeit in Tagen
            gdpr_lawful_basis: GDPR-Rechtsgrundlage
            consent_evidence: Consent-Nachweis

        Returns:
            Consent-ID
        """
        import uuid

        consent_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            expires_at=now + timedelta(days=expires_in_days) if expires_in_days else None,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            consent_method=consent_method,
            consent_evidence=consent_evidence,
            gdpr_lawful_basis=gdpr_lawful_basis
        )

        self._consent_records[consent_id] = consent

        if user_id not in self._user_consents:
            self._user_consents[user_id] = []
        self._user_consents[user_id].append(consent_id)

        self._consents_granted += 1

        logger.info(f"Consent gewährt: {consent_id} für User {user_id}")
        return consent_id

    def withdraw_consent(self, consent_id: str, reason: str | None = None) -> bool:
        """Widerruft Consent.

        Args:
            consent_id: Consent-ID
            reason: Grund für Widerruf

        Returns:
            True wenn erfolgreich
        """
        consent = self._consent_records.get(consent_id)
        if not consent:
            return False

        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.now(UTC)

        if reason and consent.consent_evidence:
            consent.consent_evidence["withdrawal_reason"] = reason

        self._consents_withdrawn += 1

        logger.info(f"Consent widerrufen: {consent_id}")
        return True

    def check_consent(
        self,
        user_id: str,
        purpose: str,
        data_category: PIICategory
    ) -> ConsentRecord | None:
        """Prüft Consent für spezifischen Zweck.

        Args:
            user_id: User-ID
            purpose: Verarbeitungszweck
            data_category: Datenkategorie

        Returns:
            Gültiger Consent oder None
        """
        user_consent_ids = self._user_consents.get(user_id, [])

        for consent_id in user_consent_ids:
            consent = self._consent_records.get(consent_id)
            if not consent or not consent.is_valid():
                continue

            if (purpose in consent.processing_purposes and
                data_category in consent.data_categories):
                return consent

        return None

    def get_user_consents(self, user_id: str) -> list[ConsentRecord]:
        """Gibt alle Consents für User zurück."""
        user_consent_ids = self._user_consents.get(user_id, [])
        return [self._consent_records[cid] for cid in user_consent_ids if cid in self._consent_records]

    def cleanup_expired_consents(self) -> int:
        """Bereinigt abgelaufene Consents."""
        now = datetime.now(UTC)
        expired_count = 0

        for consent in self._consent_records.values():
            if (consent.status == ConsentStatus.GRANTED and
                consent.expires_at and
                now > consent.expires_at):
                consent.status = ConsentStatus.EXPIRED
                expired_count += 1

        self._consents_expired += expired_count
        return expired_count

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Consent-Manager-Statistiken zurück."""
        return {
            "total_consents": len(self._consent_records),
            "consents_granted": self._consents_granted,
            "consents_withdrawn": self._consents_withdrawn,
            "consents_expired": self._consents_expired,
            "active_users": len(self._user_consents)
        }


class AuditAnonymizer:
    """Anonymisierer für Audit-Daten."""

    def __init__(self):
        """Initialisiert Audit Anonymizer."""
        self._anonymization_cache: dict[str, str] = {}
        self._salt = secrets.token_bytes(32)

    def anonymize_identifier(self, identifier: str, category: PIICategory) -> str:
        """Anonymisiert Identifier.

        Args:
            identifier: Zu anonymisierender Identifier
            category: PII-Kategorie

        Returns:
            Anonymisierter Identifier
        """
        # Cache-Check
        cache_key = f"{category.value}:{identifier}"
        if cache_key in self._anonymization_cache:
            return self._anonymization_cache[cache_key]

        # Erstelle anonymisierten Identifier
        if category == PIICategory.PERSONAL_IDENTIFIER:
            # Für persönliche Identifier: Hash mit Salt
            anonymized = self._hash_with_salt(identifier, "user")
        elif category == PIICategory.CONTACT_INFO:
            # Für Kontaktdaten: Domänen-erhaltende Anonymisierung
            anonymized = self._anonymize_contact_info(identifier)
        elif category == PIICategory.LOCATION_DATA:
            # Für Standortdaten: Geo-Anonymisierung
            anonymized = self._anonymize_location(identifier)
        else:
            # Standard-Hash für andere Kategorien
            anonymized = self._hash_with_salt(identifier, str(category.value) if hasattr(category, "value") else str(category))

        # Cache Ergebnis
        self._anonymization_cache[cache_key] = anonymized

        return anonymized

    def _hash_with_salt(self, value: str, prefix: str) -> str:
        """Erstellt Hash mit Salt."""
        combined = f"{prefix}:{value}".encode()
        hash_value = hmac.new(self._salt, combined, hashlib.sha256).hexdigest()
        return f"{prefix}_{hash_value[:16]}"

    def _anonymize_contact_info(self, contact: str) -> str:
        """Anonymisiert Kontaktinformationen."""
        if "@" in contact:  # E-Mail
            local, domain = contact.split("@", 1)
            anonymized_local = self._hash_with_salt(local, "email")[:8]
            return f"{anonymized_local}@{domain}"
        if contact.startswith("+") or contact.isdigit():  # Telefonnummer
            return f"+{self._hash_with_salt(contact, 'phone')[:10]}"
        return self._hash_with_salt(contact, "contact")

    def _anonymize_location(self, location: str) -> str:
        """Anonymisiert Standortdaten."""
        # Vereinfachte Geo-Anonymisierung
        return f"location_{self._hash_with_salt(location, 'geo')[:12]}"

    def get_anonymization_statistics(self) -> dict[str, Any]:
        """Gibt Anonymisierungs-Statistiken zurück."""
        return {
            "cached_anonymizations": len(self._anonymization_cache),
            "salt_length": len(self._salt)
        }


class AuditPIIRedactor:
    """Hauptklasse für PII-Redaction in Audit-Trails."""

    def __init__(self):
        """Initialisiert Audit PII Redactor."""
        self.consent_manager = ConsentManager()
        self.anonymizer = AuditAnonymizer()

        # Verschlüsselung für reversible Redaction
        self._encryption_key = Fernet.generate_key()
        self._fernet = Fernet(self._encryption_key)

        # Redaction-Policies
        self._policies: dict[str, PIIAuditPolicy] = {}
        self._default_policy = self._create_default_policy()

        # Redaction-Records
        self._redaction_records: dict[str, ReversibleRedaction] = {}

        # Statistiken
        self._redactions_performed = 0
        self._reversals_performed = 0
        self._consent_checks = 0

    def _create_default_policy(self) -> PIIAuditPolicy:
        """Erstellt Standard-PII-Policy."""
        return PIIAuditPolicy(
            policy_id="default_audit_pii_policy",
            name="Default Audit PII Policy",
            description="Standard-Policy für PII-Redaction in Audit-Trails",
            default_redaction_level=RedactionLevel.ENHANCED,
            category_redaction_levels={
                PIICategory.PERSONAL_IDENTIFIER: RedactionLevel.FULL,
                PIICategory.FINANCIAL_DATA: RedactionLevel.FULL,
                PIICategory.HEALTH_DATA: RedactionLevel.FULL,
                PIICategory.CONTACT_INFO: RedactionLevel.ENHANCED,
                PIICategory.LOCATION_DATA: RedactionLevel.BASIC
            },
            require_consent=True,
            gdpr_compliant=True,
            ccpa_compliant=True
        )

    def register_policy(self, policy: PIIAuditPolicy) -> None:
        """Registriert PII-Policy."""
        self._policies[policy.policy_id] = policy
        logger.info(f"PII-Policy registriert: {policy.policy_id}")

    @trace_function("audit_pii.redact_event")
    async def redact_audit_event(
        self,
        event: AuditEvent,
        policy_id: str | None = None,
        consent_override: bool = False
    ) -> AuditEvent:
        """Redaktiert PII in Audit-Event.

        Args:
            event: Zu redaktierendes Event
            policy_id: Policy-ID (Standard wenn None)
            consent_override: Consent-Prüfung überspringen

        Returns:
            Redaktiertes Event
        """
        policy = self._policies.get(policy_id) or self._default_policy

        # Prüfe Ausnahmen
        if ((event.context.user_id and policy.is_user_exempted(event.context.user_id)) or
            (event.context.agent_id and policy.is_agent_exempted(event.context.agent_id))):
            return event

        # Redaktiere verschiedene Event-Bereiche
        if event.input_data:
            event.input_data = await self._redact_data_dict(
                event.input_data, policy, event.context.user_id, consent_override
            )

        if event.output_data:
            event.output_data = await self._redact_data_dict(
                event.output_data, policy, event.context.user_id, consent_override
            )

        # Redaktiere Kontext-Daten
        event.context = await self._redact_context(
            event.context, policy, consent_override
        )

        self._redactions_performed += 1

        return event

    async def _redact_data_dict(
        self,
        data: dict[str, Any],
        policy: PIIAuditPolicy,
        user_id: str | None,
        consent_override: bool
    ) -> dict[str, Any]:
        """Redaktiert PII in Dictionary."""
        redacted_data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                redacted_data[key] = await self._redact_data_dict(
                    value, policy, user_id, consent_override
                )
            elif isinstance(value, list):
                redacted_data[key] = [
                    await self._redact_data_dict(item, policy, user_id, consent_override)
                    if isinstance(item, dict) else await self._redact_value(
                        item, key, policy, user_id, consent_override
                    )
                    for item in value
                ]
            else:
                redacted_data[key] = await self._redact_value(
                    value, key, policy, user_id, consent_override
                )

        return redacted_data

    async def _redact_value(
        self,
        value: Any,
        field_name: str,
        policy: PIIAuditPolicy,
        user_id: str | None,
        consent_override: bool
    ) -> Any:
        """Redaktiert einzelnen Wert."""
        if not isinstance(value, str):
            return value

        # Bestimme PII-Kategorie basierend auf Feldname
        pii_category = self._detect_pii_category(field_name, value)
        if pii_category is None:
            return value

        # Prüfe Consent
        if not consent_override and policy.require_consent and user_id:
            self._consent_checks += 1
            consent = self.consent_manager.check_consent(
                user_id, "audit_logging", pii_category
            )
            if not consent:
                # Kein Consent - vollständige Redaction
                return self._apply_redaction(value, pii_category, RedactionLevel.FULL)

        # Wende Policy-Redaction an
        redaction_level = policy.get_redaction_level_for_category(pii_category)
        return self._apply_redaction(value, pii_category, redaction_level)

    def _detect_pii_category(self, field_name: str, value: str) -> PIICategory | None:
        """Detektiert PII-Kategorie basierend auf Feldname und Wert."""
        field_lower = field_name.lower()

        # Personal Identifiers
        if any(keyword in field_lower for keyword in ["user_id", "username", "name", "id"]):
            return PIICategory.PERSONAL_IDENTIFIER

        # Contact Info
        if any(keyword in field_lower for keyword in ["email", "phone", "address"]):
            return PIICategory.CONTACT_INFO

        # Financial Data
        if any(keyword in field_lower for keyword in ["credit", "card", "payment", "bank"]):
            return PIICategory.FINANCIAL_DATA

        # Location Data
        if any(keyword in field_lower for keyword in ["location", "address", "gps", "coordinates"]):
            return PIICategory.LOCATION_DATA

        # Pattern-basierte Detection
        if "@" in value and "." in value:  # E-Mail-Pattern
            return PIICategory.CONTACT_INFO

        if value.startswith("+") or (value.isdigit() and len(value) > 8):  # Telefon-Pattern
            return PIICategory.CONTACT_INFO

        return None

    def _apply_redaction(
        self,
        value: str,
        category: PIICategory,
        level: RedactionLevel
    ) -> str:
        """Wendet Redaction-Level an."""
        if level == RedactionLevel.NONE:
            return value

        if level == RedactionLevel.BASIC:
            # Teilweise Maskierung
            if len(value) <= 4:
                return "*" * len(value)
            return value[:2] + "*" * (len(value) - 4) + value[-2:]

        if level == RedactionLevel.ENHANCED:
            # Reversible Redaction
            return self._create_reversible_redaction(value, category)

        if level == RedactionLevel.FULL:
            # Vollständige Redaction mit Typ-Erhaltung
            return f"[REDACTED_{category.value.upper()}]"

        if level == RedactionLevel.IRREVERSIBLE:
            # Irreversible Anonymisierung
            return self.anonymizer.anonymize_identifier(value, category)

        # Fallback: Rückgabe des ursprünglichen Werts
        return value

    def _create_reversible_redaction(self, value: str, category: PIICategory) -> str:
        """Erstellt reversible Redaction."""
        import uuid

        redaction_id = str(uuid.uuid4())

        # Verschlüssele Original-Wert
        encrypted_original = self._fernet.encrypt(value.encode("utf-8"))

        # Erstelle Redaction-Record
        redaction = ReversibleRedaction(
            redaction_id=redaction_id,
            original_value=value,
            redacted_value=f"[REDACTED_{category.value.upper()}_{redaction_id[:8]}]",
            pii_category=category,
            encrypted_original=encrypted_original,
            encryption_key_id="default_key"
        )

        self._redaction_records[redaction_id] = redaction

        return redaction.redacted_value

    async def _redact_context(
        self,
        context: AuditContext,
        policy: PIIAuditPolicy,
        consent_override: bool
    ) -> AuditContext:
        """Redaktiert Audit-Context."""
        # User-ID redaktieren falls erforderlich
        if context.user_id and not consent_override:
            context.user_id = await self._redact_value(
                context.user_id, "user_id", policy, context.user_id, consent_override
            )

        # Client-IP redaktieren
        if context.client_ip:
            context.client_ip = await self._redact_value(
                context.client_ip, "client_ip", policy, context.user_id, consent_override
            )

        return context

    async def reverse_redaction(
        self,
        redacted_value: str,
        authorized_user: str,
        purpose: str
    ) -> str | None:
        """Macht Redaction rückgängig (für autorisierte Benutzer).

        Args:
            redacted_value: Redaktierter Wert
            authorized_user: Autorisierter Benutzer
            purpose: Zweck der Rückgängigmachung

        Returns:
            Original-Wert oder None
        """
        # Extrahiere Redaction-ID aus redaktiertem Wert
        if not redacted_value.startswith("[REDACTED_"):
            return None

        try:
            # Parse Redaction-ID
            parts = redacted_value.split("_")
            if len(parts) < 3:
                return None

            redaction_id_part = parts[-1].rstrip("]")

            # Finde passende Redaction
            for redaction_id, redaction in self._redaction_records.items():
                if redaction_id.startswith(redaction_id_part):
                    # Entschlüssele Original-Wert
                    original_bytes = self._fernet.decrypt(redaction.encrypted_original)
                    original_value = original_bytes.decode("utf-8")

                    self._reversals_performed += 1

                    logger.info(f"Redaction rückgängig gemacht: {redaction_id} von {authorized_user}")
                    return original_value

        except Exception as e:
            logger.exception(f"Redaction-Rückgängigmachung fehlgeschlagen: {e}")

        return None

    def get_redaction_statistics(self) -> dict[str, Any]:
        """Gibt Redaction-Statistiken zurück."""
        return {
            "redactions_performed": self._redactions_performed,
            "reversals_performed": self._reversals_performed,
            "consent_checks": self._consent_checks,
            "redaction_records": len(self._redaction_records),
            "registered_policies": len(self._policies),
            "consent_manager_stats": self.consent_manager.get_statistics(),
            "anonymizer_stats": self.anonymizer.get_anonymization_statistics()
        }


# Globale Audit PII Redactor Instanz
audit_pii_redactor = AuditPIIRedactor()
