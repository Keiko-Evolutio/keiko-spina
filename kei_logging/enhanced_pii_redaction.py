# backend/kei_logging/enhanced_pii_redaction.py
"""Erweiterte PII-Redaction für Keiko Personal Assistant

Überprüft und erweitert die bestehende PII-Redaction-Funktionalität mit
konfigurierbaren Redaction-Patterns für verschiedene Datentypen und
zusätzlichen sensitiven Feldern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any

from .clickable_logging_formatter import get_logger
from .pii_redaction import redact_structure, redact_text

logger = get_logger(__name__)


class RedactionLevel(str, Enum):
    """Redaction-Level für verschiedene Sensitivitäts-Grade."""
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"
    HASH = "hash"
    TOKENIZE = "tokenize"


class DataType(str, Enum):
    """Datentypen für spezifische Redaction-Patterns."""
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    URL = "url"
    USER_AGENT = "user_agent"
    SESSION_ID = "session_id"
    CORRELATION_ID = "correlation_id"
    PERSONAL_NAME = "personal_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_DATA = "medical_data"
    FINANCIAL_DATA = "financial_data"
    BIOMETRIC_DATA = "biometric_data"


@dataclass
class RedactionPattern:
    """Pattern für spezifische Redaction-Regeln."""
    data_type: DataType
    pattern: Pattern[str]
    redaction_level: RedactionLevel
    replacement_template: str
    description: str

    def apply(self, text: str, mask: str = "***REDACTED***") -> str:
        """Wendet Redaction-Pattern auf Text an.

        Args:
            text: Zu redaktierender Text
            mask: Redaction-Mask

        Returns:
            Redaktierter Text
        """
        if self.redaction_level == RedactionLevel.NONE:
            return text

        def replace_match(match):
            matched_text = match.group()

            if self.redaction_level == RedactionLevel.FULL:
                return mask
            if self.redaction_level == RedactionLevel.PARTIAL:
                return self._partial_redact(matched_text, mask)
            if self.redaction_level == RedactionLevel.HASH:
                return self._hash_redact(matched_text)
            if self.redaction_level == RedactionLevel.TOKENIZE:
                return self._tokenize_redact(matched_text)
            return mask

        return self.pattern.sub(replace_match, text)

    def _partial_redact(self, text: str, mask: str) -> str:
        """Partielle Redaction (konsolidiert)."""
        from policy_engine.redaction_strategies import RedactionStrategy, redaction_engine
        return redaction_engine.redact(text, RedactionStrategy.PARTIAL_MASK, self.data_type.value, mask)

    def _hash_redact(self, text: str) -> str:
        """Hash-basierte Redaction (konsolidiert)."""
        from policy_engine.redaction_strategies import RedactionStrategy, redaction_engine
        return redaction_engine.redact(text, RedactionStrategy.HASH, self.data_type.value)

    def _tokenize_redact(self, text: str) -> str:
        """Token-basierte Redaction (konsolidiert)."""
        from policy_engine.redaction_strategies import RedactionStrategy, redaction_engine
        return redaction_engine.redact(text, RedactionStrategy.TOKENIZE, self.data_type.value)


@dataclass
class EnhancedPIIRedactionConfig:
    """Erweiterte Konfiguration für PII-Redaction."""
    # Basis-Konfiguration
    mask: str = "***REDACTED***"
    enabled: bool = True

    # Erweiterte Field-Namen (zusätzlich zu bestehenden)
    additional_field_names: set[str] = field(default_factory=lambda: {
        # Authentifizierung
        "access_token", "refresh_token", "bearer_token", "jwt_token",
        "client_secret", "client_id", "oauth_token", "auth_header",
        "authorization", "x-api-key", "x-auth-token",

        # Persönliche Daten
        "first_name", "last_name", "full_name", "display_name",
        "date_of_birth", "birth_date", "age", "gender",
        "address", "street", "city", "postal_code", "zip_code",
        "country", "state", "province",

        # Kontaktdaten
        "email_address", "phone_number", "mobile", "telephone",
        "fax", "contact_info",

        # Finanzielle Daten
        "credit_card_number", "card_number", "cvv", "cvc",
        "iban", "bic", "account_number", "routing_number",
        "bank_account", "payment_info",

        # Medizinische Daten
        "medical_record", "diagnosis", "treatment", "medication",
        "health_info", "patient_id", "medical_id",

        # Biometrische Daten
        "fingerprint", "face_id", "voice_print", "retina_scan",
        "biometric_data", "biometric_id",

        # System-IDs
        "user_agent", "device_id", "machine_id", "hardware_id",
        "mac_address", "serial_number",

        # Session-Daten
        "session_token", "csrf_token", "state_token",
        "nonce", "challenge", "verification_code"
    })

    # Redaction-Level pro Datentyp
    redaction_levels: dict[DataType, RedactionLevel] = field(default_factory=lambda: {
        DataType.PASSWORD: RedactionLevel.FULL,
        DataType.API_KEY: RedactionLevel.FULL,
        DataType.TOKEN: RedactionLevel.FULL,
        DataType.EMAIL: RedactionLevel.PARTIAL,
        DataType.PHONE: RedactionLevel.PARTIAL,
        DataType.SSN: RedactionLevel.FULL,
        DataType.CREDIT_CARD: RedactionLevel.PARTIAL,
        DataType.IBAN: RedactionLevel.PARTIAL,
        DataType.IP_ADDRESS: RedactionLevel.PARTIAL,
        DataType.SESSION_ID: RedactionLevel.HASH,
        DataType.CORRELATION_ID: RedactionLevel.TOKENIZE,
        DataType.PERSONAL_NAME: RedactionLevel.PARTIAL,
        DataType.ADDRESS: RedactionLevel.PARTIAL,
        DataType.DATE_OF_BIRTH: RedactionLevel.PARTIAL,
        DataType.MEDICAL_DATA: RedactionLevel.FULL,
        DataType.FINANCIAL_DATA: RedactionLevel.FULL,
        DataType.BIOMETRIC_DATA: RedactionLevel.FULL
    })

    # Custom Patterns
    custom_patterns: list[RedactionPattern] = field(default_factory=list)

    # Whitelist für Ausnahmen
    whitelist_patterns: list[Pattern[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert zu Dictionary."""
        return {
            "mask": self.mask,
            "enabled": self.enabled,
            "additional_field_names": list(self.additional_field_names),
            "redaction_levels": {dt.value: level.value for dt, level in self.redaction_levels.items()},
            "custom_patterns_count": len(self.custom_patterns),
            "whitelist_patterns_count": len(self.whitelist_patterns)
        }


class EnhancedPIIRedactor:
    """Erweiterte PII-Redaction mit konfigurierbaren Patterns."""

    def __init__(self, config: EnhancedPIIRedactionConfig | None = None):
        """Initialisiert Enhanced PII Redactor.

        Args:
            config: Erweiterte PII-Redaction-Konfiguration
        """
        self.config = config or EnhancedPIIRedactionConfig()
        self._patterns = self._build_patterns()

        # Statistiken
        self._redactions_performed = 0
        self._patterns_matched = 0
        self._whitelist_matches = 0

    def _build_patterns(self) -> list[RedactionPattern]:
        """Erstellt Redaction-Patterns für verschiedene Datentypen.

        Returns:
            Liste von Redaction-Patterns
        """
        # Optimierte List-Creation mit direkter Literal-Syntax
        patterns = [
            # Password-Patterns
            RedactionPattern(
                data_type=DataType.PASSWORD,
                pattern=re.compile(r'(?i)(?:password|pwd|pass)\s*[:=]\s*["\']?([^"\'\s]+)["\']?'),
                redaction_level=self.config.redaction_levels.get(DataType.PASSWORD, RedactionLevel.FULL),
                replacement_template="password=***REDACTED***",
                description="Password in key-value pairs"
            ),
            # API-Key-Patterns
            RedactionPattern(
                data_type=DataType.API_KEY,
                pattern=re.compile(r'(?i)(?:api[_-]?key|access[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?'),
                redaction_level=self.config.redaction_levels.get(DataType.API_KEY, RedactionLevel.FULL),
                replacement_template="api_key=***REDACTED***",
                description="API keys"
            ),
            # Token-Patterns
            RedactionPattern(
                data_type=DataType.TOKEN,
                pattern=re.compile(r'(?i)(?:bearer\s+|token\s*[:=]\s*)["\']?([a-zA-Z0-9_.-]{20,})["\']?'),
                redaction_level=self.config.redaction_levels.get(DataType.TOKEN, RedactionLevel.FULL),
                replacement_template="token=***REDACTED***",
                description="Bearer tokens and access tokens"
            ),
            # Email-Patterns
            RedactionPattern(
                data_type=DataType.EMAIL,
                pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
                redaction_level=self.config.redaction_levels.get(DataType.EMAIL, RedactionLevel.PARTIAL),
                replacement_template="***@***.***",
                description="Email addresses"
            ),
            # Phone-Patterns
            RedactionPattern(
                data_type=DataType.PHONE,
                pattern=re.compile(r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"),
                redaction_level=self.config.redaction_levels.get(DataType.PHONE, RedactionLevel.PARTIAL),
                replacement_template="***-***-****",
                description="Phone numbers"
            ),
            # SSN-Patterns
            RedactionPattern(
                data_type=DataType.SSN,
                pattern=re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
                redaction_level=self.config.redaction_levels.get(DataType.SSN, RedactionLevel.FULL),
                replacement_template="***-**-****",
                description="Social Security Numbers"
            ),
            # Credit Card-Patterns
            RedactionPattern(
                data_type=DataType.CREDIT_CARD,
                pattern=re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
                redaction_level=self.config.redaction_levels.get(DataType.CREDIT_CARD, RedactionLevel.PARTIAL),
                replacement_template="****-****-****-****",
                description="Credit card numbers"
            ),
            # IBAN-Patterns
            RedactionPattern(
                data_type=DataType.IBAN,
                pattern=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"),
                redaction_level=self.config.redaction_levels.get(DataType.IBAN, RedactionLevel.PARTIAL),
                replacement_template="DE**************",
                description="IBAN numbers"
            ),
            # IP-Address-Patterns
            RedactionPattern(
                data_type=DataType.IP_ADDRESS,
                pattern=re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
                redaction_level=self.config.redaction_levels.get(DataType.IP_ADDRESS, RedactionLevel.PARTIAL),
                replacement_template="***.***.***.***",
                description="IPv4 addresses"
            ),
            # Session-ID-Patterns
            RedactionPattern(
                data_type=DataType.SESSION_ID,
                pattern=re.compile(r'(?i)session[_-]?id\s*[:=]\s*["\']?([a-zA-Z0-9_-]{16,})["\']?'),
                redaction_level=self.config.redaction_levels.get(DataType.SESSION_ID, RedactionLevel.HASH),
                replacement_template="session_id=[HASH]",
                description="Session IDs"
            ),
            # User-Agent-Patterns
            RedactionPattern(
                data_type=DataType.USER_AGENT,
                pattern=re.compile(r"(?i)user-agent\s*:\s*([^\r\n]+)"),
                redaction_level=self.config.redaction_levels.get(DataType.USER_AGENT, RedactionLevel.PARTIAL),
                replacement_template="User-Agent: ***REDACTED***",
                description="User-Agent strings"
            ),
        ]

        # Füge Custom-Patterns hinzu
        patterns.extend(self.config.custom_patterns)

        return patterns

    def redact_text(self, text: str) -> str:
        """Redaktiert PII in Text mit erweiterten Patterns.

        Args:
            text: Zu redaktierender Text

        Returns:
            Redaktierter Text
        """
        if not self.config.enabled or not text:
            return text

        # Prüfe Whitelist
        for whitelist_pattern in self.config.whitelist_patterns:
            if whitelist_pattern.search(text):
                self._whitelist_matches += 1
                return text

        redacted_text = text

        # Wende alle Patterns an
        for pattern in self._patterns:
            if pattern.pattern.search(redacted_text):
                self._patterns_matched += 1
                redacted_text = pattern.apply(redacted_text, self.config.mask)

        # Fallback auf Legacy-Redaction
        redacted_text = redact_text(redacted_text)

        if redacted_text != text:
            self._redactions_performed += 1

        return redacted_text

    def redact_structure(self, obj: Any) -> Any:
        """Redaktiert PII in verschachtelten Strukturen.

        Args:
            obj: Zu redaktierende Struktur

        Returns:
            Redaktierte Struktur
        """
        if not self.config.enabled:
            return obj

        try:
            if obj is None:
                return None

            if isinstance(obj, str):
                return self.redact_text(obj)

            if isinstance(obj, dict):
                redacted_dict = {}
                for key, value in obj.items():
                    key_str = str(key).lower()

                    # Prüfe erweiterte Field-Namen
                    if any(field_name in key_str for field_name in self.config.additional_field_names):
                        redacted_dict[key] = self.config.mask
                    else:
                        redacted_dict[key] = self.redact_structure(value)

                return redacted_dict

            if isinstance(obj, list | tuple):
                redacted_seq = [self.redact_structure(item) for item in obj]
                return tuple(redacted_seq) if isinstance(obj, tuple) else redacted_seq

            return obj

        except Exception as e:
            logger.warning(f"Enhanced PII-Redaction fehlgeschlagen: {e}")
            # Fallback auf Legacy-Redaction
            return redact_structure(obj)

    def add_custom_pattern(
        self,
        data_type: DataType,
        pattern: str,
        redaction_level: RedactionLevel,
        description: str
    ) -> None:
        """Fügt Custom-Redaction-Pattern hinzu.

        Args:
            data_type: Datentyp
            pattern: Regex-Pattern
            redaction_level: Redaction-Level
            description: Beschreibung
        """
        try:
            compiled_pattern = re.compile(pattern)
            custom_pattern = RedactionPattern(
                data_type=data_type,
                pattern=compiled_pattern,
                redaction_level=redaction_level,
                replacement_template="***REDACTED***",
                description=description
            )

            self.config.custom_patterns.append(custom_pattern)
            self._patterns.append(custom_pattern)

            logger.info(f"Custom-Redaction-Pattern hinzugefügt: {description}")

        except re.error as e:
            logger.exception(f"Ungültiges Regex-Pattern: {pattern} - {e}")

    def add_whitelist_pattern(self, pattern: str) -> None:
        """Fügt Whitelist-Pattern hinzu.

        Args:
            pattern: Regex-Pattern für Whitelist
        """
        try:
            compiled_pattern = re.compile(pattern)
            self.config.whitelist_patterns.append(compiled_pattern)

            logger.info(f"Whitelist-Pattern hinzugefügt: {pattern}")

        except re.error as e:
            logger.exception(f"Ungültiges Whitelist-Pattern: {pattern} - {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Redaction-Statistiken zurück.

        Returns:
            Statistiken-Dictionary
        """
        return {
            "config": self.config.to_dict(),
            "redactions_performed": self._redactions_performed,
            "patterns_matched": self._patterns_matched,
            "whitelist_matches": self._whitelist_matches,
            "total_patterns": len(self._patterns),
            "custom_patterns": len(self.config.custom_patterns),
            "whitelist_patterns": len(self.config.whitelist_patterns)
        }


class EnhancedPIIRedactionFilter(logging.Filter):
    """Erweiterte Logging-Filter für PII-Redaction."""

    def __init__(self, config: EnhancedPIIRedactionConfig | None = None):
        """Initialisiert Enhanced PII Redaction Filter.

        Args:
            config: Erweiterte PII-Redaction-Konfiguration
        """
        super().__init__()
        self.redactor = EnhancedPIIRedactor(config)

    def filter(self, record: logging.LogRecord) -> bool:
        """Wendet erweiterte PII-Redaction auf Log-Record an.

        Args:
            record: Log-Record

        Returns:
            True (Record wird nicht gefiltert)
        """
        try:
            # Redaktiere Message
            if isinstance(record.msg, str):
                record.msg = self.redactor.redact_text(record.msg)

            # Redaktiere Args
            if hasattr(record, "args") and record.args:
                if isinstance(record.args, dict):
                    record.args = self.redactor.redact_structure(record.args)
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        self.redactor.redact_structure(arg) for arg in record.args
                    )

            # Redaktiere Extra-Felder
            for extra_key in ("payload", "headers", "context", "tags", "fields", "additional_context"):
                if hasattr(record, extra_key):
                    setattr(
                        record,
                        extra_key,
                        self.redactor.redact_structure(getattr(record, extra_key))
                    )

        except Exception as e:
            logger.warning(f"Enhanced PII-Redaction-Filter-Fehler: {e}")

        return True


# Globale Enhanced PII Redactor Instanz
_global_redactor: EnhancedPIIRedactor | None = None


def get_enhanced_pii_redactor() -> EnhancedPIIRedactor:
    """Holt oder erstellt globale Enhanced PII Redactor Instanz.

    Returns:
        Globale Enhanced PII Redactor Instanz
    """
    global _global_redactor

    if _global_redactor is None:
        _global_redactor = EnhancedPIIRedactor()

    return _global_redactor


def setup_enhanced_pii_redaction(config: EnhancedPIIRedactionConfig | None = None) -> EnhancedPIIRedactor:
    """Setzt erweiterte PII-Redaction auf.

    Args:
        config: Erweiterte PII-Redaction-Konfiguration

    Returns:
        Konfigurierte Enhanced PII Redactor Instanz
    """
    global _global_redactor

    _global_redactor = EnhancedPIIRedactor(config)

    logger.info("Erweiterte PII-Redaction erfolgreich konfiguriert")

    return _global_redactor


def enhanced_redact_text(text: str) -> str:
    """Redaktiert Text mit erweiterter PII-Redaction.

    Args:
        text: Zu redaktierender Text

    Returns:
        Redaktierter Text
    """
    redactor = get_enhanced_pii_redactor()
    return redactor.redact_text(text)


def enhanced_redact_structure(obj: Any) -> Any:
    """Redaktiert Struktur mit erweiterter PII-Redaction.

    Args:
        obj: Zu redaktierende Struktur

    Returns:
        Redaktierte Struktur
    """
    redactor = get_enhanced_pii_redactor()
    return redactor.redact_structure(obj)
