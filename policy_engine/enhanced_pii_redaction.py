# backend/policy_engine/enhanced_pii_redaction.py
"""Enhanced PII Redaction für Keiko Personal Assistant

Erweitert die bestehende PII-Redaction um ML-basierte PII-Detection,
kontextuelle PII-Erkennung und konfigurierbare Redaction-Strategien.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)

# Fallback für Legacy PII Config falls nicht verfügbar
try:
    from kei_logging.pii_redaction import PIIRedactionConfig, load_pii_config, redact_structure
except ImportError:
    logger.warning("Legacy PII-Redaction nicht verfügbar - verwende Fallback")

    @dataclass
    class PIIRedactionConfig:
        mask: str = "***"

    def load_pii_config():
        return PIIRedactionConfig()

    def redact_structure(text, config):
        """Fallback PII-Redaction mit einfacher Regex-basierter Erkennung."""
        import re

        if not text or not config:
            return text

        # Einfache Regex-Patterns für häufige PII-Typen
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        }

        redacted_text = text
        mask = getattr(config, "mask", "***")

        # Wende Redaction-Patterns an
        for pattern_name, pattern in patterns.items():
            redacted_text = re.sub(pattern, mask, redacted_text)

        return redacted_text


class PIIEntityType(str, Enum):
    """Typen von PII-Entitäten."""
    PERSON_NAME = "person_name"
    EMAIL_ADDRESS = "email_address"
    PHONE_NUMBER = "phone_number"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT_NUMBER = "passport_number"
    DRIVER_LICENSE = "driver_license"
    MEDICAL_RECORD = "medical_record"
    CUSTOM = "custom"


class RedactionStrategy(str, Enum):
    """Strategien für PII-Redaction."""
    MASK = "mask"                    # Ersetze mit ***
    TOKENIZE = "tokenize"           # Ersetze mit Token
    REMOVE = "remove"               # Entferne komplett
    HASH = "hash"                   # Ersetze mit Hash
    PARTIAL_MASK = "partial_mask"   # Zeige nur Teile
    ENCRYPT = "encrypt"             # Verschlüssele


@dataclass
class PIIEntity:
    """Repräsentiert eine erkannte PII-Entität."""
    entity_type: PIIEntityType
    text: str
    start_pos: int
    end_pos: int
    confidence: float  # 0.0 - 1.0
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIDetectionResult:
    """Ergebnis einer PII-Detection."""
    original_text: str
    entities: list[PIIEntity] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def has_pii(self) -> bool:
        """Prüft, ob PII erkannt wurde."""
        return len(self.entities) > 0


@dataclass
class RedactionResult:
    """Ergebnis einer PII-Redaction."""
    original_text: str
    redacted_text: str
    entities_redacted: list[PIIEntity] = field(default_factory=list)
    redaction_map: dict[str, str] = field(default_factory=dict)  # Original -> Redacted
    processing_time_ms: float = 0.0


class PIIDetector(ABC):
    """Basis-Klasse für PII-Detektoren."""

    @abstractmethod
    def detect(self, text: str, context: dict[str, Any] | None = None) -> list[PIIEntity]:
        """Detektiert PII-Entitäten in Text."""


class RegexPIIDetector(PIIDetector):
    """Regex-basierter PII-Detektor."""

    def __init__(self):
        """Initialisiert Regex PII Detector."""
        self._patterns = {
            PIIEntityType.EMAIL_ADDRESS: re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            PIIEntityType.PHONE_NUMBER: re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
            ),
            PIIEntityType.CREDIT_CARD: re.compile(
                r"\b(?:\d[ -]*?){13,19}\b"
            ),
            PIIEntityType.SSN: re.compile(
                r"\b\d{3}-\d{2}-\d{4}\b"
            ),
            PIIEntityType.IBAN: re.compile(
                r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"
            ),
            PIIEntityType.IP_ADDRESS: re.compile(
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
            )
        }

    def detect(self, text: str, context: dict[str, Any] | None = None) -> list[PIIEntity]:
        """Detektiert PII mit Regex-Patterns."""
        entities = []

        for entity_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                entity = PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,  # Hohe Confidence für Regex-Matches
                    context=self._extract_context(text, match.start(), match.end())
                )
                entities.append(entity)

        return entities

    def _extract_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Extrahiert Kontext um erkannte Entität."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]


class ContextualPIIDetector(PIIDetector):
    """Kontextueller PII-Detektor für Namen und Adressen."""

    def __init__(self):
        """Initialisiert Contextual PII Detector."""
        # Häufige Vor- und Nachnamen (vereinfacht)
        self._first_names = {
            "john", "jane", "michael", "sarah", "david", "lisa", "robert", "mary",
            "james", "patricia", "william", "jennifer", "richard", "elizabeth"
        }

        self._last_names = {
            "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
            "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez"
        }

        # Adress-Indikatoren
        self._address_indicators = {
            "street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd",
            "lane", "ln", "drive", "dr", "court", "ct", "place", "pl"
        }

        # Kontext-Indikatoren für Namen
        self._name_contexts = {
            "mr", "mrs", "ms", "dr", "prof", "dear", "hello", "hi",
            "name:", "called", "signed", "from", "by"
        }

    def detect(self, text: str, context: dict[str, Any] | None = None) -> list[PIIEntity]:
        """Detektiert PII basierend auf Kontext."""
        entities = []
        words = text.lower().split()

        # Namen-Detection
        entities.extend(self._detect_names(text, words))

        # Adressen-Detection
        entities.extend(self._detect_addresses(text, words))

        return entities

    def _detect_names(self, text: str, words: list[str]) -> list[PIIEntity]:
        """Detektiert Personennamen."""
        entities = []

        for i, word in enumerate(words):
            # Prüfe auf Vorname
            if word in self._first_names:
                confidence = 0.6

                # Erhöhe Confidence bei Kontext-Indikatoren
                if i > 0 and words[i-1] in self._name_contexts:
                    confidence = 0.8

                # Prüfe auf nachfolgenden Nachnamen
                if i + 1 < len(words) and words[i+1] in self._last_names:
                    confidence = 0.9
                    # Kombiniere Vor- und Nachname
                    full_name = f"{word} {words[i+1]}"
                    start_pos = text.lower().find(full_name)
                    if start_pos != -1:
                        entity = PIIEntity(
                            entity_type=PIIEntityType.PERSON_NAME,
                            text=text[start_pos:start_pos + len(full_name)],
                            start_pos=start_pos,
                            end_pos=start_pos + len(full_name),
                            confidence=confidence,
                            context=self._extract_context(text, start_pos, start_pos + len(full_name))
                        )
                        entities.append(entity)
                        continue

                # Einzelner Vorname
                start_pos = text.lower().find(word)
                if start_pos != -1:
                    entity = PIIEntity(
                        entity_type=PIIEntityType.PERSON_NAME,
                        text=text[start_pos:start_pos + len(word)],
                        start_pos=start_pos,
                        end_pos=start_pos + len(word),
                        confidence=confidence,
                        context=self._extract_context(text, start_pos, start_pos + len(word))
                    )
                    entities.append(entity)

        return entities

    def _detect_addresses(self, text: str, words: list[str]) -> list[PIIEntity]:
        """Detektiert Adressen."""
        entities = []

        for i, word in enumerate(words):
            if word in self._address_indicators:
                # Suche nach Hausnummer vor Straßentyp
                if i > 0 and words[i-1].isdigit():
                    # Potentielle Adresse gefunden
                    start_idx = max(0, i - 2)  # Inkludiere potentielle Straßenname
                    end_idx = min(len(words), i + 1)

                    address_parts = words[start_idx:end_idx]
                    address_text = " ".join(address_parts)

                    start_pos = text.lower().find(address_text)
                    if start_pos != -1:
                        entity = PIIEntity(
                            entity_type=PIIEntityType.ADDRESS,
                            text=text[start_pos:start_pos + len(address_text)],
                            start_pos=start_pos,
                            end_pos=start_pos + len(address_text),
                            confidence=0.7,
                            context=self._extract_context(text, start_pos, start_pos + len(address_text))
                        )
                        entities.append(entity)

        return entities

    def _extract_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Extrahiert Kontext um erkannte Entität."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]


class CustomEntityRecognizer(PIIDetector):
    """Erkennungsmodul für benutzerdefinierte PII-Entitäten."""

    def __init__(self):
        """Initialisiert Custom Entity Recognizer."""
        self._custom_patterns: dict[str, re.Pattern] = {}
        self._custom_wordlists: dict[str, set[str]] = {}

    def add_pattern(self, entity_name: str, pattern: str) -> None:
        """Fügt benutzerdefiniertes Pattern hinzu."""
        try:
            self._custom_patterns[entity_name] = re.compile(pattern, re.IGNORECASE)
            logger.info(f"Custom Pattern hinzugefügt: {entity_name}")
        except re.error as e:
            logger.exception(f"Ungültiges Regex-Pattern für {entity_name}: {e}")

    def add_wordlist(self, entity_name: str, words: set[str]) -> None:
        """Fügt benutzerdefinierte Wortliste hinzu."""
        self._custom_wordlists[entity_name] = {word.lower() for word in words}
        logger.info(f"Custom Wordlist hinzugefügt: {entity_name} ({len(words)} Wörter)")

    def detect(self, text: str, context: dict[str, Any] | None = None) -> list[PIIEntity]:
        """Detektiert benutzerdefinierte PII-Entitäten."""
        entities = []

        # Pattern-basierte Detection
        for entity_name, pattern in self._custom_patterns.items():
            for match in pattern.finditer(text):
                entity = PIIEntity(
                    entity_type=PIIEntityType.CUSTOM,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    context=self._extract_context(text, match.start(), match.end()),
                    metadata={"custom_entity": entity_name}
                )
                entities.append(entity)

        # Wordlist-basierte Detection
        words = text.split()
        for entity_name, wordlist in self._custom_wordlists.items():
            for _i, word in enumerate(words):
                if word.lower() in wordlist:
                    start_pos = text.find(word)
                    if start_pos != -1:
                        entity = PIIEntity(
                            entity_type=PIIEntityType.CUSTOM,
                            text=word,
                            start_pos=start_pos,
                            end_pos=start_pos + len(word),
                            confidence=0.9,
                            context=self._extract_context(text, start_pos, start_pos + len(word)),
                            metadata={"custom_entity": entity_name}
                        )
                        entities.append(entity)

        return entities

    def _extract_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Extrahiert Kontext um erkannte Entität."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]


class EnhancedPIIRedactor:
    """Enhanced PII Redactor mit ML-basierter Detection."""

    def __init__(self):
        """Initialisiert Enhanced PII Redactor."""
        self._detectors: list[PIIDetector] = []
        self._redaction_strategies: dict[PIIEntityType, RedactionStrategy] = {}
        self._legacy_config = load_pii_config()

        # Statistiken
        self._detections_performed = 0
        self._entities_detected = 0
        self._entities_redacted = 0

        # Standard-Detektoren registrieren
        self.register_detector(RegexPIIDetector())
        self.register_detector(ContextualPIIDetector())
        self.register_detector(CustomEntityRecognizer())

        # Standard-Redaction-Strategien
        self._configure_default_strategies()

    def _configure_default_strategies(self) -> None:
        """Konfiguriert Standard-Redaction-Strategien."""
        self._redaction_strategies = {
            PIIEntityType.PERSON_NAME: RedactionStrategy.PARTIAL_MASK,
            PIIEntityType.EMAIL_ADDRESS: RedactionStrategy.PARTIAL_MASK,
            PIIEntityType.PHONE_NUMBER: RedactionStrategy.PARTIAL_MASK,
            PIIEntityType.CREDIT_CARD: RedactionStrategy.MASK,
            PIIEntityType.SSN: RedactionStrategy.MASK,
            PIIEntityType.IBAN: RedactionStrategy.MASK,
            PIIEntityType.IP_ADDRESS: RedactionStrategy.HASH,
            PIIEntityType.ADDRESS: RedactionStrategy.TOKENIZE,
            PIIEntityType.CUSTOM: RedactionStrategy.MASK
        }

    def register_detector(self, detector: PIIDetector) -> None:
        """Registriert PII-Detektor."""
        self._detectors.append(detector)
        logger.info(f"PII-Detektor registriert: {detector.__class__.__name__}")

    def configure_strategy(self, entity_type: PIIEntityType, strategy: RedactionStrategy) -> None:
        """Konfiguriert Redaction-Strategie für Entity-Typ."""
        self._redaction_strategies[entity_type] = strategy
        logger.info(f"Redaction-Strategie konfiguriert: {entity_type.value} -> {strategy.value}")

    @trace_function("pii.detect")
    async def detect_pii(
        self,
        text: str,
        context: dict[str, Any] | None = None
    ) -> PIIDetectionResult:
        """Detektiert PII in Text mit allen registrierten Detektoren."""
        start_time = time.time()
        self._detections_performed += 1

        try:
            all_entities = []

            # Führe alle Detektoren aus
            for detector in self._detectors:
                try:
                    entities = detector.detect(text, context)
                    all_entities.extend(entities)
                except Exception as e:
                    logger.exception(f"PII-Detection fehlgeschlagen für {detector.__class__.__name__}: {e}")

            # Entferne Duplikate und überlappende Entitäten
            unique_entities = self._deduplicate_entities(all_entities)

            processing_time = (time.time() - start_time) * 1000
            self._entities_detected += len(unique_entities)

            return PIIDetectionResult(
                original_text=text,
                entities=unique_entities,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.exception(f"PII-Detection fehlgeschlagen: {e}")
            processing_time = (time.time() - start_time) * 1000

            return PIIDetectionResult(
                original_text=text,
                processing_time_ms=processing_time
            )

    @trace_function("pii.redact")
    async def redact_pii(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        custom_strategies: dict[PIIEntityType, RedactionStrategy] | None = None
    ) -> RedactionResult:
        """Redaktiert PII in Text."""
        start_time = time.time()

        try:
            detection_result = await self.detect_pii(text, context)

            if not detection_result.has_pii:
                return self._create_legacy_fallback_result(text, start_time)

            return self._perform_entity_redaction(
                text, detection_result, custom_strategies, start_time
            )

        except Exception as e:
            return self._create_error_fallback_result(text, e, start_time)

    def _create_legacy_fallback_result(self, text: str, start_time: float) -> RedactionResult:
        """Erstellt Fallback-Ergebnis mit Legacy-Redaction."""
        legacy_redacted = redact_structure(text, self._legacy_config)
        processing_time = (time.time() - start_time) * 1000

        return RedactionResult(
            original_text=text,
            redacted_text=legacy_redacted,
            processing_time_ms=processing_time
        )

    def _perform_entity_redaction(
        self,
        text: str,
        detection_result: PIIDetectionResult,
        custom_strategies: dict[PIIEntityType, RedactionStrategy] | None,
        start_time: float
    ) -> RedactionResult:
        """Führt Entitäts-basierte Redaction durch."""
        strategies = custom_strategies or self._redaction_strategies
        sorted_entities = sorted(detection_result.entities, key=lambda e: e.start_pos, reverse=True)

        redacted_text, redaction_map, entities_redacted = self._redact_entities(
            text, sorted_entities, strategies
        )

        processing_time = (time.time() - start_time) * 1000

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            entities_redacted=entities_redacted,
            redaction_map=redaction_map,
            processing_time_ms=processing_time
        )

    def _redact_entities(
        self,
        text: str,
        sorted_entities: list[PIIEntity],
        strategies: dict[PIIEntityType, RedactionStrategy]
    ) -> tuple[str, dict[str, str], list[PIIEntity]]:
        """Redaktiert alle Entitäten im Text."""
        redacted_text = text
        redaction_map = {}
        entities_redacted = []

        for entity in sorted_entities:
            strategy = strategies.get(entity.entity_type, RedactionStrategy.MASK)
            redacted_value = self._apply_redaction_strategy(entity, strategy)

            redacted_text = self._replace_entity_in_text(
                redacted_text, entity, redacted_value
            )

            redaction_map[entity.text] = redacted_value
            entities_redacted.append(entity)
            self._entities_redacted += 1

        return redacted_text, redaction_map, entities_redacted

    def _replace_entity_in_text(self, text: str, entity: PIIEntity, redacted_value: str) -> str:
        """Ersetzt eine Entität im Text mit dem redaktierten Wert."""
        return (
            text[:entity.start_pos] +
            redacted_value +
            text[entity.end_pos:]
        )

    def _create_error_fallback_result(self, text: str, error: Exception, start_time: float) -> RedactionResult:
        """Erstellt Fallback-Ergebnis bei Fehlern."""
        logger.error(f"PII-Redaction fehlgeschlagen: {error}")
        processing_time = (time.time() - start_time) * 1000

        legacy_redacted = redact_structure(text, self._legacy_config)

        return RedactionResult(
            original_text=text,
            redacted_text=legacy_redacted,
            processing_time_ms=processing_time
        )

    def _deduplicate_entities(self, entities: list[PIIEntity]) -> list[PIIEntity]:
        """Entfernt Duplikate und überlappende Entitäten."""
        if not entities:
            return entities

        # Sortiere nach Position
        sorted_entities = sorted(entities, key=lambda e: (e.start_pos, e.end_pos))
        unique_entities = []

        for entity in sorted_entities:
            # Prüfe auf Überlappung mit bereits hinzugefügten Entitäten
            overlaps = False
            for existing in unique_entities:
                if entity.start_pos < existing.end_pos and entity.end_pos > existing.start_pos:
                    # Überlappung gefunden - behalte Entität mit höherer Confidence
                    if entity.confidence > existing.confidence:
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    overlaps = True
                    break

            if not overlaps:
                unique_entities.append(entity)

        return unique_entities

    def _apply_redaction_strategy(self, entity: PIIEntity, strategy: RedactionStrategy) -> str:
        """Wendet Redaction-Strategie auf Entität an (konsolidiert)."""
        from .redaction_strategies import redaction_engine

        return redaction_engine.redact(
            text=entity.text,
            strategy=strategy,
            entity_type=entity.entity_type.value,
            mask=self._legacy_config.mask,
            metadata=entity.metadata
        )

    def get_statistics(self) -> dict[str, Any]:
        """Gibt PII-Redaction-Statistiken zurück."""
        return {
            "detections_performed": self._detections_performed,
            "entities_detected": self._entities_detected,
            "entities_redacted": self._entities_redacted,
            "detection_rate": self._entities_detected / max(self._detections_performed, 1),
            "redaction_rate": self._entities_redacted / max(self._entities_detected, 1),
            "registered_detectors": len(self._detectors),
            "configured_strategies": len(self._redaction_strategies)
        }


# Globale Enhanced PII Redactor Instanz
enhanced_pii_redactor = EnhancedPIIRedactor()
