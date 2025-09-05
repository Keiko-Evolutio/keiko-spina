# backend/policy_engine/redaction_strategies.py
"""Konsolidierte Redaction-Strategien für PII-Redaction.

Vereint die Redaction-Logik aus enhanced_pii_redaction.py und
kei_logging/enhanced_pii_redaction.py in einer gemeinsamen Implementierung.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

# Import konsolidierter Konstanten
from .constants import (
    DEFAULT_ENCRYPT_MODULO,
    DEFAULT_HASH_LENGTH,
    DEFAULT_MASK,
    DEFAULT_TOKEN_MODULO,
    ENCRYPT_FORMAT_WIDTH,
    ENCRYPT_PREFIX,
    HASH_PREFIX,
    TOKEN_FORMAT_WIDTH,
    TOKEN_PREFIX,
)


class RedactionLevel(str, Enum):
    """Konsolidierte Redaction-Level."""
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"
    HASH = "hash"
    TOKENIZE = "tokenize"
    ENCRYPT = "encrypt"
    REMOVE = "remove"


class RedactionStrategy(str, Enum):
    """Konsolidierte Redaction-Strategien."""
    MASK = "mask"                    # Ersetze mit ***
    TOKENIZE = "tokenize"           # Ersetze mit Token
    REMOVE = "remove"               # Entferne komplett
    HASH = "hash"                   # Ersetze mit Hash
    PARTIAL_MASK = "partial_mask"   # Zeige nur Teile
    ENCRYPT = "encrypt"             # Verschlüssele


@dataclass
class RedactionContext:
    """Kontext für Redaction-Operationen."""
    entity_type: str
    original_text: str
    mask: str = DEFAULT_MASK
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRedactionStrategy(ABC):
    """Basis-Klasse für Redaction-Strategien."""

    @abstractmethod
    def apply(self, context: RedactionContext) -> str:
        """Wendet Redaction-Strategie an."""


class MaskRedactionStrategy(BaseRedactionStrategy):
    """Vollständige Maskierung mit konfigurierbarem Mask."""

    def apply(self, context: RedactionContext) -> str:
        """Ersetzt Text vollständig mit Mask."""
        return context.mask


class RemoveRedactionStrategy(BaseRedactionStrategy):
    """Entfernt Text vollständig."""

    def apply(self, context: RedactionContext) -> str:
        """Entfernt Text vollständig."""
        return ""


class HashRedactionStrategy(BaseRedactionStrategy):
    """Hash-basierte Redaction mit konfigurierbarer Länge."""

    def __init__(self, hash_length: int = DEFAULT_HASH_LENGTH):
        self.hash_length = hash_length

    def apply(self, context: RedactionContext) -> str:
        """Ersetzt Text mit Hash-Wert."""
        hash_value = hashlib.sha256(context.original_text.encode()).hexdigest()[:self.hash_length]
        return f"{HASH_PREFIX}{hash_value}]"


class TokenizeRedactionStrategy(BaseRedactionStrategy):
    """Token-basierte Redaction mit konfigurierbarem Modulo."""

    def __init__(self, token_modulo: int = DEFAULT_TOKEN_MODULO):
        self.token_modulo = token_modulo

    def apply(self, context: RedactionContext) -> str:
        """Ersetzt Text mit Token."""
        token_id = hash(context.original_text) % self.token_modulo
        entity_type_upper = context.entity_type.upper()
        return f"{TOKEN_PREFIX}{entity_type_upper}_{token_id:0{TOKEN_FORMAT_WIDTH}d}]"


class PartialMaskRedactionStrategy(BaseRedactionStrategy):
    """Partielle Maskierung - zeigt erste und letzte Zeichen."""

    def apply(self, context: RedactionContext) -> str:
        """Zeigt erste und letzte Zeichen, maskiert den Rest."""
        text = context.original_text

        if len(text) <= 3:
            return context.mask
        if len(text) <= 6:
            return text[0] + context.mask + text[-1]
        return text[:2] + context.mask + text[-2:]


class EncryptRedactionStrategy(BaseRedactionStrategy):
    """Vereinfachte Verschlüsselung (für Demo-Zwecke)."""

    def __init__(self, encrypt_modulo: int = DEFAULT_ENCRYPT_MODULO):
        self.encrypt_modulo = encrypt_modulo

    def apply(self, context: RedactionContext) -> str:
        """Ersetzt Text mit verschlüsseltem Wert."""
        encrypted_id = hash(context.original_text) % self.encrypt_modulo
        return f"{ENCRYPT_PREFIX}{encrypted_id:0{ENCRYPT_FORMAT_WIDTH}d}]"


class ConsolidatedRedactionEngine:
    """Konsolidierte Redaction-Engine."""

    def __init__(self):
        """Initialisiert Redaction-Engine."""
        self._strategies: dict[RedactionStrategy, BaseRedactionStrategy] = {
            RedactionStrategy.MASK: MaskRedactionStrategy(),
            RedactionStrategy.REMOVE: RemoveRedactionStrategy(),
            RedactionStrategy.HASH: HashRedactionStrategy(),
            RedactionStrategy.TOKENIZE: TokenizeRedactionStrategy(),
            RedactionStrategy.PARTIAL_MASK: PartialMaskRedactionStrategy(),
            RedactionStrategy.ENCRYPT: EncryptRedactionStrategy(),
        }

        # Statistiken
        self._redactions_performed = 0
        self._strategies_used = {}

    def redact(
        self,
        text: str,
        strategy: RedactionStrategy,
        entity_type: str = "unknown",
        mask: str = DEFAULT_MASK,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Führt Redaction mit angegebener Strategie durch."""
        if not text:
            return text

        context = RedactionContext(
            entity_type=entity_type,
            original_text=text,
            mask=mask,
            metadata=metadata or {}
        )

        redaction_strategy = self._strategies.get(strategy)
        if not redaction_strategy:
            logger.warning(f"Unbekannte Redaction-Strategie: {strategy}")
            return MaskRedactionStrategy().apply(context)

        try:
            result = redaction_strategy.apply(context)
            self._redactions_performed += 1
            self._strategies_used[strategy.value] = self._strategies_used.get(strategy.value, 0) + 1
            return result
        except Exception as e:
            logger.exception(f"Redaction fehlgeschlagen: {e}")
            return MaskRedactionStrategy().apply(context)

    def register_strategy(self, strategy: RedactionStrategy, implementation: BaseRedactionStrategy) -> None:
        """Registriert eine neue Redaction-Strategie."""
        self._strategies[strategy] = implementation
        logger.info(f"Redaction-Strategie registriert: {strategy.value}")

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Redaction-Statistiken zurück."""
        return {
            "redactions_performed": self._redactions_performed,
            "strategies_used": self._strategies_used.copy(),
            "available_strategies": list(self._strategies.keys()),
            "total_strategies": len(self._strategies)
        }


# Globale Redaction-Engine Instanz
redaction_engine = ConsolidatedRedactionEngine()
