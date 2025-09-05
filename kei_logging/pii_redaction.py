"""PII-Redaction für Logging und Metriken.

Dieses Modul stellt konfigurierbare Redaction-Regeln bereit, die vor der
Persistierung/Emission von Logs und Metriken angewendet werden. Sensible
Felder in Strukturen werden maskiert; freie Texte werden per Regex bereinigt.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

_DEFAULT_MASK = os.getenv("KEI_LOGGING_PII_MASK", "***REDACTED***")


@dataclass
class PIIRedactionConfig:
    """Konfiguration für PII-Redaction.

    Attributes:
        mask: Maskierungsstring
        field_names: Feldnamen (case-insensitive) die maskiert werden
        regex_patterns: Kompilierte Regex-Muster für freie Texte
    """

    mask: str
    field_names: list[str]
    regex_patterns: list[re.Pattern[str]]


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    """Kompiliert Regex-Muster robust (IGNORECASE)."""
    compiled: list[re.Pattern[str]] = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except Exception:
            # Ungültige Muster ignorieren
            continue
    return compiled


def _default_patterns() -> list[str]:
    """Liefert Standard-Patterndefinitionen für PII."""
    return [
        # E-Mail
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        # Telefonnummern (einfach)
        r"\+?\d[\d\s().-]{6,}\d",
        # Kreditkarten (einfach, 13-19 Ziffern mit Trennern)
        r"\b(?:\d[ -]*?){13,19}\b",
        # IBAN (einfach)
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
        # US SSN (xxx-xx-xxxx)
        r"\b\d{3}-\d{2}-\d{4}\b",
        # JWT/Token-ähnlich (Base64URL Segmente)
        r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}",
    ]


def _default_field_names() -> list[str]:
    """Liefert Standard-Feldnamen für strukturierte Redaction."""
    return [
        "password",
        "pass",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "id_token",
        "refresh_token",
        "authorization",
        "auth",
        "ssn",
        "iban",
        "credit_card",
        "card_number",
        "email",
        "phone",
        "tel",
        "cookie",
        "set-cookie",
    ]


def load_pii_config() -> PIIRedactionConfig:
    """Lädt Redaction-Konfiguration aus ENV JSON und Defaults."""
    mask = os.getenv("KEI_LOGGING_PII_MASK", _DEFAULT_MASK)
    raw = os.getenv("KEI_LOGGING_PII_RULES", "").strip()
    patterns: list[str] = _default_patterns()
    fields: list[str] = _default_field_names()
    if raw:
        try:
            cfg = json.loads(raw)
            if isinstance(cfg.get("patterns"), list):
                patterns = list(cfg["patterns"])  # type: ignore[assignment]
            if isinstance(cfg.get("fields"), list):
                fields = [str(x) for x in cfg["fields"]]
            mask = str(cfg.get("mask", mask))
        except Exception:
            pass
    return PIIRedactionConfig(mask=mask, field_names=fields, regex_patterns=_compile_patterns(patterns))


_GLOBAL_CFG = load_pii_config()


def redact_text(text: str, cfg: PIIRedactionConfig | None = None) -> str:
    """Reduziert PII in freien Texten gemäß Regex-Patterns."""
    if not text:
        return text
    c = cfg or _GLOBAL_CFG
    out = text
    for pat in c.regex_patterns:
        out = pat.sub(c.mask, out)
    return out


def _should_redact_key(key: str, field_names: Iterable[str]) -> bool:
    """Prüft, ob ein Schlüsselname zu maskieren ist (case-insensitive, enthält)."""
    k = key.lower()
    for f in field_names:
        f_l = f.lower()
        if k == f_l or k.endswith(f"_{f_l}") or f_l in k:
            return True
    return False


def redact_structure(obj: Any, cfg: PIIRedactionConfig | None = None) -> Any:
    """Reduziert PII in verschachtelten Strukturen (dict/list/tuple/str)."""
    c = cfg or _GLOBAL_CFG
    try:
        if obj is None:
            return None
        if isinstance(obj, str):
            return redact_text(obj, c)
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if _should_redact_key(str(k), c.field_names):
                    out[k] = c.mask
                else:
                    out[k] = redact_structure(v, c)
            return out
        if isinstance(obj, list | tuple):
            seq = [redact_structure(v, c) for v in obj]
            return tuple(seq) if isinstance(obj, tuple) else seq
        return obj
    except Exception:
        return obj


def redact_tags(tags: dict[str, Any] | None, cfg: PIIRedactionConfig | None = None) -> dict[str, Any] | None:
    """Reduziert PII in Metrics-Tags (flache Struktur)."""
    if not tags:
        return tags
    return redact_structure(tags, cfg)


class PIIRedactionFilter(logging.Filter):
    """Logging-Filter, der PII in msg/args reduziert bevor formatiert wird."""

    def __init__(self, cfg: PIIRedactionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or _GLOBAL_CFG

    def filter(self, record: logging.LogRecord) -> bool:
        """Wird vor Formatter aufgerufen: msg/args redaktieren."""
        try:
            if isinstance(record.msg, str):
                record.msg = redact_text(record.msg, self.cfg)
            # Args redaktieren (tuple, dict)
            if hasattr(record, "args") and record.args:
                if isinstance(record.args, dict):
                    record.args = redact_structure(record.args, self.cfg)
                elif isinstance(record.args, tuple):
                    record.args = tuple(redact_structure(a, self.cfg) for a in record.args)
            # Zusatzfelder (üblich bei strukturierter Log-Nutzung)
            for extra_key in ("payload", "headers", "context", "tags"):
                if hasattr(record, extra_key):
                    setattr(record, extra_key, redact_structure(getattr(record, extra_key), self.cfg))
        except Exception:
            # Fehler in Filter dürfen Logging nicht verhindern
            return True
        return True


__all__ = [
    "PIIRedactionConfig",
    "PIIRedactionFilter",
    "load_pii_config",
    "redact_structure",
    "redact_tags",
    "redact_text",
]
