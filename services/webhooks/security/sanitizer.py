"""Sanitizer für eingehende Webhook‑Payloads.

Diese Komponente bietet XSS‑Schutz, einfache SQL‑Injection‑Detektion sowie
Content‑Validierung nach konfigurierbaren Regeln. Sie wird im Inbound‑Flow
verwendet, um Payloads vor weiterer Verarbeitung zu bereinigen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Any

from fastapi import HTTPException

from kei_logging import get_logger

logger = get_logger(__name__)


_SCRIPT_TAG_RE = re.compile(r"<\s*/?\s*script[^>]*>", re.IGNORECASE)
_EVENT_ATTR_RE = re.compile(r"on[a-zA-Z]+\s*=\s*\"[^\"]*\"|on[a-zA-Z]+\s*=\s*'[^']*'", re.IGNORECASE)
_JS_PROTOCOL_RE = re.compile(r"javascript:\s*", re.IGNORECASE)

_SQLI_PATTERNS = [
    re.compile(r"\bunion\b\s+\bselect\b", re.IGNORECASE),
    re.compile(r"(--|#).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r";\s*drop\s+table", re.IGNORECASE),
    re.compile(r";\s*update\s+", re.IGNORECASE),
    re.compile(r"\bor\b\s+1\s*=\s*1", re.IGNORECASE),
]


@dataclass
class SanitizerConfig:
    """Konfiguration für den WebhookPayloadSanitizer.

    Attributes:
        allowed_html_tags: Liste erlaubter HTML‑Tags (ohne spitze Klammern)
        max_payload_bytes: Maximale Größe der Roh‑Payload in Bytes
        allowed_content_types: Whitelist erlaubter Content‑Types
    """

    allowed_html_tags: list[str]
    max_payload_bytes: int
    allowed_content_types: list[str]


class WebhookPayloadSanitizer:
    """Bietet Sanitization und Validierung für eingehende Payloads."""

    def __init__(self, config: SanitizerConfig) -> None:
        self.config = config

    def validate_envelope(self, *, raw: bytes, content_type: str) -> None:
        """Validiert Envelope‑Eigenschaften (Größe/Content‑Type).

        Raises:
            HTTPException: wenn Größe oder Content‑Type gegen Richtlinien verstößt
        """
        if len(raw) > self.config.max_payload_bytes:
            raise HTTPException(status_code=413, detail="Payload too large")
        if not any(content_type.lower().startswith(typ.lower()) for typ in self.config.allowed_content_types):
            raise HTTPException(status_code=415, detail="Unsupported content-type")

    def sanitize_json(self, obj: Any) -> Any:
        """Sanitisiert ein JSON‑Objekt rekursiv (Strings, Objekte, Listen).

        Entfernt gefährliche HTML‑Tags/Attribute aus Strings, neutralisiert
        `javascript:`‑URLs und prüft auf offensichtliche SQL‑Injection‑Muster.
        """
        if obj is None:
            return None
        if isinstance(obj, str):
            return self._sanitize_string(obj)
        if isinstance(obj, dict):
            return {k: self.sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.sanitize_json(v) for v in obj]
        return obj

    def _sanitize_string(self, text: str) -> str:
        """Sanitisiert einen String gegen XSS/SQLi."""
        value = unescape(text)
        # XSS: entferne <script> tags
        value = _SCRIPT_TAG_RE.sub("", value)
        # XSS: entferne on* Event‑Attribute in HTML‑Attributen
        value = _EVENT_ATTR_RE.sub("", value)
        # XSS: neutralisiere javascript: URLs
        value = _JS_PROTOCOL_RE.sub("", value)
        # HTML‑Tag Whitelist durch einfache Entfernung nicht erlaubter Tags
        value = self._strip_disallowed_tags(value)
        # SQLi Heuristik: für Logging
        try:
            for pattern in _SQLI_PATTERNS:
                if pattern.search(value or ""):
                    logger.warning("Sanitizer: Mögliche SQL‑Injection erkannt")
                    break
        except Exception:
            pass
        return value

    def _strip_disallowed_tags(self, text: str) -> str:
        """Entfernt HTML‑Tags außerhalb der Whitelist (rudimentär)."""
        # Erlaube einfache Inline‑Tags wie <b>, <i>, <u> etc.
        allowed = {t.lower() for t in self.config.allowed_html_tags}

        def _replace_tag(match: re.Match[str]) -> str:
            tag = match.group(1) or ""
            if tag.lower() in allowed:
                return match.group(0)
            return ""

        # Regex für generische Tags <tag ...> oder </tag>
        tag_re = re.compile(r"<\s*/?\s*([a-zA-Z0-9]+)(?:\s+[^>]*)?>", re.IGNORECASE)
        return tag_re.sub(_replace_tag, text)


__all__ = [
    "SanitizerConfig",
    "WebhookPayloadSanitizer",
]
