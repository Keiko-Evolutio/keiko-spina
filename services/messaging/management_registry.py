"""Persistente Registry für Bus-Management (Topics/Policies/Keys).
Die Persistenz erfolgt einfach als JSON-Datei im Projektverzeichnis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kei_logging import get_logger

logger = get_logger(__name__)

# Konstanten für Topic-Management
_REGISTRY_FILE: Path = Path("management_registry.json").resolve()
DEFAULT_MAX_DELIVERY = 5
DEFAULT_RETENTION_POLICY = "limits"


@dataclass
class TopicPolicy:
    """Policy-Definition für ein Topic/Subject.

    Attributes:
        name: Vollständiger Subject-/Topic-Name oder Pattern
        retention: Retention-Policy (z. B. limits, workqueue)
        max_delivery: Maximale Redeliveries
        allowed_producer_keys: Erlaubte Producer Keys (optional)
        allowed_consumer_keys: Erlaubte Consumer Keys (optional)
    """

    name: str
    retention: str = DEFAULT_RETENTION_POLICY
    max_delivery: int = DEFAULT_MAX_DELIVERY
    allowed_producer_keys: list[str] | None = None
    allowed_consumer_keys: list[str] | None = None


def _load_registry() -> dict[str, Any]:
    """Lädt Registry aus Datei oder liefert Default-Struktur.

    Returns:
        Registry-Dict mit Schlüsseln `topics`.
    """
    try:
        if not _REGISTRY_FILE.exists():
            return {"topics": {}}
        content = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            return {"topics": {}}
        content.setdefault("topics", {})
        return content
    except Exception as exc:  # pragma: no cover - defensiv
        logger.warning(f"Registry laden fehlgeschlagen: {exc}")
        return {"topics": {}}


def _save_registry(data: dict[str, Any]) -> None:
    """Speichert Registry atomares JSON."""
    try:
        tmp = _REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_REGISTRY_FILE)
    except Exception as exc:  # pragma: no cover - defensiv
        logger.exception(f"Registry speichern fehlgeschlagen: {exc}")


async def register_topic(name: str, *, retention: str = DEFAULT_RETENTION_POLICY, max_delivery: int = DEFAULT_MAX_DELIVERY) -> None:
    """Registriert ein Topic/Subject mit Basis-Policy.

    Args:
        name: Subject-/Topic-Name
        retention: Retentions-Policy
        max_delivery: Max Redeliveries
    """
    reg = _load_registry()
    reg["topics"][name] = {
        "name": name,
        "retention": retention,
        "max_delivery": max_delivery,
        "allowed_producer_keys": [],
        "allowed_consumer_keys": [],
    }
    _save_registry(reg)


async def update_topic_policy(name: str, updates: dict[str, Any]) -> None:
    """Aktualisiert Policy-Felder eines Topics.

    Args:
        name: Subject-/Topic-Name
        updates: Zu ändernde Felder
    """
    reg = _load_registry()
    topic = reg["topics"].get(name)
    if not topic:
        topic = {"name": name}
        reg["topics"][name] = topic
    for key, value in updates.items():
        topic[key] = value
    _save_registry(reg)


async def list_topics() -> list[dict[str, Any]]:
    """Listet alle Topics mit Policies."""
    reg = _load_registry()
    return list(reg.get("topics", {}).values())


async def get_topic(name: str) -> dict[str, Any] | None:
    """Gibt Policy eines Topics zurück oder None."""
    reg = _load_registry()
    return reg.get("topics", {}).get(name)


async def set_allowed_keys(pattern: str, role: str, keys: list[str]) -> None:
    """Setzt erlaubte Keys für Producer/Consumer auf allen passenden Topics.

    Args:
        pattern: Exaktes Topic oder Pattern (kein glob; Demo: exaktes Match)
        role: "producer" oder "consumer"
        keys: Liste erlaubter Keys
    """
    reg = _load_registry()
    if role not in {"producer", "consumer"}:
        from core.exceptions import KeikoValidationError
        raise KeikoValidationError("role muss 'producer' oder 'consumer' sein", details={"role": role})
    for name, topic in reg.get("topics", {}).items():
        if name == pattern:
            if role == "producer":
                topic["allowed_producer_keys"] = list(keys)
            else:
                topic["allowed_consumer_keys"] = list(keys)
    _save_registry(reg)


__all__ = [
    "TopicPolicy",
    "get_topic",
    "list_topics",
    "register_topic",
    "set_allowed_keys",
    "update_topic_policy",
]
