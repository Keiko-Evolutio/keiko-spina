# backend/agents/registry/utils/helpers.py
"""Utility-Funktionen für das Registry-System.

Konsolidiert gemeinsame Funktionen und eliminiert Code-Duplikation.
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from .constants import (
    AgentConstants,
    MatchingConstants,
    ValidationConstants,
)
from .types import (
    AgentID,
    CapabilityList,
    MatchScore,
    Timestamp,
)


def generate_agent_id(prefix: str = "agent") -> AgentID:
    """Generiert eine eindeutige Agent-ID.

    Args:
        prefix: Präfix für die ID

    Returns:
        Eindeutige Agent-ID
    """
    unique_suffix = str(uuid.uuid4()).replace("-", "")[:8]
    return f"{prefix}_{unique_suffix}"


def validate_agent_id(agent_id: AgentID) -> bool:
    """Validiert eine Agent-ID.

    Args:
        agent_id: Zu validierende Agent-ID

    Returns:
        True wenn gültig, sonst False
    """
    if not agent_id or not isinstance(agent_id, str):
        return False

    if len(agent_id) > ValidationConstants.MAX_AGENT_NAME_LENGTH:
        return False

    return bool(re.match(ValidationConstants.AGENT_ID_PATTERN, agent_id))


def sanitize_agent_name(name: str) -> str:
    """Bereinigt einen Agent-Namen.

    Args:
        name: Zu bereinigender Name

    Returns:
        Bereinigter Name
    """
    if not name or not isinstance(name, str):
        return ValidationConstants.DEFAULT_AGENT_NAME

    # Entferne führende/nachfolgende Leerzeichen
    sanitized = name.strip()

    # Begrenze Länge
    if len(sanitized) > ValidationConstants.MAX_AGENT_NAME_LENGTH:
        sanitized = sanitized[:ValidationConstants.MAX_AGENT_NAME_LENGTH].strip()

    # Fallback wenn leer
    if not sanitized:
        return ValidationConstants.DEFAULT_AGENT_NAME

    return sanitized


def extract_capabilities(agent: Any) -> CapabilityList:
    """Extrahiert Capabilities aus einem Agent.

    Konsolidierte Logik für Capability-Extraktion aus verschiedenen Agent-Typen.

    Args:
        agent: Agent-Instanz

    Returns:
        Liste der Capabilities
    """
    capabilities: set[str] = set()

    # Direkte Capabilities-Attribute
    if hasattr(agent, "capabilities") and agent.capabilities:
        if isinstance(agent.capabilities, (list, set, tuple)):
            capabilities.update(str(cap).lower() for cap in agent.capabilities)

    # Tool-basierte Capabilities
    if hasattr(agent, "tools") and agent.tools:
        try:
            # Prüfe ob tools iterierbar ist
            tools_list = list(agent.tools) if hasattr(agent.tools, "__iter__") else []
            for tool in tools_list:
                if isinstance(tool, dict) and "type" in tool:
                    capabilities.add(str(tool["type"]).lower())
                elif hasattr(tool, "type"):
                    capabilities.add(str(tool.type).lower())
        except (TypeError, AttributeError):
            # Falls tools nicht iterierbar ist, ignoriere es
            pass

    # Beschreibungsbasierte Capabilities
    if hasattr(agent, "description") and agent.description:
        desc_lower = str(agent.description).lower()
        for capability, keywords in AgentConstants.CAPABILITY_KEYWORDS.items():
            if any(keyword in desc_lower for keyword in keywords):
                capabilities.add(capability)

    # Name-basierte Capabilities
    if hasattr(agent, "name") and agent.name:
        name_lower = str(agent.name).lower()
        for capability, keywords in AgentConstants.CAPABILITY_KEYWORDS.items():
            if any(keyword in name_lower for keyword in keywords):
                capabilities.add(capability)

    return sorted(list(capabilities))


def calculate_match_score(
    agent: Any,
    query: str | None = None,
    required_capabilities: CapabilityList | None = None,
    category: str | None = None,
) -> MatchScore:
    """Berechnet Match-Score für einen Agent.

    Konsolidierte Matching-Logik mit semantischen Konstanten.

    Args:
        agent: Agent-Instanz
        query: Suchtext
        required_capabilities: Erforderliche Capabilities
        category: Gewünschte Kategorie

    Returns:
        Match-Score zwischen 0.0 und 1.0
    """
    score = 0.0

    # Text-Matching
    if query:
        agent_text = _build_agent_text(agent)
        if query.lower() in agent_text.lower():
            score += MatchingConstants.TEXT_MATCH_WEIGHT

    # Capabilities-Matching
    if required_capabilities:
        agent_caps = extract_capabilities(agent)
        matching_caps = set(required_capabilities) & set(agent_caps)
        if matching_caps:
            capability_ratio = len(matching_caps) / len(required_capabilities)
            score += MatchingConstants.CAPABILITY_MATCH_WEIGHT * capability_ratio

    # Category-Matching
    if category:
        agent_category = _infer_agent_category(agent)
        if agent_category == category.lower():
            score += MatchingConstants.CATEGORY_MATCH_WEIGHT

    # Score normalisieren
    return min(score, MatchingConstants.MAX_MATCH_SCORE)


def _build_agent_text(agent: Any) -> str:
    """Erstellt Suchtext aus Agent-Attributen."""
    text_parts = []

    if hasattr(agent, "name") and agent.name:
        text_parts.append(str(agent.name))

    if hasattr(agent, "description") and agent.description:
        text_parts.append(str(agent.description))

    if hasattr(agent, "type") and agent.type:
        text_parts.append(str(agent.type))

    return " ".join(text_parts)


def _infer_agent_category(agent: Any) -> str:
    """Inferiert Agent-Kategorie aus Name und Beschreibung."""
    if hasattr(agent, "category") and agent.category:
        return str(agent.category).lower()

    # Fallback: Inferenz aus Name
    if hasattr(agent, "name") and agent.name:
        name_lower = str(agent.name).lower()
        for category, keywords in AgentConstants.CATEGORY_KEYWORDS.items():
            if any(keyword in name_lower for keyword in keywords):
                return category

    return "custom"


def is_cache_expired(
    last_update: Timestamp | None,
    max_age: timedelta,
    current_time: Timestamp | None = None,
) -> bool:
    """Prüft ob Cache abgelaufen ist.

    Args:
        last_update: Zeitpunkt des letzten Updates
        max_age: Maximales Cache-Alter
        current_time: Aktueller Zeitpunkt (optional)

    Returns:
        True wenn Cache abgelaufen ist
    """
    if not last_update:
        return True

    if current_time is None:
        current_time = datetime.now()

    return current_time - last_update > max_age


def normalize_capability_name(capability: str) -> str:
    """Normalisiert einen Capability-Namen.

    Args:
        capability: Zu normalisierender Capability-Name

    Returns:
        Normalisierter Name
    """
    if not capability or not isinstance(capability, str):
        return ""

    # Zu Kleinbuchstaben, Leerzeichen und Bindestriche durch Unterstriche ersetzen
    normalized = capability.lower().strip().replace(" ", "_").replace("-", "_")

    # Nur alphanumerische Zeichen und Unterstriche behalten
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)

    # Mehrfache Unterstriche reduzieren
    normalized = re.sub(r"_+", "_", normalized)

    # Führende/nachfolgende Unterstriche entfernen
    normalized = normalized.strip("_")

    return normalized


def validate_version_constraint(constraint: str) -> bool:
    """Validiert eine Versions-Constraint.

    Args:
        constraint: Zu validierende Constraint

    Returns:
        True wenn gültig
    """
    if not constraint or not isinstance(constraint, str):
        return False

    # Einfache Semantic Version Validierung
    return bool(re.match(ValidationConstants.VERSION_PATTERN, constraint.strip()))


def merge_capabilities(
    *capability_lists: CapabilityList,
) -> CapabilityList:
    """Merged mehrere Capability-Listen.

    Args:
        *capability_lists: Listen von Capabilities

    Returns:
        Merged und deduplizierte Liste
    """
    merged: set[str] = set()

    for cap_list in capability_lists:
        if cap_list:
            merged.update(normalize_capability_name(cap) for cap in cap_list)

    # Leere Capabilities entfernen
    merged.discard("")

    return sorted(list(merged))
