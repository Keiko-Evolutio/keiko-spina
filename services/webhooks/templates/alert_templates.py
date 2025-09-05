"""Vorlagen für Alert-Nachrichten (deutsche Lokalisierung).

Enthält kompakte Render-Funktionen für Slack/Teams, E-Mail und SMS.
"""

from __future__ import annotations

from typing import Any


def render_compact_summary(*, labels: dict[str, Any], annotations: dict[str, Any], tenant_id: str | None) -> dict[str, Any]:
    """Erzeugt eine kompakte Zusammenfassung für Chat-Kanäle.

    Args:
        labels: Alert-Labels
        annotations: Alert-Annotationen
        tenant_id: Optionaler Tenant

    Returns:
        Strukturierte, kompakte Nachricht
    """
    return {
        "alert": labels.get("alertname", "KeikoAlert"),
        "severity": labels.get("severity", "warning"),
        "tenant": tenant_id,
        "summary": annotations.get("summary") or annotations.get("description") or "Keiko Alert",
        "source": annotations.get("generatorURL") or labels.get("instance"),
    }


def render_email_subject(*, title: str, severity: str) -> str:
    """Erzeugt einen E-Mail-Betreff in Deutsch.

    Args:
        title: Kurztitel des Alerts
        severity: Schweregrad

    Returns:
        E-Mail Betreff
    """
    sev = severity.upper()
    return f"[{sev}] {title}"


def render_email_body(*, labels: dict[str, Any], annotations: dict[str, Any], tenant_id: str | None) -> str:
    """Erzeugt einen einfachen E-Mail-Body in Deutsch.

    Args:
        labels: Alert-Labels
        annotations: Alert-Annotationen
        tenant_id: Optionaler Tenant

    Returns:
        E-Mail Textkörper (Plain-Text)
    """
    lines = [
        f"Alarm: {labels.get('alertname', 'KeikoAlert')}",
        f"Severity: {labels.get('severity', 'warning')}",
        f"Tenant: {tenant_id or '-'}",
        f"Zusammenfassung: {annotations.get('summary') or annotations.get('description') or '-'}",
    ]
    if labels.get("instance"):
        lines.append(f"Quelle: {labels.get('instance')}")
    return "\n".join(lines)


def render_sms_text(*, title: str, labels: dict[str, Any], tenant_id: str | None) -> str:
    """Erzeugt einen sehr kompakten SMS-Text.

    Args:
        title: Titel
        labels: Alert-Labels
        tenant_id: Optionaler Tenant

    Returns:
        SMS-Text
    """
    sev = str(labels.get("severity", "warning")).upper()
    tenant_txt = tenant_id or "-"
    return f"{sev} {tenant_txt}: {title}"


__all__ = [
    "render_compact_summary",
    "render_email_body",
    "render_email_subject",
    "render_sms_text",
]
