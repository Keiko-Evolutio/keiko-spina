# backend/services/limits/rate_limiter.py
"""Rate Limiter Adapter für Backward-Compatibility.

Nutzt das zentrale backend/quotas_limits System und bietet
Backward-Compatibility für bestehende API-Endpunkte.
"""
from __future__ import annotations

import os

from kei_logging import get_logger
from quotas_limits import AgentRateLimit, RateLimitWindow, agent_rate_limiter

# Type Aliases für bessere Lesbarkeit
RateLimitHeaders = dict[str, str]
RateLimitResponse = tuple[bool, RateLimitHeaders, int | None, str | None]
CameraLimitResponse = tuple[bool, RateLimitHeaders, int | None]

logger = get_logger(__name__)

# Legacy-Konstanten für Backward-Compatibility
USER_LIMIT: int = 10
USER_WINDOW_SECONDS: int = 3600
SESSION_LIMIT: int = 3
SESSION_WINDOW_SECONDS: int = 600

# Kamera-Capture: 10 Fotos / Minute / User
CAMERA_USER_PER_MIN_LIMIT: int = 10
CAMERA_USER_PER_MIN_WINDOW_SECONDS: int = 60

# Agent/Capability Quoten (konfigurierbar per ENV)
DEFAULT_AGENT_LIMIT: int = int(os.getenv("KEI_AGENT_PER_MIN_LIMIT", "120"))
DEFAULT_AGENT_WINDOW_SECONDS: int = int(os.getenv("KEI_AGENT_PER_MIN_WINDOW_SECONDS", "60"))
DEFAULT_CAPABILITY_LIMIT: int = int(os.getenv("KEI_CAPABILITY_PER_MIN_LIMIT", "300"))
DEFAULT_CAPABILITY_WINDOW_SECONDS: int = int(
    os.getenv("KEI_CAPABILITY_PER_MIN_WINDOW_SECONDS", "60")
)


def _build_rate_limit_headers(
    limit_type: str,
    limit: int,
    remaining: int
) -> dict[str, str]:
    """Erstellt standardisierte Rate-Limit-Header für HTTP-Responses.

    Args:
        limit_type: Typ des Limits (z.B. "User", "Session", "Agent")
        limit: Maximales Limit für den Zeitraum
        remaining: Verbleibende Anfragen im aktuellen Zeitfenster

    Returns:
        Dictionary mit Rate-Limit-Headern für HTTP-Response
    """
    return {
        f"X-RateLimit-Limit-{limit_type}": str(limit),
        f"X-RateLimit-Remaining-{limit_type}": str(max(0, remaining))
    }


async def _ensure_agent_limit_exists(
    agent_id: str,
    limit: float,
    _window: RateLimitWindow
) -> None:
    """Stellt sicher, dass ein Agent-Limit im Rate-Limiter registriert ist.

    Erstellt automatisch ein Default-Limit falls noch keines existiert.
    Dies ermöglicht die nahtlose Integration mit dem zentralen Quota-System.

    Args:
        agent_id: Eindeutige Agent-ID für Rate-Limiting
        limit: Basis-Limit pro Minute
        _window: Zeitfenster für das Rate-Limiting (wird für Skalierung verwendet)
    """
    if not agent_rate_limiter._agent_limits.get(agent_id):
        # Erstelle Default-Limit für Agent basierend auf Zeitfenster
        agent_limit = AgentRateLimit(
            agent_id=agent_id,
            requests_per_minute=limit,
            requests_per_hour=limit * 60,
            requests_per_day=limit * 60 * 24,
            enabled=True
        )
        agent_rate_limiter._agent_limits[agent_id] = agent_limit


async def check_image_limits(
    user_id: str | None,
    session_id: str | None
) -> RateLimitResponse:
    """Prüft Bildgenerierungs-Limits mit zentralem Quota-System.

    Args:
        user_id: Benutzer-ID für User-spezifische Limits
        session_id: Session-ID für Session-spezifische Limits

    Returns:
        allowed: Ob Anfrage erlaubt ist
        headers: Rate-Limit-Header für HTTP-Response
        retry_after: Sekunden bis zum Reset (falls limitiert)
        limited_scope: "user" | "session" | None
    """
    headers: dict[str, str] = {}

    # User-Limit prüfen
    if user_id:
        user_agent_id = f"user:{user_id}"
        await _ensure_agent_limit_exists(user_agent_id, USER_LIMIT, RateLimitWindow.PER_HOUR)

        user_result = await agent_rate_limiter.check_agent_rate_limit(
            agent_id=user_agent_id,
            window=RateLimitWindow.PER_HOUR,
            requested_amount=1.0,
            context={"operation": "image_generation", "scope": "user"}
        )

        headers.update(_build_rate_limit_headers("User", USER_LIMIT, int(user_result.remaining)))

        if not user_result.allowed:
            return False, headers, user_result.retry_after_seconds, "user"

    # Session-Limit prüfen (10-Minuten-Fenster simuliert durch PER_MINUTE mit angepasstem Limit)
    if session_id:
        session_agent_id = f"session:{session_id}"
        # Session-Limit: 3 pro 10 Minuten = 0.3 pro Minute
        session_limit_per_minute = SESSION_LIMIT / 10.0
        await _ensure_agent_limit_exists(
            session_agent_id, session_limit_per_minute, RateLimitWindow.PER_MINUTE
        )

        session_result = await agent_rate_limiter.check_agent_rate_limit(
            agent_id=session_agent_id,
            window=RateLimitWindow.PER_MINUTE,
            requested_amount=1.0,
            context={"operation": "image_generation", "scope": "session"}
        )

        headers.update(_build_rate_limit_headers(
            "Session", SESSION_LIMIT, int(session_result.remaining * 10)
        ))

        if not session_result.allowed:
            return False, headers, session_result.retry_after_seconds, "session"

    return True, headers, None, None


async def check_agent_capability_quota(
    *,
    agent_id: str | None,
    capability_id: str | None,
    tenant_id: str | None = None,
) -> RateLimitResponse:
    """Prüft Agent- und Capability-Quoten mit zentralem System.

    Args:
        agent_id: Agent-ID für Agent-spezifische Limits
        capability_id: Capability-ID für Capability-spezifische Limits
        tenant_id: Tenant-ID für Multi-Tenancy-Support

    Returns:
        allowed: Ob Anfrage erlaubt ist
        headers: Rate-Limit-Header für HTTP-Response
        retry_after: Sekunden bis zum Reset (falls limitiert)
        limited_scope: "agent" | "capability" | None
    """
    headers: dict[str, str] = {}

    # Agent-Limit prüfen
    if agent_id:
        full_agent_id = f"agent:{tenant_id or 'global'}:{agent_id}"
        await _ensure_agent_limit_exists(
            full_agent_id, DEFAULT_AGENT_LIMIT, RateLimitWindow.PER_MINUTE
        )

        agent_result = await agent_rate_limiter.check_agent_rate_limit(
            agent_id=full_agent_id,
            window=RateLimitWindow.PER_MINUTE,
            requested_amount=1.0,
            context={
                "tenant_id": tenant_id or "global",
                "operation": "agent_execution",
                "scope": "agent"
            }
        )

        headers.update(_build_rate_limit_headers(
            "Agent", DEFAULT_AGENT_LIMIT, int(agent_result.remaining)
        ))

        if not agent_result.allowed:
            return False, headers, agent_result.retry_after_seconds, "agent"

    # Capability-Limit prüfen (über Agent-Rate-Limiter mit Capability-ID)
    if capability_id:
        capability_agent_id = f"capability:{tenant_id or 'global'}:{capability_id}"
        await _ensure_agent_limit_exists(
            capability_agent_id, DEFAULT_CAPABILITY_LIMIT, RateLimitWindow.PER_MINUTE
        )

        capability_result = await agent_rate_limiter.check_agent_rate_limit(
            agent_id=capability_agent_id,
            capability_id=capability_id,
            window=RateLimitWindow.PER_MINUTE,
            requested_amount=1.0,
            context={
                "tenant_id": tenant_id or "global",
                "operation": "capability_execution",
                "scope": "capability"
            }
        )

        headers.update(_build_rate_limit_headers(
            "Capability", DEFAULT_CAPABILITY_LIMIT, int(capability_result.remaining)
        ))

        if not capability_result.allowed:
            return False, headers, capability_result.retry_after_seconds, "capability"

    return True, headers, None, None


async def check_camera_limits_per_minute(
    user_id: str | None
) -> CameraLimitResponse:
    """Prüft Kamera-Capture-Limits mit zentralem Quota-System.

    Args:
        user_id: Benutzer-ID für User-spezifische Kamera-Limits

    Returns:
        allowed: Ob Anfrage erlaubt ist
        headers: Rate-Limit-Header für HTTP-Response
        retry_after: Sekunden bis zum Reset (falls limitiert)
    """
    if not user_id:
        # Ohne User-ID kein Limiting möglich - konservativ ablehnen
        return False, {}, None

    camera_agent_id = f"camera_user:{user_id}"
    await _ensure_agent_limit_exists(
        camera_agent_id, CAMERA_USER_PER_MIN_LIMIT, RateLimitWindow.PER_MINUTE
    )

    camera_result = await agent_rate_limiter.check_agent_rate_limit(
        agent_id=camera_agent_id,
        window=RateLimitWindow.PER_MINUTE,
        requested_amount=1.0,
        context={"operation": "camera_capture", "scope": "user"}
    )

    headers = _build_rate_limit_headers(
        "Camera-User-Min",
        CAMERA_USER_PER_MIN_LIMIT,
        int(camera_result.remaining)
    )

    if not camera_result.allowed:
        return False, headers, camera_result.retry_after_seconds

    return True, headers, None
