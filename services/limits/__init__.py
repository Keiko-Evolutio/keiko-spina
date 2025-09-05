# backend/services/limits/__init__.py
"""Rate Limiting Services für Backward-Compatibility.

Dieses Modul bietet Backward-Compatibility für bestehende Rate-Limiting-Funktionalitäten
und delegiert an das zentrale backend/quotas_limits System.

DEPRECATED: Verwende backend.quotas_limits für neue Implementierungen.
"""

import warnings

# Deprecation-Warnung beim Import
warnings.warn(
    "services.limits ist deprecated. Verwende backend.quotas_limits für neue Implementierungen.",
    DeprecationWarning,
    stacklevel=2
)

from .rate_limiter import (
    CAMERA_USER_PER_MIN_LIMIT,
    CAMERA_USER_PER_MIN_WINDOW_SECONDS,
    DEFAULT_AGENT_LIMIT,
    DEFAULT_AGENT_WINDOW_SECONDS,
    DEFAULT_CAPABILITY_LIMIT,
    DEFAULT_CAPABILITY_WINDOW_SECONDS,
    SESSION_LIMIT,
    SESSION_WINDOW_SECONDS,
    USER_LIMIT,
    USER_WINDOW_SECONDS,
    check_agent_capability_quota,
    check_camera_limits_per_minute,
    check_image_limits,
)

__all__ = [
    "CAMERA_USER_PER_MIN_LIMIT",
    "CAMERA_USER_PER_MIN_WINDOW_SECONDS",
    "DEFAULT_AGENT_LIMIT",
    "DEFAULT_AGENT_WINDOW_SECONDS",
    "DEFAULT_CAPABILITY_LIMIT",
    "DEFAULT_CAPABILITY_WINDOW_SECONDS",
    "SESSION_LIMIT",
    "SESSION_WINDOW_SECONDS",
    "USER_LIMIT",
    "USER_WINDOW_SECONDS",
    "check_agent_capability_quota",
    "check_camera_limits_per_minute",
    "check_image_limits",
]
