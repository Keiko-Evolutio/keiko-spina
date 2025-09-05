"""DEPRECATED: Legacy capabilities.py - Verwende api.capabilities Modul.

Dieses Modul ist deprecated und wird durch das neue modulare
api.capabilities Paket ersetzt. Für Backward-Compatibility werden
die wichtigsten Exports hier re-exportiert.
"""

import warnings

# Re-exports aus dem neuen modularen System
from api.capabilities import (
    APICapability,
    CapabilitiesResponse,
    CapabilityCategory,
    CapabilityResponse,
    CapabilityService,
    FeatureFlag,
    FeatureFlagResponse,
    FeatureFlagService,
    FeatureScope,
    FeatureStatus,
    get_capability_service,
    get_client_id,
    get_feature_context,
    get_feature_flag_service,
)
from api.capabilities import router as capabilities_router

# Deprecation Warning
warnings.warn(
    "api.capabilities module is deprecated. Use api.capabilities package instead.",
    DeprecationWarning,
    stacklevel=2
)

# ============================================================================
# LEGACY COMPATIBILITY - Globale Instanzen (DEPRECATED)
# ============================================================================

# Legacy globale Instanzen für Backward-Compatibility
_feature_flag_service = None
_capability_service = None


def get_legacy_feature_flag_manager():
    """Legacy-Funktion für Backward-Compatibility."""
    global _feature_flag_service
    if _feature_flag_service is None:
        _feature_flag_service = get_feature_flag_service()
    return _feature_flag_service


def get_legacy_capability_manager():
    """Legacy-Funktion für Backward-Compatibility."""
    global _capability_service
    if _capability_service is None:
        _capability_service = get_capability_service()
    return _capability_service


# Legacy aliases
feature_flag_manager = get_legacy_feature_flag_manager()
capability_manager = get_legacy_capability_manager()

# Legacy compatibility - Router
router = capabilities_router

# ============================================================================
# LEGACY EXPORTS (DEPRECATED)
# ============================================================================

# Alle wichtigen Klassen und Funktionen werden aus dem neuen Modul re-exportiert
# für Backward-Compatibility

__all__ = [
    "APICapability",
    "CapabilitiesResponse",
    "CapabilityCategory",
    "CapabilityResponse",
    "CapabilityService",
    # Models
    "FeatureFlag",
    "FeatureFlagResponse",
    # Services
    "FeatureFlagService",
    "FeatureScope",
    # Enums
    "FeatureStatus",
    "capability_manager",
    # Legacy compatibility
    "feature_flag_manager",
    "get_capability_service",
    "get_client_id",
    "get_feature_context",
    # Dependencies
    "get_feature_flag_service",
    # Router
    "router"
]
