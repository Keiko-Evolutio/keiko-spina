"""Capabilities-Modul für Feature-Flag-System und API-Capabilities-Management.

Dieses refactored Modul ersetzt die monolithische capabilities.py und
stellt eine modulare, testbare Architektur bereit.
"""

from .dependencies.capability_deps import (
    get_capability_service,
    get_client_id,
    get_feature_context,
    get_feature_flag_service,
)
from .models.capability_models import (
    APICapability,
    CapabilitiesResponse,
    CapabilityCategory,
    CapabilityResponse,
)
from .models.feature_flag_models import (
    FeatureFlag,
    FeatureFlagResponse,
    FeatureScope,
    FeatureStatus,
)
from .routes.capabilities_routes import router
from .services.capability_service import CapabilityService
from .services.feature_flag_service import FeatureFlagService

__all__ = [
    "APICapability",
    "CapabilitiesResponse",
    "CapabilityCategory",
    "CapabilityResponse",
    "CapabilityService",
    "FeatureFlag",
    "FeatureFlagResponse",
    # Services
    "FeatureFlagService",
    "FeatureScope",
    # Models
    "FeatureStatus",
    "get_capability_service",
    "get_client_id",
    "get_feature_context",
    # Dependencies
    "get_feature_flag_service",
    # Router
    "router"
]
