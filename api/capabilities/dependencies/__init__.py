"""Dependencies-Paket für Capabilities Dependency Injection."""

from .capability_deps import (
    get_capability_service,
    get_client_id,
    get_feature_context,
    get_feature_flag_service,
)

__all__ = [
    "get_capability_service",
    "get_client_id",
    "get_feature_context",
    "get_feature_flag_service"
]
