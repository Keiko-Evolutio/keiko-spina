"""API v1 Package - REST Endpunkte Version 1."""

from .configurations import router as configurations_router
from .events_api import events_router
from .management_api import management_router

__all__ = ["configurations_router", "events_router", "management_router"]
