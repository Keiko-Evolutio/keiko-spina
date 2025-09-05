"""API Versionierungspaket.

Stellt Router und Middleware für Versionserkennung, Deprecation‑Handling und
Content‑Negotiation bereit. Konsolidierte Implementierung mit gemeinsamen
Utilities und Modellen zur Reduzierung von Code-Duplikation.
"""

# Exportiere auch die neuen Module für erweiterte Nutzung
from . import constants, models, utils
from .middleware import VersioningMiddleware
from .v1_router import v1_router
from .v2_router import v2_router

__all__ = [
    "VersioningMiddleware",
    "constants",
    "models",
    "utils",
    "v1_router",
    "v2_router",
]
