"""KEI MCP Routes - Alias für MCP Routes.

Dieses Modul stellt eine Kompatibilitätsschicht für bestehende Imports bereit.
Die eigentliche Implementierung befindet sich in mcp_routes.py.
"""

# Import all exports from mcp_routes for compatibility
from .mcp_routes import *  # noqa: F403,F401

# Re-export commonly used functions for compatibility
# Ensure router is available
from .mcp_routes import (  # noqa: F401
    require_auth,
    require_rate_limit,
    router,
)

# Optional imports that might be expected
try:
    from .mcp_routes import require_domain_validation_for_registration
except ImportError:
    # Fallback if not available
    def require_domain_validation_for_registration(*_args, **_kwargs):
        """Fallback function for domain validation."""
        return True

try:
    from .mcp_routes import kei_mcp_auth
except ImportError:
    # Fallback if not available
    kei_mcp_auth = None

try:
    from .mcp_routes import mcp_registry
except ImportError:
    # Fallback if not available
    mcp_registry = None
