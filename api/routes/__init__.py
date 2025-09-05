"""API Routes Package für Azure AI Foundry Multi-Agent System."""

from .agents_metrics_routes import router as agents_metrics_router
from .agents_routes import router as agents_router
from .alertmanager_admin_routes import router as alertmanager_admin_router
from .alertmanager_routes import router as alertmanager_router
from .capabilities_routes import router as capabilities_router
from .chat_routes import router as chat_router
from .functions_routes import router as functions_router
from .health_routes import router as health_router
from .image_health_routes import router as image_health_router
from .logs_routes import router as logs_router
from .spec_routes import router as spec_router
from .voice_routes import router as voice_router
from .webhook_admin_routes import router as webhook_admin_router
from .webhook_deliveries_routes import router as webhook_deliveries_router
from .webhook_dlq_routes import router as webhook_dlq_router
from .webhook_replay_routes import router as webhook_replay_router
from .webhook_routes import router as webhook_router

# Erweiterte Agent-Management-Router (optional)
try:
    from .agent_policies_management import router as policies_router
    from .agent_quotas_management import router as quotas_router
    from .agent_scaling_health import router as scaling_health_router
    from .agent_statistics import router as statistics_router
    from .enhanced_agents_management import router as enhanced_agents_router
    ENHANCED_MANAGEMENT_AVAILABLE = True
except ImportError:
    ENHANCED_MANAGEMENT_AVAILABLE = False

# Enhanced Registry Router (optional)
try:
    from .enhanced_registry_routes import router as enhanced_registry_router
    ENHANCED_REGISTRY_AVAILABLE = True
except ImportError:
    ENHANCED_REGISTRY_AVAILABLE = False

__all__ = [
    "agents_metrics_router",
    "agents_router",
    "alertmanager_admin_router",
    "alertmanager_router",
    "capabilities_router",
    "chat_router",
    "functions_router",
    "health_router",
    "image_health_router",
    "logs_router",
    "spec_router",
    "voice_router",
    "webhook_admin_router",
    "webhook_deliveries_router",
    "webhook_dlq_router",
    "webhook_replay_router",
    "webhook_router",
]

# Erweiterte Management-Router zu __all__ hinzufügen (falls verfügbar)
if ENHANCED_MANAGEMENT_AVAILABLE:
    __all__.extend([
        "enhanced_agents_router",
        "policies_router",
        "quotas_router",
        "scaling_health_router",
        "statistics_router"
    ])

# Enhanced Registry Router zu __all__ hinzufügen (falls verfügbar)
if ENHANCED_REGISTRY_AVAILABLE:
    __all__.append("enhanced_registry_router")

# API Specifications Router (optional)
try:
    from .api_specs_routes import router as api_specs_router
    API_SPECS_AVAILABLE = True
except ImportError:
    API_SPECS_AVAILABLE = False

# API Specs Router zu __all__ hinzufügen (falls verfügbar)
if API_SPECS_AVAILABLE:
    __all__.append("api_specs_router")
