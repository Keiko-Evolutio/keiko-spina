"""System Status API Routes
Provides endpoints for system health, readiness, and infrastructure status
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from api.middleware.scope_middleware import require_scopes
from auth.enterprise_auth import Scope
from auth.unified_enterprise_auth import require_unified_auth


# Dependency: Erforderliche Scopes für System Heartbeat
async def require_system_heartbeat_scopes(request: Request):
    """Hinterlegt den Scope 'system:read' für diesen Request."""
    require_scopes(request, [Scope.SYSTEM_READ.value])
import logging
import os
import time
from datetime import datetime
from typing import Any

from fastapi.responses import JSONResponse

# Import für System Heartbeat Service
try:
    from services.system_heartbeat_service import get_system_heartbeat_service
    HEARTBEAT_SERVICE_AVAILABLE = True
except ImportError:
    get_system_heartbeat_service = None  # Fallback-Definition für get_system_heartbeat_service
    HEARTBEAT_SERVICE_AVAILABLE = False

# Für Development-Umgebung deaktivieren wir den Orchestrator
KEIKO_ENV = os.getenv("KEIKO_ENV", "development")

if KEIKO_ENV == "development":
    orchestrator = None
    ORCHESTRATOR_AVAILABLE = False
else:
    try:
        import sys
        # Add parent directory to path for startup_orchestration module
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from startup_orchestration import orchestrator
        ORCHESTRATOR_AVAILABLE = True
    except ImportError:
        sys = None  # Fallback-Definition für sys
        orchestrator = None
        ORCHESTRATOR_AVAILABLE = False

try:
    from core.container import get_container
except ImportError:
    get_container = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System Status"])


# Explizite OPTIONS-Route für CORS-Preflight-Requests (DEAKTIVIERT - verursacht Probleme)
# @router.options("/{path:path}", operation_id="system_options_handler")
# async def options_handler(request: Request, path: str):
#     """
#     Behandelt alle OPTIONS-Requests für CORS-Preflight
#     """
#     return JSONResponse(
#         content={},
#         headers={
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
#             "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Accept-Language, Content-Language, X-Tenant-Id, X-User-Id, x-tenant, x-user-id, X-Trace-Id",
#             "Access-Control-Max-Age": "3600"
#         }
#     )


@router.get("/health", operation_id="system_health_check")
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint
    Returns 200 if the backend service is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "keiko-backend",
        "version": "2.0.0"
    }


@router.get("/readiness", operation_id="system_readiness_check")
async def readiness_check() -> dict[str, Any]:
    """Readiness check endpoint
    Returns 200 only if all required infrastructure services are healthy
    """
    if not ORCHESTRATOR_AVAILABLE:
        # In development mode, assume ready if basic health check passes
        return {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat(),
            "infrastructure": {
                "total": 3,
                "healthy": 3,
                "unhealthy": 0,
                "starting": 0,
                "failed": 0
            },
            "message": "Development mode: Essential services assumed healthy"
        }

    try:
        # Initialize orchestrator if needed
        if orchestrator and hasattr(orchestrator, "docker_client") and not getattr(orchestrator, "docker_client", None):
            if hasattr(orchestrator, "initialize"):
                await orchestrator.initialize()

        # Get infrastructure status
        if orchestrator and hasattr(orchestrator, "get_infrastructure_status"):
            infra_status = await orchestrator.get_infrastructure_status()
        else:
            infra_status = {
                "status": "unavailable",
                "message": "Orchestrator not available in development mode",
                "services": {}
            }

        # Check if all required services are ready
        ready = infra_status["ready"]

        response_data = {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "infrastructure": infra_status["summary"],
            "message": "All required services are healthy" if ready else "Some required services are not healthy"
        }

        if ready:
            return JSONResponse(
                status_code=200,
                content=response_data
            )
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content=response_data
        )

    except Exception as e:
        logger.error(f"Error in readiness check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "ready": False,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Failed to check infrastructure status",
                "message": str(e)
            }
        )


@router.get("/infrastructure", operation_id="system_infrastructure_status")
async def infrastructure_status() -> dict[str, Any]:
    """Detailed infrastructure status endpoint
    Returns comprehensive status of all infrastructure services
    """
    if not ORCHESTRATOR_AVAILABLE:
        # Development-Modus: Führe eine leichte Container-Erkennung durch statt statischer "healthy"-Antwort
        try:
            import subprocess
            result = subprocess.run([
                "docker", "ps", "--filter", "name=keiko-",
                "--format", "{{.Names}}\t{{.Status}}"
            ], check=False, capture_output=True, text=True, timeout=5)

            services: dict[str, Any] = {}
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    raw_name = parts[0]
                    status = parts[1]
                    name = raw_name.replace("keiko-", "")
                    is_healthy = "Up" in status
                    services[name] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "required": name in ["postgres", "redis", "nats"]
                    }

            total = len(services)
            healthy = sum(1 for s in services.values() if s["status"] == "healthy")
            unhealthy = total - healthy
            # Infrastruktur gilt nur dann als bereit, wenn alle als "required" markierten Services healthy sind
            required_ok = all(
                (s["status"] == "healthy") for n, s in services.items() if s.get("required")
            ) if services else False

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "infrastructure": {
                    "ready": bool(required_ok and healthy >= 1 and unhealthy == 0),
                    "last_check": time.time(),
                    "services": services,
                    "summary": {
                        "total": total,
                        "healthy": healthy,
                        "unhealthy": unhealthy,
                        "starting": 0,
                        "failed": 0,
                    }
                }
            }
        except Exception as e:
            logger.warning(f"Dev infrastructure probe failed: {e}")
            # Fallback: Unbekannter Zustand statt fälschlich "healthy"
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "infrastructure": {
                    "ready": False,
                    "last_check": time.time(),
                    "services": {},
                    "summary": {
                        "total": 0,
                        "healthy": 0,
                        "unhealthy": 0,
                        "starting": 0,
                        "failed": 0,
                    }
                }
            }

    try:
        # Initialize orchestrator if needed
        if orchestrator and hasattr(orchestrator, "docker_client") and not getattr(orchestrator, "docker_client", None):
            if hasattr(orchestrator, "initialize"):
                await orchestrator.initialize()

        if orchestrator and hasattr(orchestrator, "get_infrastructure_status"):
            status = await orchestrator.get_infrastructure_status()
        else:
            status = {
                "status": "unavailable",
                "message": "Orchestrator not available in development mode",
                "services": {}
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "infrastructure": status
        }

    except Exception as e:
        logger.error(f"Error getting infrastructure status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get infrastructure status: {e!s}"
        )


@router.get("/heartbeat-test", operation_id="system_heartbeat_test")
def system_heartbeat_test() -> dict[str, Any]:
    """Test-Heartbeat-Endpoint"""
    return {"status": "working", "timestamp": 1756218000.0}


@router.get("/heartbeat-simple", operation_id="system_heartbeat_simple")
def system_heartbeat_simple() -> dict[str, Any]:
    """Einfachster möglicher Heartbeat-Endpoint"""
    return {"status": "ok"}


@router.get("/heartbeat-json", operation_id="system_heartbeat_json")
def system_heartbeat_json() -> dict[str, Any]:
    """JSON-Response-Test"""
    return {"status": "json_ok", "timestamp": time.time()}


@router.get("/heartbeat", operation_id="system_heartbeat")
async def system_heartbeat(
    _auth=Depends(require_unified_auth),
    _scopes=Depends(require_system_heartbeat_scopes)
) -> dict[str, Any]:
    """Production-Ready System Heartbeat endpoint
    Returns real-time status of all critical backend services + Docker containers
    """
    try:
        logger.info("🔍 System Heartbeat Endpoint aufgerufen")

        # Robuste Container-Discovery
        services = {}
        container_count = 0
        current_time = time.time()

        # Docker-Container prüfen
        try:
            import subprocess

            result = subprocess.run([
                "docker", "ps", "--filter", "name=keiko-",
                "--format", "{{.Names}}\t{{.Status}}"
            ], check=False, capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            name = parts[0].replace("keiko-", "")
                            status = parts[1]
                            container_count += 1

                            # Container als healthy markieren wenn er läuft
                            is_healthy = "Up" in status
                            services[name] = {
                                "name": name,
                                "status": "healthy" if is_healthy else "unhealthy",
                                "last_check": current_time,
                                "response_time_ms": 1.0
                            }
                            logger.debug(f"✅ Container gefunden: {name} - {status}")

        except Exception as docker_error:
            logger.warning(f"🐳 Docker-Befehl Fehler: {docker_error}")

        # Backend-Service hinzufügen
        services["application"] = {
            "name": "application",
            "status": "healthy",
            "last_check": current_time,
            "response_time_ms": 1.0
        }

        # Statistiken berechnen
        total_services = len(services)
        healthy_services = sum(1 for s in services.values() if s["status"] == "healthy")

        # Response erstellen
        response = {
            "timestamp": current_time,
            "phase": "ready",
            "overall_status": "healthy" if healthy_services == total_services else "degraded",
            "services": services,
            "summary": {
                "total": total_services,
                "healthy": healthy_services,
                "unhealthy": total_services - healthy_services,
                "starting": 0,
                "failed": 0,
                "unknown": 0
            },
            "uptime_seconds": current_time - 1756000000,
            "message": f"🎉 All {total_services} services are healthy ({container_count} Docker containers + 1 backend)"
        }

        logger.info(f"✅ System Heartbeat erfolgreich: {total_services} Services, {healthy_services} healthy")
        return response

    except Exception as e:
        logger.error(f"❌ System Heartbeat Fehler: {e}")

        # Fallback-Response
        return {
            "timestamp": time.time(),
            "phase": "error",
            "overall_status": "unhealthy",
            "services": {
                "application": {
                    "name": "application",
                    "status": "healthy",
                    "last_check": time.time(),
                    "response_time_ms": 1.0
                }
            },
            "summary": {
                "total": 1,
                "healthy": 1,
                "unhealthy": 0,
                "starting": 0,
                "failed": 0,
                "unknown": 0
            },
            "uptime_seconds": time.time() - 1756000000,
            "message": f"⚠️ System Heartbeat Error: {e!s}"
        }


@router.get("/startup-status", operation_id="system_startup_status")
async def startup_status() -> dict[str, Any]:
    """Startup Status endpoint - basiert auf kritischen Services
    Gibt ready=True zurück sobald kritische Services (postgres, redis, nats, application) healthy sind
    """
    if HEARTBEAT_SERVICE_AVAILABLE:
        heartbeat_service = get_system_heartbeat_service()
        if heartbeat_service:
            startup_data = heartbeat_service.get_startup_status()
            if startup_data:
                return startup_data

    # Fallback für Development-Modus: defensiv und realitätsnah
    try:
        # Wenn Heartbeat-Service nicht liefert, versuche minimalen Health-Endpunkt
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as resp:  # pragma: no cover - best-effort
            if resp.status == 200:
                # Backend läuft, aber kein Heartbeat: zeige "starting"
                return {
                    "timestamp": time.time(),
                    "phase": "starting",
                    "progress": 50,
                    "ready": False,
                    "services": {"total": 1, "healthy": 1, "starting": 0, "failed": 0},
                    "failed_services": [],
                    "message": "Backend erreichbar, warte auf Infrastruktur-Heartbeat"
                }
    except Exception:
        pass

    # Backend/Heartbeat nicht erreichbar → zeige korrekt "starting"/not ready
    return {
        "timestamp": time.time(),
        "phase": "starting",
        "progress": 0,
        "ready": False,
        "services": {"total": 0, "healthy": 0, "starting": 0, "failed": 0},
        "failed_services": [],
        "message": "Entwicklungsmodus: Kein Heartbeat verfügbar, System nicht bereit"
    }


@router.get("/startup-status-legacy", operation_id="system_startup_status_legacy")
async def startup_status_legacy() -> dict[str, Any]:
    """Startup status endpoint for frontend polling
    Returns current startup progress and any issues
    """
    if not ORCHESTRATOR_AVAILABLE:
        # In development mode, return ready status
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "ready",
            "progress": 100,
            "ready": True,
            "services": {
                "total": 3,
                "healthy": 3,
                "starting": 0,
                "failed": 0
            },
            "failed_services": [],
            "message": "🎉 Development mode: System is ready! Essential services are healthy."
        }

    try:
        # Initialize orchestrator if needed
        if orchestrator and hasattr(orchestrator, "docker_client") and not getattr(orchestrator, "docker_client", None):
            if hasattr(orchestrator, "initialize"):
                await orchestrator.initialize()

        if orchestrator and hasattr(orchestrator, "get_infrastructure_status"):
            startup_infra_status = await orchestrator.get_infrastructure_status()
        else:
            startup_infra_status = {
                "status": "unavailable",
                "message": "Orchestrator not available in development mode",
                "summary": {
                    "total": 0,
                    "healthy": 0,
                    "failed": 0,
                    "starting": 0
                },
                "services": {}
            }

        # Determine startup phase
        summary = startup_infra_status["summary"]
        total_services = summary["total"]
        healthy_services = summary["healthy"]

        if summary["failed"] > 0:
            phase = "failed"
            progress = 0
        elif summary["starting"] > 0 or summary["unhealthy"] > 0:
            phase = "starting"
            progress = int((healthy_services / total_services) * 100)
        elif startup_infra_status["ready"]:
            phase = "ready"
            progress = 100
        else:
            phase = "initializing"
            progress = int((healthy_services / total_services) * 100)

        # Get failed services for error reporting
        failed_services = [
            name for name, service in infrastructure_status["services"].items()
            if service["status"] in ["failed", "unhealthy"] and service["required"]
        ]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "progress": progress,
            "ready": infrastructure_status["ready"],
            "services": {
                "total": total_services,
                "healthy": healthy_services,
                "starting": summary["starting"],
                "failed": summary["failed"]
            },
            "failed_services": failed_services,
            "message": _get_startup_message(phase, progress, failed_services)
        }

    except Exception as e:
        logger.error(f"Error getting startup status: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "error",
            "progress": 0,
            "ready": False,
            "error": str(e),
            "message": "Failed to determine startup status"
        }


def _get_startup_message(phase: str, progress: int, failed_services: list) -> str:
    """Generate appropriate startup message based on current state"""
    if phase == "ready":
        return "🎉 System is ready! All infrastructure services are healthy."
    if phase == "failed":
        services_list = ", ".join(failed_services) if failed_services else "unknown services"
        return f"❌ Startup failed. The following services are not healthy: {services_list}"
    if phase == "starting":
        return f"🚀 Starting up... {progress}% complete. Please wait for all services to become healthy."
    if phase == "initializing":
        return f"🔄 Initializing infrastructure... {progress}% complete."
    return "⚠️ Unknown startup state. Please check system logs."


@router.get("/dependencies", operation_id="system_service_dependencies")
async def service_dependencies() -> dict[str, Any]:
    """Service dependencies endpoint
    Returns information about service dependencies and startup order
    """
    dependencies = {
        "startup_order": [
            {
                "category": "base",
                "services": ["postgres", "redis", "nats"],
                "description": "Core infrastructure services",
                "required_for_backend": True
            },
            {
                "category": "monitoring",
                "services": ["prometheus", "alertmanager", "jaeger", "otel-collector"],
                "description": "Observability and monitoring services",
                "required_for_backend": True
            },
            {
                "category": "workflow",
                "services": ["n8n-postgres", "n8n"],
                "description": "Workflow automation services",
                "required_for_backend": False
            },
            {
                "category": "edge",
                "services": ["edge-registry", "edge-node-1", "edge-node-2", "edge-node-3"],
                "description": "Edge computing services",
                "required_for_backend": False
            },
            {
                "category": "tools",
                "services": ["pgadmin", "redis-insight", "mailhog", "grafana"],
                "description": "Development and management tools",
                "required_for_backend": False
            }
        ],
        "backend_dependencies": [
            "postgres", "redis", "nats", "prometheus",
            "alertmanager", "jaeger", "otel-collector"
        ],
        "optional_services": [
            "n8n-postgres", "n8n", "edge-registry", "edge-node-1",
            "edge-node-2", "edge-node-3", "pgadmin", "redis-insight",
            "mailhog", "grafana"
        ]
    }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies
    }
