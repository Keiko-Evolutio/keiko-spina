"""Router Factory für die Keiko-Anwendung.

Diese Factory kapselt die Router-Registrierungs-Logik um die Komplexität
der KeikoApplication zu reduzieren.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.logger_utils import get_module_logger, safe_log_exception

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI

logger = get_module_logger(__name__)


class RouterGroup:
    """Repräsentiert eine Gruppe von verwandten Routern."""

    def __init__(
        self,
        name: str,
        prefix: str = "",
        tags: list[str] | None = None,
        dependencies: list[Any] | None = None
    ) -> None:
        self.name = name
        self.prefix = prefix
        self.tags = tags or []
        self.dependencies = dependencies or []
        self.routers: list[APIRouter] = []

    def add_router(self, router: APIRouter) -> None:
        """Fügt einen Router zur Gruppe hinzu."""
        self.routers.append(router)

    def register_to_app(self, app: FastAPI) -> None:
        """Registriert alle Router der Gruppe in der App."""
        for router in self.routers:
            app.include_router(
                router,
                prefix=self.prefix,
                tags=self.tags,
                dependencies=self.dependencies
            )


class RouterFactory:
    """Factory für Router-Registrierung."""

    def __init__(self) -> None:
        self.router_groups: dict[str, RouterGroup] = {}
        self._setup_router_groups()

    def _setup_router_groups(self) -> None:
        """Richtet die Router-Gruppen ein."""
        # Core API Routers
        self.router_groups["core"] = RouterGroup(
            name="core",
            prefix="/api/v1",
            tags=["core"]
        )

        # Agent System Routers
        self.router_groups["agents"] = RouterGroup(
            name="agents",
            prefix="/api/v1/agents",
            tags=["agents"]
        )

        # Voice System Routers
        self.router_groups["voice"] = RouterGroup(
            name="voice",
            prefix="/api/voice",
            tags=["voice"]
        )

        # WebSocket Routers
        self.router_groups["websocket"] = RouterGroup(
            name="websocket",
            prefix="/ws",
            tags=["websocket"]
        )

        # Admin/Management Routers
        self.router_groups["admin"] = RouterGroup(
            name="admin",
            prefix="/api/v1/admin",
            tags=["admin"]
        )

        # Camera Routers
        self.router_groups["camera"] = RouterGroup(
            name="camera",
            prefix="",  # Camera endpoints use their own prefix
            tags=["camera"]
        )

        # Health/Monitoring Routers
        self.router_groups["monitoring"] = RouterGroup(
            name="monitoring",
            prefix="",  # Health endpoints at root level
            tags=["monitoring"]
        )

    def register_all_routers(self, app: FastAPI) -> None:
        """Registriert alle Router-Gruppen in der App."""
        # Core Routers
        self._register_core_routers(app)

        # Agent Routers
        self._register_agent_routers(app)

        # Voice Routers
        self._register_voice_routers(app)

        # WebSocket Routers
        self._register_websocket_routers(app)

        # Camera Routers
        self._register_camera_routers(app)

        # Admin Routers
        self._register_admin_routers(app)

        # Health/Monitoring Routers
        self._register_monitoring_routers(app)

        logger.info("Alle Router registriert")

    def _register_core_routers(self, app: FastAPI) -> None:
        """Registriert Core API Router."""
        try:
            from api.routes.core_routes import router as core_router
            self.router_groups["core"].add_router(core_router)
            logger.debug("Core Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Core Router konnte nicht geladen werden")

        # Logs Router hinzufügen
        try:
            from api.routes.logs_routes import router as logs_router
            self.router_groups["core"].add_router(logs_router)
            logger.debug("Logs Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Logs Router konnte nicht geladen werden")

        # Alerts Router hinzufügen (Prometheus Alertmanager Integration)
        try:
            from api.routes.alerts_routes import router as alerts_router
            self.router_groups["core"].add_router(alerts_router)
            logger.debug("Alerts Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Alerts Router konnte nicht geladen werden")

        # System Status Router hinzufügen
        try:
            from api.routes.system_status import router as system_status_router
            self.router_groups["core"].add_router(system_status_router)
            logger.debug("System Status Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "System Status Router konnte nicht geladen werden")

        # System Heartbeat Fixed Router hinzufügen (Production-Ready)
        try:
            from api.routes.system_heartbeat_fixed import router as system_heartbeat_fixed_router
            self.router_groups["core"].add_router(system_heartbeat_fixed_router)
            logger.debug("System Heartbeat Fixed Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "System Heartbeat Fixed Router konnte nicht geladen werden")

        # Debug Router hinzufügen (ROOT CAUSE Analysis)
        try:
            from api.routes.debug_routes import router as debug_router
            self.router_groups["core"].add_router(debug_router)
            logger.debug("Debug Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Debug Router konnte nicht geladen werden")

        # System Heartbeat WebSocket Router hinzufügen
        try:
            from api.routes.system_heartbeat_ws import router as system_heartbeat_ws_router
            self.router_groups["core"].add_router(system_heartbeat_ws_router)
            logger.debug("System Heartbeat WebSocket Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "System Heartbeat WebSocket Router konnte nicht geladen werden")

        # Core Router-Gruppe zur App registrieren
        self.router_groups["core"].register_to_app(app)
        logger.debug("Core Router-Gruppe registriert")

    def _register_agent_routers(self, app: FastAPI) -> None:
        """Registriert Agent System Router."""
        try:
            from api.routes.agents_routes import router as agent_router
            self.router_groups["agents"].add_router(agent_router)
            logger.debug("Agent Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Agent Router konnte nicht geladen werden")

        try:
            from api.routes.kei_agents_routes import router as kei_agents_router
            self.router_groups["agents"].add_router(kei_agents_router)
            logger.debug("KEI Agents Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "KEI Agents Router konnte nicht geladen werden")

        # Agents Router-Gruppe zur App registrieren
        self.router_groups["agents"].register_to_app(app)
        logger.debug("Agents Router-Gruppe registriert")

    def _register_voice_routers(self, app: FastAPI) -> None:
        """Registriert Voice System Router."""
        try:
            from api.routes.voice_routes import router as voice_router
            self.router_groups["voice"].add_router(voice_router)
            self.router_groups["voice"].register_to_app(app)
            logger.debug("Voice Router registriert")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Voice Router konnte nicht geladen werden")

    def _register_websocket_routers(self, app: FastAPI) -> None:
        """Registriert WebSocket Router."""
        try:
            from api.routes.websocket_routes import router as websocket_router
            self.router_groups["websocket"].add_router(websocket_router)
            self.router_groups["websocket"].register_to_app(app)
            logger.debug("WebSocket Router registriert")
        except ImportError as exc:
            safe_log_exception(logger, exc, "WebSocket Router konnte nicht geladen werden")

        # Registriere WebSocket-Handler-Routen (Client/Agent WebSockets)
        try:
            from app.websocket_handlers import register_websocket_routes
            register_websocket_routes(app)
            logger.debug("WebSocket Handler Routen registriert (/ws/client, /ws/agent)")
        except ImportError as exc:
            safe_log_exception(logger, exc, "WebSocket Handler Routen konnten nicht geladen werden")

    def _register_admin_routers(self, app: FastAPI) -> None:
        """Registriert Admin/Management Router."""
        try:
            from api.routes.admin_routes import router as admin_router
            self.router_groups["admin"].add_router(admin_router)
            logger.debug("Admin Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Admin Router konnte nicht geladen werden")

        try:
            from api.routes.kei_mcp_routes import router as mcp_router
            self.router_groups["admin"].add_router(mcp_router)
            logger.debug("MCP Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "MCP Router konnte nicht geladen werden")

        try:
            from api.routes.voice_performance_routes import router as voice_performance_router
            self.router_groups["admin"].add_router(voice_performance_router)
            logger.debug("Voice Performance Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Voice Performance Router konnte nicht geladen werden")

        try:
            from api.routes.dlq_admin_routes import router as dlq_admin_router
            self.router_groups["admin"].add_router(dlq_admin_router)
            logger.debug("DLQ Admin Router hinzugefügt")
        except ImportError as exc:
            safe_log_exception(logger, exc, "DLQ Admin Router konnte nicht geladen werden")

        # Admin Router-Gruppe zur App registrieren
        self.router_groups["admin"].register_to_app(app)
        logger.debug("Admin Router-Gruppe registriert")

    def _register_camera_routers(self, app: FastAPI) -> None:
        """Registriert Camera System Router."""
        try:
            from api.routes.camera_routes import router as camera_router
            self.router_groups["camera"].add_router(camera_router)
            self.router_groups["camera"].register_to_app(app)
            logger.debug("Camera Router registriert")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Camera Router konnte nicht geladen werden")

    def _register_monitoring_routers(self, app: FastAPI) -> None:
        """Registriert Health/Monitoring Router."""
        try:
            # Health Endpoints werden direkt registriert (nicht über Router)
            from ..health_endpoints import register_health_and_metrics
            register_health_and_metrics(app)
            logger.debug("Health Endpoints registriert")
        except ImportError as exc:
            safe_log_exception(logger, exc, "Health Endpoints konnten nicht geladen werden")



    def get_router_group(self, name: str) -> RouterGroup | None:
        """Gibt eine Router-Gruppe zurück."""
        return self.router_groups.get(name)

    def add_custom_router(
        self,
        group_name: str,
        router: APIRouter,
        create_group_if_missing: bool = False
    ) -> bool:
        """Fügt einen benutzerdefinierten Router hinzu.

        Args:
            group_name: Name der Router-Gruppe
            router: Router-Instanz
            create_group_if_missing: Erstelle Gruppe falls sie nicht existiert

        Returns:
            True wenn erfolgreich hinzugefügt
        """
        if group_name not in self.router_groups:
            if create_group_if_missing:
                self.router_groups[group_name] = RouterGroup(name=group_name)
            else:
                logger.warning(f"Router-Gruppe '{group_name}' existiert nicht")
                return False

        self.router_groups[group_name].add_router(router)
        logger.debug(f"Custom Router zu Gruppe '{group_name}' hinzugefügt")
        return True


class RouterRegistrationValidator:
    """Validiert Router-Registrierungen."""

    @staticmethod
    def validate_router_paths(app: FastAPI) -> dict[str, list[str]]:
        """Validiert Router-Pfade auf Konflikte.

        Args:
            app: FastAPI-Anwendungsinstanz

        Returns:
            Dictionary mit gefundenen Konflikten
        """
        conflicts = {}
        seen_paths = {}

        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set())

            if path:
                for method in methods:
                    route_key = f"{method} {path}"
                    if route_key in seen_paths:
                        if path not in conflicts:
                            conflicts[path] = []
                        conflicts[path].append(f"Duplicate {method} route")
                    else:
                        seen_paths[route_key] = route

        return conflicts

    @staticmethod
    def get_router_statistics(app: FastAPI) -> dict[str, Any]:
        """Sammelt Statistiken über registrierte Router.

        Args:
            app: FastAPI-Anwendungsinstanz

        Returns:
            Dictionary mit Router-Statistiken
        """
        stats = {
            "total_routes": len(app.routes),
            "methods": {},
            "prefixes": {},
            "tags": {},
        }

        for route in app.routes:
            # Methoden zählen
            methods = getattr(route, "methods", set())
            for method in methods:
                stats["methods"][method] = stats["methods"].get(method, 0) + 1

            # Pfad-Präfixe analysieren
            path = getattr(route, "path", "")
            if path:
                prefix = path.split("/")[1] if len(path.split("/")) > 1 else "root"
                stats["prefixes"][prefix] = stats["prefixes"].get(prefix, 0) + 1

            # Tags sammeln
            tags = getattr(route, "tags", [])
            for tag in tags:
                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1

        return stats


__all__ = [
    "RouterFactory",
    "RouterGroup",
    "RouterRegistrationValidator",
]
