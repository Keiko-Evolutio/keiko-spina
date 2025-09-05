"""Scope/Authorization Middleware für Webhook‑RBAC.

Validiert JWT‑ableiteten Principal und prüft Scopes pro Route.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from kei_logging import get_logger
from security.authorization_service import Principal, WebhookAuthorizationService
from security.rbac_config_loader import load_rbac_config
from security.rbac_models import RBACConfig, RoleDefinition

logger = get_logger(__name__)


def _extract_principal(request: Request, settings_service=None) -> Principal:
    """Erzeugt Principal aus Request‑Kontext (JWT Payload + Header).

    Falls keine Scopes im `request.state.user` vorhanden sind, wird als
    Fallback der Header `X-Scopes` ausgewertet (whitespace‑separiert). Dies
    erleichtert Tests und einfache Integrationen ohne vollständiges JWT.
    """
    # Fallback für Backward Compatibility
    if settings_service is None:
        from config.settings import get_settings
        settings = get_settings()
    else:
        settings = settings_service

    payload = getattr(request.state, "user", {}) or {}
    subject = payload.get("sub") or payload.get("client_id") or settings.default_user_id
    roles = payload.get("roles") or []
    scopes = payload.get("scopes") or payload.get("scope") or []
    if isinstance(scopes, str):
        scopes = [s.strip() for s in scopes.split()] if scopes else []
    if not scopes:
        hdr = request.headers.get("X-Scopes") or request.headers.get("x-scopes")
        if hdr:
            scopes = [s.strip() for s in hdr.split() if s.strip()]

    # Fallback für Default-Benutzer: Standard-Scopes zuweisen
    if not scopes and settings.allow_anonymous_access:
        default_scopes = settings.default_user_scopes
        if default_scopes:
            scopes = [s.strip() for s in default_scopes.split(",") if s.strip()]

    # Tenant-ID mit Fallback auf Default-Tenant
    forced_tenant = getattr(request.state, "forced_tenant", None)
    tenant_id = (
        forced_tenant or
        request.headers.get("X-Tenant-Id") or
        request.headers.get("x-tenant") or
        settings.default_tenant_id
    )

    return Principal(subject=subject, roles=list(roles or []), scopes=list(scopes or []), tenant_id=tenant_id)


class ScopeRequirement:
    """Definiert die Scope‑Anforderung für eine Route."""

    def __init__(self, scopes: list[str]):
        self.scopes = scopes

    def expand_with_tenant(self, tenant_id: str | None) -> list[str]:
        """Erweitert Scopes um tenant‑spezifische Varianten.

        Beispiel: webhook:targets:manage -> tenant:{id}:webhook:targets:manage
        """
        if not tenant_id:
            return self.scopes
        expanded = list(self.scopes)
        for s in self.scopes:
            expanded.append(f"tenant:{tenant_id}:{s}")
        return expanded


class ScopeMiddleware(BaseHTTPMiddleware):
    """Middleware, die pro Route Scopes prüft.

    Erwartet, dass Vor‑Middleware JWT validiert und `request.state.user` setzt.
    """

    def __init__(self, app, rbac_config: RBACConfig | None = None, config_service=None):
        super().__init__(app)
        self._config_service = config_service

        # Default‑Rollen falls keine Konfiguration gesetzt wurde
        if rbac_config is None:
            try:
                rbac_config = load_rbac_config()
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"RBAC-Konfiguration konnte nicht geladen werden: {e}. Verwende Default-Konfiguration.")
                rbac_config = RBACConfig(roles=[RoleDefinition(name="webhook_admin", scopes=["webhook:admin:*"])])
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Laden der RBAC-Konfiguration: {e}. Verwende Default-Konfiguration.")
                rbac_config = RBACConfig(roles=[RoleDefinition(name="webhook_admin", scopes=["webhook:admin:*"])])
        self.authz = WebhookAuthorizationService(rbac_config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # WebSocket‑Handshakes sowie WebSocket‑Pfade grundsätzlich nicht durch Scope‑RBAC prüfen
        # (WS‑Endpunkte besitzen eigene Auth‑Logik vor dem Accept; Handshake sonst 403)
        try:
            if request.headers.get("upgrade", "").lower() == "websocket":
                return await call_next(request)
        except AttributeError:
            # Request hat keine headers - das sollte nicht passieren, aber sicherheitshalber
            pass
        except Exception as e:
            logger.debug(f"Fehler beim Prüfen des WebSocket-Upgrade-Headers: {e}")
        if request.url.path.startswith("/ws/") or request.url.path.startswith("/websocket/"):
            return await call_next(request)

        # Bestimmte Pfade ohne Scope‑Check erlauben (z. B. Health, Inbound, Camera)
        if any(seg in request.url.path for seg in ["/health", "/specs", "/openapi.json", "/inbound/", "/metrics", "/api/camera"]):
            return await call_next(request)

        # Tenant Pflicht für alle geschützten Webhook Pfade mit BC‑Fallback → "public"
        bc_forced_tenant = False
        if request.url.path.startswith("/api/v1/webhooks"):
            tenant_header = request.headers.get("X-Tenant-Id") or request.headers.get("x-tenant")
            if not tenant_header:
                request.state.forced_tenant = "public"
                bc_forced_tenant = True

        # Erforderliche Scopes ermitteln: explizit gesetzt oder heuristisch aus Pfad
        requirement: ScopeRequirement | None = getattr(request.state, "required_scopes", None)
        if requirement is None:
            required_scopes: list[str] = []
            path = request.url.path
            method = request.method.upper()
            try:
                if path.startswith("/api/v1/webhooks/admin"):
                    required_scopes = ["webhook:admin:*"]
                elif path.startswith("/api/v1/webhooks/targets"):
                    required_scopes = ["webhook:targets:manage"]
                elif path.startswith("/api/v1/webhooks/dlq"):
                    required_scopes = ["webhook:dlq:manage"]
                elif path.startswith("/api/v1/webhooks/deliveries"):
                    required_scopes = ["webhook:admin:*"]
                elif path == "/api/v1/webhooks/outbound/enqueue" and method == "POST":
                    # Body lesen, um event_type zu extrahieren
                    try:
                        body = await request.body()
                        import json as _json
                        data = _json.loads(body.decode("utf-8")) if body else {}
                        event_type = data.get("event_type", "*")
                    except (ValueError, KeyError, UnicodeDecodeError) as e:
                        logger.debug(f"Fehler beim Parsen des Request-Body für Event-Type: {e}")
                        event_type = "*"
                    except Exception as e:
                        logger.warning(f"Unerwarteter Fehler beim Extrahieren des Event-Types: {e}")
                        event_type = "*"
                    required_scopes = [f"webhook:outbound:send:{event_type}"]
            except (AttributeError, TypeError) as e:
                logger.debug(f"Fehler beim Bestimmen der Required-Scopes: {e}")
                required_scopes = []
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler bei Scope-Bestimmung: {e}")
                required_scopes = []
            requirement = ScopeRequirement(required_scopes)
            request.state.required_scopes = requirement

        # Wenn keine expliziten Scopes erforderlich sind, keinen RBAC-Check erzwingen
        if not requirement.scopes:
            return await call_next(request)

        # Autorisierung durchführen
        # Verwende injizierte Config oder Fallback
        if self._config_service:
            settings = self._config_service
        else:
            # Fallback für Backward Compatibility
            from config.settings import get_settings
            settings = get_settings()

        principal = _extract_principal(request, settings)
        tenant_id = principal.tenant_id

        # Prüfe ob Tenant-Header erforderlich ist

        if settings.tenant_header_required and not request.headers.get("X-Tenant-Id"):
            # Nur in Production oder wenn explizit aktiviert
            if settings.tenant_isolation_enabled:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="X-Tenant-Id header required"
                )

        # Autorisierung mit Tenant-Kontext
        try:
            decision = self.authz.authorize(principal, requirement.expand_with_tenant(tenant_id), tenant_id)
            if not decision.allowed:
                # Fallback: Versuche ohne Tenant-spezifische Scopes
                if tenant_id != settings.default_tenant_id:
                    decision = self.authz.authorize(principal, requirement.scopes, settings.default_tenant_id)

                if not decision.allowed:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason or "forbidden")
        except HTTPException:
            # HTTP-Exceptions sollen durchgereicht werden
            raise
        except (AttributeError, ValueError) as e:
            # Fallback für Development: Erlaube Zugriff mit Default-Tenant
            if not settings.tenant_isolation_enabled:
                logger.warning(f"Authorization fallback for {principal.subject}: {e}")
            else:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="authorization_failed")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Authorization: {e}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="authorization_failed")

        response = await call_next(request)
        if bc_forced_tenant:
            try:
                response.headers["Warning"] = "299 - Tenant header missing; defaulted to 'public' (deprecated)."
                response.headers["Sunset"] = "true"
            except (AttributeError, TypeError) as e:
                logger.debug(f"Fehler beim Setzen der Deprecation-Header: {e}")
            except Exception as e:
                logger.warning(f"Unerwarteter Fehler beim Setzen der Response-Header: {e}")
        return response


# Factory-Funktion für Dependency Injection
def create_scope_middleware_with_di(rbac_config: RBACConfig | None = None):
    """Factory-Funktion für ScopeMiddleware mit Dependency Injection.

    Args:
        rbac_config: RBAC-Konfiguration

    Returns:
        Middleware-Factory-Funktion
    """
    def middleware_factory(app):
        # Enhanced DI Container nicht mehr verfügbar - verwende Standard-Konfiguration
        config_service = None

        return ScopeMiddleware(app, rbac_config=rbac_config, config_service=config_service)

    return middleware_factory


def require_scopes(request: Request, scopes: list[str]) -> None:
    """Hinterlegt die Scope‑Anforderung für den aktuellen Request in `request.state`."""
    request.state.required_scopes = ScopeRequirement(scopes)


__all__ = ["ScopeMiddleware", "ScopeRequirement", "require_scopes"]
