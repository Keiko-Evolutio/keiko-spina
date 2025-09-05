# backend/agents/enhanced_security/security_manager.py
"""Enterprise Security Manager

Implementiert umfassende Sicherheitsfunktionen für Multi-Agent-Systeme:
- Authentifizierung und Autorisierung
- Policy-basierte Zugriffskontrolle
- Security Context Management
- Threat Detection Integration
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Sicherheitsstufen für Agent-Operationen."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityViolationType(Enum):
    """Typen von Sicherheitsverletzungen."""

    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_DENIED = "authorization_denied"
    POLICY_VIOLATION = "policy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH = "data_breach"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class SecurityConfig:
    """Konfiguration für Security Manager."""

    # Authentifizierung
    enable_authentication: bool = True
    auth_token_ttl_seconds: int = 3600
    max_failed_attempts: int = 3
    lockout_duration_seconds: int = 300

    # Autorisierung
    enable_authorization: bool = True
    default_security_level: SecurityLevel = SecurityLevel.MEDIUM
    require_explicit_permissions: bool = True

    # Verschlüsselung
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval_hours: int = 24

    # Audit & Monitoring
    enable_audit_logging: bool = True
    audit_retention_days: int = 90
    enable_threat_detection: bool = True

    # Rate Limiting
    enable_rate_limiting: bool = True
    default_rate_limit: int = 100
    rate_limit_window_seconds: int = 60

    # Compliance
    compliance_mode: str = "standard"  # standard, strict, custom
    data_classification_required: bool = False


@dataclass
class SecurityPolicy:
    """Sicherheitsrichtlinie für Agent-Operationen."""

    policy_id: str
    name: str
    description: str
    security_level: SecurityLevel
    allowed_operations: set[str] = field(default_factory=set)
    denied_operations: set[str] = field(default_factory=set)
    required_permissions: set[str] = field(default_factory=set)
    rate_limits: dict[str, int] = field(default_factory=dict)
    data_access_rules: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    is_active: bool = True


@dataclass
class SecurityContext:
    """Sicherheitskontext für Agent-Ausführung."""

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    user_id: str | None = None
    session_id: str | None = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    permissions: set[str] = field(default_factory=set)
    policies: list[SecurityPolicy] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    is_authenticated: bool = False
    is_authorized: bool = False


@dataclass
class AuthenticationResult:
    """Ergebnis einer Authentifizierung."""

    success: bool
    user_id: str | None = None
    session_id: str | None = None
    token: str | None = None
    expires_at: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorizationResult:
    """Ergebnis einer Autorisierung."""

    success: bool
    allowed_operations: set[str] = field(default_factory=set)
    denied_operations: set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityViolation:
    """Sicherheitsverletzung."""

    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    violation_type: SecurityViolationType = SecurityViolationType.POLICY_VIOLATION
    agent_id: str = ""
    user_id: str | None = None
    operation: str = ""
    description: str = ""
    severity: SecurityLevel = SecurityLevel.MEDIUM
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class SecurityManager:
    """Enterprise Security Manager"""

    def __init__(self, config: SecurityConfig):
        """Initialisiert Security Manager.

        Args:
            config: Security-Konfiguration
        """
        self.config = config
        self._active_sessions: dict[str, SecurityContext] = {}
        self._security_policies: dict[str, SecurityPolicy] = {}
        self._violations: list[SecurityViolation] = []
        self._rate_limits: dict[str, dict[str, Any]] = {}
        self._failed_attempts: dict[str, int] = {}
        self._lockouts: dict[str, float] = {}

        # Komponenten (werden bei Bedarf initialisiert)
        self._encryption_manager = None
        self._audit_logger = None
        self._threat_detector = None

        logger.info("Security Manager initialisiert")

    @trace_function("security.authenticate")
    async def authenticate(
        self,
        credentials: dict[str, Any],
        agent_id: str
    ) -> AuthenticationResult:
        """Authentifiziert einen Agent oder Benutzer.

        Args:
            credentials: Authentifizierungsdaten
            agent_id: Agent-ID

        Returns:
            Authentifizierungsergebnis
        """
        if not self.config.enable_authentication:
            return AuthenticationResult(success=True, metadata={"bypass": True})

        try:
            # Lockout-Prüfung
            if await self._is_locked_out(agent_id):
                return AuthenticationResult(
                    success=False,
                    error_message="Account temporär gesperrt"
                )

            # Authentifizierung durchführen
            auth_result = await self._perform_authentication(credentials, agent_id)

            if auth_result.success:
                # Erfolgreiche Authentifizierung
                await self._reset_failed_attempts(agent_id)
                await self._create_session(auth_result, agent_id)
            else:
                # Fehlgeschlagene Authentifizierung
                await self._record_failed_attempt(agent_id)

            return auth_result

        except Exception as e:
            logger.error(f"Authentifizierung fehlgeschlagen: {e}")
            return AuthenticationResult(
                success=False,
                error_message=f"Authentifizierungsfehler: {e!s}"
            )

    @trace_function("security.authorize")
    async def authorize(
        self,
        context: SecurityContext,
        operation: str,
        _resource: str | None = None
    ) -> AuthorizationResult:
        """Autorisiert eine Operation.

        Args:
            context: Sicherheitskontext
            operation: Angeforderte Operation
            _resource: Optionale Ressource

        Returns:
            Autorisierungsergebnis
        """
        if not self.config.enable_authorization:
            return AuthorizationResult(success=True)

        try:
            # Rate Limiting prüfen
            if not await self._check_rate_limit(context.agent_id, operation):
                await self._record_violation(
                    SecurityViolationType.RATE_LIMIT_EXCEEDED,
                    context.agent_id,
                    operation,
                    "Rate Limit überschritten"
                )
                return AuthorizationResult(
                    success=False,
                    error_message="Rate Limit überschritten"
                )

            # Policy-basierte Autorisierung
            auth_result = await self._check_policies(context, operation, _resource)

            if not auth_result.success:
                await self._record_violation(
                    SecurityViolationType.AUTHORIZATION_DENIED,
                    context.agent_id,
                    operation,
                    auth_result.error_message or "Autorisierung verweigert"
                )

            return auth_result

        except Exception as e:
            logger.error(f"Autorisierung fehlgeschlagen: {e}")
            return AuthorizationResult(
                success=False,
                error_message=f"Autorisierungsfehler: {e!s}"
            )

    async def _perform_authentication(
        self,
        credentials: dict[str, Any],
        agent_id: str
    ) -> AuthenticationResult:
        """Führt die eigentliche Authentifizierung durch."""
        # Vereinfachte Implementierung - in Produktion würde hier
        # eine echte Authentifizierung gegen Identity Provider erfolgen

        logger.debug(f"Authentifizierung für Agent {agent_id} gestartet")

        token = credentials.get("token")
        if not token:
            logger.warning(f"Authentifizierung für Agent {agent_id} fehlgeschlagen: Token fehlt")
            return AuthenticationResult(
                success=False,
                error_message="Token erforderlich"
            )

        # Token-Validierung (vereinfacht)
        if len(token) < 10:
            logger.warning(f"Authentifizierung für Agent {agent_id} fehlgeschlagen: Ungültiges Token")
            return AuthenticationResult(
                success=False,
                error_message="Ungültiges Token"
            )

        # Erfolgreiche Authentifizierung
        session_id = str(uuid.uuid4())
        expires_at = time.time() + self.config.auth_token_ttl_seconds

        logger.info(f"Authentifizierung für Agent {agent_id} erfolgreich")
        return AuthenticationResult(
            success=True,
            user_id=credentials.get("user_id"),
            session_id=session_id,
            token=token,
            expires_at=expires_at
        )

    async def _create_session(
        self,
        auth_result: AuthenticationResult,
        agent_id: str
    ) -> None:
        """Erstellt eine neue Sicherheitssession."""
        if not auth_result.session_id:
            return

        context = SecurityContext(
            context_id=auth_result.session_id,
            agent_id=agent_id,
            user_id=auth_result.user_id,
            session_id=auth_result.session_id,
            expires_at=auth_result.expires_at,
            is_authenticated=True
        )

        self._active_sessions[auth_result.session_id] = context
        logger.debug(f"Session erstellt: {auth_result.session_id}")

    async def _is_locked_out(self, agent_id: str) -> bool:
        """Prüft ob ein Agent gesperrt ist."""
        lockout_until = self._lockouts.get(agent_id)
        if lockout_until and time.time() < lockout_until:
            return True
        return False

    async def _record_failed_attempt(self, agent_id: str) -> None:
        """Zeichnet fehlgeschlagenen Authentifizierungsversuch auf."""
        self._failed_attempts[agent_id] = self._failed_attempts.get(agent_id, 0) + 1

        if self._failed_attempts[agent_id] >= self.config.max_failed_attempts:
            # Account sperren
            lockout_until = time.time() + self.config.lockout_duration_seconds
            self._lockouts[agent_id] = lockout_until
            logger.warning(f"Agent {agent_id} gesperrt bis {lockout_until}")

    async def _reset_failed_attempts(self, agent_id: str) -> None:
        """Setzt fehlgeschlagene Versuche zurück."""
        self._failed_attempts.pop(agent_id, None)
        self._lockouts.pop(agent_id, None)

    async def _check_rate_limit(self, agent_id: str, operation: str) -> bool:
        """Prüft Rate Limiting."""
        if not self.config.enable_rate_limiting:
            return True

        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window_seconds

        # Rate Limit Daten für Agent initialisieren
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = {}

        if operation not in self._rate_limits[agent_id]:
            self._rate_limits[agent_id][operation] = []

        # Alte Einträge entfernen
        self._rate_limits[agent_id][operation] = [
            timestamp for timestamp in self._rate_limits[agent_id][operation]
            if timestamp > window_start
        ]

        # Aktuellen Request hinzufügen
        self._rate_limits[agent_id][operation].append(current_time)

        # Limit prüfen
        return len(self._rate_limits[agent_id][operation]) <= self.config.default_rate_limit

    async def _check_policies(
        self,
        context: SecurityContext,
        operation: str,
        _resource: str | None
    ) -> AuthorizationResult:
        """Prüft Security Policies."""
        # Vereinfachte Policy-Prüfung
        # In Produktion würde hier eine komplexe Policy-Engine verwendet

        if not context.is_authenticated and self.config.require_explicit_permissions:
            return AuthorizationResult(
                success=False,
                error_message="Authentifizierung erforderlich"
            )

        # Standard-Autorisierung basierend auf Security Level
        if context.security_level == SecurityLevel.CRITICAL:
            # Kritische Operationen erfordern explizite Berechtigung
            if operation not in context.permissions:
                return AuthorizationResult(
                    success=False,
                    error_message="Explizite Berechtigung erforderlich"
                )

        return AuthorizationResult(
            success=True,
            allowed_operations={operation},
            security_level=context.security_level
        )

    async def _record_violation(
        self,
        violation_type: SecurityViolationType,
        agent_id: str,
        operation: str,
        description: str
    ) -> None:
        """Zeichnet Sicherheitsverletzung auf."""
        violation = SecurityViolation(
            violation_type=violation_type,
            agent_id=agent_id,
            operation=operation,
            description=description
        )

        self._violations.append(violation)
        logger.warning(f"Sicherheitsverletzung: {violation_type.value} - {description}")

        # Threat Detector benachrichtigen (falls verfügbar)
        if self._threat_detector:
            await self._threat_detector.process_violation(violation)

    @asynccontextmanager
    async def security_context(
        self,
        agent_id: str,
        credentials: dict[str, Any] | None = None
    ):
        """Context Manager für sichere Agent-Ausführung."""
        context = None
        try:
            # Authentifizierung (falls Credentials vorhanden)
            if credentials:
                auth_result = await self.authenticate(credentials, agent_id)
                if not auth_result.success:
                    raise SecurityError(f"Authentifizierung fehlgeschlagen: {auth_result.error_message}")

                context = self._active_sessions.get(auth_result.session_id)

            if not context:
                # Fallback: Basis-Kontext erstellen
                context = SecurityContext(
                    agent_id=agent_id,
                    security_level=self.config.default_security_level
                )

            yield context

        except Exception as e:
            logger.error(f"Security Context Fehler: {e}")
            raise
        finally:
            pass

    def get_security_status(self) -> dict[str, Any]:
        """Gibt aktuellen Security-Status zurück."""
        return {
            "active_sessions": len(self._active_sessions),
            "security_policies": len(self._security_policies),
            "violations_count": len(self._violations),
            "locked_agents": len(self._lockouts),
            "config": {
                "authentication_enabled": self.config.enable_authentication,
                "authorization_enabled": self.config.enable_authorization,
                "encryption_enabled": self.config.enable_encryption,
                "audit_enabled": self.config.enable_audit_logging,
                "threat_detection_enabled": self.config.enable_threat_detection,
            }
        }

    async def invalidate_session(self, session_id: str) -> None:
        """Invalidiert eine aktive Session.

        Args:
            session_id: ID der zu invalidierenden Session
        """
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.debug(f"Session invalidiert: {session_id}")

    async def close(self) -> None:
        """Schließt Security Manager und bereinigt Ressourcen."""
        try:
            # Aktive Sessions beenden
            for session_id in list(self._active_sessions.keys()):
                await self.invalidate_session(session_id)

            # Komponenten herunterfahren
            if self._encryption_manager:
                # Encryption Manager cleanup falls nötig
                pass

            if self._audit_logger:
                # Audit Logger cleanup falls nötig
                pass

            if self._threat_detector:
                # Threat Detector cleanup falls nötig
                pass

            # Caches leeren
            self._security_policies.clear()
            self._violations.clear()
            self._rate_limits.clear()
            self._failed_attempts.clear()
            self._lockouts.clear()

            logger.info("Security Manager erfolgreich geschlossen")

        except Exception as e:
            logger.error(f"Fehler beim Schließen des Security Managers: {e}")
            raise


class SecurityError(Exception):
    """Security-spezifische Exception."""
