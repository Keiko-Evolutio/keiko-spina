# backend/kei_agents/security/authentication.py
"""Authentication und Authorization für KEI-Agents.

Implementiert umfassende Auth-Funktionalitäten:
- Multi-Factor Authentication (MFA)
- Role-Based Access Control (RBAC)
- Token-basierte Authentifizierung
- Session-Management und Security
"""

try:
    import jwt
except ImportError:
    jwt = None
import asyncio
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from kei_logging import get_logger

try:
    import pyotp
except ImportError:
    pyotp = None

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    InvalidTokenError,
    TokenExpiredError,
)

logger = get_logger(__name__)


class RegistrationStatus(Enum):
    """Registration Status Enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    STALE = "stale"


@dataclass
class Permission:
    """Permission Data Structure."""
    name: str
    description: str
    resource_type: str | None = None
    actions: list[str] | None = None


@dataclass
class Role:
    """Role Data Structure."""
    name: str
    description: str
    permissions: list[Permission]
    parent_role: str | None = None


@dataclass
class AuthToken:
    """Authentication Token Data Structure."""
    token: str
    user_id: str
    agent_id: str | None
    expires_at: datetime
    permissions: list[str]
    token_type: str = "bearer"


@dataclass
class AuthSession:
    """Authentication Session Data Structure."""
    session_id: str
    user_id: str
    agent_id: str | None
    created_at: datetime
    last_activity: datetime | None = None
    expires_at: datetime | None = None
    is_active: bool = True


@dataclass
class AuthResult:
    """Authentication Result Data Structure."""
    success: bool
    user_id: str | None = None
    agent_id: str | None = None
    token: str | None = None
    expires_at: datetime | None = None
    mfa_required: bool = False
    error_message: str | None = None


class TokenManager:
    """Token Management System."""

    def __init__(self, secret_key: str, algorithm: str = "HS256",
                 default_expiry: timedelta = timedelta(hours=1)):
        """Initialisiert TokenManager.

        Args:
            secret_key: Secret Key für Token-Signierung
            algorithm: JWT-Algorithmus
            default_expiry: Standard-Ablaufzeit
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.default_expiry = default_expiry
        self._blacklisted_tokens: set = set()

        logger.info(f"TokenManager initialisiert mit {algorithm}")

    def create_token(self, payload: dict[str, Any],
                    expires_in: timedelta | None = None) -> str:
        """Erstellt JWT-Token.

        Args:
            payload: Token-Payload
            expires_in: Optional Ablaufzeit

        Returns:
            JWT-Token als String
        """
        try:
            expiry = expires_in or self.default_expiry
            expires_at = datetime.now(UTC) + expiry

            token_payload = payload.copy()
            token_payload.update({
                "exp": expires_at,
                "iat": datetime.now(UTC),
                "jti": secrets.token_hex(16)  # JWT ID
            })

            token = jwt.encode(token_payload, self.secret_key, algorithm=self.algorithm)
            return token

        except Exception as e:
            logger.error(f"Token-Erstellung fehlgeschlagen: {e}")
            raise AuthenticationError(f"Token-Erstellung fehlgeschlagen: {e}")

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verifiziert JWT-Token.

        Args:
            token: JWT-Token

        Returns:
            Dekodiertes Token-Payload
        """
        try:
            if not token:
                raise InvalidTokenError("Token ist leer")

            if token in self._blacklisted_tokens:
                raise InvalidTokenError("Token ist blacklisted")

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token ist abgelaufen")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Ungültiger Token: {e}")
        except Exception as e:
            logger.error(f"Token-Verifikation fehlgeschlagen: {e}")
            raise InvalidTokenError(f"Token-Verifikation fehlgeschlagen: {e}")

    def blacklist_token(self, token: str) -> None:
        """Fügt Token zur Blacklist hinzu.

        Args:
            token: Zu blacklistender Token
        """
        self._blacklisted_tokens.add(token)
        logger.info("Token zur Blacklist hinzugefügt")

    def is_token_expiring(self, token: str, threshold: timedelta = timedelta(minutes=5)) -> bool:
        """Prüft ob Token bald abläuft.

        Args:
            token: JWT-Token
            threshold: Schwellwert für "bald ablaufend"

        Returns:
            True wenn Token bald abläuft
        """
        try:
            payload = self.verify_token(token)
            expires_at = datetime.fromtimestamp(payload["exp"])
            return datetime.now(UTC) + threshold >= expires_at
        except (TokenExpiredError, InvalidTokenError):
            return True  # Bei Token-Fehlern als ablaufend betrachten
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Token-Expiry-Prüfung: {e}")
            return True  # Bei anderen Fehlern als ablaufend betrachten


class MFAManager:
    """Multi-Factor Authentication Manager."""

    def __init__(self, totp_window: int = 30, backup_codes_count: int = 10,
                 sms_enabled: bool = False):
        """Initialisiert MFAManager.

        Args:
            totp_window: TOTP-Zeitfenster in Sekunden
            backup_codes_count: Anzahl Backup-Codes
            sms_enabled: SMS-MFA aktiviert
        """
        self.totp_window = totp_window
        self.backup_codes_count = backup_codes_count
        self.sms_enabled = sms_enabled

        # MFA-Daten Storage
        self._user_secrets: dict[str, str] = {}
        self._backup_codes: dict[str, list[str]] = {}
        self._used_backup_codes: dict[str, set] = {}
        self._sms_codes: dict[str, dict[str, Any]] = {}  # SMS-Codes für temporäre Speicherung

        logger.info(f"MFAManager initialisiert mit TOTP-Window {totp_window}s")

    async def setup_totp(self, user_id: str) -> dict[str, Any]:
        """Richtet TOTP MFA für User ein.

        Args:
            user_id: User-ID

        Returns:
            Setup-Informationen mit Secret und QR-Code
        """
        try:
            if pyotp is None:
                raise AuthenticationError("pyotp nicht verfügbar - MFA nicht unterstützt")

            # Generiere Secret
            secret = pyotp.random_base32()
            self._user_secrets[user_id] = secret

            # Erstelle TOTP-Instanz
            totp = pyotp.TOTP(secret)

            # Generiere QR-Code URL
            qr_url = totp.provisioning_uri(
                name=user_id,
                issuer_name="KEI-Agent System"
            )

            # Generiere Backup-Codes
            backup_codes = [secrets.token_hex(4) for _ in range(self.backup_codes_count)]
            self._backup_codes[user_id] = backup_codes
            self._used_backup_codes[user_id] = set()

            logger.info(f"TOTP MFA für User {user_id} eingerichtet")

            return {
                "secret_key": secret,
                "qr_code_url": qr_url,
                "backup_codes": backup_codes
            }

        except Exception as e:
            logger.error(f"TOTP-Setup für User {user_id} fehlgeschlagen: {e}")
            raise AuthenticationError(f"TOTP-Setup fehlgeschlagen: {e}")

    async def verify_totp(self, user_id: str, totp_code: str) -> dict[str, Any]:
        """Verifiziert TOTP-Code.

        Args:
            user_id: User-ID
            totp_code: TOTP-Code

        Returns:
            Verifikationsergebnis
        """
        try:
            if pyotp is None:
                raise AuthenticationError("pyotp nicht verfügbar - MFA nicht unterstützt")

            if user_id not in self._user_secrets:
                raise AuthenticationError(f"Kein TOTP-Secret für User {user_id}")

            secret = self._user_secrets[user_id]
            totp = pyotp.TOTP(secret)

            # Verifiziere Code
            is_valid = totp.verify(totp_code, valid_window=1)

            return {
                "valid": is_valid,
                "user_id": user_id,
                "timestamp": datetime.now(UTC)
            }

        except Exception as e:
            logger.error(f"TOTP-Verifikation für User {user_id} fehlgeschlagen: {e}")
            return {
                "valid": False,
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now(UTC)
            }

    async def verify_backup_code(self, user_id: str, backup_code: str) -> dict[str, Any]:
        """Verifiziert Backup-Code.

        Args:
            user_id: User-ID
            backup_code: Backup-Code

        Returns:
            Verifikationsergebnis
        """
        try:
            if user_id not in self._backup_codes:
                raise AuthenticationError(f"Keine Backup-Codes für User {user_id}")

            # Prüfe ob Code bereits verwendet
            if backup_code in self._used_backup_codes.get(user_id, set()):
                return {
                    "valid": False,
                    "user_id": user_id,
                    "reason": "code_already_used",
                    "timestamp": datetime.now(UTC)
                }

            # Prüfe ob Code gültig
            if backup_code in self._backup_codes[user_id]:
                # Markiere als verwendet
                self._used_backup_codes[user_id].add(backup_code)

                return {
                    "valid": True,
                    "user_id": user_id,
                    "timestamp": datetime.now(UTC)
                }
            return {
                "valid": False,
                "user_id": user_id,
                "reason": "invalid_code",
                "timestamp": datetime.now(UTC)
            }

        except Exception as e:
            logger.error(f"Backup-Code-Verifikation für User {user_id} fehlgeschlagen: {e}")
            return {
                "valid": False,
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now(UTC)
            }

    async def send_sms_code(self, user_id: str, phone_number: str) -> dict[str, Any]:
        """Sendet SMS-Code für MFA.

        Args:
            user_id: User-ID
            phone_number: Telefonnummer

        Returns:
            SMS-Versand-Ergebnis
        """
        if not self.sms_enabled:
            raise AuthenticationError("SMS-MFA ist nicht aktiviert")

        try:
            # Generiere 6-stelligen Code
            sms_code = f"{secrets.randbelow(1000000):06d}"
            code_id = secrets.token_hex(8)

            # Simuliere SMS-Versand (in echter Implementierung würde hier SMS-Service aufgerufen)
            await self._send_sms(phone_number, sms_code)

            # Speichere Code temporär (in echter Implementierung in Redis/Cache)
            # Hier vereinfacht als Instanz-Variable
            self._sms_codes[code_id] = {
                "user_id": user_id,
                "code": sms_code,
                "phone_number": phone_number,
                "created_at": datetime.now(UTC),
                "expires_at": datetime.now(UTC) + timedelta(minutes=5)
            }

            return {
                "sent": True,
                "code_id": code_id,
                "expires_in": 300  # 5 Minuten
            }

        except Exception as e:
            logger.error(f"SMS-Code-Versand für User {user_id} fehlgeschlagen: {e}")
            return {
                "sent": False,
                "error": str(e)
            }

    async def verify_sms_code(self, user_id: str, code_id: str, sms_code: str) -> dict[str, Any]:
        """Verifiziert SMS-Code.

        Args:
            user_id: User-ID
            code_id: Code-ID
            sms_code: SMS-Code

        Returns:
            Verifikationsergebnis
        """
        try:
            if not hasattr(self, "_sms_codes") or code_id not in self._sms_codes:
                return {
                    "valid": False,
                    "reason": "code_not_found"
                }

            code_data = self._sms_codes[code_id]

            # Prüfe Ablauf
            if datetime.now(UTC) > code_data["expires_at"]:
                del self._sms_codes[code_id]
                return {
                    "valid": False,
                    "reason": "code_expired"
                }

            # Prüfe User-ID
            if code_data["user_id"] != user_id:
                return {
                    "valid": False,
                    "reason": "user_mismatch"
                }

            # Prüfe Code
            if code_data["code"] == sms_code:
                del self._sms_codes[code_id]  # Code nach Verwendung löschen
                return {
                    "valid": True,
                    "user_id": user_id
                }
            return {
                "valid": False,
                "reason": "invalid_code"
            }

        except Exception as e:
            logger.error(f"SMS-Code-Verifikation fehlgeschlagen: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    async def _send_sms(self, phone_number: str, code: str) -> None:
        """Sendet SMS (Mock-Implementierung).

        Args:
            phone_number: Telefonnummer
            code: SMS-Code
        """
        # Mock SMS-Versand
        logger.info(f"SMS-Code {code} an {phone_number} gesendet (Mock)")
        await asyncio.sleep(0.1)  # Simuliere Netzwerk-Delay

    def has_user_mfa_setup(self, user_id: str) -> bool:
        """Prüft ob User MFA eingerichtet hat.

        Args:
            user_id: User-ID

        Returns:
            True wenn User MFA eingerichtet hat
        """
        return user_id in self._user_secrets


class AuthorizationManager:
    """Role-Based Access Control (RBAC) Manager."""

    def __init__(self):
        """Initialisiert AuthorizationManager."""
        self._roles: dict[str, Role] = {}
        self._user_roles: dict[str, list[str]] = {}
        self._permissions: dict[str, Permission] = {}

        # Initialisiere Standard-Permissions und -Roles
        self._initialize_default_permissions()
        self._initialize_default_roles()

        logger.info("AuthorizationManager initialisiert")

    def _initialize_default_permissions(self) -> None:
        """Initialisiert Standard-Permissions."""
        default_permissions = [
            Permission("agent.read", "Agent-Informationen lesen"),
            Permission("agent.write", "Agent-Informationen schreiben"),
            Permission("agent.execute", "Agent-Aktionen ausführen"),
            Permission("agent.admin", "Agent-Administration"),
            Permission("system.read", "System-Informationen lesen"),
            Permission("system.write", "System-Konfiguration ändern"),
            Permission("system.admin", "System-Administration"),
            Permission("security.read", "Security-Logs lesen"),
            Permission("security.admin", "Security-Administration")
        ]

        for perm in default_permissions:
            self._permissions[perm.name] = perm

    def _initialize_default_roles(self) -> None:
        """Initialisiert Standard-Roles."""
        # Agent User Role
        agent_user_perms = [
            self._permissions["agent.read"],
            self._permissions["agent.execute"]
        ]
        self._roles["agent_user"] = Role(
            "agent_user", "Standard Agent User", agent_user_perms
        )

        # Agent Admin Role
        agent_admin_perms = [
            self._permissions["agent.read"],
            self._permissions["agent.write"],
            self._permissions["agent.execute"],
            self._permissions["agent.admin"]
        ]
        self._roles["agent_admin"] = Role(
            "agent_admin", "Agent Administrator", agent_admin_perms
        )

        # System Admin Role
        system_admin_perms = list(self._permissions.values())
        self._roles["system_admin"] = Role(
            "system_admin", "System Administrator", system_admin_perms
        )

    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """Weist User eine Role zu.

        Args:
            user_id: User-ID
            role_name: Role-Name

        Returns:
            True wenn erfolgreich
        """
        try:
            if role_name not in self._roles:
                raise AuthorizationError(f"Role {role_name} existiert nicht")

            if user_id not in self._user_roles:
                self._user_roles[user_id] = []

            if role_name not in self._user_roles[user_id]:
                self._user_roles[user_id].append(role_name)
                logger.info(f"Role {role_name} User {user_id} zugewiesen")

            return True

        except Exception as e:
            logger.error(f"Role-Zuweisung fehlgeschlagen: {e}")
            raise AuthorizationError(f"Role-Zuweisung fehlgeschlagen: {e}")

    async def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Entzieht User eine Role.

        Args:
            user_id: User-ID
            role_name: Role-Name

        Returns:
            True wenn erfolgreich
        """
        try:
            if user_id in self._user_roles and role_name in self._user_roles[user_id]:
                self._user_roles[user_id].remove(role_name)
                logger.info(f"Role {role_name} von User {user_id} entzogen")
                return True

            return False

        except Exception as e:
            logger.error(f"Role-Entzug fehlgeschlagen: {e}")
            raise AuthorizationError(f"Role-Entzug fehlgeschlagen: {e}")

    async def check_permission(self, user_id: str, permission_name: str) -> bool:
        """Prüft ob User Permission hat.

        Args:
            user_id: User-ID
            permission_name: Permission-Name

        Returns:
            True wenn Permission vorhanden
        """
        try:
            if user_id not in self._user_roles:
                return False

            user_roles = self._user_roles[user_id]

            for role_name in user_roles:
                if role_name in self._roles:
                    role = self._roles[role_name]
                    for perm in role.permissions:
                        if perm.name == permission_name:
                            return True

            return False

        except Exception as e:
            logger.error(f"Permission-Check fehlgeschlagen: {e}")
            return False

    async def get_user_permissions(self, user_id: str) -> list[str]:
        """Gibt alle Permissions eines Users zurück.

        Args:
            user_id: User-ID

        Returns:
            Liste der Permission-Namen
        """
        try:
            if user_id not in self._user_roles:
                return []

            permissions = set()
            user_roles = self._user_roles[user_id]

            for role_name in user_roles:
                if role_name in self._roles:
                    role = self._roles[role_name]
                    for perm in role.permissions:
                        permissions.add(perm.name)

            return list(permissions)

        except Exception as e:
            logger.error(f"User-Permissions-Abfrage fehlgeschlagen: {e}")
            return []

    async def create_role(self, role_name: str, description: str,
                         permission_names: list[str]) -> bool:
        """Erstellt neue Role.

        Args:
            role_name: Role-Name
            description: Role-Beschreibung
            permission_names: Liste der Permission-Namen

        Returns:
            True wenn erfolgreich
        """
        try:
            if role_name in self._roles:
                raise AuthorizationError(f"Role {role_name} existiert bereits")

            # Validiere Permissions
            permissions = []
            for perm_name in permission_names:
                if perm_name not in self._permissions:
                    raise AuthorizationError(f"Permission {perm_name} existiert nicht")
                permissions.append(self._permissions[perm_name])

            # Erstelle Role
            self._roles[role_name] = Role(role_name, description, permissions)
            logger.info(f"Role {role_name} erstellt")

            return True

        except Exception as e:
            logger.error(f"Role-Erstellung fehlgeschlagen: {e}")
            raise AuthorizationError(f"Role-Erstellung fehlgeschlagen: {e}")


class SessionManager:
    """Session Management System."""

    def __init__(self, session_timeout: timedelta = timedelta(hours=8),
                 max_sessions_per_user: int = 5):
        """Initialisiert SessionManager.

        Args:
            session_timeout: Session-Timeout
            max_sessions_per_user: Max Sessions pro User
        """
        self.session_timeout = session_timeout
        self.max_sessions_per_user = max_sessions_per_user

        self._sessions: dict[str, AuthSession] = {}
        self._user_sessions: dict[str, list[str]] = {}

        logger.info(f"SessionManager initialisiert mit {session_timeout} Timeout")

    async def create_session(self, user_id: str, agent_id: str | None = None) -> AuthSession:
        """Erstellt neue Session.

        Args:
            user_id: User-ID
            agent_id: Optional Agent-ID

        Returns:
            Neue AuthSession
        """
        try:
            # Bereinige abgelaufene Sessions
            await self._cleanup_expired_sessions()

            # Prüfe Session-Limit
            if user_id in self._user_sessions:
                if len(self._user_sessions[user_id]) >= self.max_sessions_per_user:
                    # Entferne älteste Session
                    oldest_session_id = self._user_sessions[user_id][0]
                    await self.destroy_session(oldest_session_id)

            # Erstelle neue Session
            session_id = secrets.token_hex(32)
            session = AuthSession(
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                created_at=datetime.now(UTC),
                last_activity=datetime.now(UTC),
                expires_at=datetime.now(UTC) + self.session_timeout
            )

            self._sessions[session_id] = session

            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)

            logger.info(f"Session {session_id} für User {user_id} erstellt")
            return session

        except Exception as e:
            logger.error(f"Session-Erstellung fehlgeschlagen: {e}")
            raise AuthenticationError(f"Session-Erstellung fehlgeschlagen: {e}")

    async def get_session(self, session_id: str) -> AuthSession | None:
        """Ruft Session ab.

        Args:
            session_id: Session-ID

        Returns:
            AuthSession oder None
        """
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]

        # Prüfe Ablauf
        if session.expires_at and datetime.now(UTC) > session.expires_at:
            await self.destroy_session(session_id)
            return None

        # Update Last Activity
        session.last_activity = datetime.now(UTC)

        return session

    async def destroy_session(self, session_id: str) -> bool:
        """Zerstört Session.

        Args:
            session_id: Session-ID

        Returns:
            True wenn erfolgreich
        """
        try:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                user_id = session.user_id

                # Entferne aus Sessions
                del self._sessions[session_id]

                # Entferne aus User-Sessions
                if user_id in self._user_sessions:
                    if session_id in self._user_sessions[user_id]:
                        self._user_sessions[user_id].remove(session_id)

                logger.info(f"Session {session_id} zerstört")
                return True

            return False

        except Exception as e:
            logger.error(f"Session-Zerstörung fehlgeschlagen: {e}")
            return False

    async def _cleanup_expired_sessions(self) -> None:
        """Bereinigt abgelaufene Sessions."""
        try:
            current_time = datetime.now(UTC)
            expired_sessions = []

            for session_id, session in self._sessions.items():
                if session.expires_at and current_time > session.expires_at:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                await self.destroy_session(session_id)

            if expired_sessions:
                logger.info(f"{len(expired_sessions)} abgelaufene Sessions bereinigt")

        except Exception as e:
            logger.error(f"Session-Cleanup fehlgeschlagen: {e}")


class AuthenticationManager:
    """Zentraler Authentication Manager für KEI-Agents."""

    def __init__(self, secret_key: str = None, enable_mfa: bool = True,
                 enable_sessions: bool = True):
        """Initialisiert AuthenticationManager.

        Args:
            secret_key: Secret Key für Token-Signierung (optional, wird generiert falls None)
            enable_mfa: MFA aktivieren
            enable_sessions: Session-Management aktivieren
        """
        # Generiere Secret Key falls nicht angegeben
        if secret_key is None:
            secret_key = secrets.token_urlsafe(32)

        self.secret_key = secret_key
        self.enable_mfa = enable_mfa
        self.enable_sessions = enable_sessions

        # Initialisiere Komponenten
        self.token_manager = TokenManager(secret_key)
        self.authorization_manager = AuthorizationManager()

        if enable_mfa:
            self.mfa_manager = MFAManager()

        if enable_sessions:
            self.session_manager = SessionManager()

        # User-Datenbank (vereinfacht)
        self._users: dict[str, dict[str, Any]] = {}

        logger.info(f"AuthenticationManager initialisiert (MFA: {enable_mfa}, Sessions: {enable_sessions})")

    async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
        """Authentifiziert User.

        Args:
            credentials: Authentifizierungs-Credentials

        Returns:
            AuthResult
        """
        try:
            user_id = credentials.get("user_id")
            password = credentials.get("password")
            agent_id = credentials.get("agent_id")

            if not user_id or not password:
                return AuthResult(
                    success=False,
                    error_message="User-ID und Password erforderlich"
                )

            # Prüfe User-Credentials
            if not await self._verify_credentials(user_id, password):
                return AuthResult(
                    success=False,
                    error_message="Ungültige Credentials"
                )

            # Prüfe MFA wenn aktiviert
            if self.enable_mfa and self.mfa_manager.has_user_mfa_setup(user_id):
                mfa_code = credentials.get("mfa_code")
                if not mfa_code:
                    return AuthResult(
                        success=False,
                        mfa_required=True,
                        user_id=user_id,
                        error_message="MFA-Code erforderlich"
                    )

                mfa_result = await self.mfa_manager.verify_totp(user_id, mfa_code)
                if not mfa_result["valid"]:
                    return AuthResult(
                        success=False,
                        error_message="Ungültiger MFA-Code"
                    )

            # Erstelle Token
            token_payload = {
                "user_id": user_id,
                "agent_id": agent_id or "default",
                "permissions": await self.authorization_manager.get_user_permissions(user_id)
            }

            token = self.token_manager.create_token(token_payload)
            expires_at = datetime.now(UTC) + self.token_manager.default_expiry

            # Erstelle Session wenn aktiviert
            if self.enable_sessions:
                await self.session_manager.create_session(user_id, agent_id)

            return AuthResult(
                success=True,
                user_id=user_id,
                agent_id=agent_id,
                token=token,
                expires_at=expires_at
            )

        except Exception as e:
            logger.error(f"Authentifizierung fehlgeschlagen: {e}")
            return AuthResult(
                success=False,
                error_message=f"Authentifizierung fehlgeschlagen: {e}"
            )

    async def authorize(self, token: str, required_permission: str) -> bool:
        """Autorisiert Token für Permission.

        Args:
            token: JWT-Token
            required_permission: Erforderliche Permission

        Returns:
            True wenn autorisiert
        """
        try:
            # Verifiziere Token
            payload = self.token_manager.verify_token(token)
            user_id = payload.get("user_id")

            if not user_id:
                return False

            # Prüfe Permission
            return await self.authorization_manager.check_permission(user_id, required_permission)

        except Exception as e:
            logger.error(f"Autorisierung fehlgeschlagen: {e}")
            return False

    async def _verify_credentials(self, user_id: str, password: str) -> bool:
        """Verifiziert User-Credentials (vereinfacht).

        Args:
            user_id: User-ID
            password: Password

        Returns:
            True wenn gültig
        """
        # Vereinfachte Implementierung - in Produktion würde hier
        # gegen echte User-Datenbank geprüft werden
        if user_id not in self._users:
            # Erstelle Test-User
            self._users[user_id] = {
                "password_hash": hashlib.sha256(password.encode()).hexdigest(),
                "created_at": datetime.now(UTC)
            }
            return True

        stored_hash = self._users[user_id]["password_hash"]
        provided_hash = hashlib.sha256(password.encode()).hexdigest()

        return stored_hash == provided_hash


class RoleManager:
    """Role Management System (Alias für AuthorizationManager)."""

    def __init__(self):
        """Initialisiert RoleManager."""
        self.authorization_manager = AuthorizationManager()

        logger.info("RoleManager initialisiert")

    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """Weist User eine Role zu."""
        return await self.authorization_manager.assign_role(user_id, role_name)

    async def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Entzieht User eine Role."""
        return await self.authorization_manager.revoke_role(user_id, role_name)

    async def check_permission(self, user_id: str, permission_name: str) -> bool:
        """Prüft ob User Permission hat."""
        return await self.authorization_manager.check_permission(user_id, permission_name)

    async def get_user_permissions(self, user_id: str) -> list[str]:
        """Gibt alle Permissions eines Users zurück."""
        return await self.authorization_manager.get_user_permissions(user_id)

    async def create_role(self, role_name: str, description: str,
                         permission_names: list[str]) -> bool:
        """Erstellt neue Role."""
        return await self.authorization_manager.create_role(role_name, description, permission_names)
