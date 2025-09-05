"""
JWT Authentication Service für Keiko Platform
Zentrale Authentifizierung zwischen Services
"""

import jwt
import secrets
from datetime import datetime, timedelta, UTC
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from kei_logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)


class ServiceRole(str, Enum):
    """Service-Rollen für Autorisierung"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    SDK = "sdk"
    API_CONTRACTS = "api_contracts"
    MONITORING = "monitoring"


@dataclass
class JWTClaims:
    """JWT Claims für Service-Authentifizierung"""
    service_id: str
    service_role: ServiceRole
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    issuer: str = "keiko-auth-service"


class JWTAuthService:
    """JWT Authentication Service"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.default_expiry = timedelta(hours=24)
        
        # Service-spezifische Permissions
        self.service_permissions = {
            ServiceRole.BACKEND: [
                "agent:register", "agent:unregister", "agent:call",
                "function:register", "function:call", "function:list",
                "user:authenticate", "metrics:read"
            ],
            ServiceRole.FRONTEND: [
                "agent:list", "agent:status", "function:call",
                "user:profile", "metrics:read"
            ],
            ServiceRole.SDK: [
                "agent:register", "agent:unregister", "agent:heartbeat",
                "function:register", "function:unregister", "function:call"
            ],
            ServiceRole.API_CONTRACTS: [
                "schema:read", "version:read"
            ],
            ServiceRole.MONITORING: [
                "metrics:read", "metrics:write", "logs:read", "alerts:read"
            ]
        }
        
        logger.info("JWT Auth Service initialisiert")
    
    def generate_service_token(
        self, 
        service_id: str, 
        service_role: ServiceRole,
        custom_permissions: List[str] = None,
        expiry: timedelta = None
    ) -> str:
        """Generiert JWT Token für Service"""
        
        now = datetime.now(UTC)
        expires_at = now + (expiry or self.default_expiry)
        
        # Permissions basierend auf Service-Rolle
        permissions = custom_permissions or self.service_permissions.get(service_role, [])
        
        claims = JWTClaims(
            service_id=service_id,
            service_role=service_role,
            permissions=permissions,
            issued_at=now,
            expires_at=expires_at
        )
        
        # JWT Payload
        payload = {
            "sub": service_id,
            "role": service_role.value,
            "permissions": permissions,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "iss": claims.issuer,
            "jti": secrets.token_urlsafe(16)  # JWT ID
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"JWT Token generiert für Service: {service_id} ({service_role.value})")
        return token
    
    def verify_token(self, token: str) -> Optional[JWTClaims]:
        """Verifiziert JWT Token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Claims extrahieren
            claims = JWTClaims(
                service_id=payload["sub"],
                service_role=ServiceRole(payload["role"]),
                permissions=payload["permissions"],
                issued_at=datetime.fromtimestamp(payload["iat"], UTC),
                expires_at=datetime.fromtimestamp(payload["exp"], UTC),
                issuer=payload["iss"]
            )
            
            logger.debug(f"JWT Token verifiziert für Service: {claims.service_id}")
            return claims
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT Token abgelaufen")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Ungültiger JWT Token: {e}")
            return None
    
    def check_permission(self, claims: JWTClaims, required_permission: str) -> bool:
        """Prüft ob Service die erforderliche Permission hat"""
        has_permission = required_permission in claims.permissions
        
        if not has_permission:
            logger.warning(
                f"Permission verweigert: Service {claims.service_id} "
                f"benötigt '{required_permission}', hat aber nur {claims.permissions}"
            )
        
        return has_permission
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Erneuert JWT Token"""
        claims = self.verify_token(token)
        if not claims:
            return None
        
        # Neuen Token mit gleichen Claims generieren
        return self.generate_service_token(
            service_id=claims.service_id,
            service_role=claims.service_role,
            custom_permissions=claims.permissions
        )


class APIKeyManager:
    """API Key Management für Service-to-Service Kommunikation"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        logger.info("API Key Manager initialisiert")
    
    def generate_api_key(
        self, 
        service_id: str, 
        service_role: ServiceRole,
        description: str = ""
    ) -> str:
        """Generiert API Key für Service"""
        
        api_key = f"keiko_{service_role.value}_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "service_id": service_id,
            "service_role": service_role.value,
            "description": description,
            "created_at": datetime.now(UTC).isoformat(),
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        logger.info(f"API Key generiert für Service: {service_id} ({service_role.value})")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verifiziert API Key"""
        key_info = self.api_keys.get(api_key)
        
        if not key_info or not key_info["active"]:
            logger.warning(f"Ungültiger oder inaktiver API Key: {api_key[:20]}...")
            return None
        
        # Usage tracking
        key_info["last_used"] = datetime.now(UTC).isoformat()
        key_info["usage_count"] += 1
        
        logger.debug(f"API Key verifiziert für Service: {key_info['service_id']}")
        return key_info
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Widerruft API Key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"API Key widerrufen: {api_key[:20]}...")
            return True
        return False


class SecurityMiddleware:
    """Security Middleware für FastAPI/Flask"""
    
    def __init__(self, jwt_service: JWTAuthService, api_key_manager: APIKeyManager):
        self.jwt_service = jwt_service
        self.api_key_manager = api_key_manager
        logger.info("Security Middleware initialisiert")
    
    def authenticate_request(self, authorization_header: str) -> Optional[Dict]:
        """Authentifiziert Request basierend auf Authorization Header"""
        
        if not authorization_header:
            return None
        
        # JWT Token Authentication
        if authorization_header.startswith("Bearer "):
            token = authorization_header[7:]  # Remove "Bearer "
            claims = self.jwt_service.verify_token(token)
            if claims:
                return {
                    "auth_type": "jwt",
                    "service_id": claims.service_id,
                    "service_role": claims.service_role.value,
                    "permissions": claims.permissions
                }
        
        # API Key Authentication
        elif authorization_header.startswith("ApiKey "):
            api_key = authorization_header[7:]  # Remove "ApiKey "
            key_info = self.api_key_manager.verify_api_key(api_key)
            if key_info:
                return {
                    "auth_type": "api_key",
                    "service_id": key_info["service_id"],
                    "service_role": key_info["service_role"],
                    "permissions": self.jwt_service.service_permissions.get(
                        ServiceRole(key_info["service_role"]), []
                    )
                }
        
        return None
    
    def require_permission(self, auth_info: Dict, required_permission: str) -> bool:
        """Prüft erforderliche Permission"""
        if not auth_info:
            return False
        
        return required_permission in auth_info.get("permissions", [])


# Beispiel-Usage
if __name__ == "__main__":
    # JWT Service initialisieren
    jwt_service = JWTAuthService()
    api_key_manager = APIKeyManager()
    security_middleware = SecurityMiddleware(jwt_service, api_key_manager)
    
    # Service Tokens generieren
    backend_token = jwt_service.generate_service_token("keiko-backend-001", ServiceRole.BACKEND)
    sdk_token = jwt_service.generate_service_token("kei-agent-sdk-001", ServiceRole.SDK)
    
    # API Keys generieren
    backend_api_key = api_key_manager.generate_api_key(
        "keiko-backend-001", 
        ServiceRole.BACKEND,
        "Backend Service API Key"
    )
    
    print(f"Backend JWT Token: {backend_token}")
    print(f"SDK JWT Token: {sdk_token}")
    print(f"Backend API Key: {backend_api_key}")
    
    # Token verifizieren
    claims = jwt_service.verify_token(backend_token)
    if claims:
        print(f"Token verifiziert für: {claims.service_id}")
        print(f"Permissions: {claims.permissions}")
    
    # Permission prüfen
    has_permission = jwt_service.check_permission(claims, "agent:register")
    print(f"Hat 'agent:register' Permission: {has_permission}")
