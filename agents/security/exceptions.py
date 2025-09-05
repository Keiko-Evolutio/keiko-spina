# backend/kei_agents/security/exceptions.py
"""Security System Exception-Klassen.

Definiert spezifische Exception-Klassen für das KEI-Agent Security System.
"""

from typing import Any


class SecurityError(Exception):
    """Basis-Exception für alle Security-Fehler."""

    def __init__(self, message: str, error_code: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


# Encryption Exceptions
class EncryptionError(SecurityError):
    """Basis-Exception für Encryption-Fehler."""


class DecryptionError(EncryptionError):
    """Exception für Decryption-Fehler."""


class KeyNotFoundError(EncryptionError):
    """Exception wenn Schlüssel nicht gefunden wird."""


class InvalidKeyError(EncryptionError):
    """Exception für ungültige Schlüssel."""


class EncryptionConfigError(EncryptionError):
    """Exception für Encryption-Konfigurationsfehler."""


# Threat Detection Exceptions
class ThreatDetectionError(SecurityError):
    """Basis-Exception für Threat Detection-Fehler."""


class SecurityViolationError(ThreatDetectionError):
    """Exception für Security-Verletzungen."""


class AnomalyDetectionError(ThreatDetectionError):
    """Exception für Anomalie-Erkennungsfehler."""


# Authentication Exceptions
class AuthenticationError(SecurityError):
    """Basis-Exception für Authentication-Fehler."""


class AuthorizationError(SecurityError):
    """Exception für Authorization-Fehler."""


class TokenExpiredError(AuthenticationError):
    """Exception für abgelaufene Tokens."""


class InvalidTokenError(AuthenticationError):
    """Exception für ungültige Tokens."""


class MFARequiredError(AuthenticationError):
    """Exception wenn MFA erforderlich ist."""

    def __init__(self, message: str, mfa_token: str, user_id: str):
        super().__init__(message)
        self.mfa_token = mfa_token
        self.user_id = user_id


class PermissionDeniedError(AuthorizationError):
    """Exception für verweigerte Permissions."""


# Secure Communication Exceptions
class CommunicationSecurityError(SecurityError):
    """Basis-Exception für Secure Communication-Fehler."""


class CertificateError(CommunicationSecurityError):
    """Exception für Certificate-Fehler."""


class MessageIntegrityError(CommunicationSecurityError):
    """Exception für Message Integrity-Fehler."""


class ChannelEstablishmentError(CommunicationSecurityError):
    """Exception für Channel Establishment-Fehler."""


# Utility Functions
def create_security_error(error_type: str, message: str,
                         error_code: str | None = None,
                         details: dict[str, Any] | None = None) -> SecurityError:
    """Factory-Funktion für Security-Exceptions.
    
    Args:
        error_type: Typ der Exception
        message: Fehlermeldung
        error_code: Optional error code
        details: Optional zusätzliche Details
        
    Returns:
        Entsprechende Security-Exception
    """
    error_map = {
        "encryption": EncryptionError,
        "decryption": DecryptionError,
        "key_not_found": KeyNotFoundError,
        "invalid_key": InvalidKeyError,
        "threat_detection": ThreatDetectionError,
        "security_violation": SecurityViolationError,
        "anomaly_detection": AnomalyDetectionError,
        "authentication": AuthenticationError,
        "authorization": AuthorizationError,
        "token_expired": TokenExpiredError,
        "invalid_token": InvalidTokenError,
        "permission_denied": PermissionDeniedError,
        "communication_security": CommunicationSecurityError,
        "certificate": CertificateError,
        "message_integrity": MessageIntegrityError,
        "channel_establishment": ChannelEstablishmentError
    }

    exception_class = error_map.get(error_type, SecurityError)
    return exception_class(message, error_code, details)


def is_security_error(exception: Exception) -> bool:
    """Prüft ob Exception eine Security-Exception ist.
    
    Args:
        exception: Zu prüfende Exception
        
    Returns:
        True wenn Security-Exception, sonst False
    """
    return isinstance(exception, SecurityError)


def get_error_details(exception: Exception) -> dict[str, Any]:
    """Extrahiert Details aus Security-Exception.
    
    Args:
        exception: Security-Exception
        
    Returns:
        Dict mit Exception-Details
    """
    if isinstance(exception, SecurityError):
        return {
            "type": type(exception).__name__,
            "message": str(exception),
            "error_code": getattr(exception, "error_code", None),
            "details": getattr(exception, "details", {})
        }
    return {
        "type": type(exception).__name__,
        "message": str(exception),
        "error_code": None,
        "details": {}
    }
