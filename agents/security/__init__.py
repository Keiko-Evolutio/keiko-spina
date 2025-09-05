# backend/kei_agents/security/__init__.py
"""KEI-Agent Security System.

Umfassendes Sicherheitssystem für KEI-Agents mit:
- Encryption und Key Management
- Threat Detection und Anomalie-Erkennung
- Authentication und Authorization
- Secure Communication zwischen Agents
"""

from typing import Any

from kei_logging import get_logger

# Setup Logging
logger = get_logger(__name__)

# Explizite Imports für IDE-Kompatibilität
try:
    from .encryption import EncryptionManager
except ImportError:
    EncryptionManager = None

try:
    from .threat_detection import ThreatDetector
except ImportError:
    ThreatDetector = None

try:
    from .authentication import AuthenticationManager
except ImportError:
    AuthenticationManager = None

try:
    from .secure_communication import SecureCommunicationManager
except ImportError:
    SecureCommunicationManager = None

try:
    from .exceptions import (
        AuthenticationError,
        AuthorizationError,
        CommunicationSecurityError,
        DecryptionError,
        EncryptionError,
        SecurityError,
        SecurityViolationError,
        ThreatDetectionError,
    )
except ImportError:
    # Fallback Exception-Klassen
    class SecurityError(Exception):
        """Basis Security Exception."""

    class EncryptionError(SecurityError):
        """Encryption-spezifische Fehler."""

    class DecryptionError(SecurityError):
        """Decryption-spezifische Fehler."""

    class AuthenticationError(SecurityError):
        """Authentication-Fehler."""

    class AuthorizationError(SecurityError):
        """Authorization-Fehler."""

    class ThreatDetectionError(SecurityError):
        """Threat Detection-Fehler."""

    class SecurityViolationError(SecurityError):
        """Security Violation-Fehler."""

    class CommunicationSecurityError(SecurityError):
        """Secure Communication-Fehler."""

# Version Information
__version__ = "1.0.0"
__author__ = "KEI-Agent Security Team"

# Security System Status
_security_initialized = False
_security_components = {
    "encryption": False,
    "threat_detection": False,
    "authentication": False,
    "secure_communication": False
}


def initialize_security_system() -> dict[str, Any]:
    """Initialisiert das KEI-Agent Security System.

    Returns:
        Dict mit Initialisierungsstatus und verfügbaren Komponenten
    """
    global _security_initialized, _security_components

    try:
        # Teste Imports der Security-Komponenten
        try:
            import agents.security.encryption  # noqa: F401
            _security_components["encryption"] = True
        except ImportError:
            _security_components["encryption"] = False

        try:
            import agents.security.threat_detection  # noqa: F401
            _security_components["threat_detection"] = True
        except ImportError:
            _security_components["threat_detection"] = False

        try:
            import agents.security.authentication  # noqa: F401
            _security_components["authentication"] = True
        except ImportError:
            _security_components["authentication"] = False

        try:
            import agents.security.secure_communication  # noqa: F401
            _security_components["secure_communication"] = True
        except ImportError:
            _security_components["secure_communication"] = False

        _security_initialized = True

        logger.info(f"KEI-Agent Security System v{__version__} erfolgreich initialisiert")

        return {
            "initialized": True,
            "version": __version__,
            "components": _security_components.copy(),
            "features": [
                "AES-256-GCM Encryption",
                "RSA-2048 Asymmetric Encryption",
                "ML-based Anomaly Detection",
                "Multi-Factor Authentication",
                "Role-Based Access Control",
                "End-to-End Secure Communication"
            ]
        }

    except Exception as security_error:
        logger.error(f"Fehler beim Initialisieren des Security Systems: {security_error}")
        return {
            "initialized": False,
            "error": str(security_error),
            "components": _security_components.copy()
        }


def get_security_status() -> dict[str, Any]:
    """Gibt aktuellen Status des Security Systems zurück.

    Returns:
        Dict mit Security-System-Status
    """
    return {
        "initialized": _security_initialized,
        "version": __version__,
        "components": _security_components.copy()
    }


# Exportiere Hauptklassen
__all__ = [
    # Core Managers
    "EncryptionManager",
    "ThreatDetector",
    "AuthenticationManager",
    "SecureCommunicationManager",
    # Exception Classes
    "SecurityError",
    "EncryptionError",
    "DecryptionError",
    "AuthenticationError",
    "AuthorizationError",
    "ThreatDetectionError",
    "SecurityViolationError",
    "CommunicationSecurityError",
    # Functions
    "initialize_security_system",
    "get_security_status"
]

# Alle Klassen sind bereits explizit importiert, keine __getattr__ Funktion nötig


# Auto-Initialisierung beim Import
try:
    _init_result = initialize_security_system()
    if _init_result["initialized"]:
        logger.info("Security System automatisch initialisiert")
except Exception as e:
    logger.warning(f"Auto-Initialisierung des Security Systems fehlgeschlagen: {e}")
