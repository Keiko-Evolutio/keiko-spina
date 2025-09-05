# backend/agents/enhanced_security/__init__.py
"""Enhanced Security Package für Keiko Personal Assistant

Enterprise-Grade Security-Management für Multi-Agent-Systeme mit:
- Authentifizierung und Autorisierung
- Verschlüsselung und Datenintegrität
- Audit-Logging und Compliance
- Threat Detection und Response
"""

from __future__ import annotations

import logging

# MIGRATION: Import von neuem audit_system für Backward-Compatibility
import warnings

# Logger für Fallback-Meldungen
logger = logging.getLogger(__name__)

try:
    from audit_system.security_audit_adapter import (
        AuditEvent,
        AuditLevel,
        AuditLogger,
        ComplianceReporter,
    )

    # Erstelle audit_logger Instanz für Backward-Compatibility
    audit_logger = AuditLogger()

    # Deprecation-Warnung für Entwickler
    warnings.warn(
        "Modul backend.agents.enhanced_security.audit_logger ist deprecated. "
        "Verwenden Sie backend.audit_system.security_audit_adapter.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.info("✅ Audit Logger Migration: Erfolgreich zu audit_system migriert")
except ImportError as e:
    # Migration abgeschlossen - keine lokale Implementierung mehr verfügbar
    logger.error(f"❌ Audit Logger Migration: audit_system nicht verfügbar: {e}")
    raise ImportError(
        "Audit Logger Migration abgeschlossen. "
        "Die veraltete audit_logger.py wurde entfernt. "
        "Verwenden Sie backend.audit_system.security_audit_adapter direkt."
    ) from e
from .encryption_manager import (
    CryptoError,
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionManager,
    KeyManager,
)
from .security_manager import (
    AuthenticationResult,
    AuthorizationResult,
    SecurityConfig,
    SecurityContext,
    SecurityLevel,
    SecurityManager,
    SecurityPolicy,
    SecurityViolation,
)
from .threat_detector import (
    SecurityAlert,
    ThreatDetector,
    ThreatEvent,
    ThreatLevel,
    ThreatResponse,
)

# Versionsinformationen
__version__ = "1.0.0"
__author__ = "Security Team"

# Paket-Exporte
__all__ = [
    # Kern-Sicherheit
    "SecurityManager",
    "SecurityConfig",
    "SecurityPolicy",
    "SecurityContext",
    "SecurityViolation",
    "SecurityLevel",
    "AuthenticationResult",
    "AuthorizationResult",
    # Verschlüsselung
    "EncryptionManager",
    "EncryptionConfig",
    "EncryptionAlgorithm",
    "KeyManager",
    "CryptoError",
    # Audit & Compliance
    "audit_logger",
    "AuditLogger",
    "AuditEvent",
    "AuditLevel",
    "ComplianceReporter",
    # Bedrohungserkennung
    "ThreatDetector",
    "ThreatLevel",
    "ThreatEvent",
    "SecurityAlert",
    "ThreatResponse",
]
