# backend/services/enhanced_security_integration/data_models.py
"""Datenmodelle für Enhanced Security Integration.

Definiert alle Datenstrukturen für Enterprise-Grade Security-Features,
Plan Persistence, State Management und Multi-Tenant Security-Isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from security.rbac_abac_system import Action, ResourceType
from security.tenant_isolation import IsolationLevel


class SecurityLevel(Enum):
    """Security-Level für Enhanced Security Integration."""

    PUBLIC = "public"                     # Öffentlich zugänglich
    INTERNAL = "internal"                 # Intern
    CONFIDENTIAL = "confidential"         # Vertraulich
    RESTRICTED = "restricted"             # Eingeschränkt
    TOP_SECRET = "top_secret"            # Streng geheim


class ThreatLevel(Enum):
    """Threat-Level für Security-Event-Monitoring."""

    LOW = "low"                          # Niedrig
    MEDIUM = "medium"                    # Mittel
    HIGH = "high"                        # Hoch
    CRITICAL = "critical"                # Kritisch


class SecurityEventType(Enum):
    """Security-Event-Typen."""

    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_DENIED = "authorization_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    TENANT_ISOLATION_BREACH = "tenant_isolation_breach"
    ENCRYPTION_FAILURE = "encryption_failure"
    CERTIFICATE_VALIDATION_FAILURE = "certificate_validation_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    THREAT_DETECTED = "threat_detected"


class EncryptionAlgorithm(Enum):
    """Encryption-Algorithmen für State Management."""

    AES_256_GCM = "aes_256_gcm"         # AES-256 in GCM Mode
    CHACHA20_POLY1305 = "chacha20_poly1305"  # ChaCha20-Poly1305
    AES_256_CBC = "aes_256_cbc"         # AES-256 in CBC Mode


class StateIntegrityStatus(Enum):
    """Status der State-Integrität."""

    VALID = "valid"                      # Gültig
    TAMPERED = "tampered"                # Manipuliert
    CORRUPTED = "corrupted"              # Beschädigt
    UNKNOWN = "unknown"                  # Unbekannt


@dataclass
class SecurityContext:
    """Security-Kontext für Enhanced Security Integration."""

    # Principal-Informationen
    user_id: str | None = None
    service_account_id: str | None = None
    tenant_id: str | None = None

    # Authentication-Informationen
    authentication_method: str | None = None
    token_type: str | None = None
    token_claims: dict[str, Any] = field(default_factory=dict)

    # Authorization-Informationen
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)

    # Security-Level
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    clearances: list[str] = field(default_factory=list)

    # Request-Kontext
    request_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None

    # mTLS-Informationen
    client_certificate_subject: str | None = None
    client_certificate_fingerprint: str | None = None

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None


@dataclass
class SecurityEvent:
    """Security-Event für Monitoring und Threat-Detection."""

    # Event-Identifikation
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    description: str

    # Event-Kontext
    security_context: SecurityContext
    resource_type: ResourceType | None = None
    resource_id: str | None = None
    action: Action | None = None
    details: dict[str, Any] = field(default_factory=dict)

    # Threat-Detection-Informationen
    threat_indicators: list[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 - 1.0

    # Response-Informationen
    response_actions: list[str] = field(default_factory=list)
    mitigation_applied: bool = False

    # Metadaten
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_component: str = "enhanced_security_integration"
    correlation_id: str | None = None


@dataclass
class EncryptedState:
    """Verschlüsselter State für Plan Persistence."""

    # Encryption-Metadaten
    state_id: str
    encryption_algorithm: EncryptionAlgorithm
    key_id: str

    # Verschlüsselte Daten
    encrypted_data: bytes
    initialization_vector: bytes
    authentication_tag: bytes

    # Integrity-Protection
    integrity_hash: str
    tamper_protection_signature: str

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    # Access-Control
    tenant_id: str | None = None
    access_permissions: list[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL


@dataclass
class StateIntegrityCheck:
    """Ergebnis einer State-Integrity-Prüfung."""

    # Check-Ergebnis
    status: StateIntegrityStatus
    is_valid: bool

    # Check-Details
    integrity_hash_valid: bool
    tamper_protection_valid: bool
    encryption_valid: bool

    # Violation-Details
    violations: list[str] = field(default_factory=list)
    threat_indicators: list[str] = field(default_factory=list)

    # Performance-Metriken
    check_duration_ms: float = 0.0

    # Metadaten
    check_timestamp: datetime = field(default_factory=datetime.utcnow)
    checker_version: str = "1.0.0"


@dataclass
class TenantSecurityBoundary:
    """Security-Boundary für Multi-Tenant-Isolation."""

    # Tenant-Identifikation
    tenant_id: str
    tenant_name: str

    # Isolation-Konfiguration
    isolation_level: IsolationLevel
    allowed_cross_tenant_access: set[str] = field(default_factory=set)

    # Security-Policies
    security_policies: list[str] = field(default_factory=list)
    encryption_requirements: dict[str, Any] = field(default_factory=dict)

    # Network-Isolation
    allowed_ip_ranges: list[str] = field(default_factory=list)
    blocked_ip_ranges: list[str] = field(default_factory=list)

    # Resource-Isolation
    isolated_resources: set[str] = field(default_factory=set)
    shared_resources: set[str] = field(default_factory=set)

    # Compliance-Anforderungen
    compliance_standards: list[str] = field(default_factory=list)
    audit_requirements: dict[str, Any] = field(default_factory=dict)

    # Metadaten
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class SecureCommunicationChannel:
    """Secure Communication Channel für Service-to-Service-Kommunikation."""

    # Channel-Identifikation
    channel_id: str
    source_service: str
    target_service: str

    # Encryption-Konfiguration
    encryption_enabled: bool = True
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_rotation_interval_hours: int = 24

    # mTLS-Konfiguration
    mtls_enabled: bool = True
    client_certificate_required: bool = True
    server_certificate_validation: bool = True

    # Authentication
    authentication_method: str = "mutual_tls"
    token_validation_enabled: bool = True

    # Security-Policies
    allowed_operations: list[str] = field(default_factory=list)
    rate_limits: dict[str, int] = field(default_factory=dict)

    # Monitoring
    security_monitoring_enabled: bool = True
    threat_detection_enabled: bool = True

    # Metadaten
    established_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = None
    is_active: bool = True


@dataclass
class SecurityCheckResult:
    """Ergebnis einer Security-Prüfung."""

    # Check-Status
    is_secure: bool
    security_score: float  # 0.0 - 1.0

    # Check-Details
    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Security-Events
    security_events: list[SecurityEvent] = field(default_factory=list)

    # Performance-Metriken
    check_duration_ms: float = 0.0
    overhead_ms: float = 0.0

    # Metadaten
    check_timestamp: datetime = field(default_factory=datetime.utcnow)
    checker_component: str = "enhanced_security_integration"
    check_version: str = "1.0.0"


@dataclass
class ThreatDetectionResult:
    """Ergebnis einer Threat-Detection-Analyse."""

    # Threat-Status
    threat_detected: bool
    threat_level: ThreatLevel
    confidence: float  # 0.0 - 1.0

    # Threat-Details
    threat_types: list[str] = field(default_factory=list)
    threat_indicators: list[str] = field(default_factory=list)
    attack_patterns: list[str] = field(default_factory=list)

    # Risk-Assessment
    risk_score: float = 0.0  # 0.0 - 1.0
    impact_assessment: str = ""
    likelihood_assessment: str = ""

    # Response-Empfehlungen
    recommended_actions: list[str] = field(default_factory=list)
    immediate_mitigations: list[str] = field(default_factory=list)

    # Metadaten
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    detector_component: str = "threat_detection_engine"
    analysis_version: str = "1.0.0"


@dataclass
class SecurityPerformanceMetrics:
    """Performance-Metriken für Security-Operationen."""

    # Authentication-Performance
    avg_authentication_time_ms: float = 0.0
    authentication_success_rate: float = 1.0
    authentication_failure_rate: float = 0.0

    # Authorization-Performance
    avg_authorization_time_ms: float = 0.0
    authorization_success_rate: float = 1.0
    authorization_denial_rate: float = 0.0

    # Encryption-Performance
    avg_encryption_time_ms: float = 0.0
    avg_decryption_time_ms: float = 0.0
    encryption_success_rate: float = 1.0

    # Threat-Detection-Performance
    avg_threat_detection_time_ms: float = 0.0
    threat_detection_accuracy: float = 1.0
    false_positive_rate: float = 0.0

    # Overall-Performance
    total_security_overhead_ms: float = 0.0
    security_sla_compliance: bool = True

    # Metadaten
    measurement_period_start: datetime = field(default_factory=datetime.utcnow)
    measurement_period_end: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0
