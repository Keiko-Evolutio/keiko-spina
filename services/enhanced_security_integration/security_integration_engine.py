# backend/services/enhanced_security_integration/security_integration_engine.py
"""Enhanced Security Integration Engine.

Implementiert Enterprise-Grade Security-Features mit Integration
aller bestehenden Security-Komponenten und Services.
"""

from __future__ import annotations

import time
from typing import Any

from kei_logging import get_logger
from security.mtls_manager import MTLSManager
from security.rbac_abac_system import (
    Action,
    AuthorizationContext,
    RBACAuthorizationService,
    ResourceType,
)
from security.tenant_isolation import TenantIsolationService

from .data_models import (
    SecureCommunicationChannel,
    SecurityCheckResult,
    SecurityContext,
    SecurityEvent,
    SecurityEventType,
    SecurityLevel,
    TenantSecurityBoundary,
    ThreatLevel,
)
from .plan_persistence_manager import PlanPersistenceManager
from .secure_communication_manager import SecureCommunicationManager
from .threat_detection_engine import ThreatDetectionEngine

logger = get_logger(__name__)


class EnhancedSecurityIntegrationEngine:
    """Enhanced Security Integration Engine für Enterprise-Grade Security."""

    def __init__(
        self,
        rbac_system: RBACAuthorizationService,
        tenant_isolation_service: TenantIsolationService,
        mtls_manager: MTLSManager,
        plan_persistence_manager: PlanPersistenceManager | None = None,
        threat_detection_engine: ThreatDetectionEngine | None = None,
        secure_communication_manager: SecureCommunicationManager | None = None
    ):
        """Initialisiert Enhanced Security Integration Engine.

        Args:
            rbac_system: RBAC/ABAC Authorization Service
            tenant_isolation_service: Tenant Isolation Service
            mtls_manager: mTLS Manager für sichere Kommunikation
            plan_persistence_manager: Plan Persistence Manager
            threat_detection_engine: Threat Detection Engine
            secure_communication_manager: Secure Communication Manager
        """
        self.rbac_system = rbac_system
        self.tenant_isolation_service = tenant_isolation_service
        self.mtls_manager = mtls_manager
        self.plan_persistence_manager = plan_persistence_manager or PlanPersistenceManager()
        self.threat_detection_engine = threat_detection_engine or ThreatDetectionEngine()
        self.secure_communication_manager = secure_communication_manager or SecureCommunicationManager()

        # Security-Konfiguration
        self.enable_enhanced_security = True
        self.enable_threat_detection = True
        self.enable_audit_logging = True
        self.security_check_timeout_ms = 100.0  # < 100ms SLA

        # Performance-Tracking
        self._security_check_count = 0
        self._total_security_check_time_ms = 0.0
        self._authentication_count = 0
        self._authorization_count = 0
        self._threat_detection_count = 0

        # Security-Events
        self._security_events: list[SecurityEvent] = []
        self._max_events_in_memory = 10000

        # Tenant-Security-Boundaries
        self._tenant_boundaries: dict[str, TenantSecurityBoundary] = {}

        # Secure Communication Channels
        self._communication_channels: dict[str, SecureCommunicationChannel] = {}

        logger.info("Enhanced Security Integration Engine initialisiert")

    async def start(self) -> None:
        """Startet Enhanced Security Integration Engine."""
        try:
            # Starte Komponenten
            await self.plan_persistence_manager.start()
            await self.threat_detection_engine.start()
            await self.secure_communication_manager.start()

            # Initialisiere Security-Boundaries
            await self._initialize_tenant_boundaries()

            # Initialisiere Secure Communication Channels
            await self._initialize_communication_channels()

            logger.info("Enhanced Security Integration Engine gestartet")

        except Exception as e:
            logger.error(f"Fehler beim Starten der Enhanced Security Integration Engine: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Enhanced Security Integration Engine."""
        try:
            await self.plan_persistence_manager.stop()
            await self.threat_detection_engine.stop()
            await self.secure_communication_manager.stop()

            logger.info("Enhanced Security Integration Engine gestoppt")

        except Exception as e:
            logger.error(f"Fehler beim Stoppen der Enhanced Security Integration Engine: {e}")

    async def perform_comprehensive_security_check(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> SecurityCheckResult:
        """Führt umfassende Security-Prüfung durch.

        Args:
            security_context: Security-Kontext
            resource_type: Resource-Type
            resource_id: Resource-ID
            action: Action

        Returns:
            Security-Check-Result
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "comprehensive_security_check_started",
                "user_id": security_context.user_id,
                "tenant_id": security_context.tenant_id,
                "resource_type": resource_type.value,
                "action": action.value
            })

            passed_checks = []
            failed_checks = []
            warnings = []
            security_events = []

            # 1. Authentication-Validation
            auth_valid = await self._validate_authentication(security_context)
            if auth_valid:
                passed_checks.append("authentication_validation")
            else:
                failed_checks.append("authentication_validation")
                security_events.append(self._create_security_event(
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    ThreatLevel.HIGH,
                    security_context,
                    "Authentication validation failed"
                ))

            # 2. Authorization-Check
            authz_result = await self._perform_authorization_check(
                security_context, resource_type, resource_id, action
            )
            if authz_result:
                passed_checks.append("authorization_check")
            else:
                failed_checks.append("authorization_check")
                security_events.append(self._create_security_event(
                    SecurityEventType.AUTHORIZATION_DENIED,
                    ThreatLevel.MEDIUM,
                    security_context,
                    f"Authorization denied for {resource_type.value}:{action.value}"
                ))

            # 3. Tenant-Isolation-Check
            tenant_isolation_valid = await self._validate_tenant_isolation(
                security_context, resource_type, resource_id, action
            )
            if tenant_isolation_valid:
                passed_checks.append("tenant_isolation_check")
            else:
                failed_checks.append("tenant_isolation_check")
                security_events.append(self._create_security_event(
                    SecurityEventType.TENANT_ISOLATION_BREACH,
                    ThreatLevel.CRITICAL,
                    security_context,
                    "Tenant isolation violation detected"
                ))

            # 4. Threat-Detection
            if self.enable_threat_detection:
                threat_result = await self.threat_detection_engine.analyze_request(
                    security_context, resource_type, resource_id, action
                )

                if not threat_result.threat_detected:
                    passed_checks.append("threat_detection")
                else:
                    failed_checks.append("threat_detection")
                    security_events.append(self._create_security_event(
                        SecurityEventType.THREAT_DETECTED,
                        threat_result.threat_level,
                        security_context,
                        f"Threat detected: {threat_result.threat_types}"
                    ))

            # 5. mTLS-Validation (wenn verfügbar)
            if security_context.client_certificate_fingerprint:
                mtls_valid = await self._validate_mtls_certificate(security_context)
                if mtls_valid:
                    passed_checks.append("mtls_validation")
                else:
                    failed_checks.append("mtls_validation")
                    security_events.append(self._create_security_event(
                        SecurityEventType.CERTIFICATE_VALIDATION_FAILURE,
                        ThreatLevel.HIGH,
                        security_context,
                        "mTLS certificate validation failed"
                    ))

            # 6. Security-Level-Check
            security_level_valid = await self._validate_security_level(
                security_context, resource_type, action
            )
            if security_level_valid:
                passed_checks.append("security_level_check")
            else:
                failed_checks.append("security_level_check")
                warnings.append("Insufficient security level for requested operation")

            # Berechne Security-Score
            total_checks = len(passed_checks) + len(failed_checks)
            security_score = len(passed_checks) / total_checks if total_checks > 0 else 0.0

            # Bestimme Overall-Security-Status
            is_secure = len(failed_checks) == 0 and security_score >= 0.8

            # Performance-Tracking
            check_duration_ms = (time.time() - start_time) * 1000
            self._update_security_performance_stats(check_duration_ms)

            # Erstelle Security-Check-Result
            result = SecurityCheckResult(
                is_secure=is_secure,
                security_score=security_score,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                warnings=warnings,
                security_events=security_events,
                check_duration_ms=check_duration_ms,
                overhead_ms=check_duration_ms  # Gesamte Zeit ist Overhead
            )

            # Logge Security-Events
            for event in security_events:
                await self._log_security_event(event)

            logger.debug({
                "event": "comprehensive_security_check_completed",
                "is_secure": is_secure,
                "security_score": security_score,
                "check_duration_ms": check_duration_ms,
                "passed_checks": len(passed_checks),
                "failed_checks": len(failed_checks)
            })

            return result

        except Exception as e:
            logger.error(f"Comprehensive Security Check fehlgeschlagen: {e}")

            # Fallback: Unsicher bei Fehler
            return SecurityCheckResult(
                is_secure=False,
                security_score=0.0,
                failed_checks=["security_check_error"],
                check_duration_ms=(time.time() - start_time) * 1000
            )

    async def _validate_authentication(self, security_context: SecurityContext) -> bool:
        """Validiert Authentication."""
        try:
            # Prüfe ob Authentication-Informationen vorhanden sind
            if not security_context.user_id and not security_context.service_account_id:
                return False

            # Prüfe Token-Claims
            if not security_context.token_claims:
                return False

            # Prüfe Token-Expiry
            from datetime import datetime
            if security_context.expires_at and security_context.expires_at < datetime.utcnow():
                return False

            self._authentication_count += 1
            return True

        except Exception as e:
            logger.error(f"Authentication validation fehlgeschlagen: {e}")
            return False

    async def _perform_authorization_check(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> bool:
        """Führt Authorization-Check durch."""
        try:
            # Erstelle Authorization-Context
            principal = security_context.user_id or security_context.service_account_id
            if not principal:
                return False

            authz_context = AuthorizationContext(
                principal=principal,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                tenant_id=security_context.tenant_id,
                additional_context={
                    "security_level": security_context.security_level.value,
                    "roles": security_context.roles,
                    "scopes": security_context.scopes
                }
            )

            # Führe Authorization durch
            decision = self.rbac_system.authorize(authz_context)

            self._authorization_count += 1
            return decision.effect.value == "allow"

        except Exception as e:
            logger.error(f"Authorization check fehlgeschlagen: {e}")
            return False

    async def _validate_tenant_isolation(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: Action
    ) -> bool:
        """Validiert Tenant-Isolation."""
        try:
            if not security_context.tenant_id:
                return True  # Keine Tenant-Isolation erforderlich

            # Extrahiere Target-Tenant aus Resource-ID (falls vorhanden)
            target_tenant_id = self._extract_tenant_from_resource(resource_id)

            if not target_tenant_id:
                return True  # Keine Cross-Tenant-Operation

            # Prüfe Cross-Tenant-Access
            access_allowed = self.tenant_isolation_service.validate_tenant_access(
                security_context.tenant_id,
                target_tenant_id,
                resource_type.value,
                action.value
            )

            return access_allowed

        except Exception as e:
            logger.error(f"Tenant isolation validation fehlgeschlagen: {e}")
            return False

    async def _validate_mtls_certificate(self, security_context: SecurityContext) -> bool:
        """Validiert mTLS-Zertifikat."""
        try:
            if not security_context.client_certificate_fingerprint:
                return True  # Kein mTLS erforderlich

            # Prüfe Zertifikat-Gültigkeit
            # TODO: Implementiere echte mTLS-Validation mit MTLSManager - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111

            return True

        except Exception as e:
            logger.error(f"mTLS certificate validation fehlgeschlagen: {e}")
            return False

    async def _validate_security_level(
        self,
        security_context: SecurityContext,
        resource_type: ResourceType,
        action: Action
    ) -> bool:
        """Validiert Security-Level."""
        try:
            # Bestimme erforderliches Security-Level für Resource/Action
            required_level = self._get_required_security_level(resource_type, action)

            # Security-Level-Hierarchie
            from .data_models import SecurityLevel
            level_hierarchy = {
                SecurityLevel.PUBLIC: 0,
                SecurityLevel.INTERNAL: 1,
                SecurityLevel.CONFIDENTIAL: 2,
                SecurityLevel.RESTRICTED: 3,
                SecurityLevel.TOP_SECRET: 4
            }

            user_level = level_hierarchy.get(security_context.security_level, 0)
            required_level_value = level_hierarchy.get(required_level, 0)

            return user_level >= required_level_value

        except Exception as e:
            logger.error(f"Security level validation fehlgeschlagen: {e}")
            return False

    def _get_required_security_level(self, resource_type: ResourceType, action: Action) -> SecurityLevel:
        """Bestimmt erforderliches Security-Level für Resource/Action."""
        from .data_models import SecurityLevel

        # Basis-Security-Level-Mapping
        security_requirements = {
            (ResourceType.AGENT, Action.CREATE): SecurityLevel.CONFIDENTIAL,
            (ResourceType.AGENT, Action.DELETE): SecurityLevel.RESTRICTED,
            (ResourceType.TASK, Action.EXECUTE): SecurityLevel.INTERNAL,
            (ResourceType.CAPABILITY, Action.EXECUTE): SecurityLevel.INTERNAL,
            (ResourceType.RPC, Action.EXECUTE): SecurityLevel.INTERNAL,
        }

        return security_requirements.get((resource_type, action), SecurityLevel.INTERNAL)

    def _extract_tenant_from_resource(self, resource_id: str) -> str | None:
        """Extrahiert Tenant-ID aus Resource-ID."""
        # Einfache Implementierung: Tenant-ID als Prefix
        if ":" in resource_id:
            parts = resource_id.split(":")
            if len(parts) >= 2:
                return parts[0]

        return None

    def _create_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        security_context: SecurityContext,
        description: str
    ) -> SecurityEvent:
        """Erstellt Security-Event."""
        import uuid

        return SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            threat_level=threat_level,
            security_context=security_context,
            description=description
        )

    async def _log_security_event(self, event: SecurityEvent) -> None:
        """Loggt Security-Event."""
        try:
            # Speichere in Memory
            self._security_events.append(event)

            # Memory-Limit prüfen
            if len(self._security_events) > self._max_events_in_memory:
                self._security_events = self._security_events[-self._max_events_in_memory:]

            # Logge Event
            logger.warning({
                "event": "security_event",
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "description": event.description,
                "user_id": event.security_context.user_id,
                "tenant_id": event.security_context.tenant_id
            })

        except Exception as e:
            logger.error(f"Security event logging fehlgeschlagen: {e}")

    async def _initialize_tenant_boundaries(self) -> None:
        """Initialisiert Tenant-Security-Boundaries."""
        try:
            # TODO: Lade Tenant-Boundaries aus Konfiguration - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
            logger.info("Tenant Security Boundaries initialisiert")

        except Exception as e:
            logger.error(f"Tenant boundaries initialization fehlgeschlagen: {e}")

    async def _initialize_communication_channels(self) -> None:
        """Initialisiert Secure Communication Channels."""
        try:
            # TODO: Initialisiere Service-to-Service Communication Channels - Issue: https://github.com/keiko-dev-team/keiko-personal-assistant/issues/111
            logger.info("Secure Communication Channels initialisiert")

        except Exception as e:
            logger.error(f"Communication channels initialization fehlgeschlagen: {e}")

    def _update_security_performance_stats(self, check_duration_ms: float) -> None:
        """Aktualisiert Security-Performance-Statistiken."""
        self._security_check_count += 1
        self._total_security_check_time_ms += check_duration_ms

    def get_security_performance_stats(self) -> dict[str, Any]:
        """Gibt Security-Performance-Statistiken zurück."""
        avg_check_time = (
            self._total_security_check_time_ms / self._security_check_count
            if self._security_check_count > 0 else 0.0
        )

        return {
            "total_security_checks": self._security_check_count,
            "avg_security_check_time_ms": avg_check_time,
            "meets_security_sla": avg_check_time < self.security_check_timeout_ms,
            "authentication_count": self._authentication_count,
            "authorization_count": self._authorization_count,
            "threat_detection_count": self._threat_detection_count,
            "security_events_count": len(self._security_events),
            "enhanced_security_enabled": self.enable_enhanced_security
        }
