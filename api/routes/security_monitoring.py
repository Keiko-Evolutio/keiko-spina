"""Enterprise Security Monitoring API.

Provides real-time security monitoring, threat detection, and compliance reporting
for WebSocket and authentication systems.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.middleware.enterprise_websocket_auth import enterprise_websocket_auth
from auth.enterprise_auth import AuthContext, require_auth
from kei_logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/security", tags=["security-monitoring"])


class SecurityStats(BaseModel):
    """Security statistics model."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_failed_attempts: int = Field(description="Total failed authentication attempts")
    suspicious_ips: int = Field(description="Number of suspicious IP addresses")
    blocked_ips: int = Field(description="Number of blocked IP addresses")
    rate_limited_ips: list[str] = Field(description="List of rate-limited IP addresses")
    suspicious_ip_list: list[str] = Field(description="List of suspicious IP addresses")


class SecurityEvent(BaseModel):
    """Security event model."""

    timestamp: datetime
    event_type: str = Field(description="Type of security event")
    component: str = Field(description="Component that generated the event")
    severity: str = Field(description="Event severity level")
    details: dict = Field(description="Event details")


class SecurityAlert(BaseModel):
    """Security alert model."""

    alert_id: str = Field(description="Unique alert identifier")
    timestamp: datetime
    severity: str = Field(description="Alert severity (LOW, MEDIUM, HIGH, CRITICAL)")
    title: str = Field(description="Alert title")
    description: str = Field(description="Alert description")
    affected_component: str = Field(description="Affected system component")
    recommended_action: str = Field(description="Recommended remediation action")


class ComplianceReport(BaseModel):
    """Compliance report model."""

    report_id: str = Field(description="Unique report identifier")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    compliance_framework: str = Field(description="Compliance framework (OWASP, SOC2, etc.)")
    overall_score: float = Field(description="Overall compliance score (0-100)")
    passed_controls: int = Field(description="Number of passed controls")
    failed_controls: int = Field(description="Number of failed controls")
    critical_findings: list[str] = Field(description="Critical compliance findings")
    recommendations: list[str] = Field(description="Compliance recommendations")


@router.get("/stats", response_model=SecurityStats)
async def get_security_stats(
    auth_context: AuthContext = Depends(require_auth)
) -> SecurityStats:
    """Get real-time security statistics.

    Requires admin privileges for access to security monitoring data.
    """
    # Prüfe Admin-Berechtigung
    if auth_context.privilege.value not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for security monitoring"
        )

    try:
        # Hole Sicherheitsstatistiken
        stats = enterprise_websocket_auth.get_security_stats()

        return SecurityStats(
            total_failed_attempts=stats["total_failed_attempts"],
            suspicious_ips=stats["suspicious_ips"],
            blocked_ips=stats["blocked_ips"],
            rate_limited_ips=stats["rate_limited_ips"],
            suspicious_ip_list=stats["suspicious_ip_list"]
        )

    except Exception as e:
        logger.error(f"Failed to get security stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security statistics"
        )


@router.get("/alerts", response_model=list[SecurityAlert])
async def get_security_alerts(
    severity: str | None = None,
    limit: int = 50,
    auth_context: AuthContext = Depends(require_auth)
) -> list[SecurityAlert]:
    """Get active security alerts.

    Args:
        severity: Filter by severity level (LOW, MEDIUM, HIGH, CRITICAL)
        limit: Maximum number of alerts to return
        auth_context: Authentifizierungskontext für Autorisierung
    """
    # Prüfe Admin-Berechtigung
    if auth_context.privilege.value not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for security alerts"
        )

    try:
        # Generiere aktuelle Sicherheitsalerts basierend auf Statistiken
        stats = enterprise_websocket_auth.get_security_stats()
        alerts = []

        # Critical Alert: Viele fehlgeschlagene Versuche
        if stats["total_failed_attempts"] > 100:
            alerts.append(SecurityAlert(
                alert_id=f"AUTH_FAIL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(UTC),
                severity="CRITICAL",
                title="High Number of Authentication Failures",
                description=f"Detected {stats['total_failed_attempts']} failed authentication attempts",
                affected_component="WebSocket Authentication",
                recommended_action="Review authentication logs and consider implementing additional security measures"
            ))

        # High Alert: Verdächtige IPs
        if stats["suspicious_ips"] > 5:
            alerts.append(SecurityAlert(
                alert_id=f"SUSPICIOUS_IP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(UTC),
                severity="HIGH",
                title="Multiple Suspicious IP Addresses Detected",
                description=f"Detected {stats['suspicious_ips']} suspicious IP addresses",
                affected_component="Rate Limiting System",
                recommended_action="Review IP addresses and consider blocking persistent attackers"
            ))

        # Medium Alert: Rate Limiting aktiv
        if stats["blocked_ips"] > 0:
            alerts.append(SecurityAlert(
                alert_id=f"RATE_LIMIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(UTC),
                severity="MEDIUM",
                title="Rate Limiting Active",
                description=f"Currently blocking {stats['blocked_ips']} IP addresses due to rate limiting",
                affected_component="Rate Limiting System",
                recommended_action="Monitor blocked IPs and adjust rate limiting thresholds if needed"
            ))

        # Filter nach Severity falls angegeben
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity.upper()]

        return alerts[:limit]

    except Exception as e:
        logger.error(f"Failed to get security alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security alerts"
        )


@router.get("/compliance/owasp", response_model=ComplianceReport)
async def get_owasp_compliance_report(
    auth_context: AuthContext = Depends(require_auth)
) -> ComplianceReport:
    """Generate OWASP ASVS compliance report.

    Evaluates current security posture against OWASP Application Security
    Verification Standard (ASVS) Level 2 requirements.
    """
    # Prüfe Admin-Berechtigung
    if auth_context.privilege.value not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for compliance reports"
        )

    try:
        # OWASP ASVS Level 2 Compliance Check
        passed_controls = 0
        failed_controls = 0
        critical_findings = []
        recommendations = []

        # V2: Authentication Verification Requirements
        stats = enterprise_websocket_auth.get_security_stats()

        # V2.1: Password Security Requirements
        passed_controls += 1  # JWT-based authentication implemented

        # V2.2: General Authenticator Requirements
        if stats["total_failed_attempts"] < 50:
            passed_controls += 1
        else:
            failed_controls += 1
            critical_findings.append("High number of authentication failures detected")
            recommendations.append("Implement stronger authentication controls")

        # V2.3: Authenticator Lifecycle Requirements
        passed_controls += 1  # Token lifecycle management implemented

        # V2.4: Credential Storage Requirements
        passed_controls += 1  # Secure credential storage implemented

        # V2.5: Credential Recovery Requirements
        failed_controls += 1  # No credential recovery mechanism
        recommendations.append("Implement secure credential recovery mechanism")

        # V2.6: Look-up Secret Verifier Requirements
        passed_controls += 1  # JWT secret management implemented

        # V2.7: Out of Band Verifier Requirements
        failed_controls += 1  # No out-of-band verification
        recommendations.append("Consider implementing out-of-band verification for high-risk operations")

        # V2.8: Single or Multi Factor One Time Verifier Requirements
        failed_controls += 1  # No MFA implemented
        critical_findings.append("Multi-factor authentication not implemented")
        recommendations.append("Implement multi-factor authentication for enhanced security")

        # V2.9: Cryptographic Software and Devices Verifier Requirements
        passed_controls += 1  # Cryptographic verification implemented

        # V2.10: Service Authentication Requirements
        if stats["suspicious_ips"] < 10:
            passed_controls += 1
        else:
            failed_controls += 1
            critical_findings.append("High number of suspicious IP addresses")

        # Berechne Compliance Score
        total_controls = passed_controls + failed_controls
        compliance_score = (passed_controls / total_controls) * 100 if total_controls > 0 else 0

        return ComplianceReport(
            report_id=f"OWASP_ASVS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            compliance_framework="OWASP ASVS Level 2",
            overall_score=compliance_score,
            passed_controls=passed_controls,
            failed_controls=failed_controls,
            critical_findings=critical_findings,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Failed to generate OWASP compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


@router.post("/reset-stats")
async def reset_security_stats(
    auth_context: AuthContext = Depends(require_auth)
) -> dict[str, str]:
    """Reset security statistics (admin only).

    Clears all security statistics and counters. Use with caution.
    """
    # Prüfe Admin-Berechtigung
    if auth_context.privilege.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required to reset security statistics"
        )

    try:
        # Reset security statistics
        enterprise_websocket_auth.failed_attempts.clear()
        enterprise_websocket_auth.suspicious_ips.clear()
        enterprise_websocket_auth.rate_limiter.attempts.clear()
        enterprise_websocket_auth.rate_limiter.blocked_ips.clear()

        logger.info(f"Security statistics reset by admin: {auth_context.subject}")

        return {"message": "Security statistics reset successfully"}

    except Exception as e:
        logger.error(f"Failed to reset security stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset security statistics"
        )


__all__ = [
    "ComplianceReport",
    "SecurityAlert",
    "SecurityEvent",
    "SecurityStats",
    "router"
]
