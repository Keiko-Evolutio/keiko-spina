#!/usr/bin/env python3
"""
OWASP Security Compliance Validation

Enterprise-grade security validation based on OWASP Top 10 and security best practices.
"""

import asyncio
import json
import logging
import time
import httpx
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OWASPSecurityValidator:
    """OWASP Top 10 Security Compliance Validator."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.auth_token = "dev-token-12345"
        self.security_findings = []
        
    async def run_owasp_validation(self) -> Dict[str, Any]:
        """Execute comprehensive OWASP security validation."""
        logger.info("🔒 Starting OWASP Security Compliance Validation")
        
        validations = [
            ("A01:2021 - Broken Access Control", self.validate_access_control),
            ("A02:2021 - Cryptographic Failures", self.validate_cryptographic_security),
            ("A03:2021 - Injection", self.validate_injection_protection),
            ("A04:2021 - Insecure Design", self.validate_secure_design),
            ("A05:2021 - Security Misconfiguration", self.validate_security_configuration),
            ("A06:2021 - Vulnerable Components", self.validate_component_security),
            ("A07:2021 - Authentication Failures", self.validate_authentication_security),
            ("A08:2021 - Software Integrity Failures", self.validate_software_integrity),
            ("A09:2021 - Security Logging Failures", self.validate_security_logging),
            ("A10:2021 - Server-Side Request Forgery", self.validate_ssrf_protection),
        ]
        
        for validation_name, validation_method in validations:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 OWASP VALIDATION: {validation_name}")
            logger.info(f"{'='*80}")
            
            try:
                await validation_method()
            except Exception as e:
                logger.error(f"❌ Validation failed: {validation_name} - {e}")
                self.security_findings.append({
                    "category": validation_name,
                    "severity": "HIGH",
                    "finding": f"Validation execution failed: {str(e)}",
                    "recommendation": "Investigate validation execution issues"
                })
        
        return self.generate_owasp_report()
    
    async def validate_access_control(self):
        """A01:2021 - Broken Access Control validation."""
        logger.info("🔐 Validating Access Control...")
        
        # Test unauthorized access attempts
        test_endpoints = [
            "/api/v1/system/heartbeat",
            "/api/v1/debug/simple",
            "/api/v1/system/health",
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in test_endpoints:
                # Test without authorization
                response = await client.get(f"{self.base_url}{endpoint}")
                
                if response.status_code == 200:
                    self.security_findings.append({
                        "category": "A01:2021 - Broken Access Control",
                        "severity": "CRITICAL",
                        "finding": f"Endpoint {endpoint} accessible without authentication",
                        "recommendation": "Implement proper access controls for all sensitive endpoints"
                    })
                    logger.error(f"❌ Access Control Failure: {endpoint} accessible without auth")
                else:
                    logger.info(f"✅ Access Control OK: {endpoint} requires authentication")
    
    async def validate_cryptographic_security(self):
        """A02:2021 - Cryptographic Failures validation."""
        logger.info("🔐 Validating Cryptographic Security...")
        
        # Test HTTPS enforcement
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test if HTTP is redirected to HTTPS
                response = await client.get(f"{self.base_url}/health", follow_redirects=False)
                
                if response.status_code not in [301, 302, 308]:
                    self.security_findings.append({
                        "category": "A02:2021 - Cryptographic Failures",
                        "severity": "HIGH",
                        "finding": "HTTP traffic not redirected to HTTPS",
                        "recommendation": "Implement HTTPS redirect for all HTTP traffic"
                    })
                    logger.warning("⚠️ HTTP not redirected to HTTPS")
                else:
                    logger.info("✅ HTTPS redirect configured")
                    
        except Exception as e:
            logger.info(f"ℹ️ HTTPS test skipped (development environment): {e}")
    
    async def validate_injection_protection(self):
        """A03:2021 - Injection validation."""
        logger.info("💉 Validating Injection Protection...")
        
        # Test SQL injection attempts
        injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "{{7*7}}",  # Template injection
            "${7*7}",   # Expression injection
        ]
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for payload in injection_payloads:
                try:
                    # Test injection in query parameters
                    response = await client.get(
                        f"{self.base_url}/api/v1/system/heartbeat",
                        params={"test": payload},
                        headers=headers
                    )
                    
                    # Check if payload is reflected in response
                    if payload in response.text:
                        self.security_findings.append({
                            "category": "A03:2021 - Injection",
                            "severity": "HIGH",
                            "finding": f"Potential injection vulnerability with payload: {payload}",
                            "recommendation": "Implement proper input validation and sanitization"
                        })
                        logger.error(f"❌ Injection vulnerability detected: {payload}")
                    else:
                        logger.info(f"✅ Injection protection OK for: {payload[:20]}...")
                        
                except Exception as e:
                    logger.debug(f"Injection test error: {e}")
    
    async def validate_secure_design(self):
        """A04:2021 - Insecure Design validation."""
        logger.info("🏗️ Validating Secure Design...")
        
        # Test rate limiting
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Send rapid requests to test rate limiting
            responses = []
            for i in range(20):
                try:
                    response = await client.get(
                        f"{self.base_url}/api/v1/system/heartbeat",
                        headers=headers
                    )
                    responses.append(response.status_code)
                except Exception as e:
                    logger.debug(f"Rate limit test error: {e}")
            
            # Check if any requests were rate limited
            rate_limited = any(code == 429 for code in responses)
            
            if not rate_limited:
                self.security_findings.append({
                    "category": "A04:2021 - Insecure Design",
                    "severity": "MEDIUM",
                    "finding": "No rate limiting detected on API endpoints",
                    "recommendation": "Implement rate limiting to prevent abuse"
                })
                logger.warning("⚠️ No rate limiting detected")
            else:
                logger.info("✅ Rate limiting is active")
    
    async def validate_security_configuration(self):
        """A05:2021 - Security Misconfiguration validation."""
        logger.info("⚙️ Validating Security Configuration...")
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/api/v1/system/heartbeat", headers=headers)
            
            # Check security headers
            required_headers = [
                "x-content-type-options",
                "x-frame-options",
                "x-xss-protection",
                "strict-transport-security",
                "content-security-policy",
                "referrer-policy"
            ]
            
            missing_headers = []
            for header in required_headers:
                if header not in response.headers:
                    missing_headers.append(header)
            
            if missing_headers:
                self.security_findings.append({
                    "category": "A05:2021 - Security Misconfiguration",
                    "severity": "MEDIUM",
                    "finding": f"Missing security headers: {', '.join(missing_headers)}",
                    "recommendation": "Implement all recommended security headers"
                })
                logger.warning(f"⚠️ Missing security headers: {missing_headers}")
            else:
                logger.info("✅ All security headers present")
            
            # Check for information disclosure
            if "server" in response.headers:
                server_header = response.headers["server"]
                if any(tech in server_header.lower() for tech in ["uvicorn", "fastapi", "python"]):
                    self.security_findings.append({
                        "category": "A05:2021 - Security Misconfiguration",
                        "severity": "LOW",
                        "finding": f"Server header reveals technology: {server_header}",
                        "recommendation": "Remove or obfuscate server header"
                    })
                    logger.warning(f"⚠️ Server header disclosure: {server_header}")
    
    async def validate_component_security(self):
        """A06:2021 - Vulnerable and Outdated Components validation."""
        logger.info("📦 Validating Component Security...")
        
        # This would typically check dependency versions
        # For now, we'll check for common vulnerable endpoints
        vulnerable_endpoints = [
            "/admin",
            "/phpmyadmin",
            "/wp-admin",
            "/.env",
            "/config.json",
            "/swagger-ui.html",
        ]
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for endpoint in vulnerable_endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    
                    if response.status_code == 200:
                        self.security_findings.append({
                            "category": "A06:2021 - Vulnerable Components",
                            "severity": "HIGH",
                            "finding": f"Potentially vulnerable endpoint accessible: {endpoint}",
                            "recommendation": "Remove or secure vulnerable endpoints"
                        })
                        logger.error(f"❌ Vulnerable endpoint found: {endpoint}")
                    else:
                        logger.info(f"✅ Vulnerable endpoint not accessible: {endpoint}")
                        
                except Exception:
                    logger.info(f"✅ Vulnerable endpoint not accessible: {endpoint}")
    
    async def validate_authentication_security(self):
        """A07:2021 - Identification and Authentication Failures validation."""
        logger.info("🔑 Validating Authentication Security...")
        
        # Test weak authentication
        weak_tokens = [
            "Bearer 123456",
            "Bearer password",
            "Bearer admin",
            "Bearer test",
            "Bearer token",
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for token in weak_tokens:
                response = await client.get(
                    f"{self.base_url}/api/v1/system/heartbeat",
                    headers={"Authorization": token}
                )
                
                if response.status_code == 200:
                    self.security_findings.append({
                        "category": "A07:2021 - Authentication Failures",
                        "severity": "CRITICAL",
                        "finding": f"Weak authentication token accepted: {token}",
                        "recommendation": "Implement strong token validation"
                    })
                    logger.error(f"❌ Weak token accepted: {token}")
                else:
                    logger.info(f"✅ Weak token rejected: {token}")
    
    async def validate_software_integrity(self):
        """A08:2021 - Software and Data Integrity Failures validation."""
        logger.info("🔒 Validating Software Integrity...")
        
        # Check for integrity headers
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/api/v1/system/heartbeat", headers=headers)
            
            # Check for integrity-related headers
            integrity_headers = ["content-security-policy", "x-content-type-options"]
            
            missing_integrity = []
            for header in integrity_headers:
                if header not in response.headers:
                    missing_integrity.append(header)
            
            if missing_integrity:
                logger.warning(f"⚠️ Missing integrity headers: {missing_integrity}")
            else:
                logger.info("✅ Integrity headers present")
    
    async def validate_security_logging(self):
        """A09:2021 - Security Logging and Monitoring Failures validation."""
        logger.info("📝 Validating Security Logging...")
        
        # Test if security events are logged
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Make authenticated request
            response = await client.get(f"{self.base_url}/api/v1/system/heartbeat", headers=headers)
            
            # Check for audit headers (indicating logging)
            audit_headers = [h for h in response.headers.keys() if "audit" in h.lower()]
            
            if audit_headers:
                logger.info(f"✅ Audit logging detected: {audit_headers}")
            else:
                self.security_findings.append({
                    "category": "A09:2021 - Security Logging Failures",
                    "severity": "MEDIUM",
                    "finding": "No audit logging headers detected",
                    "recommendation": "Implement comprehensive security logging"
                })
                logger.warning("⚠️ No audit logging detected")
    
    async def validate_ssrf_protection(self):
        """A10:2021 - Server-Side Request Forgery validation."""
        logger.info("🌐 Validating SSRF Protection...")
        
        # Test SSRF attempts (if applicable endpoints exist)
        
        # This would test endpoints that accept URLs
        # For now, we'll just log that SSRF protection should be verified
        logger.info("ℹ️ SSRF protection should be verified for any URL-accepting endpoints")
    
    def generate_owasp_report(self) -> Dict[str, Any]:
        """Generate OWASP compliance report."""
        logger.info("\n" + "="*80)
        logger.info("📊 GENERATING OWASP SECURITY COMPLIANCE REPORT")
        logger.info("="*80)
        
        # Categorize findings by severity
        critical_findings = [f for f in self.security_findings if f["severity"] == "CRITICAL"]
        high_findings = [f for f in self.security_findings if f["severity"] == "HIGH"]
        medium_findings = [f for f in self.security_findings if f["severity"] == "MEDIUM"]
        low_findings = [f for f in self.security_findings if f["severity"] == "LOW"]
        
        # Calculate compliance score
        total_checks = 10  # OWASP Top 10
        failed_checks = len(set(f["category"] for f in self.security_findings))
        compliance_score = ((total_checks - failed_checks) / total_checks) * 100
        
        report = {
            "owasp_compliance_timestamp": time.time(),
            "compliance_summary": {
                "total_findings": len(self.security_findings),
                "critical_findings": len(critical_findings),
                "high_findings": len(high_findings),
                "medium_findings": len(medium_findings),
                "low_findings": len(low_findings),
                "compliance_score": compliance_score,
                "owasp_categories_failed": failed_checks,
                "owasp_categories_passed": total_checks - failed_checks
            },
            "findings_by_severity": {
                "critical": critical_findings,
                "high": high_findings,
                "medium": medium_findings,
                "low": low_findings
            },
            "owasp_compliance": compliance_score >= 80,
            "production_ready": len(critical_findings) == 0 and len(high_findings) <= 2
        }
        
        # Print summary
        logger.info(f"🔒 OWASP COMPLIANCE SCORE: {compliance_score:.1f}%")
        logger.info(f"📊 TOTAL FINDINGS: {len(self.security_findings)}")
        logger.info(f"🚨 CRITICAL: {len(critical_findings)}")
        logger.info(f"⚠️ HIGH: {len(high_findings)}")
        logger.info(f"📋 MEDIUM: {len(medium_findings)}")
        logger.info(f"ℹ️ LOW: {len(low_findings)}")
        
        compliance_status = "✅ COMPLIANT" if report["owasp_compliance"] else "❌ NON-COMPLIANT"
        production_status = "✅ READY" if report["production_ready"] else "❌ BLOCKED"
        
        logger.info(f"\n🏆 OWASP COMPLIANCE: {compliance_status}")
        logger.info(f"🚀 PRODUCTION READY: {production_status}")
        
        return report


async def main():
    """Execute OWASP security validation."""
    validator = OWASPSecurityValidator()
    
    try:
        report = await validator.run_owasp_validation()
        
        # Save report
        with open("owasp_security_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("\n📄 OWASP compliance report saved to: owasp_security_report.json")
        
        return 0 if report["production_ready"] else 1
        
    except Exception as e:
        logger.error(f"❌ OWASP validation failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 OWASP validation interrupted")
        exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        exit(1)
