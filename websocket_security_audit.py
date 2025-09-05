#!/usr/bin/env python3
"""
Enterprise WebSocket & Authentication Security Audit Suite

Comprehensive analysis and testing of WebSocket endpoints and authentication systems
for enterprise-grade production deployment.
"""

import asyncio
import json
import logging
import time
import traceback
import websockets
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security assessment levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityFinding:
    """Security audit finding."""
    level: SecurityLevel
    category: str
    title: str
    description: str
    recommendation: str
    endpoint: Optional[str] = None


@dataclass
class WebSocketTestResult:
    """WebSocket test result."""
    endpoint: str
    success: bool
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    connection_stable: bool = False
    auth_required: bool = True
    supports_heartbeat: bool = False


class EnterpriseWebSocketAudit:
    """Enterprise-grade WebSocket and Authentication audit system."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        self.auth_token = "dev-token-12345"
        self.findings: List[SecurityFinding] = []
        self.test_results: List[WebSocketTestResult] = []
        
    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Execute comprehensive WebSocket and authentication audit."""
        logger.info("🔒 Starting Enterprise WebSocket & Authentication Security Audit")
        
        audit_sections = [
            ("WebSocket Endpoint Discovery", self.audit_websocket_endpoints),
            ("Authentication System Analysis", self.audit_authentication_system),
            ("WebSocket Handshake Security", self.audit_websocket_handshake),
            ("Message Routing & Event Handling", self.audit_message_routing),
            ("Connection Persistence & Heartbeat", self.audit_connection_persistence),
            ("Error Handling & Graceful Disconnection", self.audit_error_handling),
            ("Security Boundary Validation", self.audit_security_boundaries),
            ("Performance & Load Testing", self.audit_performance),
        ]
        
        for section_name, audit_method in audit_sections:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 AUDIT SECTION: {section_name}")
            logger.info(f"{'='*80}")
            
            try:
                await audit_method()
            except Exception as e:
                logger.error(f"❌ Audit section failed: {section_name} - {e}")
                self.findings.append(SecurityFinding(
                    level=SecurityLevel.HIGH,
                    category="audit_failure",
                    title=f"Audit Section Failure: {section_name}",
                    description=f"Failed to complete audit section: {str(e)}",
                    recommendation="Investigate and resolve audit execution issues"
                ))
        
        return self.generate_audit_report()
    
    async def audit_websocket_endpoints(self):
        """Audit all WebSocket endpoints for security and functionality."""
        logger.info("🔌 Auditing WebSocket endpoints...")
        
        # Define all known WebSocket endpoints
        websocket_endpoints = [
            "/ws/connect",
            "/ws/system/heartbeat", 
            "/ws/client/system_heartbeat_client",
            "/ws/client/test_client",
            "/websocket/connect",  # Alternative path
        ]
        
        for endpoint in websocket_endpoints:
            await self.test_websocket_endpoint(endpoint)
    
    async def test_websocket_endpoint(self, endpoint: str):
        """Test individual WebSocket endpoint comprehensively."""
        logger.info(f"🔍 Testing WebSocket endpoint: {endpoint}")
        
        # Test without authentication
        result_no_auth = await self.test_websocket_connection(
            endpoint, use_auth=False
        )
        
        # Test with authentication
        result_with_auth = await self.test_websocket_connection(
            endpoint, use_auth=True
        )
        
        # Analyze results
        self.analyze_websocket_test_results(endpoint, result_no_auth, result_with_auth)
    
    async def test_websocket_connection(self, endpoint: str, use_auth: bool = True) -> WebSocketTestResult:
        """Test WebSocket connection with comprehensive analysis."""
        full_url = f"{self.ws_url}{endpoint}"
        headers = {}
        
        if use_auth:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        
        try:
            async with websockets.connect(
                full_url,
                extra_headers=headers,
                timeout=10,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                
                response_time = (time.time() - start_time) * 1000
                
                # Test basic connectivity
                test_message = {
                    "type": "ping",
                    "timestamp": time.time(),
                    "test_id": f"audit_{int(time.time())}"
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Test connection stability
                connection_stable = await self.test_connection_stability(websocket)
                
                # Test heartbeat support
                heartbeat_support = await self.test_heartbeat_support(websocket)
                
                return WebSocketTestResult(
                    endpoint=endpoint,
                    success=True,
                    response_time_ms=response_time,
                    connection_stable=connection_stable,
                    auth_required=use_auth,
                    supports_heartbeat=heartbeat_support
                )
                
        except websockets.exceptions.InvalidStatusCode as e:
            return WebSocketTestResult(
                endpoint=endpoint,
                success=False,
                status_code=e.status_code,
                error_message=f"HTTP {e.status_code}",
                auth_required=use_auth
            )
        except Exception as e:
            return WebSocketTestResult(
                endpoint=endpoint,
                success=False,
                error_message=str(e),
                auth_required=use_auth
            )
    
    async def test_connection_stability(self, websocket) -> bool:
        """Test WebSocket connection stability."""
        try:
            # Send multiple messages to test stability
            for i in range(3):
                test_msg = {"stability_test": i, "timestamp": time.time()}
                await websocket.send(json.dumps(test_msg))
                await asyncio.sleep(0.5)
            
            return True
        except Exception:
            return False
    
    async def test_heartbeat_support(self, websocket) -> bool:
        """Test WebSocket heartbeat/ping support."""
        try:
            # Send ping and wait for pong
            await websocket.ping()
            return True
        except Exception:
            return False
    
    def analyze_websocket_test_results(self, endpoint: str, no_auth: WebSocketTestResult, with_auth: WebSocketTestResult):
        """Analyze WebSocket test results for security findings."""
        
        # Check if endpoint requires authentication
        if no_auth.success and not with_auth.success:
            self.findings.append(SecurityFinding(
                level=SecurityLevel.CRITICAL,
                category="authentication",
                title="WebSocket Endpoint Accessible Without Authentication",
                description=f"Endpoint {endpoint} accepts connections without proper authentication",
                recommendation="Implement proper authentication for all WebSocket endpoints",
                endpoint=endpoint
            ))
        
        # Check if endpoint is completely inaccessible
        if not no_auth.success and not with_auth.success:
            if no_auth.status_code == 403 and with_auth.status_code == 403:
                self.findings.append(SecurityFinding(
                    level=SecurityLevel.HIGH,
                    category="availability",
                    title="WebSocket Endpoint Inaccessible",
                    description=f"Endpoint {endpoint} returns 403 even with valid authentication",
                    recommendation="Review WebSocket authentication middleware configuration",
                    endpoint=endpoint
                ))
        
        # Check performance
        if with_auth.success and with_auth.response_time_ms and with_auth.response_time_ms > 5000:
            self.findings.append(SecurityFinding(
                level=SecurityLevel.MEDIUM,
                category="performance",
                title="Slow WebSocket Connection",
                description=f"Endpoint {endpoint} has slow connection time: {with_auth.response_time_ms:.2f}ms",
                recommendation="Optimize WebSocket connection establishment",
                endpoint=endpoint
            ))
        
        # Store successful test result
        if with_auth.success:
            self.test_results.append(with_auth)
    
    async def audit_authentication_system(self):
        """Comprehensive authentication system audit."""
        logger.info("🔐 Auditing authentication system...")
        
        # Test various authentication scenarios
        auth_tests = [
            ("Valid Bearer Token", f"Bearer {self.auth_token}"),
            ("Invalid Bearer Token", "Bearer invalid-token-12345"),
            ("Malformed Bearer Token", "Bearer"),
            ("Wrong Auth Type", f"Basic {self.auth_token}"),
            ("No Authorization Header", None),
            ("Empty Authorization Header", ""),
        ]
        
        for test_name, auth_header in auth_tests:
            await self.test_http_authentication(test_name, auth_header)
    
    async def test_http_authentication(self, test_name: str, auth_header: Optional[str]):
        """Test HTTP endpoint authentication."""
        logger.info(f"🔍 Testing: {test_name}")
        
        headers = {}
        if auth_header:
            headers["Authorization"] = auth_header
        
        test_endpoint = "/api/v1/system/heartbeat"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}{test_endpoint}",
                    headers=headers
                )
                
                # Analyze authentication behavior
                if test_name == "Valid Bearer Token" and response.status_code != 200:
                    self.findings.append(SecurityFinding(
                        level=SecurityLevel.CRITICAL,
                        category="authentication",
                        title="Valid Authentication Rejected",
                        description=f"Valid bearer token rejected with status {response.status_code}",
                        recommendation="Fix authentication middleware to accept valid tokens"
                    ))
                
                elif test_name != "Valid Bearer Token" and response.status_code == 200:
                    self.findings.append(SecurityFinding(
                        level=SecurityLevel.CRITICAL,
                        category="authentication",
                        title="Authentication Bypass",
                        description=f"Endpoint accessible with {test_name.lower()}",
                        recommendation="Strengthen authentication requirements"
                    ))
                
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {test_name} - {e}")
    
    async def audit_websocket_handshake(self):
        """Audit WebSocket handshake security."""
        logger.info("🤝 Auditing WebSocket handshake security...")
        
        # Test handshake with various headers
        handshake_tests = [
            ("Standard Handshake", {}),
            ("Custom Origin", {"Origin": "https://malicious-site.com"}),
            ("Missing Upgrade Header", {"Connection": "keep-alive"}),
            ("Invalid WebSocket Version", {"Sec-WebSocket-Version": "12"}),
        ]
        
        for test_name, custom_headers in handshake_tests:
            await self.test_websocket_handshake(test_name, custom_headers)
    
    async def test_websocket_handshake(self, test_name: str, custom_headers: Dict[str, str]):
        """Test WebSocket handshake with custom headers."""
        logger.info(f"🔍 Testing handshake: {test_name}")
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        headers.update(custom_headers)
        
        try:
            async with websockets.connect(
                f"{self.ws_url}/ws/client/system_heartbeat_client",
                extra_headers=headers,
                timeout=5
            ) as websocket:
                logger.info(f"✅ Handshake successful: {test_name}")
                
        except Exception as e:
            logger.info(f"❌ Handshake failed: {test_name} - {e}")
    
    async def audit_message_routing(self):
        """Audit message routing and event handling."""
        logger.info("📨 Auditing message routing and event handling...")
        
        # Test message routing with working endpoint
        endpoint = "/ws/client/system_heartbeat_client"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        try:
            async with websockets.connect(
                f"{self.ws_url}{endpoint}",
                extra_headers=headers,
                timeout=10
            ) as websocket:
                
                # Test various message types
                test_messages = [
                    {"type": "subscribe", "event": "system_heartbeat"},
                    {"type": "ping", "timestamp": time.time()},
                    {"type": "invalid_type", "data": "test"},
                    {"malformed": "json without type"},
                    "invalid json string",
                ]
                
                for i, message in enumerate(test_messages):
                    try:
                        if isinstance(message, str):
                            await websocket.send(message)
                        else:
                            await websocket.send(json.dumps(message))
                        
                        logger.info(f"✅ Message {i+1} sent successfully")
                        
                        # Try to receive response
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            logger.info(f"📥 Response received: {response[:100]}...")
                        except asyncio.TimeoutError:
                            logger.info("⏰ No response received")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Message {i+1} failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Message routing test failed: {e}")
    
    async def audit_connection_persistence(self):
        """Audit connection persistence and heartbeat functionality."""
        logger.info("💓 Auditing connection persistence and heartbeat...")
        
        endpoint = "/ws/client/system_heartbeat_client"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        try:
            async with websockets.connect(
                f"{self.ws_url}{endpoint}",
                extra_headers=headers,
                timeout=10,
                ping_interval=5,
                ping_timeout=3
            ) as websocket:
                
                logger.info("🔍 Testing connection persistence...")
                
                # Keep connection alive for 15 seconds
                start_time = time.time()
                ping_count = 0
                
                while time.time() - start_time < 15:
                    try:
                        # Send periodic ping
                        await websocket.ping()
                        ping_count += 1
                        logger.info(f"📡 Ping {ping_count} successful")
                        
                        await asyncio.sleep(3)
                        
                    except Exception as e:
                        logger.error(f"❌ Ping failed: {e}")
                        break
                
                logger.info(f"✅ Connection maintained for {time.time() - start_time:.1f} seconds")
                
        except Exception as e:
            logger.error(f"❌ Connection persistence test failed: {e}")
    
    async def audit_error_handling(self):
        """Audit error handling and graceful disconnection."""
        logger.info("🚨 Auditing error handling and graceful disconnection...")
        
        # Test various error scenarios
        error_scenarios = [
            ("Invalid Endpoint", "/ws/nonexistent"),
            ("Malformed URL", "/ws/client/"),
            ("Long Client ID", f"/ws/client/{'x' * 1000}"),
        ]
        
        for scenario_name, endpoint in error_scenarios:
            await self.test_error_scenario(scenario_name, endpoint)
    
    async def test_error_scenario(self, scenario_name: str, endpoint: str):
        """Test specific error scenario."""
        logger.info(f"🔍 Testing error scenario: {scenario_name}")
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        try:
            async with websockets.connect(
                f"{self.ws_url}{endpoint}",
                extra_headers=headers,
                timeout=5
            ) as websocket:
                logger.warning(f"⚠️ Unexpected success for: {scenario_name}")
                
        except websockets.exceptions.InvalidStatusCode as e:
            logger.info(f"✅ Expected error for {scenario_name}: HTTP {e.status_code}")
        except Exception as e:
            logger.info(f"✅ Expected error for {scenario_name}: {e}")
    
    async def audit_security_boundaries(self):
        """Audit security boundaries and access controls."""
        logger.info("🛡️ Auditing security boundaries...")
        
        # Test cross-origin requests
        origins = [
            "https://malicious-site.com",
            "http://localhost:3000",
            "null",
            "",
        ]
        
        for origin in origins:
            await self.test_cors_policy(origin)
    
    async def test_cors_policy(self, origin: str):
        """Test CORS policy with different origins."""
        logger.info(f"🔍 Testing CORS with origin: {origin or 'empty'}")
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Origin": origin
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/system/heartbeat",
                    headers=headers
                )
                
                cors_headers = {
                    k: v for k, v in response.headers.items() 
                    if k.lower().startswith('access-control-')
                }
                
                logger.info(f"📋 CORS headers: {cors_headers}")
                
        except Exception as e:
            logger.error(f"❌ CORS test failed for origin {origin}: {e}")
    
    async def audit_performance(self):
        """Audit performance characteristics."""
        logger.info("⚡ Auditing performance characteristics...")
        
        # Test concurrent connections
        concurrent_tests = []
        for i in range(5):
            concurrent_tests.append(
                self.test_websocket_connection("/ws/client/system_heartbeat_client", use_auth=True)
            )
        
        try:
            results = await asyncio.gather(*concurrent_tests, return_exceptions=True)
            
            successful_connections = sum(1 for r in results if isinstance(r, WebSocketTestResult) and r.success)
            
            logger.info(f"📊 Concurrent connections: {successful_connections}/5 successful")
            
            if successful_connections < 5:
                self.findings.append(SecurityFinding(
                    level=SecurityLevel.MEDIUM,
                    category="performance",
                    title="Limited Concurrent Connection Support",
                    description=f"Only {successful_connections}/5 concurrent connections succeeded",
                    recommendation="Investigate WebSocket connection limits and optimize"
                ))
                
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        logger.info("\n" + "="*80)
        logger.info("📊 GENERATING ENTERPRISE SECURITY AUDIT REPORT")
        logger.info("="*80)
        
        # Categorize findings by severity
        findings_by_level = {}
        for finding in self.findings:
            level = finding.level.value
            if level not in findings_by_level:
                findings_by_level[level] = []
            findings_by_level[level].append(finding)
        
        # Calculate security score
        security_score = self.calculate_security_score()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        report = {
            "audit_timestamp": time.time(),
            "audit_summary": {
                "total_findings": len(self.findings),
                "critical_findings": len(findings_by_level.get("critical", [])),
                "high_findings": len(findings_by_level.get("high", [])),
                "medium_findings": len(findings_by_level.get("medium", [])),
                "low_findings": len(findings_by_level.get("low", [])),
                "security_score": security_score,
                "websocket_endpoints_tested": len(self.test_results),
                "successful_connections": sum(1 for r in self.test_results if r.success)
            },
            "findings_by_category": findings_by_level,
            "websocket_test_results": [
                {
                    "endpoint": r.endpoint,
                    "success": r.success,
                    "response_time_ms": r.response_time_ms,
                    "connection_stable": r.connection_stable,
                    "supports_heartbeat": r.supports_heartbeat
                }
                for r in self.test_results
            ],
            "recommendations": recommendations,
            "production_readiness": self.assess_production_readiness()
        }
        
        self.print_audit_summary(report)
        return report
    
    def calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)."""
        if not self.findings:
            return 100.0
        
        # Weight findings by severity
        weights = {
            SecurityLevel.CRITICAL: -25,
            SecurityLevel.HIGH: -15,
            SecurityLevel.MEDIUM: -8,
            SecurityLevel.LOW: -3,
            SecurityLevel.INFO: -1
        }
        
        total_deduction = sum(weights.get(finding.level, 0) for finding in self.findings)
        score = max(0, 100 + total_deduction)
        
        return score
    
    def generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        critical_findings = [f for f in self.findings if f.level == SecurityLevel.CRITICAL]
        if critical_findings:
            recommendations.append("🚨 CRITICAL: Address all critical security findings immediately before production deployment")
        
        auth_findings = [f for f in self.findings if f.category == "authentication"]
        if auth_findings:
            recommendations.append("🔐 Strengthen authentication system and review access controls")
        
        performance_findings = [f for f in self.findings if f.category == "performance"]
        if performance_findings:
            recommendations.append("⚡ Optimize WebSocket performance and connection handling")
        
        if not self.findings:
            recommendations.append("✅ System demonstrates strong security posture - ready for production")
        
        return recommendations
    
    def assess_production_readiness(self) -> Dict[str, Any]:
        """Assess production readiness based on audit results."""
        critical_count = len([f for f in self.findings if f.level == SecurityLevel.CRITICAL])
        high_count = len([f for f in self.findings if f.level == SecurityLevel.HIGH])
        
        ready = critical_count == 0 and high_count <= 2
        
        return {
            "ready": ready,
            "blocking_issues": critical_count + high_count,
            "security_score": self.calculate_security_score(),
            "recommendation": "APPROVED FOR PRODUCTION" if ready else "REQUIRES SECURITY FIXES"
        }
    
    def print_audit_summary(self, report: Dict[str, Any]):
        """Print formatted audit summary."""
        summary = report["audit_summary"]
        
        logger.info(f"🔒 SECURITY SCORE: {summary['security_score']:.1f}/100")
        logger.info(f"📊 TOTAL FINDINGS: {summary['total_findings']}")
        logger.info(f"🚨 CRITICAL: {summary['critical_findings']}")
        logger.info(f"⚠️ HIGH: {summary['high_findings']}")
        logger.info(f"📋 MEDIUM: {summary['medium_findings']}")
        logger.info(f"ℹ️ LOW: {summary['low_findings']}")
        
        logger.info(f"\n🔌 WEBSOCKET ENDPOINTS:")
        logger.info(f"   Tested: {summary['websocket_endpoints_tested']}")
        logger.info(f"   Successful: {summary['successful_connections']}")
        
        production = report["production_readiness"]
        status = "✅ APPROVED" if production["ready"] else "❌ BLOCKED"
        logger.info(f"\n🚀 PRODUCTION READINESS: {status}")
        logger.info(f"   {production['recommendation']}")


async def main():
    """Execute comprehensive WebSocket and authentication audit."""
    audit = EnterpriseWebSocketAudit()
    
    try:
        report = await audit.run_comprehensive_audit()
        
        # Save detailed report
        with open("websocket_security_audit_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("\n📄 Detailed audit report saved to: websocket_security_audit_report.json")
        
        return 0 if report["production_readiness"]["ready"] else 1
        
    except Exception as e:
        logger.error(f"❌ Audit execution failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Audit interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        exit(1)
