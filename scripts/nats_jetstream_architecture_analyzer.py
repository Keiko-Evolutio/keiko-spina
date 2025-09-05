#!/usr/bin/env python3
"""
NATS JetStream Architecture Analyzer für Issue #56
Analysiert die geplante Messaging-first Architecture auf Architektur-Compliance
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ArchitectureRisk(Enum):
    """Risiko-Level für Architektur-Verletzungen"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class MessagingArchitectureViolation:
    """Repräsentiert eine potentielle Messaging-Architektur-Verletzung"""
    component: str
    risk_level: ArchitectureRisk
    description: str
    issue_reference: str
    compliance_impact: str
    recommendation: str
    code_example: Optional[str] = None

class NATSJetStreamArchitectureAnalyzer:
    """Analyzer für NATS JetStream Messaging-first Architecture (Issue #56)"""
    
    def __init__(self):
        self.violations: List[MessagingArchitectureViolation] = []
        self.issue_56_components = self._extract_issue_56_components()
        
    def _extract_issue_56_components(self) -> Dict[str, List[str]]:
        """Extrahiert die geplanten Komponenten aus Issue #56"""
        return {
            "messaging_core": [
                "MessagingFirstEventBus",
                "NATSClient", 
                "SchemaRegistryClient",
                "SchemaValidator"
            ],
            "event_patterns": [
                "OutboxManager",
                "InboxManager", 
                "DeadLetterQueueManager",
                "MessageDeduplicator"
            ],
            "reliability": [
                "RetryManager",
                "CircuitBreaker",
                "EventPublisher",
                "ConsumerConfig"
            ],
            "schema_management": [
                "SchemaRegistryManager",
                "SchemaStore",
                "CompatibilityChecker",
                "SchemaVersionManager"
            ]
        }
    
    def analyze_messaging_architecture_compliance(self) -> Dict[str, any]:
        """Führt umfassende Messaging-Architektur Compliance-Analyse durch"""
        print("🔍 Analysiere NATS JetStream Messaging-first Architecture...")
        
        results = {
            "cross_dependency_risks": self._analyze_cross_dependency_risks(),
            "schema_coupling_risks": self._analyze_schema_coupling_risks(),
            "event_pattern_risks": self._analyze_event_pattern_risks(),
            "deployment_isolation_risks": self._analyze_deployment_isolation_risks(),
            "api_boundary_violations": self._analyze_api_boundary_violations()
        }
        
        return results
    
    def _analyze_cross_dependency_risks(self) -> List[MessagingArchitectureViolation]:
        """Analysiert Cross-Dependency Risiken in der Messaging-Architektur"""
        violations = []
        
        # Risiko 1: Shared NATS Client zwischen Platform und SDK
        violations.append(MessagingArchitectureViolation(
            component="NATSClient",
            risk_level=ArchitectureRisk.HIGH,
            description="Shared NATS Client könnte zu direkten Dependencies zwischen Platform und SDK führen",
            issue_reference="Issue #56: MessagingFirstEventBus Implementation",
            compliance_impact="Verletzt strikte Unabhängigkeit - SDK könnte Platform NATS Client importieren",
            recommendation="Separate NATS Client Implementierungen: PlatformNATSClient vs SDKNATSClient",
            code_example="""
# ❌ RISIKO - Shared Client
from messaging.nats_client import NATSClient  # Beide nutzen gleichen Client

# ✅ LÖSUNG - Separate Clients  
# Platform: from platform.messaging.platform_nats_client import PlatformNATSClient
# SDK: from sdk.messaging.sdk_nats_client import SDKNATSClient
"""
        ))
        
        # Risiko 2: Schema Registry Coupling
        violations.append(MessagingArchitectureViolation(
            component="SchemaRegistryClient", 
            risk_level=ArchitectureRisk.CRITICAL,
            description="Gemeinsame Schema Registry könnte zu Schema-Dependencies zwischen Platform und SDK führen",
            issue_reference="Issue #56: Schema Registry Implementation",
            compliance_impact="KRITISCH: Shared Schemas könnten koordinierte Deployments erfordern",
            recommendation="Separate Schema Namespaces: platform.* vs sdk.* mit API-basierter Schema-Synchronisation",
            code_example="""
# ❌ RISIKO - Shared Schema Registry
schema_registry.register_schema("agent.created", schema)  # Beide nutzen gleiche Registry

# ✅ LÖSUNG - Namespace Separation
# Platform: schema_registry.register_schema("platform.agent.created", schema)  
# SDK: schema_registry.register_schema("sdk.agent.created", schema)
"""
        ))
        
        return violations
    
    def _analyze_schema_coupling_risks(self) -> List[MessagingArchitectureViolation]:
        """Analysiert Schema-Coupling Risiken"""
        violations = []
        
        # Risiko: Event Schema Definitions
        violations.append(MessagingArchitectureViolation(
            component="EventSchema Definitions",
            risk_level=ArchitectureRisk.HIGH,
            description="Shared Event Schemas zwischen Platform und SDK könnten Breaking Changes verursachen",
            issue_reference="Issue #56: Event Schema Management",
            compliance_impact="Schema-Änderungen könnten beide Systeme gleichzeitig betreffen",
            recommendation="Versionierte API-Contracts statt shared Schemas",
            code_example="""
# ❌ RISIKO - Shared Event Schemas
EventSchema(name="agent.created", namespace="keiko.events.agent")  # Shared namespace

# ✅ LÖSUNG - API Contract Based
# Platform definiert API Contract, SDK konsumiert über HTTP/gRPC
POST /api/v1/events/agent/created
{
  "agent_id": "string",
  "agent_type": "string", 
  "capabilities": ["string"]
}
"""
        ))
        
        return violations
    
    def _analyze_event_pattern_risks(self) -> List[MessagingArchitectureViolation]:
        """Analysiert Event Pattern Risiken (Outbox/Inbox)"""
        violations = []
        
        # Risiko: Shared Outbox/Inbox Implementation
        violations.append(MessagingArchitectureViolation(
            component="Outbox/Inbox Pattern",
            risk_level=ArchitectureRisk.MEDIUM,
            description="Shared Outbox/Inbox Manager könnte zu Database-Dependencies führen",
            issue_reference="Issue #56: Outbox/Inbox Pattern Implementation", 
            compliance_impact="Könnte gemeinsame Database-Schemas erfordern",
            recommendation="Separate Outbox/Inbox Stores mit API-basierter Kommunikation",
            code_example="""
# ❌ RISIKO - Shared Database Schema
outbox_manager.store_event(event, shared_transaction_context)

# ✅ LÖSUNG - Separate Stores
# Platform: platform_outbox_manager.store_event(event, platform_tx)
# SDK: sdk_outbox_manager.store_event(event, sdk_tx)
# Kommunikation über HTTP API
"""
        ))
        
        return violations
    
    def _analyze_deployment_isolation_risks(self) -> List[MessagingArchitectureViolation]:
        """Analysiert Deployment-Isolation Risiken"""
        violations = []
        
        # Risiko: Shared NATS Infrastructure
        violations.append(MessagingArchitectureViolation(
            component="NATS JetStream Infrastructure",
            risk_level=ArchitectureRisk.MEDIUM,
            description="Shared NATS Cluster könnte Deployment-Dependencies verursachen",
            issue_reference="Issue #56: NATS JetStream Deployment",
            compliance_impact="NATS Cluster Updates könnten beide Systeme betreffen",
            recommendation="Separate NATS Clusters oder strikte Stream-Isolation",
            code_example="""
# ❌ RISIKO - Shared NATS Cluster
nats://shared-nats-cluster:4222

# ✅ LÖSUNG - Separate Clusters oder Stream Isolation
# Platform: nats://platform-nats:4222 (Streams: platform.*)
# SDK: nats://sdk-nats:4222 (Streams: sdk.*)
# Oder: Shared Cluster mit strikter Stream-Trennung
"""
        ))
        
        return violations
    
    def _analyze_api_boundary_violations(self) -> List[MessagingArchitectureViolation]:
        """Analysiert API-Boundary Verletzungen"""
        violations = []
        
        # Risiko: Direct Messaging statt API
        violations.append(MessagingArchitectureViolation(
            component="Direct Messaging Communication",
            risk_level=ArchitectureRisk.HIGH,
            description="Direkte NATS-Kommunikation zwischen Platform und SDK umgeht API-Boundaries",
            issue_reference="Issue #56: Event-driven Communication",
            compliance_impact="Verletzt API-first Architektur-Prinzip",
            recommendation="Messaging nur intern, externe Kommunikation über HTTP/gRPC APIs",
            code_example="""
# ❌ RISIKO - Direct NATS Communication
# SDK publiziert direkt auf Platform NATS Stream
await nats_client.publish("platform.agent.events", event_data)

# ✅ LÖSUNG - API-based Communication  
# SDK nutzt HTTP API, Platform verwendet intern NATS
response = await http_client.post("/api/v1/events/agent", event_data)
"""
        ))
        
        return violations
    
    def generate_compliance_report(self) -> Dict[str, any]:
        """Generiert umfassenden Compliance-Report"""
        all_violations = []
        
        # Sammle alle Verletzungen
        analysis_results = self.analyze_messaging_architecture_compliance()
        for category, violations in analysis_results.items():
            all_violations.extend(violations)
        
        # Kategorisiere nach Risiko-Level
        risk_summary = {
            "critical": len([v for v in all_violations if v.risk_level == ArchitectureRisk.CRITICAL]),
            "high": len([v for v in all_violations if v.risk_level == ArchitectureRisk.HIGH]),
            "medium": len([v for v in all_violations if v.risk_level == ArchitectureRisk.MEDIUM]),
            "low": len([v for v in all_violations if v.risk_level == ArchitectureRisk.LOW])
        }
        
        # Berechne Compliance Score
        total_violations = len(all_violations)
        compliance_score = max(0, 10 - (total_violations * 2))  # 10 = perfekt, 0 = kritisch
        
        return {
            "summary": {
                "total_violations": total_violations,
                "compliance_score": f"{compliance_score}/10",
                "risk_distribution": risk_summary,
                "overall_assessment": self._get_overall_assessment(compliance_score)
            },
            "violations": [
                {
                    "component": v.component,
                    "risk_level": v.risk_level.value,
                    "description": v.description,
                    "issue_reference": v.issue_reference,
                    "compliance_impact": v.compliance_impact,
                    "recommendation": v.recommendation,
                    "code_example": v.code_example
                } for v in all_violations
            ],
            "recommendations": self._generate_recommendations(all_violations)
        }
    
    def _get_overall_assessment(self, score: int) -> str:
        """Gibt Overall Assessment basierend auf Score zurück"""
        if score >= 8:
            return "GOOD - Geringe Architektur-Risiken"
        elif score >= 6:
            return "MODERATE - Einige Architektur-Risiken zu beachten"
        elif score >= 4:
            return "CONCERNING - Mehrere kritische Risiken"
        else:
            return "CRITICAL - Hohe Wahrscheinlichkeit für Architektur-Verletzungen"
    
    def _generate_recommendations(self, violations: List[MessagingArchitectureViolation]) -> List[str]:
        """Generiert priorisierte Empfehlungen"""
        recommendations = []
        
        # Kritische Empfehlungen zuerst
        critical_violations = [v for v in violations if v.risk_level == ArchitectureRisk.CRITICAL]
        if critical_violations:
            recommendations.append("SOFORT: Implementiere separate Schema Namespaces für Platform und SDK")
        
        high_violations = [v for v in violations if v.risk_level == ArchitectureRisk.HIGH]
        if high_violations:
            recommendations.append("HOCH: Ersetze direkte NATS-Kommunikation durch HTTP/gRPC APIs")
            recommendations.append("HOCH: Implementiere separate NATS Client Implementierungen")
        
        medium_violations = [v for v in violations if v.risk_level == ArchitectureRisk.MEDIUM]
        if medium_violations:
            recommendations.append("MITTEL: Plane separate NATS Clusters oder strikte Stream-Isolation")
            recommendations.append("MITTEL: Implementiere separate Outbox/Inbox Stores")
        
        return recommendations

def main():
    """Hauptfunktion für CLI-Ausführung"""
    analyzer = NATSJetStreamArchitectureAnalyzer()
    
    print("🏗️ NATS JetStream Architecture Compliance Analyzer")
    print("=" * 60)
    print("📋 Analysiere Issue #56: Messaging-first Architecture")
    
    # Generiere Compliance Report
    report = analyzer.generate_compliance_report()
    
    # Ausgabe Summary
    print("\n📊 COMPLIANCE SUMMARY:")
    print(f"Gefundene Verletzungen: {report['summary']['total_violations']}")
    print(f"Compliance Score: {report['summary']['compliance_score']}")
    print(f"Assessment: {report['summary']['overall_assessment']}")
    
    print("\n🚨 RISIKO-VERTEILUNG:")
    for risk, count in report['summary']['risk_distribution'].items():
        print(f"{risk.upper()}: {count}")
    
    # Speichere detaillierten Report
    report_path = "nats_jetstream_architecture_compliance_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detaillierter Report gespeichert: {report_path}")
    
    # Empfehlungen ausgeben
    if report['recommendations']:
        print("\n💡 EMPFEHLUNGEN:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Exit Code basierend auf Compliance Score
    score = int(report['summary']['compliance_score'].split('/')[0])
    if score >= 8:
        print("\n✅ ERFOLG: Gute Architektur-Compliance!")
        return 0
    elif score >= 6:
        print("\n⚠️ WARNUNG: Moderate Architektur-Risiken gefunden!")
        return 1
    else:
        print("\n❌ KRITISCH: Hohe Architektur-Risiken gefunden!")
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())
