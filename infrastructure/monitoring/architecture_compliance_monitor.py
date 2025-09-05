#!/usr/bin/env python3
"""
Architecture Compliance Monitor für Keiko Platform-SDK
Kontinuierliche Überwachung der Architektur-Compliance mit Metriken und Alerting
"""

import os
import sys
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import subprocess
import requests
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import schedule

class ComplianceStatus(Enum):
    """Compliance-Status Levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComplianceMetric:
    """Repräsentiert eine Compliance-Metrik"""
    name: str
    value: float
    status: ComplianceStatus
    threshold: float
    description: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

@dataclass
class ComplianceAlert:
    """Repräsentiert einen Compliance-Alert"""
    alert_id: str
    severity: str
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False

class ArchitectureComplianceMonitor:
    """Hauptklasse für kontinuierliche Architektur-Compliance Überwachung"""
    
    def __init__(self, config_path: str = "monitoring/compliance_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Prometheus Metriken
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Compliance-Tracking
        self.current_metrics: Dict[str, ComplianceMetric] = {}
        self.active_alerts: List[ComplianceAlert] = []
        self.compliance_history: List[Dict[str, Any]] = []
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Lädt Monitoring-Konfiguration"""
        default_config = {
            "monitoring": {
                "interval_seconds": 300,  # 5 Minuten
                "prometheus_port": 8090,
                "alert_webhook_url": None,
                "slack_webhook_url": None
            },
            "thresholds": {
                "cross_imports": 0,  # Keine Cross-Imports erlaubt
                "api_contract_violations": 0,
                "deployment_independence_score": 95.0,
                "architecture_compliance_score": 90.0
            },
            "checks": {
                "cross_import_analysis": True,
                "api_contract_validation": True,
                "deployment_independence": True,
                "docker_isolation": False,  # Optional
                "dependency_graph_analysis": True
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge mit Default-Config
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def _setup_prometheus_metrics(self):
        """Initialisiert Prometheus-Metriken"""
        
        # Compliance Score Gauges
        self.compliance_score_gauge = Gauge(
            'keiko_architecture_compliance_score',
            'Overall architecture compliance score (0-100)',
            registry=self.registry
        )
        
        self.deployment_independence_gauge = Gauge(
            'keiko_deployment_independence_score',
            'Deployment independence score (0-100)',
            registry=self.registry
        )
        
        # Violation Counters
        self.cross_imports_counter = Counter(
            'keiko_cross_imports_total',
            'Total number of cross-imports detected',
            registry=self.registry
        )
        
        self.api_violations_counter = Counter(
            'keiko_api_contract_violations_total',
            'Total number of API contract violations',
            registry=self.registry
        )
        
        # Check Duration Histograms
        self.check_duration_histogram = Histogram(
            'keiko_compliance_check_duration_seconds',
            'Duration of compliance checks',
            ['check_type'],
            registry=self.registry
        )
        
        # Alert Gauges
        self.active_alerts_gauge = Gauge(
            'keiko_active_compliance_alerts',
            'Number of active compliance alerts',
            ['severity'],
            registry=self.registry
        )
    
    async def start_monitoring(self):
        """Startet kontinuierliche Überwachung"""
        self.logger.info("🔍 Starting Architecture Compliance Monitor...")
        
        # Starte Prometheus HTTP Server
        prometheus_port = self.config["monitoring"]["prometheus_port"]
        start_http_server(prometheus_port, registry=self.registry)
        self.logger.info(f"📊 Prometheus metrics server started on port {prometheus_port}")
        
        # Schedule regelmäßige Checks
        interval = self.config["monitoring"]["interval_seconds"]
        schedule.every(interval).seconds.do(self._run_compliance_checks)
        
        # Führe initialen Check durch
        await self._run_compliance_checks()
        
        # Hauptschleife
        while True:
            schedule.run_pending()
            await asyncio.sleep(10)  # Check alle 10 Sekunden auf scheduled jobs
    
    async def _run_compliance_checks(self):
        """Führt alle konfigurierten Compliance-Checks durch"""
        self.logger.info("🔍 Running compliance checks...")
        
        start_time = time.time()
        
        try:
            # 1. Cross-Import Analysis
            if self.config["checks"]["cross_import_analysis"]:
                await self._check_cross_imports()
            
            # 2. API Contract Validation
            if self.config["checks"]["api_contract_validation"]:
                await self._check_api_contracts()
            
            # 3. Deployment Independence
            if self.config["checks"]["deployment_independence"]:
                await self._check_deployment_independence()
            
            # 4. Dependency Graph Analysis
            if self.config["checks"]["dependency_graph_analysis"]:
                await self._check_dependency_graph()
            
            # 5. Berechne Overall Compliance Score
            await self._calculate_compliance_scores()
            
            # 6. Prüfe Alerts
            await self._process_alerts()
            
            # 7. Speichere History
            await self._save_compliance_history()
            
        except Exception as e:
            self.logger.error(f"Error during compliance checks: {e}")
        
        duration = time.time() - start_time
        self.logger.info(f"✅ Compliance checks completed in {duration:.2f}s")
    
    async def _check_cross_imports(self):
        """Prüft auf Cross-Imports zwischen Platform und SDK"""
        with self.check_duration_histogram.labels(check_type='cross_imports').time():
            try:
                # Führe Architecture Compliance Analyzer aus
                result = subprocess.run([
                    sys.executable, "scripts/architecture_compliance_analyzer.py"
                ], capture_output=True, text=True, timeout=60)
                
                # Parse Ergebnisse
                violations = 0
                if result.returncode != 0:
                    violations = 1  # Mindestens eine Verletzung
                
                # Versuche JSON-Report zu parsen
                try:
                    with open("architecture_compliance_report.json", 'r') as f:
                        report = json.load(f)
                    violations = report.get("summary", {}).get("total_violations", 0)
                except:
                    pass
                
                # Update Metriken
                self.cross_imports_counter._value._value = violations
                
                # Erstelle Compliance-Metrik
                threshold = self.config["thresholds"]["cross_imports"]
                status = ComplianceStatus.COMPLIANT if violations <= threshold else ComplianceStatus.VIOLATION
                
                metric = ComplianceMetric(
                    name="cross_imports",
                    value=violations,
                    status=status,
                    threshold=threshold,
                    description=f"Cross-imports detected between Platform and SDK",
                    timestamp=datetime.now(),
                    details={"analyzer_output": result.stdout[:500]}
                )
                
                self.current_metrics["cross_imports"] = metric
                
                if status == ComplianceStatus.VIOLATION:
                    await self._create_alert(
                        severity="critical",
                        title="Cross-Import Violation Detected",
                        description=f"Found {violations} cross-imports between Platform and SDK",
                        metric_name="cross_imports",
                        current_value=violations,
                        threshold=threshold
                    )
                
            except Exception as e:
                self.logger.error(f"Cross-import check failed: {e}")
                self.current_metrics["cross_imports"] = ComplianceMetric(
                    name="cross_imports",
                    value=-1,
                    status=ComplianceStatus.UNKNOWN,
                    threshold=0,
                    description="Cross-import check failed",
                    timestamp=datetime.now(),
                    details={"error": str(e)}
                )
    
    async def _check_api_contracts(self):
        """Prüft API-Contract Compliance"""
        with self.check_duration_histogram.labels(check_type='api_contracts').time():
            try:
                # Führe API Contract Validator aus
                result = subprocess.run([
                    sys.executable, "scripts/api_contract_validator.py"
                ], capture_output=True, text=True, timeout=120)
                
                violations = 0
                if result.returncode != 0:
                    violations = 1
                
                # Parse Violations aus Output
                if "VALIDATION FAILED" in result.stdout:
                    # Extrahiere Anzahl der Violations
                    import re
                    match = re.search(r"Failed: (\d+)", result.stdout)
                    if match:
                        violations = int(match.group(1))
                
                # Update Metriken
                self.api_violations_counter._value._value = violations
                
                threshold = self.config["thresholds"]["api_contract_violations"]
                status = ComplianceStatus.COMPLIANT if violations <= threshold else ComplianceStatus.WARNING
                
                metric = ComplianceMetric(
                    name="api_contract_violations",
                    value=violations,
                    status=status,
                    threshold=threshold,
                    description="API contract validation violations",
                    timestamp=datetime.now(),
                    details={"validator_output": result.stdout[:500]}
                )
                
                self.current_metrics["api_contract_violations"] = metric
                
                if status != ComplianceStatus.COMPLIANT:
                    await self._create_alert(
                        severity="warning",
                        title="API Contract Violations",
                        description=f"Found {violations} API contract violations",
                        metric_name="api_contract_violations",
                        current_value=violations,
                        threshold=threshold
                    )
                
            except Exception as e:
                self.logger.error(f"API contract check failed: {e}")
    
    async def _check_deployment_independence(self):
        """Prüft Deployment-Unabhängigkeit"""
        with self.check_duration_histogram.labels(check_type='deployment_independence').time():
            try:
                # Führe Deployment Independence Tests aus
                result = subprocess.run([
                    sys.executable, "tests/deployment_independence_tests.py"
                ], capture_output=True, text=True, timeout=300)
                
                # Parse Test-Ergebnisse
                passed_tests = 0
                total_tests = 0
                
                import re
                passed_match = re.search(r"Passed: (\d+)", result.stdout)
                total_match = re.search(r"Total Tests: (\d+)", result.stdout)
                
                if passed_match and total_match:
                    passed_tests = int(passed_match.group(1))
                    total_tests = int(total_match.group(1))
                
                # Berechne Score
                score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                
                # Update Metriken
                self.deployment_independence_gauge.set(score)
                
                threshold = self.config["thresholds"]["deployment_independence_score"]
                status = ComplianceStatus.COMPLIANT if score >= threshold else ComplianceStatus.WARNING
                
                metric = ComplianceMetric(
                    name="deployment_independence_score",
                    value=score,
                    status=status,
                    threshold=threshold,
                    description=f"Deployment independence test score ({passed_tests}/{total_tests})",
                    timestamp=datetime.now(),
                    details={"passed_tests": passed_tests, "total_tests": total_tests}
                )
                
                self.current_metrics["deployment_independence_score"] = metric
                
                if status != ComplianceStatus.COMPLIANT:
                    await self._create_alert(
                        severity="warning",
                        title="Deployment Independence Issues",
                        description=f"Deployment independence score: {score:.1f}% (threshold: {threshold}%)",
                        metric_name="deployment_independence_score",
                        current_value=score,
                        threshold=threshold
                    )
                
            except Exception as e:
                self.logger.error(f"Deployment independence check failed: {e}")
    
    async def _check_dependency_graph(self):
        """Prüft Dependency-Graph auf unerlaubte Dependencies"""
        with self.check_duration_histogram.labels(check_type='dependency_graph').time():
            try:
                # Führe Dependency-Analyse durch
                result = subprocess.run([
                    "pipdeptree", "--packages", "backend", "--json"
                ], capture_output=True, text=True, timeout=60, cwd="backend")
                
                if result.returncode == 0:
                    dependencies = json.loads(result.stdout)
                    
                    # Prüfe auf SDK-Dependencies
                    sdk_deps_found = False
                    for dep in dependencies:
                        package_name = dep.get("package", {}).get("package_name", "").lower()
                        if "kei" in package_name and "sdk" in package_name:
                            sdk_deps_found = True
                            break
                    
                    metric = ComplianceMetric(
                        name="dependency_graph_compliance",
                        value=0 if not sdk_deps_found else 1,
                        status=ComplianceStatus.COMPLIANT if not sdk_deps_found else ComplianceStatus.VIOLATION,
                        threshold=0,
                        description="Platform dependency graph analysis",
                        timestamp=datetime.now(),
                        details={"total_dependencies": len(dependencies), "sdk_deps_found": sdk_deps_found}
                    )
                    
                    self.current_metrics["dependency_graph_compliance"] = metric
                    
                    if sdk_deps_found:
                        await self._create_alert(
                            severity="critical",
                            title="SDK Dependencies in Platform",
                            description="Platform has dependencies on SDK packages",
                            metric_name="dependency_graph_compliance",
                            current_value=1,
                            threshold=0
                        )
                
            except Exception as e:
                self.logger.error(f"Dependency graph check failed: {e}")
    
    async def _calculate_compliance_scores(self):
        """Berechnet Overall Compliance Scores"""
        
        # Sammle alle Metriken
        metrics = list(self.current_metrics.values())
        
        if not metrics:
            return
        
        # Berechne gewichteten Compliance Score
        weights = {
            "cross_imports": 0.4,  # Höchste Gewichtung
            "api_contract_violations": 0.2,
            "deployment_independence_score": 0.3,
            "dependency_graph_compliance": 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for metric in metrics:
            weight = weights.get(metric.name, 0.1)
            
            if metric.status == ComplianceStatus.COMPLIANT:
                score = 100
            elif metric.status == ComplianceStatus.WARNING:
                score = 70
            elif metric.status == ComplianceStatus.VIOLATION:
                score = 30
            elif metric.status == ComplianceStatus.CRITICAL:
                score = 0
            else:
                continue  # Skip unknown status
            
            total_score += score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # Update Prometheus Metrik
        self.compliance_score_gauge.set(overall_score)
        
        # Erstelle Overall Compliance Metrik
        threshold = self.config["thresholds"]["architecture_compliance_score"]
        status = ComplianceStatus.COMPLIANT if overall_score >= threshold else ComplianceStatus.WARNING
        
        overall_metric = ComplianceMetric(
            name="overall_architecture_compliance",
            value=overall_score,
            status=status,
            threshold=threshold,
            description="Overall architecture compliance score",
            timestamp=datetime.now(),
            details={"component_scores": {m.name: m.value for m in metrics}}
        )
        
        self.current_metrics["overall_architecture_compliance"] = overall_metric
        
        self.logger.info(f"📊 Overall Compliance Score: {overall_score:.1f}%")
    
    async def _create_alert(self, severity: str, title: str, description: str, 
                          metric_name: str, current_value: float, threshold: float):
        """Erstellt einen neuen Alert"""
        
        alert_id = f"{metric_name}_{int(time.time())}"
        
        alert = ComplianceAlert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.active_alerts.append(alert)
        
        # Update Prometheus Alert Metrik
        self.active_alerts_gauge.labels(severity=severity).inc()
        
        # Sende Alert-Benachrichtigung
        await self._send_alert_notification(alert)
        
        self.logger.warning(f"🚨 Alert created: {title}")
    
    async def _send_alert_notification(self, alert: ComplianceAlert):
        """Sendet Alert-Benachrichtigungen"""
        
        # Webhook-Benachrichtigung
        webhook_url = self.config["monitoring"].get("alert_webhook_url")
        if webhook_url:
            try:
                payload = {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "title": alert.title,
                    "description": alert.description,
                    "metric": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                }
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
            except Exception as e:
                self.logger.error(f"Failed to send webhook alert: {e}")
        
        # Slack-Benachrichtigung
        slack_url = self.config["monitoring"].get("slack_webhook_url")
        if slack_url:
            try:
                color = {"critical": "danger", "warning": "warning", "info": "good"}.get(alert.severity, "warning")
                
                payload = {
                    "attachments": [{
                        "color": color,
                        "title": f"🚨 Architecture Compliance Alert: {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Severity", "value": alert.severity.upper(), "short": True}
                        ],
                        "ts": int(alert.timestamp.timestamp())
                    }]
                }
                
                response = requests.post(slack_url, json=payload, timeout=10)
                response.raise_for_status()
                
            except Exception as e:
                self.logger.error(f"Failed to send Slack alert: {e}")
    
    async def _process_alerts(self):
        """Verarbeitet und bereinigt Alerts"""
        
        # Entferne resolved Alerts älter als 24h
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if not (alert.resolved and alert.timestamp < cutoff_time)
        ]
        
        # Update Prometheus Alert Metriken
        for severity in ["critical", "warning", "info"]:
            count = len([a for a in self.active_alerts if a.severity == severity and not a.resolved])
            self.active_alerts_gauge.labels(severity=severity).set(count)
    
    async def _save_compliance_history(self):
        """Speichert Compliance-History für Trend-Analyse"""
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {name: asdict(metric) for name, metric in self.current_metrics.items()},
            "active_alerts": len([a for a in self.active_alerts if not a.resolved])
        }
        
        self.compliance_history.append(history_entry)
        
        # Behalte nur letzte 1000 Einträge
        if len(self.compliance_history) > 1000:
            self.compliance_history = self.compliance_history[-1000:]
        
        # Speichere in Datei
        try:
            with open("monitoring/compliance_history.json", "w") as f:
                json.dump(self.compliance_history[-100:], f, indent=2)  # Nur letzte 100 Einträge
        except Exception as e:
            self.logger.error(f"Failed to save compliance history: {e}")

def main():
    """Hauptfunktion für CLI-Ausführung"""
    
    monitor = ArchitectureComplianceMonitor()
    
    print("🔍 Keiko Architecture Compliance Monitor")
    print("=" * 50)
    print(f"📊 Prometheus metrics: http://localhost:{monitor.config['monitoring']['prometheus_port']}/metrics")
    print(f"⏱️ Check interval: {monitor.config['monitoring']['interval_seconds']}s")
    
    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
