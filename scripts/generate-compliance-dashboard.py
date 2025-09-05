#!/usr/bin/env python3
"""
Compliance Dashboard Generator

Generates interactive compliance dashboards with real-time data visualization
for regulatory compliance monitoring and reporting.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
import argparse


class ComplianceDashboardGenerator:
    """Generates comprehensive compliance dashboards."""
    
    def __init__(self, framework: str = "SOC2", period_days: int = 30):
        """Initialize the dashboard generator.
        
        Args:
            framework: Compliance framework (SOC2, ISO27001, GDPR, etc.)
            period_days: Reporting period in days
        """
        self.framework = framework
        self.period_days = period_days
        self.generated_at = datetime.utcnow()
        
    def collect_compliance_data(self) -> Dict[str, Any]:
        """Collect compliance data from various sources.
        
        Returns:
            Dictionary containing compliance metrics and data
        """
        # Mock compliance data - in real implementation, this would
        # fetch actual data from monitoring systems, databases, etc.
        
        compliance_data = {
            "framework": self.framework,
            "period_days": self.period_days,
            "generated_at": self.generated_at.isoformat(),
            "overall_score": 93,
            "trend": "improving",
            "violations": {
                "critical": 0,
                "high": 1,
                "medium": 2,
                "low": 3
            },
            "controls": {
                "implemented": 47,
                "partially_implemented": 3,
                "not_implemented": 0,
                "total": 50
            },
            "security_metrics": {
                "vulnerabilities_fixed": 15,
                "security_scans_passed": 98.2,
                "access_violations": 2,
                "incident_response_time": "2.3 hours"
            },
            "quality_metrics": {
                "code_coverage": 87.5,
                "test_success_rate": 98.1,
                "build_success_rate": 94.8,
                "deployment_success_rate": 99.2
            },
            "audit_metrics": {
                "audit_events_logged": 15847,
                "failed_access_attempts": 23,
                "configuration_changes": 156,
                "policy_violations": 4
            }
        }
        
        return compliance_data
    
    def generate_html_dashboard(self, data: Dict[str, Any]) -> str:
        """Generate HTML compliance dashboard.
        
        Args:
            data: Compliance data dictionary
            
        Returns:
            HTML content for the dashboard
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance Dashboard - {self.framework}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .metric-large {{
            font-size: 3em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #718096;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .status-excellent {{ color: #48bb78; }}
        .status-good {{ color: #38b2ac; }}
        .status-warning {{ color: #ed8936; }}
        .status-critical {{ color: #f56565; }}
        .progress-bar {{
            width: 100%;
            height: 12px;
            background: #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            margin: 15px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #48bb78, #38b2ac);
            transition: width 0.3s ease;
        }}
        .violations-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        .violation-item {{
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .violation-critical {{ background: #fed7d7; color: #742a2a; }}
        .violation-high {{ background: #faf089; color: #744210; }}
        .violation-medium {{ background: #bee3f8; color: #2a4365; }}
        .violation-low {{ background: #c6f6d5; color: #22543d; }}
        .chart-placeholder {{
            height: 200px;
            background: #f7fafc;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #718096;
            margin: 20px 0;
            border: 2px dashed #e2e8f0;
        }}
        .controls-summary {{
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
        }}
        .control-stat {{
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            background: #f7fafc;
            flex: 1;
            margin: 0 5px;
        }}
        .timestamp {{
            color: #718096;
            font-size: 0.9em;
            margin-top: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Compliance Dashboard</h1>
            <h2>{self.framework} Framework Monitoring</h2>
            <p><strong>Reporting Period:</strong> Last {self.period_days} days</p>
            <p><strong>Generated:</strong> {data['generated_at']}</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="card">
                <div class="metric-large status-excellent">{data['overall_score']}/100</div>
                <div class="metric-label">Overall Compliance Score</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {data['overall_score']}%;"></div>
                </div>
                <p class="status-excellent">Excellent compliance posture</p>
            </div>
            
            <div class="card">
                <div class="metric-large status-good">{data['controls']['implemented']}/{data['controls']['total']}</div>
                <div class="metric-label">Controls Implemented</div>
                <div class="controls-summary">
                    <div class="control-stat">
                        <div style="font-weight: bold; color: #48bb78;">{data['controls']['implemented']}</div>
                        <div style="font-size: 0.8em;">Implemented</div>
                    </div>
                    <div class="control-stat">
                        <div style="font-weight: bold; color: #ed8936;">{data['controls']['partially_implemented']}</div>
                        <div style="font-size: 0.8em;">Partial</div>
                    </div>
                    <div class="control-stat">
                        <div style="font-weight: bold; color: #f56565;">{data['controls']['not_implemented']}</div>
                        <div style="font-size: 0.8em;">Missing</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🚨 Compliance Violations</h3>
                <div class="violations-grid">
                    <div class="violation-item violation-critical">
                        <div style="font-weight: bold; font-size: 1.5em;">{data['violations']['critical']}</div>
                        <div>Critical</div>
                    </div>
                    <div class="violation-item violation-high">
                        <div style="font-weight: bold; font-size: 1.5em;">{data['violations']['high']}</div>
                        <div>High</div>
                    </div>
                    <div class="violation-item violation-medium">
                        <div style="font-weight: bold; font-size: 1.5em;">{data['violations']['medium']}</div>
                        <div>Medium</div>
                    </div>
                    <div class="violation-item violation-low">
                        <div style="font-weight: bold; font-size: 1.5em;">{data['violations']['low']}</div>
                        <div>Low</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🔒 Security Metrics</h3>
                <p><strong>Vulnerabilities Fixed:</strong> {data['security_metrics']['vulnerabilities_fixed']}</p>
                <p><strong>Security Scans Passed:</strong> {data['security_metrics']['security_scans_passed']}%</p>
                <p><strong>Access Violations:</strong> {data['security_metrics']['access_violations']}</p>
                <p><strong>Incident Response Time:</strong> {data['security_metrics']['incident_response_time']}</p>
            </div>
            
            <div class="card">
                <h3>📊 Quality Metrics</h3>
                <p><strong>Code Coverage:</strong> {data['quality_metrics']['code_coverage']}%</p>
                <p><strong>Test Success Rate:</strong> {data['quality_metrics']['test_success_rate']}%</p>
                <p><strong>Build Success Rate:</strong> {data['quality_metrics']['build_success_rate']}%</p>
                <p><strong>Deployment Success:</strong> {data['quality_metrics']['deployment_success_rate']}%</p>
            </div>
            
            <div class="card">
                <h3>📝 Audit Metrics</h3>
                <p><strong>Events Logged:</strong> {data['audit_metrics']['audit_events_logged']:,}</p>
                <p><strong>Failed Access Attempts:</strong> {data['audit_metrics']['failed_access_attempts']}</p>
                <p><strong>Configuration Changes:</strong> {data['audit_metrics']['configuration_changes']}</p>
                <p><strong>Policy Violations:</strong> {data['audit_metrics']['policy_violations']}</p>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Compliance Trends</h3>
            <div class="chart-placeholder">
                Compliance trend chart would be displayed here
                <br>
                (Integration with Chart.js or D3.js for interactive charts)
            </div>
        </div>
        
        <div class="card">
            <h3>🎯 Recommendations</h3>
            <ul>
                <li><strong>Address High Priority Violation:</strong> Review and remediate the 1 high-priority compliance gap</li>
                <li><strong>Complete Partial Controls:</strong> Finish implementation of 3 partially implemented controls</li>
                <li><strong>Enhance Security Monitoring:</strong> Reduce access violations through improved monitoring</li>
                <li><strong>Maintain Excellence:</strong> Continue excellent performance in quality metrics</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Dashboard automatically refreshes every hour | Last updated: {data['generated_at']}
        </div>
    </div>
    
    <script>
        // Auto-refresh dashboard every hour
        setTimeout(() => {{
            window.location.reload();
        }}, 3600000);
        
        // Add real-time timestamp updates
        setInterval(() => {{
            const timestamps = document.querySelectorAll('.timestamp');
            timestamps.forEach(ts => {{
                const now = new Date().toISOString();
                ts.innerHTML = ts.innerHTML.replace(/Last updated: .*$/, `Last updated: ${{now}}`);
            }});
        }}, 60000);
    </script>
</body>
</html>
        """
        
        return html_template
    
    def generate_json_report(self, data: Dict[str, Any]) -> str:
        """Generate JSON compliance report.
        
        Args:
            data: Compliance data dictionary
            
        Returns:
            JSON string containing compliance report
        """
        return json.dumps(data, indent=2, default=str)
    
    def save_dashboard(self, output_dir: str = "compliance-dashboard") -> Dict[str, str]:
        """Generate and save compliance dashboard files.
        
        Args:
            output_dir: Directory to save dashboard files
            
        Returns:
            Dictionary with file paths of generated files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect compliance data
        compliance_data = self.collect_compliance_data()
        
        # Generate HTML dashboard
        html_content = self.generate_html_dashboard(compliance_data)
        html_path = os.path.join(output_dir, "compliance-dashboard.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate JSON report
        json_content = self.generate_json_report(compliance_data)
        json_path = os.path.join(output_dir, "compliance-data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        # Generate summary report
        summary = {
            "dashboard_generated": True,
            "framework": self.framework,
            "period_days": self.period_days,
            "overall_score": compliance_data["overall_score"],
            "total_violations": sum(compliance_data["violations"].values()),
            "controls_implemented": compliance_data["controls"]["implemented"],
            "controls_total": compliance_data["controls"]["total"],
            "generated_at": compliance_data["generated_at"],
            "files_generated": {
                "html_dashboard": html_path,
                "json_data": json_path
            }
        }
        
        summary_path = os.path.join(output_dir, "dashboard-summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return {
            "html_dashboard": html_path,
            "json_data": json_path,
            "summary": summary_path
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate compliance dashboard")
    parser.add_argument("--framework", default="SOC2", 
                       choices=["SOC2", "ISO27001", "GDPR", "HIPAA", "PCI-DSS"],
                       help="Compliance framework")
    parser.add_argument("--period", type=int, default=30,
                       help="Reporting period in days")
    parser.add_argument("--output", default="compliance-dashboard",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Generate dashboard
    generator = ComplianceDashboardGenerator(args.framework, args.period)
    files = generator.save_dashboard(args.output)
    
    print(f"✅ Compliance dashboard generated successfully!")
    print(f"📊 Framework: {args.framework}")
    print(f"📅 Period: {args.period} days")
    print(f"📁 Output directory: {args.output}")
    print(f"🌐 HTML Dashboard: {files['html_dashboard']}")
    print(f"📄 JSON Data: {files['json_data']}")
    print(f"📋 Summary: {files['summary']}")


if __name__ == "__main__":
    main()
