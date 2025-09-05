#!/usr/bin/env python3
"""
SonarQube Report Generator for Keiko Backend
===========================================

This script generates comprehensive reports for SonarQube analysis by:
1. Running all quality tools (pytest, bandit, mypy, ruff)
2. Converting outputs to SonarQube-compatible formats
3. Generating a summary report with metrics

Usage:
    python scripts/sonar-report-generator.py [--output-dir reports]
    
Requirements:
    - All development dependencies installed (uv sync --group dev --group test)
    - Project tests and source code in expected locations
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET
from datetime import datetime


class SonarReportGenerator:
    """Generates comprehensive reports for SonarQube analysis."""
    
    def __init__(self, output_dir: Path = Path("sonar-reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.reports = {}
    
    def run_command(self, cmd: list, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        print(f"🔄 Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output,
                text=True,
                cwd=Path.cwd()
            )
            return result
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            return subprocess.CompletedProcess(cmd, 1, "", f"Command not found: {cmd[0]}")
    
    def generate_coverage_report(self) -> bool:
        """Generate test coverage report in XML format."""
        print("📊 Generating coverage report...")
        
        cmd = [
            "uv", "run", "pytest", 
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing",
            "--junitxml=pytest-junit.xml",
            "--tb=short"
        ]
        
        result = self.run_command(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Coverage report generated successfully")
            self.reports['coverage'] = "coverage.xml"
            self.reports['test_results'] = "pytest-junit.xml"
            return True
        else:
            print(f"⚠️ Coverage generation had issues (exit code: {result.returncode})")
            return False
    
    def generate_security_report(self) -> bool:
        """Generate security scan report using Bandit."""
        print("🔒 Generating security report...")
        
        cmd = [
            "uv", "run", "bandit",
            "-r", ".",
            "-x", "tests/",
            "-f", "json",
            "-o", "bandit-report.json"
        ]
        
        result = self.run_command(cmd)
        
        if result.returncode in [0, 1]:  # Bandit returns 1 when issues found
            print("✅ Security report generated successfully")
            self.reports['security'] = "bandit-report.json"
            return True
        else:
            print(f"❌ Security scan failed (exit code: {result.returncode})")
            return False
    
    def generate_type_checking_report(self) -> bool:
        """Generate type checking report using MyPy."""
        print("🎯 Generating type checking report...")
        
        # Try JSON format first, fall back to text
        cmd_json = [
            "uv", "run", "mypy",
            "--no-error-summary",
            "."
        ]
        
        result = self.run_command(cmd_json)
        
        # Save MyPy output to file
        mypy_output_file = self.output_dir / "mypy-report.txt"
        with open(mypy_output_file, "w") as f:
            f.write(f"MyPy Type Checking Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Exit Code: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        print("✅ Type checking report generated")
        self.reports['typing'] = str(mypy_output_file)
        return True
    
    def generate_linting_report(self) -> bool:
        """Generate linting report using Ruff."""
        print("🧹 Generating linting report...")
        
        # Generate JSON format for SonarQube
        cmd = [
            "uv", "run", "ruff",
            "check", ".",
            "--output-format=json"
        ]
        
        result = self.run_command(cmd)
        
        # Save Ruff output to file
        ruff_output_file = self.output_dir / "ruff-report.json"
        try:
            # Try to parse as JSON
            if result.stdout:
                ruff_data = json.loads(result.stdout)
                with open(ruff_output_file, "w") as f:
                    json.dump(ruff_data, f, indent=2)
            else:
                # Create empty JSON if no issues
                with open(ruff_output_file, "w") as f:
                    json.dump([], f)
        except json.JSONDecodeError:
            # Fall back to text format
            with open(ruff_output_file, "w") as f:
                f.write(result.stdout)
        
        print("✅ Linting report generated")
        self.reports['linting'] = str(ruff_output_file)
        return True
    
    def generate_project_metrics(self) -> Dict[str, Any]:
        """Generate basic project metrics."""
        print("📈 Calculating project metrics...")
        
        # Count lines of code
        py_files = list(Path(".").rglob("*.py"))
        # Exclude common non-source directories
        exclude_patterns = [
            "venv", "__pycache__", ".pytest_cache", ".mypy_cache", 
            ".ruff_cache", "build", "dist", "tests", "migrations"
        ]
        
        source_files = []
        test_files = []
        
        for py_file in py_files:
            path_str = str(py_file)
            if any(pattern in path_str for pattern in exclude_patterns):
                continue
            
            if "test" in path_str or path_str.startswith("tests/"):
                test_files.append(py_file)
            else:
                source_files.append(py_file)
        
        # Count lines
        total_lines = 0
        total_source_lines = 0
        
        for source_file in source_files:
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    # Count non-empty, non-comment lines
                    source_lines = sum(1 for line in lines 
                                     if line.strip() and not line.strip().startswith('#'))
                    total_source_lines += source_lines
            except Exception:
                continue
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "project_name": "Keiko Backend",
            "total_python_files": len(source_files),
            "test_files": len(test_files),
            "total_lines": total_lines,
            "source_lines": total_source_lines,
            "reports_generated": self.reports,
            "analysis_tools": {
                "coverage": "pytest with coverage.py",
                "security": "bandit",
                "typing": "mypy",
                "linting": "ruff"
            }
        }
        
        return metrics
    
    def generate_summary_report(self, metrics: Dict[str, Any]) -> None:
        """Generate a summary report of the analysis."""
        summary_file = self.output_dir / "analysis-summary.json"
        
        with open(summary_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Also generate a human-readable summary
        summary_md_file = self.output_dir / "analysis-summary.md"
        with open(summary_md_file, "w") as f:
            f.write("# SonarQube Analysis Summary\n\n")
            f.write(f"**Generated:** {metrics['timestamp']}\n")
            f.write(f"**Project:** {metrics['project_name']}\n\n")
            f.write("## Project Metrics\n\n")
            f.write(f"- Total Python Files: {metrics['total_python_files']}\n")
            f.write(f"- Test Files: {metrics['test_files']}\n")
            f.write(f"- Total Lines: {metrics['total_lines']}\n")
            f.write(f"- Source Lines: {metrics['source_lines']}\n\n")
            f.write("## Analysis Tools Used\n\n")
            for tool, description in metrics['analysis_tools'].items():
                f.write(f"- **{tool.title()}**: {description}\n")
            f.write("\n## Generated Reports\n\n")
            for report_type, file_path in self.reports.items():
                f.write(f"- **{report_type.title()}**: `{file_path}`\n")
        
        print(f"✅ Summary reports generated:")
        print(f"   - {summary_file}")
        print(f"   - {summary_md_file}")
    
    def run_full_analysis(self) -> bool:
        """Run complete analysis and generate all reports."""
        print("🚀 Starting comprehensive SonarQube report generation...\n")
        
        success_count = 0
        
        # Generate all reports
        if self.generate_coverage_report():
            success_count += 1
        
        if self.generate_security_report():
            success_count += 1
        
        if self.generate_type_checking_report():
            success_count += 1
        
        if self.generate_linting_report():
            success_count += 1
        
        # Generate project metrics and summary
        metrics = self.generate_project_metrics()
        self.generate_summary_report(metrics)
        
        print(f"\n🎉 Analysis complete! {success_count}/4 report types generated successfully.")
        print(f"📁 Reports saved in: {self.output_dir.absolute()}")
        
        if success_count == 4:
            print("\n✅ All reports generated successfully! Ready for SonarQube analysis.")
            print("\nNext steps:")
            print("1. Start SonarQube: docker run -d --name sonarqube -p 9000:9000 sonarqube:latest")
            print("2. Set SONAR_TOKEN environment variable")
            print("3. Run: make sonar-analyze")
            return True
        else:
            print(f"\n⚠️ Some reports failed to generate. Check the output above.")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive reports for SonarQube analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sonar-reports"),
        help="Output directory for reports (default: sonar-reports)"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: This script must be run from the project root directory")
        sys.exit(1)
    
    generator = SonarReportGenerator(args.output_dir)
    success = generator.run_full_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()