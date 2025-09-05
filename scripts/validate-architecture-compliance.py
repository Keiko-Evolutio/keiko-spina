#!/usr/bin/env python3
"""
Automatisierte Validierung der Repository-Trennung und Architektur-Compliance
Überprüft dass keine Cross-Dependencies zwischen den Repositories bestehen
"""

import re
import sys
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    """Ergebnis einer Validierung"""
    test_name: str
    passed: bool
    message: str
    details: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []

@dataclass
class ArchitectureComplianceReport:
    """Umfassender Architektur-Compliance-Report"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    compliance_score: float
    results: List[ValidationResult]
    critical_issues: List[str]
    warnings: List[str]


class ArchitectureValidator:
    """Validator für Repository-Architektur-Compliance"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.repositories = {
            "keiko-backend": self.root_path / "keiko-backend",
            "keiko-frontend": self.root_path / "keiko-frontend", 
            "kei-agent-py-sdk": self.root_path / "kei-agent-py-sdk",
            "keiko-api-contracts": self.root_path / "keiko-api-contracts"
        }
        
        # Verbotene Cross-Dependencies
        self.forbidden_imports = {
            "keiko-backend": [
                r"from kei_agent\.",
                r"import kei_agent",
                r"from kei_agents\.",
                r"import kei_agents"
            ],
            "keiko-frontend": [
                r"from kei_agent",
                r"import kei_agent",
                r"from keiko_backend",
                r"import keiko_backend"
            ],
            "kei-agent-py-sdk": [
                r"from keiko_backend",
                r"import keiko_backend",
                r"from keiko\.backend",
                r"import keiko\.backend"
            ]
        }
        
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> ArchitectureComplianceReport:
        """Führt alle Validierungen durch"""
        print("🔍 Starte Architektur-Compliance-Validierung...")
        
        # Test 1: Cross-Dependencies
        self._validate_cross_dependencies()
        
        # Test 2: Repository-Struktur
        self._validate_repository_structure()
        
        # Test 3: API-Contracts
        self._validate_api_contracts()
        
        # Test 4: Docker-Konfigurationen
        self._validate_docker_configs()
        
        # Test 5: Eigenständige Deploybarkeit
        self._validate_standalone_deployability()
        
        # Test 6: Logging-Compliance
        self._validate_logging_compliance()
        
        # Test 7: Import-Patterns
        self._validate_import_patterns()
        
        # Test 8: Configuration-Isolation
        self._validate_configuration_isolation()
        
        # Generiere Report
        return self._generate_report()
    
    def _validate_cross_dependencies(self):
        """Validiert dass keine Cross-Dependencies bestehen"""
        print("  📋 Teste Cross-Dependencies...")
        
        for repo_name, repo_path in self.repositories.items():
            if not repo_path.exists():
                continue
                
            forbidden_patterns = self.forbidden_imports.get(repo_name, [])
            if not forbidden_patterns:
                continue
            
            violations = []
            
            # Durchsuche Python-Dateien
            for py_file in repo_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for line_num, line in enumerate(content.splitlines(), 1):
                        for pattern in forbidden_patterns:
                            if re.search(pattern, line):
                                violations.append(f"{py_file.relative_to(repo_path)}:{line_num} - {line.strip()}")
                except Exception as e:
                    print(f"    ⚠️ Fehler beim Lesen von {py_file}: {e}")
            
            # Durchsuche TypeScript-Dateien (für Frontend)
            if repo_name == "keiko-frontend":
                for ts_file in repo_path.rglob("*.ts"):
                    try:
                        content = ts_file.read_text(encoding='utf-8')
                        for line_num, line in enumerate(content.splitlines(), 1):
                            if "kei_agent" in line or "kei-agent-py-sdk" in line:
                                violations.append(f"{ts_file.relative_to(repo_path)}:{line_num} - {line.strip()}")
                    except Exception as e:
                        print(f"    ⚠️ Fehler beim Lesen von {ts_file}: {e}")
            
            # Ergebnis
            # Filter out false positives (API operation names)
            real_violations = []
            for violation in violations:
                if not any(pattern in violation for pattern in [
                    "operations[\"agents-kei_agents_",  # API operation names
                    "\"agents-kei_agents_",  # API operation names
                    "kei_agents_health",  # API endpoint names
                    "list_kei_agents"  # API endpoint names
                ]):
                    real_violations.append(violation)

            if real_violations:
                self.results.append(ValidationResult(
                    test_name=f"Cross-Dependencies: {repo_name}",
                    passed=False,
                    message=f"❌ {len(real_violations)} Cross-Dependencies gefunden",
                    details=real_violations
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=f"Cross-Dependencies: {repo_name}",
                    passed=True,
                    message="✅ Keine Cross-Dependencies gefunden"
                ))
    
    def _validate_repository_structure(self):
        """Validiert Repository-Strukturen"""
        print("  📁 Teste Repository-Strukturen...")
        
        expected_structures = {
            "keiko-backend": [
                "src/", "tests/", "Dockerfile", "pyproject.toml", "README.md"
            ],
            "keiko-frontend": [
                "src/", "tests/", "Dockerfile", "package.json", "README.md"
            ],
            "kei-agent-py-sdk": [
                "kei_agent/", "tests/", "Dockerfile", "pyproject.toml", "README.md"
            ],
            "keiko-api-contracts": [
                "openapi/", "asyncapi/", "protobuf/", "README.md"
            ]
        }
        
        for repo_name, expected_files in expected_structures.items():
            repo_path = self.repositories[repo_name]
            if not repo_path.exists():
                self.results.append(ValidationResult(
                    test_name=f"Repository-Struktur: {repo_name}",
                    passed=False,
                    message=f"❌ Repository {repo_name} existiert nicht"
                ))
                continue
            
            missing_files = []
            for expected_file in expected_files:
                if not (repo_path / expected_file).exists():
                    missing_files.append(expected_file)
            
            if missing_files:
                self.results.append(ValidationResult(
                    test_name=f"Repository-Struktur: {repo_name}",
                    passed=False,
                    message=f"❌ {len(missing_files)} erwartete Dateien/Ordner fehlen",
                    details=missing_files
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=f"Repository-Struktur: {repo_name}",
                    passed=True,
                    message="✅ Repository-Struktur vollständig"
                ))
    
    def _validate_api_contracts(self):
        """Validiert API-Contracts"""
        print("  🔌 Teste API-Contracts...")
        
        contracts_path = self.repositories["keiko-api-contracts"]
        if not contracts_path.exists():
            self.results.append(ValidationResult(
                test_name="API-Contracts",
                passed=False,
                message="❌ keiko-api-contracts Repository fehlt"
            ))
            return
        
        required_contracts = [
            "openapi/backend-frontend-api-v1.yaml",
            "openapi/backend-sdk-api-v1.yaml",
            "asyncapi/backend-events-v1.yaml",
            "protobuf/agent_service.proto"
        ]
        
        missing_contracts = []
        for contract in required_contracts:
            if not (contracts_path / contract).exists():
                missing_contracts.append(contract)
        
        if missing_contracts:
            self.results.append(ValidationResult(
                test_name="API-Contracts",
                passed=False,
                message=f"❌ {len(missing_contracts)} API-Contracts fehlen",
                details=missing_contracts
            ))
        else:
            self.results.append(ValidationResult(
                test_name="API-Contracts",
                passed=True,
                message="✅ Alle API-Contracts vorhanden"
            ))
    
    def _validate_docker_configs(self):
        """Validiert Docker-Konfigurationen"""
        print("  🐳 Teste Docker-Konfigurationen...")
        
        for repo_name, repo_path in self.repositories.items():
            if not repo_path.exists():
                continue
            
            dockerfile_path = repo_path / "Dockerfile"
            if dockerfile_path.exists():
                self.results.append(ValidationResult(
                    test_name=f"Docker-Config: {repo_name}",
                    passed=True,
                    message="✅ Dockerfile vorhanden"
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=f"Docker-Config: {repo_name}",
                    passed=False,
                    message="❌ Dockerfile fehlt"
                ))
    
    def _validate_standalone_deployability(self):
        """Validiert eigenständige Deploybarkeit"""
        print("  🚀 Teste eigenständige Deploybarkeit...")
        
        # Check für docker-compose.dev-multi-repo.yml
        compose_file = self.root_path / "docker-compose.dev-multi-repo.yml"
        if compose_file.exists():
            self.results.append(ValidationResult(
                test_name="Multi-Repository Docker Compose",
                passed=True,
                message="✅ Multi-Repository Docker Compose vorhanden"
            ))
        else:
            self.results.append(ValidationResult(
                test_name="Multi-Repository Docker Compose",
                passed=False,
                message="❌ docker-compose.dev-multi-repo.yml fehlt"
            ))
    
    def _validate_logging_compliance(self):
        """Validiert Logging-Compliance"""
        print("  📝 Teste Logging-Compliance...")
        
        # Backend sollte kei_logging verwenden
        backend_path = self.repositories["keiko-backend"]
        if backend_path.exists():
            violations = []
            for py_file in backend_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    lines = content.splitlines()
                    for line_num, line in enumerate(lines, 1):
                        if "print(" in line and "# Debug" not in line and not line.strip().startswith("#"):
                            # Exclude false positives like cert.fingerprint(), function definitions, etc.
                            if not any(fp in line for fp in [
                                ".fingerprint(",
                                "pprint(",
                                "blueprint(",
                                "imprint(",
                                "def ",  # Function definitions
                                "fingerprint:",  # Type hints
                                "fingerprint ="  # Assignments
                            ]) and line.strip().startswith("print("):
                                violations.append(f"{py_file.relative_to(backend_path)}:{line_num}")
                                break  # Only report first occurrence per file
                except Exception:
                    pass
            
            if violations:
                self.results.append(ValidationResult(
                    test_name="Logging-Compliance: Backend",
                    passed=False,
                    message=f"❌ {len(violations)} Dateien verwenden print() statt kei_logging",
                    details=violations[:10]  # Nur erste 10 anzeigen
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="Logging-Compliance: Backend",
                    passed=True,
                    message="✅ Korrekte Logging-Verwendung"
                ))
    
    def _validate_import_patterns(self):
        """Validiert Import-Patterns"""
        print("  📦 Teste Import-Patterns...")
        
        # Backend sollte relative Imports verwenden
        backend_path = self.repositories["keiko-backend"]
        if backend_path.exists():
            absolute_imports = []
            for py_file in (backend_path / "src").rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if re.search(r"from src\.", line) or re.search(r"import src\.", line):
                            absolute_imports.append(f"{py_file.relative_to(backend_path)}:{line_num}")
                except Exception:
                    pass
            
            if absolute_imports:
                self.results.append(ValidationResult(
                    test_name="Import-Patterns: Backend",
                    passed=False,
                    message=f"❌ {len(absolute_imports)} absolute Imports gefunden",
                    details=absolute_imports[:5]
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="Import-Patterns: Backend",
                    passed=True,
                    message="✅ Korrekte relative Imports"
                ))
    
    def _validate_configuration_isolation(self):
        """Validiert Konfigurations-Isolation"""
        print("  ⚙️ Teste Konfigurations-Isolation...")
        
        # Jedes Repository sollte eigene Konfiguration haben
        config_files = {
            "keiko-backend": ["pyproject.toml"],
            "keiko-frontend": ["package.json", "vite.config.ts"],
            "kei-agent-py-sdk": ["pyproject.toml"],
            "keiko-api-contracts": ["README.md"]
        }
        
        for repo_name, expected_configs in config_files.items():
            repo_path = self.repositories[repo_name]
            if not repo_path.exists():
                continue
            
            missing_configs = []
            for config_file in expected_configs:
                if not (repo_path / config_file).exists():
                    missing_configs.append(config_file)
            
            if missing_configs:
                self.results.append(ValidationResult(
                    test_name=f"Konfigurations-Isolation: {repo_name}",
                    passed=False,
                    message=f"❌ {len(missing_configs)} Konfigurationsdateien fehlen",
                    details=missing_configs
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=f"Konfigurations-Isolation: {repo_name}",
                    passed=True,
                    message="✅ Konfiguration vollständig isoliert"
                ))
    
    def _generate_report(self) -> ArchitectureComplianceReport:
        """Generiert umfassenden Compliance-Report"""
        passed_tests = sum(1 for result in self.results if result.passed)
        failed_tests = len(self.results) - passed_tests
        compliance_score = (passed_tests / len(self.results)) * 100 if self.results else 0
        
        # Kritische Issues sammeln
        critical_issues = []
        warnings = []
        
        for result in self.results:
            if not result.passed:
                if "Cross-Dependencies" in result.test_name:
                    critical_issues.append(f"KRITISCH: {result.message}")
                else:
                    warnings.append(f"WARNUNG: {result.message}")
        
        return ArchitectureComplianceReport(
            timestamp=datetime.now().isoformat(),
            total_tests=len(self.results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            compliance_score=compliance_score,
            results=self.results,
            critical_issues=critical_issues,
            warnings=warnings
        )


def main():
    """Hauptfunktion"""
    print("🏗️ Repository-Trennung Architektur-Compliance-Validator")
    print("=" * 60)
    
    validator = ArchitectureValidator()
    report = validator.validate_all()
    
    print("\n📊 COMPLIANCE-REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Tests gesamt: {report.total_tests}")
    print(f"Tests bestanden: {report.passed_tests}")
    print(f"Tests fehlgeschlagen: {report.failed_tests}")
    print(f"Compliance-Score: {report.compliance_score:.1f}%")
    
    # Kritische Issues
    if report.critical_issues:
        print(f"\n🚨 KRITISCHE ISSUES ({len(report.critical_issues)}):")
        for issue in report.critical_issues:
            print(f"  {issue}")
    
    # Warnungen
    if report.warnings:
        print(f"\n⚠️ WARNUNGEN ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  {warning}")
    
    # Detaillierte Ergebnisse
    print("\n📋 DETAILLIERTE ERGEBNISSE:")
    for result in report.results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.test_name}: {result.message}")
        if result.details and not result.passed:
            for detail in result.details[:3]:  # Nur erste 3 Details anzeigen
                print(f"    - {detail}")
            if len(result.details) > 3:
                print(f"    ... und {len(result.details) - 3} weitere")
    
    # Report als JSON speichern
    report_file = Path("architecture-compliance-report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": report.timestamp,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "compliance_score": report.compliance_score,
            "critical_issues": report.critical_issues,
            "warnings": report.warnings,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                } for r in report.results
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Report gespeichert: {report_file}")
    
    # Exit-Code basierend auf kritischen Issues
    if report.critical_issues:
        print(f"\n❌ VALIDIERUNG FEHLGESCHLAGEN: {len(report.critical_issues)} kritische Issues")
        sys.exit(1)
    elif report.compliance_score < 90:
        print(f"\n⚠️ VALIDIERUNG MIT WARNUNGEN: Compliance-Score {report.compliance_score:.1f}% < 90%")
        sys.exit(2)
    else:
        print(f"\n✅ VALIDIERUNG ERFOLGREICH: Compliance-Score {report.compliance_score:.1f}%")
        sys.exit(0)


if __name__ == "__main__":
    main()
