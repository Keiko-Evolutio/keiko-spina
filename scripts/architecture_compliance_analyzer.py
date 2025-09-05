#!/usr/bin/env python3
"""
Architektur-Compliance Analyzer für Keiko Personal Assistant
Analysiert Cross-Dependencies zwischen Platform und SDK
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ViolationType(Enum):
    """Typen von Architektur-Verletzungen"""
    DIRECT_IMPORT = "direct_import"
    INDIRECT_REFERENCE = "indirect_reference"
    SHARED_MODULE = "shared_module"
    PROTOCOL_VIOLATION = "protocol_violation"

class CriticalityLevel(Enum):
    """Kritikalitätsstufen für gefundene Verletzungen"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ArchitectureViolation:
    """Repräsentiert eine Architektur-Verletzung"""
    file_path: str
    line_number: int
    violation_type: ViolationType
    criticality: CriticalityLevel
    description: str
    code_snippet: str
    recommendation: str

class ArchitectureComplianceAnalyzer:
    """Hauptklasse für Architektur-Compliance Analyse"""
    
    def __init__(self, backend_path: str = "backend"):
        self.backend_path = Path(backend_path)
        self.violations: List[ArchitectureViolation] = []
        
        # Verbotene Import-Patterns
        self.forbidden_patterns = [
            r"from\s+kei_agent_py_sdk",
            r"import\s+kei_agent_py_sdk",
            r"from\s+kei\.agent",
            r"import\s+kei\.agent"
        ]
        
        # Verdächtige Referenzen
        self.suspicious_patterns = [
            r"kei-agent-py-sdk",
            r"kei_agent_py_sdk",
            r"agent\.py\.sdk",
            r"sdk\.agent"
        ]
        
    def analyze_cross_imports(self) -> Dict[str, List[ArchitectureViolation]]:
        """Führt umfassende Cross-Import Analyse durch"""
        print("🔍 Starte Cross-Import Analyse...")
        
        results = {
            "direct_imports": [],
            "indirect_references": [],
            "suspicious_patterns": [],
            "ast_violations": []
        }
        
        # 1. Direkte Import-Analyse
        results["direct_imports"] = self._analyze_direct_imports()
        
        # 2. AST-basierte Analyse
        results["ast_violations"] = self._analyze_with_ast()
        
        # 3. Verdächtige Pattern-Suche
        results["suspicious_patterns"] = self._analyze_suspicious_patterns()
        
        # 4. Konfigurationsdateien prüfen
        results["config_violations"] = self._analyze_config_files()
        
        return results
    
    def _analyze_direct_imports(self) -> List[ArchitectureViolation]:
        """Analysiert direkte Imports zwischen Platform und SDK"""
        violations = []
        
        for py_file in self.backend_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.forbidden_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            violation = ArchitectureViolation(
                                file_path=str(py_file.relative_to(self.backend_path.parent)),
                                line_number=line_num,
                                violation_type=ViolationType.DIRECT_IMPORT,
                                criticality=CriticalityLevel.CRITICAL,
                                description=f"Direkter SDK-Import gefunden: {line.strip()}",
                                code_snippet=line.strip(),
                                recommendation="Ersetze durch API-basierte Kommunikation"
                            )
                            violations.append(violation)
                            
            except Exception as e:
                print(f"⚠️ Fehler beim Analysieren von {py_file}: {e}")
                
        return violations
    
    def _analyze_with_ast(self) -> List[ArchitectureViolation]:
        """AST-basierte Analyse für komplexere Import-Patterns"""
        violations = []
        
        for py_file in self.backend_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                try:
                    tree = ast.parse(content)
                    visitor = ImportVisitor(str(py_file.relative_to(self.backend_path.parent)))
                    visitor.visit(tree)
                    violations.extend(visitor.violations)
                except SyntaxError:
                    # Ignoriere Syntax-Fehler in Dateien
                    pass
                    
            except Exception as e:
                print(f"⚠️ AST-Fehler bei {py_file}: {e}")
                
        return violations
    
    def _analyze_suspicious_patterns(self) -> List[ArchitectureViolation]:
        """Sucht nach verdächtigen Patterns die auf SDK-Dependencies hindeuten"""
        violations = []
        
        for py_file in self.backend_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.suspicious_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Ignoriere Kommentare und Dokumentation
                            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                                continue
                                
                            violation = ArchitectureViolation(
                                file_path=str(py_file.relative_to(self.backend_path.parent)),
                                line_number=line_num,
                                violation_type=ViolationType.INDIRECT_REFERENCE,
                                criticality=CriticalityLevel.MEDIUM,
                                description=f"Verdächtige SDK-Referenz: {line.strip()}",
                                code_snippet=line.strip(),
                                recommendation="Prüfe ob dies eine versteckte SDK-Abhängigkeit ist"
                            )
                            violations.append(violation)
                            
            except Exception as e:
                print(f"⚠️ Fehler beim Pattern-Scan von {py_file}: {e}")
                
        return violations
    
    def _analyze_config_files(self) -> List[ArchitectureViolation]:
        """Analysiert Konfigurationsdateien auf SDK-Dependencies"""
        violations = []
        config_files = [
            "pyproject.toml", "requirements.txt", "setup.py", 
            "Pipfile", "poetry.lock", "uv.lock"
        ]
        
        for config_file in config_files:
            config_path = self.backend_path / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                    for line_num, line in enumerate(lines, 1):
                        if "kei-agent-py-sdk" in line.lower() or "kei_agent_py_sdk" in line.lower():
                            violation = ArchitectureViolation(
                                file_path=str(config_path.relative_to(self.backend_path.parent)),
                                line_number=line_num,
                                violation_type=ViolationType.PROTOCOL_VIOLATION,
                                criticality=CriticalityLevel.CRITICAL,
                                description=f"SDK-Dependency in Konfiguration: {line.strip()}",
                                code_snippet=line.strip(),
                                recommendation="Entferne SDK-Dependency aus Platform-Konfiguration"
                            )
                            violations.append(violation)
                            
                except Exception as e:
                    print(f"⚠️ Fehler beim Analysieren von {config_path}: {e}")
                    
        return violations

class ImportVisitor(ast.NodeVisitor):
    """AST Visitor für Import-Analyse"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations: List[ArchitectureViolation] = []
        
    def visit_Import(self, node):
        """Besucht Import-Statements"""
        for alias in node.names:
            if self._is_forbidden_import(alias.name):
                violation = ArchitectureViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    violation_type=ViolationType.DIRECT_IMPORT,
                    criticality=CriticalityLevel.CRITICAL,
                    description=f"AST: Verbotener Import gefunden: {alias.name}",
                    code_snippet=f"import {alias.name}",
                    recommendation="Ersetze durch API-basierte Kommunikation"
                )
                self.violations.append(violation)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Besucht From-Import-Statements"""
        if node.module and self._is_forbidden_import(node.module):
            violation = ArchitectureViolation(
                file_path=self.file_path,
                line_number=node.lineno,
                violation_type=ViolationType.DIRECT_IMPORT,
                criticality=CriticalityLevel.CRITICAL,
                description=f"AST: Verbotener From-Import: from {node.module}",
                code_snippet=f"from {node.module} import ...",
                recommendation="Ersetze durch API-basierte Kommunikation"
            )
            self.violations.append(violation)
        self.generic_visit(node)
        
    def _is_forbidden_import(self, module_name: str) -> bool:
        """Prüft ob ein Import verboten ist"""
        forbidden_modules = [
            "kei_agent_py_sdk",
            "kei.agent",
            "agent.sdk"
        ]
        return any(forbidden in module_name.lower() for forbidden in forbidden_modules)

def main():
    """Hauptfunktion für CLI-Ausführung"""
    analyzer = ArchitectureComplianceAnalyzer()
    
    print("🏗️ Keiko Architecture Compliance Analyzer")
    print("=" * 50)
    
    # Führe Analyse durch
    results = analyzer.analyze_cross_imports()
    
    # Erstelle Bericht
    total_violations = sum(len(violations) for violations in results.values())
    
    print(f"\n📊 ANALYSE-ERGEBNISSE:")
    print(f"Direkte Imports: {len(results['direct_imports'])}")
    print(f"AST-Verletzungen: {len(results['ast_violations'])}")
    print(f"Verdächtige Patterns: {len(results['suspicious_patterns'])}")
    print(f"Konfigurations-Verletzungen: {len(results['config_violations'])}")
    print(f"GESAMT: {total_violations} Verletzungen")
    
    # Speichere detaillierten Bericht
    report_path = "architecture_compliance_report.json"
    report_data = {
        "summary": {
            "total_violations": total_violations,
            "direct_imports": len(results['direct_imports']),
            "ast_violations": len(results['ast_violations']),
            "suspicious_patterns": len(results['suspicious_patterns']),
            "config_violations": len(results['config_violations'])
        },
        "violations": {
            key: [
                {
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "violation_type": v.violation_type.value,
                    "criticality": v.criticality.value,
                    "description": v.description,
                    "code_snippet": v.code_snippet,
                    "recommendation": v.recommendation
                } for v in violations
            ] for key, violations in results.items()
        }
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detaillierter Bericht gespeichert: {report_path}")
    
    if total_violations == 0:
        print("\n✅ ERFOLG: Keine Architektur-Verletzungen gefunden!")
        return 0
    else:
        print(f"\n❌ WARNUNG: {total_violations} Architektur-Verletzungen gefunden!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
