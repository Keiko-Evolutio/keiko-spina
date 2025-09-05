#!/usr/bin/env python3
"""
API Contract Validator für Keiko Platform-SDK Kommunikation
Automatisierte Validierung der API-Verträge in CI/CD Pipeline
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests
import jsonschema
from jsonschema import validate, ValidationError
import openapi_spec_validator
from openapi_spec_validator import validate_spec
from openapi_spec_validator.readers import read_from_filename

class ValidationResult(Enum):
    """Validierungs-Ergebnisse"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class ContractValidationIssue:
    """Repräsentiert ein Contract-Validierungs-Problem"""
    contract_file: str
    issue_type: str
    severity: str
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

@dataclass
class ContractValidationReport:
    """Validierungs-Report für API-Contracts"""
    total_contracts: int
    passed_contracts: int
    failed_contracts: int
    warnings: int
    issues: List[ContractValidationIssue]
    execution_time_ms: int

class APIContractValidator:
    """Hauptklasse für API-Contract Validierung"""
    
    def __init__(self, contracts_dir: str = "api-contracts"):
        self.contracts_dir = Path(contracts_dir)
        self.issues: List[ContractValidationIssue] = []
        
        # Validierungs-Konfiguration
        self.openapi_files = list(self.contracts_dir.glob("openapi/**/*.yaml"))
        self.asyncapi_files = list(self.contracts_dir.glob("asyncapi/**/*.yaml"))
        self.protobuf_files = list(self.contracts_dir.glob("protobuf/**/*.proto"))
        
    def validate_all_contracts(self) -> ContractValidationReport:
        """Führt vollständige Contract-Validierung durch"""
        print("🔍 Starte API-Contract Validierung...")
        
        import time
        start_time = time.time()
        
        total_contracts = 0
        passed_contracts = 0
        failed_contracts = 0
        warnings = 0
        
        # 1. OpenAPI Validierung
        openapi_results = self._validate_openapi_contracts()
        total_contracts += len(self.openapi_files)
        passed_contracts += openapi_results["passed"]
        failed_contracts += openapi_results["failed"]
        warnings += openapi_results["warnings"]
        
        # 2. AsyncAPI Validierung
        asyncapi_results = self._validate_asyncapi_contracts()
        total_contracts += len(self.asyncapi_files)
        passed_contracts += asyncapi_results["passed"]
        failed_contracts += asyncapi_results["failed"]
        warnings += asyncapi_results["warnings"]
        
        # 3. Protocol Buffers Validierung
        protobuf_results = self._validate_protobuf_contracts()
        total_contracts += len(self.protobuf_files)
        passed_contracts += protobuf_results["passed"]
        failed_contracts += protobuf_results["failed"]
        warnings += protobuf_results["warnings"]
        
        # 4. Cross-Contract Consistency Checks
        consistency_results = self._validate_cross_contract_consistency()
        warnings += consistency_results["warnings"]
        failed_contracts += consistency_results["failed"]
        
        # 5. Architecture Compliance Checks
        compliance_results = self._validate_architecture_compliance()
        warnings += compliance_results["warnings"]
        failed_contracts += compliance_results["failed"]
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return ContractValidationReport(
            total_contracts=total_contracts,
            passed_contracts=passed_contracts,
            failed_contracts=failed_contracts,
            warnings=warnings,
            issues=self.issues,
            execution_time_ms=execution_time
        )
    
    def _validate_openapi_contracts(self) -> Dict[str, int]:
        """Validiert OpenAPI 3.0 Spezifikationen"""
        print("📋 Validiere OpenAPI Contracts...")
        
        passed = 0
        failed = 0
        warnings = 0
        
        for openapi_file in self.openapi_files:
            try:
                # OpenAPI Spec laden und validieren
                spec_dict, spec_url = read_from_filename(str(openapi_file))
                validate_spec(spec_dict)
                
                # Zusätzliche Validierungen
                self._validate_openapi_architecture_compliance(openapi_file, spec_dict)
                self._validate_openapi_versioning(openapi_file, spec_dict)
                self._validate_openapi_security(openapi_file, spec_dict)
                
                passed += 1
                print(f"  ✅ {openapi_file.name}")
                
            except Exception as e:
                failed += 1
                self.issues.append(ContractValidationIssue(
                    contract_file=str(openapi_file),
                    issue_type="openapi_validation",
                    severity="error",
                    message=f"OpenAPI validation failed: {str(e)}",
                    suggestion="Fix OpenAPI specification syntax and structure"
                ))
                print(f"  ❌ {openapi_file.name}: {str(e)}")
        
        return {"passed": passed, "failed": failed, "warnings": warnings}
    
    def _validate_openapi_architecture_compliance(self, file_path: Path, spec: Dict[str, Any]):
        """Prüft OpenAPI auf Architektur-Compliance"""
        
        # Prüfe auf verbotene direkte NATS-Referenzen
        spec_str = json.dumps(spec).lower()
        if "nats" in spec_str or "jetstream" in spec_str:
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="architecture_compliance",
                severity="error",
                message="OpenAPI contains direct NATS/JetStream references - violates API-first architecture",
                suggestion="Remove NATS references, use HTTP/gRPC APIs only"
            ))
        
        # Prüfe auf korrekte Versionierung
        if not spec.get("info", {}).get("version"):
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="versioning",
                severity="warning",
                message="Missing API version in OpenAPI spec",
                suggestion="Add version field to info section"
            ))
    
    def _validate_openapi_versioning(self, file_path: Path, spec: Dict[str, Any]):
        """Validiert OpenAPI Versionierung"""
        
        info = spec.get("info", {})
        version = info.get("version")
        
        if not version:
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="versioning",
                severity="error",
                message="Missing version in OpenAPI info section"
            ))
            return
        
        # Prüfe Semantic Versioning
        import re
        semver_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9\-\.]+)?$"
        if not re.match(semver_pattern, version):
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="versioning",
                severity="warning",
                message=f"Version '{version}' does not follow semantic versioning",
                suggestion="Use semantic versioning format (e.g., 1.0.0)"
            ))
    
    def _validate_openapi_security(self, file_path: Path, spec: Dict[str, Any]):
        """Validiert OpenAPI Security-Definitionen"""
        
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
        
        if not security_schemes:
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="security",
                severity="warning",
                message="No security schemes defined in OpenAPI spec",
                suggestion="Add authentication mechanisms (API Key, Bearer Token, etc.)"
            ))
        
        # Prüfe auf unsichere Authentifizierung
        for scheme_name, scheme in security_schemes.items():
            if scheme.get("type") == "apiKey" and scheme.get("in") == "query":
                self.issues.append(ContractValidationIssue(
                    contract_file=str(file_path),
                    issue_type="security",
                    severity="warning",
                    message=f"API Key in query parameter is less secure: {scheme_name}",
                    suggestion="Consider using header-based API key authentication"
                ))
    
    def _validate_asyncapi_contracts(self) -> Dict[str, int]:
        """Validiert AsyncAPI Spezifikationen"""
        print("🔄 Validiere AsyncAPI Contracts...")
        
        passed = 0
        failed = 0
        warnings = 0
        
        for asyncapi_file in self.asyncapi_files:
            try:
                with open(asyncapi_file, 'r', encoding='utf-8') as f:
                    spec = yaml.safe_load(f)
                
                # AsyncAPI Version prüfen
                asyncapi_version = spec.get("asyncapi")
                if not asyncapi_version or not asyncapi_version.startswith("3."):
                    self.issues.append(ContractValidationIssue(
                        contract_file=str(asyncapi_file),
                        issue_type="asyncapi_version",
                        severity="warning",
                        message=f"AsyncAPI version {asyncapi_version} - recommend 3.x",
                        suggestion="Upgrade to AsyncAPI 3.x for better features"
                    ))
                
                # Architektur-Compliance prüfen
                self._validate_asyncapi_architecture_compliance(asyncapi_file, spec)
                
                passed += 1
                print(f"  ✅ {asyncapi_file.name}")
                
            except Exception as e:
                failed += 1
                self.issues.append(ContractValidationIssue(
                    contract_file=str(asyncapi_file),
                    issue_type="asyncapi_validation",
                    severity="error",
                    message=f"AsyncAPI validation failed: {str(e)}"
                ))
                print(f"  ❌ {asyncapi_file.name}: {str(e)}")
        
        return {"passed": passed, "failed": failed, "warnings": warnings}
    
    def _validate_asyncapi_architecture_compliance(self, file_path: Path, spec: Dict[str, Any]):
        """Prüft AsyncAPI auf Architektur-Compliance"""
        
        # Prüfe Server-Definitionen auf erlaubte Protokolle
        servers = spec.get("servers", {})
        for server_name, server in servers.items():
            protocol = server.get("protocol", "").lower()
            
            # Erlaubte Protokolle für SDK-Platform Kommunikation
            allowed_protocols = ["wss", "ws", "https", "http"]
            
            if protocol not in allowed_protocols:
                self.issues.append(ContractValidationIssue(
                    contract_file=str(file_path),
                    issue_type="architecture_compliance",
                    severity="error",
                    message=f"Protocol '{protocol}' not allowed for SDK-Platform communication",
                    suggestion=f"Use allowed protocols: {', '.join(allowed_protocols)}"
                ))
    
    def _validate_protobuf_contracts(self) -> Dict[str, int]:
        """Validiert Protocol Buffers Definitionen"""
        print("📦 Validiere Protocol Buffers Contracts...")
        
        passed = 0
        failed = 0
        warnings = 0
        
        for proto_file in self.protobuf_files:
            try:
                # Protobuf Syntax-Validierung mit protoc
                result = subprocess.run([
                    "protoc", "--proto_path", str(proto_file.parent),
                    "--descriptor_set_out=/dev/null", str(proto_file)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Zusätzliche Architektur-Validierung
                    self._validate_protobuf_architecture_compliance(proto_file)
                    passed += 1
                    print(f"  ✅ {proto_file.name}")
                else:
                    failed += 1
                    self.issues.append(ContractValidationIssue(
                        contract_file=str(proto_file),
                        issue_type="protobuf_syntax",
                        severity="error",
                        message=f"Protobuf compilation failed: {result.stderr}",
                        suggestion="Fix protobuf syntax errors"
                    ))
                    print(f"  ❌ {proto_file.name}: Compilation failed")
                    
            except FileNotFoundError:
                # protoc nicht verfügbar - Skip mit Warning
                warnings += 1
                self.issues.append(ContractValidationIssue(
                    contract_file=str(proto_file),
                    issue_type="protobuf_validation",
                    severity="warning",
                    message="protoc not available - skipping protobuf validation",
                    suggestion="Install Protocol Buffers compiler for validation"
                ))
                print(f"  ⚠️ {proto_file.name}: protoc not available")
        
        return {"passed": passed, "failed": failed, "warnings": warnings}
    
    def _validate_protobuf_architecture_compliance(self, file_path: Path):
        """Prüft Protocol Buffers auf Architektur-Compliance"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Prüfe auf korrekte Package-Trennung
        if "platform" in str(file_path) and "sdk" in content.lower():
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="architecture_compliance",
                severity="error",
                message="Platform protobuf file contains SDK references",
                suggestion="Keep platform and SDK protobuf definitions separate"
            ))
        
        if "sdk" in str(file_path) and "platform" in content.lower():
            self.issues.append(ContractValidationIssue(
                contract_file=str(file_path),
                issue_type="architecture_compliance",
                severity="error",
                message="SDK protobuf file contains platform references",
                suggestion="Keep platform and SDK protobuf definitions separate"
            ))
    
    def _validate_cross_contract_consistency(self) -> Dict[str, int]:
        """Validiert Konsistenz zwischen verschiedenen Contract-Typen"""
        print("🔗 Validiere Cross-Contract Consistency...")
        
        warnings = 0
        failed = 0
        
        # Sammle Event-Typen aus verschiedenen Contracts
        openapi_events = self._extract_openapi_event_types()
        asyncapi_events = self._extract_asyncapi_event_types()
        
        # Prüfe auf Inkonsistenzen
        openapi_only = openapi_events - asyncapi_events
        asyncapi_only = asyncapi_events - openapi_events
        
        if openapi_only:
            warnings += 1
            self.issues.append(ContractValidationIssue(
                contract_file="cross-contract",
                issue_type="consistency",
                severity="warning",
                message=f"Event types in OpenAPI but not AsyncAPI: {', '.join(openapi_only)}",
                suggestion="Ensure event types are consistent across all contract types"
            ))
        
        if asyncapi_only:
            warnings += 1
            self.issues.append(ContractValidationIssue(
                contract_file="cross-contract",
                issue_type="consistency",
                severity="warning",
                message=f"Event types in AsyncAPI but not OpenAPI: {', '.join(asyncapi_only)}",
                suggestion="Ensure event types are consistent across all contract types"
            ))
        
        return {"warnings": warnings, "failed": failed}
    
    def _validate_architecture_compliance(self) -> Dict[str, int]:
        """Validiert allgemeine Architektur-Compliance"""
        print("🏗️ Validiere Architecture Compliance...")
        
        warnings = 0
        failed = 0
        
        # Prüfe auf verbotene direkte Messaging-Referenzen
        all_files = list(self.openapi_files) + list(self.asyncapi_files) + list(self.protobuf_files)
        
        for contract_file in all_files:
            with open(contract_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Verbotene Patterns für direkte Messaging
            forbidden_patterns = [
                "nats://", "jetstream", "direct.messaging", 
                "shared.schema.registry", "platform.nats"
            ]
            
            for pattern in forbidden_patterns:
                if pattern in content:
                    failed += 1
                    self.issues.append(ContractValidationIssue(
                        contract_file=str(contract_file),
                        issue_type="architecture_compliance",
                        severity="error",
                        message=f"Contract contains forbidden pattern: {pattern}",
                        suggestion="Use API-first communication instead of direct messaging"
                    ))
        
        return {"warnings": warnings, "failed": failed}
    
    def _extract_openapi_event_types(self) -> Set[str]:
        """Extrahiert Event-Typen aus OpenAPI Specs"""
        event_types = set()
        
        for openapi_file in self.openapi_files:
            try:
                with open(openapi_file, 'r', encoding='utf-8') as f:
                    spec = yaml.safe_load(f)
                
                # Suche nach Event-Typen in Schemas
                schemas = spec.get("components", {}).get("schemas", {})
                for schema_name, schema in schemas.items():
                    if "event" in schema_name.lower():
                        properties = schema.get("properties", {})
                        event_type_prop = properties.get("event_type", {})
                        if "enum" in event_type_prop:
                            event_types.update(event_type_prop["enum"])
                            
            except Exception:
                continue
        
        return event_types
    
    def _extract_asyncapi_event_types(self) -> Set[str]:
        """Extrahiert Event-Typen aus AsyncAPI Specs"""
        event_types = set()
        
        for asyncapi_file in self.asyncapi_files:
            try:
                with open(asyncapi_file, 'r', encoding='utf-8') as f:
                    spec = yaml.safe_load(f)
                
                # Suche nach Event-Typen in Messages
                channels = spec.get("channels", {})
                for channel_name, channel in channels.items():
                    messages = channel.get("messages", {})
                    for message_name in messages.keys():
                        if "." in message_name:
                            event_types.add(message_name)
                            
            except Exception:
                continue
        
        return event_types
    
    def generate_report(self, report: ContractValidationReport) -> str:
        """Generiert detaillierten Validierungs-Report"""
        
        # Gruppiere Issues nach Severity
        errors = [issue for issue in report.issues if issue.severity == "error"]
        warnings = [issue for issue in report.issues if issue.severity == "warning"]
        
        report_lines = [
            "# API Contract Validation Report",
            f"**Execution Time:** {report.execution_time_ms}ms",
            "",
            "## Summary",
            f"- **Total Contracts:** {report.total_contracts}",
            f"- **Passed:** {report.passed_contracts} ✅",
            f"- **Failed:** {report.failed_contracts} ❌",
            f"- **Warnings:** {report.warnings} ⚠️",
            "",
            f"**Overall Status:** {'✅ PASSED' if report.failed_contracts == 0 else '❌ FAILED'}",
            ""
        ]
        
        if errors:
            report_lines.extend([
                "## ❌ Errors",
                ""
            ])
            for error in errors:
                report_lines.extend([
                    f"### {error.contract_file}",
                    f"**Type:** {error.issue_type}",
                    f"**Message:** {error.message}",
                    f"**Suggestion:** {error.suggestion or 'N/A'}",
                    ""
                ])
        
        if warnings:
            report_lines.extend([
                "## ⚠️ Warnings",
                ""
            ])
            for warning in warnings:
                report_lines.extend([
                    f"### {warning.contract_file}",
                    f"**Type:** {warning.issue_type}",
                    f"**Message:** {warning.message}",
                    f"**Suggestion:** {warning.suggestion or 'N/A'}",
                    ""
                ])
        
        return "\n".join(report_lines)

def main():
    """Hauptfunktion für CLI-Ausführung"""
    validator = APIContractValidator()
    
    print("🔍 Keiko API Contract Validator")
    print("=" * 50)
    
    # Führe Validierung durch
    report = validator.validate_all_contracts()
    
    # Generiere und speichere Report
    report_content = validator.generate_report(report)
    
    with open("api_contract_validation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # Ausgabe Summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"Total Contracts: {report.total_contracts}")
    print(f"Passed: {report.passed_contracts}")
    print(f"Failed: {report.failed_contracts}")
    print(f"Warnings: {report.warnings}")
    print(f"Execution Time: {report.execution_time_ms}ms")
    
    print(f"\n📄 Detailed report saved: api_contract_validation_report.md")
    
    # Exit Code für CI/CD
    if report.failed_contracts > 0:
        print(f"\n❌ VALIDATION FAILED: {report.failed_contracts} contracts failed")
        return 1
    elif report.warnings > 0:
        print(f"\n⚠️ VALIDATION PASSED WITH WARNINGS: {report.warnings} warnings")
        return 0
    else:
        print(f"\n✅ VALIDATION PASSED: All contracts valid")
        return 0

if __name__ == "__main__":
    sys.exit(main())
