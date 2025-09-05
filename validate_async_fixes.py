#!/usr/bin/env python3
"""
Validierung der async/await-Korrekturen.

Überprüft, ob alle korrigierten async/await-Aufrufe syntaktisch korrekt sind
und keine offensichtlichen Probleme aufweisen.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class AsyncCallValidator(ast.NodeVisitor):
    """AST-Visitor zur Validierung von async/await-Aufrufen."""
    
    def __init__(self):
        self.issues: List[Tuple[int, str]] = []
        self.current_function_is_async = False
        self.function_stack = []
    
    def visit_FunctionDef(self, node):
        """Besucht Funktionsdefinitionen."""
        self.function_stack.append(self.current_function_is_async)
        self.current_function_is_async = False
        self.generic_visit(node)
        self.current_function_is_async = self.function_stack.pop()
    
    def visit_AsyncFunctionDef(self, node):
        """Besucht async Funktionsdefinitionen."""
        self.function_stack.append(self.current_function_is_async)
        self.current_function_is_async = True
        self.generic_visit(node)
        self.current_function_is_async = self.function_stack.pop()
    
    def visit_Await(self, node):
        """Besucht await-Ausdrücke."""
        if not self.current_function_is_async:
            self.issues.append((
                node.lineno,
                "await verwendet außerhalb einer async-Funktion"
            ))
        
        # Prüfe spezifische Patterns
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                attr_name = node.value.func.attr
                if attr_name == "abort" and isinstance(node.value.func.value, ast.Name):
                    if node.value.func.value.id == "context":
                        print(f"✅ Zeile {node.lineno}: Korrekter await context.abort() Aufruf")
                elif attr_name == "record_attempt":
                    print(f"✅ Zeile {node.lineno}: Korrekter await record_attempt() Aufruf")
                elif attr_name == "handle_function_confirmation":
                    print(f"✅ Zeile {node.lineno}: Korrekter await handle_function_confirmation() Aufruf")
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Besucht Funktionsaufrufe."""
        # Prüfe auf potentiell vergessene awaits
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name == "abort" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "context":
                    self.issues.append((
                        node.lineno,
                        "Möglicher fehlender await bei context.abort()"
                    ))
        
        self.generic_visit(node)


def validate_file(file_path: Path) -> Tuple[bool, List[Tuple[int, str]]]:
    """Validiert eine Python-Datei auf async/await-Probleme."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        validator = AsyncCallValidator()
        validator.visit(tree)
        
        return len(validator.issues) == 0, validator.issues
    
    except SyntaxError as e:
        return False, [(e.lineno or 0, f"Syntaxfehler: {e.msg}")]
    except Exception as e:
        return False, [(0, f"Fehler beim Parsen: {e}")]


def main():
    """Hauptfunktion zur Validierung aller korrigierten Dateien."""
    files_to_check = [
        "backend/api/grpc/kei_rpc_service.py",
        "backend/api/middleware/enterprise_websocket_auth.py", 
        "backend/services/streaming/manager.py"
    ]
    
    print("🔍 Validierung der async/await-Korrekturen\n")
    
    all_valid = True
    
    for file_path_str in files_to_check:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"❌ Datei nicht gefunden: {file_path}")
            all_valid = False
            continue
        
        print(f"📁 Prüfe: {file_path}")
        is_valid, issues = validate_file(file_path)
        
        if is_valid:
            print(f"✅ {file_path}: Alle async/await-Aufrufe korrekt")
        else:
            print(f"❌ {file_path}: {len(issues)} Problem(e) gefunden:")
            for line_no, issue in issues:
                print(f"   Zeile {line_no}: {issue}")
            all_valid = False
        
        print()
    
    if all_valid:
        print("🎉 Alle Korrekturen erfolgreich validiert!")
        return 0
    else:
        print("⚠️  Einige Probleme gefunden. Bitte überprüfen.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
