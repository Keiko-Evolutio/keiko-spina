#!/usr/bin/env python3
"""
Backend Logging Independence Implementation
Replaces kei_agent.enterprise_logging imports with backend-native logging system
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
import re
from pathlib import Path


class KeikoBackendLogger:
    """Enterprise-Grade Logger für Keiko Backend (independent from SDK)"""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional metadata"""
        if extra:
            self.logger.info(message, extra=extra)
        else:
            self.logger.info(message)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message with optional metadata"""
        if extra:
            self.logger.error(message, extra=extra)
        else:
            self.logger.error(message)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with optional metadata"""
        if extra:
            self.logger.debug(message, extra=extra)
        else:
            self.logger.debug(message)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with optional metadata"""
        if extra:
            self.logger.warning(message, extra=extra)
        else:
            self.logger.warning(message)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message with optional metadata"""
        if extra:
            self.logger.critical(message, extra=extra)
        else:
            self.logger.critical(message)


def get_logger(name: str) -> KeikoBackendLogger:
    """Factory function für Backend Logger (replaces kei_agent.enterprise_logging.get_logger)"""
    return KeikoBackendLogger(name)


class BackendLoggingMigrator:
    """Handles migration of logging imports from SDK to backend-native"""

    def __init__(self, backend_path: str = "backend"):
        self.backend_path = Path(backend_path)
        self.migration_report = {
            "timestamp": datetime.now().isoformat(),
            "files_processed": 0,
            "imports_replaced": 0,
            "issues": []
        }

    def find_problematic_imports(self) -> list:
        """Find all files with kei_agent.enterprise_logging imports"""
        problematic_files = []
        
        for py_file in self.backend_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for problematic imports
                    if "from kei_agent.enterprise_logging import" in content or \
                       "import kei_agent.enterprise_logging" in content:
                        problematic_files.append(py_file)
                        
            except Exception as e:
                self.migration_report["issues"].append(f"Error reading {py_file}: {str(e)}")
        
        return problematic_files

    def replace_logging_imports(self, file_path: Path) -> bool:
        """Replace SDK logging imports with backend-native imports"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace different import patterns
            replacements = [
                # Standard import replacement
                (r"from kei_agent\.enterprise_logging import get_logger",
                 "from kei_logging import get_logger"),
                
                # Alternative import patterns
                (r"import kei_agent\.enterprise_logging as (\w+)",
                 r"import kei_logging as \1"),
                
                # Direct module import
                (r"from kei_agent\.enterprise_logging import (\w+)",
                 r"from kei_logging import \1"),
                
                # Full module import
                (r"import kei_agent\.enterprise_logging",
                 "import kei_logging")
            ]
            
            # Apply replacements
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Check if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
        except Exception as e:
            self.migration_report["issues"].append(f"Error processing {file_path}: {str(e)}")
            return False
        
        return False

    def create_backend_logging_module(self):
        """Create the backend-native kei_logging module"""
        kei_logging_dir = self.backend_path / "kei_logging"
        kei_logging_dir.mkdir(exist_ok=True)
        
        # Create __init__.py with the backend-native logger
        init_file = kei_logging_dir / "__init__.py"
        
        init_content = '''"""
Backend-eigenes Logging-System (unabhängig vom SDK)
Ersetzt alle kei_agent.enterprise_logging Imports
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json


class KeikoBackendLogger:
    """Enterprise-Grade Logger für Keiko Backend"""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.info(message, extra=extra or {})

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.error(message, extra=extra or {})

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.debug(message, extra=extra or {})

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.warning(message, extra=extra or {})

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self.logger.critical(message, extra=extra or {})


def get_logger(name: str) -> KeikoBackendLogger:
    """Factory function für Backend Logger"""
    return KeikoBackendLogger(name)


# Backward compatibility aliases
enterprise_logger = get_logger
create_logger = get_logger
'''
        
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)
        
        print(f"✅ Created backend-native logging module at {init_file}")

    def migrate_all_files(self):
        """Migrate all backend files from SDK logging to backend-native logging"""
        print("🔄 Starting backend logging migration...")
        
        # Create backend-native logging module first
        self.create_backend_logging_module()
        
        # Find problematic files
        problematic_files = self.find_problematic_imports()
        print(f"📁 Found {len(problematic_files)} files with SDK logging imports")
        
        # Process each file
        for file_path in problematic_files:
            print(f"🔧 Processing {file_path}")
            
            if self.replace_logging_imports(file_path):
                self.migration_report["imports_replaced"] += 1
                print(f"   ✅ Replaced imports in {file_path.name}")
            else:
                print(f"   ⚠️  No changes made to {file_path.name}")
            
            self.migration_report["files_processed"] += 1

        # Generate migration report
        self.generate_report()
        
        print("\n🎯 Backend logging migration completed!")
        print(f"   Files processed: {self.migration_report['files_processed']}")
        print(f"   Imports replaced: {self.migration_report['imports_replaced']}")

    def generate_report(self):
        """Generate migration report"""
        report_file = f"backend_logging_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        print(f"📊 Migration report saved to: {report_file}")


def main():
    """Main migration execution"""
    print("🚀 Backend Logging Independence Migration")
    print("=" * 50)
    
    # Check if backend directory exists
    if not Path("backend").exists():
        print("❌ Error: Backend directory not found!")
        print("   Please run this script from the repository root.")
        sys.exit(1)
    
    # Create migrator instance
    migrator = BackendLoggingMigrator("backend")
    
    # Execute migration
    migrator.migrate_all_files()
    
    print("\n✨ Next steps:")
    print("   1. Review migration report for any issues")
    print("   2. Test backend services with new logging")
    print("   3. Run backend tests to ensure compatibility")
    print("   4. Continue with orchestrator service migration")


if __name__ == "__main__":
    main()