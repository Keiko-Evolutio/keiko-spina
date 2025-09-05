# backend/services/llm/prompt_templates.py
"""Prompt Template Management System für LLM Client.

Implementiert versioniertes Template-System mit Jinja2-Integration
für strukturierte und wiederverwendbare Prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, TemplateError

from kei_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Prompt Template Datenmodell."""

    name: str
    version: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def render(self, **kwargs: Any) -> str:
        """Rendert Template mit gegebenen Variablen.

        Args:
            **kwargs: Template-Variablen

        Returns:
            Gerenderte Prompt-String

        Raises:
            TemplateError: Bei Rendering-Fehlern
        """
        try:
            jinja_template = Template(self.template)
            return jinja_template.render(**kwargs)
        except TemplateError as e:
            logger.error({
                "event": "template_render_failed",
                "template_name": self.name,
                "version": self.version,
                "error": str(e)
            })
            raise

    def validate_variables(self, variables: dict[str, Any]) -> list[str]:
        """Validiert ob alle erforderlichen Variablen vorhanden sind.

        Args:
            variables: Zu validierende Variablen

        Returns:
            Liste fehlender Variablen
        """
        missing = []
        for var in self.variables:
            if var not in variables:
                missing.append(var)
        return missing


@dataclass
class PromptTemplateConfig:
    """Konfiguration für Template Manager."""

    templates_directory: str = "templates/prompts"
    default_version: str = "1.0.0"
    enable_caching: bool = True
    auto_reload: bool = False


class PromptTemplateManager:
    """Manager für Prompt Templates mit Versionierung."""

    def __init__(self, config: PromptTemplateConfig):
        """Initialisiert Template Manager.

        Args:
            config: Manager-Konfiguration
        """
        self.config = config
        self._templates: dict[str, dict[str, PromptTemplate]] = {}
        self._jinja_env: Environment | None = None

        # Templates-Verzeichnis erstellen falls nicht vorhanden
        self.templates_path = Path(config.templates_directory)
        self.templates_path.mkdir(parents=True, exist_ok=True)

        logger.info({
            "event": "template_manager_initialized",
            "templates_directory": config.templates_directory,
            "caching_enabled": config.enable_caching
        })

    def _ensure_jinja_env(self) -> Environment:
        """Initialisiert Jinja2 Environment lazily."""
        if self._jinja_env is None:
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_path)),
                autoescape=False,
                auto_reload=self.config.auto_reload
            )
        return self._jinja_env

    async def load_templates(self) -> None:
        """Lädt alle Templates aus dem Templates-Verzeichnis."""
        try:
            for template_file in self.templates_path.glob("*.json"):
                await self._load_template_file(template_file)

            logger.info({
                "event": "templates_loaded",
                "total_templates": sum(len(versions) for versions in self._templates.values()),
                "unique_templates": len(self._templates)
            })
        except Exception as e:
            logger.error(f"Fehler beim Laden der Templates: {e}")

    async def _load_template_file(self, file_path: Path) -> None:
        """Lädt einzelne Template-Datei."""
        try:
            with open(file_path, encoding="utf-8") as f:
                template_data = json.load(f)

            template = PromptTemplate(
                name=template_data["name"],
                version=template_data["version"],
                template=template_data["template"],
                description=template_data.get("description", ""),
                variables=template_data.get("variables", []),
                metadata=template_data.get("metadata", {}),
                created_at=datetime.fromisoformat(template_data.get("created_at", datetime.utcnow().isoformat()))
            )

            # Template in Cache speichern
            if template.name not in self._templates:
                self._templates[template.name] = {}

            self._templates[template.name][template.version] = template

            logger.debug({
                "event": "template_loaded",
                "name": template.name,
                "version": template.version,
                "file": str(file_path)
            })

        except Exception as e:
            logger.error({
                "event": "template_load_failed",
                "file": str(file_path),
                "error": str(e)
            })

    def get_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Holt Template nach Name und Version.

        Args:
            name: Template-Name
            version: Template-Version (optional, verwendet default)

        Returns:
            Template oder None falls nicht gefunden
        """
        if name not in self._templates:
            logger.warning({
                "event": "template_not_found",
                "name": name,
                "version": version
            })
            return None

        versions = self._templates[name]

        if version is None:
            # Verwende neueste Version
            version = max(versions.keys())

        if version not in versions:
            logger.warning({
                "event": "template_version_not_found",
                "name": name,
                "version": version,
                "available_versions": list(versions.keys())
            })
            return None

        return versions[version]

    def list_templates(self) -> list[dict[str, Any]]:
        """Listet alle verfügbaren Templates auf.

        Returns:
            Liste mit Template-Informationen
        """
        templates_info = []

        for name, versions in self._templates.items():
            for version, template in versions.items():
                templates_info.append({
                    "name": name,
                    "version": version,
                    "description": template.description,
                    "variables": template.variables,
                    "created_at": template.created_at.isoformat()
                })

        return templates_info

    async def save_template(self, template: PromptTemplate) -> None:
        """Speichert Template in Datei und Cache.

        Args:
            template: Zu speicherndes Template
        """
        try:
            # Template-Daten für JSON-Serialisierung vorbereiten
            template_data = {
                "name": template.name,
                "version": template.version,
                "template": template.template,
                "description": template.description,
                "variables": template.variables,
                "metadata": template.metadata,
                "created_at": template.created_at.isoformat()
            }

            # Dateiname erstellen
            filename = f"{template.name}_v{template.version.replace('.', '_')}.json"
            file_path = self.templates_path / filename

            # Template in Datei speichern
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)

            # Template in Cache speichern
            if template.name not in self._templates:
                self._templates[template.name] = {}

            self._templates[template.name][template.version] = template

            logger.info({
                "event": "template_saved",
                "name": template.name,
                "version": template.version,
                "file": str(file_path)
            })

        except Exception as e:
            logger.error({
                "event": "template_save_failed",
                "name": template.name,
                "version": template.version,
                "error": str(e)
            })
            raise

    def render_template(self, name: str, variables: dict[str, Any], version: str | None = None) -> str:
        """Rendert Template mit gegebenen Variablen.

        Args:
            name: Template-Name
            variables: Template-Variablen
            version: Template-Version (optional)

        Returns:
            Gerenderte Prompt-String

        Raises:
            ValueError: Falls Template nicht gefunden
            TemplateError: Bei Rendering-Fehlern
        """
        template = self.get_template(name, version)
        if template is None:
            raise ValueError(f"Template '{name}' (Version: {version}) nicht gefunden")

        # Validiere erforderliche Variablen
        missing_vars = template.validate_variables(variables)
        if missing_vars:
            raise ValueError(f"Fehlende Template-Variablen: {missing_vars}")

        return template.render(**variables)


# Vordefinierte Templates für Task Decomposition
TASK_DECOMPOSITION_TEMPLATES = {
    "task_analysis": {
        "name": "task_analysis",
        "version": "1.0.0",
        "template": """Analyze the following task for intelligent decomposition:

Task: {{ task_description }}
Context: {{ context }}
Constraints: {{ constraints }}
SLA Requirements: {{ sla_requirements }}

Please provide a detailed analysis in JSON format with:
1. Task complexity assessment (1-10 scale)
2. Required capabilities and skills
3. Potential subtasks and dependencies
4. Risk factors and failure scenarios
5. Estimated execution time and resources
6. Success criteria and validation steps

Format your response as structured JSON.""",
        "description": "Template für LLM-basierte Task-Analyse",
        "variables": ["task_description", "context", "constraints", "sla_requirements"]
    },

    "task_decomposition": {
        "name": "task_decomposition",
        "version": "1.0.0",
        "template": """Based on the task analysis and available agents, decompose the task into executable subtasks:

Task Analysis:
- Complexity: {{ complexity_score }}/10
- Required Capabilities: {{ required_capabilities }}
- Potential Subtasks: {{ potential_subtasks }}
- Dependencies: {{ dependencies }}

Available Agents and Capabilities:
{{ agent_capabilities }}

Create an optimal decomposition that:
1. Maximizes parallelization where possible
2. Respects dependencies and constraints
3. Assigns subtasks to most suitable agents
4. Includes error handling and rollback steps
5. Defines clear success criteria for each subtask

Provide the decomposition as structured JSON with:
- subtasks: List of executable subtasks
- dependencies: Task dependency graph
- agent_assignments: Recommended agent for each subtask
- execution_order: Optimal execution sequence
- rollback_plan: Compensation actions for failures""",
        "description": "Template für LLM-basierte Task-Dekomposition",
        "variables": ["complexity_score", "required_capabilities", "potential_subtasks", "dependencies", "agent_capabilities"]
    }
}
