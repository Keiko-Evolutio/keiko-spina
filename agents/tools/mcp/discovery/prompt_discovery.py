# backend/kei_mcp/discovery/prompt_discovery.py
"""Prompt-Discovery für KEI-MCP Interface.

Implementiert Prompt-Template-Discovery, Kategorisierung, Versionierung,
Parameter-Substitution und Usage-Tracking für MCP-Server-Prompts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kei_logging import get_logger
from observability import trace_function, trace_span

if TYPE_CHECKING:
    from ..kei_mcp_registry import MCPPromptDefinition, RegisteredMCPServer

logger = get_logger(__name__)


class PromptCategory(str, Enum):
    """Kategorien für Prompt-Klassifizierung."""
    SYSTEM = "system"
    USER_INTERACTION = "user_interaction"
    DATA_PROCESSING = "data_processing"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"
    TEMPLATE = "template"
    CUSTOM = "custom"


class PromptComplexity(str, Enum):
    """Komplexitätsstufen für Prompts."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class PromptStatus(str, Enum):
    """Status eines Prompts."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class PromptParameter:
    """Parameter-Definition für Prompt-Template."""
    name: str
    type: str
    description: str
    required: bool = True
    default_value: Any | None = None
    validation_pattern: str | None = None
    enum_values: list[str] | None = None
    min_length: int | None = None
    max_length: int | None = None

    def validate_value(self, value: Any) -> tuple[bool, str | None]:
        """Validiert Parameter-Wert.

        Args:
            value: Zu validierender Wert

        Returns:
            Tuple (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' ist erforderlich"
            return True, None

        # Typ-Validierung
        if self.type == "string" and not isinstance(value, str):
            return False, f"Parameter '{self.name}' muss ein String sein"
        if self.type == "integer" and not isinstance(value, int):
            return False, f"Parameter '{self.name}' muss eine Ganzzahl sein"
        if self.type == "number" and not isinstance(value, int | float):
            return False, f"Parameter '{self.name}' muss eine Zahl sein"
        if self.type == "boolean" and not isinstance(value, bool):
            return False, f"Parameter '{self.name}' muss ein Boolean sein"

        # String-spezifische Validierungen
        if isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                return False, f"Parameter '{self.name}' muss mindestens {self.min_length} Zeichen haben"
            if self.max_length and len(value) > self.max_length:
                return False, f"Parameter '{self.name}' darf maximal {self.max_length} Zeichen haben"

            if self.validation_pattern and not re.match(self.validation_pattern, value):
                return False, f"Parameter '{self.name}' entspricht nicht dem erwarteten Format"

            if self.enum_values and value not in self.enum_values:
                return False, f"Parameter '{self.name}' muss einer von {self.enum_values} sein"

        return True, None


@dataclass
class PromptMetadata:
    """Erweiterte Metadaten für Prompts."""
    category: PromptCategory
    complexity: PromptComplexity = PromptComplexity.SIMPLE
    tags: set[str] = field(default_factory=set)
    author: str | None = None
    language: str = "en"
    use_cases: list[str] = field(default_factory=list)
    expected_output_type: str | None = None
    estimated_tokens: int | None = None
    model_compatibility: list[str] = field(default_factory=list)
    performance_notes: str | None = None
    safety_level: str = "safe"
    content_rating: str = "general"


@dataclass
class PromptUsageStats:
    """Nutzungsstatistiken für Prompts."""
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    avg_response_time_ms: float = 0.0
    avg_output_tokens: float = 0.0
    last_used: datetime | None = None
    user_ratings: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses

    @property
    def avg_rating(self) -> float:
        """Berechnet durchschnittliche Bewertung."""
        if not self.user_ratings:
            return 0.0
        return sum(self.user_ratings) / len(self.user_ratings)


@dataclass
class DiscoveredPrompt:
    """Entdeckter Prompt mit vollständigen Informationen."""
    id: str
    name: str
    description: str
    server_name: str
    template: str
    parameters: list[PromptParameter] = field(default_factory=list)
    metadata: PromptMetadata = field(default_factory=lambda: PromptMetadata(category=PromptCategory.CUSTOM))
    status: PromptStatus = PromptStatus.ACTIVE
    version: str = "1.0.0"
    usage_stats: PromptUsageStats = field(default_factory=PromptUsageStats)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def substitute_parameters(self, parameter_values: dict[str, Any]) -> tuple[str, list[str]]:
        """Substituiert Parameter im Prompt-Template.

        Args:
            parameter_values: Dictionary mit Parameter-Werten

        Returns:
            Tuple (substituted_prompt, validation_errors)
        """
        errors = []
        substituted = self.template

        # Validiere alle Parameter
        for param in self.parameters:
            value = parameter_values.get(param.name)
            is_valid, error = param.validate_value(value)
            if not is_valid:
                errors.append(error)
                continue

            # Verwende Default-Wert falls verfügbar
            if value is None and param.default_value is not None:
                value = param.default_value

            # Substituiere Parameter im Template
            if value is not None:
                placeholder_patterns = [
                    f"{{{param.name}}}",  # {param_name}
                    f"{{{{ {param.name} }}}}",  # {{ param_name }}
                    f"${param.name}",  # $param_name
                    f"${{{param.name}}}"  # ${param_name}
                ]

                for pattern in placeholder_patterns:
                    substituted = substituted.replace(pattern, str(value))

        return substituted, errors

    def extract_parameters_from_template(self) -> list[str]:
        """Extrahiert Parameter-Namen aus Template.

        Returns:
            Liste der gefundenen Parameter-Namen
        """
        # Verschiedene Parameter-Patterns erkennen
        patterns = [
            r"\{(\w+)\}",  # {param_name}
            r"\{\{\s*(\w+)\s*\}\}",  # {{ param_name }}
            r"\$\{(\w+)\}",  # ${param_name}
            r"\$(\w+)"  # $param_name
        ]

        found_params = set()
        for pattern in patterns:
            matches = re.findall(pattern, self.template)
            found_params.update(matches)

        return list(found_params)

    def matches_criteria(self, criteria: dict[str, Any]) -> bool:
        """Prüft, ob Prompt Suchkriterien erfüllt."""
        # Kategorie-Filter
        if "category" in criteria and self.metadata.category != criteria["category"]:
            return False

        # Tag-Filter
        if "tags" in criteria:
            required_tags = set(criteria["tags"])
            if not required_tags.issubset(self.metadata.tags):
                return False

        # Komplexitäts-Filter
        if "complexity" in criteria and self.metadata.complexity != criteria["complexity"]:
            return False

        # Sprach-Filter
        if "language" in criteria and self.metadata.language != criteria["language"]:
            return False

        # Modell-Kompatibilitäts-Filter
        return not ("model" in criteria and (self.metadata.model_compatibility and criteria["model"] not in self.metadata.model_compatibility))


class PromptAnalyzer:
    """Analyzer für Prompt-Eigenschaften."""

    @staticmethod
    def classify_category(name: str, description: str, template: str) -> PromptCategory:
        """Klassifiziert Prompt-Kategorie."""
        text = f"{name} {description} {template}".lower()

        # Kategorie-Keywords
        category_keywords = {
            PromptCategory.SYSTEM: ["system", "initialize", "setup", "configure"],
            PromptCategory.USER_INTERACTION: ["user", "chat", "conversation", "dialog"],
            PromptCategory.DATA_PROCESSING: ["process", "transform", "parse", "clean"],
            PromptCategory.ANALYSIS: ["analyze", "examine", "evaluate", "assess"],
            PromptCategory.GENERATION: ["generate", "create", "produce", "write"],
            PromptCategory.TRANSLATION: ["translate", "convert", "transform language"],
            PromptCategory.SUMMARIZATION: ["summarize", "summary", "brief", "abstract"],
            PromptCategory.CLASSIFICATION: ["classify", "categorize", "label", "tag"],
            PromptCategory.EXTRACTION: ["extract", "find", "identify", "locate"],
            PromptCategory.CONVERSATION: ["conversation", "chat", "talk", "discuss"],
            PromptCategory.INSTRUCTION: ["instruction", "guide", "how to", "step"],
            PromptCategory.TEMPLATE: ["template", "format", "structure", "pattern"]
        }

        # Finde beste Kategorie-Übereinstimmung
        best_category = PromptCategory.CUSTOM
        best_score = 0

        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    @staticmethod
    def estimate_complexity(template: str, parameters: list[PromptParameter]) -> PromptComplexity:
        """Schätzt Prompt-Komplexität."""
        # Faktoren für Komplexität
        template_length = len(template)
        param_count = len(parameters)
        required_params = sum(1 for p in parameters if p.required)

        # Komplexe Patterns im Template
        complex_patterns = [
            r"\{[^}]+\|[^}]+\}",  # Conditional logic
            r"if\s+\w+",  # If statements
            r"for\s+\w+",  # Loops
            r"\[\[.*?\]\]",  # Complex placeholders
        ]

        complex_pattern_count = sum(
            len(re.findall(pattern, template)) for pattern in complex_patterns
        )

        # Komplexitäts-Score berechnen
        complexity_score = (
            template_length / 100 +
            param_count * 2 +
            required_params * 3 +
            complex_pattern_count * 5
        )

        if complexity_score < 5:
            return PromptComplexity.SIMPLE
        if complexity_score < 15:
            return PromptComplexity.MODERATE
        if complexity_score < 30:
            return PromptComplexity.COMPLEX
        return PromptComplexity.ADVANCED

    @staticmethod
    def extract_tags(name: str, description: str, template: str) -> set[str]:
        """Extrahiert Tags aus Prompt-Informationen."""
        tags = set()

        # Keyword-basierte Tag-Extraktion
        keywords = [
            "json", "xml", "csv", "markdown", "html", "code",
            "creative", "technical", "business", "educational",
            "research", "analysis", "report", "email", "letter"
        ]

        text = f"{name} {description} {template}".lower()
        for keyword in keywords:
            if keyword in text:
                tags.add(keyword)

        return tags


class PromptDiscoveryEngine:
    """Engine für Prompt-Discovery und -Management."""

    def __init__(self) -> None:
        """Initialisiert Prompt-Discovery-Engine."""
        self._discovered_prompts: dict[str, DiscoveredPrompt] = {}
        self._prompts_by_server: dict[str, list[str]] = {}
        self._prompts_by_category: dict[PromptCategory, list[str]] = {}
        self._usage_analytics: dict[str, list[dict[str, Any]]] = {}
        self._last_discovery_run: datetime | None = None
        self._discovery_interval_seconds = 900  # 15 Minuten

    @trace_function("mcp.prompt_discovery.discover_all")
    async def discover_all_prompts(
        self,
        servers: dict[str, RegisteredMCPServer],
        force_refresh: bool = False
    ) -> list[DiscoveredPrompt]:
        """Entdeckt alle Prompts von registrierten MCP-Servern.

        Args:
            servers: Dictionary der registrierten Server
            force_refresh: Erzwingt Neuentdeckung

        Returns:
            Liste aller entdeckten Prompts
        """
        current_time = datetime.now(UTC)

        # Prüfe, ob Discovery erforderlich ist
        if (not force_refresh and
            self._last_discovery_run and
            (current_time - self._last_discovery_run).total_seconds() < self._discovery_interval_seconds):
            return list(self._discovered_prompts.values())

        logger.info(f"Starte Prompt-Discovery für {len(servers)} Server")

        discovered_prompts = []

        for server_name, server in servers.items():
            if not server.is_healthy:
                logger.warning(f"Server {server_name} ist nicht gesund, überspringe Prompt-Discovery")
                continue

            try:
                server_prompts = await self._discover_server_prompts(server_name, server)
                discovered_prompts.extend(server_prompts)

                # Server-Mapping aktualisieren
                self._prompts_by_server[server_name] = [prompt.id for prompt in server_prompts]

            except Exception as e:
                logger.exception(f"Prompt-Discovery für Server {server_name} fehlgeschlagen: {e}")

        # Prompts nach Kategorie organisieren
        self._organize_prompts_by_category(discovered_prompts)

        # Discovery-Cache aktualisieren
        self._discovered_prompts = {prompt.id: prompt for prompt in discovered_prompts}
        self._last_discovery_run = current_time

        logger.info(f"Prompt-Discovery abgeschlossen: {len(discovered_prompts)} Prompts entdeckt")
        return discovered_prompts

    async def _discover_server_prompts(
        self,
        server_name: str,
        server: RegisteredMCPServer
    ) -> list[DiscoveredPrompt]:
        """Entdeckt Prompts eines spezifischen Servers."""
        prompts = []

        with trace_span("mcp.prompt_discovery.server", {"server": server_name}):
            for mcp_prompt in server.available_prompts:
                try:
                    discovered_prompt = await self._convert_mcp_prompt_to_discovered(
                        mcp_prompt, server_name
                    )

                    prompts.append(discovered_prompt)

                except Exception as e:
                    logger.exception(f"Fehler beim Verarbeiten von Prompt {mcp_prompt.name}: {e}")

        return prompts

    async def _convert_mcp_prompt_to_discovered(
        self,
        mcp_prompt: MCPPromptDefinition,
        server_name: str
    ) -> DiscoveredPrompt:
        """Konvertiert MCP-Prompt zu DiscoveredPrompt."""
        # Prompt-ID generieren
        prompt_id = f"{server_name}:{mcp_prompt.name}"

        # Template aus MCP-Prompt extrahieren (vereinfacht)
        template = mcp_prompt.description  # In echter Implementierung würde Template separat geladen

        # Parameter aus MCP-Prompt extrahieren
        parameters = self._extract_parameters_from_mcp_prompt(mcp_prompt)

        # Metadaten ableiten
        metadata = self._extract_prompt_metadata(mcp_prompt, template)

        return DiscoveredPrompt(
            id=prompt_id,
            name=mcp_prompt.name,
            description=mcp_prompt.description,
            server_name=server_name,
            template=template,
            parameters=parameters,
            metadata=metadata,
            version=mcp_prompt.version
        )

    def _extract_parameters_from_mcp_prompt(self, mcp_prompt: MCPPromptDefinition) -> list[PromptParameter]:
        """Extrahiert Parameter aus MCP-Prompt."""
        parameters = []

        if mcp_prompt.parameters and isinstance(mcp_prompt.parameters, dict):
            for param_name, param_def in mcp_prompt.parameters.items():
                if isinstance(param_def, dict):
                    parameter = PromptParameter(
                        name=param_name,
                        type=param_def.get("type", "string"),
                        description=param_def.get("description", ""),
                        required=param_def.get("required", True),
                        default_value=param_def.get("default"),
                        validation_pattern=param_def.get("pattern"),
                        enum_values=param_def.get("enum"),
                        min_length=param_def.get("minLength"),
                        max_length=param_def.get("maxLength")
                    )
                    parameters.append(parameter)

        return parameters

    def _extract_prompt_metadata(self, mcp_prompt: MCPPromptDefinition, template: str) -> PromptMetadata:
        """Extrahiert Metadaten aus MCP-Prompt."""
        # Kategorie klassifizieren
        category = PromptAnalyzer.classify_category(
            mcp_prompt.name, mcp_prompt.description, template
        )

        # Komplexität schätzen
        parameters = self._extract_parameters_from_mcp_prompt(mcp_prompt)
        complexity = PromptAnalyzer.estimate_complexity(template, parameters)

        # Tags extrahieren
        tags = PromptAnalyzer.extract_tags(mcp_prompt.name, mcp_prompt.description, template)

        # Tags aus MCP-Prompt hinzufügen
        if mcp_prompt.tags:
            tags.update(mcp_prompt.tags)

        return PromptMetadata(
            category=category,
            complexity=complexity,
            tags=tags,
            language="en",  # Default, könnte aus Prompt erkannt werden
            use_cases=[],
            model_compatibility=[]
        )

    def _organize_prompts_by_category(self, prompts: list[DiscoveredPrompt]) -> None:
        """Organisiert Prompts nach Kategorien."""
        self._prompts_by_category.clear()

        for prompt in prompts:
            category = prompt.metadata.category
            if category not in self._prompts_by_category:
                self._prompts_by_category[category] = []
            self._prompts_by_category[category].append(prompt.id)

    # Public Interface Methods

    def get_prompt_by_id(self, prompt_id: str) -> DiscoveredPrompt | None:
        """Gibt Prompt anhand ID zurück."""
        return self._discovered_prompts.get(prompt_id)

    def get_prompts_by_server(self, server_name: str) -> list[DiscoveredPrompt]:
        """Gibt alle Prompts eines Servers zurück."""
        prompt_ids = self._prompts_by_server.get(server_name, [])
        return [self._discovered_prompts[pid] for pid in prompt_ids
                if pid in self._discovered_prompts]

    def get_prompts_by_category(self, category: PromptCategory) -> list[DiscoveredPrompt]:
        """Gibt alle Prompts einer Kategorie zurück."""
        prompt_ids = self._prompts_by_category.get(category, [])
        return [self._discovered_prompts[pid] for pid in prompt_ids
                if pid in self._discovered_prompts]

    def search_prompts(self, criteria: dict[str, Any]) -> list[DiscoveredPrompt]:
        """Sucht Prompts nach Kriterien."""
        matching_prompts = []

        for prompt in self._discovered_prompts.values():
            if prompt.matches_criteria(criteria):
                matching_prompts.append(prompt)

        # Sortiere nach Erfolgsrate und Bewertung
        matching_prompts.sort(
            key=lambda p: (p.usage_stats.success_rate, p.usage_stats.avg_rating),
            reverse=True
        )

        return matching_prompts

    def record_prompt_usage(
        self,
        prompt_id: str,
        success: bool,
        response_time_ms: float,
        output_tokens: int | None = None,
        user_rating: float | None = None
    ) -> None:
        """Zeichnet Prompt-Nutzung auf."""
        prompt = self.get_prompt_by_id(prompt_id)
        if not prompt:
            return

        stats = prompt.usage_stats
        stats.total_uses += 1

        if success:
            stats.successful_uses += 1
        else:
            stats.failed_uses += 1

        # Aktualisiere durchschnittliche Antwortzeit
        if stats.total_uses == 1:
            stats.avg_response_time_ms = response_time_ms
        else:
            stats.avg_response_time_ms = (
                (stats.avg_response_time_ms * (stats.total_uses - 1) + response_time_ms) /
                stats.total_uses
            )

        # Aktualisiere durchschnittliche Output-Tokens
        if output_tokens is not None:
            if stats.total_uses == 1:
                stats.avg_output_tokens = output_tokens
            else:
                stats.avg_output_tokens = (
                    (stats.avg_output_tokens * (stats.total_uses - 1) + output_tokens) /
                    stats.total_uses
                )

        # Füge Bewertung hinzu
        if user_rating is not None:
            stats.user_ratings.append(user_rating)

        stats.last_used = datetime.now(UTC)

        # Analytics-Event aufzeichnen
        if prompt_id not in self._usage_analytics:
            self._usage_analytics[prompt_id] = []

        self._usage_analytics[prompt_id].append({
            "timestamp": time.time(),
            "success": success,
            "response_time_ms": response_time_ms,
            "output_tokens": output_tokens,
            "user_rating": user_rating
        })

    def get_usage_analytics(self, prompt_id: str) -> list[dict[str, Any]]:
        """Gibt Nutzungsanalytics für Prompt zurück."""
        return self._usage_analytics.get(prompt_id, [])


# Globale Prompt-Discovery-Engine
prompt_discovery_engine = PromptDiscoveryEngine()
