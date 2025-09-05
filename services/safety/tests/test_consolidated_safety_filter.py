"""Unit-Tests für ConsolidatedSafetyFilter.

Testet alle Funktionalitäten des konsolidierten Safety-Filters mit
umfassender Coverage und isolierten Test-Cases.
"""

import time

from services.safety import (
    ConsolidatedSafetyFilter,
    SafetyConfig,
    SafetyFilterResult,
    SafetyLevel,
    SafetyViolation,
    ViolationType,
    create_safety_filter,
)


class TestSafetyConfig:
    """Tests für SafetyConfig Klasse."""

    def test_default_config_values(self):
        """Testet Standard-Konfigurationswerte."""
        config = SafetyConfig()

        assert config.MAX_PROMPT_LENGTH == 500
        assert config.MAX_SANITIZED_LENGTH == 400
        assert config.SAFE_THRESHOLD == 0.5
        assert config.KEYWORD_PENALTY == 0.2
        assert config.PATTERN_PENALTY == 0.3
        assert config.LENGTH_PENALTY == 0.1
        assert config.LANGUAGE_PENALTY == 0.1
        assert config.MAX_SUGGESTIONS == 3


class TestSafetyViolation:
    """Tests für SafetyViolation Dataclass."""

    def test_violation_creation(self):
        """Testet Erstellung von SafetyViolation."""
        violation = SafetyViolation(
            violation_type=ViolationType.BLOCKED_KEYWORD,
            description="Test violation",
            confidence=0.9,
            detected_content="test",
            suggested_replacement="[REDACTED]"
        )

        assert violation.violation_type == ViolationType.BLOCKED_KEYWORD
        assert violation.description == "Test violation"
        assert violation.confidence == 0.9
        assert violation.detected_content == "test"
        assert violation.suggested_replacement == "[REDACTED]"


class TestSafetyFilterResult:
    """Tests für SafetyFilterResult Dataclass."""

    def test_result_creation(self):
        """Testet Erstellung von SafetyFilterResult."""
        violations = [
            SafetyViolation(
                violation_type=ViolationType.BLOCKED_KEYWORD,
                description="Test",
                confidence=0.9,
                detected_content="test"
            )
        ]

        result = SafetyFilterResult(
            is_safe=False,
            sanitized_prompt="safe prompt",
            original_prompt="unsafe prompt",
            violations=violations,
            confidence_score=0.3,
            safety_level=SafetyLevel.HIGH_RISK,
            processing_time_ms=10.5
        )

        assert result.is_safe is False
        assert result.sanitized_prompt == "safe prompt"
        assert result.original_prompt == "unsafe prompt"
        assert len(result.violations) == 1
        assert result.confidence_score == 0.3
        assert result.safety_level == SafetyLevel.HIGH_RISK
        assert result.processing_time_ms == 10.5


class TestConsolidatedSafetyFilter:
    """Tests für ConsolidatedSafetyFilter Hauptklasse."""

    def setup_method(self):
        """Setup für jeden Test."""
        self.config = SafetyConfig()
        self.filter = ConsolidatedSafetyFilter(self.config)

    def test_initialization(self):
        """Testet korrekte Initialisierung."""
        assert self.filter.config == self.config
        assert len(self.filter._blocked_keywords) > 0
        assert len(self.filter._suspicious_patterns) > 0
        assert len(self.filter._compiled_patterns) > 0

    def test_safe_prompt_analysis(self):
        """Testet Analyse eines sicheren Prompts."""
        safe_prompt = "A beautiful landscape with mountains and trees"
        result = self.filter.analyze_prompt(safe_prompt)

        assert result.is_safe is True
        assert result.sanitized_prompt == safe_prompt
        assert result.original_prompt == safe_prompt
        assert len(result.violations) == 0
        assert result.confidence_score == 1.0
        assert result.safety_level == SafetyLevel.SAFE
        assert result.processing_time_ms is not None

    def test_blocked_keyword_detection(self):
        """Testet Erkennung blockierter Keywords."""
        unsafe_prompt = "Show me a weapon in action"
        result = self.filter.analyze_prompt(unsafe_prompt)

        assert result.is_safe is False
        assert len(result.violations) > 0
        assert any(v.violation_type == ViolationType.BLOCKED_KEYWORD for v in result.violations)
        assert result.confidence_score < 1.0
        assert result.safety_level != SafetyLevel.SAFE

    def test_suspicious_pattern_detection(self):
        """Testet Erkennung verdächtiger Patterns."""
        unsafe_prompt = "Create a nude realistic person"
        result = self.filter.analyze_prompt(unsafe_prompt)

        assert result.is_safe is False
        assert len(result.violations) > 0
        assert any(v.violation_type == ViolationType.SUSPICIOUS_PATTERN for v in result.violations)

    def test_prompt_length_check(self):
        """Testet Längen-Validierung."""
        long_prompt = "A" * 600  # Über dem Limit von 500
        result = self.filter.analyze_prompt(long_prompt)

        assert result.is_safe is False
        assert any(v.violation_type == ViolationType.PROMPT_TOO_LONG for v in result.violations)

    def test_multiple_language_detection(self):
        """Testet Erkennung mehrerer Sprachen."""
        mixed_prompt = "Create a beautiful Landschaft with trees und Bäume"
        result = self.filter.analyze_prompt(mixed_prompt)

        # Sollte Multiple Languages erkennen
        has_language_violation = any(
            v.violation_type == ViolationType.MULTIPLE_LANGUAGES
            for v in result.violations
        )
        assert has_language_violation

    def test_confidence_score_calculation(self):
        """Testet Confidence Score Berechnung."""
        # Test mit verschiedenen Violation-Typen
        violations = [
            SafetyViolation(ViolationType.BLOCKED_KEYWORD, "test", 0.9, "weapon"),
            SafetyViolation(ViolationType.SUSPICIOUS_PATTERN, "test", 0.8, "pattern")
        ]

        score = self.filter._calculate_confidence_score(violations)
        expected_score = 1.0 - self.config.KEYWORD_PENALTY - self.config.PATTERN_PENALTY
        assert score == expected_score

    def test_safety_level_determination(self):
        """Testet Safety Level Bestimmung."""
        # Test verschiedene Confidence Scores
        assert self.filter._determine_safety_level(0.95, []) == SafetyLevel.SAFE
        assert self.filter._determine_safety_level(0.8, []) == SafetyLevel.LOW_RISK
        assert self.filter._determine_safety_level(0.6, []) == SafetyLevel.MEDIUM_RISK
        assert self.filter._determine_safety_level(0.4, []) == SafetyLevel.HIGH_RISK
        assert self.filter._determine_safety_level(0.2, []) == SafetyLevel.UNSAFE

    def test_prompt_sanitization(self):
        """Testet Prompt-Sanitization."""
        violations = [
            SafetyViolation(
                ViolationType.BLOCKED_KEYWORD,
                "test",
                0.9,
                "weapon",
                "[REDACTED]"
            )
        ]

        unsafe_prompt = "Show me a weapon"
        sanitized = self.filter._sanitize_prompt(unsafe_prompt, violations)

        # Prüfe dass Sanitization stattgefunden hat
        assert "A safe, artistic image of" in sanitized
        # Das Wort "weapon" sollte durch "tool" ersetzt worden sein (via replacement patterns)
        assert "tool" in sanitized or "[REDACTED]" in sanitized

    def test_safe_fallback_prompts(self):
        """Testet sichere Fallback-Prompts."""
        # Test verschiedene Intents
        landscape_fallback = self.filter.get_safe_fallback_prompt("landscape scene")
        assert "landscape" in landscape_fallback.lower()

        garden_fallback = self.filter.get_safe_fallback_prompt("garden view")
        assert "garden" in garden_fallback.lower()

        default_fallback = self.filter.get_safe_fallback_prompt("unknown intent")
        assert len(default_fallback) > 0

    def test_safe_prompt_suggestions(self):
        """Testet sichere Prompt-Vorschläge."""
        unsafe_prompt = "Show me a violent person with weapons"
        suggestions = self.filter.create_safe_prompt_suggestions(unsafe_prompt)

        assert len(suggestions) <= self.config.MAX_SUGGESTIONS
        assert all(len(suggestion) > 0 for suggestion in suggestions)

        # Teste dass Vorschläge sicher sind
        for suggestion in suggestions:
            result = self.filter.analyze_prompt(suggestion)
            assert result.is_safe is True

    def test_core_concept_extraction(self):
        """Testet Kern-Konzept Extraktion."""
        prompt = "A beautiful natural landscape with a person and animals"
        concepts = self.filter._extract_core_concepts(prompt)

        assert "nature" in concepts
        assert "person" in concepts
        assert "animal" in concepts

    def test_multiple_language_detection_logic(self):
        """Testet Multiple Language Detection Logik."""
        # Nur Englisch
        english_only = "The beautiful landscape with trees"
        assert not self.filter._contains_multiple_languages(english_only)

        # Nur Deutsch
        german_only = "Die schöne Landschaft mit Bäumen"
        assert not self.filter._contains_multiple_languages(german_only)

        # Gemischt
        mixed = "The beautiful Landschaft with trees und Bäume"
        assert self.filter._contains_multiple_languages(mixed)

    def test_safety_prefix_addition(self):
        """Testet Hinzufügung von Safety-Präfixen."""
        content_without_prefix = "beautiful landscape"
        result = self.filter._add_safety_prefix(content_without_prefix)
        assert "A safe, artistic image of" in result

        content_with_prefix = "A safe, family-friendly image of landscape"
        result = self.filter._add_safety_prefix(content_with_prefix)
        assert result == content_with_prefix  # Sollte unverändert bleiben


class TestFactoryFunction:
    """Tests für Factory-Funktionen."""

    def test_create_safety_filter_default(self):
        """Testet Factory-Funktion mit Standard-Config."""
        filter_instance = create_safety_filter()

        assert isinstance(filter_instance, ConsolidatedSafetyFilter)
        assert isinstance(filter_instance.config, SafetyConfig)

    def test_create_safety_filter_custom_config(self):
        """Testet Factory-Funktion mit Custom-Config."""
        custom_config = SafetyConfig()
        custom_config.MAX_PROMPT_LENGTH = 1000

        filter_instance = create_safety_filter(custom_config)

        assert filter_instance.config.MAX_PROMPT_LENGTH == 1000


class TestPerformance:
    """Performance-Tests für Safety-Filter."""

    def test_analysis_performance(self):
        """Testet Performance der Prompt-Analyse."""
        filter_instance = create_safety_filter()
        test_prompt = "A beautiful landscape with mountains and trees"

        start_time = time.time()
        result = filter_instance.analyze_prompt(test_prompt)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000

        # Sollte unter 100ms sein
        assert processing_time < 100
        assert result.processing_time_ms is not None
        assert result.processing_time_ms > 0
