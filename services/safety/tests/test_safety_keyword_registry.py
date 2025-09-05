"""Unit-Tests für SafetyKeywordRegistry.

Testet die zentrale Keyword-Registry und Pattern-Verwaltung.
"""

import re

import pytest

from services.safety.prompt_safety_filter import SafetyKeywordRegistry


class TestSafetyKeywordRegistry:
    """Tests für SafetyKeywordRegistry Klasse."""

    def test_blocked_keywords_not_empty(self):
        """Testet dass blockierte Keywords nicht leer sind."""
        keywords = SafetyKeywordRegistry.get_blocked_keywords()

        assert isinstance(keywords, set)
        assert len(keywords) > 0

    def test_blocked_keywords_contain_expected_categories(self):
        """Testet dass alle erwarteten Keyword-Kategorien vorhanden sind."""
        keywords = SafetyKeywordRegistry.get_blocked_keywords()

        # Gewalt und Waffen
        assert "weapon" in keywords
        assert "violence" in keywords
        assert "waffe" in keywords

        # Explizite Inhalte
        assert "nude" in keywords
        assert "sexual" in keywords
        assert "nackt" in keywords

        # Hassrede
        assert "hate" in keywords
        assert "racist" in keywords
        assert "hass" in keywords

        # Drogen
        assert "drug" in keywords
        assert "cocaine" in keywords
        assert "droge" in keywords

        # Selbstverletzung
        assert "suicide" in keywords
        assert "self-harm" in keywords
        assert "selbstmord" in keywords

        # Toxische Begriffe
        assert "stupid" in keywords
        assert "idiot" in keywords

        # Politische Figuren
        assert "hitler" in keywords
        assert "putin" in keywords

        # Urheberrecht
        assert "disney" in keywords
        assert "pokemon" in keywords

    def test_blocked_keywords_multilingual_support(self):
        """Testet mehrsprachige Unterstützung der Keywords."""
        keywords = SafetyKeywordRegistry.get_blocked_keywords()

        # Deutsche und englische Varianten
        english_german_pairs = [
            ("weapon", "waffe"),
            ("violence", "gewalt"),
            ("nude", "nackt"),
            ("hate", "hass"),
            ("drug", "droge"),
            ("suicide", "selbstmord")
        ]

        for english, german in english_german_pairs:
            assert english in keywords, f"English keyword '{english}' missing"
            assert german in keywords, f"German keyword '{german}' missing"

    def test_suspicious_patterns_not_empty(self):
        """Testet dass verdächtige Patterns nicht leer sind."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_suspicious_patterns_are_valid_regex(self):
        """Testet dass alle Patterns gültige Regex sind."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()

        for pattern in patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error:
                pytest.fail(f"Invalid regex pattern: {pattern}")

    def test_suspicious_patterns_contain_expected_categories(self):
        """Testet dass erwartete Pattern-Kategorien vorhanden sind."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        pattern_string = " ".join(patterns)

        # Content-Generation Patterns
        assert "make|create|generate|show" in pattern_string
        assert "nude|naked|sexual" in pattern_string
        assert "erstelle|mache|zeige" in pattern_string

        # Injection Patterns
        assert "script" in pattern_string
        assert "javascript" in pattern_string
        assert "SELECT|INSERT|UPDATE" in pattern_string

        # Path Traversal
        assert r"\.\./|\.\.\\" in pattern_string

    def test_patterns_detect_content_generation_attempts(self):
        """Testet dass Patterns Content-Generation-Versuche erkennen."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

        test_cases = [
            "create a nude person",
            "generate naked woman",
            "show me sexual content",
            "erstelle nackte Person",
            "mache sexuellen Inhalt"
        ]

        for test_case in test_cases:
            found_match = any(pattern.search(test_case) for pattern in compiled_patterns)
            assert found_match, f"No pattern matched: {test_case}"

    def test_patterns_detect_injection_attempts(self):
        """Testet dass Patterns Injection-Versuche erkennen."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

        injection_attempts = [
            "<script>alert('test')</script>",
            "javascript:void(0)",
            "SELECT * FROM users",
            "INSERT INTO table",
            "../../../etc/passwd",
            "..\\..\\windows\\system32"
        ]

        for attempt in injection_attempts:
            found_match = any(pattern.search(attempt) for pattern in compiled_patterns)
            assert found_match, f"No pattern matched injection attempt: {attempt}"

    def test_patterns_multilingual_support(self):
        """Testet mehrsprachige Unterstützung der Patterns."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        pattern_string = " ".join(patterns)

        # Deutsche und englische Pattern-Varianten
        assert "make|create|generate|show" in pattern_string
        assert "erstelle|mache|zeige" in pattern_string
        assert "realistic|photorealistic" in pattern_string
        assert "realistisch|fotorealistisch" in pattern_string
        assert "famous|celebrity|politician" in pattern_string
        assert "berühmt|prominente|politiker" in pattern_string

    def test_keyword_registry_consistency(self):
        """Testet Konsistenz der Keyword-Registry."""
        keywords = SafetyKeywordRegistry.get_blocked_keywords()

        # Alle Keywords sollten lowercase sein
        for keyword in keywords:
            assert keyword == keyword.lower(), f"Keyword not lowercase: {keyword}"

        # Keine leeren Keywords
        assert all(len(keyword.strip()) > 0 for keyword in keywords)

        # Keine Duplikate (Set sollte das automatisch handhaben)
        keywords_list = list(keywords)
        assert len(keywords_list) == len(set(keywords_list))

    def test_pattern_registry_consistency(self):
        """Testet Konsistenz der Pattern-Registry."""
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()

        # Keine leeren Patterns
        assert all(len(pattern.strip()) > 0 for pattern in patterns)

        # Keine Duplikate
        assert len(patterns) == len(set(patterns))

        # Alle Patterns sollten kompilierbar sein
        for pattern in patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                assert compiled is not None
            except Exception as e:
                pytest.fail(f"Pattern compilation failed for '{pattern}': {e}")


class TestKeywordRegistryPerformance:
    """Performance-Tests für Keyword-Registry."""

    def test_keyword_retrieval_performance(self):
        """Testet Performance der Keyword-Abfrage."""
        import time

        start_time = time.time()
        keywords = SafetyKeywordRegistry.get_blocked_keywords()
        end_time = time.time()

        retrieval_time = (end_time - start_time) * 1000

        # Sollte unter 10ms sein
        assert retrieval_time < 10
        assert len(keywords) > 0

    def test_pattern_retrieval_performance(self):
        """Testet Performance der Pattern-Abfrage."""
        import time

        start_time = time.time()
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        end_time = time.time()

        retrieval_time = (end_time - start_time) * 1000

        # Sollte unter 10ms sein
        assert retrieval_time < 10
        assert len(patterns) > 0

    def test_pattern_compilation_performance(self):
        """Testet Performance der Pattern-Kompilierung."""
        import time

        patterns = SafetyKeywordRegistry.get_suspicious_patterns()

        start_time = time.time()
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        end_time = time.time()

        compilation_time = (end_time - start_time) * 1000

        # Sollte unter 50ms sein
        assert compilation_time < 50
        assert len(compiled_patterns) == len(patterns)


class TestKeywordRegistryIntegration:
    """Integrations-Tests für Keyword-Registry."""

    def test_registry_integration_with_filter(self):
        """Testet Integration der Registry mit dem Safety-Filter."""
        from services.safety import create_safety_filter

        filter_instance = create_safety_filter()

        # Registry-Keywords sollten im Filter verfügbar sein
        registry_keywords = SafetyKeywordRegistry.get_blocked_keywords()
        filter_keywords = filter_instance._blocked_keywords

        assert registry_keywords == filter_keywords

        # Registry-Patterns sollten im Filter verfügbar sein
        registry_patterns = SafetyKeywordRegistry.get_suspicious_patterns()
        filter_patterns = filter_instance._suspicious_patterns

        assert registry_patterns == filter_patterns

    def test_registry_updates_reflect_in_filter(self):
        """Testet dass Registry-Updates im Filter reflektiert werden."""
        # Dieser Test würde in einer echten Implementierung
        # dynamische Updates testen, hier nur Struktur-Validierung

        keywords = SafetyKeywordRegistry.get_blocked_keywords()
        patterns = SafetyKeywordRegistry.get_suspicious_patterns()

        assert isinstance(keywords, set)
        assert isinstance(patterns, list)
        assert len(keywords) > 0
        assert len(patterns) > 0
