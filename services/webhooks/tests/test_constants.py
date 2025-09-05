"""Tests für das Constants-Modul."""

from unittest.mock import patch

import pytest

from services.webhooks.constants import (
    HTTP_REQUEST_TIMEOUT_SECONDS,
    IDEMPOTENCY_TTL_SECONDS,
    # TTL-Konstanten
    NONCE_TTL_SECONDS,
    # Redis-Key-Präfixe
    REDIS_KEY_PREFIX,
    REDIS_NONCE_PREFIX,
    # Timeout-Konstanten
    SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS,
    WORKER_SHUTDOWN_TIMEOUT_SECONDS,
    get_redis_key,
    # Utility-Funktionen
    get_tenant_normalized,
)


class TestConstants:
    """Tests für Konstanten-Definitionen."""

    def test_timeout_constants_have_reasonable_defaults(self):
        """Testet, dass Timeout-Konstanten vernünftige Standardwerte haben."""
        assert SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS >= 60  # Mindestens 1 Minute
        assert HTTP_REQUEST_TIMEOUT_SECONDS >= 5.0  # Mindestens 5 Sekunden
        assert WORKER_SHUTDOWN_TIMEOUT_SECONDS >= 5.0  # Mindestens 5 Sekunden

    def test_ttl_constants_have_reasonable_defaults(self):
        """Testet, dass TTL-Konstanten vernünftige Standardwerte haben."""
        assert NONCE_TTL_SECONDS >= 60  # Mindestens 1 Minute
        assert IDEMPOTENCY_TTL_SECONDS >= 300  # Mindestens 5 Minuten

    def test_redis_key_prefixes_are_consistent(self):
        """Testet, dass Redis-Key-Präfixe konsistent sind."""
        assert REDIS_KEY_PREFIX == "kei:webhook"
        assert REDIS_NONCE_PREFIX.startswith(REDIS_KEY_PREFIX)
        assert "nonce" in REDIS_NONCE_PREFIX

    @patch("config.settings.settings")
    def test_constants_use_settings_when_available(self, mock_settings):
        """Testet, dass Konstanten Settings verwenden wenn verfügbar."""
        # Mock Settings mit benutzerdefinierten Werten
        mock_settings.webhook_signature_tolerance_seconds = 600
        mock_settings.webhook_http_timeout_seconds = 45.0

        # Reimport um neue Settings zu laden
        import importlib

        from services.webhooks import constants
        importlib.reload(constants)

        assert constants.SIGNATURE_TIMESTAMP_TOLERANCE_SECONDS == 600
        assert constants.HTTP_REQUEST_TIMEOUT_SECONDS == 45.0


class TestUtilityFunctions:
    """Tests für Utility-Funktionen."""

    def test_get_tenant_normalized_with_valid_tenant(self):
        """Testet Tenant-Normalisierung mit gültiger Tenant-ID."""
        result = get_tenant_normalized("tenant123")
        assert result == "tenant123"

    def test_get_tenant_normalized_with_none(self):
        """Testet Tenant-Normalisierung mit None."""
        result = get_tenant_normalized(None)
        assert result == "default"

    def test_get_tenant_normalized_with_empty_string(self):
        """Testet Tenant-Normalisierung mit leerem String."""
        result = get_tenant_normalized("")
        assert result == "default"

    def test_get_tenant_normalized_with_whitespace(self):
        """Testet Tenant-Normalisierung mit Whitespace."""
        result = get_tenant_normalized("  ")
        assert result == "default"

    def test_get_redis_key_with_all_parts(self):
        """Testet Redis-Key-Erstellung mit allen Teilen."""
        result = get_redis_key("prefix", "part1", "part2", "part3")
        assert result == "prefix:part1:part2:part3"

    def test_get_redis_key_with_none_parts(self):
        """Testet Redis-Key-Erstellung mit None-Teilen."""
        result = get_redis_key("prefix", "part1", None, "part3")
        assert result == "prefix:part1:part3"

    def test_get_redis_key_with_empty_parts(self):
        """Testet Redis-Key-Erstellung mit leeren Teilen."""
        result = get_redis_key("prefix", "", "part2")
        assert result == "prefix::part2"

    def test_get_redis_key_only_prefix(self):
        """Testet Redis-Key-Erstellung nur mit Präfix."""
        result = get_redis_key("prefix")
        assert result == "prefix"


class TestConstantsIntegration:
    """Integrationstests für Konstanten."""

    def test_all_timeout_constants_are_positive(self):
        """Testet, dass alle Timeout-Konstanten positiv sind."""
        from services.webhooks import constants

        timeout_attrs = [
            attr for attr in dir(constants)
            if "TIMEOUT" in attr and not attr.startswith("_") and attr.isupper()
        ]

        for attr_name in timeout_attrs:
            value = getattr(constants, attr_name)
            # Skip MagicMock objects from patched settings
            if hasattr(value, "_mock_name"):
                continue
            assert isinstance(value, int | float), f"{attr_name} sollte numerisch sein"
            assert value > 0, f"{attr_name} sollte positiv sein, ist aber {value}"

    def test_all_ttl_constants_are_positive(self):
        """Testet, dass alle TTL-Konstanten positiv sind."""
        from services.webhooks import constants

        ttl_attrs = [
            attr for attr in dir(constants)
            if "TTL" in attr and not attr.startswith("_") and attr.isupper()
        ]

        for attr_name in ttl_attrs:
            value = getattr(constants, attr_name)
            # Skip MagicMock objects from patched settings
            if hasattr(value, "_mock_name"):
                continue
            assert isinstance(value, int | float), f"{attr_name} sollte numerisch sein"
            assert value > 0, f"{attr_name} sollte positiv sein, ist aber {value}"

    def test_redis_prefixes_follow_naming_convention(self):
        """Testet, dass Redis-Präfixe der Namenskonvention folgen."""
        from services.webhooks import constants

        prefix_attrs = [
            attr for attr in dir(constants)
            if attr.startswith("REDIS_") and attr.endswith("_PREFIX") and not attr.startswith("_")
        ]

        for attr_name in prefix_attrs:
            value = getattr(constants, attr_name)
            assert isinstance(value, str), f"{attr_name} sollte ein String sein"
            assert "kei:" in value, f"{attr_name} sollte 'kei:' enthalten"
            assert value.count(":") >= 1, f"{attr_name} sollte mindestens einen Doppelpunkt enthalten"

    def test_header_constants_follow_convention(self):
        """Testet, dass Header-Konstanten der Konvention folgen."""
        from services.webhooks import constants

        header_attrs = [
            attr for attr in dir(constants)
            if attr.startswith("KEI_") and attr.endswith("_HEADER") and not attr.startswith("_")
        ]

        for attr_name in header_attrs:
            value = getattr(constants, attr_name)
            assert isinstance(value, str), f"{attr_name} sollte ein String sein"
            assert value.startswith("x-kei-"), f"{attr_name} sollte mit 'x-kei-' beginnen"
            assert value.islower(), f"{attr_name} sollte lowercase sein"


if __name__ == "__main__":
    pytest.main([__file__])
