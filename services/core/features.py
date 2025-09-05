# backend/services/core/features.py
"""Feature-Detection für Service-Komponenten."""


from kei_logging import get_logger

logger = get_logger(__name__)


class ServiceFeatures:
    """Zentrale Feature-Detection."""

    def __init__(self) -> None:
        self._features = self._detect_all_features()
        available = [k for k, v in self._features.items() if v]
        logger.debug(f"Verfügbare Features: {', '.join(available)}")

    def _detect_all_features(self) -> dict[str, bool]:
        """Detektiert alle verfügbaren Features."""
        return {
            "http_clients": self._check_import("aiohttp") or self._check_import("httpx"),
            "azure_core": self._check_import("azure.identity"),
            # Azure AI Foundry (Deep Research)
            "azure_ai": self._check_import("azure.ai.projects"),
            "pools": self._check_module_path("services.pools"),
            "clients": self._check_module_path("services.clients.clients"),
        }

    def _check_import(self, module_name: str) -> bool:
        """Prüft Import-Verfügbarkeit."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _check_module_path(self, module_path: str) -> bool:
        """Prüft Modul-Pfad-Verfügbarkeit."""
        return self._check_import(module_path)

    def is_available(self, feature: str) -> bool:
        """Prüft Feature-Verfügbarkeit."""
        return self._features.get(feature, False)

    @property
    def all_features(self) -> dict[str, bool]:
        """Alle Feature-Stati."""
        return self._features.copy()


# Globale Instanz
features = ServiceFeatures()
