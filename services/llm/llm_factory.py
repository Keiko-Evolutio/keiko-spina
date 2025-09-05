# backend/services/llm/llm_factory.py
"""LLM Service Factory für einfache Integration.

Stellt vorkonfigurierte LLM Services basierend auf Environment-Variablen
und Application-Settings bereit.
"""

from __future__ import annotations

import os

from config.settings import settings
from kei_logging import get_logger

from .llm_client import LLMClient, LLMClientConfig
from .llm_monitoring import AlertConfig, LLMMonitor
from .prompt_templates import (
    TASK_DECOMPOSITION_TEMPLATES,
    PromptTemplateConfig,
    PromptTemplateManager,
)

logger = get_logger(__name__)


class LLMServiceFactory:
    """Factory für LLM Services mit automatischer Konfiguration."""

    _client_instance: LLMClient | None = None
    _monitor_instance: LLMMonitor | None = None
    _template_manager_instance: PromptTemplateManager | None = None

    @classmethod
    def get_llm_client(cls) -> LLMClient:
        """Gibt Singleton LLM Client zurück.

        Returns:
            Konfigurierter LLM Client

        Raises:
            ValueError: Falls API-Key nicht konfiguriert
        """
        if cls._client_instance is None:
            cls._client_instance = cls._create_llm_client()

        return cls._client_instance

    @classmethod
    def get_llm_monitor(cls) -> LLMMonitor:
        """Gibt Singleton LLM Monitor zurück.

        Returns:
            Konfigurierter LLM Monitor
        """
        if cls._monitor_instance is None:
            cls._monitor_instance = cls._create_llm_monitor()

        return cls._monitor_instance

    @classmethod
    def get_template_manager(cls) -> PromptTemplateManager:
        """Gibt Singleton Template Manager zurück.

        Returns:
            Konfigurierter Template Manager
        """
        if cls._template_manager_instance is None:
            cls._template_manager_instance = cls._create_template_manager()

        return cls._template_manager_instance

    @classmethod
    def _create_llm_client(cls) -> LLMClient:
        """Erstellt LLM Client basierend auf Environment-Konfiguration."""
        # API-Key aus Environment oder Settings
        api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None)

        if not api_key:
            raise ValueError("OPENAI_API_KEY nicht konfiguriert")

        # Azure OpenAI Konfiguration (optional)
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # Rate Limiting Konfiguration
        max_requests_per_minute = int(os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", "60"))
        max_tokens_per_minute = int(os.getenv("LLM_MAX_TOKENS_PER_MINUTE", "150000"))

        # Cost Management Konfiguration
        max_cost_per_hour = float(os.getenv("LLM_MAX_COST_PER_HOUR_USD", "10.0"))
        cost_alert_threshold = float(os.getenv("LLM_COST_ALERT_THRESHOLD_USD", "5.0"))

        # Caching Konfiguration
        enable_caching = os.getenv("LLM_ENABLE_CACHING", "true").lower() == "true"
        cache_ttl = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))

        # Retry Konfiguration
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        initial_delay = float(os.getenv("LLM_INITIAL_DELAY", "1.0"))
        backoff_multiplier = float(os.getenv("LLM_BACKOFF_MULTIPLIER", "2.0"))

        # Fallback Konfiguration
        enable_fallback = os.getenv("LLM_ENABLE_FALLBACK", "true").lower() == "true"
        fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gpt-3.5-turbo")

        # PII Redaction Konfiguration
        enable_pii_redaction = os.getenv("LLM_ENABLE_PII_REDACTION", "true").lower() == "true"

        config = LLMClientConfig(
            api_key=api_key,
            endpoint=azure_endpoint,
            deployment=azure_deployment,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            max_cost_per_hour_usd=max_cost_per_hour,
            cost_alert_threshold_usd=cost_alert_threshold,
            enable_caching=enable_caching,
            cache_ttl_seconds=cache_ttl,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_multiplier=backoff_multiplier,
            enable_fallback=enable_fallback,
            fallback_model=fallback_model,
            enable_pii_redaction=enable_pii_redaction
        )

        logger.info({
            "event": "llm_client_created",
            "azure_enabled": bool(azure_endpoint),
            "caching_enabled": enable_caching,
            "pii_redaction_enabled": enable_pii_redaction,
            "max_requests_per_minute": max_requests_per_minute
        })

        return LLMClient(config)

    @classmethod
    def _create_llm_monitor(cls) -> LLMMonitor:
        """Erstellt LLM Monitor basierend auf Environment-Konfiguration."""
        # Alert Konfiguration
        cost_threshold = float(os.getenv("LLM_ALERT_COST_THRESHOLD_USD", "5.0"))
        cost_alert_interval = int(os.getenv("LLM_ALERT_COST_INTERVAL_MINUTES", "15"))
        response_time_threshold = float(os.getenv("LLM_ALERT_RESPONSE_TIME_MS", "5000.0"))
        error_rate_threshold = float(os.getenv("LLM_ALERT_ERROR_RATE", "0.1"))
        rate_limit_threshold = int(os.getenv("LLM_ALERT_RATE_LIMIT_HITS", "5"))
        token_usage_threshold = int(os.getenv("LLM_ALERT_TOKEN_USAGE_PER_HOUR", "100000"))

        alert_config = AlertConfig(
            cost_threshold_usd=cost_threshold,
            cost_alert_interval_minutes=cost_alert_interval,
            response_time_threshold_ms=response_time_threshold,
            error_rate_threshold=error_rate_threshold,
            rate_limit_alert_threshold=rate_limit_threshold,
            token_usage_threshold_per_hour=token_usage_threshold
        )

        logger.info({
            "event": "llm_monitor_created",
            "cost_threshold_usd": cost_threshold,
            "response_time_threshold_ms": response_time_threshold,
            "error_rate_threshold": error_rate_threshold
        })

        return LLMMonitor(alert_config)

    @classmethod
    def _create_template_manager(cls) -> PromptTemplateManager:
        """Erstellt Template Manager basierend auf Environment-Konfiguration."""
        # Template Konfiguration
        templates_directory = os.getenv("LLM_TEMPLATES_DIRECTORY", "templates/prompts")
        enable_caching = os.getenv("LLM_TEMPLATES_ENABLE_CACHING", "true").lower() == "true"
        auto_reload = os.getenv("LLM_TEMPLATES_AUTO_RELOAD", "false").lower() == "true"

        config = PromptTemplateConfig(
            templates_directory=templates_directory,
            enable_caching=enable_caching,
            auto_reload=auto_reload
        )

        manager = PromptTemplateManager(config)

        # Vordefinierte Templates laden
        cls._load_predefined_templates(manager)

        # Templates aus Dateien laden falls vorhanden
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(manager.load_templates())
            else:
                loop.run_until_complete(manager.load_templates())
        except RuntimeError:
            # Fallback: Templates sind bereits in Memory geladen
            pass

        logger.info({
            "event": "template_manager_created",
            "templates_directory": templates_directory,
            "caching_enabled": enable_caching,
            "auto_reload": auto_reload
        })

        return manager

    @classmethod
    def _load_predefined_templates(cls, manager: PromptTemplateManager) -> None:
        """Lädt vordefinierte Templates in den Manager."""
        try:
            from .prompt_templates import PromptTemplate

            for template_data in TASK_DECOMPOSITION_TEMPLATES.values():
                template = PromptTemplate(
                    name=template_data["name"],
                    version=template_data["version"],
                    template=template_data["template"],
                    description=template_data["description"],
                    variables=template_data["variables"]
                )

                # Template asynchron speichern (wird in Event Loop ausgeführt)
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Wenn Event Loop läuft, Task erstellen
                        asyncio.create_task(manager.save_template(template))
                    else:
                        # Wenn keine Event Loop läuft, synchron ausführen
                        loop.run_until_complete(manager.save_template(template))
                except RuntimeError:
                    # Fallback: Template nur in Memory speichern
                    if template.name not in manager._templates:
                        manager._templates[template.name] = {}
                    manager._templates[template.name][template.version] = template

            logger.info({
                "event": "predefined_templates_loaded",
                "template_count": len(TASK_DECOMPOSITION_TEMPLATES)
            })

        except Exception as e:
            logger.warning(f"Fehler beim Laden vordefinierter Templates: {e}")

    @classmethod
    def reset_instances(cls) -> None:
        """Setzt alle Singleton-Instanzen zurück (für Tests)."""
        cls._client_instance = None
        cls._monitor_instance = None
        cls._template_manager_instance = None


# Convenience-Funktionen für einfache Nutzung
def get_llm_client() -> LLMClient:
    """Gibt konfigurierten LLM Client zurück."""
    return LLMServiceFactory.get_llm_client()


def get_llm_monitor() -> LLMMonitor:
    """Gibt konfigurierten LLM Monitor zurück."""
    return LLMServiceFactory.get_llm_monitor()


def get_template_manager() -> PromptTemplateManager:
    """Gibt konfigurierten Template Manager zurück."""
    return LLMServiceFactory.get_template_manager()
