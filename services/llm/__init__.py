# backend/services/llm/__init__.py
"""LLM Services Package für Orchestrator Service.

Stellt Enterprise-Grade LLM Client Integration mit Rate-Limiting,
Cost-Tracking, Prompt-Template-Management und Monitoring bereit.
"""

from __future__ import annotations

from .llm_client import LLMClient, LLMClientConfig, LLMRequest, LLMResponse
from .llm_factory import LLMServiceFactory, get_llm_client, get_llm_monitor, get_template_manager
from .llm_monitoring import AlertConfig, LLMMonitor
from .prompt_templates import PromptTemplate, PromptTemplateConfig, PromptTemplateManager

__all__ = [
    # LLM Client
    "LLMClient",
    "LLMClientConfig",
    "LLMRequest",
    "LLMResponse",
    # Monitoring
    "LLMMonitor",
    "AlertConfig",
    # Templates
    "PromptTemplate",
    "PromptTemplateManager",
    "PromptTemplateConfig",
    # Factory Functions
    "create_llm_client",
    "create_llm_monitor",
    "create_template_manager",
    # Service Factory
    "LLMServiceFactory",
    "get_llm_client",
    "get_llm_monitor",
    "get_template_manager",
]

__version__ = "1.0.0"


def create_llm_client(
    api_key: str,
    endpoint: str | None = None,
    deployment: str | None = None,
    **kwargs
) -> LLMClient:
    """Factory-Funktion für LLM Client.

    Args:
        api_key: OpenAI/Azure OpenAI API Key
        endpoint: Azure OpenAI Endpoint (optional)
        deployment: Azure OpenAI Deployment (optional)
        **kwargs: Zusätzliche Konfigurationsparameter

    Returns:
        Konfigurierter LLM Client
    """
    config = LLMClientConfig(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        **kwargs
    )

    return LLMClient(config)


def create_llm_monitor(**kwargs) -> LLMMonitor:
    """Factory-Funktion für LLM Monitor.

    Args:
        **kwargs: Alert-Konfigurationsparameter

    Returns:
        Konfigurierter LLM Monitor
    """
    alert_config = AlertConfig(**kwargs)
    return LLMMonitor(alert_config)


def create_template_manager(templates_directory: str = "templates/prompts", **kwargs) -> PromptTemplateManager:
    """Factory-Funktion für Template Manager.

    Args:
        templates_directory: Verzeichnis für Templates
        **kwargs: Zusätzliche Konfigurationsparameter

    Returns:
        Konfigurierter Template Manager
    """
    config = PromptTemplateConfig(
        templates_directory=templates_directory,
        **kwargs
    )

    return PromptTemplateManager(config)
