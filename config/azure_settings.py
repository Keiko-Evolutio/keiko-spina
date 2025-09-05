"""Azure Settings für Keiko Personal Assistant.

Alle Azure-bezogenen Konfigurationen in einer fokussierten Klasse.
Folgt Single Responsibility Principle.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

from .constants import (
    DEFAULT_API_VERSION,
    DEFAULT_AZURE_LOCATION,
    DEFAULT_CONTAINER_NAME,
    DEFAULT_DATABASE_NAME,
    DEFAULT_KEIKO_STORAGE_CONTAINER,
    DEFAULT_MODEL_DEPLOYMENT_NAME,
    DEFAULT_STORAGE_ACCOUNT_URL,
)
from .env_utils import get_env_str


class AzureSettings(BaseSettings):
    """Azure-spezifische Konfigurationen."""

    # Azure Subscription und Resource Group
    azure_subscription_id: str = Field(
        default="",
        description="Azure Subscription ID"
    )

    azure_resource_group: str = Field(
        default="",
        description="Azure Resource Group Name"
    )

    azure_location: str = Field(
        default=DEFAULT_AZURE_LOCATION,
        description="Azure Region/Location"
    )

    # Azure Storage
    storage_account_url: str = Field(
        default=DEFAULT_STORAGE_ACCOUNT_URL,
        description="Azure Storage Account URL"
    )

    keiko_storage_container_for_img: str = Field(
        default=DEFAULT_KEIKO_STORAGE_CONTAINER,
        description="Container für Keiko Bilder"
    )

    # Azure App Registration für Storage Authentication
    azure_client_id: str = Field(
        default="",
        description="Azure App Registration Client ID"
    )

    azure_client_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Azure App Registration Client Secret"
    )

    azure_tenant_id: str = Field(
        default="",
        description="Azure Tenant ID"
    )

    # SSL-Konfiguration
    azure_ssl_verify: bool = Field(
        default=True,
        description="SSL-Verifikation für Azure-Verbindungen aktivieren"
    )

    # Azure AI Services
    cosmosdb_connection: SecretStr = Field(
        default=SecretStr(""),
        description="CosmosDB Connection String"
    )

    database_name: str = Field(
        default=DEFAULT_DATABASE_NAME,
        description="CosmosDB Database Name"
    )

    container_name: str = Field(
        default=DEFAULT_CONTAINER_NAME,
        description="CosmosDB Container Name"
    )

    # Azure OpenAI Configuration
    project_keiko_model_inference_endpoint: str = Field(
        default="",
        description="Azure OpenAI Inference Endpoint"
    )

    project_keiko_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI Endpoint"
    )

    project_keiko_services_endpoint: str = Field(
        default="",
        description="Azure Services Endpoint"
    )

    project_keiko_api_key: str = Field(
        default="",
        description="Azure OpenAI API Key"
    )

    project_keiko_project_id: str = Field(
        default="",
        description="Azure Project ID"
    )

    project_keiko_model_deployment_name: str = Field(
        default=DEFAULT_MODEL_DEPLOYMENT_NAME,
        description="Azure OpenAI Model Deployment Name"
    )

    project_keiko_api_version: str = Field(
        default=DEFAULT_API_VERSION,
        description="Azure OpenAI API Version"
    )

    # Azure Voice/Realtime API
    project_keiko_voice_endpoint: str = Field(
        default="",
        description="Azure OpenAI Voice/Realtime API Endpoint"
    )

    project_keiko_voice_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Azure OpenAI Voice/Realtime API Key"
    )

    # Azure Image Generation (DALL·E-3)
    project_keiko_image_endpoint: str = Field(
        default="",
        description="Azure OpenAI Image Generation Endpoint"
    )

    project_keiko_image_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Azure OpenAI Image Generation API Key"
    )

    # Azure Content Safety
    azure_content_safety_endpoint: str = Field(
        default="",
        description="Azure Content Safety Endpoint"
    )

    azure_content_safety_key: SecretStr = Field(
        default=SecretStr(""),
        description="Azure Content Safety API Key"
    )

    # Azure Key Vault
    azure_key_vault_url: str = Field(
        default="",
        description="Azure Key Vault URL"
    )

    class Config:
        """Pydantic-Konfiguration."""
        env_prefix = "AZURE_"
        case_sensitive = False


def load_azure_settings() -> AzureSettings:
    """Lädt Azure Settings aus Umgebungsvariablen.

    Returns:
        AzureSettings-Instanz
    """
    # .env Datei explizit laden
    from dotenv import load_dotenv
    load_dotenv()
    return AzureSettings(
        azure_subscription_id=get_env_str("AZURE_SUBSCRIPTION_ID"),
        azure_resource_group=get_env_str("AZURE_RESOURCE_GROUP"),
        azure_location=get_env_str("AZURE_LOCATION", DEFAULT_AZURE_LOCATION),
        storage_account_url=get_env_str("STORAGE_ACCOUNT_URL", DEFAULT_STORAGE_ACCOUNT_URL),
        keiko_storage_container_for_img=get_env_str("KEIKO_STORAGE_CONTAINER_FOR_IMG", DEFAULT_KEIKO_STORAGE_CONTAINER),
        azure_client_id=get_env_str("AZURE_CLIENT_ID"),
        azure_client_secret=SecretStr(get_env_str("AZURE_CLIENT_SECRET")),
        azure_tenant_id=get_env_str("AZURE_TENANT_ID"),
        azure_ssl_verify=get_env_str("AZURE_SSL_VERIFY", "true").lower() == "true",
        cosmosdb_connection=SecretStr(get_env_str("COSMOSDB_CONNECTION")),
        database_name=get_env_str("DATABASE_NAME", DEFAULT_DATABASE_NAME),
        container_name=get_env_str("CONTAINER_NAME", DEFAULT_CONTAINER_NAME),
        project_keiko_model_inference_endpoint=get_env_str("PROJECT_KEIKO_MODEL_INFERENCE_ENDPOINT"),
        project_keiko_openai_endpoint=get_env_str("PROJECT_KEIKO_OPENAI_ENDPOINT"),
        project_keiko_services_endpoint=get_env_str("PROJECT_KEIKO_SERVICES_ENDPOINT"),
        project_keiko_api_key=get_env_str("PROJECT_KEIKO_API_KEY"),
        project_keiko_project_id=get_env_str("PROJECT_KEIKO_PROJECT_ID"),
        project_keiko_model_deployment_name=get_env_str("PROJECT_KEIKO_MODEL_DEPLOYMENT_NAME", DEFAULT_MODEL_DEPLOYMENT_NAME),
        project_keiko_api_version=get_env_str("PROJECT_KEIKO_API_VERSION", DEFAULT_API_VERSION),
        project_keiko_voice_endpoint=get_env_str("PROJECT_KEIKO_VOICE_ENDPOINT"),
        project_keiko_voice_api_key=SecretStr(get_env_str("PROJECT_KEIKO_VOICE_API_KEY")),
        project_keiko_image_endpoint=get_env_str("PROJECT_KEIKO_IMAGE_ENDPOINT"),
        project_keiko_image_api_key=SecretStr(get_env_str("PROJECT_KEIKO_IMAGE_API_KEY")),
        azure_content_safety_endpoint=get_env_str("AZURE_CONTENT_SAFETY_ENDPOINT"),
        azure_content_safety_key=SecretStr(get_env_str("AZURE_CONTENT_SAFETY_KEY")),
        azure_key_vault_url=get_env_str("AZURE_KEY_VAULT_URL")
    )


# Globale Azure Settings Instanz
azure_settings = load_azure_settings()


__all__ = [
    "AzureSettings",
    "azure_settings",
    "load_azure_settings"
]
