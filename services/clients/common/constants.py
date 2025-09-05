# backend/services/clients/common/constants.py
"""Konstanten für Client Services.

Alle Magic Numbers und Hard-coded Strings sind hier zentralisiert.
"""

from __future__ import annotations

from typing import Final

# =====================================================================
# HTTP Client Konfiguration
# =====================================================================

# Timeouts (in Sekunden)
DEFAULT_TIMEOUT: Final[float] = 30.0
DEFAULT_CONNECT_TIMEOUT: Final[float] = 5.0
DEFAULT_REQUEST_TIMEOUT: Final[float] = 10.0

# Connection Pool Limits
DEFAULT_CONNECTION_LIMIT: Final[int] = 100
DEFAULT_CONNECTION_LIMIT_PER_HOST: Final[int] = 30
DEFAULT_KEEPALIVE_TIMEOUT: Final[int] = 30

# =====================================================================
# Retry Konfiguration
# =====================================================================

# Retry Versuche
DEFAULT_MAX_RETRIES: Final[int] = 2
CONTENT_SAFETY_MAX_RETRIES: Final[int] = 2
IMAGE_GENERATION_MAX_RETRIES: Final[int] = 2
DEEP_RESEARCH_MAX_ITERATIONS: Final[int] = 3

# Retry Delays (in Sekunden)
DEFAULT_INITIAL_DELAY: Final[float] = 0.5
DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
DEFAULT_MAX_DELAY: Final[float] = 60.0

# =====================================================================
# API Versionen
# =====================================================================

# Azure Content Safety
CONTENT_SAFETY_API_VERSION: Final[str] = "2024-09-01"

# Azure OpenAI Images
DEFAULT_IMAGE_API_VERSION: Final[str] = "2024-12-01-preview"
DEFAULT_IMAGE_DEPLOYMENT: Final[str] = "dall-e-3"

# =====================================================================
# Content Safety Konfiguration
# =====================================================================

# Severity Levels (Azure Content Safety)
SAFE_SEVERITY_THRESHOLD: Final[int] = 1  # 0-1 sicher, 2-3 unsicher
MAX_SEVERITY_LEVEL: Final[int] = 3

# Content Safety Kategorien
CONTENT_SAFETY_CATEGORIES: Final[list[str]] = [
    "Hate",
    "SelfHarm",
    "Sexual",
    "Violence"
]

# =====================================================================
# Image Generation Konfiguration
# =====================================================================

# Standard Image Größen
DEFAULT_IMAGE_SIZE: Final[str] = "1024x1024"
VALID_IMAGE_SIZES: Final[list[str]] = [
    "1024x1024",
    "1024x1792",
    "1792x1024"
]

# Image Quality
DEFAULT_IMAGE_QUALITY: Final[str] = "standard"
VALID_IMAGE_QUALITIES: Final[list[str]] = ["standard", "hd"]

# Image Formats
DEFAULT_IMAGE_RESPONSE_FORMAT: Final[str] = "b64_json"
DEFAULT_IMAGE_CONTENT_TYPE: Final[str] = "image/png"

# =====================================================================
# Error Messages
# =====================================================================

# Service Availability
SERVICE_NOT_CONFIGURED_ERROR: Final[str] = "Service nicht konfiguriert"
SERVICE_UNAVAILABLE_ERROR: Final[str] = "Service nicht verfügbar"

# Content Safety
CONTENT_SAFETY_UNAVAILABLE_REASON: Final[str] = "unavailable"
CONTENT_SAFETY_FALLBACK_CATEGORY: Final[str] = "unknown"

# Image Generation
IMAGE_GENERATION_NO_CONTENT_ERROR: Final[str] = "Kein Bild-Content in Antwort"
CONTENT_POLICY_VIOLATION_ERROR: Final[str] = "content_policy_violation"
SAFETY_SYSTEM_ERROR: Final[str] = "safety system"

# Deep Research
DEEP_RESEARCH_SDK_UNAVAILABLE: Final[str] = "SDK nicht verfügbar oder nicht konfiguriert"
DEEP_RESEARCH_FALLBACK_MESSAGE: Final[str] = "Azure Deep Research nicht verfügbar. Fallback aktiv."

# =====================================================================
# Logging Events
# =====================================================================

# Client Initialization
CLIENT_INIT_EVENT: Final[str] = "client_init"
CLIENT_READY_EVENT: Final[str] = "client_ready"

# Content Safety Events
CONTENT_SAFETY_REQUEST_EVENT: Final[str] = "content_safety_request"
CONTENT_SAFETY_UNAVAILABLE_EVENT: Final[str] = "content_safety_unavailable_fallback"

# Image Generation Events
IMAGE_GENERATION_REQUEST_EVENT: Final[str] = "image_generation_request"
IMAGE_SERVICE_CLIENT_READY_EVENT: Final[str] = "image_service_client_ready"
OPENAI_IMAGES_GENERATE_CALL_EVENT: Final[str] = "openai_images_generate_call"
OPENAI_IMAGES_GENERATE_OK_EVENT: Final[str] = "openai_images_generate_ok"
OPENAI_IMAGES_NO_CONTENT_EVENT: Final[str] = "openai_images_no_content"
OPENAI_IMAGES_GENERATE_ERROR_EVENT: Final[str] = "openai_images_generate_error"

# Prompt Safety Events
PROMPT_SAFETY_VIOLATION_EVENT: Final[str] = "prompt_safety_violation"
PROMPT_SANITIZED_EVENT: Final[str] = "prompt_sanitized"
CONTENT_POLICY_VIOLATION_DETECTED_EVENT: Final[str] = "content_policy_violation_detected"
USING_FALLBACK_PROMPT_EVENT: Final[str] = "using_fallback_prompt"

# =====================================================================
# Default Values
# =====================================================================

# User Identifiers
DEFAULT_USER_ID: Final[str] = "keiko"

# Confidence Scores
DEFAULT_CONFIDENCE_SCORE: Final[float] = 0.0
HIGH_CONFIDENCE_SCORE: Final[float] = 0.8
MEDIUM_CONFIDENCE_SCORE: Final[float] = 0.6

# Response Formats
DEFAULT_RESPONSE_FORMAT: Final[str] = "json"
