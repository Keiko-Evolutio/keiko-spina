"""Custom Agents Paket.

Implementiert native Python-Agenten für Bildgenerierung und andere Aufgaben.
"""
from agents.constants import (
    ALLOWED_IMAGE_QUALITIES,
    ALLOWED_IMAGE_SIZES,
    ALLOWED_IMAGE_STYLES,
    DEFAULT_SAFETY_SCORE,
    IMAGE_GENERATION_ERROR_THRESHOLD_MS,
    IMAGE_GENERATION_WARNING_THRESHOLD_MS,
    METRICS_INCREMENT,
    SECONDS_TO_MILLISECONDS,
    AgentStatus,
    AgentType,
    ImageQuality,
    ImageSize,
    ImageStyle,
)

from .image_generator_agent import ImageGeneratorAgent, ImageTask

__all__ = [
    "ALLOWED_IMAGE_QUALITIES",
    "ALLOWED_IMAGE_SIZES",
    "ALLOWED_IMAGE_STYLES",
    "DEFAULT_SAFETY_SCORE",
    "IMAGE_GENERATION_ERROR_THRESHOLD_MS",
    "IMAGE_GENERATION_WARNING_THRESHOLD_MS",
    "METRICS_INCREMENT",
    "SECONDS_TO_MILLISECONDS",
    "AgentStatus",
    "AgentType",
    "ImageGeneratorAgent",
    "ImageQuality",
    "ImageSize",
    "ImageStyle",
    "ImageTask",
]
