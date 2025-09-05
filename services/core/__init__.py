# backend/services/core/__init__.py
"""Zentrale Kernkomponenten für Service-Management."""

from .features import ServiceFeatures, features
from .manager import ServiceManager, service_manager

__all__ = [
    "ServiceFeatures",
    "ServiceManager",
    "features",
    "service_manager",
]
