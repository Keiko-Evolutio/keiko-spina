# backend/services/enhanced_dependency_resolution/__init__.py
"""Enhanced Dependency Resolution Package.

Implementiert Enterprise-Grade Dependency-Management mit Intelligent Graph Analysis,
Circular Dependency Detection und Real-time Dependency-Tracking.
"""

from __future__ import annotations

from .circular_dependency_engine import CircularDependencyEngine
from .data_models import (
    CircularDependency,
    CircularResolutionStrategy,
    DependencyAnalytics,
    DependencyCache,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    DependencyPerformanceMetrics,
    DependencyRelation,
    DependencyResolutionRequest,
    DependencyResolutionResult,
    DependencyStatus,
    DependencyType,
    ResolutionStrategy,
    ResourceDependencyContext,
    TaskDependencyContext,
)
from .dependency_cache_engine import DependencyCacheEngine
from .dependency_graph_engine import EnhancedDependencyGraphEngine
from .dependency_resolution_engine import EnhancedDependencyResolutionEngine
from .service_integration_layer import DependencyServiceIntegrationLayer

__all__ = [
    # Core Components
    "EnhancedDependencyGraphEngine",
    "EnhancedDependencyResolutionEngine",
    "CircularDependencyEngine",
    "DependencyCacheEngine",
    "DependencyServiceIntegrationLayer",

    # Data Models
    "DependencyNode",
    "DependencyEdge",
    "DependencyGraph",
    "DependencyResolutionRequest",
    "DependencyResolutionResult",
    "CircularDependency",
    "DependencyCache",
    "DependencyAnalytics",
    "DependencyPerformanceMetrics",
    "TaskDependencyContext",
    "ResourceDependencyContext",

    # Enums
    "DependencyType",
    "DependencyRelation",
    "DependencyStatus",
    "ResolutionStrategy",
    "CircularResolutionStrategy",

    # Factory Functions
    "create_enhanced_dependency_graph_engine",
    "create_enhanced_dependency_resolution_engine",
    "create_circular_dependency_engine",
    "create_dependency_cache_engine",
    "create_dependency_service_integration_layer",
    "create_integrated_dependency_resolution_system",
]

__version__ = "1.0.0"


def create_enhanced_dependency_graph_engine() -> EnhancedDependencyGraphEngine:
    """Factory-Funktion für Enhanced Dependency Graph Engine.

    Returns:
        Konfigurierte Enhanced Dependency Graph Engine
    """
    return EnhancedDependencyGraphEngine()


def create_enhanced_dependency_resolution_engine(
    graph_engine: EnhancedDependencyGraphEngine,
    quota_management_engine=None
) -> EnhancedDependencyResolutionEngine:
    """Factory-Funktion für Enhanced Dependency Resolution Engine.

    Args:
        graph_engine: Dependency Graph Engine
        quota_management_engine: Quota Management Engine (optional)

    Returns:
        Konfigurierte Enhanced Dependency Resolution Engine
    """
    return EnhancedDependencyResolutionEngine(
        graph_engine=graph_engine,
        quota_management_engine=quota_management_engine
    )


def create_circular_dependency_engine() -> CircularDependencyEngine:
    """Factory-Funktion für Circular Dependency Engine.

    Returns:
        Konfigurierte Circular Dependency Engine
    """
    return CircularDependencyEngine()


def create_dependency_cache_engine() -> DependencyCacheEngine:
    """Factory-Funktion für Dependency Cache Engine.

    Returns:
        Konfigurierte Dependency Cache Engine
    """
    return DependencyCacheEngine()


def create_dependency_service_integration_layer(
    dependency_resolution_engine: EnhancedDependencyResolutionEngine,
    dependency_graph_engine: EnhancedDependencyGraphEngine,
    security_integration_engine=None,
    quota_management_engine=None
) -> DependencyServiceIntegrationLayer:
    """Factory-Funktion für Dependency Service Integration Layer.

    Args:
        dependency_resolution_engine: Dependency Resolution Engine
        dependency_graph_engine: Dependency Graph Engine
        security_integration_engine: Security Integration Engine (optional)
        quota_management_engine: Quota Management Engine (optional)

    Returns:
        Konfigurierte Dependency Service Integration Layer
    """
    return DependencyServiceIntegrationLayer(
        dependency_resolution_engine=dependency_resolution_engine,
        dependency_graph_engine=dependency_graph_engine,
        security_integration_engine=security_integration_engine,
        quota_management_engine=quota_management_engine
    )


def create_integrated_dependency_resolution_system(
    security_integration_engine=None,
    quota_management_engine=None
) -> dict:
    """Factory-Funktion für integriertes Dependency Resolution System.

    Args:
        security_integration_engine: Security Integration Engine (optional)
        quota_management_engine: Quota Management Engine (optional)

    Returns:
        Dictionary mit allen konfigurierten Komponenten
    """
    # Erstelle alle Komponenten
    dependency_graph_engine = create_enhanced_dependency_graph_engine()

    dependency_resolution_engine = create_enhanced_dependency_resolution_engine(
        graph_engine=dependency_graph_engine,
        quota_management_engine=quota_management_engine
    )

    circular_dependency_engine = create_circular_dependency_engine()

    dependency_cache_engine = create_dependency_cache_engine()

    dependency_service_integration_layer = create_dependency_service_integration_layer(
        dependency_resolution_engine=dependency_resolution_engine,
        dependency_graph_engine=dependency_graph_engine,
        security_integration_engine=security_integration_engine,
        quota_management_engine=quota_management_engine
    )

    return {
        "dependency_graph_engine": dependency_graph_engine,
        "dependency_resolution_engine": dependency_resolution_engine,
        "circular_dependency_engine": circular_dependency_engine,
        "dependency_cache_engine": dependency_cache_engine,
        "dependency_service_integration_layer": dependency_service_integration_layer
    }
