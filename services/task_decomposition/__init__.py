# backend/services/task_decomposition/__init__.py
"""LLM-powered Task Decomposition Engine Package.

Implementiert intelligente Task-Zerlegung mit LLM-Integration,
Agent-Capability-Matching und Performance-Optimierung.
"""

from __future__ import annotations

from .agent_matcher import AgentCapabilityMatcher
from .data_models import (
    AgentMatch,
    ComplexityLevel,
    DecompositionPlan,
    DecompositionRequest,
    DecompositionResult,
    DecompositionStrategy,
    DependencyType,
    FallbackRule,
    SubtaskDefinition,
    TaskAnalysis,
    ValidationResult,
)
from .decomposition_engine import TaskDecompositionEngine
from .fallback_decomposer import FallbackDecomposer
from .llm_analyzer import LLMTaskAnalyzer
from .plan_validator import PlanValidator

__all__ = [
    # Data Models
    "TaskAnalysis",
    "SubtaskDefinition",
    "AgentMatch",
    "DecompositionPlan",
    "DecompositionRequest",
    "DecompositionResult",
    "ValidationResult",
    "FallbackRule",
    "ComplexityLevel",
    "DecompositionStrategy",
    "DependencyType",

    # Core Components
    "TaskDecompositionEngine",
    "LLMTaskAnalyzer",
    "AgentCapabilityMatcher",
    "PlanValidator",
    "FallbackDecomposer",

    # Factory Functions
    "create_task_decomposition_engine",
    "create_llm_analyzer",
    "create_agent_matcher",
    "create_plan_validator",
    "create_fallback_decomposer",
]

__version__ = "1.0.0"


def create_task_decomposition_engine(
    task_manager=None,
    agent_registry=None,
    performance_predictor=None
) -> TaskDecompositionEngine:
    """Factory-Funktion für Task Decomposition Engine.
    
    Args:
        task_manager: Task Manager Instanz
        agent_registry: Agent Registry Instanz
        performance_predictor: Performance Predictor aus TASK 2
        
    Returns:
        Konfigurierte Task Decomposition Engine
    """
    if not task_manager:
        from task_management.core_task_manager import task_manager as default_task_manager
        task_manager = default_task_manager

    if not agent_registry:
        from agents.registry.dynamic_registry import DynamicAgentRegistry
        agent_registry = DynamicAgentRegistry()

    return TaskDecompositionEngine(
        task_manager=task_manager,
        agent_registry=agent_registry,
        performance_predictor=performance_predictor
    )


def create_llm_analyzer() -> LLMTaskAnalyzer:
    """Factory-Funktion für LLM Task Analyzer.
    
    Returns:
        Konfigurierter LLM Task Analyzer
    """
    return LLMTaskAnalyzer()


def create_agent_matcher(
    agent_registry=None,
    performance_predictor=None
) -> AgentCapabilityMatcher:
    """Factory-Funktion für Agent Capability Matcher.
    
    Args:
        agent_registry: Agent Registry Instanz
        performance_predictor: Performance Predictor aus TASK 2
        
    Returns:
        Konfigurierter Agent Capability Matcher
    """
    if not agent_registry:
        from agents.registry.dynamic_registry import DynamicAgentRegistry
        agent_registry = DynamicAgentRegistry()

    return AgentCapabilityMatcher(
        agent_registry=agent_registry,
        performance_predictor=performance_predictor
    )


def create_plan_validator() -> PlanValidator:
    """Factory-Funktion für Plan Validator.
    
    Returns:
        Konfigurierter Plan Validator
    """
    return PlanValidator()


def create_fallback_decomposer() -> FallbackDecomposer:
    """Factory-Funktion für Fallback Decomposer.
    
    Returns:
        Konfigurierter Fallback Decomposer
    """
    return FallbackDecomposer()
