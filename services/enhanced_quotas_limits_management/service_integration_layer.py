# backend/services/enhanced_quotas_limits_management/service_integration_layer.py
"""Service Integration Layer für Enhanced Quotas & Limits Management.

Integriert Enhanced Quotas & Limits Management mit allen bestehenden Services
und erweitert sie um Enterprise-Grade Resource-Management-Features.
"""

from __future__ import annotations

import time
from typing import Any

from kei_logging import get_logger
from services.enhanced_security_integration import (
    EnhancedSecurityIntegrationEngine,
    SecurityContext,
)

from .data_models import QuotaScope, ResourceType
from .quota_management_engine import EnhancedQuotaManagementEngine

logger = get_logger(__name__)


class ServiceIntegrationLayer:
    """Service Integration Layer für Enhanced Quotas & Limits Management."""

    def __init__(
        self,
        quota_management_engine: EnhancedQuotaManagementEngine,
        security_integration_engine: EnhancedSecurityIntegrationEngine
    ):
        """Initialisiert Service Integration Layer.

        Args:
            quota_management_engine: Enhanced Quota Management Engine
            security_integration_engine: Enhanced Security Integration Engine
        """
        self.quota_management_engine = quota_management_engine
        self.security_integration_engine = security_integration_engine

        # Integration-Konfiguration
        self.enable_service_integration = True
        self.enable_quota_aware_orchestration = True
        self.enable_quota_aware_agent_selection = True
        self.enable_quota_aware_task_decomposition = True

        # Performance-Tracking
        self._integration_count = 0
        self._total_integration_time_ms = 0.0
        self._quota_aware_operations = 0

        logger.info("Service Integration Layer initialisiert")

    async def start(self) -> None:
        """Startet Service Integration Layer."""
        try:
            # Starte Quota Management Engine
            await self.quota_management_engine.start()

            logger.info("Service Integration Layer gestartet")

        except Exception as e:
            logger.error(f"Service Integration Layer start fehlgeschlagen: {e}")
            raise

    async def stop(self) -> None:
        """Stoppt Service Integration Layer."""
        try:
            await self.quota_management_engine.stop()

            logger.info("Service Integration Layer gestoppt")

        except Exception as e:
            logger.error(f"Service Integration Layer stop fehlgeschlagen: {e}")

    async def enhance_orchestrator_service_with_quotas(
        self,
        orchestration_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Orchestrator Service um Quota-Management.

        Args:
            orchestration_request: Orchestration Request
            security_context: Security Context

        Returns:
            Enhanced Orchestration Result mit Quota-Informationen
        """
        start_time = time.time()

        try:
            logger.debug({
                "event": "enhance_orchestrator_with_quotas_started",
                "orchestration_id": orchestration_request.get("orchestration_id"),
                "user_id": security_context.user_id,
                "tenant_id": security_context.tenant_id
            })

            # 1. Quota-Check für Orchestration
            orchestration_quota_check = await self.quota_management_engine.check_quota_with_security(
                resource_type=ResourceType.TASK,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                amount=1,
                security_context=security_context
            )

            if not orchestration_quota_check.allowed:
                return {
                    "success": False,
                    "error": "Orchestration quota exceeded",
                    "quota_result": orchestration_quota_check,
                    "retry_after_seconds": orchestration_quota_check.retry_after_seconds
                }

            # 2. Erweitere Request mit Quota-Constraints
            enhanced_request = orchestration_request.copy()
            enhanced_request["quota_constraints"] = {
                "max_agents": self._calculate_max_agents_for_quota(security_context),
                "max_tasks": self._calculate_max_tasks_for_quota(security_context),
                "max_execution_time_minutes": self._calculate_max_execution_time_for_quota(security_context),
                "quota_monitoring_enabled": True
            }

            # 3. Füge Quota-Tracking hinzu
            enhanced_request["quota_tracking"] = {
                "track_agent_usage": True,
                "track_task_usage": True,
                "track_api_calls": True,
                "track_llm_requests": True
            }

            # Performance-Tracking
            integration_time_ms = (time.time() - start_time) * 1000
            self._update_integration_performance_stats(integration_time_ms)

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "quota_result": orchestration_quota_check,
                "quota_constraints": enhanced_request["quota_constraints"]
            }

        except Exception as e:
            logger.error(f"Orchestrator Service Quota Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_agent_selection_with_quotas(
        self,
        agent_selection_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Agent Selection um Quota-Constraints.

        Args:
            agent_selection_request: Agent Selection Request
            security_context: Security Context

        Returns:
            Enhanced Agent Selection Result mit Quota-Constraints
        """
        try:
            logger.debug({
                "event": "enhance_agent_selection_with_quotas_started",
                "request_id": agent_selection_request.get("request_id"),
                "user_id": security_context.user_id
            })

            # 1. Quota-Check für Agent Selection
            agent_quota_check = await self.quota_management_engine.check_quota_with_security(
                resource_type=ResourceType.AGENT,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                amount=agent_selection_request.get("required_agents", 1),
                security_context=security_context
            )

            if not agent_quota_check.allowed:
                return {
                    "success": False,
                    "error": "Agent quota exceeded",
                    "quota_result": agent_quota_check,
                    "max_allowed_agents": agent_quota_check.remaining
                }

            # 2. Erweitere Request mit Quota-Aware Selection
            enhanced_request = agent_selection_request.copy()
            enhanced_request["quota_aware_selection"] = {
                "max_agents": agent_quota_check.remaining,
                "prefer_efficient_agents": True,
                "consider_resource_usage": True,
                "quota_optimization_enabled": True
            }

            # 3. Füge Agent-Usage-Tracking hinzu
            enhanced_request["usage_tracking"] = {
                "track_agent_lifecycle": True,
                "track_resource_consumption": True,
                "track_performance_metrics": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "quota_result": agent_quota_check,
                "quota_constraints": enhanced_request["quota_aware_selection"]
            }

        except Exception as e:
            logger.error(f"Agent Selection Quota Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_task_decomposition_with_quotas(
        self,
        decomposition_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Task Decomposition um Quota-Constraints.

        Args:
            decomposition_request: Task Decomposition Request
            security_context: Security Context

        Returns:
            Enhanced Task Decomposition Result mit Quota-Constraints
        """
        try:
            logger.debug({
                "event": "enhance_task_decomposition_with_quotas_started",
                "task_id": decomposition_request.get("task_id"),
                "user_id": security_context.user_id
            })

            # 1. Quota-Check für Task Decomposition
            task_quota_check = await self.quota_management_engine.check_quota_with_security(
                resource_type=ResourceType.TASK,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                amount=decomposition_request.get("estimated_subtasks", 5),
                security_context=security_context
            )

            if not task_quota_check.allowed:
                return {
                    "success": False,
                    "error": "Task decomposition quota exceeded",
                    "quota_result": task_quota_check,
                    "max_allowed_subtasks": task_quota_check.remaining
                }

            # 2. Erweitere Request mit Quota-Aware Decomposition
            enhanced_request = decomposition_request.copy()
            enhanced_request["quota_aware_decomposition"] = {
                "max_subtasks": task_quota_check.remaining,
                "optimize_for_efficiency": True,
                "consider_resource_limits": True,
                "quota_balanced_splitting": True
            }

            # 3. Füge Task-Resource-Estimation hinzu
            enhanced_request["resource_estimation"] = {
                "estimate_agent_requirements": True,
                "estimate_execution_time": True,
                "estimate_resource_consumption": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "quota_result": task_quota_check,
                "quota_constraints": enhanced_request["quota_aware_decomposition"]
            }

        except Exception as e:
            logger.error(f"Task Decomposition Quota Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_llm_client_with_quotas(
        self,
        llm_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert LLM Client um Quota-Management.

        Args:
            llm_request: LLM Request
            security_context: Security Context

        Returns:
            Enhanced LLM Request Result mit Quota-Management
        """
        try:
            logger.debug({
                "event": "enhance_llm_client_with_quotas_started",
                "request_id": llm_request.get("request_id"),
                "user_id": security_context.user_id
            })

            # 1. Quota-Check für LLM Request
            llm_quota_check = await self.quota_management_engine.check_quota_with_security(
                resource_type=ResourceType.LLM_REQUEST,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                amount=1,
                security_context=security_context
            )

            if not llm_quota_check.allowed:
                return {
                    "success": False,
                    "error": "LLM request quota exceeded",
                    "quota_result": llm_quota_check,
                    "retry_after_seconds": llm_quota_check.retry_after_seconds
                }

            # 2. Erweitere Request mit Quota-Aware Features
            enhanced_request = llm_request.copy()
            enhanced_request["quota_features"] = {
                "track_token_usage": True,
                "optimize_for_efficiency": True,
                "respect_rate_limits": True,
                "quota_aware_caching": True
            }

            # 3. Füge Cost-Tracking hinzu
            enhanced_request["cost_tracking"] = {
                "track_token_costs": True,
                "track_request_costs": True,
                "budget_monitoring": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "quota_result": llm_quota_check
            }

        except Exception as e:
            logger.error(f"LLM Client Quota Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def enhance_performance_prediction_with_quotas(
        self,
        prediction_request: dict[str, Any],
        security_context: SecurityContext
    ) -> dict[str, Any]:
        """Erweitert Performance Prediction um Quota-Vorhersagen.

        Args:
            prediction_request: Performance Prediction Request
            security_context: Security Context

        Returns:
            Enhanced Performance Prediction Result mit Quota-Vorhersagen
        """
        try:
            logger.debug({
                "event": "enhance_performance_prediction_with_quotas_started",
                "prediction_id": prediction_request.get("prediction_id"),
                "user_id": security_context.user_id
            })

            # 1. Quota-Check für Performance Prediction
            prediction_quota_check = await self.quota_management_engine.check_quota_with_security(
                resource_type=ResourceType.API_CALL,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                amount=1,
                security_context=security_context
            )

            if not prediction_quota_check.allowed:
                return {
                    "success": False,
                    "error": "Performance prediction quota exceeded",
                    "quota_result": prediction_quota_check
                }

            # 2. Erweitere Request mit Quota-Prediction-Features
            enhanced_request = prediction_request.copy()
            enhanced_request["quota_prediction"] = {
                "predict_quota_usage": True,
                "predict_resource_exhaustion": True,
                "optimize_for_quota_efficiency": True,
                "include_quota_recommendations": True
            }

            # 3. Füge Quota-Analytics-Integration hinzu
            enhanced_request["analytics_integration"] = {
                "use_quota_history": True,
                "consider_usage_patterns": True,
                "include_trend_analysis": True
            }

            return {
                "success": True,
                "enhanced_request": enhanced_request,
                "quota_result": prediction_quota_check
            }

        except Exception as e:
            logger.error(f"Performance Prediction Quota Enhancement fehlgeschlagen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _calculate_max_agents_for_quota(self, security_context: SecurityContext) -> int:
        """Berechnet maximale Anzahl Agents basierend auf Quota."""
        try:
            # Vereinfachte Berechnung - in Realität würde dies aus Quota-Limits kommen
            if security_context.tenant_id:
                return 50  # Tenant-Level
            return 10  # User-Level

        except Exception as e:
            logger.error(f"Max agents calculation fehlgeschlagen: {e}")
            return 5  # Conservative fallback

    def _calculate_max_tasks_for_quota(self, security_context: SecurityContext) -> int:
        """Berechnet maximale Anzahl Tasks basierend auf Quota."""
        try:
            # Vereinfachte Berechnung
            if security_context.tenant_id:
                return 500  # Tenant-Level
            return 100  # User-Level

        except Exception as e:
            logger.error(f"Max tasks calculation fehlgeschlagen: {e}")
            return 50  # Conservative fallback

    def _calculate_max_execution_time_for_quota(self, security_context: SecurityContext) -> int:
        """Berechnet maximale Execution-Zeit basierend auf Quota."""
        try:
            # Vereinfachte Berechnung in Minuten
            if security_context.tenant_id:
                return 120  # 2 Stunden für Tenant
            return 30   # 30 Minuten für User

        except Exception as e:
            logger.error(f"Max execution time calculation fehlgeschlagen: {e}")
            return 15  # Conservative fallback

    async def track_service_usage(
        self,
        service_name: str,
        operation: str,
        security_context: SecurityContext,
        resource_usage: dict[str, Any],
        response_time_ms: float,
        success: bool
    ) -> None:
        """Trackt Service-Usage für Quota-Analytics.

        Args:
            service_name: Service-Name
            operation: Operation-Name
            security_context: Security Context
            resource_usage: Resource-Usage-Informationen
            response_time_ms: Response-Zeit
            success: Erfolg-Status
        """
        try:
            # Bestimme Resource-Type basierend auf Service
            resource_type_mapping = {
                "orchestrator_service": ResourceType.TASK,
                "policy_aware_selection": ResourceType.AGENT,
                "task_decomposition_engine": ResourceType.TASK,
                "llm_client_infrastructure": ResourceType.LLM_REQUEST,
                "performance_prediction": ResourceType.API_CALL
            }

            resource_type = resource_type_mapping.get(service_name, ResourceType.API_CALL)

            # Tracke Usage in Analytics Engine
            await self.quota_management_engine.analytics_engine.track_quota_usage(
                quota_id=f"{service_name}_{operation}",
                resource_type=resource_type,
                scope=QuotaScope.TENANT if security_context.tenant_id else QuotaScope.USER,
                scope_id=security_context.tenant_id or security_context.user_id,
                usage_amount=resource_usage.get("amount", 1),
                response_time_ms=response_time_ms,
                success=success
            )

            self._quota_aware_operations += 1

        except Exception as e:
            logger.error(f"Service usage tracking fehlgeschlagen: {e}")

    def _update_integration_performance_stats(self, integration_time_ms: float) -> None:
        """Aktualisiert Integration-Performance-Statistiken."""
        self._integration_count += 1
        self._total_integration_time_ms += integration_time_ms

    def get_performance_stats(self) -> dict[str, Any]:
        """Gibt Performance-Statistiken zurück."""
        avg_integration_time = (
            self._total_integration_time_ms / self._integration_count
            if self._integration_count > 0 else 0.0
        )

        # Hole Statistiken von Komponenten
        quota_engine_stats = self.quota_management_engine.get_performance_metrics()

        return {
            "service_integration": {
                "total_integrations": self._integration_count,
                "avg_integration_time_ms": avg_integration_time,
                "quota_aware_operations": self._quota_aware_operations,
                "meets_integration_sla": avg_integration_time < 50.0,  # < 50ms SLA
                "service_integration_enabled": self.enable_service_integration,
                "quota_aware_orchestration_enabled": self.enable_quota_aware_orchestration,
                "quota_aware_agent_selection_enabled": self.enable_quota_aware_agent_selection,
                "quota_aware_task_decomposition_enabled": self.enable_quota_aware_task_decomposition
            },
            "quota_management_engine": {
                "total_quota_checks": quota_engine_stats.total_quota_checks,
                "avg_quota_check_time_ms": quota_engine_stats.avg_quota_check_time_ms,
                "meets_quota_sla": quota_engine_stats.meets_quota_sla,
                "cache_hit_rate": quota_engine_stats.cache_hit_rate
            }
        }
