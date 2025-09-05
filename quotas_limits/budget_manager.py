# backend/quotas_limits/budget_manager.py
"""Budget Manager für Keiko Personal Assistant

Implementiert Cost-Tracking, Budget-Caps, Timeout-Propagation
und automatische Budget-Übertragung zwischen Parent/Child-Agents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from kei_logging import get_logger
from observability import trace_function

from .base_manager import BaseManager, ManagerConfig
from .constants import (
    DEFAULT_BUDGET_ALERT_THRESHOLD,
    MAX_BUDGET_AMOUNT,
    MIN_BUDGET_AMOUNT,
)
from .utils import (
    calculate_percentage,
    clamp_value,
    format_decimal_currency,
    generate_uuid,
)

logger = get_logger(__name__)


class BudgetType(str, Enum):
    """Typen von Budgets."""
    MONETARY = "monetary"
    COMPUTE_CREDITS = "compute_credits"
    API_CALLS = "api_calls"
    DATA_TRANSFER = "data_transfer"
    STORAGE_USAGE = "storage_usage"
    PROCESSING_TIME = "processing_time"


class BudgetStatus(str, Enum):
    """Status von Budgets."""
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class TransferType(str, Enum):
    """Typen von Budget-Transfers."""
    ALLOCATION = "allocation"
    REALLOCATION = "reallocation"
    INHERITANCE = "inheritance"
    EMERGENCY = "emergency"
    REFUND = "refund"


class CostCategory(str, Enum):
    """Kategorien von Kosten."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    DATA_PROCESSING = "data_processing"
    ML_INFERENCE = "ml_inference"
    CUSTOM = "custom"


@dataclass
class OperationCost:
    """Kosten einer Operation."""
    operation_id: str
    operation_type: str
    category: CostCategory

    # Kosten-Details
    base_cost: Decimal
    variable_cost: Decimal = Decimal("0.0")
    total_cost: Decimal = field(init=False)

    # Ressourcen-Verbrauch
    compute_time_seconds: float = 0.0
    data_volume_mb: float = 0.0
    api_calls_count: int = 0
    storage_mb_hours: float = 0.0

    # Metadaten
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_id: str | None = None
    capability_id: str | None = None
    tenant_id: str | None = None

    def __post_init__(self):
        """Berechnet Gesamtkosten."""
        self.total_cost = self.base_cost + self.variable_cost

    def add_variable_cost(self, amount: Decimal, reason: str = "") -> None:
        """Fügt variable Kosten hinzu."""
        self.variable_cost += amount
        self.total_cost = self.base_cost + self.variable_cost

        if reason:
            logger.debug(f"Variable Kosten hinzugefügt: {amount} ({reason})")


@dataclass
class Budget:
    """Budget-Definition."""
    budget_id: str
    name: str
    description: str
    budget_type: BudgetType

    # Budget-Beträge
    total_amount: Decimal
    allocated_amount: Decimal = Decimal("0.0")
    used_amount: Decimal = Decimal("0.0")
    reserved_amount: Decimal = Decimal("0.0")

    # Hierarchie
    parent_budget_id: str | None = None
    child_budget_ids: set[str] = field(default_factory=set)

    # Scope
    tenant_id: str | None = None
    agent_id: str | None = None
    capability_id: str | None = None

    # Gültigkeit
    status: BudgetStatus = BudgetStatus.ACTIVE
    valid_from: datetime = field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None

    # Auto-Renewal
    auto_renewal: bool = False
    renewal_amount: Decimal | None = None
    renewal_period_days: int = 30

    # Alerts
    alert_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])  # 50%, 80%, 95%

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = None
    tags: set[str] = field(default_factory=set)

    @property
    def available_amount(self) -> Decimal:
        """Gibt verfügbaren Betrag zurück."""
        return self.total_amount - self.used_amount - self.reserved_amount

    @property
    def utilization_percentage(self) -> float:
        """Gibt Auslastung in Prozent zurück."""
        if self.total_amount == 0:
            return 0.0
        return float((self.used_amount / self.total_amount) * 100)

    @property
    def is_exhausted(self) -> bool:
        """Prüft, ob Budget erschöpft ist."""
        return self.available_amount <= 0

    @property
    def is_expired(self) -> bool:
        """Prüft, ob Budget abgelaufen ist."""
        if not self.valid_until:
            return False
        return datetime.now(UTC) > self.valid_until

    def check_alert_threshold(self) -> float | None:
        """Prüft, ob Alert-Schwellwert erreicht wurde."""
        utilization = self.utilization_percentage / 100.0

        for threshold in sorted(self.alert_thresholds, reverse=True):
            if utilization >= threshold:
                return threshold

        return None

    def can_spend(self, amount: Decimal) -> bool:
        """Prüft, ob Betrag ausgegeben werden kann."""
        return (
            self.status == BudgetStatus.ACTIVE and
            not self.is_expired and
            self.available_amount >= amount
        )


@dataclass
class BudgetAllocation:
    """Budget-Zuteilung."""
    allocation_id: str
    source_budget_id: str
    target_budget_id: str
    amount: Decimal
    allocation_type: TransferType
    reason: str

    # Status
    status: str = "pending"  # pending, completed, failed, cancelled

    # Zeitstempel
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Metadaten
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetUsage:
    """Budget-Nutzung."""
    usage_id: str
    budget_id: str
    operation_cost: OperationCost
    amount_charged: Decimal

    # Zeitstempel
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadaten
    agent_id: str | None = None
    capability_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetTransfer:
    """Budget-Transfer zwischen Agents."""
    transfer_id: str
    from_agent_id: str
    to_agent_id: str
    budget_type: BudgetType
    amount: Decimal
    transfer_type: TransferType
    reason: str

    # Status
    status: str = "pending"

    # Zeitstempel
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Approval
    requires_approval: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None


@dataclass
class BudgetPropagationRule:
    """Regel für Budget-Propagation."""
    rule_id: str
    name: str
    description: str

    # Bedingungen
    source_scope: str  # tenant, agent, capability
    target_scope: str
    budget_type: BudgetType

    # Propagation-Logik
    propagation_percentage: float = 100.0  # Prozent des Parent-Budgets
    max_amount: Decimal | None = None
    min_amount: Decimal | None = None

    # Trigger
    trigger_on_creation: bool = True
    trigger_on_exhaustion: bool = True
    trigger_on_threshold: float | None = None  # z.B. 0.9 für 90%

    # Gültigkeit
    enabled: bool = True
    priority: int = 100

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class CostTracker:
    """Tracker für Operation-Kosten."""

    def __init__(self):
        """Initialisiert Cost Tracker."""
        self._cost_history: list[OperationCost] = []
        self._cost_cache: dict[str, Decimal] = {}
        self._cache_ttl = 300  # 5 Minuten

        # Pricing-Modell (vereinfacht)
        self._pricing_model = {
            CostCategory.COMPUTE: Decimal("0.001"),  # Pro Sekunde
            CostCategory.STORAGE: Decimal("0.0001"),  # Pro MB-Stunde
            CostCategory.NETWORK: Decimal("0.01"),   # Pro MB
            CostCategory.API_CALLS: Decimal("0.001"), # Pro Call
            CostCategory.DATA_PROCESSING: Decimal("0.005"), # Pro MB
            CostCategory.ML_INFERENCE: Decimal("0.01"),     # Pro Request
        }

    def calculate_operation_cost(
        self,
        operation_type: str,
        category: CostCategory,
        compute_time_seconds: float = 0.0,
        data_volume_mb: float = 0.0,
        api_calls_count: int = 0,
        storage_mb_hours: float = 0.0,
        custom_multiplier: float = 1.0
    ) -> OperationCost:
        """Berechnet Kosten für Operation."""
        import uuid

        base_cost = self._pricing_model.get(category, Decimal("0.001"))

        # Berechne variable Kosten basierend auf Ressourcen-Verbrauch
        variable_cost = Decimal("0.0")

        if compute_time_seconds > 0:
            variable_cost += Decimal(str(compute_time_seconds)) * self._pricing_model[CostCategory.COMPUTE]

        if data_volume_mb > 0:
            if category == CostCategory.NETWORK:
                variable_cost += Decimal(str(data_volume_mb)) * self._pricing_model[CostCategory.NETWORK]
            elif category == CostCategory.DATA_PROCESSING:
                variable_cost += Decimal(str(data_volume_mb)) * self._pricing_model[CostCategory.DATA_PROCESSING]

        if api_calls_count > 0:
            variable_cost += Decimal(str(api_calls_count)) * self._pricing_model[CostCategory.API_CALLS]

        if storage_mb_hours > 0:
            variable_cost += Decimal(str(storage_mb_hours)) * self._pricing_model[CostCategory.STORAGE]

        # Wende Custom-Multiplier an
        if abs(custom_multiplier - 1.0) > 1e-6:
            variable_cost *= Decimal(str(custom_multiplier))

        operation_cost = OperationCost(
            operation_id=str(uuid.uuid4()),
            operation_type=operation_type,
            category=category,
            base_cost=base_cost,
            variable_cost=variable_cost,
            compute_time_seconds=compute_time_seconds,
            data_volume_mb=data_volume_mb,
            api_calls_count=api_calls_count,
            storage_mb_hours=storage_mb_hours
        )

        self._cost_history.append(operation_cost)
        return operation_cost

    def get_cost_summary(
        self,
        agent_id: str | None = None,
        capability_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> dict[str, Any]:
        """Gibt Kosten-Zusammenfassung zurück."""
        filtered_costs = self._cost_history

        # Filter anwenden
        if agent_id:
            filtered_costs = [c for c in filtered_costs if c.agent_id == agent_id]

        if capability_id:
            filtered_costs = [c for c in filtered_costs if c.capability_id == capability_id]

        if start_time:
            filtered_costs = [c for c in filtered_costs if c.timestamp >= start_time]

        if end_time:
            filtered_costs = [c for c in filtered_costs if c.timestamp <= end_time]

        # Berechne Zusammenfassung
        total_cost = sum(cost.total_cost for cost in filtered_costs)
        cost_by_category = {}

        for cost in filtered_costs:
            category = cost.category.value
            if category not in cost_by_category:
                cost_by_category[category] = Decimal("0.0")
            cost_by_category[category] += cost.total_cost

        return {
            "total_cost": total_cost,
            "cost_by_category": cost_by_category,
            "operation_count": len(filtered_costs),
            "average_cost": total_cost / len(filtered_costs) if filtered_costs else Decimal("0.0"),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        }


class BudgetManager(BaseManager):
    """Manager für Budget-Verwaltung."""

    def __init__(self, config: ManagerConfig | None = None):
        """Initialisiert Budget Manager."""
        super().__init__(config)

        self._budgets: dict[str, Budget] = {}
        self._allocations: dict[str, BudgetAllocation] = {}
        self._usage_history: list[BudgetUsage] = []
        self._transfers: dict[str, BudgetTransfer] = {}
        self._propagation_rules: dict[str, BudgetPropagationRule] = {}

        # Transfer-Statistiken
        self._total_transfers_completed: int = 0
        self._total_transfers_failed: int = 0

        self.cost_tracker = CostTracker()

        # Zusätzliche Locks für spezifische Operationen
        self._budget_lock = asyncio.Lock()
        self._transfer_lock = asyncio.Lock()

    def get_manager_type(self) -> str:
        """Gibt Manager-Typ zurück."""
        return "BudgetManager"

    def create_budget(
        self,
        name: str,
        description: str,
        budget_type: BudgetType,
        total_amount: Decimal,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        capability_id: str | None = None,
        parent_budget_id: str | None = None,
        valid_until: datetime | None = None
    ) -> str:
        """Erstellt neues Budget."""
        # Validiere Budget-Betrag
        total_amount = clamp_value(float(total_amount), float(MIN_BUDGET_AMOUNT), float(MAX_BUDGET_AMOUNT))
        total_amount = Decimal(str(total_amount))

        budget_id = generate_uuid()

        budget = Budget(
            budget_id=budget_id,
            name=name,
            description=description,
            budget_type=budget_type,
            total_amount=total_amount,
            tenant_id=tenant_id,
            agent_id=agent_id,
            capability_id=capability_id,
            parent_budget_id=parent_budget_id,
            valid_until=valid_until
        )

        self._budgets[budget_id] = budget

        # Füge zu Parent-Budget hinzu
        if parent_budget_id and parent_budget_id in self._budgets:
            self._budgets[parent_budget_id].child_budget_ids.add(budget_id)

        logger.info(f"Budget erstellt: {budget_id} ({name}) - {format_decimal_currency(total_amount)}")
        return budget_id

    @trace_function("budget.charge")
    async def charge_budget(
        self,
        budget_id: str,
        operation_cost: OperationCost,
        agent_id: str | None = None
    ) -> bool:
        """Belastet Budget mit Operation-Kosten."""
        return await self.execute_operation(
            operation_name="charge_budget",
            operation_func=self._charge_budget_impl,
            budget_id=budget_id,
            operation_cost=operation_cost,
            agent_id=agent_id
        )

    async def _charge_budget_impl(
        self,
        budget_id: str,
        operation_cost: OperationCost,
        agent_id: str | None
    ) -> bool:
        """Implementierung der Budget-Belastung."""
        async with self._budget_lock:
            budget = self._budgets.get(budget_id)
            if not budget:
                logger.error(f"Budget nicht gefunden: {budget_id}")
                return False

            if not budget.can_spend(operation_cost.total_cost):
                logger.warning(f"Budget erschöpft: {budget_id}")
                return False

            # Belasten
            budget.used_amount += operation_cost.total_cost

            # Usage-Eintrag erstellen
            usage = BudgetUsage(
                usage_id=generate_uuid(),
                budget_id=budget_id,
                operation_cost=operation_cost,
                amount_charged=operation_cost.total_cost,
                agent_id=agent_id
            )

            self._usage_history.append(usage)

            # Prüfe Alert-Schwellwerte
            usage_percentage = calculate_percentage(float(budget.used_amount), float(budget.total_amount))
            if usage_percentage >= DEFAULT_BUDGET_ALERT_THRESHOLD * 100:
                logger.warning(f"Budget-Alert: {budget_id} hat {usage_percentage:.1f}% erreicht")

            # Prüfe Erschöpfung
            if budget.is_exhausted:
                budget.status = BudgetStatus.EXHAUSTED
                await self._handle_budget_exhaustion(budget_id)

            return True

    async def transfer_budget(
        self,
        from_agent_id: str,
        to_agent_id: str,
        budget_type: BudgetType,
        amount: Decimal,
        transfer_type: TransferType = TransferType.REALLOCATION,
        reason: str = ""
    ) -> str | None:
        """Überträgt Budget zwischen Agents."""
        async with self._transfer_lock:
            # Finde Source-Budget
            source_budget = None
            for budget in self._budgets.values():
                if budget.agent_id == from_agent_id and budget.budget_type == budget_type:
                    source_budget = budget
                    break

            if not source_budget or not source_budget.can_spend(amount):
                logger.error("Budget-Transfer fehlgeschlagen: Unzureichendes Budget")
                return None

            # Finde oder erstelle Target-Budget
            target_budget = None
            for budget in self._budgets.values():
                if budget.agent_id == to_agent_id and budget.budget_type == budget_type:
                    target_budget = budget
                    break

            if not target_budget:
                # Erstelle neues Budget für Target-Agent
                target_budget_id = self.create_budget(
                    name=f"Auto-created budget for {to_agent_id}",
                    description=f"Budget created via transfer from {from_agent_id}",
                    budget_type=budget_type,
                    total_amount=amount,
                    agent_id=to_agent_id
                )
                target_budget = self._budgets[target_budget_id]

            # Führe Transfer durch
            transfer_id = str(__import__("uuid").uuid4())

            transfer = BudgetTransfer(
                transfer_id=transfer_id,
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                budget_type=budget_type,
                amount=amount,
                transfer_type=transfer_type,
                reason=reason
            )

            # Aktualisiere Budgets
            source_budget.used_amount += amount  # Reduziert verfügbares Budget
            target_budget.total_amount += amount  # Erhöht Ziel-Budget

            transfer.status = "completed"
            transfer.completed_at = datetime.now(UTC)

            self._transfers[transfer_id] = transfer
            self._total_transfers_completed += 1

            logger.info(f"Budget-Transfer abgeschlossen: {transfer_id}")
            return transfer_id

    async def _handle_budget_exhaustion(self, budget_id: str) -> None:
        """Behandelt Budget-Erschöpfung."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return

        # Prüfe Propagation-Regeln
        for rule in self._propagation_rules.values():
            if (rule.enabled and
                rule.trigger_on_exhaustion and
                rule.budget_type == budget.budget_type):

                await self._apply_propagation_rule(rule, budget_id)

        # Auto-Renewal prüfen
        if budget.auto_renewal and budget.renewal_amount:
            await self._renew_budget(budget_id)

    async def _apply_propagation_rule(self, rule: BudgetPropagationRule, budget_id: str) -> None:
        """Wendet Propagation-Regel an."""
        budget = self._budgets.get(budget_id)
        if not budget or not budget.parent_budget_id:
            return

        parent_budget = self._budgets.get(budget.parent_budget_id)
        if not parent_budget:
            return

        # Berechne Propagation-Betrag
        propagation_amount = parent_budget.available_amount * Decimal(str(rule.propagation_percentage / 100.0))

        if rule.max_amount:
            propagation_amount = min(propagation_amount, rule.max_amount)

        if rule.min_amount:
            propagation_amount = max(propagation_amount, rule.min_amount)

        if propagation_amount > 0:
            # Führe automatischen Transfer durch
            await self.transfer_budget(
                from_agent_id=parent_budget.agent_id or "system",
                to_agent_id=budget.agent_id or "system",
                budget_type=budget.budget_type,
                amount=propagation_amount,
                transfer_type=TransferType.INHERITANCE,
                reason=f"Auto-propagation via rule {rule.name}"
            )

    async def _renew_budget(self, budget_id: str) -> None:
        """Erneuert Budget automatisch."""
        budget = self._budgets.get(budget_id)
        if not budget or not budget.renewal_amount:
            return

        budget.total_amount += budget.renewal_amount
        budget.used_amount = Decimal("0.0")
        budget.status = BudgetStatus.ACTIVE
        budget.valid_until = datetime.now(UTC) + timedelta(days=budget.renewal_period_days)

        logger.info(f"Budget erneuert: {budget_id}")

    def get_budget_status(self, budget_id: str) -> dict[str, Any] | None:
        """Gibt Budget-Status zurück."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return None

        return {
            "budget_id": budget.budget_id,
            "name": budget.name,
            "budget_type": budget.budget_type.value,
            "status": budget.status.value,
            "total_amount": float(budget.total_amount),
            "used_amount": float(budget.used_amount),
            "available_amount": float(budget.available_amount),
            "utilization_percentage": budget.utilization_percentage,
            "is_exhausted": budget.is_exhausted,
            "is_expired": budget.is_expired,
            "alert_threshold": budget.check_alert_threshold(),
            "child_budgets": len(budget.child_budget_ids),
            "created_at": budget.created_at.isoformat(),
            "valid_until": budget.valid_until.isoformat() if budget.valid_until else None
        }

    def get_agent_budgets(self, agent_id: str) -> list[dict[str, Any]]:
        """Gibt alle Budgets für Agent zurück."""
        agent_budgets = [
            budget for budget in self._budgets.values()
            if budget.agent_id == agent_id
        ]

        return [self.get_budget_status(budget.budget_id) for budget in agent_budgets]

    def get_budget_statistics(self) -> dict[str, Any]:
        """Gibt Budget-spezifische Statistiken zurück."""
        base_status = self.get_status()

        active_budgets = sum(1 for b in self._budgets.values() if b.status == BudgetStatus.ACTIVE)
        exhausted_budgets = sum(1 for b in self._budgets.values() if b.status == BudgetStatus.EXHAUSTED)

        budget_stats = {
            "total_budgets": len(self._budgets),
            "active_budgets": active_budgets,
            "exhausted_budgets": exhausted_budgets,
            "total_transfers": len(self._transfers),
            "propagation_rules": len(self._propagation_rules),
            "usage_entries": len(self._usage_history),
            "total_amount_managed": sum(float(b.total_amount) for b in self._budgets.values()),
            "total_amount_used": sum(float(b.used_amount) for b in self._budgets.values())
        }

        # Kombiniere mit Base-Manager-Statistiken
        base_status.update(budget_stats)
        return base_status


# Globale Budget Manager Instanz
budget_manager = BudgetManager()
