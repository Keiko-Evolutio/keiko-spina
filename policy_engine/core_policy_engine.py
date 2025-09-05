# backend/policy_engine/core_policy_engine.py
"""Core Policy Engine für Keiko Personal Assistant

Implementiert zentrale Policy-Evaluation, -Management und -Enforcement
mit asynchroner Verarbeitung und Circuit-Breaker-Pattern.
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from kei_logging import get_logger
from observability import trace_function

if TYPE_CHECKING:
    from collections.abc import Callable

# Oso-Integration (optional) - konsolidiert aus mesh/policy_engine.py
try:
    from oso import Oso
    _oso_available = True
except ImportError:
    _oso_available = False
    Oso = None  # type: ignore

# Final-Deklaration außerhalb des try-except Blocks
OSO_AVAILABLE: Final[bool] = _oso_available

logger = get_logger(__name__)

# Policy-Entscheidungen (konsolidiert aus mesh_constants)
POLICY_DECISION_ALLOW: Final[str] = "allow"
POLICY_DECISION_DENY: Final[str] = "deny"
POLICY_DECISION_UNKNOWN: Final[str] = "unknown"


class PolicyType(str, Enum):
    """Typen von Policies."""
    SAFETY_GUARDRAILS = "safety_guardrails"
    COMPLIANCE = "compliance"
    PII_REDACTION = "pii_redaction"
    DATA_MINIMIZATION = "data_minimization"
    PROMPT_VALIDATION = "prompt_validation"
    CONTENT_FILTERING = "content_filtering"
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"


class PolicyEffect(str, Enum):
    """Effekt einer Policy-Entscheidung."""
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    AUDIT = "audit"
    WARN = "warn"


class PolicyPriority(int, Enum):
    """Priorität von Policies."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class PolicyContext:
    """Kontext für Policy-Evaluation."""
    user_id: str | None = None
    tenant_id: str | None = None
    agent_id: str | None = None
    operation: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Gibt Kontext-Attribut zurück."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)


@dataclass
class PolicyRule:
    """Definition einer Policy-Regel."""
    rule_id: str
    name: str
    description: str
    policy_type: PolicyType
    effect: PolicyEffect
    priority: PolicyPriority = PolicyPriority.NORMAL

    # Bedingungen
    conditions: dict[str, Any] = field(default_factory=dict)

    # Aktionen
    actions: dict[str, Any] = field(default_factory=dict)

    # Gültigkeit
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Metadaten
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = None
    version: str = "1.0.0"

    def is_valid(self) -> bool:
        """Prüft, ob Regel gültig ist."""
        if not self.enabled:
            return False

        now = datetime.now(UTC)

        if self.valid_from and now < self.valid_from:
            return False

        return not (self.valid_until and now > self.valid_until)

    def matches_context(self, context: PolicyContext) -> bool:
        """Prüft, ob Regel auf Kontext zutrifft."""
        if not self.is_valid():
            return False

        for condition_key, condition_value in self.conditions.items():
            context_value = context.get_attribute(condition_key)

            if not self._evaluate_condition(context_value, condition_value):
                return False

        return True

    def _evaluate_condition(self, context_value: Any, condition_value: Any) -> bool:
        """Evaluiert einzelne Bedingung."""
        if isinstance(condition_value, dict):
            # Komplexe Bedingung (z.B. {"operator": "in", "values": [...]})
            operator = condition_value.get("operator", "equals")

            if operator == "equals":
                return context_value == condition_value.get("value")
            if operator == "in":
                return context_value in condition_value.get("values", [])
            if operator == "not_in":
                return context_value not in condition_value.get("values", [])
            if operator == "contains":
                return condition_value.get("value") in str(context_value)
            if operator == "regex":
                import re
                pattern = condition_value.get("pattern")
                return bool(re.search(pattern, str(context_value)))

        else:
            # Einfache Gleichheitsprüfung
            return context_value == condition_value

        return False


@dataclass
class PolicyDecision:
    """Ergebnis einer Policy-Entscheidung."""
    effect: PolicyEffect
    rule_id: str | None = None
    reason: str = ""
    modifications: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluationResult:
    """Ergebnis einer Policy-Evaluation."""
    decisions: list[PolicyDecision] = field(default_factory=list)
    final_effect: PolicyEffect = PolicyEffect.ALLOW
    evaluation_time_ms: float = 0.0
    context: PolicyContext | None = None

    @property
    def is_allowed(self) -> bool:
        """Prüft, ob Operation erlaubt ist."""
        return self.final_effect in [PolicyEffect.ALLOW, PolicyEffect.MODIFY, PolicyEffect.WARN]

    @property
    def requires_modification(self) -> bool:
        """Prüft, ob Modifikation erforderlich ist."""
        return self.final_effect == PolicyEffect.MODIFY

    @property
    def has_warnings(self) -> bool:
        """Prüft, ob Warnungen vorhanden sind."""
        return self.final_effect == PolicyEffect.WARN or any(
            decision.effect == PolicyEffect.WARN for decision in self.decisions
        )


class PolicyEvaluator(ABC):
    """Basis-Klasse für Policy-Evaluatoren."""

    @abstractmethod
    async def evaluate(self, context: PolicyContext, rules: list[PolicyRule]) -> list[PolicyDecision]:
        """Evaluiert Policies für gegebenen Kontext."""


class DefaultPolicyEvaluator(PolicyEvaluator):
    """Standard-Policy-Evaluator."""

    async def evaluate(self, context: PolicyContext, rules: list[PolicyRule]) -> list[PolicyDecision]:
        """Evaluiert Policies sequenziell."""
        decisions = []

        # Sortiere Regeln nach Priorität (höchste zuerst)
        sorted_rules = sorted(rules, key=lambda r: r.priority.value, reverse=True)

        for rule in sorted_rules:
            if rule.matches_context(context):
                decision = PolicyDecision(
                    effect=rule.effect,
                    rule_id=rule.rule_id,
                    reason=f"Regel {rule.name} angewendet",
                    modifications=rule.actions.copy(),
                    metadata={"rule_priority": rule.priority.value}
                )
                decisions.append(decision)

        return decisions


class CircuitBreaker:
    """Circuit Breaker für Policy-Evaluation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """Führt Funktion mit Circuit Breaker aus."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Prüft, ob Reset-Versuch gemacht werden soll."""
        if not self.last_failure_time:
            return False

        return (datetime.now(UTC) - self.last_failure_time).total_seconds() > self.recovery_timeout

    def _on_success(self):
        """Behandelt erfolgreiche Ausführung."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Behandelt fehlgeschlagene Ausführung."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class PolicyEngine:
    """Zentrale Policy Engine."""

    def __init__(self, policy_source: str | None = None):
        """Initialisiert Policy Engine mit optionaler Oso-Integration."""
        self._rules: dict[PolicyType, list[PolicyRule]] = {}
        self._evaluators: dict[PolicyType, PolicyEvaluator] = {}
        self._circuit_breakers: dict[PolicyType, CircuitBreaker] = {}
        self._evaluation_cache: dict[str, PolicyEvaluationResult] = {}
        self._cache_ttl = 300  # 5 Minuten

        # Oso-Integration (konsolidiert aus mesh/policy_engine.py)
        self._oso = Oso() if OSO_AVAILABLE and Oso is not None else None
        self._tool_access_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        if self._oso and policy_source:
            try:
                self._oso.load_str(policy_source)
            except Exception:
                self._oso = None

        # Statistiken
        self._evaluation_count = 0
        self._cache_hits = 0
        self._circuit_breaker_trips = 0

        # Standard-Evaluator registrieren
        default_evaluator = DefaultPolicyEvaluator()
        for policy_type in PolicyType:
            self._evaluators[policy_type] = default_evaluator
            self._circuit_breakers[policy_type] = CircuitBreaker()

    def register_rule(self, rule: PolicyRule) -> None:
        """Registriert Policy-Regel."""
        if rule.policy_type not in self._rules:
            self._rules[rule.policy_type] = []

        # Entferne existierende Regel mit gleicher ID
        self._rules[rule.policy_type] = [
            r for r in self._rules[rule.policy_type] if r.rule_id != rule.rule_id
        ]

        self._rules[rule.policy_type].append(rule)
        self._invalidate_cache()

        logger.info(f"Policy-Regel registriert: {rule.rule_id} ({rule.policy_type.value})")

    def register_evaluator(self, policy_type: PolicyType, evaluator: PolicyEvaluator) -> None:
        """Registriert Policy-Evaluator."""
        self._evaluators[policy_type] = evaluator
        logger.info(f"Policy-Evaluator registriert für {policy_type.value}")

    @trace_function("policy.evaluate")
    async def evaluate(
        self,
        context: PolicyContext,
        policy_types: list[PolicyType] | None = None
    ) -> PolicyEvaluationResult:
        """Evaluiert Policies für gegebenen Kontext."""
        start_time = time.time()
        self._evaluation_count += 1

        # Cache-Check
        cached_result = self._check_cache(context, policy_types)
        if cached_result:
            return cached_result

        try:
            all_decisions = await self._evaluate_all_policy_types(context, policy_types)
            return self._create_evaluation_result(all_decisions, context, start_time)

        except Exception as e:
            return self._create_error_result(e, context, start_time)

    def _check_cache(self, context: PolicyContext, policy_types: list[PolicyType] | None) -> PolicyEvaluationResult | None:
        """Prüft Cache auf vorhandenes Ergebnis."""
        cache_key = self._get_cache_key(context, policy_types)
        cached_result = self._evaluation_cache.get(cache_key)

        if cached_result and self._is_cache_valid(cached_result):
            self._cache_hits += 1
            return cached_result

        return None

    async def _evaluate_all_policy_types(
        self,
        context: PolicyContext,
        policy_types: list[PolicyType] | None
    ) -> list[PolicyDecision]:
        """Evaluiert alle angegebenen Policy-Typen."""
        types_to_evaluate = policy_types or list(PolicyType)
        all_decisions = []

        for policy_type in types_to_evaluate:
            decisions = await self._evaluate_single_policy_type(context, policy_type)
            all_decisions.extend(decisions)

        return all_decisions

    async def _evaluate_single_policy_type(
        self,
        context: PolicyContext,
        policy_type: PolicyType
    ) -> list[PolicyDecision]:
        """Evaluiert einen einzelnen Policy-Typ."""
        rules = self._rules.get(policy_type, [])
        if not rules:
            return []

        evaluator = self._evaluators[policy_type]
        circuit_breaker = self._circuit_breakers[policy_type]

        try:
            return await circuit_breaker.call(evaluator.evaluate, context, rules)
        except Exception as e:
            return self._create_fallback_decision(policy_type, e)

    def _create_fallback_decision(self, policy_type: PolicyType, error: Exception) -> list[PolicyDecision]:
        """Erstellt Fallback-Entscheidung bei Fehlern."""
        logger.error(f"Policy-Evaluation fehlgeschlagen für {policy_type.value}: {error}")
        self._circuit_breaker_trips += 1

        fallback_decision = PolicyDecision(
            effect=PolicyEffect.WARN,
            reason=f"Policy-Evaluation fehlgeschlagen: {error!s}"
        )
        return [fallback_decision]

    def _create_evaluation_result(
        self,
        all_decisions: list[PolicyDecision],
        context: PolicyContext,
        start_time: float
    ) -> PolicyEvaluationResult:
        """Erstellt finales Evaluation-Ergebnis."""
        final_effect = self._determine_final_effect(all_decisions)
        execution_time = (time.time() - start_time) * 1000

        result = PolicyEvaluationResult(
            decisions=all_decisions,
            final_effect=final_effect,
            evaluation_time_ms=execution_time,
            context=context
        )

        # Cache aktualisieren
        cache_key = self._get_cache_key(context, None)
        self._evaluation_cache[cache_key] = result

        return result

    def _create_error_result(
        self,
        error: Exception,
        context: PolicyContext,
        start_time: float
    ) -> PolicyEvaluationResult:
        """Erstellt Error-Ergebnis bei kritischen Fehlern."""
        logger.error(f"Policy-Engine-Fehler: {error}")
        execution_time = (time.time() - start_time) * 1000

        return PolicyEvaluationResult(
            decisions=[PolicyDecision(
                effect=PolicyEffect.WARN,
                reason=f"Policy-Engine-Fehler: {error!s}"
            )],
            final_effect=PolicyEffect.WARN,
            evaluation_time_ms=execution_time,
            context=context
        )

    def _determine_final_effect(self, decisions: list[PolicyDecision]) -> PolicyEffect:
        """Bestimmt finalen Effekt basierend auf allen Entscheidungen."""
        if not decisions:
            return PolicyEffect.ALLOW

        # DENY hat höchste Priorität
        if any(d.effect == PolicyEffect.DENY for d in decisions):
            return PolicyEffect.DENY

        # MODIFY hat zweithöchste Priorität
        if any(d.effect == PolicyEffect.MODIFY for d in decisions):
            return PolicyEffect.MODIFY

        # WARN hat dritthöchste Priorität
        if any(d.effect == PolicyEffect.WARN for d in decisions):
            return PolicyEffect.WARN

        # AUDIT hat niedrigste Priorität
        if any(d.effect == PolicyEffect.AUDIT for d in decisions):
            return PolicyEffect.AUDIT

        return PolicyEffect.ALLOW

    def _get_cache_key(self, context: PolicyContext, policy_types: list[PolicyType] | None) -> str:
        """Generiert Cache-Key für Kontext."""
        key_data = {
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "agent_id": context.agent_id,
            "operation": context.operation,
            "resource_type": context.resource_type,
            "resource_id": context.resource_id,
            "policy_types": [pt.value for pt in (policy_types or [])]
        }

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_cache_valid(self, result: PolicyEvaluationResult) -> bool:
        """Prüft, ob Cache-Eintrag gültig ist."""
        if not result.context:
            return False

        age = (datetime.now(UTC) - result.context.timestamp).total_seconds()
        return age < self._cache_ttl

    def _invalidate_cache(self) -> None:
        """Invalidiert Policy-Cache."""
        self._evaluation_cache.clear()

    def get_rules(self, policy_type: PolicyType | None = None) -> list[PolicyRule]:
        """Gibt Policy-Regeln zurück."""
        if policy_type:
            return self._rules.get(policy_type, [])

        all_rules = []
        for rules in self._rules.values():
            all_rules.extend(rules)
        return all_rules

    def remove_rule(self, rule_id: str) -> bool:
        """Entfernt Policy-Regel."""
        for policy_type, rules in self._rules.items():
            original_count = len(rules)
            self._rules[policy_type] = [r for r in rules if r.rule_id != rule_id]

            if len(self._rules[policy_type]) < original_count:
                self._invalidate_cache()
                logger.info(f"Policy-Regel entfernt: {rule_id}")
                return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Gibt Policy-Engine-Statistiken zurück."""
        total_rules = sum(len(rules) for rules in self._rules.values())

        return {
            "total_rules": total_rules,
            "rules_by_type": {pt.value: len(self._rules.get(pt, [])) for pt in PolicyType},
            "evaluation_count": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(self._evaluation_count, 1),
            "circuit_breaker_trips": self._circuit_breaker_trips,
            "cache_size": len(self._evaluation_cache),
            "tool_access_cache_size": len(self._tool_access_cache),
            "oso_available": OSO_AVAILABLE
        }

    # Tool-Access-Control-Methoden (konsolidiert aus mesh/policy_engine.py)
    def allow_tool_access(self, subject: dict[str, Any], tool: dict[str, Any], ctx: dict[str, Any] | None = None) -> bool:
        """Prüft, ob Subjekt auf Tool zugreifen darf."""
        return self.check_policy(subject, "use", tool, ctx) == POLICY_DECISION_ALLOW

    def check_policy(self, subject: dict[str, Any], action: str, resource: dict[str, Any], context: dict[str, Any] | None = None) -> str:
        """Prüft Policy-Entscheidung mit Caching."""
        cache_key = f"{hash(str(subject))}-{action}-{hash(str(resource))}"

        with self._lock:
            cached = self._tool_access_cache.get(cache_key)
            if cached and time.time() - cached["timestamp"] < self._cache_ttl:
                return cached["decision"]

        decision = self._evaluate_tool_policy(subject, action, resource, context)

        with self._lock:
            self._tool_access_cache[cache_key] = {"decision": decision, "timestamp": time.time()}

        return decision

    def _evaluate_tool_policy(self, subject: dict[str, Any], action: str, resource: dict[str, Any], context: dict[str, Any] | None = None) -> str:
        """Evaluiert Tool-Policy mit Oso oder Fallback."""
        if self._oso:
            try:
                # Erweitere Oso-Evaluation mit Kontext
                return POLICY_DECISION_ALLOW if self._oso.is_allowed(subject, action, resource) else POLICY_DECISION_DENY
            except Exception:
                pass

        # Fallback-Logik mit Kontext-Berücksichtigung
        if subject.get("type") == "system" or subject.get("id") == "admin":
            return POLICY_DECISION_ALLOW
        if action == "read" and resource.get("public", False):
            return POLICY_DECISION_ALLOW

        # Prüfe Kontext-basierte Regeln
        if context:
            # Zeitbasierte Beschränkungen
            if context.get("time_restricted") and not self._is_within_allowed_time(context):
                return POLICY_DECISION_DENY

            # IP-basierte Beschränkungen
            if context.get("ip_restricted") and not self._is_allowed_ip(context.get("client_ip")):
                return POLICY_DECISION_DENY

            # Rollen-basierte Beschränkungen
            user_roles = context.get("user_roles", [])
            required_roles = resource.get("required_roles", [])
            if required_roles and not any(role in user_roles for role in required_roles):
                return POLICY_DECISION_DENY

        return POLICY_DECISION_ALLOW

    def _is_within_allowed_time(self, context: dict[str, Any]) -> bool:
        """Prüft zeitbasierte Zugriffsbeschränkungen."""
        # Vereinfachte Implementierung - in Produktion komplexere Logik
        allowed_hours = context.get("allowed_hours", [])
        if not allowed_hours:
            return True

        import datetime
        current_hour = datetime.datetime.now().hour
        return current_hour in allowed_hours

    def _is_allowed_ip(self, client_ip: str | None) -> bool:
        """Prüft IP-basierte Zugriffsbeschränkungen."""
        # Vereinfachte Implementierung - in Produktion komplexere Logik
        if not client_ip:
            return False

        # Beispiel: Erlaube nur lokale IPs
        allowed_prefixes = ["127.", "192.168.", "10.", "172."]
        return any(client_ip.startswith(prefix) for prefix in allowed_prefixes)

    def clear_cache(self) -> int:
        """Löscht alle Caches."""
        with self._lock:
            tool_cache_size = len(self._tool_access_cache)
            self._tool_access_cache.clear()

        policy_cache_size = len(self._evaluation_cache)
        self._evaluation_cache.clear()

        return tool_cache_size + policy_cache_size


# Globale Policy Engine Instanz (konsolidiert)
policy_engine = PolicyEngine()
