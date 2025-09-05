# backend/agents/core/component_manager.py
"""Component-Manager für Keiko Personal Assistant

Implementiert Dependency Injection und Component-Lifecycle-Management:
- Service-Container-Pattern
- Async Component-Initialization
- Health-Check-Aggregation
- Resource-Management
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from .utils import (
    ValidationError,
    get_module_logger,
    validate_required_field,
)

logger = get_module_logger(__name__)

T = TypeVar("T")
ComponentFactory = Callable[[], Awaitable[T]]


@dataclass
class ComponentInfo:
    """Informationen über eine registrierte Komponente."""

    name: str
    component_type: type
    factory: ComponentFactory
    singleton: bool = True
    dependencies: list[str] = None
    initialized: bool = False
    instance: Any | None = None
    initialization_time: float = 0.0

    def __post_init__(self) -> None:
        """Post-Initialisierung."""
        if self.dependencies is None:
            self.dependencies = []


class ComponentLifecycle(ABC):
    """Interface für Component-Lifecycle-Management."""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialisiert Komponente."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Führt Health-Check durch."""

    @abstractmethod
    async def close(self) -> None:
        """Schließt Komponente."""


class ComponentManager:
    """Component-Manager für Dependency Injection und Lifecycle-Management.

    Implementiert Service-Container-Pattern mit:
    - Dependency Resolution
    - Singleton-Management
    - Async Initialization
    - Health-Check-Aggregation
    """

    def __init__(self):
        """Initialisiert Component-Manager."""
        self._components: dict[str, ComponentInfo] = {}
        self._instances: dict[str, Any] = {}
        self._initialization_order: list[str] = []
        self._is_initialized = False

    def register_component(
        self,
        name: str,
        component_type: type[T],
        factory: ComponentFactory[T],
        singleton: bool = True,
        dependencies: list[str] | None = None,
    ) -> None:
        """Registriert Komponente im Container.

        Args:
            name: Komponenten-Name
            component_type: Komponenten-Typ
            factory: Factory-Funktion
            singleton: Ob Singleton-Pattern verwendet werden soll
            dependencies: Liste von Abhängigkeiten
        """
        try:
            validate_required_field(name, "name")
            validate_required_field(component_type, "component_type")
            validate_required_field(factory, "factory")
        except ValidationError as e:
            raise ValueError(str(e)) from e

        if name in self._components:
            raise ValueError(f"Komponente '{name}' bereits registriert")

        self._components[name] = ComponentInfo(
            name=name,
            component_type=component_type,
            factory=factory,
            singleton=singleton,
            dependencies=dependencies or [],
        )

        logger.debug(f"Komponente '{name}' registriert")

    def register_singleton(
        self,
        name: str,
        component_type: type[T],
        factory: ComponentFactory[T],
        dependencies: list[str] | None = None,
    ) -> None:
        """Registriert Singleton-Komponente."""
        self.register_component(name, component_type, factory, True, dependencies)

    def register_transient(
        self,
        name: str,
        component_type: type[T],
        factory: ComponentFactory[T],
        dependencies: list[str] | None = None,
    ) -> None:
        """Registriert Transient-Komponente (neue Instanz bei jedem Aufruf)."""
        self.register_component(name, component_type, factory, False, dependencies)

    async def get_component(self, name: str) -> Any:
        """Holt Komponente aus Container.

        Args:
            name: Komponenten-Name

        Returns:
            Komponenten-Instanz
        """
        if name not in self._components:
            raise ValueError(f"Komponente '{name}' nicht registriert")

        component_info = self._components[name]

        if component_info.singleton:
            if name in self._instances:
                return self._instances[name]

            instance = await self._create_component_instance(name)
            self._instances[name] = instance
            return instance

        return await self._create_component_instance(name)

    async def _create_component_instance(self, name: str) -> Any:
        """Erstellt Komponenten-Instanz mit Dependency-Resolution."""
        component_info = self._components[name]

        dependencies = {}
        for dep_name in component_info.dependencies:
            dependencies[dep_name] = await self.get_component(dep_name)

        start_time = time.time()

        try:
            # Factory-Funktionen erwarten keine Parameter basierend auf ComponentFactory-Definition
            instance = await component_info.factory()

            initialization_time = time.time() - start_time
            component_info.initialization_time = initialization_time
            component_info.initialized = True

            logger.debug(f"Komponente '{name}' erstellt ({initialization_time:.3f}s)")

            return instance

        except Exception as e:
            logger.error(f"Komponenten-Erstellung '{name}' fehlgeschlagen: {e}")
            raise

    async def initialize_all(self) -> bool:
        """Initialisiert alle registrierten Komponenten.

        Returns:
            True wenn alle Komponenten erfolgreich initialisiert
        """
        if self._is_initialized:
            logger.warning("Component-Manager bereits initialisiert")
            return True

        try:
            self._initialization_order = self._resolve_initialization_order()

            for component_name in self._initialization_order:
                component_info = self._components[component_name]

                if component_info.singleton:
                    instance = await self.get_component(component_name)

                    if isinstance(instance, ComponentLifecycle):
                        success = await instance.initialize()
                        if not success:
                            raise RuntimeError(
                                f"Komponenten-Initialisierung '{component_name}' fehlgeschlagen"
                            )

            self._is_initialized = True
            logger.info(
                f"Alle {len(self._initialization_order)} Komponenten erfolgreich initialisiert"
            )
            return True

        except Exception as e:
            logger.error(f"Component-Manager-Initialisierung fehlgeschlagen: {e}")
            return False

    def _resolve_initialization_order(self) -> list[str]:
        """Bestimmt Initialisierungs-Reihenfolge basierend auf Dependencies."""
        visited = set()
        temp_visited = set()
        order = []

        def visit(comp_name: str):
            if comp_name in temp_visited:
                raise ValueError(f"Zirkuläre Dependency erkannt: {comp_name}")

            if comp_name in visited:
                return

            temp_visited.add(comp_name)

            component_info = self._components[comp_name]
            for dependency in component_info.dependencies:
                if dependency not in self._components:
                    raise ValueError(
                        f"Unbekannte Dependency '{dependency}' für Komponente '{comp_name}'"
                    )
                visit(dependency)

            temp_visited.remove(comp_name)
            visited.add(comp_name)
            order.append(comp_name)

        for component_name in self._components:
            visit(component_name)

        return order

    async def health_check_all(self) -> dict[str, Any]:
        """Führt Health-Check für alle Komponenten durch.

        Returns:
            Aggregierter Health-Status
        """
        health_status = {
            "healthy": True,
            "component_manager_initialized": self._is_initialized,
            "total_components": len(self._components),
            "initialized_components": len(self._instances),
            "components": {},
        }

        for name, instance in self._instances.items():
            try:
                if isinstance(instance, ComponentLifecycle):
                    component_health = await instance.health_check()
                else:
                    component_health = {
                        "healthy": True,
                        "type": type(instance).__name__,
                        "initialized": True,
                    }

                health_status["components"][name] = component_health

                if not component_health.get("healthy", False):
                    health_status["healthy"] = False

            except Exception as e:
                logger.error(f"Health-Check für Komponente '{name}' fehlgeschlagen: {e}")
                health_status["components"][name] = {"healthy": False, "error": str(e)}
                health_status["healthy"] = False

        return health_status

    async def get_component_metrics(self) -> dict[str, Any]:
        """Holt Metriken für alle Komponenten.

        Returns:
            Komponenten-Metriken
        """
        metrics = {
            "total_registered": len(self._components),
            "total_initialized": len(self._instances),
            "initialization_order": self._initialization_order,
            "components": {},
        }

        for name, component_info in self._components.items():
            component_metrics = {
                "type": component_info.component_type.__name__,
                "singleton": component_info.singleton,
                "dependencies": component_info.dependencies,
                "initialized": component_info.initialized,
                "initialization_time": component_info.initialization_time,
            }

            if name in self._instances:
                instance = self._instances[name]
                if hasattr(instance, "get_metrics"):
                    try:
                        instance_metrics = await instance.get_metrics()
                        component_metrics["instance_metrics"] = instance_metrics
                    except Exception as e:
                        logger.warning(f"Metriken für Komponente '{name}' nicht verfügbar: {e}")

            metrics["components"][name] = component_metrics

        return metrics

    @asynccontextmanager
    async def component_scope(self, component_names: list[str]):
        """Context-Manager für temporäre Komponenten-Nutzung."""
        instances = {}

        try:
            for name in component_names:
                instances[name] = await self.get_component(name)

            yield instances

        finally:
            for name in component_names:
                component_info = self._components[name]
                if not component_info.singleton and name in instances:
                    instance = instances[name]
                    if isinstance(instance, ComponentLifecycle):
                        try:
                            await instance.close()
                        except Exception as e:
                            logger.warning(f"Cleanup für Komponente '{name}' fehlgeschlagen: {e}")

    async def close_all(self) -> None:
        """Schließt alle Komponenten in umgekehrter Reihenfolge."""
        logger.info("Schließe alle Komponenten")

        # Schließe in umgekehrter Initialisierungs-Reihenfolge
        close_order = list(reversed(self._initialization_order))

        for component_name in close_order:
            if component_name in self._instances:
                instance = self._instances[component_name]

                if isinstance(instance, ComponentLifecycle):
                    try:
                        await instance.close()
                        logger.debug(f"Komponente '{component_name}' geschlossen")
                    except Exception as e:
                        logger.error(
                            f"Fehler beim Schließen von Komponente '{component_name}': {e}"
                        )

        # Reset State
        self._instances.clear()
        self._initialization_order.clear()
        self._is_initialized = False

        logger.info("Alle Komponenten geschlossen")

    def is_registered(self, name: str) -> bool:
        """Prüft ob Komponente registriert ist."""
        return name in self._components

    def is_initialized(self, name: str) -> bool:
        """Prüft ob Komponente initialisiert ist."""
        return name in self._instances

    def get_registered_components(self) -> list[str]:
        """Holt Liste aller registrierten Komponenten."""
        return list(self._components.keys())

    def get_initialized_components(self) -> list[str]:
        """Holt Liste aller initialisierten Komponenten."""
        return list(self._instances.keys())


class ComponentRegistry:
    """Registry für Component-Typen und Factories.

    Vereinfachte Registry für Backward-Compatibility.
    """

    def __init__(self):
        """Initialisiert Component Registry."""
        self._component_types: dict[str, type] = {}
        self._factories: dict[str, ComponentFactory] = {}

    def register_type(self, name: str, component_type: type) -> None:
        """Registriert Component-Typ.

        Args:
            name: Name der Komponente
            component_type: Typ der Komponente
        """
        self._component_types[name] = component_type

    def register_factory(self, name: str, factory: ComponentFactory) -> None:
        """Registriert Component-Factory.

        Args:
            name: Name der Komponente
            factory: Factory-Funktion
        """
        self._factories[name] = factory

    def get_type(self, name: str) -> type | None:
        """Holt Component-Typ.

        Args:
            name: Name der Komponente

        Returns:
            Component-Typ oder None
        """
        return self._component_types.get(name)

    def get_factory(self, name: str) -> ComponentFactory | None:
        """Holt Component-Factory.

        Args:
            name: Name der Komponente

        Returns:
            Factory-Funktion oder None
        """
        return self._factories.get(name)

    def list_components(self) -> list[str]:
        """Listet alle registrierten Komponenten auf.

        Returns:
            Liste der Komponenten-Namen
        """
        return list(set(self._component_types.keys()) | set(self._factories.keys()))
