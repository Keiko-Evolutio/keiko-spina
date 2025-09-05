# backend/agents/factory/singleton_mixin.py
"""Thread-sichere Singleton-Implementierung für Factory-Komponenten.

Konsolidiert alle Singleton-Patterns in der Keiko-Personal-Assistant Codebase
zu einer wiederverwendbaren, enterprise-grade Implementierung.
"""
from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any, TypeVar

from kei_logging import get_logger

from .constants import FactoryState, LogLevel

# Type Variable für Generic Singleton
T = TypeVar("T", bound="SingletonMixin")

logger = get_logger(__name__)


class SingletonMeta(type):
    """Metaclass für thread-sichere Singleton-Implementierung.

    Implementiert das Singleton-Pattern auf Metaclass-Ebene für maximale
    Performance und Thread-Sicherheit. Verwendet WeakReferences für
    besseres Memory-Management.
    """

    _instances: dict[type, Any] = {}
    _locks: dict[type, threading.Lock] = {}
    _creation_times: dict[type, datetime] = {}

    def __call__(cls, *args, **kwargs):
        """Thread-sichere Singleton-Instanz-Erstellung."""
        # Lazy Lock-Erstellung für bessere Performance
        if cls not in cls._locks:
            with threading.Lock():
                if cls not in cls._locks:
                    cls._locks[cls] = threading.Lock()

        # Double-checked locking für optimale Performance
        if cls not in cls._instances:
            with cls._locks[cls]:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
                    cls._creation_times[cls] = datetime.now(UTC)

                    logger.debug(
                        f"Singleton-Instanz erstellt: {cls.__name__}",
                        extra={
                            "class_name": cls.__name__,
                            "creation_time": cls._creation_times[cls].isoformat(),
                            "log_level": LogLevel.DEBUG
                        }
                    )

        return cls._instances[cls]

    @classmethod
    def reset_instance(mcs, cls: type) -> None:
        """Setzt eine spezifische Singleton-Instanz zurück.

        Hauptsächlich für Tests und Debugging verwendet.
        """
        if cls in mcs._locks:
            with mcs._locks[cls]:
                if cls in mcs._instances:
                    instance = mcs._instances.pop(cls)
                    mcs._creation_times.pop(cls, None)

                    # Cleanup-Hook aufrufen falls vorhanden
                    if hasattr(instance, "singleton_cleanup"):
                        try:
                            instance.singleton_cleanup()
                        except Exception as e:
                            logger.warning(
                                f"Fehler beim Singleton-Cleanup: {e}",
                                extra={
                                    "class_name": cls.__name__,
                                    "error": str(e),
                                    "log_level": LogLevel.WARNING
                                }
                            )
                    elif hasattr(instance, "_singleton_cleanup"):
                        try:
                            instance._singleton_cleanup()
                        except Exception as e:
                            logger.warning(
                                f"Fehler beim Singleton-Cleanup: {e}",
                                extra={
                                    "class_name": cls.__name__,
                                    "error": str(e),
                                    "log_level": LogLevel.WARNING
                                }
                            )

                    logger.debug(
                        f"Singleton-Instanz zurückgesetzt: {cls.__name__}",
                        extra={
                            "class_name": cls.__name__,
                            "log_level": LogLevel.DEBUG
                        }
                    )

    @classmethod
    def has_instance(mcs, cls: type) -> bool:
        """Public API für Instanz-Existenz-Prüfung.

        Args:
            cls: Klasse für die geprüft werden soll

        Returns:
            True wenn Instanz existiert
        """
        return cls in mcs._instances

    @classmethod
    def get_instance_by_class(mcs, cls: type) -> Any | None:
        """Public API für Instanz-Zugriff.

        Args:
            cls: Klasse deren Instanz geholt werden soll

        Returns:
            Instanz oder None
        """
        return mcs._instances.get(cls)

    @classmethod
    def reset_all_instances(mcs) -> None:
        """Setzt alle Singleton-Instanzen zurück.

        Vorsicht: Nur für Tests oder komplette System-Resets verwenden.
        """
        classes_to_reset = list(mcs._instances.keys())
        for cls in classes_to_reset:
            mcs.reset_instance(cls)

        logger.info(
            f"Alle {len(classes_to_reset)} Singleton-Instanzen zurückgesetzt",
            extra={
                "reset_count": len(classes_to_reset),
                "log_level": LogLevel.INFO
            }
        )

    @classmethod
    def get_singleton_stats(cls) -> dict[str, Any]:
        """Gibt Statistiken über alle Singleton-Instanzen zurück."""
        return {
            "total_singletons": len(cls._instances),
            "active_classes": [cls.__name__ for cls in cls._instances],
            "creation_times": {
                cls.__name__: time.isoformat()
                for cls, time in cls._creation_times.items()
            },
            "memory_usage": {
                cls.__name__: id(instance)
                for cls, instance in cls._instances.items()
            }
        }


class SingletonMixin(metaclass=SingletonMeta):
    """Abstrakte Basisklasse für Singleton-Implementierungen.

    Bietet eine konsistente Schnittstelle für alle Singleton-Klassen
    in der Factory-Architektur mit eingebauter Initialisierung und
    Cleanup-Funktionalität.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialisiert die Singleton-Instanz.

        Verwendet ein Flag um mehrfache Initialisierung zu verhindern.
        """
        if hasattr(self, "_singleton_initialized"):
            return

        self._singleton_initialized = False
        self._initialization_lock = threading.Lock()
        self._state = FactoryState.UNINITIALIZED
        self._creation_time = datetime.now(UTC)

        # Subclass-spezifische Initialisierung
        self._initialize_singleton(*args, **kwargs)

        self._singleton_initialized = True
        self._state = FactoryState.READY

        logger.debug(
            f"Singleton initialisiert: {self.__class__.__name__}",
            extra={
                "class_name": self.__class__.__name__,
                "state": self._state.value,
                "log_level": LogLevel.DEBUG
            }
        )

    def _initialize_singleton(self, *args, **kwargs) -> None:
        """Subclass-spezifische Initialisierung.

        Kann von jeder Singleton-Klasse überschrieben werden.
        """

    def singleton_cleanup(self) -> None:
        """Public API für Singleton-Cleanup.

        Kann von Subclasses überschrieben werden für spezifische Cleanup-Logik.
        """
        self._singleton_cleanup()

    def _singleton_cleanup(self) -> None:
        """Cleanup-Hook für Singleton-Reset.

        Kann von Subclasses überschrieben werden für spezifische Cleanup-Logik.
        """
        self._state = FactoryState.SHUTTING_DOWN
        logger.debug(
            f"Singleton-Cleanup: {self.__class__.__name__}",
            extra={
                "class_name": self.__class__.__name__,
                "state": self._state.value,
                "log_level": LogLevel.DEBUG
            }
        )

    @property
    def is_initialized(self) -> bool:
        """Prüft ob die Singleton-Instanz vollständig initialisiert ist."""
        return getattr(self, "_singleton_initialized", False)

    @property
    def state(self) -> FactoryState:
        """Gibt den aktuellen Zustand der Singleton-Instanz zurück."""
        return getattr(self, "_state", FactoryState.UNINITIALIZED)

    @property
    def creation_time(self) -> datetime:
        """Gibt die Erstellungszeit der Singleton-Instanz zurück."""
        return getattr(self, "_creation_time", datetime.now(UTC))

    @classmethod
    def reset_singleton(cls) -> None:
        """Setzt diese Singleton-Instanz zurück."""
        SingletonMeta.reset_instance(cls)

    @classmethod
    def get_instance_info(cls) -> dict[str, Any]:
        """Gibt Informationen über diese Singleton-Instanz zurück."""
        if SingletonMeta.has_instance(cls):
            instance = SingletonMeta.get_instance_by_class(cls)
            return {
                "class_name": cls.__name__,
                "is_initialized": instance.is_initialized,
                "state": instance.state.value,
                "creation_time": instance.creation_time.isoformat(),
                "memory_id": id(instance)
            }
        return {
            "class_name": cls.__name__,
            "is_initialized": False,
            "state": FactoryState.UNINITIALIZED.value,
            "creation_time": None,
            "memory_id": None
        }


class ThreadSafeSingleton(SingletonMixin):
    """Konkrete Implementierung für einfache Thread-sichere Singletons.

    Kann direkt verwendet werden für Klassen die keine spezielle
    Initialisierung benötigen.
    """

    def _initialize_singleton(self, *args, **kwargs) -> None:
        """Standard-Initialisierung ohne spezielle Logik."""


# =============================================================================
# Utility-Funktionen für Singleton-Management
# =============================================================================

def reset_all_singletons() -> None:
    """Setzt alle Singleton-Instanzen zurück.

    Hauptsächlich für Tests und System-Resets verwendet.
    """
    SingletonMeta.reset_all_instances()


def get_all_singleton_stats() -> dict[str, Any]:
    """Gibt Statistiken über alle Singleton-Instanzen zurück."""
    return SingletonMeta.get_singleton_stats()


def is_singleton_active(cls: type[SingletonMixin]) -> bool:
    """Prüft ob eine Singleton-Klasse eine aktive Instanz hat."""
    return SingletonMeta.has_instance(cls)


# =============================================================================
# Export für einfachen Import
# =============================================================================

__all__ = [
    "SingletonMeta",
    "SingletonMixin",
    "ThreadSafeSingleton",
    "get_all_singleton_stats",
    "is_singleton_active",
    "reset_all_singletons",
]
