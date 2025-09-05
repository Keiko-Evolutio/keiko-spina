# backend/agents/registry/core/singleton_mixin.py
"""Einheitliches Singleton-Pattern für Registry-Komponenten.

Konsolidiert die verschiedenen Singleton-Implementierungen in eine saubere,
thread-sichere Lösung.
"""

import threading
from typing import Any, TypeVar

T = TypeVar("T", bound="SingletonMixin")


class SingletonMixin:
    """Thread-sichere Singleton-Implementierung als Mixin.
    
    Ersetzt die verschiedenen Singleton-Patterns in der Codebase durch
    eine einheitliche, saubere Implementierung.
    """

    _instances: dict[type, Any] = {}
    _lock = threading.Lock()

    def __new__(cls: type[T], *args, **kwargs) -> T:
        """Thread-sichere Singleton-Erstellung."""
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    instance = super(SingletonMixin, cls).__new__(cls)
                    cls._instances[cls] = instance
                    # Flag für Initialisierung setzen
                    instance._singleton_initialized = False

        return cls._instances[cls]

    def __init__(self, *args, **kwargs):
        """Initialisierung nur einmal pro Singleton-Instanz."""
        if hasattr(self, "_singleton_initialized") and self._singleton_initialized:
            return

        # Rufe die Initialisierung der Kindklasse auf
        super().__init__(*args, **kwargs)
        self._singleton_initialized = True

    @classmethod
    def get_instance(cls: type[T]) -> T:
        """Holt die Singleton-Instanz.
        
        Returns:
            Singleton-Instanz der Klasse
        """
        return cls()

    @classmethod
    def reset_instance(cls: type[T]) -> None:
        """Setzt die Singleton-Instanz zurück.
        
        Nützlich für Tests und Cleanup.
        """
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]

    @classmethod
    def has_instance(cls: type[T]) -> bool:
        """Prüft ob eine Singleton-Instanz existiert.
        
        Returns:
            True wenn Instanz existiert
        """
        return cls in cls._instances

    def is_initialized(self) -> bool:
        """Prüft ob die Singleton-Instanz initialisiert ist.
        
        Returns:
            True wenn initialisiert
        """
        return getattr(self, "_singleton_initialized", False)
