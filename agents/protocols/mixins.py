"""Protocol Mixins - Wiederverwendbare Mixin-Klassen."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


class SerializableMixin:
    """Mixin für Serialisierung von Dataclasses.

    Stellt einheitliche to_dict() Methode für alle Dataclasses bereit.
    """

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert Dataclass zu Dictionary.

        Returns:
            Dictionary-Repräsentation der Dataclass

        Example:
            >>> @dataclass
            ... class MyClass(SerializableMixin):
            ...     name: str
            ...     value: int
            >>> obj = MyClass("test", 42)
            >>> assert obj.to_dict() == {"name": "test", "value": 42}
        """
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} ist keine Dataclass. "
                          "SerializableMixin kann nur mit @dataclass dekorierte Klassen verwendet werden.")
        return asdict(self)

    def to_json_dict(self) -> dict[str, Any]:
        """Konvertiert zu JSON-serialisierbarem Dictionary.

        Behandelt spezielle Typen wie datetime, Enum, etc.

        Returns:
            JSON-serialisierbares Dictionary
        """
        result = self.to_dict()
        return self._make_json_serializable(result)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Macht Objekt JSON-serialisierbar.

        Args:
            obj: Zu konvertierendes Objekt

        Returns:
            JSON-serialisierbares Objekt
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if hasattr(obj, "isoformat"):  # datetime
            return obj.isoformat()
        return obj


class ValidatableMixin:
    """Mixin für Validierung von Dataclasses.

    Stellt Basis-Validierungsfunktionalitäten bereit.
    """

    def validate(self) -> bool:
        """Validiert die Dataclass-Instanz.

        Returns:
            True wenn valide

        Raises:
            ValueError: Bei Validierungsfehlern
        """
        self._validate_required_fields()
        self._validate_field_types()
        return True

    def _validate_required_fields(self) -> None:
        """Prüft erforderliche Felder."""
        required_fields = getattr(self, "_required_fields", [])
        for field_name in required_fields:
            if not hasattr(self, field_name):
                raise ValueError(f"Erforderliches Feld fehlt: {field_name}")

            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Feld '{field_name}' darf nicht leer sein")

    def _validate_field_types(self) -> None:
        """Prüft Feldtypen (Basis-Implementierung)."""
        # Kann in Subklassen überschrieben werden


class AzureAIEntityMixin(SerializableMixin, ValidatableMixin):
    """Mixin für Azure AI Entities.

    Kombiniert Serialisierung und Validierung für Azure AI Foundry Entities.
    """

    @property
    def metadata(self) -> dict[str, Any]:
        """Ermittelt Entity-Metadaten mit Lazy Loading.

        Returns:
            Metadaten-Dictionary
        """
        if not hasattr(self, "_metadata"):
            self._metadata: dict[str, Any] = {}
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        """Setzt Entity-Metadaten.

        Args:
            value: Metadaten-Dictionary
        """
        self._metadata = value or {}

    def get_entity_type(self) -> str:
        """Ermittelt Entity-Typ.

        Returns:
            Entity-Typ-String
        """
        return getattr(self, "_type", "unknown")

    def get_metadata(self) -> dict[str, Any]:
        """Ermittelt Entity-Metadaten.

        Returns:
            Metadaten-Dictionary
        """
        return self.metadata

    def set_metadata(self, key: str, value: Any) -> None:
        """Setzt Metadaten-Wert.

        Args:
            key: Metadaten-Schlüssel
            value: Metadaten-Wert
        """
        self.metadata[key] = value

    def get_id(self) -> str:
        """Ermittelt Entity-ID.

        Returns:
            Entity-ID

        Raises:
            ValueError: Wenn keine ID vorhanden
        """
        if not hasattr(self, "id") or not self.id:  # type: ignore
            raise ValueError("Entity hat keine gültige ID")
        return self.id  # type: ignore


__all__ = [
    "AzureAIEntityMixin",
    "SerializableMixin",
    "ValidatableMixin",
]
