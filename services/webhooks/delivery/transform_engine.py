"""Webhook-Transform-Engine für Event-Transformation.

Transformiert Webhook-Events basierend auf Target-spezifischen Regeln
wie Feld-Filterung, Umbenennung und Anreicherung.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kei_logging import get_logger

if TYPE_CHECKING:
    from ..models import TargetTransform, WebhookEvent

logger = get_logger(__name__)


class WebhookTransformEngine:
    """Transformiert Events basierend auf Target-Konfiguration."""

    def transform_event(
        self,
        event: WebhookEvent,
        transform: TargetTransform | None = None,
    ) -> dict[str, Any]:
        """Wendet Transformationsregeln auf Event an.

        Args:
            event: Ursprüngliches Webhook-Event
            transform: Optionale Transformationsregeln

        Returns:
            Transformiertes Event als Dictionary
        """
        # Basis-Event-Struktur
        result = event.model_dump(mode="json")

        if not transform:
            return result

        # Transformiere nur die Daten, Meta-Informationen bleiben unverändert
        data = result.get("data", {})
        if not isinstance(data, dict):
            logger.warning("Event data is not a dictionary, skipping transformation")
            return result

        # Wende Transformationsregeln an
        transformed_data = self._apply_transformations(data, transform)
        result["data"] = transformed_data

        return result

    def _apply_transformations(
        self,
        data: dict[str, Any],
        transform: TargetTransform,
    ) -> dict[str, Any]:
        """Wendet alle Transformationsregeln auf Daten an.

        Args:
            data: Ursprüngliche Event-Daten
            transform: Transformationsregeln

        Returns:
            Transformierte Daten
        """
        # 1. Include-Filter anwenden (wenn definiert)
        if transform.include_fields:
            data = self._apply_include_filter(data, transform.include_fields)

        # 2. Exclude-Filter anwenden (wenn definiert)
        if transform.exclude_fields:
            data = self._apply_exclude_filter(data, transform.exclude_fields)

        # 3. Felder umbenennen (wenn definiert)
        if transform.rename_map:
            data = self._apply_field_renaming(data, transform.rename_map)

        # 4. Zusätzliche Felder hinzufügen (wenn definiert)
        if transform.add_fields:
            data = self._apply_field_addition(data, transform.add_fields)

        # 5. Null-Werte entfernen (wenn aktiviert)
        if transform.drop_nulls:
            data = self._apply_null_removal(data)

        return data

    def _apply_include_filter(
        self,
        data: dict[str, Any],
        include_fields: list[str],
    ) -> dict[str, Any]:
        """Behält nur die angegebenen Felder.

        Args:
            data: Ursprüngliche Daten
            include_fields: Liste der zu behaltenden Felder

        Returns:
            Gefilterte Daten
        """
        include_set: set[str] = set(include_fields)
        return {
            key: value
            for key, value in data.items()
            if key in include_set
        }

    def _apply_exclude_filter(
        self,
        data: dict[str, Any],
        exclude_fields: list[str],
    ) -> dict[str, Any]:
        """Entfernt die angegebenen Felder.

        Args:
            data: Ursprüngliche Daten
            exclude_fields: Liste der zu entfernenden Felder

        Returns:
            Gefilterte Daten
        """
        exclude_set: set[str] = set(exclude_fields)
        return {
            key: value
            for key, value in data.items()
            if key not in exclude_set
        }

    def _apply_field_renaming(
        self,
        data: dict[str, Any],
        rename_map: dict[str, str],
    ) -> dict[str, Any]:
        """Benennt Felder basierend auf Mapping um.

        Args:
            data: Ursprüngliche Daten
            rename_map: Mapping von alten zu neuen Feldnamen

        Returns:
            Daten mit umbenannten Feldern
        """
        renamed: dict[str, Any] = {}

        for key, value in data.items():
            new_key = rename_map.get(key, key)
            renamed[new_key] = value

        return renamed

    def _apply_field_addition(
        self,
        data: dict[str, Any],
        add_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Fügt zusätzliche Felder hinzu.

        Args:
            data: Ursprüngliche Daten
            add_fields: Zusätzliche Felder

        Returns:
            Daten mit zusätzlichen Feldern
        """
        result = data.copy()
        result.update(add_fields)
        return result

    def _apply_null_removal(self, data: dict[str, Any]) -> dict[str, Any]:
        """Entfernt Felder mit None-Werten.

        Args:
            data: Ursprüngliche Daten

        Returns:
            Daten ohne None-Werte
        """
        return {
            key: value
            for key, value in data.items()
            if value is not None
        }

    def validate_transform(self, transform: TargetTransform) -> list[str]:
        """Validiert Transformationsregeln und gibt Warnungen zurück.

        Args:
            transform: Zu validierende Transformationsregeln

        Returns:
            Liste von Validierungswarnungen
        """
        warnings: list[str] = []

        # Prüfe auf Konflikte zwischen include und exclude
        if transform.include_fields and transform.exclude_fields:
            include_set = set(transform.include_fields)
            exclude_set = set(transform.exclude_fields)
            conflicts = include_set.intersection(exclude_set)

            if conflicts:
                warnings.append(
                    f"Felder sowohl in include als auch exclude definiert: {conflicts}"
                )

        # Prüfe auf Konflikte zwischen rename und add
        if transform.rename_map and transform.add_fields:
            renamed_targets = set(transform.rename_map.values())
            added_fields = set(transform.add_fields.keys())
            conflicts = renamed_targets.intersection(added_fields)

            if conflicts:
                warnings.append(
                    f"Umbenannte Felder kollidieren mit hinzugefügten Feldern: {conflicts}"
                )

        # Prüfe auf leere Transformationsregeln
        if not any([
            transform.include_fields,
            transform.exclude_fields,
            transform.rename_map,
            transform.add_fields,
            transform.drop_nulls,
        ]):
            warnings.append("Keine Transformationsregeln definiert")

        return warnings

    def get_transform_summary(self, transform: TargetTransform) -> dict[str, Any]:
        """Gibt Zusammenfassung der Transformationsregeln zurück.

        Args:
            transform: Transformationsregeln

        Returns:
            Zusammenfassung als Dictionary
        """
        return {
            "include_fields_count": len(transform.include_fields or []),
            "exclude_fields_count": len(transform.exclude_fields or []),
            "rename_mappings_count": len(transform.rename_map or {}),
            "additional_fields_count": len(transform.add_fields or {}),
            "drop_nulls_enabled": bool(transform.drop_nulls),
            "has_transformations": any([
                transform.include_fields,
                transform.exclude_fields,
                transform.rename_map,
                transform.add_fields,
                transform.drop_nulls,
            ]),
        }


__all__ = ["WebhookTransformEngine"]
