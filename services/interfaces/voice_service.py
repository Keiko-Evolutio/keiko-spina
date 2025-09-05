"""VoiceService Interface-Definition."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._base import FeatureService

if TYPE_CHECKING:
    from ._types import ServiceConfig, ServiceResult


class VoiceService(FeatureService):
    """Definiert den Vertrag für Voice-bezogene Funktionen.

    Feature-Service für Audio-Verarbeitung, Transkription und Sprachsynthese.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        config: ServiceConfig | None = None
    ) -> ServiceResult:
        """Transkribiert Audio-Daten und liefert Metadaten zurück.

        Args:
            audio_bytes: Audio-Daten als Bytes.
            config: Optionale Konfiguration für die Transkription.

        Returns:
            Transkriptions-Ergebnis mit Text und Metadaten.

        Raises:
            ValueError: Bei ungültigen Audio-Daten.
            RuntimeError: Bei Transkriptions-Fehlern.
        """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        config: ServiceConfig | None = None
    ) -> bytes:
        """Synthetisiert Sprache aus Text.

        Args:
            text: Zu synthetisierender Text.
            config: Optionale Konfiguration für die Synthese.

        Returns:
            Audio-Daten als Bytes.

        Raises:
            ValueError: Bei ungültigem Text.
            RuntimeError: Bei Synthese-Fehlern.
        """

    @abstractmethod
    async def detect_language(self, audio_bytes: bytes) -> ServiceResult:
        """Erkennt die Sprache in Audio-Daten.

        Args:
            audio_bytes: Audio-Daten als Bytes.

        Returns:
            Erkannte Sprache mit Konfidenz-Score.

        Raises:
            ValueError: Bei ungültigen Audio-Daten.
            RuntimeError: Bei Erkennungs-Fehlern.
        """
