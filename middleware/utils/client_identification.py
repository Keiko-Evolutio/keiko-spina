"""Client-Identifikations-Utilities für Rate-Limiting.

Dieses Modul konsolidiert alle Client-Identifikations-Logik,
die zuvor in verschiedenen Rate-Limiting-Modulen dupliziert war.
"""

import hashlib
from enum import Enum

from fastapi import Request

from ..constants import ConfigConstants, ErrorConstants, HeaderConstants


class IdentificationStrategy(str, Enum):
    """Strategien für Client-Identifikation."""
    IP_ADDRESS = "ip_address"
    USER_ID = "user_id"
    API_KEY = "api_key"
    TENANT_ID = "tenant_id"
    SESSION_ID = "session_id"
    STREAM_ID = "stream_id"
    COMBINED = "combined"


class ClientIdentificationUtils:
    """Utility-Klasse für einheitliche Client-Identifikation.

    Konsolidiert alle Client-Identifikations-Logik aus verschiedenen
    Rate-Limiting-Modulen in eine wiederverwendbare Komponente.
    """

    @staticmethod
    def extract_client_ip(request: Request) -> str:
        """Extrahiert Client-IP aus Request mit standardisierter Logik.

        Prüft verschiedene Headers in der korrekten Reihenfolge:
        1. X-Forwarded-For (Load Balancer)
        2. X-Real-IP (Nginx)
        3. request.client.host (Fallback)

        Args:
            request: FastAPI-Request-Objekt

        Returns:
            Client-IP-Adresse als String
        """
        # Prüfe X-Forwarded-For Header (Load Balancer)
        forwarded_for = request.headers.get(HeaderConstants.X_FORWARDED_FOR)
        if forwarded_for:
            # Nimm die erste IP (Client-IP) aus der Comma-separated Liste
            return forwarded_for.split(",")[0].strip()

        # Prüfe X-Real-IP Header (Nginx)
        real_ip = request.headers.get(HeaderConstants.X_REAL_IP)
        if real_ip:
            return real_ip.strip()

        # Fallback auf Client-IP
        return request.client.host if request.client else ErrorConstants.UNKNOWN_CLIENT

    @staticmethod
    def extract_api_key(request: Request) -> str | None:
        """Extrahiert API-Key aus Authorization-Header.

        Args:
            request: FastAPI-Request-Objekt

        Returns:
            API-Key als String oder None wenn nicht vorhanden
        """
        auth_header = request.headers.get(HeaderConstants.AUTHORIZATION, "")
        if auth_header.startswith(HeaderConstants.BEARER_PREFIX):
            return auth_header[len(HeaderConstants.BEARER_PREFIX):]
        return None

    @staticmethod
    def hash_api_key(api_key: str, length: int = 16) -> str:
        """Erstellt gehashte Version des API-Keys für Privacy.

        Args:
            api_key: Original API-Key
            length: Länge des gehashten Keys

        Returns:
            Gehashter API-Key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()[:length]

    @staticmethod
    def extract_user_id(request: Request) -> str | None:
        """Extrahiert User-ID aus Request-Headers.

        Args:
            request: FastAPI-Request-Objekt

        Returns:
            User-ID als String oder None wenn nicht vorhanden
        """
        return request.headers.get(HeaderConstants.X_USER_ID)

    @staticmethod
    def extract_tenant_id(request: Request) -> str | None:
        """Extrahiert Tenant-ID aus Request-Headers.

        Args:
            request: FastAPI-Request-Objekt

        Returns:
            Tenant-ID als String oder None wenn nicht vorhanden
        """
        return request.headers.get(HeaderConstants.X_TENANT_ID)

    @staticmethod
    def extract_session_id(request: Request) -> str | None:
        """Extrahiert Session-ID aus Request (Header oder URL-Path).

        Args:
            request: FastAPI-Request-Objekt

        Returns:
            Session-ID als String oder None wenn nicht vorhanden
        """
        # Zuerst aus Header versuchen
        session_id = request.headers.get(HeaderConstants.X_SESSION_ID)
        if session_id:
            return session_id

        # Dann aus URL-Path extrahieren (für WebSocket/SSE)
        path_parts = request.url.path.split("/")
        for i, part in enumerate(path_parts):
            if part in ["ws", "sse"] and i + 1 < len(path_parts):
                return path_parts[i + 1]

        return None

    @staticmethod
    def extract_stream_id(request: Request) -> str | None:
        """Extrahiert Stream-ID aus Request (Header oder URL-Path).

        Args:
            request: FastAPI-Request-Objekt

        Returns:
            Stream-ID als String oder None wenn nicht vorhanden
        """
        # Zuerst aus Header versuchen
        stream_id = request.headers.get(HeaderConstants.X_STREAM_ID)
        if stream_id:
            return stream_id

        # Dann aus URL-Path extrahieren (für SSE)
        path_parts = request.url.path.split("/")
        if "sse" in path_parts:
            try:
                sse_index = path_parts.index("sse")
                if sse_index + 2 < len(path_parts):
                    return path_parts[sse_index + 2]
            except ValueError:
                pass

        # Aus Query-Parameter (für WebSocket)
        if "ws" in path_parts:
            stream_id = request.query_params.get("stream_id")
            if stream_id:
                return stream_id

        return None

    @classmethod
    def generate_client_id(
        cls,
        request: Request,
        strategy: IdentificationStrategy
    ) -> str:
        """Generiert Client-ID basierend auf der gewählten Strategie.

        Args:
            request: FastAPI-Request-Objekt
            strategy: Identifikations-Strategie

        Returns:
            Client-ID als String
        """
        if strategy == IdentificationStrategy.IP_ADDRESS:
            client_ip = cls.extract_client_ip(request)
            return f"{ErrorConstants.IP_PREFIX}:{client_ip}"

        if strategy == IdentificationStrategy.USER_ID:
            user_id = cls.extract_user_id(request)
            if user_id:
                return f"{ErrorConstants.USER_PREFIX}:{user_id}"
            # Fallback auf IP-Adresse
            client_ip = cls.extract_client_ip(request)
            return f"{ErrorConstants.IP_PREFIX}:{client_ip}"

        if strategy == IdentificationStrategy.API_KEY:
            api_key = cls.extract_api_key(request)
            if api_key:
                hashed_key = cls.hash_api_key(api_key)
                return f"{ErrorConstants.API_KEY_PREFIX}:{hashed_key}"
            return ErrorConstants.NO_API_KEY

        if strategy == IdentificationStrategy.TENANT_ID:
            tenant_id = cls.extract_tenant_id(request)
            if tenant_id:
                return f"{ErrorConstants.TENANT_PREFIX}:{tenant_id}"
            return ErrorConstants.NO_TENANT

        if strategy == IdentificationStrategy.SESSION_ID:
            session_id = cls.extract_session_id(request)
            if session_id:
                return f"{ErrorConstants.SESSION_PREFIX}:{session_id}"
            return ErrorConstants.NO_SESSION

        if strategy == IdentificationStrategy.STREAM_ID:
            stream_id = cls.extract_stream_id(request)
            if stream_id:
                return f"{ErrorConstants.STREAM_PREFIX}:{stream_id}"
            return ErrorConstants.NO_STREAM

        if strategy == IdentificationStrategy.COMBINED:
            # Kombinierte Strategie: IP + User-ID
            client_ip = cls.extract_client_ip(request)
            user_id = cls.extract_user_id(request) or ErrorConstants.ANONYMOUS_CLIENT
            return f"{ErrorConstants.COMBINED_PREFIX}:{ErrorConstants.IP_PREFIX}:{client_ip}:{user_id}"

        # Fallback auf IP-Adresse
        client_ip = cls.extract_client_ip(request)
        return f"{ErrorConstants.IP_PREFIX}:{client_ip}"

    @classmethod
    def get_client_info_with_tier(
        cls,
        request: Request,
        api_key_tiers: dict,
        default_tier: str = "basic"
    ) -> tuple[str, str]:
        """Extrahiert Client-ID und Tier aus Request.

        Kompatibilitäts-Methode für bestehende Rate-Limiting-Module.

        Args:
            request: FastAPI-Request-Objekt
            api_key_tiers: Mapping von API-Key zu Tier
            default_tier: Standard-Tier wenn kein API-Key gefunden

        Returns:
            Tuple von (client_id, tier)
        """
        # Versuche API-Key aus Authorization-Header
        api_key = cls.extract_api_key(request)
        if api_key:
            # Prüfe API-Key-Tier-Mapping
            if api_key in api_key_tiers:
                tier = api_key_tiers[api_key]
                # Gekürzte API-Key für Privacy
                client_id = f"{ErrorConstants.API_KEY_PREFIX}:{api_key[:ConfigConstants.API_KEY_DISPLAY_LENGTH]}"
                return client_id, tier
            # Unbekannter API-Key -> Default-Tier
            client_id = f"{ErrorConstants.API_KEY_PREFIX}:{api_key[:ConfigConstants.API_KEY_DISPLAY_LENGTH]}"
            return client_id, default_tier

        # Fallback auf IP-basierte Identifikation
        client_ip = cls.extract_client_ip(request)
        client_id = f"{ErrorConstants.IP_PREFIX}:{client_ip}"
        return client_id, "basic"  # IP-basiert immer Basic-Tier


__all__ = [
    "ClientIdentificationUtils",
    "IdentificationStrategy",
]
