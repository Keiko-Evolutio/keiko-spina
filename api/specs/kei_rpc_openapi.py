# backend/api/specs/kei_rpc_openapi.py
"""OpenAPI-Spezifikation für KEI-RPC Interface.

Refactored Modul mit modularer Struktur für bessere Wartbarkeit.
Verwendet gemeinsame Komponenten zur Elimination von Code-Duplikaten.
"""

from __future__ import annotations

from typing import Any

from .common_parameters import get_parameter_components
from .common_responses import get_response_components
from .common_schemas import get_common_schemas, get_security_schemas
from .kei_rpc_paths import get_kei_rpc_paths


def get_kei_rpc_openapi_extensions() -> dict[str, Any]:
    """Gibt KEI-RPC OpenAPI-Erweiterungen zurück.

    Konsolidiert alle KEI-RPC spezifischen OpenAPI-Definitionen
    unter Verwendung modularer Komponenten.

    Returns:
        Dictionary mit OpenAPI-Erweiterungen für KEI-RPC
    """
    return {
        "paths": get_kei_rpc_paths(),
        "components": {
            "parameters": get_parameter_components(),
            "responses": get_response_components(),
            "schemas": get_common_schemas(),
            "securitySchemes": get_security_schemas()
        }
    }




