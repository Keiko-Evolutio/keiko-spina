"""CLI‑Tool zur Generierung von AsyncAPI/OpenAPI Spezifikationen.

Bietet Ausgabe in JSON/YAML, Validierung und Markdown‑Export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from kei_logging import get_logger

from .asyncapi_generator import build_asyncapi_dict
from .constants import DirectoryNames, FileNames
from .openapi_generator import generate_openapi_dict

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)


def _write_spec(doc: dict, out_json: Path, out_yaml: Path) -> None:
    """Schreibt Spezifikation in JSON/YAML Dateien."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def generate_all_specs(app: FastAPI | None = None, *, out_dir: Path = Path(f"backend/{DirectoryNames.SPECS_DIR}/out")) -> dict:
    """Erzeugt AsyncAPI und OpenAPI und schreibt Ausgabedateien.

    Args:
        app: Optional laufende FastAPI App (für OpenAPI)
        out_dir: Ausgabeverzeichnis

    Returns:
        Zusammenfassung mit Dateipfaden
    """
    summary: dict = {"asyncapi": {}, "openapi": {}}

    # AsyncAPI
    asyncapi = build_asyncapi_dict()
    _write_spec(asyncapi, out_dir / FileNames.ASYNCAPI_JSON, out_dir / FileNames.ASYNCAPI_YAML)
    summary["asyncapi"] = {"json": str(out_dir / FileNames.ASYNCAPI_JSON), "yaml": str(out_dir / FileNames.ASYNCAPI_YAML)}

    # OpenAPI
    if app is not None:
        openapi = generate_openapi_dict(app)
        _write_spec(openapi, out_dir / FileNames.OPENAPI_JSON, out_dir / FileNames.OPENAPI_YAML)
        summary["openapi"] = {"json": str(out_dir / FileNames.OPENAPI_JSON), "yaml": str(out_dir / FileNames.OPENAPI_YAML)}
    else:
        logger.warning("OpenAPI konnte nicht erzeugt werden: App nicht übergeben")

    return summary


__all__ = ["generate_all_specs"]

