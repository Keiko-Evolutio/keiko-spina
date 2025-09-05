"""Laufzeit-Generierung der gRPC Python-Module aus .proto Dateien.

Verwendet grpc_tools.protoc um `kei_rpc.proto` in `kei_rpc_pb2.py` und
`kei_rpc_pb2_grpc.py` zu kompilieren. Diese Funktion kann vom Server
aufgerufen werden, falls die generierten Module fehlen.
"""

from __future__ import annotations

from pathlib import Path


def generate_protos(proto_dir: Path | None = None) -> bool:
    """Generiert gRPC Python-Code aus Protobuf-Definitionen.

    Args:
        proto_dir: Verzeichnis mit den .proto Dateien

    Returns:
        True bei Erfolg, sonst False
    """
    try:
        from grpc_tools import protoc  # type: ignore
    except Exception:
        return False

    root = Path(__file__).resolve().parent
    src_dir = proto_dir or root
    out_dir = root
    proto_file = src_dir / "kei_rpc.proto"

    if not proto_file.exists():
        return False

    cmd = [
        "protoc",
        f"-I{src_dir!s}",
        f"--python_out={out_dir!s}",
        f"--grpc_python_out={out_dir!s}",
        str(proto_file),
    ]
    try:
        result = protoc.main(cmd)
        return result == 0
    except Exception:
        return False


__all__ = ["generate_protos"]
