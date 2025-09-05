"""Laufzeit-Generierung der gRPC Python-Module für KEI-Stream.

Kompiliert `grpc_stream.proto` nach `grpc_stream_pb2.py` und
`grpc_stream_pb2_grpc.py`.
"""

from __future__ import annotations

from pathlib import Path


def generate_stream_protos(proto_dir: Path | None = None) -> bool:
    """Generiert Python-Module aus `grpc_stream.proto`.

    Args:
        proto_dir: Verzeichnis mit .proto Datei

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
    proto_file = src_dir / "grpc_stream.proto"

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


__all__ = ["generate_stream_protos"]
