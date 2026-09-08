"""Small helpers for protocol JSON files and checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def write_json_model(model: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write a pydantic model to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model.model_dump_json(indent=indent), encoding="utf-8")
    return output_path


def read_json_model(model_type: Any, path: str | Path) -> Any:
    """Read a pydantic model from JSON."""

    return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["read_json_model", "sha256_file", "write_json_model"]
