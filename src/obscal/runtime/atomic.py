"""Atomic artefact writes: a reader never observes a half-written file."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "obscal/1"


def _fsync_replace(tmp_path: Path, dest: Path) -> None:
    with open(tmp_path, "rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, dest)


def write_text(dest: Path, text: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=dest.parent, prefix=".tmp_")
    tmp_path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_replace(tmp_path, dest)
    return dest


def write_json(dest: Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    return write_text(
        dest,
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True, default=_encode) + "\n",
    )


def read_json(source: Path) -> Any:
    with open(source, encoding="utf-8") as handle:
        return json.load(handle)


def _encode(value: object) -> object:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(payload: Sequence[float]) -> str:
    """Digest a numeric payload rather than the container bytes that hold it."""
    digest = hashlib.sha256()
    for value in payload:
        digest.update(repr(round(float(value), 12)).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()
