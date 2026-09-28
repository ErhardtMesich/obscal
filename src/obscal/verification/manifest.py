"""The integrity manifest: one digest per shipped file, plus a digest of the manifest.

The manifest is always the last artefact written, and it excludes itself, so the
report and the manifest cannot both be final if the report embeds manifest
digests. The report records only the fact that the manifest exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from obscal.runtime.atomic import file_digest, read_json, write_json
from obscal.runtime.config import RELEASE_SLUG

MANIFEST_NAME = "integrity_manifest.json"
SKIP_DIRECTORY_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "build",
    "dist",
    ".venv",
    "node_modules",
}
SKIP_TOP_LEVEL = {"runs", "artefacts", "cohort", "checkpoints"}


def inventory(root: Path, *, exclude: Path | None = None) -> list[Path]:
    """Every shipped file, in a stable walk order."""
    excluded = exclude.resolve() if exclude is not None else None
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRECTORY_NAMES for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in SKIP_TOP_LEVEL:
            continue
        if excluded is not None and path.resolve() == excluded:
            continue
        files.append(path)
    return files


def build_manifest(root: Path, *, target: Path | None = None) -> dict[str, Any]:
    destination = (target or (root / MANIFEST_NAME)).resolve()
    entries: list[dict[str, str]] = []
    for path in inventory(root, exclude=destination):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_digest(path),
            }
        )
    digest = manifest_digest(entries)
    return {
        "algorithm": "sha256",
        "root": RELEASE_SLUG,
        "excluded": [MANIFEST_NAME],
        "skipped_directories": sorted(SKIP_DIRECTORY_NAMES),
        "skipped_top_level": sorted(SKIP_TOP_LEVEL),
        "n_files": len(entries),
        "files": entries,
        "manifest_digest": digest,
    }


def manifest_digest(entries: list[dict[str, str]]) -> str:
    """Digest the inventory itself, so the manifest is checkable without re-hashing."""
    import hashlib

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}:{entry['sha256']}\n".encode())
    return digest.hexdigest()


def write_integrity_manifest(root: Path, target: Path | None = None) -> Path:
    destination = target or (root / MANIFEST_NAME)
    payload = build_manifest(root, target=destination)
    write_json(destination, payload)
    return destination


def verify_manifest(root: Path, target: Path | None = None) -> dict[str, Any]:
    """Compare the shipped manifest with the live tree, in both directions."""
    destination = target or (root / MANIFEST_NAME)
    payload = read_json(destination)
    listed = {entry["path"]: entry["sha256"] for entry in payload["files"]}
    on_disk = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in inventory(root, exclude=destination.resolve())
    }
    stale = sorted(name for name, digest in listed.items() if on_disk.get(name) != digest)
    unlisted = sorted(name for name in on_disk if name not in listed)
    recomputed = manifest_digest(payload["files"])
    return {
        "n_listed": len(listed),
        "n_on_disk": len(on_disk),
        "stale": stale,
        "unlisted": unlisted,
        "digest_matches": recomputed == payload["manifest_digest"],
        "intact": not stale and not unlisted and recomputed == payload["manifest_digest"],
    }
