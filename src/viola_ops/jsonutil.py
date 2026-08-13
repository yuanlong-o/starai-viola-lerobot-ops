"""Small, strict JSON and hashing helpers used by Repo-A operations.

The handoff package owns the wire format.  These helpers keep the operator
code readable while using that exact canonical representation at the
repository boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from viola_handoff import canonical_json_bytes

from .errors import ValidationError


def _reject_json_constant(token: str) -> None:
    raise ValidationError(f"non-finite JSON value is forbidden: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_object(path: str | Path, *, label: str = "JSON file") -> dict[str, Any]:
    """Read a human-formatted JSON object without relaxing JSON safety."""

    source = regular_file(path, label=label)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {label} {source}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must contain one JSON object")
    return value


def sha256_file(path: str | Path) -> str:
    """Hash one regular file while detecting replacement during the read."""

    source = regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValidationError(f"cannot safely open {source}: {exc}") from exc

    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"expected a regular file: {source}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ValidationError(f"file changed while it was being hashed: {source}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def sha256_json(value: Any) -> str:
    """Hash JSON using the exact canonical encoding shared with Repo B."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_canonical_json(path: str | Path, value: Any) -> Path:
    """Create, or exactly reuse, one canonical JSON file.

    Existing divergent bytes are never overwritten.  This makes retries safe
    and avoids quietly changing evidence that another machine may already use.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        if destination.is_symlink() or not destination.is_file():
            raise ValidationError(f"existing output is not a regular file: {destination}") from exc
        if destination.read_bytes() != payload:
            raise ValidationError(f"refusing to replace different evidence: {destination}") from exc
        return destination
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def write_text_once(path: str | Path, text: str) -> Path:
    """Create, or exactly reuse, a UTF-8 text file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        if destination.is_symlink() or not destination.is_file():
            raise ValidationError(f"existing output is not a regular file: {destination}") from exc
        if destination.read_bytes() != payload:
            raise ValidationError(f"refusing to replace different evidence: {destination}") from exc
        return destination
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def copy_regular_file(source: str | Path, destination: str | Path) -> Path:
    """Copy immutable input bytes without following a source or target symlink."""

    source_path = regular_file(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = source_path.read_bytes()
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        if target.is_symlink() or not target.is_file():
            raise ValidationError(f"existing output is not a regular file: {target}") from exc
        if target.read_bytes() != payload:
            raise ValidationError(f"refusing to replace different material: {target}") from exc
        return target
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if sha256_file(source_path) != sha256_file(target):
        raise ValidationError(f"copy verification failed for {source_path}")
    return target


def regular_file(path: str | Path, *, label: str = "file") -> Path:
    """Return an absolute regular path only when no component is a symlink."""

    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValidationError(f"{label} does not exist: {candidate}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink path is forbidden for {label}: {current}")
    if not stat.S_ISREG(candidate.lstat().st_mode):
        raise ValidationError(f"{label} is not a regular file: {candidate}")
    return candidate


def require_exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    """Keep strict boundary checks legible at their call sites."""

    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{label} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


__all__ = [
    "copy_regular_file",
    "read_json_object",
    "regular_file",
    "require_exact_keys",
    "sha256_file",
    "sha256_json",
    "write_canonical_json",
    "write_text_once",
]
