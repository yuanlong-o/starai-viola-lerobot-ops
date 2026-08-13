"""Small publication-boundary guards for immutable Repo-A evidence."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import viola_handoff

from .errors import ValidationError
from .jsonutil import regular_file


@dataclass(frozen=True, slots=True)
class TreeSnapshot:
    """One directory's identity and exact public handoff inventory."""

    label: str
    path: Path
    device: int
    inode: int
    inventory: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """One source file's identity and bytes, including same-byte replacement."""

    label: str
    path: Path
    device: int
    inode: int
    size_bytes: int
    mode: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicationSnapshot:
    """All local material that must remain stable through publication."""

    payload: TreeSnapshot
    artifacts: tuple[tuple[str, TreeSnapshot], ...]
    sources: tuple[FileSnapshot, ...] = ()

    @property
    def aggregate_inventory_sha256(self) -> str:
        material = {
            "payload": self.payload.inventory["inventory_sha256"],
            "artifacts": [
                {
                    "name": name,
                    "inventory_sha256": tree.inventory["inventory_sha256"],
                }
                for name, tree in self.artifacts
            ],
        }
        return hashlib.sha256(viola_handoff.canonical_json_bytes(material)).hexdigest()

    def require_unchanged(self, *, boundary: str) -> None:
        trees = (self.payload, *(tree for _name, tree in self.artifacts))
        for expected in trees:
            current = snapshot_tree(expected.path, label=expected.label)
            if (current.device, current.inode) != (expected.device, expected.inode):
                raise ValidationError(f"{expected.label} changed {boundary}")
            if current.inventory != expected.inventory:
                raise ValidationError(f"{expected.label} bytes changed {boundary}")
        for expected in self.sources:
            current = snapshot_file(expected.path, label=expected.label)
            if current != expected:
                raise ValidationError(f"{expected.label} changed {boundary}")


@dataclass(frozen=True, slots=True)
class GuardedEvidenceLogger:
    """Recheck exact local material on both sides of the handoff W&B call."""

    delegate: viola_handoff.EvidenceLogger
    snapshot: PublicationSnapshot
    identity: viola_handoff.RuntimeIdentity
    identity_capture: Callable[[], viola_handoff.RuntimeIdentity]
    operation: str
    source_validator: Callable[[], None] | None = None

    def record(
        self,
        *,
        project: str,
        run_id: str,
        event: str,
        metadata: Mapping[str, str | int | float | bool | None],
    ) -> str:
        self._require_expected_metadata(metadata)
        require_same_runtime(
            self.identity, capture=self.identity_capture, operation=self.operation
        )
        self.snapshot.require_unchanged(boundary="before handoff W&B publication")
        if self.source_validator is not None:
            self.source_validator()
        url = self.delegate.record(
            project=project,
            run_id=run_id,
            event=event,
            metadata=metadata,
        )
        self._require_expected_metadata(metadata)
        require_same_runtime(
            self.identity, capture=self.identity_capture, operation=self.operation
        )
        self.snapshot.require_unchanged(boundary="during handoff W&B publication")
        if self.source_validator is not None:
            self.source_validator()
        return url

    def _require_expected_metadata(
        self, metadata: Mapping[str, str | int | float | bool | None]
    ) -> None:
        if metadata.get("inventory_sha256") != self.snapshot.aggregate_inventory_sha256:
            raise ValidationError(
                f"handoff sealer inventoried different {self.operation} material"
            )
        if metadata.get("actor_commit") != self.identity.repository_commit:
            raise ValidationError(
                f"handoff sealer used a different {self.operation} identity"
            )


def snapshot_tree(path: str | Path, *, label: str) -> TreeSnapshot:
    """Capture a nonsymlink directory inode and its public handoff inventory."""

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValidationError(f"cannot inspect {label} {root}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValidationError(f"symlink path is forbidden for {label}: {current}")
    try:
        before = root.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise ValidationError(f"{label} is not a directory: {root}")
        inventory = viola_handoff.inventory_root(root)
        after = root.lstat()
    except ValidationError:
        raise
    except (OSError, viola_handoff.HandoffError) as exc:
        raise ValidationError(f"cannot inventory {label} {root}: {exc}") from exc
    if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise ValidationError(f"{label} directory changed while it was inventoried")
    return TreeSnapshot(label, root, before.st_dev, before.st_ino, inventory)


def snapshot_file(path: str | Path, *, label: str) -> FileSnapshot:
    """Hash one nonsymlink file through its descriptor and bind the path inode."""

    source = regular_file(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValidationError(f"cannot safely open {label} {source}: {exc}") from exc

    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValidationError(f"{label} is not a regular file: {source}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        named = source.lstat()
    except OSError as exc:
        raise ValidationError(f"cannot recheck {label} {source}: {exc}") from exc
    opened_identity = _file_identity(opened)
    if opened_identity != _file_identity(finished) or opened_identity != _file_identity(named):
        raise ValidationError(f"{label} changed while it was snapshotted")
    return FileSnapshot(
        label,
        source,
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mode,
        digest.hexdigest(),
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_mtime_ns,
    )


def require_same_runtime(
    expected: viola_handoff.RuntimeIdentity,
    *,
    capture: Callable[[], viola_handoff.RuntimeIdentity],
    operation: str,
) -> None:
    """Recapture and require the exact clean identity used in the manifest."""

    try:
        current = capture()
    except viola_handoff.HandoffError as exc:
        raise ValidationError(f"cannot recapture clean Repo-A runtime: {exc}") from exc
    if not isinstance(current, viola_handoff.RuntimeIdentity) or current != expected:
        raise ValidationError(
            f"Repo-A commit or runtime identity changed before {operation} sealing"
        )


def require_sealed_bundle_matches(
    bundle: viola_handoff.VerifiedBundle,
    request: viola_handoff.SealRequest,
    snapshot: PublicationSnapshot,
    *,
    operation: str,
    permission: str,
    consumer_role: str,
) -> None:
    """Reject a new or idempotent READY result for different material."""

    if not isinstance(bundle, viola_handoff.VerifiedBundle):
        raise ValidationError(f"handoff sealer returned no verified {operation} bundle")
    expected_artifacts = [
        {"name": name, "root": str(tree.path), **tree.inventory}
        for name, tree in snapshot.artifacts
    ]
    manifest = bundle.manifest
    producer = request.producer
    expected_producer = {
        "role": producer.role,
        "repository_commit": producer.repository_commit,
        "repository_clean": producer.repository_clean,
        "hostname": producer.hostname,
        "python_version": producer.python_version,
        "lerobot_version": producer.lerobot_version,
        "conda_environment": producer.conda_environment,
    }
    bindings_match = (
        manifest.get("kind") == request.kind
        and manifest.get("experiment") == request.experiment
        and manifest.get("subject") == request.subject
        and manifest.get("producer") == expected_producer
        and manifest.get("lineage") == dict(request.lineage)
        and manifest.get("permission") == permission
        and manifest.get("consumer") == {"role": consumer_role}
        and manifest.get("wandb", {}).get("project") == request.wandb_project
    )
    if not bindings_match:
        raise ValidationError(f"sealed {operation} bindings differ from the request")
    if manifest.get("payload") != snapshot.payload.inventory:
        raise ValidationError(f"sealed payload differs from the prepared {operation}")
    if manifest.get("artifacts") != expected_artifacts:
        raise ValidationError(f"sealed artifacts differ from the prepared {operation}")
    manifest_sha256 = hashlib.sha256(
        viola_handoff.canonical_json_bytes(manifest)
    ).hexdigest()
    ready = bundle.ready
    ready_matches = (
        ready.get("bundle_id") == manifest.get("bundle_id")
        and ready.get("content_id") == manifest.get("content_id")
        and ready.get("manifest_sha256") == manifest_sha256
        and ready.get("inventory_sha256") == snapshot.aggregate_inventory_sha256
    )
    if not ready_matches:
        raise ValidationError(f"{operation} READY differs from the sealed manifest")


__all__ = [
    "FileSnapshot",
    "GuardedEvidenceLogger",
    "PublicationSnapshot",
    "TreeSnapshot",
    "require_same_runtime",
    "require_sealed_bundle_matches",
    "snapshot_file",
    "snapshot_tree",
]
