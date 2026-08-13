"""Reviewed setup operations that never command a motor."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viola_handoff import RuntimeIdentity, canonical_json_bytes

from .errors import SafetyGateError, ValidationError
from .hardware import (
    JOINTS,
    PublicFashionStarPort,
    degrees_to_raw,
    load_calibration,
    raw_to_normalized,
)
from .jsonutil import (
    read_json_object,
    regular_file,
    require_exact_keys,
    sha256_file,
    sha256_json,
    write_canonical_json,
)
from .session_inputs import load_reviewed_setup
from .wandb_ops import WandbRunIdentity, planned_run, publish_finished_run

_STATE_FILE = "frozen_state.json"
_CAPTURE_FILE = "frozen_state_capture.json"
_SYNC_FILE = "frozen_state_capture_WANDB_SYNCED.json"
_STATE_FIELDS = {
    "schema_version",
    "state",
    "captured_at",
    "robot_connected_at_capture",
    "motor_disconnected_at",
}
_CAPTURE_FIELDS = {
    "schema_version",
    "kind",
    "status",
    "setup_id",
    "frozen_state_file",
    "frozen_state_sha256",
    "state_sha256",
    "setup_hashes",
    "operator",
    "captured_at",
    "motor_disconnected_at",
    "repo",
    "wandb",
}
_SYNC_FIELDS = {
    "schema_version",
    "operation",
    "evidence_file",
    "evidence_sha256",
    "wandb",
    "binding",
    "synced_at",
}
_REPO_FIELDS = {"commit", "clean", "hostname", "python"}
_SETUP_HASH_FIELDS = {"calibration", "camera", "robot", "reset"}
_SYNC_BINDING_FIELDS = {
    "setup_id",
    "state_sha256",
    "status",
    "repo_commit",
    "setup_hashes",
    "frozen_state_sha256",
}


@dataclass(frozen=True, slots=True)
class FrozenStateCapture:
    output_root: Path
    state_path: Path
    evidence_path: Path
    sync_path: Path
    state: tuple[float, ...]
    wandb: WandbRunIdentity
    upload_only: bool = False

    @property
    def binding(self) -> dict[str, Any]:
        capture = _json(self.evidence_path)
        return {
            "setup_id": capture["setup_id"],
            "evidence_sha256": sha256_file(self.evidence_path),
            "frozen_state_sha256": sha256_file(self.state_path),
            "state_sha256": capture["state_sha256"],
            "wandb": self.wandb.binding(),
        }

    def render_text(self) -> str:
        """Describe the completed read-only capture in operator language."""

        title = (
            "Frozen-state evidence upload recovered"
            if self.upload_only
            else "Frozen state captured"
        )
        connection = (
            "not opened during upload-only recovery"
            if self.upload_only
            else "closed before evidence publication"
        )
        return "\n".join(
            [
                title,
                f"  Seven-axis state: {self.state_path}",
                f"  Capture evidence: {self.evidence_path}",
                f"  W&B receipt: {self.sync_path}",
                f"  W&B run: {self.wandb.url}",
                "  Motor commands sent: none",
                f"  Serial connection: {connection}",
            ]
        )


def capture_frozen_state(
    setup_path: str | Path,
    *,
    output_root: str | Path,
    operator: str,
    repo_root: str | Path,
    wandb_entity: str,
    wandb_project: str = "starai-viola-policy-benchmark",
    confirm: Callable[[str], bool] | None = None,
    port_factory: Callable[[str, int], PublicFashionStarPort] | None = None,
    publisher: Callable[..., WandbRunIdentity] = publish_finished_run,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    upload_only: bool = False,
) -> FrozenStateCapture:
    """Capture positions, or recover publication of an immutable capture.

    Fresh capture requires a new output directory and an operator confirmation.
    ``upload_only`` accepts only an exact capture left after publication failed;
    it does not confirm, construct a serial port, or read hardware again.
    """

    setup = load_reviewed_setup(setup_path)
    if (
        not isinstance(operator, str)
        or not operator.strip()
        or operator != setup.estop["operator"]
    ):
        raise ValidationError("capture operator must equal the reviewed E-stop operator")
    identity = RuntimeIdentity.capture(role="pc_a", repo_root=repo_root)
    _validate_identity(identity, setup)
    root = _absolute_path(output_root)
    _require_external_root(root, repo_root)

    if upload_only:
        pinned_root = _pin_capture_root(root)
        try:
            material = _load_capture(
                root,
                setup,
                operator,
                identity,
                wandb_entity,
                wandb_project,
            )
            return _publish_and_sync(
                root,
                pinned_root,
                setup,
                identity,
                material,
                publisher=publisher,
                now=now,
                upload_only=True,
                repo_root=repo_root,
            )
        finally:
            pinned_root.close()

    # The leaf mkdir is the atomic reservation for one fresh capture.  It is
    # deliberately complete before confirmation or serial-port construction.
    reservation = _create_capture_directory(root)
    try:
        try:
            challenge = f"READ FROZEN STATE {setup.setup_id}"
            if confirm is None or confirm(challenge) is not True:
                raise SafetyGateError(
                    "frozen-state capture requires explicit operator confirmation"
                )
            calibration = load_calibration(setup.calibration_path)
            values, captured_at, disconnected_at = _read_positions(
                setup,
                calibration,
                port_factory=port_factory,
                now=now,
            )
            material = _write_capture(
                root,
                setup,
                operator,
                identity,
                values,
                captured_at,
                disconnected_at,
                wandb_entity,
                wandb_project,
            )
            # Acquire the publication pin while the reservation descriptor is
            # still open, so no close/reopen gap exists for this directory.
            pinned_root = _pin_capture_root(root, reservation.identity)
        except BaseException as exc:
            _clean_incomplete_capture(reservation, exc)
            raise
    finally:
        reservation.close()

    try:
        return _publish_and_sync(
            root,
            pinned_root,
            setup,
            identity,
            material,
            publisher=publisher,
            now=now,
            upload_only=False,
            repo_root=repo_root,
        )
    finally:
        pinned_root.close()


@dataclass(frozen=True, slots=True)
class _CaptureMaterial:
    state_path: Path
    evidence_path: Path
    sync_path: Path
    state: tuple[float, ...]
    captured_at: datetime
    disconnected_at: datetime
    wandb: WandbRunIdentity
    existing_sync: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _PublicationSnapshot:
    """Exact local evidence and reviewed sources bound to one W&B upload."""

    layout: frozenset[str]
    state_file_sha256: str
    evidence_file_sha256: str
    sync_file_sha256: str | None
    setup_source_hashes: tuple[tuple[str, str], ...]
    setup_hashes: tuple[tuple[str, str], ...]
    state_sha256: str
    status: str
    frozen_state_sha256: str

    def setup_hash_mapping(self) -> dict[str, str]:
        return dict(self.setup_hashes)


@dataclass(frozen=True, slots=True)
class _PinnedCaptureRoot:
    """An open descriptor that pins one capture directory during publication."""

    path: Path
    descriptor: int
    identity: _DirectoryIdentity

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(frozen=True, slots=True)
class _CaptureReservation:
    """Open handles that keep cleanup bound to the directory we created."""

    root: Path
    parent_fd: int
    directory_fd: int
    device: int
    inode: int

    @property
    def identity(self) -> _DirectoryIdentity:
        return _DirectoryIdentity(self.device, self.inode)

    def close(self) -> None:
        os.close(self.directory_fd)
        os.close(self.parent_fd)


def _read_positions(
    setup: Any,
    calibration: Mapping[str, Any],
    *,
    port_factory: Callable[[str, int], PublicFashionStarPort] | None,
    now: Callable[[], datetime],
) -> tuple[tuple[float, ...], datetime, datetime]:
    """Read one complete state and preserve an active error if close also fails."""

    factory = port_factory or _fashionstar_port
    port = factory(setup.robot_port, 1_000_000)
    open_attempted = False
    try:
        open_attempted = True
        if port.openPort() is False:
            raise ValidationError("FashionStar serial port did not open")
        for joint in JOINTS:
            if not port.ping(calibration[joint].id):
                raise ValidationError(f"servo {calibration[joint].id} ({joint}) did not respond")
        ids = {joint: calibration[joint].id for joint in JOINTS}
        monitors = port.SyncServoMonitor(ids, realtime=True)
        if set(monitors) != set(JOINTS):
            raise ValidationError("synchronous monitor did not return all seven joints")
        values = tuple(
            raw_to_normalized(
                degrees_to_raw(_finite(monitors[joint].current_position, f"{joint} position")),
                calibration[joint],
                gripper=joint == "gripper",
            )
            for joint in JOINTS
        )
        captured_at = _utc(now())
    except BaseException as exc:
        if open_attempted:
            try:
                _close_port(port)
            except BaseException as close_error:
                exc.add_note(
                    "Serial-port cleanup also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        raise
    else:
        if open_attempted:
            _close_port(port)

    disconnected_at = _utc(now())
    if disconnected_at <= captured_at:
        raise ValidationError("disconnect timestamp must follow frozen-state capture")
    return values, captured_at, disconnected_at


def _close_port(port: PublicFashionStarPort) -> None:
    if port.closePort() is False:
        raise ValidationError("FashionStar serial port did not report a clean close")


def _write_capture(
    root: Path,
    setup: Any,
    operator: str,
    identity: RuntimeIdentity,
    values: tuple[float, ...],
    captured_at: datetime,
    disconnected_at: datetime,
    wandb_entity: str,
    wandb_project: str,
) -> _CaptureMaterial:
    state_payload = {
        "schema_version": 1,
        "state": list(values),
        "captured_at": captured_at.isoformat(),
        "robot_connected_at_capture": True,
        "motor_disconnected_at": disconnected_at.isoformat(),
    }
    state_path = write_canonical_json(root / _STATE_FILE, state_payload)
    setup_hashes = _setup_hashes(setup)
    state_sha256 = sha256_json(list(values))
    wandb = _capture_run(
        setup.setup_id,
        state_sha256,
        captured_at,
        identity.repository_commit,
        wandb_entity,
        wandb_project,
    )
    repo = _repo_binding(identity)
    evidence = {
        "schema_version": 1,
        "kind": "frozen_state_capture",
        "status": "captured_disconnected",
        "setup_id": setup.setup_id,
        "frozen_state_file": _STATE_FILE,
        "frozen_state_sha256": sha256_file(state_path),
        "state_sha256": state_sha256,
        "setup_hashes": setup_hashes,
        "operator": operator,
        "captured_at": captured_at.isoformat(),
        "motor_disconnected_at": disconnected_at.isoformat(),
        "repo": repo,
        "wandb": wandb.binding(),
    }
    evidence_path = write_canonical_json(root / _CAPTURE_FILE, evidence)
    return _CaptureMaterial(
        state_path,
        evidence_path,
        root / _SYNC_FILE,
        values,
        captured_at,
        disconnected_at,
        wandb,
    )


def _publish_and_sync(
    root: Path,
    pinned_root: _PinnedCaptureRoot,
    setup: Any,
    identity: RuntimeIdentity,
    material: _CaptureMaterial,
    *,
    publisher: Callable[..., WandbRunIdentity],
    now: Callable[[], datetime],
    upload_only: bool,
    repo_root: str | Path,
) -> FrozenStateCapture:
    """Publish the deterministic run, then create or reuse its exact receipt."""

    publication_changed = (
        "capture evidence or reviewed setup changed during W&B publication"
    )
    sync_changed = (
        "capture evidence, reviewed setup, or Repo-A identity changed while "
        "writing the W&B sync receipt"
    )
    snapshot = _publication_snapshot(
        root,
        setup,
        identity,
        material,
        expected_sync=material.existing_sync,
    )
    setup_hashes = snapshot.setup_hash_mapping()
    config = {
        "operation": "frozen_state_capture",
        "evidence_sha256": snapshot.evidence_file_sha256,
        "setup_id": setup.setup_id,
        "status": snapshot.status,
        "repo_commit": identity.repository_commit,
        "setup_hashes": setup_hashes,
        "setup_source_hashes": dict(snapshot.setup_source_hashes),
        "frozen_state_sha256": snapshot.frozen_state_sha256,
        "state_sha256": snapshot.state_sha256,
    }
    _require_capture_root(pinned_root)
    _require_identity_unchanged(identity, repo_root)
    try:
        published = publisher(
            material.wandb,
            job_type="viola-frozen-state-capture",
            config=config,
            summary={"state_dimensions": 7, "serial_closed": True},
        )
        if published != material.wandb:
            raise ValidationError("W&B publisher returned another frozen-state run identity")
    except BaseException as exc:
        exc.add_note(
            f"Immutable capture evidence remains at {root}; retry with --upload-only."
        )
        raise

    try:
        current = _revalidate_publication(
            root,
            pinned_root,
            setup,
            identity,
            material,
            repo_root,
            expected_sync=material.existing_sync,
            failure_message=publication_changed,
        )
        if current != snapshot:
            raise ValidationError(publication_changed)
        if material.existing_sync is None:
            synced_at = _utc(now())
            if synced_at < material.disconnected_at:
                raise ValidationError("W&B sync timestamp cannot precede motor disconnection")
            sync = _sync_payload(
                setup_id=setup.setup_id,
                identity=identity,
                wandb=material.wandb,
                evidence_sha256=snapshot.evidence_file_sha256,
                setup_hashes=setup_hashes,
                state_sha256=snapshot.state_sha256,
                frozen_state_sha256=snapshot.frozen_state_sha256,
                synced_at=synced_at,
            )
        else:
            sync = material.existing_sync
        sync_path = write_canonical_json(material.sync_path, sync)

        expected_final = replace(
            snapshot,
            layout=snapshot.layout | {_SYNC_FILE},
            sync_file_sha256=sha256_json(sync),
        )
        current = _revalidate_publication(
            root,
            pinned_root,
            setup,
            identity,
            material,
            repo_root,
            expected_sync=sync,
            failure_message=sync_changed,
        )
        if current != expected_final:
            raise ValidationError(sync_changed)
    except BaseException as exc:
        exc.add_note(
            f"The finished run has no reusable local receipt at {root}; retry --upload-only."
        )
        raise

    return FrozenStateCapture(
        root,
        material.state_path,
        material.evidence_path,
        sync_path,
        material.state,
        material.wandb,
        upload_only,
    )


def _load_capture(
    root: Path,
    setup: Any,
    operator: str,
    identity: RuntimeIdentity,
    wandb_entity: str,
    wandb_project: str,
) -> _CaptureMaterial:
    """Validate every byte needed for a hardware-free publication retry."""

    _require_existing_directory(root)
    names = _directory_names(root)
    required = {_STATE_FILE, _CAPTURE_FILE}
    allowed = required | {_SYNC_FILE}
    if not required.issubset(names) or not names.issubset(allowed):
        raise ValidationError(
            "upload-only capture directory must contain exactly frozen_state.json, "
            "frozen_state_capture.json, and optionally its W&B sync receipt"
        )

    state_path = root / _STATE_FILE
    evidence_path = root / _CAPTURE_FILE
    sync_path = root / _SYNC_FILE
    state = _read_immutable_json(state_path, "frozen state")
    capture = _read_immutable_json(evidence_path, "frozen-state capture")
    require_exact_keys(state, _STATE_FIELDS, label="frozen state")
    require_exact_keys(capture, _CAPTURE_FIELDS, label="frozen-state capture")

    values = _state_values(state["state"])
    captured_at = _canonical_timestamp(state["captured_at"], "frozen state captured_at")
    disconnected_at = _canonical_timestamp(
        state["motor_disconnected_at"], "frozen state motor_disconnected_at"
    )
    expected_state = {
        "schema_version": 1,
        "state": list(values),
        "captured_at": captured_at.isoformat(),
        "robot_connected_at_capture": True,
        "motor_disconnected_at": disconnected_at.isoformat(),
    }
    if state != expected_state or state_path.read_bytes() != canonical_json_bytes(expected_state):
        raise ValidationError("frozen state differs from the exact capture format")
    if disconnected_at <= captured_at:
        raise ValidationError("disconnect timestamp must follow frozen-state capture")

    setup_hashes = _setup_hashes(setup)
    repo = _repo_binding(identity)
    state_sha256 = sha256_json(list(values))
    wandb = _capture_run(
        setup.setup_id,
        state_sha256,
        captured_at,
        identity.repository_commit,
        wandb_entity,
        wandb_project,
    )
    require_exact_keys(capture["setup_hashes"], _SETUP_HASH_FIELDS, label="setup hashes")
    require_exact_keys(capture["repo"], _REPO_FIELDS, label="capture repository")
    expected_capture = {
        "schema_version": 1,
        "kind": "frozen_state_capture",
        "status": "captured_disconnected",
        "setup_id": setup.setup_id,
        "frozen_state_file": _STATE_FILE,
        "frozen_state_sha256": sha256_file(state_path),
        "state_sha256": state_sha256,
        "setup_hashes": setup_hashes,
        "operator": operator,
        "captured_at": captured_at.isoformat(),
        "motor_disconnected_at": disconnected_at.isoformat(),
        "repo": repo,
        "wandb": wandb.binding(),
    }
    if capture != expected_capture:
        raise ValidationError(
            "frozen-state capture does not match the reviewed setup, operator, revision, or W&B run"
        )

    existing_sync: Mapping[str, Any] | None = None
    if _SYNC_FILE in names:
        sync = _read_immutable_json(sync_path, "frozen-state W&B sync receipt")
        require_exact_keys(sync, _SYNC_FIELDS, label="frozen-state W&B sync receipt")
        require_exact_keys(sync["binding"], _SYNC_BINDING_FIELDS, label="W&B sync binding")
        synced_at = _canonical_timestamp(sync["synced_at"], "W&B sync timestamp")
        if synced_at < disconnected_at:
            raise ValidationError("W&B sync timestamp cannot precede motor disconnection")
        expected_sync = _sync_payload(
            setup_id=setup.setup_id,
            identity=identity,
            wandb=wandb,
            evidence_sha256=sha256_file(evidence_path),
            setup_hashes=setup_hashes,
            state_sha256=state_sha256,
            frozen_state_sha256=sha256_file(state_path),
            synced_at=synced_at,
        )
        if sync != expected_sync:
            raise ValidationError("frozen-state W&B sync receipt is stale or mismatched")
        existing_sync = sync

    return _CaptureMaterial(
        state_path,
        evidence_path,
        sync_path,
        values,
        captured_at,
        disconnected_at,
        wandb,
        existing_sync,
    )


def _publication_snapshot(
    root: Path,
    setup: Any,
    identity: RuntimeIdentity,
    material: _CaptureMaterial,
    *,
    expected_sync: Mapping[str, Any] | None,
) -> _PublicationSnapshot:
    """Read the exact local inputs that a finished W&B run will describe."""

    expected_paths = {
        _STATE_FILE: root / _STATE_FILE,
        _CAPTURE_FILE: root / _CAPTURE_FILE,
        _SYNC_FILE: root / _SYNC_FILE,
    }
    if (
        material.state_path != expected_paths[_STATE_FILE]
        or material.evidence_path != expected_paths[_CAPTURE_FILE]
        or material.sync_path != expected_paths[_SYNC_FILE]
    ):
        raise ValidationError("frozen-state material paths do not match the capture directory")

    expected_layout = {_STATE_FILE, _CAPTURE_FILE}
    if expected_sync is not None:
        expected_layout.add(_SYNC_FILE)
    _require_existing_directory(root)
    if _directory_names(root) != expected_layout:
        raise ValidationError("frozen-state capture directory layout changed before publication")

    state = _read_immutable_json(material.state_path, "frozen state")
    evidence = _read_immutable_json(material.evidence_path, "frozen-state capture")
    require_exact_keys(state, _STATE_FIELDS, label="frozen state")
    require_exact_keys(evidence, _CAPTURE_FIELDS, label="frozen-state capture")
    expected_state = {
        "schema_version": 1,
        "state": list(material.state),
        "captured_at": material.captured_at.isoformat(),
        "robot_connected_at_capture": True,
        "motor_disconnected_at": material.disconnected_at.isoformat(),
    }
    if state != expected_state:
        raise ValidationError("frozen state changed before publication")

    setup_hashes = _setup_hashes(setup)
    state_sha256 = sha256_json(list(material.state))
    state_file_sha256 = sha256_file(material.state_path)
    if (
        evidence["setup_id"] != setup.setup_id
        or evidence["status"] != "captured_disconnected"
        or evidence["state_sha256"] != state_sha256
        or evidence["frozen_state_sha256"] != state_file_sha256
        or evidence["setup_hashes"] != setup_hashes
        or evidence["repo"] != _repo_binding(identity)
        or evidence["wandb"] != material.wandb.binding()
    ):
        raise ValidationError("frozen-state capture binding changed before publication")

    sync_file_sha256: str | None = None
    if expected_sync is not None:
        sync = _read_immutable_json(material.sync_path, "frozen-state W&B sync receipt")
        if sync != expected_sync:
            raise ValidationError("frozen-state W&B sync receipt changed before publication")
        sync_file_sha256 = sha256_file(material.sync_path)

    return _PublicationSnapshot(
        layout=frozenset(expected_layout),
        state_file_sha256=state_file_sha256,
        evidence_file_sha256=sha256_file(material.evidence_path),
        sync_file_sha256=sync_file_sha256,
        setup_source_hashes=_reviewed_setup_source_hashes(setup),
        setup_hashes=tuple(sorted(setup_hashes.items())),
        state_sha256=state_sha256,
        status=evidence["status"],
        frozen_state_sha256=evidence["frozen_state_sha256"],
    )


def _revalidate_publication(
    root: Path,
    pinned_root: _PinnedCaptureRoot,
    setup: Any,
    identity: RuntimeIdentity,
    material: _CaptureMaterial,
    repo_root: str | Path,
    *,
    expected_sync: Mapping[str, Any] | None,
    failure_message: str,
) -> _PublicationSnapshot:
    """Recapture every local input while the original output root stays pinned."""

    try:
        _require_identity_unchanged(identity, repo_root)
        _require_capture_root(pinned_root)
        current_setup = load_reviewed_setup(setup.source_path)
        if current_setup != setup:
            raise ValidationError("reviewed setup changed")
        snapshot = _publication_snapshot(
            root,
            current_setup,
            identity,
            material,
            expected_sync=expected_sync,
        )
        # Snapshot construction performs several filesystem reads.  Pin and
        # identity checks on both sides keep its result tied to this checkout
        # and this exact output directory.
        _require_capture_root(pinned_root)
        _require_identity_unchanged(identity, repo_root)
        return snapshot
    except ValidationError as exc:
        raise ValidationError(failure_message) from exc


def _reviewed_setup_source_hashes(setup: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "reviewed_setup": sha256_file(setup.source_path),
                "calibration": sha256_file(setup.calibration_path),
                "reset": sha256_file(setup.reset_protocol_path),
                "entrypoint": sha256_file(setup.executor_entrypoint),
            }.items()
        )
    )


def _sync_payload(
    setup_id: str,
    identity: RuntimeIdentity,
    *,
    wandb: WandbRunIdentity,
    evidence_sha256: str,
    setup_hashes: Mapping[str, str],
    state_sha256: str,
    frozen_state_sha256: str,
    synced_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "operation": "frozen_state_capture",
        "evidence_file": _CAPTURE_FILE,
        "evidence_sha256": evidence_sha256,
        "wandb": wandb.binding(),
        "binding": {
            "setup_id": setup_id,
            "state_sha256": state_sha256,
            "status": "captured_disconnected",
            "repo_commit": identity.repository_commit,
            "setup_hashes": dict(setup_hashes),
            "frozen_state_sha256": frozen_state_sha256,
        },
        "synced_at": synced_at.isoformat(),
    }


def _capture_run(
    setup_id: str,
    state_sha256: str,
    captured_at: datetime,
    repository_commit: str,
    wandb_entity: str,
    wandb_project: str,
) -> WandbRunIdentity:
    run_seed = sha256_json(
        {
            "setup_id": setup_id,
            "state_sha256": state_sha256,
            "captured_at": captured_at.isoformat(),
            "repo": repository_commit,
        }
    )
    return planned_run(wandb_entity, wandb_project, f"capture-{run_seed[:16]}")


def _validate_identity(identity: RuntimeIdentity, setup: Any) -> None:
    if identity.role != "pc_a" or identity.repository_clean is not True:
        raise ValidationError("frozen-state evidence requires a current clean pc_a identity")
    if identity.repository_commit != setup.executor["commit"]:
        raise ValidationError("current clean revision differs from the reviewed executor")
    if (
        identity.python_version != setup.executor["python_version"]
        or identity.lerobot_version != setup.executor["lerobot_version"]
        or identity.conda_environment != setup.executor["conda_environment"]
    ):
        raise ValidationError("current runtime differs from the reviewed executor environment")


def _repo_binding(identity: RuntimeIdentity) -> dict[str, Any]:
    return {
        "commit": identity.repository_commit,
        "clean": identity.repository_clean,
        "hostname": identity.hostname,
        "python": identity.python_version,
    }


def _pin_capture_root(
    root: Path,
    expected: _DirectoryIdentity | None = None,
) -> _PinnedCaptureRoot:
    """Keep one output directory open and verify its pathname still names it."""

    _require_existing_directory(root)
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise ValidationError(f"cannot safely open capture output_root {root}: {exc}") from exc

    opened = os.fstat(descriptor)
    identity = _DirectoryIdentity(opened.st_dev, opened.st_ino)
    pinned = _PinnedCaptureRoot(root, descriptor, identity)
    try:
        _require_capture_root(pinned)
        if expected is not None and identity != expected:
            raise ValidationError(
                "capture output_root no longer names the reserved directory"
            )
    except BaseException:
        pinned.close()
        raise
    return pinned


def _require_capture_root(pinned: _PinnedCaptureRoot) -> None:
    """Require the path and open descriptor to identify the same directory."""

    try:
        opened = os.fstat(pinned.descriptor)
        current = pinned.path.lstat()
    except OSError as exc:
        raise ValidationError(
            f"capture output_root changed during publication: {pinned.path}"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or _DirectoryIdentity(opened.st_dev, opened.st_ino) != pinned.identity
        or _DirectoryIdentity(current.st_dev, current.st_ino) != pinned.identity
    ):
        raise ValidationError(
            f"capture output_root changed during publication: {pinned.path}"
        )


def _create_capture_directory(root: Path) -> _CaptureReservation:
    """Atomically reserve and open a new leaf without following symlinks."""

    _safe_directory(root.parent, create=True)
    try:
        parent_fd = os.open(
            root.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise ValidationError(
            f"cannot open capture output parent {root.parent}: {exc}"
        ) from exc
    try:
        os.mkdir(root.name, 0o755, dir_fd=parent_fd)
    except FileExistsError as exc:
        os.close(parent_fd)
        raise ValidationError(
            f"fresh capture output_root must not already exist: {root}"
        ) from exc
    except OSError as exc:
        os.close(parent_fd)
        raise ValidationError(f"cannot create capture output_root {root}: {exc}") from exc

    try:
        directory_fd = os.open(
            root.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        opened = os.fstat(directory_fd)
        current = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise ValidationError(
                f"capture output_root changed while it was being reserved: {root}"
            )
    except BaseException:
        if "directory_fd" in locals():
            os.close(directory_fd)
        os.close(parent_fd)
        raise
    return _CaptureReservation(
        root,
        parent_fd,
        directory_fd,
        opened.st_dev,
        opened.st_ino,
    )


def _require_existing_directory(path: Path) -> None:
    _safe_directory(path, create=False)


def _safe_directory(path: Path, *, create: bool) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            if not create:
                raise ValidationError(f"capture output directory does not exist: {path}") from exc
            try:
                os.mkdir(current, 0o755)
                mode = current.lstat().st_mode
            except FileExistsError:
                mode = current.lstat().st_mode
            except OSError as create_error:
                raise ValidationError(
                    f"cannot prepare capture output parent {current}: {create_error}"
                ) from create_error
        except OSError as exc:
            raise ValidationError(f"cannot inspect capture output path {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink capture output path is forbidden: {current}")
        if not stat.S_ISDIR(mode):
            raise ValidationError(f"capture output parent is not a directory: {current}")


def _directory_names(root: Path) -> set[str]:
    try:
        return {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise ValidationError(f"cannot inspect capture output directory {root}: {exc}") from exc


def _read_immutable_json(path: Path, label: str) -> Mapping[str, Any]:
    source = regular_file(path, label=label)
    if source.lstat().st_mode & 0o222:
        raise ValidationError(f"{label} must be immutable (no write permission): {source}")
    value = read_json_object(source, label=label)
    if source.read_bytes() != canonical_json_bytes(value):
        raise ValidationError(f"{label} must use exact canonical JSON bytes")
    # sha256_file performs a descriptor-level replacement/change check.
    sha256_file(source)
    return value


def _state_values(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(JOINTS):
        raise ValidationError("frozen state must contain exactly seven positions")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError(f"frozen state position {index} must be a number")
        result.append(_finite(item, f"frozen state position {index}"))
    return tuple(result)


def _canonical_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a UTC offset")
    result = parsed.astimezone(UTC)
    if value != result.isoformat():
        raise ValidationError(f"{label} must use the canonical UTC representation")
    return result


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _require_external_root(root: Path, repo_root: str | Path) -> None:
    repository = _absolute_path(repo_root)
    try:
        root.relative_to(repository)
    except ValueError:
        return
    raise ValidationError(
        f"frozen-state evidence must be outside the Repo-A worktree: {root}"
    )


def _require_identity_unchanged(
    expected: RuntimeIdentity,
    repo_root: str | Path,
) -> None:
    current = RuntimeIdentity.capture(role="pc_a", repo_root=repo_root)
    if current != expected:
        raise ValidationError(
            "Repo-A revision or runtime identity changed during frozen-state capture"
        )


def _clean_incomplete_capture(
    reservation: _CaptureReservation,
    original: BaseException,
) -> None:
    """Clean only the exact directory reserved for this capture attempt."""

    cleanup_errors: list[str] = []
    for name in (_SYNC_FILE, _CAPTURE_FILE, _STATE_FILE):
        try:
            os.unlink(name, dir_fd=reservation.directory_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            cleanup_errors.append(f"{reservation.root / name}: {exc}")

    try:
        current = os.stat(
            reservation.root.name,
            dir_fd=reservation.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        cleanup_errors.append(
            f"{reservation.root}: reserved directory path was removed or replaced; "
            "replacement left untouched"
        )
    except OSError as exc:
        cleanup_errors.append(f"{reservation.root}: {exc}")
    else:
        same_directory = stat.S_ISDIR(current.st_mode) and (
            current.st_dev,
            current.st_ino,
        ) == (reservation.device, reservation.inode)
        if not same_directory:
            cleanup_errors.append(
                f"{reservation.root}: reserved directory path was replaced; "
                "replacement left untouched"
            )
        else:
            try:
                os.rmdir(reservation.root.name, dir_fd=reservation.parent_fd)
            except OSError as exc:
                cleanup_errors.append(f"{reservation.root}: {exc}")

    if cleanup_errors:
        original.add_note("Incomplete-capture cleanup also failed: " + "; ".join(cleanup_errors))
    else:
        original.add_note(f"Incomplete capture directory removed: {reservation.root}")


def interactive_confirmation(challenge: str) -> bool:
    """Read the exact capture challenge from a real terminal."""

    try:
        with open("/dev/tty", "r+", encoding="utf-8", buffering=1) as terminal:
            if not terminal.isatty():
                return False
            terminal.write(
                "This reads positions only and sends no motor command.\n"
                f"Type exactly: {challenge}\n> "
            )
            terminal.flush()
            return terminal.readline().rstrip("\r\n") == challenge
    except OSError:
        return False


def _setup_hashes(setup: Any) -> dict[str, str]:
    robot = {
        "robot_port": setup.robot_port,
        "joint_limits": setup.joint_limits,
        "max_step_deltas": setup.max_step_deltas,
        "speed_scale": setup.speed_scale,
    }
    return {
        "calibration": sha256_file(setup.calibration_path),
        "camera": sha256_json(setup.cameras),
        "robot": sha256_json(robot),
        "reset": sha256_file(setup.reset_protocol_path),
    }


def _fashionstar_port(port: str, baudrate: int) -> PublicFashionStarPort:
    from fashionstar_uart_sdk import PortHandler

    return PortHandler(port, baudrate)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not numeric") from exc
    if not float("-inf") < result < float("inf"):
        raise ValidationError(f"{label} is not finite")
    return result


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("capture clock must be timezone-aware")
    return value.astimezone(UTC)


def _json(path: Path) -> Mapping[str, Any]:
    import json

    return json.loads(path.read_bytes())


__all__ = ["FrozenStateCapture", "capture_frozen_state", "interactive_confirmation"]
