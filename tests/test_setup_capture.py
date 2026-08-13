from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import viola_handoff
from viola_ops.errors import SafetyGateError, ValidationError
from viola_ops.jsonutil import sha256_file, sha256_json
from viola_ops.schemas import EXECUTOR_CAPABILITIES, VIOLA_JOINTS
from viola_ops.setup import capture_frozen_state


class _ReadOnlyPort:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.monitor_calls = 0

    def openPort(self):
        self.opened = True
        return True

    def closePort(self):
        self.closed = True

    def ping(self, _servo_id):
        return True

    def SyncServoMonitor(self, motors, realtime=False):
        assert realtime is True
        self.monitor_calls += 1
        return {joint: SimpleNamespace(current_position=0.0) for joint in motors}


def _setup(tmp_path: Path) -> Path:
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                joint: {
                    "id": index,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": 0,
                    "range_max": 4096,
                }
                for index, joint in enumerate(VIOLA_JOINTS)
            }
        ),
        encoding="utf-8",
    )
    reset = tmp_path / "reset.json"
    reset.write_text('{"manual":true}\n', encoding="utf-8")
    executor = tmp_path / "execution.py"
    executor.write_text("# reviewed\n", encoding="utf-8")
    cameras = {
        name: {
            "type": "opencv",
            "index_or_path": f"/dev/{name}",
            "width": 640,
            "height": 480,
            "fps": 30,
        }
        for name in ("front", "up")
    }
    robot = {
        "robot_port": "/dev/robot",
        "joint_limits": {
            joint: ([0.0, 100.0] if joint == "gripper" else [-100.0, 100.0])
            for joint in VIOLA_JOINTS
        },
        "max_step_deltas": {joint: 1.0 for joint in VIOLA_JOINTS},
        "speed_scale": 1.0,
    }
    value = {
        "schema_version": 1,
        "setup_id": "setup-one",
        **robot,
        "cameras": cameras,
        "calibration_path": str(calibration),
        "calibration_sha256": sha256_file(calibration),
        "reset_protocol_path": str(reset),
        "camera_config_sha256": sha256_json(cameras),
        "robot_config_sha256": sha256_json(robot),
        "reset_protocol_sha256": sha256_file(reset),
        "executor_attestation": {
            "reviewed": True,
            "clean_commit": True,
            "commit": "a" * 40,
            "reviewer": "reviewer",
            "reviewed_at": now,
            "repository": "starai-viola-lerobot-ops",
            "python_version": "3.12.13",
            "lerobot_version": "0.6.1",
            "conda_environment": "lerobot",
            "execution_backend": "direct_lerobot_fashionstar",
            "entrypoint_path": str(executor),
            "entrypoint_sha256": sha256_file(executor),
            "capabilities": list(EXECUTOR_CAPABILITIES),
        },
        "estop": {"tested_at": now, "operator": "operator", "passed": True},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _identity(*, role: str = "pc_a", clean: bool = True):
    return viola_handoff.RuntimeIdentity(
        role=role,
        repository_commit="a" * 40,
        repository_clean=clean,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    return root


def test_capture_reads_then_closes_before_wandb_publish(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    port = _ReadOnlyPort()
    events = []
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1), base + timedelta(seconds=1)))

    def publish(identity, **kwargs):
        assert port.closed is True
        events.append((identity, kwargs))
        return identity

    result = capture_frozen_state(
        setup,
        output_root=tmp_path / "capture",
        operator="operator",
        repo_root=_repo_root(tmp_path),
        wandb_entity="entity",
        confirm=lambda challenge: challenge == "READ FROZEN STATE setup-one",
        port_factory=lambda path, baud: port,
        publisher=publish,
        now=lambda: next(times),
    )

    assert port.opened and port.closed and port.monitor_calls == 1
    assert len(result.state) == 7
    assert result.sync_path.is_file()
    assert events[0][1]["config"]["operation"] == "frozen_state_capture"


@pytest.mark.parametrize("mutation", ["bytes", "permissions", "layout"])
def test_finished_publish_rejects_capture_material_changed_inside_publisher(
    tmp_path, monkeypatch, mutation: str
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1)))

    def publish_then_mutate(identity, **_kwargs):
        if mutation == "bytes":
            state_path = root / "frozen_state.json"
            state = json.loads(state_path.read_bytes())
            state["state"][0] = 0.5
            state_path.chmod(0o644)
            state_path.write_bytes(viola_handoff.canonical_json_bytes(state))
            state_path.chmod(0o444)
        elif mutation == "permissions":
            (root / "frozen_state_capture.json").chmod(0o644)
        else:
            (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
        return identity

    with pytest.raises(
        ValidationError,
        match="changed during W&B publication",
    ) as rejected:
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: True,
            port_factory=lambda _path, _baud: _ReadOnlyPort(),
            publisher=publish_then_mutate,
            now=lambda: next(times),
        )

    assert not (root / "frozen_state_capture_WANDB_SYNCED.json").exists()
    assert "finished run has no reusable local receipt" in " ".join(
        rejected.value.__notes__
    )


@pytest.mark.parametrize(
    "source",
    ["reviewed_setup", "calibration", "reset", "entrypoint"],
)
def test_finished_publish_rejects_reviewed_source_changed_inside_publisher(
    tmp_path, monkeypatch, source: str
) -> None:
    setup_path = _setup(tmp_path)
    setup = json.loads(setup_path.read_bytes())
    source_paths = {
        "reviewed_setup": setup_path,
        "calibration": Path(setup["calibration_path"]),
        "reset": Path(setup["reset_protocol_path"]),
        "entrypoint": Path(setup["executor_attestation"]["entrypoint_path"]),
    }
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1)))

    def publish_then_mutate(identity, **_kwargs):
        with source_paths[source].open("ab") as handle:
            handle.write(b"\n")
        return identity

    with pytest.raises(
        ValidationError,
        match="changed during W&B publication",
    ):
        capture_frozen_state(
            setup_path,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: True,
            port_factory=lambda _path, _baud: _ReadOnlyPort(),
            publisher=publish_then_mutate,
            now=lambda: next(times),
        )

    assert not (root / "frozen_state_capture_WANDB_SYNCED.json").exists()


def test_capture_refuses_without_operator_action_before_port_factory(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    calls = []
    with pytest.raises(SafetyGateError, match="explicit operator"):
        capture_frozen_state(
            setup,
            output_root=tmp_path / "capture",
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda challenge: False,
            port_factory=lambda path, baud: calls.append((path, baud)),
        )
    assert calls == []
    assert not (tmp_path / "capture").exists()


def test_cleanup_leaves_replacement_symlink_and_its_contents_untouched(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"
    moved_reservation = tmp_path / "moved-reservation"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    marker = unrelated / "frozen_state.json"
    marker.write_text("must remain\n", encoding="utf-8")

    def replace_before_refusal(_challenge: str) -> bool:
        root.rename(moved_reservation)
        root.symlink_to(unrelated, target_is_directory=True)
        return False

    with pytest.raises(SafetyGateError, match="explicit operator") as refused:
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=replace_before_refusal,
            port_factory=lambda _path, _baud: pytest.fail("port constructed"),
        )

    assert marker.read_text(encoding="utf-8") == "must remain\n"
    assert root.is_symlink()
    assert moved_reservation.is_dir()
    assert "replacement left untouched" in " ".join(refused.value.__notes__)


def test_capture_requires_external_evidence_before_confirmation_or_port(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    repository = _repo_root(tmp_path)
    calls = []

    with pytest.raises(ValidationError, match="outside the Repo-A worktree"):
        capture_frozen_state(
            setup,
            output_root=repository / "capture",
            operator="operator",
            repo_root=repository,
            wandb_entity="entity",
            confirm=lambda _challenge: calls.append("confirm") or True,
            port_factory=lambda _path, _baud: calls.append("port"),
        )

    assert calls == []


def _leave_publishable_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, datetime]:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1)))
    root = tmp_path / "capture"
    with pytest.raises(RuntimeError, match="W&B unavailable"):
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: True,
            port_factory=lambda _path, _baud: _ReadOnlyPort(),
            publisher=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("W&B unavailable")
            ),
            now=lambda: next(times),
        )
    assert {entry.name for entry in root.iterdir()} == {
        "frozen_state.json",
        "frozen_state_capture.json",
    }
    return setup, root, base


def test_upload_only_retries_same_run_without_confirmation_or_port(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1)))
    publish_calls = []

    def first_publish(identity, **kwargs):
        publish_calls.append((identity, kwargs))
        raise RuntimeError("W&B unavailable")

    with pytest.raises(RuntimeError) as failed:
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: True,
            port_factory=lambda _path, _baud: _ReadOnlyPort(),
            publisher=first_publish,
            now=lambda: next(times),
        )
    assert "retry with --upload-only" in " ".join(failed.value.__notes__)
    original = {
        name: (root / name).read_bytes()
        for name in ("frozen_state.json", "frozen_state_capture.json")
    }

    def recovered_publish(identity, **kwargs):
        publish_calls.append((identity, kwargs))
        return identity

    result = capture_frozen_state(
        setup,
        output_root=root,
        operator="operator",
        repo_root=_repo_root(tmp_path),
        wandb_entity="entity",
        upload_only=True,
        confirm=lambda _challenge: (_ for _ in ()).throw(AssertionError("confirmation called")),
        port_factory=lambda _path, _baud: (_ for _ in ()).throw(
            AssertionError("port constructed")
        ),
        publisher=recovered_publish,
        now=lambda: base + timedelta(seconds=1),
    )

    assert result.upload_only is True
    assert result.sync_path.is_file()
    assert publish_calls[0] == publish_calls[1]
    assert "not opened during upload-only recovery" in result.render_text()
    for name, payload in original.items():
        assert (root / name).read_bytes() == payload

    receipt = result.sync_path.read_bytes()
    second = capture_frozen_state(
        setup,
        output_root=root,
        operator="operator",
        repo_root=_repo_root(tmp_path),
        wandb_entity="entity",
        upload_only=True,
        publisher=lambda identity, **_kwargs: identity,
        now=lambda: (_ for _ in ()).throw(AssertionError("existing sync must be reused")),
    )
    assert second.sync_path.read_bytes() == receipt


@pytest.mark.parametrize("target_kind", ["directory", "file", "symlink"])
def test_fresh_capture_rejects_existing_target_before_confirmation_or_port(
    tmp_path, monkeypatch, target_kind: str
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"
    if target_kind == "directory":
        root.mkdir()
    elif target_kind == "file":
        root.write_text("occupied", encoding="utf-8")
    else:
        destination = tmp_path / "elsewhere"
        destination.mkdir()
        root.symlink_to(destination, target_is_directory=True)
    calls = []

    with pytest.raises(ValidationError, match="must not already exist"):
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: calls.append("confirm") or True,
            port_factory=lambda _path, _baud: calls.append("port"),
        )
    assert calls == []


def test_fresh_capture_rejects_non_directory_parent_before_confirmation_or_port(
    tmp_path, monkeypatch
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied", encoding="utf-8")
    calls = []
    with pytest.raises(ValidationError, match="parent"):
        capture_frozen_state(
            setup,
            output_root=parent / "capture",
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: calls.append("confirm") or True,
            port_factory=lambda _path, _baud: calls.append("port"),
        )
    assert calls == []


def test_fresh_capture_reserves_directory_before_constructing_port(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "nested" / "capture"
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1), base + timedelta(seconds=1)))

    def port_factory(_path, _baud):
        assert root.is_dir()
        return _ReadOnlyPort()

    capture_frozen_state(
        setup,
        output_root=root,
        operator="operator",
        repo_root=_repo_root(tmp_path),
        wandb_entity="entity",
        confirm=lambda _challenge: True,
        port_factory=port_factory,
        publisher=lambda identity, **_kwargs: identity,
        now=lambda: next(times),
    )


@pytest.mark.parametrize("damage", ["missing", "tampered", "writable", "extra", "symlink"])
def test_upload_only_rejects_partial_tampered_or_symlinked_material_without_publish(
    tmp_path, monkeypatch, damage: str
) -> None:
    setup, root, base = _leave_publishable_capture(tmp_path, monkeypatch)
    state_path = root / "frozen_state.json"
    if damage == "missing":
        (root / "frozen_state_capture.json").unlink()
    elif damage == "tampered":
        state = json.loads(state_path.read_bytes())
        state["state"][0] = 0.5
        state_path.chmod(0o644)
        state_path.write_bytes(viola_handoff.canonical_json_bytes(state))
        state_path.chmod(0o444)
    elif damage == "writable":
        state_path.chmod(0o644)
    elif damage == "extra":
        (root / "partial.tmp").write_text("partial", encoding="utf-8")
    else:
        saved = tmp_path / "saved-state.json"
        state_path.rename(saved)
        state_path.symlink_to(saved)
    publish_calls = []

    with pytest.raises(ValidationError):
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            upload_only=True,
            confirm=lambda _challenge: (_ for _ in ()).throw(
                AssertionError("confirmation called")
            ),
            port_factory=lambda _path, _baud: (_ for _ in ()).throw(
                AssertionError("port constructed")
            ),
            publisher=lambda identity, **kwargs: publish_calls.append((identity, kwargs)),
            now=lambda: base + timedelta(seconds=1),
        )
    assert publish_calls == []


def test_upload_only_rejects_tampered_optional_sync_before_publish(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"
    base = datetime.now(UTC)
    times = iter((base, base + timedelta(milliseconds=1), base + timedelta(seconds=1)))
    result = capture_frozen_state(
        setup,
        output_root=root,
        operator="operator",
        repo_root=_repo_root(tmp_path),
        wandb_entity="entity",
        confirm=lambda _challenge: True,
        port_factory=lambda _path, _baud: _ReadOnlyPort(),
        publisher=lambda identity, **_kwargs: identity,
        now=lambda: next(times),
    )
    sync = json.loads(result.sync_path.read_bytes())
    sync["binding"]["state_sha256"] = "0" * 64
    result.sync_path.chmod(0o644)
    result.sync_path.write_bytes(viola_handoff.canonical_json_bytes(sync))
    result.sync_path.chmod(0o444)
    publish_calls = []
    with pytest.raises(ValidationError, match="stale or mismatched"):
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            upload_only=True,
            publisher=lambda identity, **kwargs: publish_calls.append((identity, kwargs)),
        )
    assert publish_calls == []


@pytest.mark.parametrize(
    "identity",
    [_identity(role="pc_b"), _identity(clean=False)],
    ids=["wrong-role", "dirty-repository"],
)
def test_upload_only_requires_current_clean_pc_a_identity(
    tmp_path, monkeypatch, identity
) -> None:
    setup, root, _base = _leave_publishable_capture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: identity),
    )
    with pytest.raises(ValidationError, match="current clean pc_a identity"):
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            upload_only=True,
            publisher=lambda *_args, **_kwargs: pytest.fail("publisher called"),
        )


@pytest.mark.parametrize("signal", [KeyboardInterrupt("stop"), SystemExit(130)])
def test_capture_preserves_process_signal_when_serial_close_also_fails(
    tmp_path, monkeypatch, signal: BaseException
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"

    class SignalledPort(_ReadOnlyPort):
        def SyncServoMonitor(self, motors, realtime=False):
            raise signal

        def closePort(self):
            self.closed = True
            raise RuntimeError("close failed")

    with pytest.raises(type(signal)) as stopped:
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: True,
            port_factory=lambda _path, _baud: SignalledPort(),
            publisher=lambda *_args, **_kwargs: pytest.fail("publisher called"),
        )
    assert stopped.value is signal
    notes = " ".join(stopped.value.__notes__)
    assert "Serial-port cleanup also failed" in notes
    assert "Incomplete capture directory removed" in notes
    assert not root.exists()


def test_close_failure_after_read_prevents_publication_and_cleans_capture(tmp_path, monkeypatch) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: _identity()),
    )
    root = tmp_path / "capture"

    class CloseFailurePort(_ReadOnlyPort):
        def closePort(self):
            self.closed = True
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed") as failed:
        capture_frozen_state(
            setup,
            output_root=root,
            operator="operator",
            repo_root=_repo_root(tmp_path),
            wandb_entity="entity",
            confirm=lambda _challenge: True,
            port_factory=lambda _path, _baud: CloseFailurePort(),
            publisher=lambda *_args, **_kwargs: pytest.fail("publisher called"),
        )
    assert "Incomplete capture directory removed" in " ".join(failed.value.__notes__)
    assert not root.exists()
