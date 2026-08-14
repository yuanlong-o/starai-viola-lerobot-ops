from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import viola_ops.safety as safety
from viola_handoff import RuntimeIdentity, inventory_root
from viola_ops.errors import SafetyGateError
from viola_ops.jsonutil import sha256_file, sha256_json
from viola_ops.safety import (
    JOINTS,
    LocalActCheckpointBinding,
    LocalActGateRequest,
    ResolvedSetup,
    authorize_local_act_motion,
    begin_permit_execution,
    finish_permit_execution,
    local_act_operator_challenge,
    revalidate_local_act_motion,
)


class _ChallengeStream:
    def __init__(self, response: str | None = None) -> None:
        self.response = response
        self.output = ""

    def write(self, value: str) -> int:
        self.output += value
        return len(value)

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        if self.response is not None:
            return self.response
        challenge = self.output.split("Type exactly: ", 1)[1].split("\n", 1)[0]
        return f"{challenge}\n"


def test_operator_prompt_terminal_supports_a_real_nonseekable_pty() -> None:
    master_fd, slave_fd = os.openpty()
    terminal = None
    try:
        terminal = safety._OperatorPromptTerminal(os.ttyname(slave_fd))
        assert terminal.isatty()

        assert terminal.write("prompt") == len("prompt")
        terminal.flush()
        assert os.read(master_fd, len("prompt")) == b"prompt"

        os.write(master_fd, b"response\n")
        assert terminal.readline() == "response\n"
    finally:
        if terminal is not None:
            terminal.close()
        os.close(slave_fd)
        os.close(master_fd)


def _identity(*, commit: str = "a" * 40, clean: bool = True) -> RuntimeIdentity:
    return RuntimeIdentity(
        role="pc_a",
        repository_commit=commit,
        repository_clean=clean,
        hostname="pc-a",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _setup(tmp_path: Path, identity: RuntimeIdentity) -> tuple[ResolvedSetup, Path]:
    repo = tmp_path / "repo"
    current_entrypoint = repo / "src/viola_ops/execution.py"
    current_entrypoint.parent.mkdir(parents=True)
    current_entrypoint.write_text("# reviewed local executor\n", encoding="utf-8")

    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()
    calibration = reviewed / "calibration.json"
    reset = reviewed / "reset_protocol.json"
    entrypoint = reviewed / "executor_entrypoint.py"
    calibration.write_text('{"calibration":"local-act"}\n', encoding="utf-8")
    reset.write_text('{"reset":"manual"}\n', encoding="utf-8")
    entrypoint.write_bytes(current_entrypoint.read_bytes())

    cameras = {
        "front": {
            "type": "opencv",
            "index_or_path": "/dev/v4l/by-id/front-video-index0",
            "width": 640,
            "height": 480,
            "fps": 30,
            "fourcc": "MJPG",
            "warmup_s": 8,
        },
        "up": {
            "type": "opencv",
            "index_or_path": "/dev/v4l/by-id/up-video-index0",
            "width": 640,
            "height": 480,
            "fps": 30,
            "fourcc": "YUYV",
            "warmup_s": 8,
        },
    }
    limits = {
        joint: ((0.0, 100.0) if joint == "gripper" else (-100.0, 100.0))
        for joint in JOINTS
    }
    deltas = {joint: 3.0 for joint in JOINTS}
    setup_hashes = {
        "calibration": sha256_file(calibration),
        "camera": sha256_json(cameras),
        "robot": sha256_json(
            {
                "robot_port": "/dev/serial/by-path/local-act-robot",
                "joint_limits": {
                    joint: list(pair) for joint, pair in limits.items()
                },
                "max_step_deltas": deltas,
                "speed_scale": 1.0,
            }
        ),
        "reset": sha256_file(reset),
    }
    return (
        ResolvedSetup(
            setup_id="local-act-setup",
            robot_port="/dev/serial/by-path/local-act-robot",
            camera_configs=cameras,
            calibration_path=calibration,
            reset_protocol_path=reset,
            executor_entrypoint=entrypoint,
            setup_hashes=setup_hashes,
            absolute_limits=limits,
            max_step_deltas=deltas,
            executor_identity={
                "repository_commit": identity.repository_commit,
                "python_version": identity.python_version,
                "lerobot_version": identity.lerobot_version,
                "conda_environment": identity.conda_environment,
            },
        ),
        repo,
    )


def _checkpoint(tmp_path: Path) -> LocalActCheckpointBinding:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"type": "act", "chunk_size": 100, "n_action_steps": 100}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"model")
    (root / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (root / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    return LocalActCheckpointBinding(
        root=root,
        inventory=inventory_root(root),
        model_sha256=sha256_file(root / "model.safetensors"),
        dataset_release_id="local-act-dataset-v1",
        dataset_inventory_sha256="d" * 64,
        dataset_metadata_inventory_sha256="f" * 64,
    )


def _request(
    tmp_path: Path,
    *,
    now: datetime | None = None,
    passed: bool = True,
) -> tuple[LocalActGateRequest, RuntimeIdentity]:
    current = now or datetime.now(UTC)
    identity = _identity()
    setup, repository = _setup(tmp_path, identity)
    return (
        LocalActGateRequest(
            setup=setup,
            checkpoint=_checkpoint(tmp_path),
            operator="operator",
            estop_tested_at=current - timedelta(minutes=5),
            estop_attestation_sha256="e" * 64,
            estop_passed=passed,
            trial="act-smoke-01",
            duration_s=10.0,
            repository_root=repository,
            now=current,
        ),
        identity,
    )


@pytest.fixture(autouse=True)
def _clean_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "check_current_checkout", lambda *_args, **_kwargs: None)


def test_local_act_gate_issues_exact_commissioning_permit(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    stream = _ChallengeStream()

    permit = authorize_local_act_motion(
        request,
        identity=identity,
        input_stream=stream,
        terminal_check=lambda _stream: True,
    )

    assert permit.policy == "act"
    assert permit.phase == "local_act"
    assert permit.trial == "act-smoke-01"
    assert permit.speed_scale == 0.25
    assert permit.session_id.startswith("local-act-")
    assert len(permit.session_bundle_id) == 64
    assert len(permit.candidate_bundle_id) == 64
    assert permit.candidate_bundle_id == sha256_json(
        {
            "policy": "act",
            "model_sha256": request.checkpoint.model_sha256,
            "checkpoint_inventory_sha256": request.checkpoint.inventory[
                "inventory_sha256"
            ],
            "dataset_release_id": request.checkpoint.dataset_release_id,
            "dataset_inventory_sha256": request.checkpoint.dataset_inventory_sha256,
            "dataset_metadata_inventory_sha256": (
                request.checkpoint.dataset_metadata_inventory_sha256
            ),
            "deployment_queue_actions": 10,
        }
    )
    assert "ESTOP TESTED" in stream.output
    assert permit.allows(
        session_id=permit.session_id,
        phase="local_act",
        trial="act-smoke-01",
    )


def test_local_act_gate_requires_exact_real_tty_phrase(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    wrong = _ChallengeStream("ARM something else\n")
    with pytest.raises(SafetyGateError, match="did not match"):
        authorize_local_act_motion(
            request,
            identity=identity,
            input_stream=wrong,
            terminal_check=lambda _stream: True,
        )

    with pytest.raises(SafetyGateError, match="interactive terminal"):
        authorize_local_act_motion(
            request,
            identity=identity,
            input_stream=_ChallengeStream(),
            terminal_check=lambda _stream: False,
        )


def test_local_act_challenge_explicitly_attests_estop_and_duration() -> None:
    assert local_act_operator_challenge("local-act-session", "trial-1", 10.0) == (
        "ARM local-act-session local_act trial-1 10s ESTOP TESTED"
    )


def test_local_act_gate_requires_historical_camera_format_fields(
    tmp_path: Path,
) -> None:
    request, identity = _request(tmp_path)
    cameras = {
        name: {
            key: value
            for key, value in camera.items()
            if key not in {"fourcc", "warmup_s"}
        }
        for name, camera in request.setup.camera_configs.items()
    }
    broken = replace(request, setup=replace(request.setup, camera_configs=cameras))

    with pytest.raises(SafetyGateError, match="camera fields differ"):
        authorize_local_act_motion(
            broken,
            identity=identity,
            input_stream=_ChallengeStream(),
            terminal_check=lambda _stream: True,
        )


@pytest.mark.parametrize(
    ("age", "passed", "message"),
    [
        (timedelta(hours=24), True, "24 hours old or older"),
        (timedelta(minutes=1), False, "passed physical E-stop test"),
        (timedelta(minutes=-1), True, "future-dated"),
    ],
)
def test_local_act_gate_requires_current_passed_estop(
    tmp_path: Path,
    age: timedelta,
    passed: bool,
    message: str,
) -> None:
    now = datetime.now(UTC)
    request, identity = _request(tmp_path, now=now, passed=passed)
    object.__setattr__(request, "estop_tested_at", now - age)

    with pytest.raises(SafetyGateError, match=message):
        authorize_local_act_motion(
            request,
            identity=identity,
            input_stream=_ChallengeStream(),
            terminal_check=lambda _stream: True,
        )


def test_local_act_revalidation_needs_no_second_prompt(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    permit = authorize_local_act_motion(
        request,
        identity=identity,
        input_stream=_ChallengeStream(),
        terminal_check=lambda _stream: True,
    )

    session_id, candidate_id = revalidate_local_act_motion(
        request,
        permit,
        identity=identity,
    )

    assert session_id == permit.session_bundle_id
    assert candidate_id == permit.candidate_bundle_id


def test_post_execution_revalidation_requires_consumed_permit(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    permit = authorize_local_act_motion(
        request,
        identity=identity,
        input_stream=_ChallengeStream(),
        terminal_check=lambda _stream: True,
    )

    with pytest.raises(SafetyGateError, match="has not completed permit teardown"):
        revalidate_local_act_motion(
            request,
            permit,
            identity=identity,
            allow_consumed=True,
        )

    begin_permit_execution(permit)
    finish_permit_execution(permit)
    assert revalidate_local_act_motion(
        request,
        permit,
        identity=identity,
        allow_consumed=True,
    ) == (permit.session_bundle_id, permit.candidate_bundle_id)


def test_local_act_revalidation_rejects_checkpoint_change(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    permit = authorize_local_act_motion(
        request,
        identity=identity,
        input_stream=_ChallengeStream(),
        terminal_check=lambda _stream: True,
    )
    (request.checkpoint.root / "model.safetensors").write_bytes(b"changed model")

    with pytest.raises(SafetyGateError, match="differs from its approved inventory"):
        revalidate_local_act_motion(request, permit, identity=identity)


def test_local_act_revalidation_rejects_setup_change(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    permit = authorize_local_act_motion(
        request,
        identity=identity,
        input_stream=_ChallengeStream(),
        terminal_check=lambda _stream: True,
    )
    request.setup.calibration_path.write_text("changed", encoding="utf-8")

    with pytest.raises(SafetyGateError, match="setup hashes do not reproduce"):
        revalidate_local_act_motion(request, permit, identity=identity)


def test_local_act_gate_rejects_unreviewed_runtime(tmp_path: Path) -> None:
    request, _identity_value = _request(tmp_path)
    dirty = _identity(clean=False)

    with pytest.raises(SafetyGateError, match="clean pc_a identity"):
        authorize_local_act_motion(
            request,
            identity=dirty,
            input_stream=_ChallengeStream(),
            terminal_check=lambda _stream: True,
        )


def test_local_act_gate_rejects_non_act_or_non_ten_action_binding(tmp_path: Path) -> None:
    request, identity = _request(tmp_path)
    config = request.checkpoint.root / "config.json"
    config.write_text(
        json.dumps({"type": "diffusion", "chunk_size": 100, "n_action_steps": 100}),
        encoding="utf-8",
    )
    object.__setattr__(
        request,
        "checkpoint",
        LocalActCheckpointBinding(
            root=request.checkpoint.root,
            inventory=inventory_root(request.checkpoint.root),
            model_sha256=request.checkpoint.model_sha256,
            dataset_release_id=request.checkpoint.dataset_release_id,
            dataset_inventory_sha256=request.checkpoint.dataset_inventory_sha256,
            dataset_metadata_inventory_sha256=(
                request.checkpoint.dataset_metadata_inventory_sha256
            ),
            deployment_action_steps=10,
        ),
    )
    with pytest.raises(SafetyGateError, match="policy type is not ACT"):
        authorize_local_act_motion(
            request,
            identity=identity,
            input_stream=_ChallengeStream(),
            terminal_check=lambda _stream: True,
        )

    object.__setattr__(
        request,
        "checkpoint",
        LocalActCheckpointBinding(
            root=request.checkpoint.root,
            inventory=inventory_root(request.checkpoint.root),
            model_sha256=request.checkpoint.model_sha256,
            dataset_release_id=request.checkpoint.dataset_release_id,
            dataset_inventory_sha256=request.checkpoint.dataset_inventory_sha256,
            dataset_metadata_inventory_sha256=(
                request.checkpoint.dataset_metadata_inventory_sha256
            ),
            deployment_action_steps=100,
        ),
    )
    with pytest.raises(SafetyGateError, match="exactly ten queued actions"):
        authorize_local_act_motion(
            request,
            identity=identity,
            input_stream=_ChallengeStream(),
            terminal_check=lambda _stream: True,
        )
