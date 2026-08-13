"""Reviewed setup operations that never command a motor."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
from .jsonutil import sha256_file, sha256_json, write_canonical_json
from .session_inputs import load_reviewed_setup
from .wandb_ops import WandbRunIdentity, planned_run, publish_finished_run


@dataclass(frozen=True, slots=True)
class FrozenStateCapture:
    output_root: Path
    state_path: Path
    evidence_path: Path
    sync_path: Path
    state: tuple[float, ...]
    wandb: WandbRunIdentity

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

        return "\n".join(
            [
                "Frozen state captured",
                f"  Seven-axis state: {self.state_path}",
                f"  Capture evidence: {self.evidence_path}",
                f"  W&B receipt: {self.sync_path}",
                f"  W&B run: {self.wandb.url}",
                "  Motor commands sent: none",
                "  Serial connection: closed before evidence publication",
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
) -> FrozenStateCapture:
    """Read seven positions through the public SDK, close, then publish evidence.

    No position-control or torque API is reachable from this function.  A
    human confirmation is nevertheless mandatory before the serial port is
    opened, and W&B work starts only after that port is closed.
    """

    setup = load_reviewed_setup(setup_path)
    if not operator.strip() or operator != setup.estop["operator"]:
        raise ValidationError("capture operator must equal the reviewed E-stop operator")
    challenge = f"READ FROZEN STATE {setup.setup_id}"
    if confirm is None or confirm(challenge) is not True:
        raise SafetyGateError("frozen-state capture requires explicit operator confirmation")

    identity = RuntimeIdentity.capture(role="pc_a", repo_root=repo_root)
    if identity.repository_commit != setup.executor["commit"]:
        raise ValidationError("current clean revision differs from the reviewed executor")
    calibration = load_calibration(setup.calibration_path)
    factory = port_factory or _fashionstar_port
    port = factory(setup.robot_port, 1_000_000)
    opened = False
    captured_at: datetime | None = None
    values: tuple[float, ...] | None = None
    try:
        if port.openPort() is False:
            raise ValidationError("FashionStar serial port did not open")
        opened = True
        for joint in JOINTS:
            if not port.ping(calibration[joint].id):
                raise ValidationError(f"servo {calibration[joint].id} ({joint}) did not respond")
        ids = {joint: calibration[joint].id for joint in JOINTS}
        monitors = port.SyncServoMonitor(ids, realtime=True)
        if set(monitors) != set(JOINTS):
            raise ValidationError("synchronous monitor did not return all seven joints")
        state: list[float] = []
        for joint in JOINTS:
            degrees = _finite(monitors[joint].current_position, f"{joint} position")
            state.append(
                raw_to_normalized(
                    degrees_to_raw(degrees),
                    calibration[joint],
                    gripper=joint == "gripper",
                )
            )
        values = tuple(state)
        captured_at = _utc(now())
    finally:
        if opened:
            port.closePort()
    disconnected_at = _utc(now())
    if values is None or captured_at is None:
        raise ValidationError("frozen-state capture did not produce a complete sample")
    if disconnected_at <= captured_at:
        raise ValidationError("disconnect timestamp must follow frozen-state capture")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "schema_version": 1,
        "state": list(values),
        "captured_at": captured_at.isoformat(),
        "robot_connected_at_capture": True,
        "motor_disconnected_at": disconnected_at.isoformat(),
    }
    state_path = write_canonical_json(root / "frozen_state.json", state_payload)
    setup_hashes = _setup_hashes(setup)
    state_sha256 = sha256_json(list(values))
    run_seed = sha256_json(
        {
            "setup_id": setup.setup_id,
            "state_sha256": state_sha256,
            "captured_at": captured_at.isoformat(),
            "repo": identity.repository_commit,
        }
    )
    wandb = planned_run(wandb_entity, wandb_project, f"capture-{run_seed[:16]}")
    repo = {
        "commit": identity.repository_commit,
        "clean": identity.repository_clean,
        "hostname": identity.hostname,
        "python": identity.python_version,
    }
    evidence = {
        "schema_version": 1,
        "kind": "frozen_state_capture",
        "status": "captured_disconnected",
        "setup_id": setup.setup_id,
        "frozen_state_file": "frozen_state.json",
        "frozen_state_sha256": sha256_file(state_path),
        "state_sha256": state_sha256,
        "setup_hashes": setup_hashes,
        "operator": operator,
        "captured_at": captured_at.isoformat(),
        "motor_disconnected_at": disconnected_at.isoformat(),
        "repo": repo,
        "wandb": wandb.binding(),
    }
    evidence_path = write_canonical_json(root / "frozen_state_capture.json", evidence)
    evidence_sha256 = sha256_file(evidence_path)
    publisher(
        wandb,
        job_type="viola-frozen-state-capture",
        config={
            "operation": "frozen_state_capture",
            "evidence_sha256": evidence_sha256,
            "setup_id": setup.setup_id,
            "status": evidence["status"],
            "repo_commit": identity.repository_commit,
            "setup_hashes": setup_hashes,
            "frozen_state_sha256": evidence["frozen_state_sha256"],
            "state_sha256": state_sha256,
        },
        summary={"state_dimensions": 7, "serial_closed": True},
    )
    sync = {
        "schema_version": 1,
        "operation": "frozen_state_capture",
        "evidence_file": "frozen_state_capture.json",
        "evidence_sha256": evidence_sha256,
        "wandb": wandb.binding(),
        "binding": {
            "setup_id": setup.setup_id,
            "state_sha256": state_sha256,
            "status": "captured_disconnected",
            "repo_commit": identity.repository_commit,
            "setup_hashes": setup_hashes,
            "frozen_state_sha256": evidence["frozen_state_sha256"],
        },
        "synced_at": _utc(now()).isoformat(),
    }
    sync_path = write_canonical_json(root / "frozen_state_capture_WANDB_SYNCED.json", sync)
    return FrozenStateCapture(root, state_path, evidence_path, sync_path, values, wandb)


def interactive_confirmation(challenge: str) -> bool:
    """Read the exact capture challenge from a real terminal."""

    try:
        with open("/dev/tty", "r+", encoding="utf-8", buffering=1) as terminal:
            if not terminal.isatty():
                return False
            terminal.write(f"This reads positions only and sends no motor command.\nType exactly: {challenge}\n> ")
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
