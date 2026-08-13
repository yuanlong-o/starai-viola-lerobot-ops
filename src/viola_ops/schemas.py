"""Small typed values shared by the human-facing Repo-A commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIOLA_JOINTS = (
    "Motor_0",
    "Motor_1",
    "Motor_2",
    "Motor_3",
    "Motor_4",
    "Motor_5",
    "gripper",
)

VIOLA_CAMERAS = ("front", "up")

# This order is part of Repo B's accepted session-input contract.  Keep the
# tuple explicit: alphabetical or set-based rewrites would change the wire data.
EXECUTOR_CAPABILITIES = (
    "exact_camera_mapping",
    "fresh_position_write_limits",
    "keep_current_pose_startup",
    "no_automatic_reset_or_return",
    "operator_estop_ownership",
    "session_trial_recording",
    "ten_action_queue",
    "torque_retained_on_disconnect",
)


@dataclass(frozen=True, slots=True)
class ReviewedSetup:
    """A reviewed setup after all producer-side semantic checks pass."""

    source_path: Path
    setup_id: str
    robot_port: str
    cameras: dict[str, dict[str, Any]]
    joint_limits: dict[str, list[float]]
    max_step_deltas: dict[str, float]
    speed_scale: float
    calibration_path: Path
    reset_protocol_path: Path
    executor_entrypoint: Path
    executor: dict[str, Any]
    estop: dict[str, Any]


__all__ = [
    "EXECUTOR_CAPABILITIES",
    "ReviewedSetup",
    "VIOLA_CAMERAS",
    "VIOLA_JOINTS",
]
