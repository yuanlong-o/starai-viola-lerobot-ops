from __future__ import annotations

import pytest

from scripts.teleoperate_keep_pose import _bounded_relative_action


def test_no_teacher_displacement_keeps_follower_startup_pose() -> None:
    follower_start = {"Motor_0.pos": 17.0, "gripper.pos": 42.0}

    result = _bounded_relative_action(
        leader_action={"Motor_0.pos": -8.0, "gripper.pos": 20.0},
        leader_start={"Motor_0.pos": -8.0, "gripper.pos": 20.0},
        follower_start=follower_start,
        previous_action=follower_start,
        max_step=3.0,
    )

    assert result == follower_start


def test_relative_target_is_limited_per_control_step() -> None:
    result = _bounded_relative_action(
        leader_action={"Motor_0.pos": 60.0},
        leader_start={"Motor_0.pos": 0.0},
        follower_start={"Motor_0.pos": 10.0},
        previous_action={"Motor_0.pos": 10.0},
        max_step=3.0,
    )

    assert result == {"Motor_0.pos": 13.0}


def test_joint_and_gripper_targets_use_different_normalized_bounds() -> None:
    result = _bounded_relative_action(
        leader_action={"Motor_0.pos": 500.0, "gripper.pos": -500.0},
        leader_start={"Motor_0.pos": 0.0, "gripper.pos": 0.0},
        follower_start={"Motor_0.pos": 95.0, "gripper.pos": 5.0},
        previous_action={"Motor_0.pos": 99.0, "gripper.pos": 1.0},
        max_step=10.0,
    )

    assert result == {"Motor_0.pos": 100.0, "gripper.pos": 0.0}


def test_missing_teacher_joint_is_rejected() -> None:
    with pytest.raises(KeyError, match="missing required joint"):
        _bounded_relative_action(
            leader_action={},
            leader_start={"Motor_0.pos": 0.0},
            follower_start={"Motor_0.pos": 0.0},
            previous_action={"Motor_0.pos": 0.0},
            max_step=3.0,
        )
