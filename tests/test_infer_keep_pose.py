from scripts.infer_keep_pose import _bounded_goal, _extract_safety_args


FEATURES = {f"Motor_{index}.pos": object() for index in range(6)} | {"gripper.pos": object()}


def test_extract_safety_args_removes_wrapper_option() -> None:
    max_step, remaining = _extract_safety_args(["--max_step=2.5", "--duration=10"])
    assert max_step == 2.5
    assert remaining == ["--duration=10"]


def test_bounded_goal_clips_absolute_and_relative_targets() -> None:
    present = {f"Motor_{index}": 0.0 for index in range(6)} | {"gripper": 98.0}
    action = {f"Motor_{index}.pos": 200.0 if index % 2 == 0 else -200.0 for index in range(6)}
    action["gripper.pos"] = 200.0
    goal = _bounded_goal(action, present, FEATURES, max_step=3.0)
    assert goal == {
        "Motor_0": 3.0,
        "Motor_1": -3.0,
        "Motor_2": 3.0,
        "Motor_3": -3.0,
        "Motor_4": 3.0,
        "Motor_5": -3.0,
        "gripper": 100.0,
    }


def test_bounded_goal_rejects_missing_joint() -> None:
    present = {f"Motor_{index}": 0.0 for index in range(6)} | {"gripper": 0.0}
    action = {key: 0.0 for key in FEATURES}
    del action["Motor_5.pos"]
    try:
        _bounded_goal(action, present, FEATURES, max_step=3.0)
    except KeyError as error:
        assert "Motor_5.pos" in str(error)
    else:
        raise AssertionError("missing policy joint was accepted")
