#!/usr/bin/env python3
"""Run LeRobot policy inference with keep-pose startup and bounded motor writes."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any


def _extract_safety_args(argv: list[str]) -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_step", type=float, default=3.0)
    known, remaining = parser.parse_known_args(argv)
    if known.max_step <= 0:
        raise ValueError("--max_step must be positive")
    return known.max_step, remaining


def _bounded_goal(
    action: dict[str, Any],
    present: dict[str, Any],
    action_features: dict[str, Any],
    max_step: float,
) -> dict[str, float]:
    bounded: dict[str, float] = {}
    for action_key in action_features:
        motor = action_key.removesuffix(".pos")
        if action_key not in action:
            raise KeyError(f"Policy action is missing required joint {action_key!r}")
        if motor not in present:
            raise KeyError(f"Present-position read is missing motor {motor!r}")
        desired = float(action[action_key])
        lower, upper = (0.0, 100.0) if motor == "gripper" else (-100.0, 100.0)
        desired = min(upper, max(lower, desired))
        current = float(present[motor])
        bounded[motor] = min(current + max_step, max(current - max_step, desired))
    return bounded


def install_inference_safety_patch(max_step: float) -> None:
    from lerobot.utils.errors import DeviceNotConnectedError
    from lerobot_robot_viola.starai_viola import StaraiViola

    logger = logging.getLogger("viola_inference_safety")
    write_count = 0

    def hold_at_current_pose(self: Any) -> dict[str, float]:
        current = self.get_action()
        goal = {
            key.removesuffix(".pos"): float(value)
            for key, value in current.items()
            if key.endswith(".pos")
        }
        self.bus.sync_write("Goal_Position", goal, motion_time=100)
        return current

    def send_bounded_action(self: Any, action: dict[str, Any]) -> dict[str, float]:
        nonlocal write_count
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        present = self.bus.sync_read("Present_Position")
        bounded = _bounded_goal(action, present, self.action_features, max_step)
        self.bus.sync_write("Goal_Position", bounded)
        write_count += 1
        if write_count == 1:
            requested = {key: round(float(action[key]), 4) for key in self.action_features}
            sent = {f"{motor}.pos": round(value, 4) for motor, value in bounded.items()}
            logger.info("First policy target: %s", requested)
            logger.info("First bounded motor command (max_step=%s): %s", max_step, sent)
        elif write_count % 25 == 0:
            logger.info("Bounded policy motor writes completed: %d", write_count)
        return {f"{motor}.pos": value for motor, value in bounded.items()}

    StaraiViola.move_to_initial_position = hold_at_current_pose
    StaraiViola.send_action = send_bounded_action


def main() -> None:
    max_step, rollout_args = _extract_safety_args(sys.argv[1:])
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

    from lerobot.scripts.lerobot_rollout import main as rollout_main
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    install_inference_safety_patch(max_step)
    sys.argv = [sys.argv[0], *rollout_args]
    rollout_main()


if __name__ == "__main__":
    main()
