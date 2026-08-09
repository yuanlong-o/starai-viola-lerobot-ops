#!/usr/bin/env python3
"""Teleoperate StarAI Viola from its current pose instead of a hard-coded startup pose.

The installed StarAI Viola and Violin plugins both call
``move_to_initial_position()`` while connecting.  Their implementation commands a
fixed joint configuration, which can make either arm jump at startup.  This
wrapper replaces those startup methods before constructing the devices.

It also anchors teleoperation at the two measured startup poses:

    follower target = follower startup + (leader current - leader startup)

As a result, the first command sent to the follower is its measured current pose,
even when the leader and follower are not physically aligned at launch.
"""

import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import rerun as rr

from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class KeepPoseTeleoperateConfig(TeleoperateConfig):
    """Standard teleoperation options plus a per-control-step safety limit."""

    max_step: float = 3.0


def _hold_follower_at_current_pose(self: Any) -> dict[str, float]:
    """Replace the plugin's hard-coded follower startup movement."""

    current_action = self.get_action()
    current_goal = {
        key.removesuffix(".pos"): value
        for key, value in current_action.items()
        if key.endswith(".pos")
    }
    # A goal equal to the measured position engages control without requesting a
    # move to the plugin's predefined pose.
    self.bus.sync_write("Goal_Position", current_goal, motion_time=100)
    return current_action


def _leave_leader_at_current_pose(self: Any) -> dict[str, float]:
    """Replace the plugin's hard-coded leader startup movement."""

    # StaraiMotorsBus.connect() has already put the leader in unlocked mode.
    # Reading only preserves that state and avoids commanding a new position.
    return self.get_action()


def install_keep_pose_startup_patch() -> None:
    """Install the startup patch on the two third-party StarAI classes."""

    from lerobot_robot_viola.starai_viola import StaraiViola
    from lerobot_teleoperator_violin.starai_violin import StaraiViolin

    StaraiViola.move_to_initial_position = _hold_follower_at_current_pose
    StaraiViolin.move_to_initial_position = _leave_leader_at_current_pose


def _bounded_relative_action(
    leader_action: dict[str, float],
    leader_start: dict[str, float],
    follower_start: dict[str, float],
    previous_action: dict[str, float],
    max_step: float,
) -> dict[str, float]:
    """Map leader displacement onto the follower startup pose with safe bounds."""

    bounded_action: dict[str, float] = {}
    for key, follower_start_value in follower_start.items():
        if key not in leader_action or key not in leader_start:
            raise KeyError(f"Leader action is missing required joint {key!r}")

        desired = follower_start_value + (leader_action[key] - leader_start[key])
        lower_limit, upper_limit = (0.0, 100.0) if key == "gripper.pos" else (-100.0, 100.0)
        desired = min(upper_limit, max(lower_limit, desired))

        previous = previous_action[key]
        bounded_action[key] = min(previous + max_step, max(previous - max_step, desired))

    return bounded_action


@parser.wrap()
def teleoperate_keep_pose(cfg: KeepPoseTeleoperateConfig) -> None:
    if cfg.max_step <= 0:
        raise ValueError("--max_step must be positive")

    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        # `sg dialout` starts a clean shell whose PATH does not include the
        # active Conda environment. Rerun's SDK discovers its viewer via PATH,
        # so explicitly expose the bin directory belonging to this Python.
        python_bin = str(Path(sys.executable).resolve().parent)
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
        init_rerun(session_name="keep-pose teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop_connected = False
    robot_connected = False
    try:
        teleop.connect()
        teleop_connected = True
        robot.connect()
        robot_connected = True

        leader_start = teleop.get_action()
        initial_observation = robot.get_observation()
        follower_start = {key: float(initial_observation[key]) for key in robot.action_features}
        previous_action = follower_start.copy()

        logging.info("Follower startup pose captured; relative teleoperation is active.")
        display_len = max(len(key) for key in robot.action_features)
        started_at = time.perf_counter()

        while True:
            loop_started_at = time.perf_counter()
            observation = robot.get_observation()
            leader_action = teleop.get_action()
            relative_action = _bounded_relative_action(
                leader_action=leader_action,
                leader_start=leader_start,
                follower_start=follower_start,
                previous_action=previous_action,
                max_step=cfg.max_step,
            )

            teleop_action = teleop_action_processor((relative_action, observation))
            robot_action = robot_action_processor((teleop_action, observation))
            sent_action = robot.send_action(robot_action)
            previous_action = {key: float(sent_action[key]) for key in follower_start}

            if cfg.display_data:
                observation_transition = robot_observation_processor(observation)
                log_rerun_data(observation=observation_transition, action=teleop_action)
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                for motor, value in sent_action.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(sent_action) + 5)

            elapsed = time.perf_counter() - loop_started_at
            precise_sleep(max(0.0, 1 / cfg.fps - elapsed))

            if cfg.teleop_time_s is not None and time.perf_counter() - started_at >= cfg.teleop_time_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        if robot_connected:
            robot.disconnect()
        if teleop_connected:
            teleop.disconnect()


def main() -> None:
    register_third_party_plugins()
    install_keep_pose_startup_patch()
    teleoperate_keep_pose()


if __name__ == "__main__":
    main()
