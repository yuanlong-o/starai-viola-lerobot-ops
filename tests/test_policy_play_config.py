from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/policy_play.yaml"
CHECKPOINT_PATH = Path("/home/yz/models/act_viola_val20_step080000")


def test_standard_lerobot_policy_play_config_parses_without_hardware() -> None:
    if not CHECKPOINT_PATH.is_dir():
        pytest.skip("PC-A direct-play ACT checkpoint is not installed")

    parse_only = r"""
import json

from lerobot.configs import parser
from lerobot.rollout import RolloutConfig
import lerobot.scripts.lerobot_rollout  # Registers built-in robot/camera configs.
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

@parser.wrap()
def inspect_config(cfg: RolloutConfig) -> None:
    print(json.dumps({
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "policy_actions": cfg.policy.n_action_steps,
        "robot_type": cfg.robot.type,
        "robot_id": cfg.robot.id,
        "robot_port": cfg.robot.port,
        "calibration_dir": str(cfg.robot.calibration_dir),
        "disable_torque_on_disconnect": cfg.robot.disable_torque_on_disconnect,
        "use_degrees": cfg.robot.use_degrees,
        "front": {
            "path": str(cfg.robot.cameras["front"].index_or_path),
            "fourcc": cfg.robot.cameras["front"].fourcc,
            "backend": cfg.robot.cameras["front"].backend.value,
            "warmup_s": cfg.robot.cameras["front"].warmup_s,
            "width": cfg.robot.cameras["front"].width,
            "height": cfg.robot.cameras["front"].height,
            "fps": cfg.robot.cameras["front"].fps,
        },
        "up": {
            "path": str(cfg.robot.cameras["up"].index_or_path),
            "fourcc": cfg.robot.cameras["up"].fourcc,
            "backend": cfg.robot.cameras["up"].backend.value,
            "warmup_s": cfg.robot.cameras["up"].warmup_s,
            "width": cfg.robot.cameras["up"].width,
            "height": cfg.robot.cameras["up"].height,
            "fps": cfg.robot.cameras["up"].fps,
        },
        "strategy": cfg.strategy.type,
        "inference": cfg.inference.type,
        "fps": cfg.fps,
        "duration": cfg.duration,
        "device": cfg.device,
        "task": cfg.task,
        "display_data": cfg.display_data,
        "play_sounds": cfg.play_sounds,
        "return_to_initial_position": cfg.return_to_initial_position,
    }, sort_keys=True))

inspect_config()
"""
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", parse_only, "--config_path", str(CONFIG_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    parsed = json.loads(completed.stdout.strip().splitlines()[-1])

    assert parsed == {
        "calibration_dir": str(REPO_ROOT / "calibration/robots/starai_viola"),
        "device": "cuda:0",
        "disable_torque_on_disconnect": True,
        "display_data": False,
        "duration": 10.0,
        "fps": 30.0,
        "front": {
            "backend": 200,
            "fps": 30,
            "fourcc": "YUYV",
            "height": 480,
            "path": "/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0",
            "warmup_s": 8,
            "width": 640,
        },
        "inference": "sync",
        "play_sounds": False,
        "policy_actions": 100,
        "policy_path": "/home/yz/models/act_viola_val20_step080000",
        "policy_type": "act",
        "return_to_initial_position": False,
        "robot_id": "my_awesome_staraiviola_arm",
        "robot_port": "/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0",
        "robot_type": "lerobot_robot_viola",
        "strategy": "base",
        "task": (
            "Move the blue cube, then the red cube, from the white pad on the right "
            "to the gray platform on the left."
        ),
        "up": {
            "backend": 200,
            "fps": 30,
            "fourcc": "YUYV",
            "height": 480,
            "path": "/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0",
            "warmup_s": 8,
            "width": 640,
        },
        "use_degrees": False,
    }
