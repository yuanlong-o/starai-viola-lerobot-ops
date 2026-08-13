from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENTRYPOINTS = (
    Path("scripts/infer_keep_pose.py"),
    Path("scripts/teleoperate_keep_pose.py"),
    Path("scripts/record_episodes_keep_pose.py"),
)
SHELL_ENTRYPOINTS = (
    Path("scripts/run_viola_inference.sh"),
    Path("scripts/run_viola_calibration.sh"),
    Path("scripts/run_viola_teleoperation.sh"),
    Path("scripts/run_viola_episode_recording.sh"),
    Path("archive/record_right_to_left_episode.sh"),
    Path("ros2/run_viola_driver.sh"),
    Path("ros2/setup_starai.sh"),
    Path("scripts/sync_policy.sh"),
)


def poisoned_environment(tmp_path: Path) -> dict[str, str]:
    """Make any accidental device lookup point at a path that cannot exist."""

    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path),
            "PATH": "/path-that-does-not-exist",
            "VIOLA_ROBOT_PORT": str(tmp_path / "must-not-be-opened-robot"),
            "VIOLA_TELEOP_PORT": str(tmp_path / "must-not-be-opened-teleop"),
            "VIOLA_FRONT_CAMERA": str(tmp_path / "must-not-be-opened-front"),
            "VIOLA_UP_CAMERA": str(tmp_path / "must-not-be-opened-up"),
        }
    )
    return environment


@pytest.mark.parametrize("relative_path", PYTHON_ENTRYPOINTS, ids=str)
def test_python_entrypoint_refuses_without_device_or_lerobot_access(
    relative_path: Path,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / relative_path), "--unexpected-option"],
        cwd=tmp_path,
        env=poisoned_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 64
    assert "disabled" in completed.stderr.lower()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative_path", SHELL_ENTRYPOINTS, ids=str)
def test_shell_entrypoint_refuses_using_only_shell_builtins(
    relative_path: Path,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["/bin/bash", str(REPO_ROOT / relative_path), "--unexpected-option"],
        cwd=tmp_path,
        env=poisoned_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 64
    assert "disabled" in completed.stderr.lower()
    assert not list(tmp_path.iterdir())


def test_python_entrypoints_have_only_standard_library_imports() -> None:
    allowed_roots = {"__future__", "argparse", "sys", "typing"}

    for relative_path in PYTHON_ENTRYPOINTS:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            (node.module or "").split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert imported_roots <= allowed_roots, relative_path


def test_retired_entrypoints_contain_no_hardware_or_patch_implementation() -> None:
    forbidden_python = (
        "import lerobot",
        "from lerobot",
        "lerobot_robot_",
        "lerobot_teleoperator_",
        "sync_write",
        "sync_read",
        ".connect(",
        "setattr(",
        "/dev/",
    )
    forbidden_shell = (
        "/dev/",
        "sg dialout",
        "ros2 launch",
        "lerobot-calibrate",
        "lerobot-teleoperate",
        "lerobot-record",
    )

    for relative_path in PYTHON_ENTRYPOINTS:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden_python), relative_path

    for relative_path in SHELL_ENTRYPOINTS:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden_shell), relative_path


def test_inference_stubs_point_to_unified_execute_command() -> None:
    inference_paths = (
        Path("scripts/infer_keep_pose.py"),
        Path("scripts/run_viola_inference.sh"),
    )

    for relative_path in inference_paths:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "viola-ops policy execute --help" in source
