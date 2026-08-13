from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_new_pc.sh"
PREFLIGHT = REPO_ROOT / "scripts" / "preflight.sh"
REQUIREMENTS = REPO_ROOT / "requirements-validated.txt"


def test_shell_scripts_are_valid_bash() -> None:
    for script in (BOOTSTRAP, PREFLIGHT):
        completed = subprocess.run(
            ["/bin/bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_bootstrap_installs_repo_editable_without_dependency_resolution() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert 'python -m pip install --no-deps --editable "${REPO_DIR}"' in source
    assert "operation.env" not in source
    assert "install_calibrations" not in source
    assert "sync_policy" not in source
    assert "dialout" not in source


def test_preflight_is_software_and_contract_only() -> None:
    source = PREFLIGHT.read_text(encoding="utf-8")
    forbidden = (
        "/dev/serial",
        "/dev/v4l",
        "VIOLA_ROBOT_PORT",
        "VIOLA_TELEOP_PORT",
        "VIOLA_FRONT_CAMERA",
        "VIOLA_UP_CAMERA",
        "POLICY_DIR",
        "CALIBRATION_ROOT",
        "MODEL_SHA256",
        "wandb login",
        "wandb.Api",
        "curl ",
        "wget ",
    )

    assert not any(token in source for token in forbidden)
    assert "passing this audit is not motion authorization" in source
    assert "W&B connectivity" in source
    assert "CONTRACT_SHA256" in source
    assert "expected_schemas" in source


def test_every_requirement_is_exactly_pinned_without_a_date_claim() -> None:
    source = REQUIREMENTS.read_text(encoding="utf-8")
    package_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert package_lines
    assert all(line.count("==") == 1 for line in package_lines)
    assert not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", source)
