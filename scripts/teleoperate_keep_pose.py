#!/usr/bin/env python3
"""Retired teleoperation launcher with its pure mapping helper retained."""

from __future__ import annotations

import sys


EXIT_RETIRED = 64
MIGRATION_MESSAGE = """\
Legacy Viola teleoperation is disabled.

Direct teleoperation is not supported by the live-session contract. Do not use
this script to connect a leader, follower, camera, serial device, or motor.
Policy motion is available only through:
  viola-ops policy execute --help
"""


def _bounded_relative_action(
    leader_action: dict[str, float],
    leader_start: dict[str, float],
    follower_start: dict[str, float],
    previous_action: dict[str, float],
    max_step: float,
) -> dict[str, float]:
    """Compute the historical relative target without performing any I/O."""

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


def main() -> None:
    """Refuse the retired path before importing or inspecting any device API."""

    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    raise SystemExit(EXIT_RETIRED)


if __name__ == "__main__":
    main()
