#!/usr/bin/env python3
"""Retired inference launcher with two retained, pure calculation helpers."""

from __future__ import annotations

import argparse
import sys
from typing import Any


EXIT_RETIRED = 64
MIGRATION_MESSAGE = """\
Legacy inference is disabled.

The exact previously deployed ACT model now has a Repo-A-local safety gate. It
requires reviewed local bytes, a clean revision, current E-stop evidence,
online W&B, and explicit operator actions, but no Repo-B handoff.

Start with:
  viola-ops act run --help
"""


def _extract_safety_args(argv: list[str]) -> tuple[float, list[str]]:
    """Parse the retired wrapper's numeric option without touching hardware."""

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
    """Return the historical bounded target as a side-effect-free calculation."""

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


def main() -> None:
    """Refuse the retired path before importing or inspecting any device API."""

    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    raise SystemExit(EXIT_RETIRED)


if __name__ == "__main__":
    main()
