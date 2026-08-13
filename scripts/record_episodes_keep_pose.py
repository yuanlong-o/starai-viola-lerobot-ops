#!/usr/bin/env python3
"""Fail-closed replacement for the retired episode-recording wrapper."""

from __future__ import annotations

import sys


EXIT_RETIRED = 64
MIGRATION_MESSAGE = """\
Legacy Viola episode recording is disabled.

Demonstration recording is not supported by the live-session contract. This
entrypoint cannot connect cameras, serial devices, a leader, or a follower.
Use the reviewed Viola operations workflow for supported work.
"""


def main() -> None:
    """Refuse before importing or inspecting any device API."""

    print(MIGRATION_MESSAGE, file=sys.stderr, end="")
    raise SystemExit(EXIT_RETIRED)


if __name__ == "__main__":
    main()
