#!/usr/bin/env python3
"""Print PIDs whose open file descriptors resolve to a requested device."""

import argparse
import glob
import os


def find_device_users(device_path: str) -> list[int]:
    target = os.path.realpath(device_path)
    users: set[int] = set()

    for descriptor_path in glob.iglob("/proc/[0-9]*/fd/*"):
        try:
            if os.path.realpath(descriptor_path) == target:
                users.add(int(descriptor_path.split("/", 3)[2]))
        except (OSError, PermissionError, ValueError):
            continue

    return sorted(users)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("device_path")
    args = parser.parse_args()
    print(" ".join(str(pid) for pid in find_device_users(args.device_path)))


if __name__ == "__main__":
    main()
