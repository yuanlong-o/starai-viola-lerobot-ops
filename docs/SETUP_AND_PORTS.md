# Setup, ports, and permissions

## Validated workstation mapping

| Role | Stable path used here | Current kernel node |
|---|---|---|
| Viola follower/robot | `/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0` | `/dev/ttyUSB0` |
| Violin teacher/teleoperator | `/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0` | `/dev/ttyUSB1` |
| Front camera | `/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0` | `/dev/video0` |
| Up camera | `/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0` | `/dev/video2` |

Use the `video-index0` camera nodes. The `video-index1` nodes are not the
capture streams used by this setup.

The two arm adapters report the same USB identity and do not have unique serial
numbers. Their `/dev/serial/by-id` name is ambiguous, so the arm commands use
physical `by-path` links. The cameras do have unique serial numbers, so their
`by-id` links are preferred.

## What changes after shutdown or moving machines

- A normal shutdown does not erase calibration and normally does not change
  these paths.
- `/dev/ttyUSB0` and `/dev/ttyUSB1` can swap after any re-enumeration. Do not use
  them as the source of truth.
- An arm `by-path` link stays stable only while the cable, hub topology, and
  physical USB socket remain the same.
- Moving an arm cable to another socket or another PC changes its `by-path`.
- Camera `by-id` links should follow each camera to another socket or PC because
  the cameras have unique serials; still verify the views visually.
- Calibration files and Python packages live on the PC. They do not travel with
  the arm automatically.

## Identify the arms safely with an unplug test

Do not guess from `ttyUSB` numbers. With motor power disabled if possible:

```bash
ls -l /dev/serial/by-path/
```

Unplug only the teacher adapter, repeat the command, and note which path
disappeared. Reconnect it to the same socket. Then repeat for the follower. Put
the verified paths into environment overrides if they differ from this machine:

```bash
export VIOLA_ROBOT_PORT=/dev/serial/by-path/REPLACE_WITH_FOLLOWER_PATH
export VIOLA_TELEOP_PORT=/dev/serial/by-path/REPLACE_WITH_TEACHER_PATH
```

Verify cameras with:

```bash
ls -l /dev/v4l/by-id/

/home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/dual_camera_view.py \
  --devices \
  /dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0 \
  /dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0
```

The first window must be front and the second must be up. Press Q or Esc to
close both before starting teleoperation or episode recording.

## Serial and camera permissions

The serial devices are `root:dialout` with mode `0660`. The current account is
listed in `dialout`, but an already-running login session may not contain the
new supplementary group. Either log out completely and log back in, or use the
documented wrapper:

```bash
sg dialout -c 'exec COMMAND_HERE'
```

Check the active process groups with:

```bash
id
```

Camera access on the current desktop comes from a user ACL. On a headless or new
machine, add the user to `video`, then log out and back in:

```bash
sudo usermod -aG dialout,video "$USER"
```

Do not solve permissions with `sudo lerobot-*`, `chmod 777`, or persistent
world-writable udev rules. Those approaches create root-owned data or allow any
local process to command the robot.

## Reproduce the validated software

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot

git clone https://github.com/huggingface/lerobot.git /home/yz/lerobot
git -C /home/yz/lerobot checkout d9e74a9d374a8f26582ad326c699740a227b483c
python -m pip install -e /home/yz/lerobot

gh repo clone yuanlong-o/starai-viola-lerobot-ops \
  /home/yz/lerobot/starai-viola-lerobot-ops

python -m pip install -r \
  /home/yz/lerobot/starai-viola-lerobot-ops/requirements-validated.txt
```

If `/home/yz/lerobot` already exists, do not clone over it. Review its branch and
local changes first. The keep-pose code monkey-patches private plugin behavior,
so an untested LeRobot or StarAI plugin upgrade must be treated as a new hardware
validation project.

If GitHub is unavailable on a machine with the shared NAS mounted, clone the
committed fallback after it has been created by the release procedure:

```bash
git clone /mnt/nas02/yz/starai/repos/starai-viola-lerobot-ops.git \
  /home/yz/lerobot/starai-viola-lerobot-ops
```

## New-PC checklist

1. Install the pinned LeRobot checkout and Python packages.
2. Clone this private operations repository.
3. Copy/restore calibration only for the same physical arm pair.
4. Add the account to `dialout` and `video`; log out and back in.
5. Remap both arms with the unplug test.
6. Verify the two camera views.
7. Export any changed paths and run `scripts/preflight.sh`.
8. With a clear workspace and power cutoff available, test keep-pose
   teleoperation at low speed before recording data.
