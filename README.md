# StarAI Viola operations and inference

Private, self-contained operating package for the StarAI Viola follower,
StarAI Violin leader, calibration, two-camera preview/video, teleoperation,
LeRobot episode recording, and ACT policy inference. This repository contains
**no model-training code**.

Validated runtime:

- Python 3.12 and LeRobot 0.6.1
- StarAI Viola, Violin, and motor plugins 0.0.4
- Logitech `front` and `up` cameras at 640×480
- ACT step-80,000 deployment checkpoint
- keep-current-pose startup and a `3.0` normalized per-write bound
- Rerun display of both cameras throughout inference

> Physical robots can injure people or damage equipment. Clear the workspace,
> secure the base, keep the physical power cutoff accessible, and run the
> read-only preflight before every session. Software bounds are not an E-stop.

## Migrate to a new PC: complete procedure

The Git repository contains the operating code, pinned dependency list, and
calibration snapshots. It deliberately does **not** contain the model checkpoint,
datasets, recordings, logs, credentials, or the machine-local
`config/operation.env`. Copy any of those that you want to keep separately.

### 1. Prepare access and the new Linux PC

Before disconnecting the old PC, confirm that the new PC can access:

- the private GitHub repository `yuanlong-o/starai-viola-lerobot-ops`;
- the complete seven-file ACT checkpoint on NAS, an external disk, or the old PC;
- this exact physical Viola/Violin pair if the tracked calibrations will be used.

Install an NVIDIA driver compatible with PyTorch 2.7.1, Conda (Miniconda or
Anaconda), GitHub CLI, Git, `rsync`, `v4l2-ctl`, and FFmpeg. On Ubuntu, the
ordinary system utilities can be installed with:

```bash
sudo apt update
sudo apt install -y git rsync v4l-utils ffmpeg
gh auth login
gh auth status
```

Do not copy the old Conda environment directory. The bootstrap recreates the
validated Python 3.12 environment from the pinned requirements.

### 2. Clone the private operations repository

The validated operations code is on the default `main` branch:

```bash
cd "${HOME}"
gh repo clone yuanlong-o/starai-viola-lerobot-ops
cd "${HOME}/starai-viola-lerobot-ops"
git status
```

`git status` should report a clean worktree on `main`.

### 3. Recreate the unified operation environment

```bash
./scripts/bootstrap_new_pc.sh
conda activate lerobot
python --version
python -m pip check
```

The bootstrap creates the `lerobot` Conda environment and installs LeRobot
0.6.1, the three StarAI plugins, camera/video support, Rerun, and inference
dependencies. Training code and training dependencies are not installed.

### 4. Grant serial and camera permissions

```bash
sudo usermod -aG dialout,video "$USER"
```

Log out of the entire graphical desktop session and log back in; opening only a
new terminal is insufficient. Then verify:

```bash
id -nG
```

Both `dialout` and `video` must appear. Do not run robot programs with `sudo`,
use `chmod 777`, or create broad device permissions.

### 5. Connect and identify the hardware

Connect the arms and cameras, preferably to USB sockets that will remain fixed.
The two CH340 arm adapters do not expose unique serial IDs, so identify them by
unplugging one arm at a time:

```bash
ls -l /dev/serial/by-path/
```

Record which path disappears for the Viola follower and which disappears for
the Violin leader. Then list the cameras:

```bash
ls -l /dev/v4l/by-id/
```

Use each Logitech camera's `video-index0` path. The camera serial IDs normally
follow the cameras to the new PC, but the arm `by-path` values will commonly
change. Never assume the old PC's port 10/port 11 mapping is still correct.

### 6. Create the machine-local configuration

The bootstrap normally creates this file. The guarded copy command also works
if setup was performed manually:

```bash
cd "${HOME}/starai-viola-lerobot-ops"
test -f config/operation.env || \
  cp config/operation.env.example config/operation.env
${EDITOR:-nano} config/operation.env
```

At minimum, verify or change these entries:

```bash
LEROBOT_ENV_NAME=lerobot
VIOLA_ROBOT_PORT=/dev/serial/by-path/REPLACE_WITH_VIOLA_FOLLOWER_PATH
VIOLA_TELEOP_PORT=/dev/serial/by-path/REPLACE_WITH_VIOLIN_LEADER_PATH
VIOLA_FRONT_CAMERA=/dev/v4l/by-id/REPLACE_WITH_FRONT-video-index0
VIOLA_UP_CAMERA=/dev/v4l/by-id/REPLACE_WITH_UP-video-index0
VIOLA_ROBOT_ID=my_awesome_staraiviola_arm
VIOLA_TELEOP_ID=my_awesome_staraiviolin_arm
VIOLA_POLICY_DIR=${HOME}/models/act_viola_val20_step080000
VIOLA_MAX_STEP=3.0
```

`config/operation.env` is intentionally ignored by Git because device and local
storage paths differ between PCs.

### 7. Verify both camera identities and framing

Close all other camera applications, then run:

```bash
./scripts/run_dual_camera_view.sh
```

Confirm `front` is the task-wide view and `up` is the overhead view. Both must
show 640×480 images continuously. Press Q or Esc to close both windows before
starting another workflow. If the views are reversed, swap only
`VIOLA_FRONT_CAMERA` and `VIOLA_UP_CAMERA` in `config/operation.env` and repeat.

### 8. Restore calibration for this exact arm pair

First inspect the destination:

```bash
./scripts/install_calibrations.sh --check || true
```

On a fresh PC, install the tracked snapshots:

```bash
./scripts/install_calibrations.sh --install
./scripts/install_calibrations.sh --check
```

Use `--replace` only when files already exist and you have confirmed that these
are the same physical arms; the script backs up replaced files. Recalibrate
instead of restoring snapshots after a motor replacement, joint reassembly, ID
change, or mechanical alignment change.

### 9. Transfer and verify the inference checkpoint

If the original NAS path is mounted at the same location:

```bash
./scripts/sync_policy.sh
```

If the checkpoint is on another mount, external disk, or copied from the old PC,
point to the directory containing all seven `pretrained_model` files:

```bash
VIOLA_POLICY_SOURCE=/path/to/pretrained_model \
  ./scripts/sync_policy.sh
```

The script copies the bundle to `VIOLA_POLICY_DIR` and refuses it unless all
seven files exist and `model.safetensors` matches SHA-256
`1093aaeddfb902e7e596425d87676baba58cb8ab617a52c954ec11940726b886`.

### 10. Run the read-only migration gate

```bash
conda activate lerobot
python -m pytest -q
./scripts/preflight.sh
```

Do not continue until the tests pass and preflight prints:

```text
Preflight passed. Hardware was not opened.
```

This checks the environment versions, imports, stable device paths,
read/write permissions, both calibration hashes, and the policy hash without
opening a serial port or camera.

### 11. Test operation in increasing-risk order

Clear the full arm workspace, secure both bases, support the follower, and keep
the physical power cutoff reachable. Close every other camera and robot process.
Then test in this order:

```bash
# Cameras only; press Q or Esc after confirming both live views.
./scripts/run_dual_camera_view.sh

# Leader/follower control with both cameras displayed in Rerun; Ctrl-C stops.
./scripts/run_viola_teleoperation.sh

# Ten-second ACT rollout with both cameras displayed throughout.
./scripts/run_viola_inference.sh 10
```

Start teleoperation with small leader movements. Stop immediately if the arm
roles are reversed, a joint direction is wrong, the follower jumps, either
camera freezes, or Rerun does not show both views. Software bounds are not an
E-stop.

### 12. Optional: migrate local datasets and recordings

This is unnecessary for operating or inference. If the old data is needed,
copy it separately after the software migration, preserving directory contents:

```bash
rsync -a --info=progress2 \
  OLD_PC_OR_DISK:/path/to/lerobot-data/ "${HOME}/lerobot-data/"
```

Update `VIOLA_DATASET_DIR` in `config/operation.env`, then inspect existing
datasets before resuming them. Never append new episodes if either calibration
changed. MP4 recordings, LeRobot datasets, logs, and checkpoints remain ignored
by Git.

Rerun opens before robot connection for teleoperation and inference. The
launchers use MJPG for `front` and YUYV for `up`; both decode to 640×480 RGB,
while YUYV avoids the observed intermittent MJPEG stall on the up camera.

## Model bundle

Weights are deliberately excluded from Git. `scripts/sync_policy.sh` copies the
complete seven-file checkpoint bundle and verifies the deployment model hash:

```text
1093aaeddfb902e7e596425d87676baba58cb8ab617a52c954ec11940726b886
```

Default source and destination:

```text
/mnt/nas02/yz/starai/outputs/act_viola_right_to_left_blue_then_red_val20_v1/checkpoints/080000/pretrained_model
~/models/act_viola_val20_step080000
```

Override the source with `VIOLA_POLICY_SOURCE=/path/to/pretrained_model`.

## Other operating workflows

| Goal | Guide |
|---|---|
| Copy/paste repeat-use commands | [Command reference](docs/COMMAND_REFERENCE.md) |
| Identify USB devices and migrate PCs | [Setup and ports](docs/SETUP_AND_PORTS.md) |
| Understand physical and software guards | [Safety](docs/SAFETY.md) |
| Keep-pose leader/follower operation | [Teleoperation](docs/TELEOPERATION.md) |
| Preview or record both cameras | [Camera and video](docs/CAMERA_VIDEO.md) |
| Record local LeRobot episodes | [Episode recording](docs/EPISODE_RECORDING.md) |
| Restore this exact arm pair's calibration | [Calibration](docs/CALIBRATION.md) |
| Diagnose devices, cameras, or Rerun | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Optional ROS 2 / MoveIt operation | [ROS 2 / MoveIt](docs/ROS2_MOVEIT.md) |

## Repository contents

```text
calibration/                    reviewed snapshots for this physical arm pair
config/operation.env.example   portable machine/device configuration template
docs/                           operating and migration procedures
scripts/bootstrap_new_pc.sh    creates the unified Conda environment
scripts/preflight.sh           read-only hardware/software/model verification
scripts/sync_policy.sh         transfers and verifies the ACT checkpoint
scripts/infer_keep_pose.py      guarded LeRobot policy rollout wrapper
scripts/run_viola_inference.sh displayed inference launcher
scripts/teleoperate_keep_pose.py keep-pose leader/follower wrapper
scripts/record_episodes_keep_pose.py guarded local episode recorder
ros2/                           optional pinned ROS 2/MoveIt workflow
tests/                          no-hardware regression tests
```

Generated datasets, checkpoints, recordings, logs, credentials, and local
`config/operation.env` are ignored. Model training remains in the separate private
`yuanlong-o/starai-viola-act-training` repository and is not required here.
