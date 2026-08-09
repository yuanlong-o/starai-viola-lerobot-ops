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

## New PC: clone to first inference

Prerequisites: Linux, an NVIDIA driver compatible with PyTorch 2.7.1, Conda,
Git, access to this private GitHub repository, and access to the checkpoint on
NAS or another machine.

```bash
gh repo clone yuanlong-o/starai-viola-lerobot-ops
cd starai-viola-lerobot-ops

./scripts/bootstrap_new_pc.sh
sudo usermod -aG dialout,video "$USER"
```

Log out completely and back in after changing groups. Then identify the arm
ports with the unplug procedure in [New-PC setup](docs/SETUP_AND_PORTS.md), and:

```bash
cp config/operation.env.example config/operation.env
${EDITOR:-nano} config/operation.env

./scripts/install_calibrations.sh --install
./scripts/sync_policy.sh
conda activate lerobot
./scripts/preflight.sh
```

With the arm workspace clear, run ten seconds of displayed inference:

```bash
./scripts/run_viola_inference.sh 10
```

Rerun opens before hardware connection. After both cameras finish warming up,
the `front` and `up` RGB streams are displayed throughout the policy-control
phase. The launcher uses MJPG for `front` and YUYV for `up`; both decode to the
same 640×480 RGB model inputs, while YUYV avoids the observed intermittent
MJPEG stall on the up camera.

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
