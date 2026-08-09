# Operation command reference

Run commands from the repository root after completing
[new-PC setup](SETUP_AND_PORTS.md). Paths and IDs come from
`config/operation.env`.

## Read-only preflight

```bash
conda activate lerobot
./scripts/preflight.sh
```

## Displayed ACT inference

```bash
./scripts/run_viola_inference.sh 10
```

The argument is the positive inference duration in seconds. Rerun displays
both cameras throughout the policy-control phase.

## Keep-pose leader/follower teleoperation

```bash
./scripts/run_viola_teleoperation.sh
```

Press Ctrl-C to stop. The wrapper measures both startup poses and bounds each
follower command by `VIOLA_MAX_STEP` (default `3.0`).

## Preview both cameras without motors

```bash
conda run --no-capture-output -n lerobot \
  python scripts/dual_camera_view.py --devices \
  "${VIOLA_FRONT_CAMERA}" "${VIOLA_UP_CAMERA}"
```

If the shell variables are not exported, copy the two values from
`config/operation.env`. Press Q or Esc before starting another camera process.

## Calibration snapshots

```bash
./scripts/install_calibrations.sh --check
```

Use `--install` only when the destination is absent. Use `--replace` only after
confirming these are the same physical arms; existing files are backed up.

## Interactive calibration

```bash
./scripts/run_viola_calibration.sh violin
./scripts/run_viola_calibration.sh viola
```

Run only the arm that actually needs calibration. These commands can move the
arm during connection; follow the safety checklist in the calibration guide.

## Five-minute two-camera video

```bash
./scripts/run_dual_camera_view.sh
./scripts/run_dual_camera_record.sh 300
```

Both commands display both cameras for the entire process. Close either viewer
with Q or Esc before starting teleoperation, episode recording, or inference.

## Local LeRobot episode recording

```bash
./scripts/run_viola_episode_recording.sh right-to-left --check
./scripts/run_viola_episode_recording.sh right-to-left

./scripts/run_viola_episode_recording.sh left-to-right --check
./scripts/run_viola_episode_recording.sh left-to-right
```

The recorder displays both cameras in Rerun throughout capture. Datasets remain
local under `VIOLA_DATASET_DIR`; no Hub upload or model training is performed.

## Refresh or verify the policy

```bash
./scripts/sync_policy.sh
```

Set `VIOLA_POLICY_SOURCE` when the NAS path differs.

## No-hardware tests

```bash
conda run --no-capture-output -n lerobot python -m pytest -q
```

## Optional ROS 2 / MoveIt

The ROS 2 workflow is isolated from LeRobot. Follow
[ROS 2 / MoveIt](ROS2_MOVEIT.md); do not run ROS and LeRobot hardware control
simultaneously.
