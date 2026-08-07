# Left-to-right episode recording

This workflow creates LeRobot training episodes with synchronized robot state,
commanded action, front video, overhead video, and one language task label.

The active task is:

> Move the blue cube, then the red cube, from the gray platform on the left to
> the white pad on the right.

The active dataset is:

```text
repo_id: bourn117/viola_cubes_left_to_right_keep_pose_v3
root:    /home/yz/lerobot/data/recordings/bourn117/viola_cubes_left_to_right_keep_pose_v3
```

## One-time preparation

Ensure the parent directory exists:

```bash
mkdir -p /home/yz/lerobot/data/recordings/bourn117
```

Close the dual-camera viewer and every old recorder. Put both cubes on the left
gray platform. Put both arms in safe starting poses. Then run the launcher's
read-only checks:

```bash
sg dialout -c 'exec \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/record_left_to_right_episode.sh \
  --check'
```

It checks software, calibration hashes, dataset metadata when resuming, required
files, permissions, and device contention. It does not open the hardware or
write a dataset.

## Record

```bash
sg dialout -c 'exec \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/record_left_to_right_episode.sh'
```

If the dataset does not exist, the first run records exactly one pilot episode.
Inspect that pilot before continuing. Later runs validate and resume the same
dataset, with up to ten accepted episodes per run.

Recording begins automatically after connection and initialization. **Right
Arrow does not start the first episode.** Before launching the command, the cubes
and arms must already be in the desired start state.

## Controls and phases

| Current phase | Key | Result |
|---|---|---|
| Recording | Right Arrow | Accept the episode after at least 5 seconds |
| Recording | Left Arrow | Discard it and re-record the same episode index |
| Recording | Esc | Discard the partial episode and stop |
| Saving/encoding | any arrow key | Ignored; wait until encoding finishes |
| Reset | Right Arrow | Skip the remaining reset time and start the next episode |
| Reset | Esc | Preserve the accepted episode and stop cleanly |

Key auto-repeat is ignored until the key is released. A Right Arrow press during
the first five seconds is ignored and recording continues.

`episode_time_s=60` is a warning boundary, not automatic acceptance. If no key is
pressed, recording continues so a task is never silently truncated. This also
means unattended recording can consume disk space.

### Why it feels like two Right Arrow presses

They control two different phases:

1. Right Arrow during **recording** accepts the completed episode.
2. The program saves and encodes the videos. Keys cannot skip this work.
3. After the console says the **reset** phase has begun, place both cubes back on
   the left platform.
4. Right Arrow during reset skips the remaining 60-second reset timer.

Pressing Right Arrow repeatedly during saving does not queue a skip. Wait for the
reset message, reset both cubes, and tap it once.

## What makes an episode valid

- Both cubes are fully on the left gray platform in the first frames.
- Blue is moved first and placed on the right white pad.
- Red is moved second and placed on the right white pad.
- Both finish supported by the right pad.
- No manual cube repositioning occurs after recording starts.
- No unfinished grasp, drop, collision recovery, long inactive segment, or
  camera corruption is present.

If the start state is wrong, press Left Arrow and re-record. Do not manually move
a cube during the episode and plan to cut it out later.

## Stop and resume

The safest planned stop is Esc during reset: the last accepted episode remains
saved. If the process is interrupted during recording, the partial episode is
not valid. Preserve the directory and log, then run `--check`; do not delete
unknown files until the failure is understood.

Logs are written under:

```text
/home/yz/lerobot/logs/episode_recording/
```

The launcher uses a dataset lock so a second writer fails instead of corrupting
the active dataset.

## Path overrides for another workstation

The launcher accepts environment overrides without editing the tracked file:

```bash
export LEROBOT_PROJECT_DIR=/path/to/lerobot
export LEROBOT_PYTHON_BIN=/path/to/conda/envs/lerobot/bin/python
export VIOLA_LTR_DATASET_ROOT=/path/to/new/dataset
export VIOLA_ROBOT_PORT=/dev/serial/by-path/FOLLOWER_PATH
export VIOLA_TELEOP_PORT=/dev/serial/by-path/TEACHER_PATH
export VIOLA_FRONT_CAMERA=/dev/v4l/by-id/FRONT-video-index0
export VIOLA_UP_CAMERA=/dev/v4l/by-id/UP-video-index0
export HF_LEROBOT_CALIBRATION=/path/to/calibration
```

If calibration changes, the launcher intentionally refuses to mix it into this
dataset version. Create a new version and update the expected hashes only after
review.
