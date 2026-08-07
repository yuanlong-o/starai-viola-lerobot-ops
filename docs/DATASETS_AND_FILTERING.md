# Dataset structure, inspection, and filtering

## What a recorded LeRobot dataset contains

The active recordings use the LeRobot v3 layout:

```text
dataset-root/
├── data/
│   └── chunk-*/file-*.parquet
├── meta/
│   ├── episodes/chunk-*/file-*.parquet
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
└── videos/
    ├── observation.images.front/chunk-*/file-*.mp4
    └── observation.images.up/chunk-*/file-*.mp4
```

- `data/*.parquet` contains frame-level timestamps, episode/frame indices,
  seven-dimensional `observation.state`, seven-dimensional `action`, task
  indices, and video timestamps.
- `meta/info.json` describes the schema, shape, frame rate, robot type, totals,
  and storage templates.
- `meta/episodes/*.parquet` stores one row per episode: length, task label, data
  location, and video location.
- `meta/tasks.parquet` maps task indices to language strings.
- `meta/stats.json` stores normalization statistics used in training.
- `videos/...` stores the two camera observations referenced by frame
  timestamps.

If the follower is replaced by a UR5 but data is still recorded through
LeRobot, the high-level folder pattern can remain similar. The robot type,
action/state dimensions, feature names, ranges, control rate, calibration,
driver, and launcher will change. Do not reuse the seven-motor StarAI wrappers,
calibration JSON, dataset schema assumptions, or trained policy for a UR5.

## Strict right-to-left result

The completed right-to-left source contained 78 episodes and 67,232 frames.
Both camera views and the trajectories were audited. The final trainable result
contains 34 episodes and 28,306 frames:

```text
/home/yz/lerobot/data/recordings/bourn117/viola_cubes_right_to_left_blue_then_red_train_v1
```

NAS copy:

```text
/mnt/nas02/yz/starai/datasets/bourn117/viola_cubes_right_to_left_blue_then_red_train_v1
```

Accepted original IDs:

```text
0, 1, 2, 4, 6, 18, 20, 23, 26, 31, 34, 35, 36, 38, 40, 43,
47, 48, 50, 52, 54, 57, 59, 61, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77
```

The immutable audit record and every rejection reason are under
`provenance/right_to_left_blue_then_red_v1/`. Common rejection reasons were red
before blue, wrong starting placement followed by a manual reset, incomplete
completion, manual movement after recording began, and excessive idle footage.

## Recheck or reproduce the filtered dataset

Non-mutating source check:

```bash
cd /home/yz/lerobot

/home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/filtering/build_right_to_left_training_dataset.py \
  --check
```

Validate the existing final dataset and refresh its validation report. This
decodes all video and writes `meta/validation_report.json`:

```bash
/home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/filtering/build_right_to_left_training_dataset.py \
  --validate-only
```

To reproduce into a new path without touching the canonical result:

```bash
VIOLA_RTL_TARGET_ROOT=/home/yz/lerobot/data/recordings/bourn117/viola_cubes_right_to_left_blue_then_red_train_v1_rebuild \
  /home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/filtering/build_right_to_left_training_dataset.py
```

The builder refuses to overwrite an existing target, locks against the source
recorder, records source tree hashes before and after, rebuilds through the
LeRobot API, validates numeric values and complete videos, and then atomically
promotes a staging directory.

## Remove known bad episodes safely

Do not delete MP4 or Parquet files manually; their indices and metadata are
coupled. Do not experiment on the only source copy.

LeRobot metadata indices are zero-based. If a person calls the demonstrations
“episodes 2–4,” their metadata indices are normally `[1, 2, 3]`.

First make a working copy:

```bash
SOURCE=/absolute/path/to/original_dataset
WORKING=/absolute/path/to/original_dataset_edited

cp -a --reflink=auto -- "$SOURCE" "$WORKING"
```

Then edit only the copy. Replace the repo ID and indices with the values from
that dataset:

```bash
/home/yz/anaconda3/envs/lerobot/bin/python \
  -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id=bourn117/REPLACE_WITH_DATASET_ID \
  --root="$WORKING" \
  --operation.type=delete_episodes \
  --operation.episode_indices='[1, 2, 3]'
```

In this LeRobot version, editing without `--new_repo_id` moves the working copy
to `${WORKING}_old` and creates the rebuilt result at `$WORKING`. Inspect and
validate the rebuilt result before treating it as trainable. Keep the untouched
source until training succeeds.

For important datasets, prefer a reviewed, task-specific manifest and canonical
builder like the right-to-left tool over ad hoc deletion.
