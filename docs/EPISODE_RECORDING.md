# Local LeRobot episode recording

The operations repository can capture synchronized robot state, bounded action,
and both camera streams for later use. It does not contain or run model training.

Before every session, close camera previews and other robot processes, clear the
workspace, support the Viola, and keep the physical cutoff reachable. Validate
the selected direction without opening hardware:

```bash
./scripts/run_viola_episode_recording.sh right-to-left --check
./scripts/run_viola_episode_recording.sh left-to-right --check
```

Record a direction with the same command minus `--check`:

```bash
./scripts/run_viola_episode_recording.sh right-to-left
./scripts/run_viola_episode_recording.sh left-to-right
```

Rerun displays `front` and `up` throughout recording. The launcher uses MJPG for
front and the stable YUYV mode for up, keeps the follower at its measured startup
pose, and bounds each action step by `VIOLA_MAX_STEP` (default `3.0`). Right Arrow
accepts an episode after the five-second safety floor, Left Arrow rejects and
repeats it, and Esc stops cleanly. Reaching 60 seconds without acceptance keeps
recording; run from the graphical desktop for Rerun and release-aware hotkeys.

The first run creates one pilot episode below `VIOLA_DATASET_DIR` (default
`~/lerobot-data`). Later runs resume in batches of up to ten only when
`meta/info.json` exists. Datasets, videos, and logs are ignored by Git and are
never pushed to Hugging Face.
