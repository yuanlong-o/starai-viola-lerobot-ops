# Troubleshooting

Start with the terminal output and the latest log. Rerun is a visualization
client; an empty or stale panel does not prove that recording is healthy.

## `Permission denied` on a serial port

Check the active groups and device ownership:

```bash
id
ls -l /dev/ttyUSB* /dev/serial/by-path/*
```

Use the exact documented wrapper for the current login:

```bash
./scripts/preflight.sh
```

For a permanent fix, add the user to `dialout`, then fully log out and back in.
Opening a new terminal inside the old desktop session may not refresh groups.
Do not run robot commands with `sudo` and do not use `chmod 777`.

## `OpenCVCamera(...) read failed (status=False)`

The path can be correct and still fail when another process owns the camera, the
wrong V4L node is used, USB bandwidth/power is unstable, or the device has not
started producing MJPEG frames.

Close all preview and recording windows. Resolve the camera path and find users
from the normal desktop shell:

```bash
source config/operation.env
resolved_up="$(readlink -f -- "${VIOLA_UP_CAMERA}")"
echo "${resolved_up}"
conda run --no-capture-output -n "${LEROBOT_ENV_NAME:-lerobot}" \
  python scripts/find_device_users.py "${resolved_up}"
```

Repeat for the front camera. The helper is best-effort; `/proc` permissions can
hide processes. Also check:

```bash
v4l2-ctl --device="${resolved_up}" --list-formats-ext
```

Use `video-index0`, not `video-index1`. Unplug/replug only after stopping robot
control, then verify both views with `dual_camera_view.py`. The custom episode
recorder retries bounded camera startup/read failures; the stock recorder does
not contain all of these protections.

## Rerun opens but shows no trajectories or cameras

Check the phase and terminal:

- During device connection, an exception may have occurred before the control
  loop sent any Rerun data.
- During saving/encoding and reset, no new trajectory is being produced.
- The command must contain `--display_data=true`.
- An old Rerun viewer can be showing another session.
- A camera may be open in a separate viewer even though Rerun itself opened.
- A remote/headless shell may not have a usable `DISPLAY` for Rerun or `pynput`.

Close old viewers, run the launcher `--check`, then launch from the graphical
desktop terminal and watch its stdout. Do not keep operating solely because a
GUI window exists.

## The follower jumps to a strange pose at enable

Cut power first. This usually means a stock command invoked the StarAI plugin's
hardcoded startup pose. Confirm the command calls either
`teleoperate_keep_pose.py` or `run_viola_inference.sh`. Then verify arm
roles, calibration hashes, `use_degrees=false`, and pinned versions. See
[Safety](SAFETY.md).

## Right Arrow appears to require two presses

One press accepts an episode during recording. After saving/encoding finishes,
a second press during reset skips the remaining reset timer. Arrow keys pressed
during saving are ignored and are not queued. Wait for the reset message before
the second press.

## Reset cannot be skipped and seems to take about 30 seconds

The apparent delay is usually video saving/encoding, not reset. Encoding a pair
of episode videos has to finish before the recorder enters the reset phase and
can accept a reset-skip key. The launcher prints the phase. Do not terminate the
process to skip encoding.

## Recording was interrupted

- During recording: treat the partial episode as rejected. Preserve the root and
  log for diagnosis.
- During reset: previously accepted data should remain finalized.
- During saving/encoding: do not immediately resume. Inspect the log and run the
  launcher `--check`; partial metadata or unexpected directories must be handled
  before another writer opens the root.

The active launcher refuses to resume when metadata, schema, referenced files,
task label, episode indices, or expected directory structure is inconsistent.
That refusal protects the dataset; do not bypass it by deleting random files.

## Serial reads report missing positions or timeouts

Stop if the arm behaves abnormally. Check USB power, cable seating, correct arm
path, and whether another process owns the port. The custom recorder retries
known transient missing-position reads for a bounded interval; repeated failure
is a hardware/communication problem, not a reason to increase the retry forever.

## Where to look for evidence

```bash
ls -lt /home/yz/lerobot/logs/episode_recording/ | head
```

Keep the most recent log, the exact command, dataset `meta/`, and the first/last
frames when reporting a problem. Do not include GitHub/Hugging Face credentials
or unrelated environment dumps.
