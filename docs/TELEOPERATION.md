# Keep-current-pose teleoperation

The command below is the tested replacement for stock `lerobot-teleoperate`.
At startup it measures the teacher and follower separately, holds the follower
at its measured pose, and maps teacher displacement relative to the two startup
poses.

## Before starting

- Close `dual_camera_view.py`, video recorders, and any other camera process.
- Put the follower in a safe pose; put the teacher in a comfortable pose.
- The arms do not need to be physically aligned.
- Put both cubes in their required task start region before the program starts.
- Run the read-only preflight and resolve every failure.

## Copy/paste command

```bash
./scripts/preflight.sh
./scripts/run_viola_teleoperation.sh
```

Stop with Ctrl+C. The follower disconnects with
`disable_torque_on_disconnect=true`.

## What should happen

1. The Violin teacher connects and remains at its current pose.
2. The Viola follower connects and receives a goal equal to its measured
   current pose, rather than the plugin's fixed startup pose.
3. Both startup poses are captured.
4. Movement of the teacher from its startup pose produces the same normalized
   displacement around the follower's startup pose.
5. Rerun opens and receives state, action, and both camera streams.

The wrapper clamps arm joints to normalized `[-100, 100]`, the gripper to
`[0, 100]`, and each requested control step to `3.0`. This is not a collision or
force limit.

## If the arm still moves unexpectedly

Cut motor power first. Then check:

- The command invokes `teleoperate_keep_pose.py`, not `lerobot-teleoperate`.
- The follower and teacher paths have not been exchanged.
- `--robot.use_degrees=false` and `--teleop.use_degrees=false` are present.
- Calibration checksums match the files in [Calibration](CALIBRATION.md).
- The StarAI and LeRobot versions pass `scripts/preflight.sh`.
- No second program has the arm port open.

Do not compensate for an unexplained startup movement by changing the physical
start pose. Diagnose the port, calibration, command, and software first.
