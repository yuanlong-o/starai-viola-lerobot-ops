# Safety

## Before enabling either arm

1. Confirm the Viola follower is the adapter on the current `usb-0:10` path and
   the Violin teacher is on `usb-0:11`. If anything was replugged, perform the
   unplug test in [Setup, ports, and permissions](SETUP_AND_PORTS.md).
2. Put both arms in comfortable, collision-free poses. The follower does not
   need to match the teacher; the keep-pose wrapper measures both starting poses.
3. Clear cubes, tools, cables, people, and camera stands from the arm envelope.
4. Ensure the arm base and cameras cannot move.
5. Keep immediate access to the hardware power or torque cutoff.
6. Run the read-only preflight from the repository root:

   ```bash
   sg dialout -c 'exec ./scripts/preflight.sh'
   ```

## Why the stock command moves to a strange pose

In the installed `lerobot_robot_viola==0.0.4` and
`lerobot_teleoperator_violin==0.0.4` plugins, `connect()` calls
`move_to_initial_position()` when a calibration file is loaded. That method
commands this normalized target over 1.5 seconds:

```text
Motor_0=0, Motor_1=-100, Motor_2=60, Motor_3=0,
Motor_4=30, Motor_5=0, gripper=50
```

This is why the arm can jump or fold into an unexpected shape after a stock
`lerobot-teleoperate`, `lerobot-record`, or even calibration connection.

The wrappers in this repository patch that behavior only inside their Python
process. They read the current follower pose, engage with a goal equal to that
pose, leave the teacher unlocked at its current pose, and then use:

```text
follower target = follower startup + (teacher current - teacher startup)
```

The first requested follower target is therefore its measured current pose.

## Limits of the software guard

- `--max_step=3.0` limits each normalized command step, not physical velocity,
  force, energy, or collision risk.
- The wrapper assumes seven normalized controls named `Motor_0` through
  `Motor_5` and `gripper`; it is not a general UR5 safety layer.
- A wrong port assignment, wrong calibration, failed sensor, plugin update, or
  external program can still cause dangerous movement.
- Never rely on a GUI, keyboard shortcut, Python exception, or USB disconnect as
  the emergency stop.

## Rules that protect recorded data

- Never run two writers against one dataset root.
- Do not move a cube manually after an episode starts. Discard the episode with
  Left Arrow and reset during the reset phase.
- Do not append to the frozen right-to-left source dataset.
- Do not mix demonstrations made with different calibrations in one dataset
  version.
- Preserve interrupted or suspicious directories for diagnosis; do not record
  over them blindly.
