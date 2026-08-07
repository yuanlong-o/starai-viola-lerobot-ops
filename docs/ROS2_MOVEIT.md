# Safe StarAI Viola ROS 2 / MoveIt workspace

This is a rootless ROS 2 Humble and MoveIt 2 implementation of Seeed Studio's
StarAI arm guide for the Viola follower on this PC. It is intentionally
separate from `/home/yz/lerobot` and does not use `/opt/ros` or `sudo`.

Official references:

- Seeed guide: <https://wiki.seeedstudio.com/cn/starai_arm_ros_moveit/>
- pinned source: <https://github.com/Seeed-Projects/fashionstar-starai-arm-ros2/tree/be498c0034f30bfbe1ceacafeb773ffd95309a69>

The local safety patch changes the upstream behavior in several important
ways:

- uses the stable Viola follower path (USB path `0:10`) instead of
  `/dev/ttyUSB0`;
- reads a fresh reply from all seven servos before startup can continue;
- holds the measured pose by default, with no automatic multi-turn reset or
  zero-pose move;
- rejects existing port owners, requests PySerial's advisory lock, and applies
  Linux `TIOCEXCL` so later non-privileged opens of the follower tty fail;
- validates joint IDs, array sizes, finite values, URDF limits, and command
  durations before sending a packet;
- limits implied joint speed at both the action and raw driver boundaries;
- checks that a zero-time trajectory start agrees with fresh measured feedback;
- replaces the vendor SDK's silent serial-write failure path with checked
  full writes and transmit-buffer drains;
- uses acknowledged command and damping services, sends arm and gripper
  packets independently, supports cancellation, and rejects concurrent goals
  for the same resource;
- latches motion off and requests checked damping after sustained missing,
  incomplete, corrupt, or implausible feedback;
- starts real-robot MoveIt in planning-only mode by default;
- uses 5% velocity and acceleration scaling in both MoveIt limits and RViz;
- restricts ROS discovery to this PC and refuses startup when fake or existing
  Viola controllers are detected.

The implementation is based on official Seeed commit
`be498c0034f30bfbe1ceacafeb773ffd95309a69`; setup and preflight also require
the exact reviewed patched Git tree
`e5b01741b6be8f9cdb69d09c90ba12a20807b4a1`.

## One-time setup or rebuild

From this operations repository:

```bash
cd /home/yz/lerobot/starai-viola-lerobot-ops
./ros2/setup_starai.sh
```

For later rebuilds, the deployed copy is equivalent:

```bash
cd /home/yz/starai_ws
./setup_starai.sh
```

This creates or updates
`/home/yz/anaconda3/envs/starai_ros_humble`, applies the tracked safety patch,
builds all 14 upstream packages, and runs the 55 custom safety tests. It never
opens a serial device.

The private operations repository is the source of truth. Its setup script
installs these entry points into `/home/yz/starai_ws`, checks out the exact
upstream commit, verifies the patched tree hash, builds, and tests without
opening hardware.

## Read-only preflight

Run this before using the arm:

```bash
sg dialout -c 'exec /home/yz/starai_ws/preflight.sh'
```

The preflight inspects paths, permissions, package versions, launch defaults,
port ownership, ROS localhost isolation, and conflicting fake/real controller
nodes. It does not open serial or command a motor.

## Simulation first

```bash
/home/yz/starai_ws/run_viola_simulation.sh
```

The wrapper first refuses any discovered LeRobot, real Viola, MoveGroup, or
controller-manager process, then starts fake ROS 2 Control hardware without
opening serial. Close it with `Ctrl-C` before starting the hardware driver.
Do not bypass the wrapper: the fake and real stacks use the same absolute
action names, so overlapping them would be unsafe.

## Staged real-arm startup

> **Physical commissioning is still outstanding.** All validation so far used
> software and fake hardware; no serial device was opened and no motor command
> was sent. Treat the first real-arm run as a supervised bring-up with the arm
> supported and the physical power cutoff immediately reachable.

Do not run LeRobot teleoperation or recording while the ROS driver is active.
Clamp the base, clear the full reachable workspace, support the arm, and keep
the power cutoff within reach.

Terminal 1 — start the patched hardware driver and trajectory-action bridge:

```bash
sg dialout -c 'exec /home/yz/starai_ws/run_viola_driver.sh'
```

This wrapper always supplies the reviewed values:

```text
startup_mode:=hold_current
reset_multiturn:=false
shutdown_mode:=damping
feedback_fault_timeout_s:=0.5
```

Terminal 2 — confirm fresh measured feedback before MoveIt:

```bash
source /home/yz/starai_ws/starai_env.sh
ros2 topic echo /joint_states --once
```

Verify that all seven names and finite positions appear. Then start MoveIt in
planning-only mode:

```bash
ros2 launch viola_moveit_config actual_robot_demo.launch.py \
  allow_trajectory_execution:=false
```

Compare the physical arm against RViz at several manually observed poses. Do
not enable execution if a joint sign, zero, order, or gripper position does not
match.

`allow_trajectory_execution:=false` gates MoveGroup execution only. It is not
a motor-power interlock: the driver is holding the arm, and a custom caller
could still reach the action/service/topic APIs. Do not run other command
publishers, and use the physical power cutoff for an actual interlock.

Only after that check, stop the planning-only launch with `Ctrl-C` and restart
it with execution enabled:

```bash
ros2 launch viola_moveit_config actual_robot_demo.launch.py \
  allow_trajectory_execution:=true
```

Keep the RViz velocity and acceleration scaling at 5% for the first small
joint-space test. Plan first, inspect the complete path, and execute only with
the workspace clear.

If feedback is lost for 0.5 seconds, becomes incomplete, or falls outside a
plausible range, the driver requests damping and permanently rejects new
motion for that process. Correct the fault and restart the driver; the latch
does not clear itself. An action cancellation is reported as safely held only
after the driver acknowledges the checked serial write. The later measured
feedback check is what confirms that the servo actually followed a motion.

## Important limitations

- `hold_current` prevents the upstream startup jump; it is not calibration.
  The ROS driver converts six arm-servo angles from degrees to ROS radians and
  maps the gripper servo to meters; it does not read the two LeRobot
  calibration JSON files.
- This package models the Viola follower. It does not provide a MoveIt model
  for the Violin teacher (USB path `0:11`).
- The upstream tree also installs Cello packages, but this local serial driver,
  limits, launch defaults, and hardware safety review are Viola-specific.
  Cello real-hardware launch is unreviewed and unsupported in this workspace.
- Both CH340 serial adapters report the same generic serial identity. The
  stable by-path names identify PC sockets, not the arms themselves. Physically
  label the follower cable/socket as USB path `0:10`; swapping the two cables
  can otherwise direct ROS commands to the teacher arm.
- `starai_env.sh` forces `ROS_LOCALHOST_ONLY=1` and the dedicated local domain
  `ROS_DOMAIN_ID=42`, so another machine on the LAN cannot discover or command
  these nodes and all StarAI terminals share one graph. Keep all driver,
  MoveIt, and RViz terminals loaded through that script.
- Fake and real controllers expose the same absolute action names. Always use
  `run_viola_simulation.sh`, which checks for active hardware/control nodes,
  and close simulation before running the real driver. This startup check is
  still software and cannot eliminate every process-race condition.
- The ownership scan plus `TIOCEXCL` closes the normal LeRobot/ROS overlap
  paths, but it is still software—not a physical interlock. A privileged
  process, or one that already held a descriptor in the tiny startup race,
  remains outside this guarantee. Never use it in place of the power cutoff.
- The desk, cameras, pads, cubes, cables, and people are not collision objects.
  Add them to the planning scene before relying on collision avoidance.
- The two installed webcams are RGB cameras, not registered depth sensors for
  MoveIt's occupancy map. The bogus upstream Kinect configuration was removed.
  Humble may still print `No 3D sensor plugin(s) defined for octomap updates`;
  that message is expected here and means there is no depth-based collision
  coverage.
- Do not run `moveit_write_read.launch.py` or the upstream `topic_publisher`
  during initial testing. Those helpers can generate physical motion.
- On `Ctrl-C`, the driver requests damping and then closes serial. Damping is
  not a rigid brake; continue supporting the arm because it may sag.
- The feedback watchdog validates fresh angle replies. It does not monitor
  servo voltage, current, temperature, or device-status fault flags.
- If the serial bus or its power is physically lost, the damping request cannot
  reach the servos. Use the physical power cutoff and support the arm; software
  cannot guarantee a stop packet was received after disconnection.
- The gripper protocol uses the upstream fixed power value `8000`; it is not
  bounded by `GripperCommand.max_effort`. Positive effort-limit requests are
  therefore rejected; use the default value `0` only.
- The driver guard limits average position change divided by command duration.
  It does not model the servo's internal peak speed during the half-interval
  acceleration/deceleration profile, which is another reason to keep MoveIt
  scaling at the reviewed 5% setting.
- Non-zero trajectory header timestamps are rejected instead of being executed
  early. Custom path/goal tolerance arrays and action feedback are not
  implemented; the bridge uses its fixed measured goal tolerances.
- During a multi-waypoint segment the bridge checks feedback freshness, but it
  does not enforce per-waypoint measured path error; it verifies measured error
  at the final target. Use the 5% scaling and inspect planned paths carefully.
- Normal operation must never use `startup_mode:=zero` or
  `reset_multiturn:=true`.

## Common checks

```bash
# Load this workspace in each terminal
source /home/yz/starai_ws/starai_env.sh

# Confirm the corrected package name from the Chinese guide
ros2 pkg prefix viola_moveit_config

# Display safe driver arguments without opening hardware
ros2 launch viola_moveit_config driver.launch.py --show-args

# Display planning-only default
ros2 launch viola_moveit_config actual_robot_demo.launch.py --show-args

# Rebuild and retest the unchanged, exact reviewed source tree
cd /home/yz/starai_ws
./setup_starai.sh

# Run only the 55 safety-patch tests
colcon test --packages-select robo_driver viola_controller \
  --return-code-on-test-failure
colcon test-result --test-result-base build/robo_driver --verbose
colcon test-result --test-result-base build/viola_controller --verbose
```

The complete upstream lint suite contains pre-existing formatting and manifest
failures outside the patched Viola packages. The custom hardware-safety tests
are kept separate so those upstream lint defects do not hide a regression.
