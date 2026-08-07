# Copy/paste command reference

These commands are for the validated current workstation. Read the linked guide
before first use; this page is the short repeat-use reference.

## Read-only robot preflight

```bash
cd /home/yz/lerobot/starai-viola-lerobot-ops
sg dialout -c 'exec ./scripts/preflight.sh'
```

## Preview front and up cameras

```bash
/home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/dual_camera_view.py
```

Press Q or Esc to close both windows before another camera command.

## Record a five-minute two-camera video

```bash
/home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/dual_camera_record.py \
  --duration 300 --countdown 3 \
  --output-dir /home/yz/lerobot/recordings
```

## Keep-current-pose teleoperation with cameras

```bash
sg dialout -c 'exec /home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/teleoperate_keep_pose.py \
  --robot.type=lerobot_robot_viola \
  --robot.port=/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0 \
  --robot.id=my_awesome_staraiviola_arm \
  --robot.disable_torque_on_disconnect=true \
  --robot.use_degrees=false \
  --robot.cameras="{\"front\":{\"type\":\"opencv\",\"index_or_path\":\"/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\"},\"up\":{\"type\":\"opencv\",\"index_or_path\":\"/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0\",\"width\":640,\"height\":480,\"fps\":30,\"fourcc\":\"MJPG\"}}" \
  --teleop.type=lerobot_teleoperator_violin \
  --teleop.port=/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0 \
  --teleop.id=my_awesome_staraiviolin_arm \
  --teleop.use_degrees=false \
  --fps=30 --max_step=3.0 --display_data=true'
```

## Check and record left-to-right episodes

```bash
sg dialout -c 'exec \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/record_left_to_right_episode.sh \
  --check'
```

```bash
sg dialout -c 'exec \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/record_left_to_right_episode.sh'
```

Recording starts automatically. Right accepts the episode; after encoding and
cube reset, Right skips the remaining reset timer. Left discards/re-records. Esc
during recording discards the partial episode; Esc during reset stops after
preserving the accepted episode.

## Check calibration snapshots

```bash
cd /home/yz/lerobot/starai-viola-lerobot-ops
./scripts/install_calibrations.sh --check
```

## Validate right-to-left filtering assumptions

```bash
/home/yz/anaconda3/envs/lerobot/bin/python \
  /home/yz/lerobot/starai-viola-lerobot-ops/scripts/filtering/build_right_to_left_training_dataset.py \
  --check
```

## ROS 2 / MoveIt for the Viola follower

One-time installation or a pinned-core rebuild (does not open serial):

```bash
cd /home/yz/lerobot/starai-viola-lerobot-ops
./ros2/setup_starai.sh
```

Read-only preflight before every hardware session:

```bash
sg dialout -c 'exec /home/yz/starai_ws/preflight.sh'
```

Start guarded fake-hardware simulation after it confirms that no real control
stack is active:

```bash
/home/yz/starai_ws/run_viola_simulation.sh
```

Close simulation and all LeRobot processes before hardware use. Start the
patched hold-current driver in terminal 1. Physical commissioning has not yet
been performed; for the first run, follow the supervised bring-up in the full
guide instead of treating this short reference as a commissioning procedure:

```bash
sg dialout -c 'exec /home/yz/starai_ws/run_viola_driver.sh'
```

In terminal 2, load the same isolated ROS graph, verify all seven measured
joints, and start MoveIt with execution disabled:

```bash
source /home/yz/starai_ws/starai_env.sh
ros2 topic echo /joint_states --once
ros2 launch viola_moveit_config actual_robot_demo.launch.py \
  allow_trajectory_execution:=false
```

Do not enable execution until the physical arm and RViz agree. The staged
hardware procedure and limitations are in [ROS 2 / MoveIt](ROS2_MOVEIT.md).

## Check and start ACT on the A100

```bash
conda activate lerobot-a100
cd ~/starai-viola-act-training
./train_act.sh --check
./train_act.sh
```

Detailed procedures: [Safety](SAFETY.md), [Teleoperation](TELEOPERATION.md),
[Episode recording](EPISODE_RECORDING.md), and
[Troubleshooting](TROUBLESHOOTING.md), plus the separate
[ROS 2 / MoveIt workflow](ROS2_MOVEIT.md).
