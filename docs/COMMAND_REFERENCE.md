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

## Check and start ACT on the A100

```bash
conda activate lerobot-a100
cd ~/starai-viola-act-training
./train_act.sh --check
./train_act.sh
```

Detailed procedures: [Safety](SAFETY.md), [Teleoperation](TELEOPERATION.md),
[Episode recording](EPISODE_RECORDING.md), and
[Troubleshooting](TROUBLESHOOTING.md).
