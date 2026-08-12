# New-PC setup, ports, and permissions

## 1. Install the operation environment

From a fresh clone:

```bash
./scripts/bootstrap_new_pc.sh
sudo usermod -aG dialout,video "$USER"
```

Log out completely and back in. Do not use `chmod 777`, run robot programs as
root, or add broad persistent udev permissions.

The bootstrap creates a Python 3.12 Conda environment named `lerobot` and
installs the pinned operation/inference stack from `requirements-validated.txt`.
It does not install training code or clone the LeRobot source tree.

## 2. Identify the arms

The identical CH340 adapters do not have unique serial identities. Their
`/dev/ttyUSB*` numbers can swap. With motor power disabled if possible:

```bash
ls -l /dev/serial/by-path/
```

Unplug only the Violin leader and note which path disappears. Reconnect it to
the same socket, then repeat for the Viola follower. Record the stable physical
paths in `config/operation.env`:

```bash
VIOLA_ROBOT_PORT=/dev/serial/by-path/REPLACE_FOLLOWER_PATH
VIOLA_TELEOP_PORT=/dev/serial/by-path/REPLACE_LEADER_PATH
```

## 3. Identify the cameras

The Logitech cameras have distinct serial IDs, so prefer `/dev/v4l/by-id`:

```bash
ls -l /dev/v4l/by-id/
conda run --no-capture-output -n lerobot \
  python scripts/dual_camera_view.py --devices \
  /dev/v4l/by-id/REPLACE_FRONT-video-index0 \
  /dev/v4l/by-id/REPLACE_UP-video-index0
```

Visually confirm the first view is `front` and the second is `up`, then store
both paths in `config/operation.env`. Use `video-index0`, not `video-index1`.

## 4. Restore calibration and checkpoint

The tracked calibration snapshots are only for this exact physical Viola and
Violin pair:

```bash
./scripts/install_calibrations.sh --check
./scripts/install_calibrations.sh --install   # only when target files are absent
```

Transfer the selected ACT checkpoint:

```bash
./scripts/sync_policy.sh
```

If NAS is unavailable, copy the complete `pretrained_model` directory to the
new machine, set `VIOLA_POLICY_SOURCE`, and rerun the script for verification.

## 5. Verify without opening hardware

```bash
conda activate lerobot
./scripts/preflight.sh
python -m pytest -q
```

The preflight checks stable paths, permissions, calibration hashes, model hash,
package versions, and imports. It does not open serial ports or cameras.

## Validated original mapping

| Role | Stable path on the validated workstation |
|---|---|
| Viola follower | `/dev/serial/by-path/pci-0000:00:14.0-usb-0:10:1.0-port0` |
| Violin leader | `/dev/serial/by-path/pci-0000:00:14.0-usb-0:11:1.0-port0` |
| Front camera | `/dev/v4l/by-id/usb-046d_0825_543F8BC0-video-index0` |
| Up camera | `/dev/v4l/by-id/usb-046d_0825_A8E49440-video-index0` |

Arm `by-path` values normally change on a new PC. Camera `by-id` values should
follow the devices but must still be visually verified.
