# Calibration

## What calibration is

For this StarAI setup, the persistent calibration state is stored in two JSON
files. Each file contains seven motor records with motor ID, drive mode, homing
offset, minimum range, and maximum range.

| Device | Installed file | Validated SHA-256 |
|---|---|---|
| Viola follower | `/home/yz/.cache/huggingface/lerobot/calibration/robots/starai_viola/my_awesome_staraiviola_arm.json` | `7e580ce32f1f4b9a37367d42ff1563edccd4d9130ea4e122febd0558517d30ac` |
| Violin teacher | `/home/yz/.cache/huggingface/lerobot/calibration/teleoperators/starai_violin/my_awesome_staraiviolin_arm.json` | `31d9cbe5471219df2087f4bd104dada86d3279965cee0666837c9dd880764fe7` |

The command IDs select these filenames. Changing `--robot.id` or `--teleop.id`
selects another calibration file and can make a valid calibration appear to be
missing.

The JSON files are the stored calibration state; there is no separate
calibration value inside the serial port or command. The result still depends on
the matching plugin, motor IDs, mechanical assembly, and physical arm. Never use
these files for a different arm pair, a rebuilt joint, changed motor IDs, or a
UR5.

Snapshots for this exact pair are tracked under `calibration/` in this private
repository. They contain motor ranges, not passwords or tokens.

## Check or restore the snapshots

From the operations repository:

```bash
cd /home/yz/lerobot/starai-viola-lerobot-ops

./scripts/install_calibrations.sh --check
```

Install only when the destination is missing:

```bash
./scripts/install_calibrations.sh --install
```

Replace differing files only after confirming this is the same physical arm
pair. The script creates timestamped backups before replacement:

```bash
./scripts/install_calibrations.sh --replace
./scripts/install_calibrations.sh --check
```

`HF_LEROBOT_CALIBRATION` can point to a different calibration root. The same
environment value must be present when running preflight, teleoperation,
recording, or calibration.

## Recalibrate only when necessary

Recalibration is needed after relevant mechanical/motor changes, a confirmed
bad range, or loss of the correct files. A shutdown, reboot, or ordinary USB
renumbering is not a reason to recalibrate.

> Warning: stock StarAI calibration connects through the plugin. With an
> existing calibration loaded, the 0.0.4 plugin may command its hardcoded pose
> before the prompts begin. Clear the workspace, support the arm, identify the
> exact port, and keep power cutoff available.

Teacher/Violin, from the repository root:

```bash
./scripts/run_viola_calibration.sh violin
```

Follower/Viola:

```bash
./scripts/run_viola_calibration.sh viola
```

After calibration, record the new checksums:

```bash
sha256sum \
  /home/yz/.cache/huggingface/lerobot/calibration/robots/starai_viola/my_awesome_staraiviola_arm.json \
  /home/yz/.cache/huggingface/lerobot/calibration/teleoperators/starai_violin/my_awesome_staraiviolin_arm.json
```

Do not append new-calibration episodes to an old dataset. Create a new dataset
version, update its launcher checksum guard, and inspect a pilot episode first.
