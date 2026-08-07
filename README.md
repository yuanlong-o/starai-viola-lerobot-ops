# StarAI Viola + LeRobot operations

This private repository is the operating manual and tested support code for the
current StarAI setup:

- **Follower/robot:** StarAI Viola
- **Teacher/leader/teleoperator:** StarAI Violin
- **Observations:** front and overhead Logitech cameras
- **Task:** move the blue cube first and the red cube second between the two desk regions
- **LeRobot:** 0.4.2 at commit `d9e74a9d374a8f26582ad326c699740a227b483c`
- **StarAI plugins:** Viola, Violin, and motor packages at 0.0.4

The normal commands in this repository preserve the arms' measured startup
poses. Do not substitute stock `lerobot-teleoperate` or `lerobot-record`: the
installed StarAI 0.0.4 plugins command a fixed pose during connection and can
make an arm jump.

> Before every motor command, clear the workspace, identify teacher and
> follower, keep a hand near power/torque cutoff, and run the read-only
> preflight. The keep-pose patch and `max_step` limit are software protections,
> not a safety-rated emergency stop.

## Start here on the current workstation

```bash
cd /home/yz/lerobot/starai-viola-lerobot-ops

sg dialout -c 'exec ./scripts/preflight.sh'
```

That command checks paths, permissions, software versions, and calibration
hashes. It does **not** open a camera, serial port, motor, or dataset.

Then choose one documented workflow:

| Goal | Documentation |
|---|---|
| Repeat a known workflow quickly | [Copy/paste command reference](docs/COMMAND_REFERENCE.md) |
| Operate the arms and keep the current pose at startup | [Teleoperation](docs/TELEOPERATION.md) |
| Preview both cameras or record a five-minute task video | [Cameras and five-minute video](docs/CAMERA_VIDEO.md) |
| Record left-to-right LeRobot episodes | [Episode recording](docs/EPISODE_RECORDING.md) |
| Understand or restore calibration | [Calibration](docs/CALIBRATION.md) |
| Identify USB ports or move to another PC | [Setup, ports, and permissions](docs/SETUP_AND_PORTS.md) |
| Inspect, reject, or rebuild demonstrations | [Datasets and filtering](docs/DATASETS_AND_FILTERING.md) |
| Decide how much data to collect | [Data collection policy](docs/DATA_COLLECTION_POLICY.md) |
| Resolve camera, Rerun, permission, or interruption errors | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Train ACT on the A100 machine | [Training handoff](docs/TRAINING_HANDOFF.md) |

Read [Safety](docs/SAFETY.md) before calibration or first use on a new machine.

## What is tracked

This repository tracks the custom keep-pose wrappers, the active left-to-right
launcher, camera tools, tests, exact documentation, two arm-specific calibration
snapshots, and right-to-left filtering provenance.

It intentionally does not track raw datasets, recordings, logs, checkpoints,
credentials, installed third-party plugin source, the unsafe stale
`operation.yaml`, or the superseded weak-machine ACT launcher. Training code is
maintained separately in the private
[`starai-viola-act-training`](https://github.com/yuanlong-o/starai-viola-act-training)
repository.

## Repository layout

```text
calibration/                 known calibration snapshots for this exact arm pair
docs/                        operating procedures and copy/paste commands
provenance/                  immutable right-to-left audit records
scripts/                     active robot, camera, and preflight tools
scripts/filtering/           task-specific canonical dataset builder
tests/                       no-hardware regression tests
archive/                     disabled historical launcher; never run for recording
```
