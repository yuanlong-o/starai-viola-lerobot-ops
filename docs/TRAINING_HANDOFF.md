# ACT training handoff to the A100

Training is intentionally separated from robot operation. The weak workstation
should record, audit, and copy data; the A100 should train from the verified NAS
copy.

## Canonical resources

- Private training repository:
  <https://github.com/yuanlong-o/starai-viola-act-training>
- Git commit currently prepared: `e71ea6c164116f61381489dd0d498241fd791a2d`
- NAS Git fallback:
  `/mnt/nas02/yz/starai/repos/starai-viola-act-training.git`
- Verified dataset:
  `/mnt/nas02/yz/starai/datasets/bourn117/viola_cubes_right_to_left_blue_then_red_train_v1`
- Dataset size: 34 episodes, 28,306 frames, two 640×480 camera streams,
  943.53 seconds; approximately 346 MiB on NAS
- Output root:
  `/mnt/nas02/yz/starai/outputs/act_viola_right_to_left_blue_then_red_v1`

The dataset is not stored in GitHub. The training repository contains hashes
and a full preflight that validates every expected file, video stream, Parquet
structure, sample decoding, software version, CUDA availability, and A100
capability.

## Fetch on the A100

```bash
gh auth status || gh auth login
gh repo clone yuanlong-o/starai-viola-act-training \
  ~/starai-viola-act-training
cd ~/starai-viola-act-training
```

NAS fallback:

```bash
git clone /mnt/nas02/yz/starai/repos/starai-viola-act-training.git \
  ~/starai-viola-act-training
cd ~/starai-viola-act-training
```

## Check before using GPU time

After preparing the `lerobot-a100` conda environment described in the training
repository README:

```bash
conda activate lerobot-a100
cd ~/starai-viola-act-training
./train_act.sh --check
```

The check must pass before training.

## Train or resume

```bash
conda activate lerobot-a100
cd ~/starai-viola-act-training
./train_act.sh
```

Resume an interrupted run:

```bash
./train_act.sh --resume
```

Defaults are one A100, batch size 16, BF16, 100,000 steps, eight data workers,
and a checkpoint every 10,000 steps. Run inside `tmux`, `screen`, Slurm, or
another persistent job environment.

ACT in this LeRobot version is not language-conditioned, so this policy covers
only right-to-left, blue-then-red. Train the opposite direction as a separate
ACT policy unless the selected policy architecture and training pipeline
explicitly support task conditioning.

π0.5 training is not implemented or validated in either repository. Do not
rename the ACT launcher or assume the same configuration applies to π0.5.
