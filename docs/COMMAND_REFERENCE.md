# Viola command reference

Run from the Repo-A root in the `lerobot` Conda environment. Evidence-producing
commands require authenticated online W&B. Use each command's `--help` for its
full path options.

## Immutable handoffs

```bash
viola-handoff inspect <bundle>
viola-handoff accept <bundle>
viola-handoff ack --status rejected <bundle>
viola-handoff ack --status revoked <bundle>
```

`inspect` is read-only. `accept` creates a receiver-local verified artifact
copy and appends an accepted receipt. `ack` never accepts; it records rejection
or revocation. Prefer typed `viola-ops` producers over raw `seal`.

External artifacts are not embedded in the small handoff payload. Their source
roots must stay on shared storage mounted at the same path on PCs A and B until
PC B accepts and checksum-copies them locally. Repo-A producer material defaults
to `/mnt/nas02/yz/starai/producer-materials/v1`; policy evidence defaults to
`/mnt/nas02/yz/starai/evidence/v1`.

## Dataset

```bash
viola-ops dataset validate
viola-ops dataset release --wandb-project starai-viola-policy-benchmark
```

`--numeric-only` is diagnostic and is insufficient for release.

## Session inputs and frozen state

```bash
viola-ops session-inputs produce \
  --setup /path/to/reviewed-setup.json --subject <setup-id> \
  --material-root /mnt/nas02/yz/starai/producer-materials/v1

viola-ops setup capture-frozen-state \
  --setup /path/to/reviewed-setup.json \
  --output-root /path/to/new/evidence \
  --operator <estop-owner> --wandb-entity <entity>
```

Session inputs are planning-only. Frozen-state capture reads positions and
sends no motor command.

## Policy verification and shadow

```bash
viola-ops policy verify \
  --bundle ~/.local/share/viola/handoffs/v1/policy_candidate/<id> \
  --wandb-entity <entity>

viola-ops policy shadow \
  --mode replay \
  --bundle ~/.local/share/viola/handoffs/v1/policy_candidate/<id> \
  --verification /path/to/verification.json \
  --output-root /mnt/nas02/yz/starai/evidence/v1 \
  --wandb-entity <entity>

viola-ops policy shadow \
  --mode live-soak \
  --bundle ~/.local/share/viola/handoffs/v1/policy_candidate/<id> \
  --verification /path/to/verification.json \
  --setup /path/to/reviewed-setup.json \
  --frozen-state /path/to/frozen-state-evidence \
  --output-root /mnt/nas02/yz/starai/evidence/v1 \
  --wandb-entity <entity>
```

Verification and replay are hardware-inert. Live soak is camera-only; it has no
robot or motor construction path. Both modes seal an external `shadow_record`;
its output root must remain visible to PC B until acceptance.
Each invocation creates a fresh immutable attempt directory. Failed attempts
remain incomplete; use only the exact successful path printed by the command.

## Supervised inference

```bash
viola-ops policy execute \
  --session ~/.local/share/viola/handoffs/v1/rollout_session/<session-id> \
  --candidate ~/.local/share/viola/handoffs/v1/policy_candidate/<candidate-id> \
  --phase hold --trial commissioning-hold \
  --evidence-root /mnt/nas02/yz/starai/evidence/v1/rollout \
  --wandb-entity <entity>
```

The hold phase is observation-only. Connection reads and validates the current
pose and connects both cameras without sending a motor command, so its evidence
records `motion: false`.

Later phases additionally require accepted predecessor evidence. Run
`viola-ops policy execute --help`. There is no valid motion command before a
blocker-free accepted live session, reviewed setup/current E-stop evidence,
online intent receipt, and exact interactive operator action all pass.
ACT is checked first. Every non-ACT session requires Repo B to bind either a
scored, accepted ACT rollout as `shared_infrastructure_proven`, or a reviewed
`policy_specific_act_blocker` attestation proving that the ACT failure does not
implicate any shared hardware, control, safety, evidence, and W&B component.
Repo A verifies the attached outcome and attestation bytes before authorization.

The rollout evidence root is required and must be shared NAS storage outside
the Git worktree. Shakedown and scored bundles inventory-bind their motion
traces and videos there so PC B can copy and verify them during acceptance.

## Report

```bash
viola-ops report inspect \
  --bundle ~/.local/share/viola/handoffs/v1/report/<id>
```

## Disconnected tests

```bash
python -m pytest -q
git diff --check
```

## Retired commands

The old inference, calibration, teleoperation, episode-recording, and ROS
hardware launchers deliberately exit with code 64 before device access. They
must not be restored as alternate motion paths.
