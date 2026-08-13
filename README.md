# StarAI Viola — PC-A operations

This repository is the PC-A half of the eight-policy Viola benchmark. It owns
the reviewed dataset release, immutable cross-PC handoffs, disconnected policy
verification, replay and camera-only shadow runs, supervised robot execution,
evidence capture, and read-only report inspection. Training and canonical
ranking belong to Repo B.

The supported runtime is the `lerobot` Conda environment with Python 3.12,
LeRobot 0.6.1, and W&B 0.27.2 in online mode. Nothing in this README is a
readiness claim. A clean revision, accepted evidence, and the live gate must
pass at the time of use.

## Safety boundary

A policy candidate permits disconnected work only. It cannot move the robot.
Benchmark motion is possible only through `viola-ops policy execute`, after all
of these conditions pass:

- Repo A has locally accepted the exact policy candidate and Repo-B-produced
  `rollout_session`.
- The session permission is `live_session` and its blocker list is exactly
  empty.
- The current Repo-A commit and clean worktree match the reviewed executor.
- Calibration, camera mapping, reset protocol, seven-axis absolute/rate limits,
  and setup hashes match the reviewed session inputs.
- E-stop evidence belongs to the operator and is still less than 24 hours old.
- Required hold/shakedown evidence has been accepted for later phases.
- Canonical NAS bundles and receipts remain active and unchanged.
- A pre-hardware online W&B intent run finishes successfully.
- The named operator uses a real TTY to type the session-specific `ARM ...`
  challenge and explicitly starts every trial.

Repo A also owns one deliberately narrow, independent compatibility path for
the exact ACT checkpoint that was deployed before the two-PC benchmark
workflow existed: `viola-ops act run`. It does not need a Repo-B candidate,
receipt, or rollout session. It still refuses motion unless the checkpoint and
dataset hashes, versioned local setup, current clean revision, Python/LeRobot
environment, online W&B intent, current E-stop assertion, absolute/step limits,
and real-TTY `ARM ... ESTOP TESTED` and `START ...` actions all pass. Its output
is clearly marked `local_only`, never a Repo-B `READY` handoff or benchmark
result.

There is no `--yes`, environment-variable bypass, task override, or piped-input
arming path. The local ACT command accepts a path only so it can prove the bytes
equal the one reviewed deployment; arbitrary checkpoints are rejected. The old inference, calibration,
teleoperation, recording, and ROS hardware launchers are retired and exit before
opening a device. Software is not an E-stop.

## Install the Repo-A commands

From this repository:

```bash
conda activate lerobot
python --version
python -c 'import lerobot, wandb; print(lerobot.__version__, wandb.__version__)'
python -m pip install --no-deps -e .
viola-handoff --help
viola-ops --help
```

`--no-deps` preserves the pinned environment. Do not edit installed LeRobot or
plugin source. Repo-owned adapters use public LeRobot and FashionStar APIs.

## What each command does

### `viola-handoff seal|inspect|accept|ack`

- `seal` validates content, records online W&B lineage, writes a unique partial
  bundle, verifies it, and atomically publishes an immutable `READY` bundle.
  It is the low-level transport command; typed Repo-A operations are preferred.
  Generic sealing cannot create a live rollout session.
- `inspect` is read-only. It checks the manifest, contract hash, canonical JSON,
  inventory, `READY`, receipts, and optionally every external artifact byte.
- `accept` checksum-copies every artifact into receiver-local storage, verifies
  it again, finishes online W&B evidence, atomically accepts it, and appends the
  receiver's `accepted` receipt. Runtime code consumes only this local copy.
- `ack` appends an immutable `rejected` or `revoked` receipt. Acceptance uses
  `accept`, not `ack`.

Bundle source layout:

```text
/mnt/nas02/yz/starai/handoffs/v1/<kind>/<content-id>/
  manifest.json
  payload/
  READY.json
  receipts/
```

Large artifacts are inventory-bound external roots, not copies inside the
small `READY` bundle. Keep each producer root on storage mounted at the same
path on PCs A and B until PC B accepts the bundle and makes its verified local
copy. Repo-A commands therefore default producer material to
`/mnt/nas02/yz/starai/producer-materials/v1` and policy evidence to
`/mnt/nas02/yz/starai/evidence/v1`; do not replace those with a PC-A-only path
such as `~/.local` for a cross-PC handoff.

### `viola-ops dataset validate|release`

- `validate` checks the exact frozen 34-episode release, all 28,306 finite 7-D
  state/action rows, split and provenance, file inventory, and—unless
  `--numeric-only` is used—fully decodes every `front` and `up` frame.
- `release` always performs the full validation, records online W&B lineage,
  and seals a `dataset_release` bundle for PC B. It never trains a model and
  never changes the source dataset.

```bash
viola-ops dataset validate
viola-ops dataset release \
  --wandb-project starai-viola-policy-benchmark
```

### `viola-ops session-inputs produce`

Validates reviewed setup artifacts, current clean executor identity, exact
seven-axis limits, camera mapping, reset protocol, and current operator-owned
E-stop evidence. It seals a planning-only `session_inputs` bundle for Repo B.
It does not authorize motion; Repo B alone may turn accepted inputs into a typed
rollout session.

```bash
viola-ops session-inputs produce \
  --setup /path/to/reviewed-setup.json \
  --subject <setup-id> \
  --material-root /mnt/nas02/yz/starai/producer-materials/v1
```

The material root retains the signed `setup_record` external artifact where PC
B can checksum-copy it during acceptance.

### `viola-ops setup capture-frozen-state`

After an exact TTY confirmation, reads seven positions through the public SDK,
closes the serial connection, and only then publishes evidence to online W&B.
It has no motor-write or torque API. The result is an input for a camera-only
live soak. A fresh capture requires a path that does not exist; the command
atomically creates that directory before constructing the serial port. If
capture or serial cleanup fails before complete evidence exists, that new
directory is cleaned up. If W&B publication fails, the two immutable local
evidence files are retained for an upload-only retry.

```bash
viola-ops setup capture-frozen-state \
  --setup /path/to/reviewed-setup.json \
  --output-root /path/to/new/frozen-state-evidence \
  --operator <estop-owner> \
  --wandb-entity <entity>
```

Retry only the failed online publication, without another confirmation or any
serial/device access, by repeating the same command and adding
`--upload-only`:

```bash
viola-ops setup capture-frozen-state \
  --setup /path/to/reviewed-setup.json \
  --output-root /path/to/existing/frozen-state-evidence \
  --operator <estop-owner> \
  --wandb-entity <entity> \
  --upload-only
```

Upload-only recovery accepts exactly canonical, read-only `frozen_state.json`
and `frozen_state_capture.json`, plus an optional matching W&B receipt. It
revalidates the reviewed setup, operator, current clean PC-A identity, hashes,
timestamps, and deterministic W&B binding before reopening that same run. It
rejects partial, modified, writable, symlinked, or extra material and never
recaptures positions.

### `viola-ops policy verify|shadow|execute`

- `verify` loads an accepted candidate and its processors/dependencies from
  receiver-local bytes, verifies exact policy/config/camera/action contracts,
  produces a finite 7-D sample, then runs 20 warmups and 200 timed calls. No
  hardware package is constructed.
- `shadow --mode replay` runs at least 9,000 proposed actions over the complete
  signed held-out set and resets policy/queue state at episode boundaries.
- `shadow --mode live-soak` opens only both reviewed cameras, uses a signed
  frozen seven-axis state, records both videos, and proposes at least 9,000
  actions over at least 300 wall seconds. It never constructs a robot or motor.
- `execute` is the shared benchmark policy-motion entrypoint. It validates every gate before
  hardware-capable imports, records a finished online intent run, requires TTY
  operator actions, executes only the canonical hold → two 25%-rate shakedowns
  → ten scored trials, and retains complete action traces plus both encoded
  camera videos.

A post-frame aborted shakedown with complete, decodable evidence is sealed as
terminal `evidence_only` for Repo B and can never advance the phase gate. A stop
before the first retained frame remains local and online non-READY with blocker
`repo_b_unsafe_terminal_video_unavailable`. An aborted scored phase likewise
remains non-READY because Repo B `6fcf643` requires contradictory
shakedown-predecessor shapes for completed and unsafe scored outcomes.

All eight policy tokens share this path:

```text
act  diffusion  vqbet  smolvla  pi0  pi0_fast  pi05  groot
```

An accepted candidate is verified like this:

```bash
viola-ops policy verify \
  --bundle ~/.local/share/viola/handoffs/v1/policy_candidate/<id> \
  --wandb-entity <entity>
```

Both shadow modes default `--output-root` to the shared
`/mnt/nas02/yz/starai/evidence/v1`. Their `shadow_record` traces and videos must
remain readable from PC B until acceptance; a PC-A-local override is not a
portable handoff.

Every verification or shadow invocation creates a fresh immutable attempt
directory. Failed attempts remain incomplete and are never overwritten; pass
forward only the exact successful evidence path printed by the command.

### `viola-ops act run`

Runs the previously deployed ACT step-80,000 checkpoint directly from Repo A.
This is the independent inference command: it does not wait for Repo B and does
not accept any Repo-B handoff. The default duration is 10 seconds; the policy is
loaded from exact reviewed local bytes with a recorded 100-to-10 action-queue
deployment overlay through public LeRobot APIs. For compatibility with the
historical runner, each raw ACT proposal is first bounded to the reviewed
absolute range and then to the 25%-scaled per-step envelope; evidence records
both the raw proposal and the exact commanded target.

From an operator-controlled terminal, after physically testing the E-stop and
clearing the workspace:

```bash
conda activate lerobot
sg dialout -c 'CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online viola-ops act run'
```

PC A's current login has not inherited its configured `dialout` membership;
the wrapper above preserves the `lerobot` environment and real terminal while
granting access to the reviewed serial device. After a fresh login that already
shows `dialout` in `id`, the inner command is equivalent:
`CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online viola-ops act run`.
The command checks serial and camera access before asking for ARM or writing a
W&B intent.

The command prints the exact session-specific `ARM ... ESTOP TESTED` phrase and
then the exact trial-specific `START ...` phrase. These are physical operator
actions, not Repo-A or Repo-B approvals. Optional human-readable overrides are
available through `viola-ops act run --help`, including `--operator`, `--trial`,
and `--duration-seconds`; a checkpoint override still has to match the reviewed
step-80,000 inventory byte for byte.

If motion or final W&B publication fails, the attempt is never repeated in
place. The command retains a bound `LOCAL_ACT_FAILURE.json`, tries to publish a
deterministic failure run, and prints an upload-only recovery command. If the
network was unavailable, run that exact command, for example:

```bash
viola-ops act recover-failure \
  --attempt /mnt/nas02/yz/starai/evidence/v1/local-act/<session>/<trial>
```

Recovery revalidates immutable failure/intent/partial-motion evidence and the
clean revision, but never prompts, loads the model, or opens a device.

### `viola-ops report inspect`

Read-only validation of an accepted Repo-B `report` bundle. It checks complete
lineage, exact eight-policy ordering, recomputed metrics, CSV/Markdown/JSON
agreement, hashes, receipts, and the finished W&B report run. It prints terminal
outcomes; it does not rank, rewrite, publish, or declare the robot ready.

The merged Repo-B implementation still hashes resolved checkout paths into its
report configuration identity. Repo A therefore accepts only explicitly
reviewed Repo-B producer commits and path-bound digests, while also requiring
that digest to match exactly across the handoff lineage, report core,
content-derived W&B run, and rendered evidence. Inspection labels it as
nonportable and never treats it as readiness proof; Repo B should eventually
replace it with one shared path-independent identity.

```bash
viola-ops report inspect \
  --bundle ~/.local/share/viola/handoffs/v1/report/<id>
```

## Live rollout commands after the evidence chain exists

First activate the environment so the interactive TTY is preserved:

```bash
conda activate lerobot
export WANDB_MODE=online
viola-ops policy execute \
  --session ~/.local/share/viola/handoffs/v1/rollout_session/<session-id> \
  --candidate ~/.local/share/viola/handoffs/v1/policy_candidate/<candidate-id> \
  --phase hold \
  --trial commissioning-hold \
  --evidence-root /mnt/nas02/yz/starai/evidence/v1/rollout \
  --handoff-root /mnt/nas02/yz/starai/handoffs/v1 \
  --wandb-entity <entity>
```

That first command is intentionally an observation-only hold commissioning
step, not policy inference. Startup reads and validates the current pose and
connects both cameras without a motor
write; its evidence therefore records `motion: false`. It does not accept a
checkpoint path, duration, task, or speed override. After Repo B accepts the
hold evidence, the first command that actually runs policy inference is:

```bash
viola-ops policy execute \
  --session ~/.local/share/viola/handoffs/v1/rollout_session/<session-id> \
  --candidate ~/.local/share/viola/handoffs/v1/policy_candidate/<candidate-id> \
  --phase shakedown --trial supervised-shakedown \
  --prior-hold ~/.local/share/viola/handoffs/v1/rollout_evidence/<hold-id> \
  --evidence-root /mnt/nas02/yz/starai/evidence/v1/rollout \
  --handoff-root /mnt/nas02/yz/starai/handoffs/v1 \
  --wandb-entity <entity>
```

After Repo B accepts the two shakedowns, run `--phase scored` with both
`--prior-hold` and `--prior-shakedown`. Each phase requires a fresh operator
challenge, and each physical trial requires an explicit `START ...` action.

ACT is the infrastructure-clearance policy. A non-ACT live session is invalid
until Repo B binds either a scored, accepted ACT rollout as
`shared_infrastructure_proven`, or a reviewed `policy_specific_act_blocker`
attestation. The latter is allowed only when it proves that the ACT failure is
policy-specific and explicitly excludes every shared hardware, control,
safety, evidence, and W&B component. Repo A verifies the attached outcome and
attestation bytes before it can issue a permit.

If an accepted blocker-free rollout session does not yet exist, no shared
eight-policy benchmark inference may move the robot. The separately documented
local ACT compatibility run remains local-only and cannot substitute for that
benchmark evidence chain.

## Disconnected verification

```bash
conda activate lerobot
python -m pytest -q
git diff --check
```

The disconnected test matrix covers the shared transport, dataset and setup
contracts, every policy token, replay and fake-camera live soak, motion-gate
truth tables, fake public hardware, evidence backpressure, phase ordering,
report inspection, and refusal by every retired launcher. Fixture coverage is
not a substitute for loading the eight real accepted Repo-B candidates when
training and handoff are complete.

See [the command reference](docs/COMMAND_REFERENCE.md) and
[safety design](docs/SAFETY.md) for a shorter operator checklist.
