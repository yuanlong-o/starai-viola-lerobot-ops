# Viola command reference

Run from the Repo-A root in the `lerobot` Conda environment. Evidence-producing
commands require authenticated online W&B. Use each command's `--help` for its
full path options.

## Immutable handoffs

```bash
viola-handoff seal --kind <non-session-kind> ...
viola-handoff inspect <bundle>
viola-handoff accept <bundle>
viola-handoff ack --status rejected <bundle>
viola-handoff ack --status revoked <bundle>
```

`inspect` is read-only. `accept` creates a receiver-local verified artifact
copy and appends an accepted receipt. `ack` never accepts; it records rejection
or revocation. `seal` is the low-level content-addressed producer used by typed
operations; prefer `viola-ops` producers. Generic sealing can never create a
`rollout_session`, even though the byte-identical shared Repo-B CLI currently
lists that kind in its generated choices. Only Repo B's typed validated session
producer may grant `live_session` permission.

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

# Only after W&B upload failed and immutable capture files remain:
viola-ops setup capture-frozen-state \
  --setup /path/to/reviewed-setup.json \
  --output-root /path/to/existing/evidence \
  --operator <estop-owner> --wandb-entity <entity> \
  --upload-only
```

Session inputs are planning-only. Frozen-state capture reads positions and
sends no motor command. Fresh capture atomically reserves a new output
directory before opening serial hardware. `--upload-only` strictly validates
the canonical, read-only state/capture files and optional sync receipt, then
retries the same deterministic W&B run without confirmation, serial, cameras,
or motors. Partial, modified, writable, symlinked, or extra material is
rejected.

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

## Supervised rollout and inference

### Independent Repo-A ACT inference

The previously deployed ACT checkpoint has a separate Repo-A-owned path and
does not require a Repo-B candidate, receipt, or rollout session:

```bash
conda activate lerobot
sg dialout -c 'CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online /home/yz/anaconda3/envs/lerobot/bin/viola-ops act run'
```

The current PC-A login needs the `sg dialout` wrapper for the reviewed serial
device. A new login whose `id` already includes `dialout` can run the inner
`CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online viola-ops act run` command directly.
The editable installation resolves Repo A independently of the terminal's
current directory.
Serial and both camera paths are checked before ARM and before any online
intent is written.

Defaults bind `/home/yz/models/act_viola_val20_step080000`, the frozen
34-episode dataset, and `config/local_act_setup.json`. The default run is 10
seconds at the shared 30 Hz safety loop and 25% of the reviewed per-step limit.
The operator must first physically test the E-stop, then type the printed
`ARM ... ESTOP TESTED` and `START ...` phrases on a real TTY. No Repo-B action
is involved. Evidence is local-only plus online W&B and is never presented as
a Repo-B `READY` bundle or benchmark result.

Use `viola-ops act run --help` for optional operator, trial, duration, path, and
evidence-root arguments. Alternate checkpoint paths are useful only for an
identical copy: different bytes are rejected.

Raw ACT outputs are recorded separately from the exact commanded actions. The
local compatibility transform clamps first to reviewed absolute limits, then
to the 25%-scaled per-step envelope; the shared benchmark path remains strict
and rejects rather than transforms out-of-envelope proposals.

If the command retains a failure after motion or an online outage, publish that
same terminal without repeating motion:

```bash
viola-ops act recover-failure --attempt <printed-attempt-directory>
```

This recovery is hardware-free and creates no `READY` handoff.

### Shared eight-policy benchmark rollout

```bash
viola-ops policy execute \
  --session ~/.local/share/viola/handoffs/v1/rollout_session/<session-id> \
  --candidate ~/.local/share/viola/handoffs/v1/policy_candidate/<candidate-id> \
  --phase hold --trial commissioning-hold \
  --evidence-root /mnt/nas02/yz/starai/evidence/v1/rollout \
  --wandb-entity <entity>
```

The hold phase is observation-only; it does not load or call the policy.
Connection reads and validates the current pose and connects both cameras
without sending a motor command, so its evidence records `motion: false`.

After Repo B accepts that hold evidence, actual policy inference begins with a
session-specific shakedown command:

```bash
viola-ops policy execute \
  --session ~/.local/share/viola/handoffs/v1/rollout_session/<session-id> \
  --candidate ~/.local/share/viola/handoffs/v1/policy_candidate/<candidate-id> \
  --phase shakedown --trial supervised-shakedown \
  --prior-hold ~/.local/share/viola/handoffs/v1/rollout_evidence/<hold-id> \
  --evidence-root /mnt/nas02/yz/starai/evidence/v1/rollout \
  --wandb-entity <entity>
```

Later shared benchmark phases additionally require accepted predecessor evidence. Run
`viola-ops policy execute --help`. There is no valid shared benchmark motion command before a
blocker-free accepted live session, reviewed setup/current E-stop evidence,
online intent receipt, and exact interactive operator action all pass.
Only a post-frame unsafe shakedown whose retained traces and videos pass the
full consumer contract produces terminal `evidence_only` for Repo B; it is
never a completed predecessor. A stop before the first retained frame remains
non-READY as `repo_b_unsafe_terminal_video_unavailable`. An unsafe scored result
remains non-READY with blocker
`repo_b_unsafe_scored_predecessor_schema_mismatch` until Repo B unifies its
mutually exclusive compact/rich predecessor schemas.
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

Inspection verifies the accepted bundle, rendered files, content-derived W&B
run, reviewed Repo-B producer revision and configuration digest, and all eight
outcomes. It labels Repo B's current checkout-path-dependent configuration
digest as nonportable; report inspection never authorizes motion.

## Disconnected tests

```bash
python -m pytest -q
git diff --check
```

## Retired commands

The old inference, calibration, teleoperation, episode-recording, and ROS
hardware launchers deliberately exit with code 64 before device access. They
must not be restored as alternate motion paths.
