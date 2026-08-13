# Safety design

Software cannot replace the physical E-stop or an attentive operator. This
repository therefore treats every missing, malformed, stale, or inconsistent
input as a blocker.

## Before hardware can exist

The shared benchmark path, `viola-ops policy execute`, checks, in order:

1. locally accepted Repo-B candidate and rollout session;
2. `live_session` permission and exactly `blockers: []`;
3. exact policy, task, queue, setup, session-input, and lineage bindings;
4. active, unchanged canonical NAS bundles and receipts;
5. current clean reviewed Repo-A commit, Python 3.12, LeRobot 0.6.1, `lerobot`
   Conda environment, W&B 0.27.2, and online mode;
6. reviewed calibration, two-camera mapping, reset protocol, seven keyed
   absolute limits, per-step limits, and executor entrypoint hash;
7. passed operator-owned E-stop evidence less than 24 hours old at rollout;
8. accepted hold/shakedown predecessor evidence when the phase needs it;
9. an exact session/phase/trial `ARM ...` phrase from a real `/dev/tty`;
10. a finished online W&B intent run.

Only then are hardware-capable modules imported. The motion permit is an
unforgeable in-process object tied to one session, phase, and challenge.

The independent `viola-ops act run` path replaces steps 1–4 and 8 with a
Repo-A-local binding to the exact historical ACT checkpoint, frozen dataset,
and versioned `config/local_act_setup.json`. It performs the same environment,
setup, E-stop, TTY, online-intent, one-use-permit, execution-lease, and boundary
revalidation checks. It imports hardware only after those checks pass and marks
all output local-only rather than creating a benchmark `READY` bundle.

## During execution

- Connection uses the reviewed public FashionStar interface to read and check
  the current pose, then connects both cameras. It performs zero motor writes;
  hold commissioning is observation-only and is recorded as `motion: false`.
- Every action must name exactly seven joints and contain finite values.
- An action that is out of bounds or would need clamping is rejected; it is not
  silently corrected.
- Shakedowns use 25% of reviewed per-step limits. Scored trials use the reviewed
  limits without a runtime override.
- Independent ACT inference likewise uses 25% of the reviewed per-step limits
  and one bounded trial (10 seconds by default, never more than 60 seconds).
  Its historical compatibility transform retains the raw proposal, bounds to
  absolute then per-step limits, and records the exact commanded action.
- Stale camera data, one missed control deadline, feedback loss, malformed
  action, limit failure, evidence-queue saturation, collision, intervention, or
  operator stop aborts before the current write whenever possible.
- Evidence capacity is reserved before a motor write. Proposed, sent, and
  feedback actions plus timings, freshness, hashes, and videos are retained.
- If the public SDK write call raises, its physical outcome is unknowable. The
  reserved frames and an immutable `write_outcome: unknown` attempt receipt are
  retained, `sent_action` stays null, and execution aborts. Failures proven to
  occur before that SDK call release the unused reservation.
- Network and ordinary filesystem work stay outside the 30 Hz loop.
- Torque is retained on disconnect; the operator owns physical safeing.

## Phase order

The only sequence is hold-only commissioning, exactly two unscored 25%-rate
shakedowns, then ten preregistered scored trials. Repo B must accept and validate
the completed prior phase before Repo A can advance.

## Explicitly unsupported paths

Direct calibration, leader/follower teleoperation, demonstration recording, the
old ACT launcher, and the ROS hardware driver have no live-session authorization
contract. Their legacy entrypoints exit before checking or opening devices.
Camera preview/recording utilities remain motor-inert but do not provide policy
or rollout evidence.

## Known cross-repository evidence limitation

Repo B can consume rich terminal evidence for a post-frame aborted shakedown.
Repo A therefore preserves its original safety trace, derives Repo B's reviewed
unsafe view, fully decodes its retained videos, and seals it with
`evidence_only` permission when every consumer invariant passes. Its `READY`
marker means only that immutable evidence is ready for transfer; it can never
satisfy the completed-predecessor gate for more motion. A terminal before the
first retained camera frame is recorded locally and online as non-READY with
blocker `repo_b_unsafe_terminal_video_unavailable`. Any other projection or
decode failure raises without being relabeled or fabricating evidence.

Repo B `6fcf643` cannot consume the equivalent scored-phase abort reliably: its
unsafe path requires a rich completed-shakedown predecessor, while its normal
scored path requires the same predecessor in a mutually exclusive compact
shape. Repo A keeps an unsafe scored result locally and online as non-READY
evidence, marked `repo_b_unsafe_scored_predecessor_schema_mismatch`, instead of
issuing a bundle that Repo B would reject. This remains a readiness blocker
until Repo B versions one predecessor schema for both paths.

Repo B also currently derives its benchmark-report configuration identity from
resolved absolute checkout paths. Identical Repo-B source trees therefore
produce different report IDs when checked out in different directories. Repo A
requires exact internal agreement among the accepted handoff lineage, report
core, content-derived W&B run, and rendered evidence, and visibly labels the
digest as nonportable. That inspection cannot independently establish the
approved benchmark configuration or readiness until Repo B publishes one
path-independent configuration identity.
