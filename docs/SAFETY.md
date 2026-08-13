# Safety design

Software cannot replace the physical E-stop or an attentive operator. This
repository therefore treats every missing, malformed, stale, or inconsistent
input as a blocker.

## Before hardware can exist

`viola-ops policy execute` checks, in order:

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

## During execution

- Connection uses the reviewed public FashionStar interface and commands only
  the freshly measured current pose at startup.
- Every action must name exactly seven joints and contain finite values.
- An action that is out of bounds or would need clamping is rejected; it is not
  silently corrected.
- Shakedowns use 25% of reviewed per-step limits. Scored trials use the reviewed
  limits without a runtime override.
- Stale camera data, one missed control deadline, feedback loss, malformed
  action, limit failure, evidence-queue saturation, collision, intervention, or
  operator stop aborts before the current write whenever possible.
- Evidence capacity is reserved before a motor write. Proposed, sent, and
  feedback actions plus timings, freshness, hashes, and videos are retained.
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

## Known cross-repository evidence blocker

The merged Repo-B contract currently uses incompatible completed and unsafe
trace shapes. Repo A preserves the richer safety trace and derives Repo B's
compact completed view, but it will not mint `READY` unsafe evidence until both
repositories share one schema that can prove the required safety facts. This
blocks a readiness claim; it is not relaxed at runtime.
