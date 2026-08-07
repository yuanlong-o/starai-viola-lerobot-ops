# Data collection policy

## Episode boundaries

Record one complete task per episode. For the current two directions:

- Right to left: start with both cubes on the right white pad; blue first, then
  red; finish with both on the left gray platform.
- Left to right: start with both cubes on the left gray platform; blue first,
  then red; finish with both on the right white pad.

Reset cubes only between episodes. A reset or search for a misplaced cube inside
an episode changes the demonstrated behavior and should be discarded.

Do not record an endless left-to-right/right-to-left video and later cut only
the camera files. A LeRobot episode couples two video timelines with robot state,
commanded action, timestamps, task metadata, and statistics. Safe post-cutting
requires a canonical rebuild of all modalities and new validation. Deliberate
episode boundaries during recording are simpler and produce cleaner supervision.

## Include or exclude

Include an episode only when:

- the first frames show the exact required start state;
- the task order matches its label;
- both cubes end fully supported in the target region;
- the trajectory is intentional and reasonably direct;
- both camera views remain usable;
- there is no manual intervention after recording begins.

Exclude or re-record when:

- a cube starts in the wrong place and is manually moved after recording starts;
- red moves before blue for a “blue then red” label;
- the final placement or last grasp is unfinished;
- a camera freezes, disconnects, moves, or becomes badly occluded;
- the operator pauses for a long time, searches, performs an unrelated motion,
  or has a poor failure recovery;
- the process is interrupted before deliberate acceptance.

Comparison of starting and ending frames is useful for screening, but it is not
enough by itself: order, mid-trajectory interventions, drops, and camera faults
require trajectory review.

## How many episodes

Episode count is meaningful only after filtering. For this narrow fixed-camera,
fixed-object task:

- 30–50 clean episodes per direction can support a proof-of-concept and reveal
  whether the learning pipeline works.
- About 75–100 clean episodes per direction is a more useful target for
  robustness to natural starting-pose and grasp variation.
- More data helps only when it is correctly labeled, visually usable, and not
  dominated by repeated mistakes or near-identical motion.

The existing 34 clean right-to-left episodes are sufficient to begin an ACT
baseline, but they are not a guarantee of robust deployment. Fifty recorded
episodes are not “enough” if filtering leaves only a small or inconsistent
subset. Track accepted count, task success, and variation—not just raw count.

For a π0.5-style vision-language-action fine-tune, this setup is a reasonable
narrow tabletop task because the language instruction, two camera views, and
seven-dimensional actions are available. The likely bottleneck is demonstration
quality and diversity rather than raw video length. Keep each direction as an
explicit task label, collect clean coverage, and evaluate on held-out start
configurations. This operations repository does not claim a validated π0.5
training stack; the provided A100 launcher is for ACT.

## Variation to collect deliberately

Vary within the intended deployment envelope:

- cube position and yaw inside the start region;
- safe teacher/follower startup poses;
- approach direction and grasp point;
- small lighting and background variation;
- placement position inside the destination region.

Keep camera mounts, task semantics, cube identity/order, calibration, and safety
limits fixed within one dataset version. If one of those changes materially,
create a new version and record the change in provenance.
