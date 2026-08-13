# Calibration status

Interactive calibration is not authorized by the live-session contract and the
legacy launcher is disabled before device access. Do not use a stock LeRobot or
plugin calibration command as a workaround: the installed 0.0.4 plugin can
command its fixed startup pose during connection.

Tracked snapshots may be inspected or copied for the same physical arm pair
with `scripts/install_calibrations.sh`, but copying a file is not a review and
does not authorize motion. A production setup must bind the exact calibration
bytes and hash inside reviewed `session_inputs`, then inside the Repo-B rollout
session.

If recalibration is needed, first extend the shared safety contract with a
separate, reviewed calibration authorization flow. Until then it is an external
blocker, not an operator command in this repository.
