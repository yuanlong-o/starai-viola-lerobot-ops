# Right-to-left training dataset provenance

The final trainable dataset was rebuilt from the original recordings after a
strict audit of both camera views and the full trajectory.

- Source: `bourn117/viola_cubes_right_to_left_keep_pose_v3_clean`
- Source size: 78 episodes, 67,232 frames
- Final: `bourn117/viola_cubes_right_to_left_blue_then_red_train_v1`
- Final size: 34 episodes, 28,306 frames, 943.53 seconds at 30 Hz
- Task: move blue first, then red, from the right white pad to the left gray platform
- Source tree SHA-256: `9e8a3107df94371b910b382891d6cc1c8285cafb10ffd0a199ea8639856898b7`

The accepted original episode IDs, rejected IDs, per-episode rejection reasons,
old-to-new index mapping, numeric hashes, and video validation results are in
`filter_manifest.json` and `validation_report.json` beside this file.

The source was unchanged before and after rebuilding. The builder decoded the
selected episodes and wrote a canonical LeRobot dataset; it did not splice MP4
files by hand.
