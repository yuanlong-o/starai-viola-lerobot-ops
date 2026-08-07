# Disabled historical launcher

`record_right_to_left_episode.sh` is preserved only to show how the completed
right-to-left source data was recorded. It exits immediately and is not an
active command.

The source dataset
`bourn117/viola_cubes_right_to_left_keep_pose_v3_clean` is frozen at 78
episodes. Appending to it would invalidate the recorded filtering provenance.
If more right-to-left demonstrations are needed, create a new dataset ID and a
new launcher; never remove the guard from this archived file.
