# Two cameras and a five-minute task video

These tools are for previewing the task or recording a human-readable task
structure video. They do not create LeRobot action/state episodes.

Current roles:

- Front: Logitech serial `543F8BC0`
- Up/overhead: Logitech serial `A8E49440`
- Requested capture: MJPEG, 640×480, 30 fps

The cameras are sampled on one software clock, but they are not hardware
synchronized. The recorder writes two equal-length MP4 files without audio and
reports duplicated frames and capture failures at the end.

## Preview both cameras

```bash
./scripts/run_dual_camera_view.sh
```

Press Q or Esc in either window to close both. Close the viewer before any
teleoperation or episode-recording command, because one V4L2 device generally
cannot be owned by both programs.

## Record exactly five minutes

```bash
./scripts/run_dual_camera_record.sh 300
```

After the three-second countdown, demonstrate this loop for five minutes:

1. Both cubes begin together in one region.
2. Move the blue cube to the other region.
3. Move the red cube to the other region.
4. Move the blue cube back.
5. Move the red cube back.
6. Repeat while keeping both views unobstructed.

Q, Esc, or Ctrl+C stops early and finalizes both files. A normal run creates:

```text
/home/yz/lerobot/recordings/dual_camera_YYYYMMDD_HHMMSS/front.mp4
/home/yz/lerobot/recordings/dual_camera_YYYYMMDD_HHMMSS/up.mp4
```

## Verify the result before relying on it

Read the recorder's final summary. Both files should have the same frame count
and approximately 300 seconds duration. Then inspect the beginning, middle, and
end of both videos:

```bash
ffprobe -v error \
  -show_entries format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 \
  /home/yz/lerobot/recordings/dual_camera_YYYYMMDD_HHMMSS/front.mp4
```

Repeat for `up.mp4`. The current 640×480 views are adequate for a tabletop
proof-of-concept when the gripper, both cubes, and both destination regions stay
visible. Re-record if color/exposure hides cube identity, the arm blocks both
views, focus is soft, a camera moves, or dropped/duplicated frames are excessive.
