#!/usr/bin/env python
"""Build and validate the strictly filtered right-to-left training dataset.

The source dataset is never modified. Selected episodes are decoded and written
through LeRobot's public recording API into a staging directory, validated, and
then atomically promoted to the final destination.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROJECT_ROOT = Path(os.environ.get("LEROBOT_PROJECT_DIR", "/home/yz/lerobot")).expanduser().resolve()
SOURCE_REPO_ID = "bourn117/viola_cubes_right_to_left_keep_pose_v3_clean"
SOURCE_ROOT = Path(
    os.environ.get(
        "VIOLA_RTL_SOURCE_ROOT",
        PROJECT_ROOT / "data/recordings/bourn117/viola_cubes_right_to_left_keep_pose_v3_clean",
    )
).expanduser().resolve()
TARGET_REPO_ID = "bourn117/viola_cubes_right_to_left_blue_then_red_train_v1"
TARGET_ROOT = Path(
    os.environ.get(
        "VIOLA_RTL_TARGET_ROOT",
        PROJECT_ROOT / "data/recordings/bourn117/viola_cubes_right_to_left_blue_then_red_train_v1",
    )
).expanduser().resolve()
DATASET_LOCK = Path(
    os.environ.get(
        "VIOLA_RTL_DATASET_LOCK",
        PROJECT_ROOT / "logs/episode_recording/right_to_left_dataset.lock",
    )
).expanduser().resolve()
TASK = (
    "Move the blue cube, then the red cube, from the white pad on the right "
    "to the gray platform on the left."
)

# Final result of a full, two-camera trajectory audit at 2 Hz or denser.
KEEP_ORIGINAL_EPISODES = [
    0,
    1,
    2,
    4,
    6,
    18,
    20,
    23,
    26,
    31,
    34,
    35,
    36,
    38,
    40,
    43,
    47,
    48,
    50,
    52,
    54,
    57,
    59,
    61,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
]

WRONG_ORDER_EPISODES = [
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    19,
    22,
    24,
    27,
    28,
    29,
    30,
    32,
    39,
    41,
    42,
    44,
    46,
    49,
    51,
    53,
    58,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
]
PRIOR_INVALID_EPISODES = [3, 5, 12, 16, 17, 21, 25, 33, 37, 45, 55, 56]
EXCESS_IDLE_EPISODES = [15]

EXPECTED_SOURCE_EPISODES = 78
EXPECTED_SOURCE_FRAMES = 67_232
EXPECTED_FPS = 30
EXPECTED_TARGET_EPISODES = 34
EXPECTED_TARGET_FRAMES = 28_306
VIDEO_KEYS = ["observation.images.front", "observation.images.up"]
NUMERIC_KEYS = ["action", "observation.state"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate source assumptions and print the planned output without building it.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an already-built final dataset and refresh its validation report.",
    )
    parser.add_argument(
        "--recover-staging",
        type=Path,
        metavar="PATH",
        help="Validate and atomically promote a complete staging dataset left by a failed validator.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tree_digest(root: Path) -> dict[str, Any]:
    """Return a deterministic content digest for every regular file below root."""
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
        total_bytes += size
        file_count += 1
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def validate_audit_partition() -> None:
    keep = KEEP_ORIGINAL_EPISODES
    require(keep == sorted(set(keep)), "Keep list must be sorted and unique")
    rejected = sorted(WRONG_ORDER_EPISODES + PRIOR_INVALID_EPISODES + EXCESS_IDLE_EPISODES)
    require(rejected == sorted(set(rejected)), "Reject groups overlap")
    require(
        sorted(keep + rejected) == list(range(EXPECTED_SOURCE_EPISODES)),
        "Keep and reject lists do not exactly partition source episodes 0..77",
    )
    require(len(keep) == EXPECTED_TARGET_EPISODES, "Unexpected accepted episode count")


def load_and_validate_source() -> LeRobotDataset:
    require(SOURCE_ROOT.is_dir(), f"Source dataset is missing: {SOURCE_ROOT}")
    source = LeRobotDataset(
        SOURCE_REPO_ID,
        root=SOURCE_ROOT,
        video_backend="torchcodec",
    )
    require(source.meta.total_episodes == EXPECTED_SOURCE_EPISODES, "Source episode count changed")
    require(source.meta.total_frames == EXPECTED_SOURCE_FRAMES, "Source frame count changed")
    require(source.meta.fps == EXPECTED_FPS, "Source FPS changed")
    require(source.meta.robot_type == "starai_viola", "Unexpected source robot type")
    require(source.meta.video_keys == VIDEO_KEYS, f"Unexpected video keys: {source.meta.video_keys}")
    require(tuple(source.meta.features["action"]["shape"]) == (7,), "Expected 7D actions")
    require(tuple(source.meta.features["observation.state"]["shape"]) == (7,), "Expected 7D state")
    for video_key in VIDEO_KEYS:
        require(
            tuple(source.meta.features[video_key]["shape"]) == (480, 640, 3),
            f"Unexpected shape for {video_key}",
        )
    selected_frames = sum(source.meta.episodes[i]["length"] for i in KEEP_ORIGINAL_EPISODES)
    require(selected_frames == EXPECTED_TARGET_FRAMES, "Selected source frame count changed")
    for episode_index in range(source.meta.total_episodes):
        require(source.meta.episodes[episode_index]["episode_index"] == episode_index, "Bad source metadata index")
    return source


def tensor_image_to_uint8_hwc(value: torch.Tensor, key: str) -> np.ndarray:
    require(isinstance(value, torch.Tensor), f"{key} is not a tensor")
    require(tuple(value.shape) == (3, 480, 640), f"Unexpected decoded shape for {key}: {value.shape}")
    require(torch.isfinite(value).all().item(), f"Non-finite pixels in {key}")
    require(value.min().item() >= 0.0 and value.max().item() <= 1.0, f"Pixel range error in {key}")
    image = value.detach().cpu().permute(1, 2, 0)
    image = torch.round(image * 255.0).clamp_(0, 255).to(torch.uint8).numpy()
    return np.ascontiguousarray(image)


def update_array_digest(digest: hashlib._Hash, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def make_staging_root() -> Path:
    suffix = f"{os.getpid()}-{uuid.uuid4().hex[:10]}"
    return TARGET_ROOT.parent / f".{TARGET_ROOT.name}.building-{suffix}"


def build_dataset(source: LeRobotDataset, staging_root: Path) -> dict[str, str]:
    features = copy.deepcopy(source.meta.features)
    destination = LeRobotDataset.create(
        repo_id=TARGET_REPO_ID,
        fps=EXPECTED_FPS,
        features=features,
        root=staging_root,
        robot_type=source.meta.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=8,
        video_backend="torchcodec",
        batch_encoding_size=1,
    )

    source_hashes = {key: hashlib.sha256() for key in NUMERIC_KEYS}
    total_written = 0
    try:
        for new_episode, old_episode in enumerate(KEEP_ORIGINAL_EPISODES):
            episode = source.meta.episodes[old_episode]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            require(stop - start == episode["length"], f"Source length mismatch for episode {old_episode}")
            description = f"original {old_episode:02d} -> train {new_episode:02d}"
            for global_index in tqdm(range(start, stop), desc=description, unit="frame"):
                item = source[global_index]
                require(int(item["episode_index"].item()) == old_episode, "Source episode boundary mismatch")
                require(item["task"] == TASK, f"Unexpected task string in source episode {old_episode}")

                action = item["action"].detach().cpu().numpy().astype(np.float32, copy=True)
                state = item["observation.state"].detach().cpu().numpy().astype(np.float32, copy=True)
                require(action.shape == (7,) and state.shape == (7,), "Bad action/state shape")
                require(np.isfinite(action).all() and np.isfinite(state).all(), "Non-finite trajectory value")
                update_array_digest(source_hashes["action"], action)
                update_array_digest(source_hashes["observation.state"], state)

                destination.add_frame(
                    {
                        "action": action,
                        "observation.state": state,
                        "observation.images.front": tensor_image_to_uint8_hwc(
                            item["observation.images.front"], "observation.images.front"
                        ),
                        "observation.images.up": tensor_image_to_uint8_hwc(
                            item["observation.images.up"], "observation.images.up"
                        ),
                        "task": TASK,
                    }
                )
                total_written += 1

            destination.save_episode()
            print(
                f"Saved train episode {new_episode}/{EXPECTED_TARGET_EPISODES - 1} "
                f"from original episode {old_episode} ({stop - start} frames)",
                flush=True,
            )
    finally:
        try:
            destination.finalize()
        finally:
            if destination.image_writer is not None:
                destination.stop_image_writer()

    require(total_written == EXPECTED_TARGET_FRAMES, "Builder wrote an unexpected frame count")
    return {key: value.hexdigest() for key, value in source_hashes.items()}


def load_data_table(root: Path) -> pa.Table:
    files = sorted((root / "data").rglob("*.parquet"))
    require(bool(files), "No data parquet files found")
    return pa.concat_tables([pq.read_table(path) for path in files])


def load_episode_table(root: Path) -> pa.Table:
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    require(bool(files), "No episode metadata parquet files found")
    return pa.concat_tables([pq.read_table(path) for path in files])


def fixed_list_to_numpy(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_pylist(), dtype=dtype)


def output_numeric_hashes(data: pa.Table) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in NUMERIC_KEYS:
        digest = hashlib.sha256()
        for row in fixed_list_to_numpy(data[key], np.float32):
            update_array_digest(digest, row)
        result[key] = digest.hexdigest()
    return result


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite metadata statistic at {path}")


def validate_video_files(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for video_key in VIDEO_KEYS:
        files = sorted((root / "videos" / video_key).rglob("*.mp4"))
        require(bool(files), f"No videos found for {video_key}")
        total_frames = 0
        file_reports = []
        for path in files:
            with av.open(str(path)) as container:
                require(len(container.streams.video) == 1, f"Expected one video stream in {path}")
                stream = container.streams.video[0]
                # PyAV reports the selected decoder implementation here (for
                # example ``libdav1d``), not necessarily the encoded format.
                # The MP4 sample entry is the stable way to assert AV1.
                decoder_name = stream.codec_context.name
                codec_tag = stream.codec_context.codec_tag
                pixel_format = stream.codec_context.format.name if stream.codec_context.format else None
                require(codec_tag == "av01", f"Unexpected codec tag {codec_tag} in {path}")
                require(stream.codec_context.width == 640, f"Unexpected width in {path}")
                require(stream.codec_context.height == 480, f"Unexpected height in {path}")
                require(pixel_format == "yuv420p", f"Unexpected pixel format {pixel_format} in {path}")
                require(stream.average_rate is not None, f"Missing frame rate in {path}")
                require(abs(float(stream.average_rate) - EXPECTED_FPS) < 1e-6, f"Unexpected FPS in {path}")
                timestamps: list[float] = []
                for frame in container.decode(stream):
                    require(frame.pts is not None, f"Missing frame PTS in {path}")
                    timestamps.append(float(frame.pts * frame.time_base))
                require(timestamps, f"Video has no decodable frames: {path}")
                differences = np.diff(np.asarray(timestamps, dtype=np.float64))
                require(np.all(differences > 0), f"Non-monotonic video timestamps in {path}")
                require(
                    np.allclose(differences, 1.0 / EXPECTED_FPS, atol=2e-6, rtol=0),
                    f"Irregular frame timestamps in {path}",
                )
                total_frames += len(timestamps)
                file_reports.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "codec": "av1",
                        "decoder": decoder_name,
                        "frames": len(timestamps),
                        "first_timestamp_s": timestamps[0],
                        "last_timestamp_s": timestamps[-1],
                    }
                )
        require(total_frames == EXPECTED_TARGET_FRAMES, f"Wrong decoded frame total for {video_key}")
        report[video_key] = {"total_frames": total_frames, "files": file_reports}
    return report


def validate_dataset(root: Path, expected_numeric_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    require(root.is_dir(), f"Dataset root does not exist: {root}")
    dataset = LeRobotDataset(TARGET_REPO_ID, root=root, video_backend="torchcodec")
    require(dataset.meta.total_episodes == EXPECTED_TARGET_EPISODES, "Wrong target episode count")
    require(dataset.meta.total_frames == EXPECTED_TARGET_FRAMES, "Wrong target frame count")
    require(dataset.meta.fps == EXPECTED_FPS, "Wrong target FPS")
    require(dataset.meta.robot_type == "starai_viola", "Wrong target robot type")
    require(dataset.meta.video_keys == VIDEO_KEYS, "Wrong target video keys")
    require(len(dataset.meta.tasks) == 1 and dataset.meta.tasks.index.tolist() == [TASK], "Wrong task table")

    source_dataset = LeRobotDataset(
        SOURCE_REPO_ID,
        root=SOURCE_ROOT,
        video_backend="torchcodec",
    )
    source_metadata = source_dataset.meta

    data = load_data_table(root)
    episodes = load_episode_table(root)
    require(data.num_rows == EXPECTED_TARGET_FRAMES, "Data parquet row count mismatch")
    require(episodes.num_rows == EXPECTED_TARGET_EPISODES, "Episode metadata row count mismatch")

    global_indices = np.asarray(data["index"].combine_chunks(), dtype=np.int64)
    episode_indices = np.asarray(data["episode_index"].combine_chunks(), dtype=np.int64)
    frame_indices = np.asarray(data["frame_index"].combine_chunks(), dtype=np.int64)
    timestamps = np.asarray(data["timestamp"].combine_chunks(), dtype=np.float32)
    task_indices = np.asarray(data["task_index"].combine_chunks(), dtype=np.int64)
    require(np.array_equal(global_indices, np.arange(EXPECTED_TARGET_FRAMES)), "Global indices are not contiguous")
    require(np.all(task_indices == 0), "Target contains an unexpected task index")

    expected_offset = 0
    visual_mae: dict[str, list[float]] = {key: [] for key in VIDEO_KEYS}
    for new_episode, old_episode in enumerate(KEEP_ORIGINAL_EPISODES):
        expected_length = int(dataset.meta.episodes[new_episode]["length"])
        source_length = int(source_metadata.episodes[old_episode]["length"])
        require(expected_length == source_length, f"Length changed for original episode {old_episode}")
        mask = episode_indices == new_episode
        require(mask.sum() == expected_length, f"Frame count mismatch in target episode {new_episode}")
        require(
            np.array_equal(frame_indices[mask], np.arange(expected_length)),
            f"Frame indices are not contiguous in target episode {new_episode}",
        )
        require(
            np.array_equal(global_indices[mask], np.arange(expected_offset, expected_offset + expected_length)),
            f"Global range mismatch in target episode {new_episode}",
        )
        require(
            np.allclose(
                timestamps[mask],
                np.arange(expected_length, dtype=np.float32) / np.float32(EXPECTED_FPS),
                atol=2e-6,
                rtol=0,
            ),
            f"Timestamp mismatch in target episode {new_episode}",
        )
        metadata = dataset.meta.episodes[new_episode]
        require(metadata["dataset_from_index"] == expected_offset, "Bad metadata start index")
        require(metadata["dataset_to_index"] == expected_offset + expected_length, "Bad metadata end index")
        require(metadata["tasks"] == [TASK], "Bad episode task metadata")
        for video_key in VIDEO_KEYS:
            duration = metadata[f"videos/{video_key}/to_timestamp"] - metadata[
                f"videos/{video_key}/from_timestamp"
            ]
            require(
                abs(duration - expected_length / EXPECTED_FPS) < 2e-6,
                f"Video duration mismatch for target episode {new_episode}, {video_key}",
            )

        source_start = int(source_metadata.episodes[old_episode]["dataset_from_index"])
        for relative_index in sorted({0, expected_length // 2, expected_length - 1}):
            source_item = source_dataset[source_start + relative_index]
            output_item = dataset[expected_offset + relative_index]
            for video_key in VIDEO_KEYS:
                difference = torch.mean(torch.abs(source_item[video_key] - output_item[video_key])).item()
                visual_mae[video_key].append(difference)
                require(
                    difference < 0.08,
                    f"Output image mismatch for original episode {old_episode}, "
                    f"frame {relative_index}, {video_key}: MAE={difference:.6f}",
                )
        expected_offset += expected_length
    require(expected_offset == EXPECTED_TARGET_FRAMES, "Target episode ranges do not cover all frames")

    numeric_arrays = {key: fixed_list_to_numpy(data[key], np.float32) for key in NUMERIC_KEYS}
    for key, values in numeric_arrays.items():
        require(values.shape == (EXPECTED_TARGET_FRAMES, 7), f"Wrong shape for {key}")
        require(np.isfinite(values).all(), f"Non-finite values in {key}")
    actual_hashes = output_numeric_hashes(data)
    if expected_numeric_hashes is not None:
        require(actual_hashes == expected_numeric_hashes, "Action/state values changed during rebuild")

    episode_rows = episodes.to_pylist()
    numeric_features = {key: dataset.meta.features[key] for key in NUMERIC_KEYS}
    recomputed_global_stats: dict[str, dict[str, np.ndarray]] | None = None
    for new_episode, row in enumerate(episode_rows):
        length = int(row["length"])
        require(row["episode_index"] == new_episode, "Episode metadata is not reindexed")
        mask = episode_indices == new_episode
        episode_values = {key: numeric_arrays[key][mask] for key in NUMERIC_KEYS}
        recomputed_episode_stats = compute_episode_stats(episode_values, numeric_features)
        for key in NUMERIC_KEYS:
            require(row[f"stats/{key}/count"] == [length], f"Bad {key} stats count")
            for statistic, expected in recomputed_episode_stats[key].items():
                stored = np.asarray(row[f"stats/{key}/{statistic}"])
                require(
                    np.allclose(stored, expected, atol=1e-7, rtol=1e-7),
                    f"Bad per-episode {key} {statistic} statistics in episode {new_episode}",
                )
        require(row["stats/episode_index/min"] == [new_episode], "Stale episode-index stats")
        require(row["stats/episode_index/max"] == [new_episode], "Stale episode-index stats")
        require(row["stats/index/min"] == [row["dataset_from_index"]], "Stale global-index stats")
        require(row["stats/index/max"] == [row["dataset_to_index"] - 1], "Stale global-index stats")
        recomputed_global_stats = (
            recomputed_episode_stats
            if recomputed_global_stats is None
            else aggregate_stats([recomputed_global_stats, recomputed_episode_stats])
        )

    stats = json.loads((root / "meta/stats.json").read_text())
    assert_finite_json(stats, "stats")
    require(recomputed_global_stats is not None, "No episode statistics were recomputed")
    for key in NUMERIC_KEYS:
        require(stats[key]["count"] == [EXPECTED_TARGET_FRAMES], f"Bad global stats count for {key}")

        for statistic, expected in recomputed_global_stats[key].items():
            stored = np.asarray(stats[key][statistic])
            require(
                np.allclose(stored, expected, atol=1e-7, rtol=1e-7),
                f"Global {key} {statistic} statistics do not match LeRobot's aggregation",
            )

        values = numeric_arrays[key]
        actual_mean = values.astype(np.float64).mean(axis=0)
        actual_std = values.astype(np.float64).std(axis=0)
        expected_statistics = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": actual_mean,
            "std": actual_std,
        }
        for statistic, expected in expected_statistics.items():
            stored = np.asarray(stats[key][statistic], dtype=np.float64)
            require(
                np.allclose(stored, expected, atol=0.05, rtol=1e-5),
                f"Global {key} {statistic} statistics do not match the rebuilt data",
            )

    # Official decoder boundary test: first, middle, and final frame of every episode.
    for episode_index, metadata in enumerate(dataset.meta.episodes):
        start = int(metadata["dataset_from_index"])
        stop = int(metadata["dataset_to_index"])
        for index in sorted({start, start + (stop - start) // 2, stop - 1}):
            item = dataset[index]
            require(int(item["episode_index"].item()) == episode_index, "Official decoder crossed a boundary")
            for video_key in VIDEO_KEYS:
                image = item[video_key]
                require(tuple(image.shape) == (3, 480, 640), "Official decoder returned a bad shape")
                require(torch.isfinite(image).all().item(), "Official decoder returned non-finite pixels")

    video_report = validate_video_files(root)
    png_files = list(root.rglob("*.png"))
    require(not png_files, f"Temporary PNG files remain: {png_files[:3]}")
    zero_byte_files = [path for path in root.rglob("*") if path.is_file() and path.stat().st_size == 0]
    require(not zero_byte_files, f"Zero-byte files found: {zero_byte_files[:3]}")

    return {
        "repo_id": TARGET_REPO_ID,
        "root": str(root),
        "episodes": EXPECTED_TARGET_EPISODES,
        "frames": EXPECTED_TARGET_FRAMES,
        "duration_s": EXPECTED_TARGET_FRAMES / EXPECTED_FPS,
        "numeric_sha256": actual_hashes,
        "source_output_visual_mae": {
            key: {
                "samples": len(values),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
            for key, values in visual_mae.items()
        },
        "videos": video_report,
    }


def exclusion_reason(episode_index: int) -> str:
    if episode_index in WRONG_ORDER_EPISODES:
        return "wrong object order: red moved before blue"
    if episode_index == 15:
        return "46.4-second timing outlier with substantial leading and trailing inactive footage"
    if episode_index == 45:
        return "manual object repositioning after episode recording began"
    if episode_index == 55:
        return "incomplete task; both cubes did not finish on the left platform"
    if episode_index == 56:
        return "incorrect start state followed by a manual object reset"
    if episode_index in PRIOR_INVALID_EPISODES:
        return "failed prior strict start-state, task-completion, or trajectory audit"
    raise ValueError(f"No exclusion reason for episode {episode_index}")


def write_manifest(
    staging_root: Path,
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    numeric_hashes: dict[str, str],
) -> None:
    rejected = sorted(set(range(EXPECTED_SOURCE_EPISODES)) - set(KEEP_ORIGINAL_EPISODES))
    mapping = {str(old): new for new, old in enumerate(KEEP_ORIGINAL_EPISODES)}
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": "strict two-camera trajectory audit followed by canonical LeRobot decode/rebuild",
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "root": str(SOURCE_ROOT),
            "episodes": EXPECTED_SOURCE_EPISODES,
            "frames": EXPECTED_SOURCE_FRAMES,
            "tree_before": source_before,
            "tree_after": source_after,
        },
        "output": {
            "repo_id": TARGET_REPO_ID,
            "root": str(TARGET_ROOT),
            "episodes": EXPECTED_TARGET_EPISODES,
            "frames": EXPECTED_TARGET_FRAMES,
            "fps": EXPECTED_FPS,
            "task": TASK,
        },
        "accepted_original_episode_indices": KEEP_ORIGINAL_EPISODES,
        "rejected_original_episode_indices": rejected,
        "old_to_new_episode_index": mapping,
        "exclusion_reasons": {str(index): exclusion_reason(index) for index in rejected},
        "numeric_source_sha256": numeric_hashes,
        "audit_acceptance_criteria": [
            "both cubes start on the right white pad",
            "blue cube is moved to the left gray platform before the red cube",
            "both cubes finish supported by the left gray platform",
            "no manual object reset after recording starts",
            "no failed grasp/drop recovery or unfinished contact ending",
            "no camera corruption or excessive inactive footage",
        ],
    }
    path = staging_root / "meta/filter_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_build() -> None:
    validate_audit_partition()
    require(not TARGET_ROOT.exists(), f"Refusing to overwrite existing target: {TARGET_ROOT}")
    TARGET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    DATASET_LOCK.parent.mkdir(parents=True, exist_ok=True)

    with DATASET_LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The right-to-left dataset is currently locked by a recorder") from error

        source = load_and_validate_source()
        source_before = tree_digest(SOURCE_ROOT)
        staging_root = make_staging_root()
        print(f"Source fingerprint: {source_before['sha256']}", flush=True)
        print(f"Building in staging directory: {staging_root}", flush=True)
        try:
            numeric_hashes = build_dataset(source, staging_root)
            source_after = tree_digest(SOURCE_ROOT)
            require(source_before == source_after, "Source dataset changed during the rebuild")
            write_manifest(staging_root, source_before, source_after, numeric_hashes)
            report = validate_dataset(staging_root, expected_numeric_hashes=numeric_hashes)
            report["validated_at"] = datetime.now().astimezone().isoformat()
            (staging_root / "meta/validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
            staging_root.rename(TARGET_ROOT)
        except Exception:
            print(f"Build failed; diagnostic staging data was left at: {staging_root}", file=sys.stderr)
            raise

    print(json.dumps(report, indent=2), flush=True)
    print(f"Validated dataset promoted atomically to: {TARGET_ROOT}", flush=True)


def recover_staging(staging_root: Path) -> None:
    """Validate and promote a complete build left behind by a validator failure."""
    require(not staging_root.is_symlink(), "Refusing to recover a symlinked staging directory")
    staging_root = staging_root.resolve()
    expected_parent = TARGET_ROOT.parent.resolve()
    require(staging_root.is_dir(), f"Staging dataset does not exist: {staging_root}")
    require(staging_root.parent == expected_parent, "Staging dataset is outside the target parent directory")
    require(
        staging_root.name.startswith(f".{TARGET_ROOT.name}.building-"),
        f"Unexpected staging directory name: {staging_root.name}",
    )
    require(
        not TARGET_ROOT.exists() and not TARGET_ROOT.is_symlink(),
        f"Refusing to overwrite existing target: {TARGET_ROOT}",
    )

    manifest_path = staging_root / "meta/filter_manifest.json"
    require(manifest_path.is_file(), f"Filter manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest["accepted_original_episode_indices"] == KEEP_ORIGINAL_EPISODES,
        "Manifest keep list does not match the audited keep list",
    )
    require(
        manifest["old_to_new_episode_index"]
        == {str(old): new for new, old in enumerate(KEEP_ORIGINAL_EPISODES)},
        "Manifest episode mapping is wrong",
    )
    require(manifest["output"]["repo_id"] == TARGET_REPO_ID, "Manifest repo ID is wrong")

    DATASET_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The right-to-left dataset is currently locked by a recorder") from error
        load_and_validate_source()
        source_digest = tree_digest(SOURCE_ROOT)
        require(source_digest == manifest["source"]["tree_before"], "Source differs from pre-build fingerprint")
        require(source_digest == manifest["source"]["tree_after"], "Source differs from post-build fingerprint")
        report = validate_dataset(staging_root, manifest["numeric_source_sha256"])
        report["validated_at"] = datetime.now().astimezone().isoformat()
        (staging_root / "meta/validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        staging_root.rename(TARGET_ROOT)

    print(json.dumps(report, indent=2), flush=True)
    print(f"Validated staging dataset promoted atomically to: {TARGET_ROOT}", flush=True)


def main() -> None:
    args = parse_args()
    require(
        sum((bool(args.check), bool(args.validate_only), args.recover_staging is not None)) <= 1,
        "Choose only one mode",
    )
    validate_audit_partition()
    if args.recover_staging is not None:
        recover_staging(args.recover_staging)
        return
    if args.validate_only:
        manifest_path = TARGET_ROOT / "meta/filter_manifest.json"
        require(manifest_path.is_file(), f"Filter manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        require(
            manifest["accepted_original_episode_indices"] == KEEP_ORIGINAL_EPISODES,
            "Manifest keep list does not match the audited keep list",
        )
        require(manifest["old_to_new_episode_index"] == {
            str(old): new for new, old in enumerate(KEEP_ORIGINAL_EPISODES)
        }, "Manifest episode mapping is wrong")
        require(manifest["output"]["repo_id"] == TARGET_REPO_ID, "Manifest repo ID is wrong")
        require(
            tree_digest(SOURCE_ROOT) == manifest["source"]["tree_after"],
            "The source dataset no longer matches the build manifest",
        )
        report = validate_dataset(TARGET_ROOT, manifest["numeric_source_sha256"])
        report["validated_at"] = datetime.now().astimezone().isoformat()
        (TARGET_ROOT / "meta/validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    if args.check:
        source = load_and_validate_source()
        print(
            json.dumps(
                {
                    "status": "preflight passed",
                    "source": str(SOURCE_ROOT),
                    "target": str(TARGET_ROOT),
                    "target_exists": TARGET_ROOT.exists(),
                    "accepted_original_episodes": KEEP_ORIGINAL_EPISODES,
                    "episodes": EXPECTED_TARGET_EPISODES,
                    "frames": EXPECTED_TARGET_FRAMES,
                    "duration_s": EXPECTED_TARGET_FRAMES / EXPECTED_FPS,
                },
                indent=2,
            )
        )
        return
    run_build()


if __name__ == "__main__":
    main()
