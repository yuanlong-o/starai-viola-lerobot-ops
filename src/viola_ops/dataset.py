"""Validate and release the frozen 34-episode Viola dataset.

Validation is intentionally explicit.  An operator can read the checks in
order, while Repo B still receives its exact canonical ``dataset_release``
payload and immutable artifact inventory.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

import viola_handoff

from .errors import ValidationError
from .jsonutil import (
    read_json_object,
    sha256_file,
    sha256_json,
    write_canonical_json,
    write_text_once,
)
from .publication_guard import (
    GuardedEvidenceLogger,
    PublicationSnapshot,
    require_same_runtime,
    require_sealed_bundle_matches,
    snapshot_tree,
)

DEFAULT_DATASET_ROOT: Final = Path(
    "/mnt/nas02/yz/starai/datasets/bourn117/"
    "viola_cubes_right_to_left_blue_then_red_train_v1"
)
DEFAULT_WANDB_PROJECT: Final = "starai-viola-policy-benchmark"
DEFAULT_MATERIAL_ROOT: Final = Path(
    "/mnt/nas02/yz/starai/producer-materials/v1"
)

_FULL_GIT_SHA: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PYTHON_312: Final = re.compile(r"^3\.12(?:\.\d+)?$")

RELEASE_ID: Final = "viola-cubes-right-to-left-blue-then-red-v1--31cf41385cd9e183"
DATASET_REPO_ID: Final = "bourn117/viola_cubes_right_to_left_blue_then_red_train_v1"
SOURCE_REPO_ID: Final = "bourn117/viola_cubes_right_to_left_keep_pose_v3_clean"
TASK: Final = (
    "Move the blue cube, then the red cube, from the white pad on the right "
    "to the gray platform on the left."
)
SOURCE_LEROBOT_VERSION: Final = "0.4.2"
SOURCE_TREE_SHA256: Final = "9e8a3107df94371b910b382891d6cc1c8285cafb10ffd0a199ea8639856898b7"
FILTER_MANIFEST_CANONICAL_SHA256: Final = (
    "cec0695adfff3d9e2dfff8857f86860fbfa79ded1638ae320a5bdfe376adb087"
)
DATASET_INVENTORY_SHA256: Final = (
    "e0508423512ff9951840febe21916c01e4a9749de82b0e731e9636ec76792b0f"
)
ACCEPTED_SOURCE_EPISODES: Final = (
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
)
RELEASE_EPISODES: Final = tuple(range(34))
TRAIN_EPISODES: Final = tuple(range(27))
EVALUATION_EPISODES: Final = tuple(range(27, 34))
EPISODE_LENGTHS: Final = (
    780,
    605,
    759,
    1066,
    956,
    770,
    757,
    946,
    904,
    714,
    780,
    820,
    897,
    751,
    665,
    878,
    780,
    860,
    992,
    816,
    771,
    889,
    728,
    846,
    816,
    725,
    956,
    797,
    797,
    868,
    853,
    889,
    910,
    965,
)
VIDEO_KEYS: Final = ("observation.images.front", "observation.images.up")
ACTION_KEYS: Final = ("action", "observation.state")


@dataclass(frozen=True, slots=True)
class ExpectedFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    release_id: str
    dataset_repo_id: str
    source_repo_id: str
    task: str
    source_lerobot_version: str
    source_tree_sha256: str
    source_episodes: int
    source_frames: int
    source_file_count: int
    source_byte_count: int
    accepted_source_episodes: tuple[int, ...]
    release_episodes: tuple[int, ...]
    train_episodes: tuple[int, ...]
    evaluation_episodes: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    frames: int
    fps: int
    robot_type: str
    video_keys: tuple[str, ...]
    filter_manifest_canonical_sha256: str
    inventory_directories: tuple[str, ...]
    inventory_files: tuple[ExpectedFile, ...]
    inventory_sha256: str
    byte_count: int

    def expected_inventory(self) -> dict[str, Any]:
        directories = list(self.inventory_directories)
        files = [
            {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
            for item in self.inventory_files
        ]
        return {
            "directories": directories,
            "files": files,
            "file_count": len(files),
            "byte_count": self.byte_count,
            "inventory_sha256": self.inventory_sha256,
        }


CURRENT_DATASET = DatasetSpec(
    release_id=RELEASE_ID,
    dataset_repo_id=DATASET_REPO_ID,
    source_repo_id=SOURCE_REPO_ID,
    task=TASK,
    source_lerobot_version=SOURCE_LEROBOT_VERSION,
    source_tree_sha256=SOURCE_TREE_SHA256,
    source_episodes=78,
    source_frames=67_232,
    source_file_count=49,
    source_byte_count=913_499_264,
    accepted_source_episodes=ACCEPTED_SOURCE_EPISODES,
    release_episodes=RELEASE_EPISODES,
    train_episodes=TRAIN_EPISODES,
    evaluation_episodes=EVALUATION_EPISODES,
    episode_lengths=EPISODE_LENGTHS,
    frames=28_306,
    fps=30,
    robot_type="starai_viola",
    video_keys=VIDEO_KEYS,
    filter_manifest_canonical_sha256=FILTER_MANIFEST_CANONICAL_SHA256,
    inventory_directories=(
        "data",
        "data/chunk-000",
        "images",
        "images/observation.images.front",
        "images/observation.images.up",
        "meta",
        "meta/episodes",
        "meta/episodes/chunk-000",
        "videos",
        "videos/observation.images.front",
        "videos/observation.images.front/chunk-000",
        "videos/observation.images.up",
        "videos/observation.images.up/chunk-000",
    ),
    inventory_files=(
        ExpectedFile(
            "data/chunk-000/file-000.parquet",
            "2259dd54be900d5dab7618b7c20fe7dfed4b12587920c7ec078c1f2790cf5591",
            1_070_271,
        ),
        ExpectedFile(
            "meta/episodes/chunk-000/file-000.parquet",
            "871572eccc2fdee6632d4892a7afba0f046085b0481b40d09e999b0493892cca",
            185_509,
        ),
        ExpectedFile(
            "meta/filter_manifest.json",
            "a687c948e4edb3cc5bf6d9a461a63ad587fafc33b4eba7620ba6b5b9c44466ab",
            5_683,
        ),
        ExpectedFile(
            "meta/info.json",
            "674ce072d5ef30dcacb908fa18acb88e294853884f6b0f66e0b55e0863903e52",
            3_247,
        ),
        ExpectedFile(
            "meta/stats.json",
            "d407701cc9ad1116652cfed6b8a4314509c5f1f3dadbe5d6cedddc00f0161c6b",
            13_494,
        ),
        ExpectedFile(
            "meta/tasks.parquet",
            "a083774b7f5afe34e4f7c0b567771b9e87be7dbe287e49e6c651a0afb1f0d482",
            2_605,
        ),
        ExpectedFile(
            "meta/validation_report.json",
            "fc93aa7ee34d0a519344b2b20b48930844f7c1c3a042853d1ccc38cc8393fd62",
            1_517,
        ),
        ExpectedFile(
            "videos/observation.images.front/chunk-000/file-000.mp4",
            "bbf261510a71748fb9c70c8827698f9f5355e2c5c2896d3f39c00be4b08e9c61",
            249_051_297,
        ),
        ExpectedFile(
            "videos/observation.images.up/chunk-000/file-000.mp4",
            "3560607ddaf51c83871a14408383c6942a96205f945f83a9e2cdc6314322368c",
            112_200_356,
        ),
    ),
    inventory_sha256=DATASET_INVENTORY_SHA256,
    byte_count=362_533_979,
)


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    root: Path
    release_id: str
    dataset_repo_id: str
    task: str
    inventory_sha256: str
    file_count: int
    byte_count: int
    episodes: int
    frames: int
    fps: int
    episode_lengths: tuple[int, ...]
    train_episodes: tuple[int, ...]
    evaluation_episodes: tuple[int, ...]
    numeric_frames_checked: int
    video_frames_decoded: int
    full_decode: bool
    filter_manifest_canonical_sha256: str
    lerobot_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "passed",
            "mode": "full" if self.full_decode else "numeric_only",
            "release_id": self.release_id,
            "dataset_repo_id": self.dataset_repo_id,
            "task": self.task,
            "root": str(self.root),
            "inventory_sha256": self.inventory_sha256,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "episodes": self.episodes,
            "frames": self.frames,
            "fps": self.fps,
            "episode_lengths": list(self.episode_lengths),
            "train_episodes": list(self.train_episodes),
            "evaluation_episodes": list(self.evaluation_episodes),
            "numeric_frames_checked": self.numeric_frames_checked,
            "video_frames_decoded": self.video_frames_decoded,
            "filter_manifest_canonical_sha256": self.filter_manifest_canonical_sha256,
            "runtime": {"python": "3.12", "lerobot": self.lerobot_version},
        }

    @property
    def evidence_sha256(self) -> str:
        return sha256_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class DatasetReleaseResult:
    bundle: viola_handoff.VerifiedBundle
    validation: DatasetValidationResult
    payload_root: Path
    material_root: Path


DatasetLoader = Callable[[str, Path], Any]
VideoDecoder = Callable[[Path, DatasetSpec], int]


def validate_dataset(
    root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    full_decode: bool = True,
    dataset_loader: DatasetLoader | None = None,
    video_decoder: VideoDecoder | None = None,
    spec: DatasetSpec | None = None,
) -> DatasetValidationResult:
    """Validate exact frozen bytes, all numeric rows, and optionally every video frame."""

    expected = spec or CURRENT_DATASET
    dataset_root = _dataset_directory(root)
    _require_lerobot_061()
    before = viola_handoff.inventory_root(dataset_root)
    if before != expected.expected_inventory():
        raise ValidationError(
            "dataset bytes differ from the frozen release "
            f"(expected {expected.inventory_sha256}, found {before['inventory_sha256']})"
        )

    info = read_json_object(dataset_root / "meta" / "info.json", label="dataset info")
    filter_manifest = read_json_object(
        dataset_root / "meta" / "filter_manifest.json", label="reviewed filter manifest"
    )
    _validate_info(info, expected)
    _validate_filter_manifest(filter_manifest, expected)

    loader = dataset_loader or _load_lerobot_dataset
    dataset = loader(expected.dataset_repo_id, dataset_root)
    if (
        dataset.num_episodes != len(expected.release_episodes)
        or dataset.num_frames != expected.frames
        or dataset.fps != expected.fps
        or len(dataset) != expected.frames
    ):
        raise ValidationError("LeRobot 0.6.1 loaded counts differ from the frozen release")

    episode_counts = [0] * len(expected.release_episodes)
    numeric_checked = 0
    previous_episode = 0
    previous_frame = -1
    for index in range(expected.frames):
        raw = dataset.get_raw_item(index)
        episode = _integer_scalar(raw.get("episode_index"), "episode_index")
        frame = _integer_scalar(raw.get("frame_index"), "frame_index")
        absolute_index = _integer_scalar(raw.get("index"), "index")
        task_index = _integer_scalar(raw.get("task_index"), "task_index")
        timestamp = _finite_scalar(raw.get("timestamp"), "timestamp")
        if episode not in expected.release_episodes:
            raise ValidationError(f"frame {index} has unexpected episode_index {episode}")
        wanted_frame = previous_frame + 1 if episode == previous_episode else 0
        if episode < previous_episode or frame != wanted_frame:
            raise ValidationError(f"frame {index} is out of canonical episode order")
        if absolute_index != index or task_index != 0:
            raise ValidationError(f"frame {index} has inconsistent index or task_index")
        if abs(timestamp - frame / expected.fps) > 1e-4:
            raise ValidationError(f"frame {index} timestamp differs from frame_index/fps")
        for key in ACTION_KEYS:
            _finite_vector(raw.get(key), size=7, label=f"frame {index} {key}")
        episode_counts[episode] += 1
        numeric_checked += 1
        previous_episode = episode
        previous_frame = frame

    if tuple(episode_counts) != expected.episode_lengths:
        raise ValidationError("episode lengths differ from the frozen 34-episode release")
    decoded = 0
    if full_decode:
        decode = video_decoder or _decode_all_videos
        decoded = decode(dataset_root, expected)
        if decoded != expected.frames:
            raise ValidationError(
                "each camera must decode exactly "
                f"{expected.frames} frames, decoder reported {decoded}"
            )
    after = viola_handoff.inventory_root(dataset_root)
    if after != before:
        raise ValidationError("dataset changed while validation was running")

    return DatasetValidationResult(
        root=dataset_root,
        release_id=expected.release_id,
        dataset_repo_id=expected.dataset_repo_id,
        task=expected.task,
        inventory_sha256=before["inventory_sha256"],
        file_count=before["file_count"],
        byte_count=before["byte_count"],
        episodes=len(expected.release_episodes),
        frames=expected.frames,
        fps=expected.fps,
        episode_lengths=expected.episode_lengths,
        train_episodes=expected.train_episodes,
        evaluation_episodes=expected.evaluation_episodes,
        numeric_frames_checked=numeric_checked,
        video_frames_decoded=decoded,
        full_decode=full_decode,
        filter_manifest_canonical_sha256=expected.filter_manifest_canonical_sha256,
        lerobot_version="0.6.1",
    )


def release_dataset(
    root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    experiment: str,
    handoff_root: str | Path = viola_handoff.DEFAULT_HANDOFF_ROOT,
    material_root: str | Path = DEFAULT_MATERIAL_ROOT,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    repo_root: str | Path,
    producer_identity: viola_handoff.RuntimeIdentity | None = None,
    evidence_logger: viola_handoff.EvidenceLogger | None = None,
    identity_capture: Callable[[], viola_handoff.RuntimeIdentity] | None = None,
    validation: DatasetValidationResult | None = None,
    dataset_loader: DatasetLoader | None = None,
    video_decoder: VideoDecoder | None = None,
    spec: DatasetSpec | None = None,
) -> DatasetReleaseResult:
    """Run full validation and seal Repo B's current-v1 dataset release."""

    repository = _existing_directory(repo_root, label="Repo-A worktree")
    material_base = _external_output_root(
        material_root, repository=repository, label="producer material root"
    )
    handoff_base = _external_output_root(
        handoff_root, repository=repository, label="handoff root"
    )

    expected = spec or CURRENT_DATASET
    checked = validation or validate_dataset(
        root,
        full_decode=True,
        dataset_loader=dataset_loader,
        video_decoder=video_decoder,
        spec=expected,
    )
    dataset_root = _dataset_directory(root)
    if checked.root != dataset_root or checked.release_id != expected.release_id:
        raise ValidationError("provided validation belongs to a different dataset release")
    if not checked.full_decode or checked.video_frames_decoded != expected.frames:
        raise ValidationError("dataset release requires a complete two-camera decode")
    if checked.inventory_sha256 != expected.inventory_sha256:
        raise ValidationError("validation inventory differs from the frozen release")
    dataset_snapshot = snapshot_tree(dataset_root, label="validated dataset")
    if dataset_snapshot.inventory != expected.expected_inventory():
        raise ValidationError("dataset changed after validation and before release")

    capture_identity = identity_capture or (
        lambda: viola_handoff.RuntimeIdentity.capture(
            role="pc_a", repo_root=repository
        )
    )
    identity = _producer_identity(
        producer_identity or capture_identity()
    )

    material = _safe_output_directory(
        material_base / expected.release_id / checked.evidence_sha256,
        repository=repository,
        label="producer material directory",
    )
    payload = material / "payload"
    validation_evidence = checked.as_dict()
    write_canonical_json(payload / "validation_evidence.json", validation_evidence)
    release_payload = _release_payload(expected, checked)
    write_canonical_json(payload / "dataset_release.json", release_payload)
    write_text_once(payload / "SUMMARY.md", _release_summary(expected, checked))

    publication = PublicationSnapshot(
        payload=snapshot_tree(payload, label="dataset-release payload"),
        artifacts=(("dataset", dataset_snapshot),),
    )
    publication.require_unchanged(boundary="immediately before release sealing")
    require_same_runtime(
        identity, capture=capture_identity, operation="dataset release"
    )

    request = viola_handoff.SealRequest(
        root=handoff_base,
        kind="dataset_release",
        experiment=experiment,
        subject=expected.release_id,
        producer=identity,
        lineage={
            "dataset_release_id": expected.release_id,
            "source_repo_id": expected.source_repo_id,
            "source_lerobot_version": expected.source_lerobot_version,
            "source_tree_sha256": expected.source_tree_sha256,
            "accepted_source_episode_ids": list(expected.accepted_source_episodes),
            "normalization_scope": "dataset_wide_legacy",
            "validation_mode": "historical_qualification",
            "dataset_inventory_sha256": expected.inventory_sha256,
        },
        wandb_project=wandb_project,
        payload_dir=payload,
        artifact_roots={"dataset": dataset_root},
    )
    guarded_logger = GuardedEvidenceLogger(
        delegate=evidence_logger or viola_handoff.WandbEvidenceLogger(),
        snapshot=publication,
        identity=identity,
        identity_capture=capture_identity,
        operation="dataset release",
    )
    bundle = viola_handoff.seal_bundle(request, evidence_logger=guarded_logger)
    require_sealed_bundle_matches(
        bundle,
        request,
        publication,
        operation="dataset release",
        permission="data_only",
        consumer_role="pc_b",
    )
    require_same_runtime(
        identity, capture=capture_identity, operation="dataset release"
    )
    publication.require_unchanged(boundary="after release sealing")
    return DatasetReleaseResult(
        bundle=bundle,
        validation=checked,
        payload_root=payload,
        material_root=material,
    )


def _release_payload(
    expected: DatasetSpec, checked: DatasetValidationResult
) -> dict[str, Any]:
    raw_hashes = {item.path: item.sha256 for item in expected.inventory_files}
    return {
        "schema_version": 1,
        "release_id": expected.release_id,
        "dataset_repo_id": expected.dataset_repo_id,
        "task": expected.task,
        "source": {
            "repo_id": expected.source_repo_id,
            "root": (
                "/home/yz/lerobot/data/recordings/bourn117/"
                "viola_cubes_right_to_left_keep_pose_v3_clean"
            ),
            "lerobot_version": expected.source_lerobot_version,
            "episodes": expected.source_episodes,
            "frames": expected.source_frames,
            "inventory": {
                "algorithm": "viola_legacy_tree_v1",
                "sha256": expected.source_tree_sha256,
                "file_count": expected.source_file_count,
                "byte_count": expected.source_byte_count,
            },
        },
        "selection": {
            "accepted_source_episode_ids": list(expected.accepted_source_episodes),
            "reviewed_filter_manifest_canonical_sha256": (
                expected.filter_manifest_canonical_sha256
            ),
        },
        "dataset": {
            "artifact_name": "dataset",
            "root": str(checked.root),
            "inventory_sha256": checked.inventory_sha256,
            "file_count": checked.file_count,
            "byte_count": checked.byte_count,
            "episodes": checked.episodes,
            "frames": checked.frames,
            "fps": checked.fps,
            "robot_type": expected.robot_type,
            "video_keys": list(expected.video_keys),
            "video_backend": "pyav",
        },
        "validation": {
            "mode": "historical_qualification",
            "status": "passed",
            "normalization_scope": "dataset_wide_legacy",
            "validation_sha256": raw_hashes["meta/validation_report.json"],
            "qualification_sha256": checked.evidence_sha256,
            "evidence_sha256": {
                "filter_manifest": raw_hashes["meta/filter_manifest.json"],
                "historical_validation_report": raw_hashes["meta/validation_report.json"],
                "pc_a_full_validation": checked.evidence_sha256,
            },
        },
    }


def _validate_info(info: Mapping[str, Any], expected: DatasetSpec) -> None:
    wanted = {
        "codebase_version": "v3.0",
        "robot_type": expected.robot_type,
        "total_episodes": len(expected.release_episodes),
        "total_frames": expected.frames,
        "total_tasks": 1,
        "fps": expected.fps,
    }
    for field, value in wanted.items():
        if info.get(field) != value:
            raise ValidationError(f"dataset info {field} differs from the frozen release")
    if info.get("splits") != {"train": "0:34"}:
        raise ValidationError("dataset's physical split must remain the complete 0:34 release")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValidationError("dataset info features must be an object")
    names = [f"Motor_{index}.pos" for index in range(6)] + ["gripper.pos"]
    for key in ACTION_KEYS:
        feature = features.get(key)
        if not isinstance(feature, Mapping) or feature.get("shape") != [7]:
            raise ValidationError(f"dataset feature {key} must be seven-dimensional")
        if feature.get("dtype") != "float32" or feature.get("names") != names:
            raise ValidationError(f"dataset feature {key} names/dtype differ from Viola v1")
    video_features = [key for key, value in features.items() if value.get("dtype") == "video"]
    if video_features != list(expected.video_keys):
        raise ValidationError("dataset must contain exactly the front and up video features")
    for key in expected.video_keys:
        feature = features[key]
        details = feature.get("info")
        if feature.get("shape") != [480, 640, 3] or not isinstance(details, Mapping):
            raise ValidationError(f"dataset video feature {key} has the wrong shape")
        if (
            details.get("video.codec") != "av1"
            or details.get("video.fps") != expected.fps
            or details.get("video.height") != 480
            or details.get("video.width") != 640
            or details.get("video.channels") != 3
        ):
            raise ValidationError(f"dataset video feature {key} differs from frozen AV1 metadata")


def _validate_filter_manifest(manifest: Mapping[str, Any], expected: DatasetSpec) -> None:
    if sha256_json(manifest) != expected.filter_manifest_canonical_sha256:
        raise ValidationError("reviewed filter manifest canonical hash differs")
    source = manifest.get("source")
    output = manifest.get("output")
    if not isinstance(source, Mapping) or not isinstance(output, Mapping):
        raise ValidationError("reviewed filter manifest lacks source/output provenance")
    if (
        source.get("repo_id") != expected.source_repo_id
        or source.get("episodes") != expected.source_episodes
        or source.get("frames") != expected.source_frames
        or source.get("tree_before") != source.get("tree_after")
        or source.get("tree_before", {}).get("sha256") != expected.source_tree_sha256
    ):
        raise ValidationError("historical source provenance differs from the reviewed manifest")
    if (
        output.get("repo_id") != expected.dataset_repo_id
        or output.get("episodes") != len(expected.release_episodes)
        or output.get("frames") != expected.frames
        or output.get("fps") != expected.fps
        or output.get("task") != expected.task
    ):
        raise ValidationError("reviewed filter output differs from the frozen release")
    if manifest.get("accepted_original_episode_indices") != list(
        expected.accepted_source_episodes
    ):
        raise ValidationError("reviewed filter must select exactly the accepted 34 episodes")


def _load_lerobot_dataset(repo_id: str, root: Path) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=root,
        download_videos=False,
        video_backend="pyav",
    )


def _decode_all_videos(root: Path, expected: DatasetSpec) -> int:
    """Decode each frozen AV1 stream sequentially through public PyAV APIs."""

    import av

    counts: dict[str, int] = {}
    av1_codec_id = av.codec.Codec("av1", "r").id
    for key in expected.video_keys:
        video = root / "videos" / key / "chunk-000" / "file-000.mp4"
        try:
            with av.open(str(video), mode="r") as container:
                if len(container.streams.video) != 1:
                    raise ValidationError(f"{key} must contain exactly one video stream")
                stream = container.streams.video[0]
                if stream.codec_context.codec.id != av1_codec_id:
                    raise ValidationError(f"{key} video codec must be AV1")
                count = 0
                previous_pts: int | None = None
                for frame in container.decode(stream):
                    if (frame.width, frame.height) != (640, 480):
                        raise ValidationError(f"{key} decoded a frame that is not 640x480")
                    if frame.pts is None:
                        raise ValidationError(f"{key} decoded a frame without a timestamp")
                    if previous_pts is not None and frame.pts <= previous_pts:
                        raise ValidationError(f"{key} decoded non-increasing frame timestamps")
                    previous_pts = frame.pts
                    count += 1
        except ValidationError:
            raise
        except (OSError, av.error.FFmpegError) as exc:
            raise ValidationError(f"cannot fully decode {key}: {exc}") from exc
        if count != expected.frames:
            raise ValidationError(
                f"{key} decoded {count} frames, expected exactly {expected.frames}"
            )
        counts[key] = count
    if set(counts.values()) != {expected.frames}:
        raise ValidationError("front and up videos did not decode the same complete frame count")
    return expected.frames


def _finite_vector(value: Any, *, size: int, label: str) -> None:
    if value is None:
        raise ValidationError(f"{label} is missing")
    raw = value.tolist() if hasattr(value, "tolist") else list(value)
    if not isinstance(raw, list) or len(raw) != size:
        raise ValidationError(f"{label} must have exactly {size} values")
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError(f"{label} contains a non-number")
        if not math.isfinite(float(item)):
            raise ValidationError(f"{label} contains a non-finite value")


def _integer_scalar(value: Any, label: str) -> int:
    raw = value.item() if hasattr(value, "item") else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValidationError(f"{label} must be an integer")
    return raw


def _finite_scalar(value: Any, label: str) -> float:
    raw = value.item() if hasattr(value, "item") else value
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValidationError(f"{label} must be a finite number")
    return float(raw)


def _dataset_directory(path: str | Path) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValidationError(f"dataset root does not exist: {candidate}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink dataset path is forbidden: {current}")
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ValidationError(f"dataset root is not a directory: {candidate}")
    return candidate


def _existing_directory(path: str | Path, *, label: str) -> Path:
    candidate = _path_without_symlinks(path, label=label, must_exist=True)
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ValidationError(f"{label} is not a directory: {candidate}")
    return candidate


def _external_output_root(
    path: str | Path,
    *,
    repository: Path,
    label: str,
) -> Path:
    candidate = _path_without_symlinks(path, label=label, must_exist=False)
    if candidate == repository or repository in candidate.parents:
        raise ValidationError(f"{label} must be outside the Repo-A worktree")
    return candidate


def _safe_output_directory(
    path: str | Path,
    *,
    repository: Path,
    label: str,
) -> Path:
    candidate = _external_output_root(path, repository=repository, label=label)
    candidate.mkdir(parents=True, exist_ok=True)
    return _existing_directory(candidate, label=label)


def _path_without_symlinks(
    path: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if must_exist:
                raise ValidationError(f"{label} does not exist: {candidate}") from None
            break
        except OSError as exc:
            raise ValidationError(f"cannot inspect {label} {candidate}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValidationError(f"symlink path is forbidden for {label}: {current}")
        if not stat.S_ISDIR(mode):
            raise ValidationError(f"{label} path component is not a directory: {current}")
    return candidate


def _producer_identity(
    identity: viola_handoff.RuntimeIdentity,
) -> viola_handoff.RuntimeIdentity:
    """Validate injected identities before any producer evidence is written."""

    if not isinstance(identity, viola_handoff.RuntimeIdentity):
        raise ValidationError("producer identity must be a RuntimeIdentity")
    if identity.role != "pc_a":
        raise ValidationError("dataset releases must use a pc_a producer identity")
    if identity.repository_clean is not True:
        raise ValidationError("dataset releases require a clean Repo-A identity")
    if _FULL_GIT_SHA.fullmatch(identity.repository_commit) is None:
        raise ValidationError("producer repository commit must be a full lowercase Git SHA")
    if not identity.hostname:
        raise ValidationError("producer hostname must be nonempty")
    if _PYTHON_312.fullmatch(identity.python_version) is None:
        raise ValidationError("dataset releases require Python 3.12")
    if identity.lerobot_version != "0.6.1":
        raise ValidationError("dataset releases require LeRobot 0.6.1")
    if identity.conda_environment != "lerobot":
        raise ValidationError("dataset releases require the lerobot Conda environment")
    return identity


def _require_lerobot_061() -> None:
    try:
        installed = version("lerobot")
    except PackageNotFoundError as exc:
        raise ValidationError("LeRobot is not installed in the active environment") from exc
    if installed != "0.6.1":
        raise ValidationError(f"dataset validation requires LeRobot 0.6.1, found {installed}")


def _release_summary(expected: DatasetSpec, checked: DatasetValidationResult) -> str:
    return (
        "# Viola dataset release\n\n"
        f"- Release: `{expected.release_id}`\n"
        f"- Episodes: {checked.episodes} (train 0-26, evaluation 27-33)\n"
        f"- Frames: {checked.frames}\n"
        f"- Cameras: {', '.join(expected.video_keys)}\n"
        f"- Full decoded frames per camera: {checked.video_frames_decoded}\n"
        f"- Artifact inventory SHA-256: `{checked.inventory_sha256}`\n"
        f"- Validation evidence SHA-256: `{checked.evidence_sha256}`\n\n"
        "Historical note: this v1 release used dataset-wide normalization. "
        "It is preserved for the existing benchmark and is not a fresh "
        "training-only-normalized release.\n"
    )


__all__ = [
    "ACCEPTED_SOURCE_EPISODES",
    "CURRENT_DATASET",
    "DATASET_INVENTORY_SHA256",
    "DATASET_REPO_ID",
    "DEFAULT_DATASET_ROOT",
    "DatasetReleaseResult",
    "DatasetSpec",
    "DatasetValidationResult",
    "EPISODE_LENGTHS",
    "EVALUATION_EPISODES",
    "FILTER_MANIFEST_CANONICAL_SHA256",
    "RELEASE_EPISODES",
    "RELEASE_ID",
    "TASK",
    "TRAIN_EPISODES",
    "release_dataset",
    "validate_dataset",
]
