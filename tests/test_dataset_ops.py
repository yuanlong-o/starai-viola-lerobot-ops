from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import viola_handoff
import viola_ops.dataset as dataset_ops
from viola_ops.dataset import (
    ACCEPTED_SOURCE_EPISODES,
    CURRENT_DATASET,
    DATASET_INVENTORY_SHA256,
    DEFAULT_MATERIAL_ROOT,
    EPISODE_LENGTHS,
    EVALUATION_EPISODES,
    ExpectedFile,
    RELEASE_EPISODES,
    TASK,
    TRAIN_EPISODES,
    DatasetSpec,
    release_dataset,
    validate_dataset,
)
from viola_ops.errors import ValidationError
from viola_ops.jsonutil import sha256_json


def test_programmatic_release_defaults_to_shared_nas_material() -> None:
    assert DEFAULT_MATERIAL_ROOT == Path("/mnt/nas02/yz/starai/producer-materials/v1")


class FakeDataset:
    def __init__(self, spec: DatasetSpec, *, bad_action_at: int | None = None) -> None:
        self.num_episodes = len(spec.release_episodes)
        self.num_frames = spec.frames
        self.fps = spec.fps
        self.spec = spec
        self.bad_action_at = bad_action_at
        self._locations: list[tuple[int, int]] = []
        for episode, length in enumerate(spec.episode_lengths):
            self._locations.extend((episode, frame) for frame in range(length))

    def __len__(self) -> int:
        return self.num_frames

    def get_raw_item(self, index: int) -> dict[str, Any]:
        episode, frame = self._locations[index]
        action = [0.0] * 7
        if index == self.bad_action_at:
            action[3] = float("nan")
        return {
            "action": action,
            "observation.state": [0.0] * 7,
            "timestamp": frame / self.fps,
            "frame_index": frame,
            "episode_index": episode,
            "index": index,
            "task_index": 0,
        }

@dataclass
class FakeEvidence:
    fail: bool = False

    def __post_init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, **event: Any) -> str:
        self.events.append(event)
        if self.fail:
            raise RuntimeError("simulated W&B failure")
        return f"https://wandb.ai/tester/{event['project']}/runs/{event['run_id']}"


def _identity() -> viola_handoff.RuntimeIdentity:
    return viola_handoff.RuntimeIdentity(
        role="pc_a",
        repository_commit="a" * 40,
        repository_clean=True,
        hostname="pc-a-test",
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _repo_root(tmp_path: Path) -> Path:
    repository = tmp_path / "repo-a"
    repository.mkdir()
    return repository


def _dataset_fixture(tmp_path: Path) -> tuple[Path, DatasetSpec, FakeDataset]:
    root = tmp_path / "dataset"
    for directory in (
        "data/chunk-000",
        "images/observation.images.front",
        "images/observation.images.up",
        "meta/episodes/chunk-000",
        "videos/observation.images.front/chunk-000",
        "videos/observation.images.up/chunk-000",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    features = {
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": [f"Motor_{index}.pos" for index in range(6)] + ["gripper.pos"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": [f"Motor_{index}.pos" for index in range(6)] + ["gripper.pos"],
        },
    }
    for key in ("observation.images.front", "observation.images.up"):
        features[key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "info": {
                "video.codec": "av1",
                "video.fps": 30,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
            },
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "starai_viola",
        "total_episodes": 34,
        "total_frames": 28_306,
        "total_tasks": 1,
        "fps": 30,
        "splits": {"train": "0:34"},
        "features": features,
    }
    source_tree = {
        "sha256": CURRENT_DATASET.source_tree_sha256,
        "file_count": CURRENT_DATASET.source_file_count,
        "total_bytes": CURRENT_DATASET.source_byte_count,
    }
    filter_manifest = {
        "source": {
            "repo_id": CURRENT_DATASET.source_repo_id,
            "episodes": CURRENT_DATASET.source_episodes,
            "frames": CURRENT_DATASET.source_frames,
            "tree_before": source_tree,
            "tree_after": source_tree,
        },
        "output": {
            "repo_id": CURRENT_DATASET.dataset_repo_id,
            "episodes": 34,
            "frames": 28_306,
            "fps": 30,
            "task": TASK,
        },
        "accepted_original_episode_indices": list(ACCEPTED_SOURCE_EPISODES),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "filter_manifest.json").write_text(
        json.dumps(filter_manifest), encoding="utf-8"
    )
    for relative in (
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/validation_report.json",
        "videos/observation.images.front/chunk-000/file-000.mp4",
        "videos/observation.images.up/chunk-000/file-000.mp4",
    ):
        (root / relative).write_bytes(f"fixture:{relative}".encode())

    inventory = viola_handoff.inventory_root(root)
    spec = replace(
        CURRENT_DATASET,
        filter_manifest_canonical_sha256=sha256_json(filter_manifest),
        inventory_directories=tuple(inventory["directories"]),
        inventory_files=tuple(
            ExpectedFile(item["path"], item["sha256"], item["size_bytes"])
            for item in inventory["files"]
        ),
        inventory_sha256=inventory["inventory_sha256"],
        byte_count=inventory["byte_count"],
    )
    return root, spec, FakeDataset(spec)


def test_current_v1_facts_are_locked() -> None:
    assert len(ACCEPTED_SOURCE_EPISODES) == 34
    assert ACCEPTED_SOURCE_EPISODES[-7:] == (71, 72, 73, 74, 75, 76, 77)
    assert RELEASE_EPISODES == tuple(range(34))
    assert TRAIN_EPISODES == tuple(range(27))
    assert EVALUATION_EPISODES == tuple(range(27, 34))
    assert len(EPISODE_LENGTHS) == 34
    assert sum(EPISODE_LENGTHS) == 28_306
    assert sum(EPISODE_LENGTHS[27:]) == 6_079
    assert CURRENT_DATASET.inventory_sha256 == DATASET_INVENTORY_SHA256
    assert CURRENT_DATASET.byte_count == 362_533_979
    assert len(CURRENT_DATASET.inventory_files) == 9


def test_full_validation_loads_all_numeric_rows_and_both_camera_frames(tmp_path: Path) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    loaded: list[tuple[str, Path]] = []

    def loader(repo_id: str, path: Path) -> FakeDataset:
        loaded.append((repo_id, path))
        return dataset

    decoded: list[Path] = []

    def decode(path: Path, expected: DatasetSpec) -> int:
        decoded.append(path)
        return expected.frames

    result = validate_dataset(
        root, dataset_loader=loader, video_decoder=decode, spec=spec
    )

    assert loaded == [(spec.dataset_repo_id, root)]
    assert result.numeric_frames_checked == 28_306
    assert result.video_frames_decoded == 28_306
    assert result.episode_lengths == EPISODE_LENGTHS
    assert decoded == [root]
    assert result.full_decode is True


def test_validation_rejects_nonfinite_actions_and_inventory_drift(tmp_path: Path) -> None:
    root, spec, _dataset = _dataset_fixture(tmp_path)
    bad = FakeDataset(spec, bad_action_at=10)
    with pytest.raises(ValidationError, match="non-finite"):
        validate_dataset(
            root,
            dataset_loader=lambda _repo, _root: bad,
            video_decoder=lambda _root, expected: expected.frames,
            spec=spec,
        )

    (root / "meta" / "stats.json").write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="dataset bytes differ"):
        validate_dataset(root, dataset_loader=lambda _repo, _root: bad, spec=spec)


def test_release_seals_repo_b_v1_payload_without_nas_or_wandb(tmp_path: Path) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    evidence = FakeEvidence()

    result = release_dataset(
        root,
        experiment="viola-policy-benchmark-v1",
        handoff_root=tmp_path / "handoffs",
        material_root=tmp_path / "materials",
        wandb_project="viola-test",
        repo_root=_repo_root(tmp_path),
        producer_identity=_identity(),
        evidence_logger=evidence,
        validation=validation,
        spec=spec,
    )

    payload = json.loads((result.payload_root / "dataset_release.json").read_text())
    assert result.bundle.kind == "dataset_release"
    assert result.bundle.permission == "data_only"
    assert payload["schema_version"] == 1
    assert payload["selection"]["accepted_source_episode_ids"] == list(
        ACCEPTED_SOURCE_EPISODES
    )
    assert payload["dataset"]["episodes"] == 34
    assert payload["dataset"]["frames"] == 28_306
    assert payload["validation"]["normalization_scope"] == "dataset_wide_legacy"
    assert result.bundle.manifest["lineage"]["source_lerobot_version"] == "0.4.2"
    assert evidence.events[0]["event"] == "sealed"


def test_release_refuses_numeric_only_validation(tmp_path: Path) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        full_decode=False,
        dataset_loader=lambda _repo, _root: dataset,
        spec=spec,
    )
    with pytest.raises(ValidationError, match="complete two-camera decode"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=tmp_path / "handoffs",
            material_root=tmp_path / "materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
            validation=validation,
            spec=spec,
        )


def test_wandb_failure_never_publishes_a_ready_dataset_release(tmp_path: Path) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    handoffs = tmp_path / "handoffs"

    with pytest.raises(RuntimeError, match="simulated W&B failure"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=handoffs,
            material_root=tmp_path / "materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(fail=True),
            validation=validation,
            spec=spec,
        )

    assert not list(handoffs.rglob("READY.json"))


@pytest.mark.parametrize("protected_root", ["material", "handoff"])
def test_release_requires_external_producer_roots(
    tmp_path: Path,
    protected_root: str,
) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    repository = _repo_root(tmp_path)
    material_root = tmp_path / "materials"
    handoff_root = tmp_path / "handoffs"
    if protected_root == "material":
        material_root = repository / "materials"
    else:
        handoff_root = repository / "handoffs"

    with pytest.raises(ValidationError, match="outside the Repo-A worktree"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=handoff_root,
            material_root=material_root,
            repo_root=repository,
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
            validation=validation,
            spec=spec,
        )

    assert not material_root.exists()
    assert not handoff_root.exists()


@pytest.mark.parametrize("protected_root", ["material", "handoff"])
def test_release_rejects_symlinked_producer_roots(
    tmp_path: Path,
    protected_root: str,
) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    repository = _repo_root(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "linked-external"
    linked.symlink_to(external, target_is_directory=True)
    material_root = tmp_path / "materials"
    handoff_root = tmp_path / "handoffs"
    if protected_root == "material":
        material_root = linked / "materials"
    else:
        handoff_root = linked / "handoffs"

    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=handoff_root,
            material_root=material_root,
            repo_root=repository,
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
            validation=validation,
            spec=spec,
        )


def test_release_rejects_a_symlink_inside_the_material_path(tmp_path: Path) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    material_root = tmp_path / "materials"
    material_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (material_root / spec.release_id).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ValidationError, match="symlink path is forbidden"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=tmp_path / "handoffs",
            material_root=material_root,
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=FakeEvidence(),
            validation=validation,
            spec=spec,
        )

    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"role": "pc_b"}, "pc_a"),
        ({"repository_clean": False}, "clean Repo-A"),
        ({"repository_commit": "A" * 40}, "full lowercase Git SHA"),
        ({"python_version": "3.11.9"}, "Python 3.12"),
        ({"lerobot_version": "0.5.0"}, "LeRobot 0.6.1"),
        ({"conda_environment": "base"}, "lerobot Conda"),
    ],
)
def test_release_validates_injected_identity_before_writing(
    tmp_path: Path,
    changes: dict[str, Any],
    message: str,
) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    material_root = tmp_path / "materials"

    with pytest.raises(ValidationError, match=message):
        release_dataset(
            root,
            experiment="test",
            handoff_root=tmp_path / "handoffs",
            material_root=material_root,
            repo_root=_repo_root(tmp_path),
            producer_identity=replace(_identity(), **changes),
            evidence_logger=FakeEvidence(),
            validation=validation,
            spec=spec,
        )

    assert not material_root.exists()


def test_release_recaptures_the_clean_runtime_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    identities = iter([_identity(), replace(_identity(), repository_commit="b" * 40)])
    monkeypatch.setattr(
        viola_handoff.RuntimeIdentity,
        "capture",
        classmethod(lambda cls, **kwargs: next(identities)),
    )
    evidence = FakeEvidence()

    with pytest.raises(ValidationError, match="changed before release sealing"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=tmp_path / "handoffs",
            material_root=tmp_path / "materials",
            repo_root=_repo_root(tmp_path),
            evidence_logger=evidence,
            validation=validation,
            spec=spec,
        )

    assert evidence.events == []
    assert not (tmp_path / "handoffs").exists()


def test_release_rechecks_dataset_inventory_immediately_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, spec, dataset = _dataset_fixture(tmp_path)
    validation = validate_dataset(
        root,
        dataset_loader=lambda _repo, _root: dataset,
        video_decoder=lambda _root, expected: expected.frames,
        spec=spec,
    )
    real_write = dataset_ops.write_canonical_json

    def write_then_change_source(path: str | Path, value: Any) -> Path:
        written = real_write(path, value)
        if Path(path).name == "dataset_release.json":
            (root / "meta" / "stats.json").write_bytes(b"changed-before-seal")
        return written

    monkeypatch.setattr(dataset_ops, "write_canonical_json", write_then_change_source)
    evidence = FakeEvidence()

    with pytest.raises(ValidationError, match="immediately before release sealing"):
        release_dataset(
            root,
            experiment="test",
            handoff_root=tmp_path / "handoffs",
            material_root=tmp_path / "materials",
            repo_root=_repo_root(tmp_path),
            producer_identity=_identity(),
            evidence_logger=evidence,
            validation=validation,
            spec=spec,
        )

    assert evidence.events == []
    assert not (tmp_path / "handoffs").exists()
