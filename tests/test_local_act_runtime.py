from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from viola_handoff import inventory_root
from viola_ops.errors import ValidationError
from viola_ops.jsonutil import sha256_file
from viola_ops.local_act import (
    LEGACY_ACT_EXPECTATION,
    LEGACY_ACT_MODEL_SHA256,
    LocalActExpectation,
    LocalActPublicApi,
    inspect_local_act_candidate,
    load_local_act_runtime,
    prepare_local_act_observation,
    snapshot_local_act_runtime,
)
from viola_ops.policies import CANONICAL_TASK, QUEUE_ACTIONS


def _checkpoint_config() -> dict[str, Any]:
    return {
        "type": "act",
        "device": "cuda",
        "chunk_size": 100,
        "n_action_steps": 100,
        "use_amp": False,
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.up": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [7]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }


def _write_fixture(
    tmp_path: Path,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[Path, Path, LocalActExpectation]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_config = config or _checkpoint_config()
    (checkpoint / "config.json").write_text(json.dumps(checkpoint_config), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"fixture model weights")
    (checkpoint / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    (checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_bytes(
        b"normalizer"
    )
    (checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        b"unnormalizer"
    )
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "test/local-act-dataset"},
                "policy": checkpoint_config,
            }
        ),
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset"
    metadata = dataset / "meta"
    (metadata / "episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (metadata / "episodes/chunk-000/file-000.parquet").write_bytes(b"episodes")
    (metadata / "filter_manifest.json").write_text("{}", encoding="utf-8")
    (metadata / "stats.json").write_text("{}", encoding="utf-8")
    (metadata / "tasks.parquet").write_bytes(b"tasks")
    (dataset / "data/chunk-000/file-000.parquet").write_bytes(b"frames")
    info = {
        "total_episodes": 34,
        "total_frames": 28_306,
        "fps": 30,
        "robot_type": "starai_viola",
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
            "observation.state": {"dtype": "float32", "shape": [7]},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.up": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (metadata / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (metadata / "validation_report.json").write_text(
        json.dumps(
            {
                "repo_id": "test/local-act-dataset",
                "episodes": 34,
                "frames": 28_306,
            }
        ),
        encoding="utf-8",
    )
    expectation = LocalActExpectation(
        checkpoint_inventory=inventory_root(checkpoint),
        dataset_inventory=inventory_root(dataset),
        dataset_metadata_inventory=inventory_root(metadata),
        model_sha256=sha256_file(checkpoint / "model.safetensors"),
        dataset_repo_id="test/local-act-dataset",
        dataset_release_id="test-local-act-release",
    )
    return checkpoint, dataset, expectation


def _candidate(tmp_path: Path):
    checkpoint, dataset, expectation = _write_fixture(tmp_path)
    return inspect_local_act_candidate(
        checkpoint,
        dataset,
        expectation=expectation,
    )


def _observation() -> dict[str, Any]:
    return {
        "observation.state": [0.0] * 6 + [50.0],
        "observation.images.front": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.up": np.zeros((480, 640, 3), dtype=np.uint8),
        "task": CANONICAL_TASK,
    }


def test_legacy_expectation_binds_the_previously_deployed_model() -> None:
    assert LEGACY_ACT_MODEL_SHA256 == (
        "1093aaeddfb902e7e596425d87676baba58cb8ab617a52c954ec11940726b886"
    )
    assert LEGACY_ACT_EXPECTATION.checkpoint_inventory["inventory_sha256"] == (
        "c3764c47677c8b33adbcf5f288429b28180bdb64542cd46e7a92aba7dbdca214"
    )
    assert LEGACY_ACT_EXPECTATION.dataset_inventory["inventory_sha256"] == (
        "e0508423512ff9951840febe21916c01e4a9749de82b0e731e9636ec76792b0f"
    )


def test_inspector_returns_one_stable_local_act_identity(tmp_path: Path) -> None:
    checkpoint, dataset, expectation = _write_fixture(tmp_path)

    first = inspect_local_act_candidate(checkpoint, dataset, expectation=expectation)
    second = inspect_local_act_candidate(checkpoint, dataset, expectation=expectation)

    assert first.policy == "act"
    assert first.spec.token == "act"
    assert first.bundle_id == first.content_id == second.candidate_id
    assert first.model_sha256 == expectation.model_sha256
    assert first.dataset_release_id == "test-local-act-release"


def test_inspector_rejects_any_extra_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint, dataset, expectation = _write_fixture(tmp_path)
    (checkpoint / "unreviewed.bin").write_bytes(b"extra")

    with pytest.raises(ValidationError, match="checkpoint inventory differs"):
        inspect_local_act_candidate(checkpoint, dataset, expectation=expectation)


def test_inspector_rejects_semantically_wrong_act_config_even_when_inventoried(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config()
    config["n_action_steps"] = 12
    checkpoint, dataset, expectation = _write_fixture(tmp_path, config=config)

    with pytest.raises(ValidationError, match="architecture differs"):
        inspect_local_act_candidate(checkpoint, dataset, expectation=expectation)


def test_private_snapshot_contains_only_checkpoint_and_dataset_metadata(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)

    with snapshot_local_act_runtime(candidate) as snapshot:
        private_root = snapshot.root
        snapshot.verify()
        assert snapshot.candidate.checkpoint.parent == private_root
        assert snapshot.candidate.dataset_root.parent == private_root
        assert (snapshot.candidate.dataset_root / "meta/info.json").is_file()
        assert not (snapshot.candidate.dataset_root / "data").exists()
        assert not (snapshot.candidate.checkpoint / "model.safetensors").stat().st_mode & 0o222

    assert not private_root.exists()


def test_private_snapshot_detects_runtime_byte_changes(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)

    with pytest.raises(ValidationError, match="runtime bytes changed"):
        with snapshot_local_act_runtime(candidate) as snapshot:
            model = snapshot.candidate.checkpoint / "model.safetensors"
            model.chmod(0o600)
            model.write_bytes(b"changed after snapshot")


class _Component:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def __call__(self, value: Any) -> Any:
        return value


def test_public_loader_binds_weights_queue_and_real_observation_pipeline(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    calls: dict[str, Any] = {}
    config = SimpleNamespace(
        type="act",
        chunk_size=100,
        n_action_steps=100,
        device="cpu",
        pretrained_path=None,
        pretrained_revision="unreviewed",
    )
    preprocessor = _Component()
    postprocessor = _Component()

    class Policy(_Component):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.evaluating = False

        def eval(self) -> None:
            self.evaluating = True

        def select_action(self, value: Any) -> np.ndarray:
            calls["selected"] = value
            return np.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)

    policy = Policy()

    def config_loader(path: Path, **kwargs: Any) -> Any:
        calls["config"] = (path, kwargs, os.environ.get("HF_HUB_OFFLINE"))
        return config

    class Metadata:
        def __init__(self, repo_id: str, **kwargs: Any) -> None:
            calls["metadata"] = (repo_id, kwargs)

    def make_policy(cfg: Any, **kwargs: Any) -> Any:
        calls["policy"] = (cfg.pretrained_path, cfg.n_action_steps, kwargs)
        return policy

    def make_processors(cfg: Any, **kwargs: Any) -> tuple[Any, Any]:
        calls["processors"] = (cfg, kwargs)
        return preprocessor, postprocessor

    def prepare(raw: dict[str, Any], device: Any, **kwargs: Any) -> dict[str, Any]:
        calls["prepared"] = (raw, str(device), kwargs)
        return {"model-ready": raw}

    api = LocalActPublicApi(
        config_from_pretrained=config_loader,
        dataset_metadata=Metadata,
        make_policy=make_policy,
        make_pre_post_processors=make_processors,
        prepare_observation_for_inference=prepare,
    )
    runtime = load_local_act_runtime(candidate, api=api)
    action = runtime.infer(_observation())
    runtime.reset()

    assert calls["config"] == (
        candidate.checkpoint,
        {"local_files_only": True},
        "1",
    )
    assert calls["metadata"] == (
        "test/local-act-dataset",
        {"root": candidate.dataset_root, "force_cache_sync": False},
    )
    pretrained_path, queued_actions, policy_arguments = calls["policy"]
    assert pretrained_path == str(candidate.checkpoint)
    assert queued_actions == QUEUE_ACTIONS
    assert isinstance(policy_arguments["ds_meta"], Metadata)
    assert policy_arguments["rename_map"] == {}
    assert calls["processors"][1] == {"pretrained_path": str(candidate.checkpoint)}
    raw, device, preparation = calls["prepared"]
    assert set(raw) == {
        "observation.state",
        "observation.images.front",
        "observation.images.up",
    }
    assert raw["observation.state"].dtype == np.float32
    assert device == "cpu"
    assert preparation == {"task": CANONICAL_TASK, "robot_type": "starai_viola"}
    assert policy.evaluating is True
    assert action == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert policy.resets == preprocessor.resets == postprocessor.resets == 1
    assert "HF_HUB_OFFLINE" not in os.environ


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(task="another task"), "task differs"),
        (
            lambda value: value.update(
                {"observation.images.front": np.zeros((480, 640, 3), dtype=np.float32)}
            ),
            "front image must be uint8",
        ),
        (lambda value: value.pop("observation.images.up"), "observation fields differ"),
        (lambda value: value.update({"observation.state": [0.0] * 6}), "seven values"),
    ],
)
def test_real_observation_validation_fails_closed(change, message: str) -> None:
    observation = _observation()
    change(observation)

    with pytest.raises(ValidationError, match=message):
        prepare_local_act_observation(observation)


def test_real_observation_is_contiguous_float_state_and_uint8_images() -> None:
    observation = _observation()
    observation["observation.images.front"] = observation[
        "observation.images.front"
    ][:, ::-1, :]

    prepared = prepare_local_act_observation(observation)

    assert prepared["observation.state"].shape == (7,)
    assert prepared["observation.state"].dtype == np.float32
    assert prepared["observation.images.front"].flags.c_contiguous
    assert "task" not in prepared
