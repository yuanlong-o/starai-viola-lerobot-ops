from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from viola_handoff import (
    RuntimeIdentity,
    SealRequest,
    accept_bundle,
    canonical_json_bytes,
    inventory_root,
    seal_bundle,
)
from viola_ops.errors import ValidationError
from viola_ops.policies import CANONICAL_TASK, POLICY_TOKENS, get_policy_spec
from viola_ops.policy_runtime import (
    AcceptedPolicyCandidate,
    LeRobotPublicApi,
    build_verification_payload,
    finite_action,
    inspect_candidate,
    load_lerobot_runtime,
    load_replay_dataset,
    verify_policy,
)


class FakeEvidence:
    def record(self, *, project: str, run_id: str, event: str, metadata: Any) -> str:
        del event, metadata
        return f"https://wandb.ai/test/{project}/runs/{run_id}"


class FastClock:
    def __init__(self, step: float = 0.001) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        result = self.value
        self.value += self.step
        return result


class FakeRuntime:
    def __init__(self, action: Any = (0.0,) * 7) -> None:
        self.action = action
        self.calls: list[dict[str, Any]] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def infer(self, observation: Any) -> Any:
        self.calls.append(dict(observation))
        return self.action


def _candidate(token: str) -> AcceptedPolicyCandidate:
    spec = get_policy_spec(token)
    digest = "a" * 64
    return AcceptedPolicyCandidate(
        bundle=SimpleNamespace(bundle_id="b" * 64, content_id="c" * 64),
        spec=spec,
        payload=MappingProxyType(
            {
                "policy": token,
                "task": CANONICAL_TASK,
                "queue_actions": 10,
                "feature_map": dict(spec.feature_map),
            }
        ),
        replay_manifest=MappingProxyType({"dataset_repo_id": "test/replay"}),
        checkpoint=Path("/accepted/checkpoint"),
        replay_dataset=Path("/accepted/replay"),
        dependencies=MappingProxyType({}),
        runtime_binding=MappingProxyType(
            {
                "checkpoint_inventory_sha256": digest,
                "checkpoint_config_sha256": digest,
                "processor_sha256": {"preprocessor": digest, "postprocessor": digest},
                "dependency_inventory_sha256": {},
            }
        ),
    )


def _observation() -> dict[str, Any]:
    return {
        "observation.state": [0.0] * 7,
        "observation.images.front": [[1]],
        "observation.images.up": [[2]],
        "task": CANONICAL_TASK,
    }


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_verify_runs_sample_twenty_warmups_and_all_two_hundred_trials(token: str) -> None:
    runtime = FakeRuntime()
    result = verify_policy(
        _candidate(token),
        _observation(),
        runtime_factory=lambda _candidate: runtime,
        clock=FastClock(),
    )

    assert result.eligible
    assert result.sample_action == (0.0,) * 7
    assert result.latency is not None
    assert result.latency["warmups"] == 20
    assert result.latency["trials"] == 200
    assert len(result.latency["samples_ms"]) == 200
    assert result.latency["p95_ms"] == pytest.approx(1.0)
    assert runtime.resets == 1
    assert len(runtime.calls) == 221
    if token == "vqbet":
        assert "observation.images.up" not in runtime.calls[0]
    else:
        assert "observation.images.up" in runtime.calls[0]


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_every_policy_rejects_a_nonfinite_or_wrong_width_action(token: str) -> None:
    action = [0.0] * 6 + [float("nan")] if token != "groot" else [0.0] * 8
    result = verify_policy(
        _candidate(token),
        _observation(),
        runtime_factory=lambda _candidate: FakeRuntime(action),
    )
    assert result.status == "ineligible_pc_runtime"
    assert result.failure is not None
    assert result.failure["code"] == "inference_failed"
    assert result.latency is None


def test_latency_at_or_above_three_hundred_ms_is_terminal() -> None:
    result = verify_policy(
        _candidate("act"),
        _observation(),
        runtime_factory=lambda _candidate: FakeRuntime(),
        clock=FastClock(step=0.301),
    )
    assert result.status == "ineligible_pc_runtime"
    assert result.failure is not None
    assert result.failure["code"] == "latency_exceeded"
    assert result.latency is not None
    assert len(result.latency["samples_ms"]) == 200


def test_verification_payload_matches_repo_b_field_shape() -> None:
    candidate = _candidate("pi0_fast")
    result = verify_policy(
        candidate,
        _observation(),
        runtime_factory=lambda _candidate: FakeRuntime(),
        clock=FastClock(),
    )
    payload = build_verification_payload(
        candidate,
        result,
        repo={
            "commit": "d" * 40,
            "clean": True,
            "hostname": "pc-a",
            "python": "3.12.13",
        },
        wandb={
            "run_id": "verify-pi0-fast",
            "url": "https://wandb.ai/test/project/runs/verify-pi0-fast",
        },
        verified_at="2026-08-13T00:00:00+00:00",
    )
    assert set(payload) == {
        "schema_version",
        "kind",
        "status",
        "bundle_id",
        "content_id",
        "policy",
        "checkpoint",
        "queue_actions",
        "task",
        "feature_map",
        "runtime_binding",
        "sample_action",
        "latency",
        "failure_reason",
        "failure",
        "repo",
        "wandb",
        "verified_at",
    }
    assert payload["queue_actions"] == 10
    assert payload["kind"] == "pc_runtime_verification"


def test_replay_loader_uses_exact_public_lerobot_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeDataset:
        episodes = tuple(range(27, 34))

        def __init__(self, repo_id: str, **kwargs: Any) -> None:
            calls.append((repo_id, kwargs))

        def __len__(self) -> int:
            return 6_079

    import lerobot.datasets.lerobot_dataset as dataset_module

    monkeypatch.setattr(dataset_module, "LeRobotDataset", FakeDataset)
    candidate = _candidate("act")
    loaded = load_replay_dataset(candidate)
    assert isinstance(loaded, FakeDataset)
    assert calls == [
        (
            "test/replay",
            {
                "root": candidate.replay_dataset,
                "episodes": list(range(27, 34)),
                "video_backend": "pyav",
                "return_uint8": True,
            },
        )
    ]


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_runtime_loader_uses_public_lerobot_factories_and_local_dependencies(
    token: str,
    tmp_path: Path,
) -> None:
    candidate = _candidate(token)
    checkpoint = tmp_path / "checkpoint"
    replay = tmp_path / "replay"
    checkpoint.mkdir()
    replay.mkdir()
    dependencies: dict[str, Path] = {}
    for dependency_id in candidate.spec.dependency_ids:
        dependency = tmp_path / dependency_id
        dependency.mkdir()
        dependencies[dependency_id] = dependency
    candidate = replace(
        candidate,
        checkpoint=checkpoint,
        replay_dataset=replay,
        dependencies=MappingProxyType(dependencies),
        replay_manifest=MappingProxyType({"dataset_repo_id": "test/replay"}),
    )

    calls: dict[str, Any] = {}
    config = SimpleNamespace(type=token)
    setattr(config, candidate.spec.queue_config_field, 10)

    def config_loader(path: Path, **kwargs: Any) -> Any:
        calls["config"] = (path, kwargs, _offline_environment())
        return config

    class Metadata:
        features = {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
            "observation.images.up": {"dtype": "video", "shape": [3, 480, 640]},
            "action": {"dtype": "float32", "shape": [7]},
        }
        stats = {key: {} for key in features}

        def __init__(self, repo_id: str, **kwargs: Any) -> None:
            calls["metadata"] = (repo_id, kwargs, _offline_environment())

    class Component:
        def __init__(self) -> None:
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def __call__(self, value: Any) -> Any:
            return value

    class Policy(Component):
        def eval(self) -> None:
            calls["eval"] = True

        def select_action(self, observation: Any) -> list[float]:
            calls["observation"] = observation
            return [0.0] * 7

    policy = Policy()
    preprocessor = Component()
    postprocessor = Component()

    def make_policy(cfg: Any, **kwargs: Any) -> Any:
        calls["policy"] = (cfg, kwargs, _offline_environment())
        return policy

    def make_processors(cfg: Any, **kwargs: Any) -> tuple[Any, Any]:
        calls["processors"] = (cfg, kwargs, _offline_environment())
        return preprocessor, postprocessor

    runtime = load_lerobot_runtime(
        candidate,
        api=LeRobotPublicApi(
            config_from_pretrained=config_loader,
            dataset_metadata=Metadata,
            make_policy=make_policy,
            make_pre_post_processors=make_processors,
        ),
    )
    runtime.reset()

    assert calls["config"][:2] == (checkpoint, {"local_files_only": True})
    assert calls["metadata"][:2] == (
        "test/replay",
        {"root": replay, "force_cache_sync": False},
    )
    metadata = calls["policy"][1]["ds_meta"]
    assert ("observation.images.up" in metadata.features) is (token != "vqbet")
    assert calls["policy"][1]["rename_map"] == dict(candidate.spec.feature_map)
    assert calls["processors"][1]["pretrained_path"] == str(checkpoint)
    assert calls["processors"][1]["preprocessor_overrides"] == _expected_overrides(
        token, dependencies
    )
    assert calls["eval"] is True
    assert policy.resets == preprocessor.resets == postprocessor.resets == 1
    assert all(item[2]["HF_HUB_OFFLINE"] == "1" for item in calls.values() if isinstance(item, tuple))

    if token == "smolvla":
        assert config.vlm_model_name == str(dependencies["smolvlm_model"])
    elif token == "pi0_fast":
        assert config.text_tokenizer_name == str(dependencies["paligemma_tokenizer"])
        assert config.action_tokenizer_name == str(dependencies["fast_action_tokenizer"])
    elif token == "groot":
        assert config.base_model_path == str(dependencies["groot_base_model"])
        assert calls["policy"][2]["HF_HOME"] == str(dependencies["hf_hub_cache"])
    assert os.environ.get("HF_HUB_OFFLINE") != "1"


@pytest.mark.parametrize("token", POLICY_TOKENS)
@pytest.mark.parametrize(
    ("qualification", "valid_checkpoint_lineage", "error_match"),
    (
        ("current_release_bound", True, None),
        ("legacy_hash_bound", True, None),
        ("qualified", True, "training qualification"),
        ("current_release_bound", False, "checkpoint evaluation hash"),
    ),
    ids=("current", "legacy", "unknown-qualification", "wrong-evaluation-checkpoint"),
)
def test_inspect_candidate_enforces_receiver_local_lineage(
    tmp_path: Path,
    token: str,
    qualification: str,
    valid_checkpoint_lineage: bool,
    error_match: str | None,
) -> None:
    checkpoint = tmp_path / f"checkpoint-{token}"
    checkpoint.mkdir()
    spec = get_policy_spec(token)
    config = {"type": token, spec.queue_config_field: 10}
    (checkpoint / "config.json").write_text(
        __import__("json").dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"fixture-model")
    (checkpoint / "policy_preprocessor.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text("{}\n", encoding="utf-8")
    replay = tmp_path / f"replay-{token}"
    replay.mkdir()
    (replay / "README.txt").write_text("fixture", encoding="utf-8")

    artifacts: dict[str, Path] = {
        "selected_checkpoint": checkpoint,
        "replay_dataset": replay,
    }
    dependency_payload: dict[str, dict[str, str]] = {}
    for dependency_id in spec.dependency_ids:
        artifact_name = f"dependency_{dependency_id}"
        root = tmp_path / f"{token}-{artifact_name}"
        root.mkdir()
        (root / "fixture.txt").write_text(dependency_id, encoding="utf-8")
        artifacts[artifact_name] = root
        dependency_payload[dependency_id] = {
            "artifact": artifact_name,
            "inventory_sha256": inventory_root(root)["inventory_sha256"],
            "relative_path": ".",
        }

    dataset_manifest = "1" * 64
    training = {
        "qualification": qualification,
        "run_id": f"train-{token}",
        "repository_commit": "2" * 40,
        "training_config_sha256": _sha256(checkpoint / "config.json"),
        "dataset_release_id": "dataset-v1",
        "dataset_manifest_sha256": dataset_manifest,
        "persisted_queue_actions": 10,
    }
    replay_payload = {
        "schema_version": 1,
        "dataset_release_id": "dataset-v1",
        "dataset_manifest_sha256": dataset_manifest,
        "dataset_inventory_sha256": inventory_root(replay)["inventory_sha256"],
        "dataset_artifact": "replay_dataset",
        "dataset_relative_path": ".",
        "dataset_repo_id": "test/replay",
        "episodes": list(range(27, 34)),
        "expected_frames": 6079,
        "task": CANONICAL_TASK,
    }
    payload_root = tmp_path / f"payload-{token}"
    payload_root.mkdir()
    replay_path = payload_root / "replay_manifest.json"
    replay_path.write_bytes(canonical_json_bytes(replay_payload))
    checkpoint_inventory = inventory_root(checkpoint)["inventory_sha256"]
    checkpoint_evaluation_sha256 = _repo_b_checkpoint_sha256(checkpoint)
    candidate_payload = {
        "schema_version": 1,
        "policy": token,
        "checkpoint_artifact": "selected_checkpoint",
        "checkpoint_relative_path": ".",
        "checkpoint_inventory_sha256": checkpoint_inventory,
        "dependency_artifacts": dependency_payload,
        "training_lineage": training,
        "dataset_release_id": "dataset-v1",
        "evaluation_sha256": "3" * 64,
        "task": CANONICAL_TASK,
        "queue_actions": 10,
        "feature_map": dict(spec.feature_map),
        "wandb_url": f"https://wandb.ai/test/project/runs/eval-{token}",
    }
    (payload_root / "policy_candidate.json").write_bytes(canonical_json_bytes(candidate_payload))
    lineage = {
        "policy": token,
        "dataset_release_id": "dataset-v1",
        "dataset_release_manifest_sha256": dataset_manifest,
        "evaluation_sha256": "3" * 64,
        "evaluation_wandb_run_id": f"eval-{token}",
        "training_wandb_run_id": f"train-{token}",
        "checkpoint_inventory_sha256": checkpoint_inventory,
        "checkpoint_evaluation_sha256": (
            checkpoint_evaluation_sha256 if valid_checkpoint_lineage else "4" * 64
        ),
        "training": training,
        "dependency_inventories": {
            name: dependency_payload[name]["inventory_sha256"] for name in spec.dependency_ids
        },
        "replay_manifest_sha256": _sha256(replay_path),
        "replay_dataset_inventory_sha256": replay_payload["dataset_inventory_sha256"],
    }
    source = seal_bundle(
        SealRequest(
            root=tmp_path / "source",
            kind="policy_candidate",
            experiment="viola-test",
            subject=token,
            producer=_identity("pc_b", "5" * 40),
            lineage=lineage,
            wandb_project="project",
            payload_dir=payload_root,
            artifact_roots=artifacts,
        ),
        evidence_logger=FakeEvidence(),
    )
    accepted = accept_bundle(
        source.path,
        tmp_path / "accepted",
        receiver=_identity("pc_a", "6" * 40),
        evidence_logger=FakeEvidence(),
    )

    if error_match is not None:
        with pytest.raises(ValidationError, match=error_match):
            inspect_candidate(accepted.path)
        return

    candidate = inspect_candidate(accepted.path)
    assert candidate.policy == token
    assert candidate.checkpoint.is_relative_to(tmp_path / "accepted" / "artifacts")
    assert set(candidate.dependencies) == set(spec.dependency_ids)
    assert candidate.runtime_binding["checkpoint_config_sha256"] == _sha256(
        checkpoint / "config.json"
    )


def test_finite_action_rejects_boolean_and_width() -> None:
    with pytest.raises(ValidationError, match="boolean"):
        finite_action([0.0] * 6 + [True])
    with pytest.raises(ValidationError, match="exactly seven"):
        finite_action([0.0] * 6)


def _identity(role: str, commit: str) -> RuntimeIdentity:
    return RuntimeIdentity(
        role=role,
        repository_commit=commit,
        repository_clean=True,
        hostname=role,
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_b_checkpoint_sha256(root: Path) -> str:
    """Mirror Repo B's path-independent ``snapshot_inventory`` digest."""

    files = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _offline_environment() -> dict[str, str | None]:
    return {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "HF_HOME": os.environ.get("HF_HOME"),
    }


def _expected_overrides(token: str, dependencies: dict[str, Path]) -> dict[str, Any]:
    if token == "smolvla":
        return {
            "tokenizer_processor": {
                "tokenizer_name": str(dependencies["smolvlm_model"]),
            }
        }
    if token in {"pi0", "pi05"}:
        return {
            "tokenizer_processor": {
                "tokenizer_name": str(dependencies["paligemma_tokenizer"]),
            }
        }
    if token == "pi0_fast":
        paligemma = str(dependencies["paligemma_tokenizer"])
        return {
            "tokenizer_processor": {"tokenizer_name": paligemma},
            "action_tokenizer_processor": {
                "action_tokenizer_name": str(dependencies["fast_action_tokenizer"]),
                "paligemma_tokenizer_name": paligemma,
            },
        }
    if token == "groot":
        return {
            "groot_n1_7_vlm_encode_v1": {
                "model_name": str(dependencies["cosmos_reason2_processor"]),
            }
        }
    return {}
