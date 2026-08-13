"""Accepted policy inspection and disconnected LeRobot inference.

Nothing in this module imports a robot, camera, serial, or motor package.  The
production loader uses LeRobot's public checkpoint, policy, processor, and
dataset-metadata APIs.  Tests can inject a tiny runtime with the same three
methods instead of pretending that a synthetic model is a qualified policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType, SimpleNamespace
from typing import Any, Protocol
from urllib.parse import urlsplit

from viola_handoff import HandoffError, VerifiedBundle, canonical_json_bytes, inspect_bundle

from .errors import ValidationError
from .policies import (
    CANONICAL_TASK,
    QUEUE_ACTIONS,
    PolicySpec,
    get_policy_spec,
)

POLICY_CANDIDATE_FILENAME = "policy_candidate.json"
REPLAY_MANIFEST_FILENAME = "replay_manifest.json"
CHECKPOINT_ARTIFACT = "selected_checkpoint"
REPLAY_DATASET_ARTIFACT = "replay_dataset"
EXPECTED_HELDOUT_EPISODES = (27, 28, 29, 30, 31, 32, 33)
EXPECTED_HELDOUT_FRAMES = 6_079
WARMUP_COUNT = 20
TIMED_COUNT = 200
LATENCY_THRESHOLD_MS = 300.0
SHADOW_SCHEMA_SHA256 = "778a3cbb73355764626a2f1b2bf915d347951d118287cc846e7a30e9a09cfd90"

_CANDIDATE_KEYS = {
    "schema_version",
    "policy",
    "checkpoint_artifact",
    "checkpoint_relative_path",
    "checkpoint_inventory_sha256",
    "dependency_artifacts",
    "training_lineage",
    "dataset_release_id",
    "evaluation_sha256",
    "task",
    "queue_actions",
    "feature_map",
    "wandb_url",
}
_REPLAY_KEYS = {
    "schema_version",
    "dataset_release_id",
    "dataset_manifest_sha256",
    "dataset_inventory_sha256",
    "dataset_artifact",
    "dataset_relative_path",
    "dataset_repo_id",
    "episodes",
    "expected_frames",
    "task",
}
_TRAINING_KEYS = {
    "qualification",
    "run_id",
    "repository_commit",
    "training_config_sha256",
    "dataset_release_id",
    "dataset_manifest_sha256",
    "persisted_queue_actions",
}
_LINEAGE_KEYS = {
    "policy",
    "dataset_release_id",
    "dataset_release_manifest_sha256",
    "evaluation_sha256",
    "evaluation_wandb_run_id",
    "training_wandb_run_id",
    "checkpoint_inventory_sha256",
    "checkpoint_evaluation_sha256",
    "training",
    "dependency_inventories",
    "replay_manifest_sha256",
    "replay_dataset_inventory_sha256",
}
_REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
_TRAINING_QUALIFICATIONS = frozenset({"current_release_bound", "legacy_hash_bound"})


class PolicyRuntime(Protocol):
    """The intentionally tiny interface shared by real and injected runtimes."""

    def reset(self) -> None:
        """Clear every policy, processor, and action-queue state."""

    def infer(self, observation: Mapping[str, Any]) -> Sequence[float] | Any:
        """Return one postprocessed action for one observation."""


@dataclass(frozen=True, slots=True)
class LeRobotPublicApi:
    """Injectable handles to the four public LeRobot factories we use."""

    config_from_pretrained: Callable[..., Any]
    dataset_metadata: Callable[..., Any]
    make_policy: Callable[..., Any]
    make_pre_post_processors: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class AcceptedPolicyCandidate:
    """A receiver-local, immutable candidate safe for disconnected loading."""

    bundle: VerifiedBundle
    spec: PolicySpec
    payload: Mapping[str, Any]
    replay_manifest: Mapping[str, Any]
    checkpoint: Path
    replay_dataset: Path
    dependencies: Mapping[str, Path]
    runtime_binding: Mapping[str, Any]

    @property
    def bundle_id(self) -> str:
        return self.bundle.bundle_id

    @property
    def content_id(self) -> str:
        return self.bundle.content_id

    @property
    def policy(self) -> str:
        return self.spec.token


@dataclass(frozen=True, slots=True)
class RuntimeVerification:
    """Result of a real or injected PC-A runtime check."""

    status: str
    sample_action: tuple[float, ...] | None
    latency: Mapping[str, Any] | None
    failure_reason: str | None
    failure: Mapping[str, Any] | None

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"


def inspect_candidate(path: str | Path) -> AcceptedPolicyCandidate:
    """Inspect a PC-A-accepted candidate and return only receiver-local paths."""

    try:
        bundle = inspect_bundle(
            path,
            verify_artifacts=True,
            expected_kind="policy_candidate",
            required_permission="disconnected_only",
        )
    except HandoffError as exc:
        raise ValidationError(f"invalid policy candidate bundle: {exc}") from exc

    _require_pc_a_acceptance(bundle)
    _require_exact_payload_files(bundle, {POLICY_CANDIDATE_FILENAME, REPLAY_MANIFEST_FILENAME})
    payload = _load_canonical_object(
        bundle.payload_file(POLICY_CANDIDATE_FILENAME), "policy candidate"
    )
    _exact_keys(payload, _CANDIDATE_KEYS, "policy candidate")
    if payload["schema_version"] != 1:
        raise ValidationError("policy candidate schema_version must be 1")
    spec = get_policy_spec(_text(payload["policy"], "policy candidate policy"))

    if payload["checkpoint_artifact"] != CHECKPOINT_ARTIFACT:
        raise ValidationError("policy candidate must name selected_checkpoint")
    if payload["checkpoint_relative_path"] != ".":
        raise ValidationError("policy candidate checkpoint path must be exactly '.'")
    _text(payload["dataset_release_id"], "policy candidate dataset_release_id")
    if payload["task"] != CANONICAL_TASK:
        raise ValidationError("policy candidate task differs from the benchmark task")
    if payload["queue_actions"] != QUEUE_ACTIONS:
        raise ValidationError("policy candidate must persist a ten-action queue")
    if payload["feature_map"] != dict(spec.feature_map):
        raise ValidationError(f"{spec.token} feature map differs from the deployment registry")
    _sha256(payload["checkpoint_inventory_sha256"], "checkpoint inventory")
    _sha256(payload["evaluation_sha256"], "policy candidate evaluation_sha256")
    evaluation_run_id = _wandb_url(payload["wandb_url"], "policy candidate wandb_url")

    artifact_by_name = {item["name"]: item for item in bundle.manifest["artifacts"]}
    checkpoint_entry = artifact_by_name.get(CHECKPOINT_ARTIFACT)
    replay_entry = artifact_by_name.get(REPLAY_DATASET_ARTIFACT)
    if checkpoint_entry is None or replay_entry is None:
        raise ValidationError("policy candidate lacks selected_checkpoint or replay_dataset")
    if payload["checkpoint_inventory_sha256"] != checkpoint_entry["inventory_sha256"]:
        raise ValidationError("checkpoint inventory differs from the signed manifest")

    checkpoint = _accepted_artifact(bundle, CHECKPOINT_ARTIFACT)
    replay_dataset = _accepted_artifact(bundle, REPLAY_DATASET_ARTIFACT)
    for filename in _REQUIRED_CHECKPOINT_FILES:
        target = checkpoint / filename
        if target.is_symlink() or not target.is_file() or target.stat().st_size <= 0:
            raise ValidationError(f"accepted checkpoint is missing nonempty {filename}")

    checkpoint_config = _load_json_object(checkpoint / "config.json", "checkpoint config")
    if checkpoint_config.get("type") != spec.token:
        raise ValidationError("checkpoint config type differs from the candidate policy")
    if checkpoint_config.get(spec.queue_config_field) != QUEUE_ACTIONS:
        raise ValidationError(
            f"checkpoint {spec.queue_config_field} must itself equal {QUEUE_ACTIONS}"
        )

    training = _object(payload["training_lineage"], "training_lineage")
    _exact_keys(training, _TRAINING_KEYS, "training_lineage")
    if training["persisted_queue_actions"] != QUEUE_ACTIONS:
        raise ValidationError("training lineage did not persist the ten-action queue")
    qualification = _text(training["qualification"], "training qualification")
    if qualification not in _TRAINING_QUALIFICATIONS:
        raise ValidationError(
            "training qualification must be current_release_bound or legacy_hash_bound"
        )
    training_run_id = _text(training["run_id"], "training run_id")
    training_commit = _text(training["repository_commit"], "training repository commit")
    if len(training_commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in training_commit
    ):
        raise ValidationError("training repository commit must be a full Git object ID")
    if training["dataset_release_id"] != payload["dataset_release_id"]:
        raise ValidationError("training and candidate dataset release IDs differ")
    if training["training_config_sha256"] != _sha256_file(checkpoint / "config.json"):
        raise ValidationError("training config hash differs from accepted config.json")
    _sha256(training["dataset_manifest_sha256"], "training dataset manifest")

    replay = _load_canonical_object(
        bundle.payload_file(REPLAY_MANIFEST_FILENAME), "candidate replay manifest"
    )
    _exact_keys(replay, _REPLAY_KEYS, "candidate replay manifest")
    if replay["schema_version"] != 1:
        raise ValidationError("replay manifest schema_version must be 1")
    if replay["dataset_release_id"] != payload["dataset_release_id"]:
        raise ValidationError("replay and candidate dataset release IDs differ")
    if replay["dataset_artifact"] != REPLAY_DATASET_ARTIFACT:
        raise ValidationError("replay manifest must name replay_dataset")
    if replay["dataset_relative_path"] != ".":
        raise ValidationError("replay dataset path must be exactly '.'")
    if replay["dataset_inventory_sha256"] != replay_entry["inventory_sha256"]:
        raise ValidationError("replay dataset inventory differs from the signed artifact")
    if replay["episodes"] != list(EXPECTED_HELDOUT_EPISODES):
        raise ValidationError("replay manifest must bind held-out episodes 27 through 33")
    if replay["expected_frames"] != EXPECTED_HELDOUT_FRAMES:
        raise ValidationError("replay manifest must bind all 6079 held-out frames")
    if replay["task"] != CANONICAL_TASK:
        raise ValidationError("replay manifest task differs from the benchmark task")
    _text(replay["dataset_repo_id"], "replay dataset_repo_id")
    _sha256(replay["dataset_manifest_sha256"], "replay dataset manifest")

    dependencies = _accepted_dependencies(
        bundle,
        payload["dependency_artifacts"],
        spec=spec,
        artifact_by_name=artifact_by_name,
    )
    expected_artifacts = {
        CHECKPOINT_ARTIFACT,
        REPLAY_DATASET_ARTIFACT,
        *(payload["dependency_artifacts"][name]["artifact"] for name in spec.dependency_ids),
    }
    if set(artifact_by_name) != expected_artifacts:
        raise ValidationError(
            "policy candidate artifact names differ; "
            f"expected={sorted(expected_artifacts)}, actual={sorted(artifact_by_name)}"
        )

    lineage = _object(bundle.manifest["lineage"], "policy candidate lineage")
    _exact_keys(lineage, _LINEAGE_KEYS, "policy candidate lineage")
    for field in (
        "dataset_release_manifest_sha256",
        "checkpoint_evaluation_sha256",
    ):
        _sha256(lineage[field], f"policy candidate lineage {field}")
    checkpoint_evaluation_sha256 = _checkpoint_evaluation_sha256(checkpoint_entry)
    if lineage["checkpoint_evaluation_sha256"] != checkpoint_evaluation_sha256:
        raise ValidationError(
            "policy candidate checkpoint evaluation hash differs from accepted checkpoint"
        )
    dependency_inventories = {
        name: payload["dependency_artifacts"][name]["inventory_sha256"]
        for name in spec.dependency_ids
    }
    expected_lineage = {
        "policy": spec.token,
        "dataset_release_id": payload["dataset_release_id"],
        "evaluation_sha256": payload["evaluation_sha256"],
        "evaluation_wandb_run_id": evaluation_run_id,
        "training_wandb_run_id": training_run_id,
        "checkpoint_inventory_sha256": payload["checkpoint_inventory_sha256"],
        "training": training,
        "dependency_inventories": dependency_inventories,
        "replay_manifest_sha256": _sha256_file(bundle.payload_file(REPLAY_MANIFEST_FILENAME)),
        "replay_dataset_inventory_sha256": replay["dataset_inventory_sha256"],
    }
    for field, expected in expected_lineage.items():
        if lineage[field] != expected:
            raise ValidationError(f"policy candidate lineage mismatch: {field}")
    if lineage["dataset_release_manifest_sha256"] != replay["dataset_manifest_sha256"]:
        raise ValidationError("candidate and replay dataset manifest hashes differ")
    if lineage["dataset_release_manifest_sha256"] != training["dataset_manifest_sha256"]:
        raise ValidationError("candidate and training dataset manifest hashes differ")

    runtime_binding = {
        "checkpoint_inventory_sha256": payload["checkpoint_inventory_sha256"],
        "checkpoint_config_sha256": _sha256_file(checkpoint / "config.json"),
        "processor_sha256": {
            "preprocessor": _sha256_file(checkpoint / "policy_preprocessor.json"),
            "postprocessor": _sha256_file(checkpoint / "policy_postprocessor.json"),
        },
        "dependency_inventory_sha256": dependency_inventories,
    }
    return AcceptedPolicyCandidate(
        bundle=bundle,
        spec=spec,
        payload=MappingProxyType(dict(payload)),
        replay_manifest=MappingProxyType(dict(replay)),
        checkpoint=checkpoint,
        replay_dataset=replay_dataset,
        dependencies=MappingProxyType(dependencies),
        runtime_binding=MappingProxyType(runtime_binding),
    )


class LeRobotPolicyRuntime:
    """Thin adapter around public LeRobot policy and processor methods."""

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

    def reset(self) -> None:
        for component in (self._policy, self._preprocessor, self._postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def infer(self, observation: Mapping[str, Any]) -> tuple[float, ...]:
        import torch

        processed = self._preprocessor(dict(observation))
        with torch.inference_mode():
            action = self._policy.select_action(processed)
        return finite_action(self._postprocessor(action))


def load_lerobot_runtime(
    candidate: AcceptedPolicyCandidate,
    *,
    api: LeRobotPublicApi | None = None,
) -> LeRobotPolicyRuntime:
    """Freshly load an accepted candidate using only LeRobot public APIs.

    Hugging Face model access is forced offline.  Every external model/tokenizer
    path is replaced by its receiver-local accepted dependency before loading.
    W&B remains online; the offline flags are deliberately HF-specific.
    """

    public_api = api or _public_lerobot_api()

    with _huggingface_offline(candidate):
        config = public_api.config_from_pretrained(
            candidate.checkpoint,
            local_files_only=True,
        )
        if config.type != candidate.policy:
            raise ValidationError("LeRobot loaded a policy type different from the candidate")
        if getattr(config, candidate.spec.queue_config_field, None) != QUEUE_ACTIONS:
            raise ValidationError("LeRobot changed the persisted ten-action queue")
        _bind_dependency_paths(config, candidate)

        metadata = public_api.dataset_metadata(
            str(candidate.replay_manifest["dataset_repo_id"]),
            root=candidate.replay_dataset,
            force_cache_sync=False,
        )
        if candidate.policy == "vqbet":
            metadata = _front_camera_metadata(metadata)

        policy = public_api.make_policy(
            config,
            ds_meta=metadata,
            rename_map=dict(candidate.spec.feature_map),
        )
        policy.eval()
        preprocessor, postprocessor = public_api.make_pre_post_processors(
            config,
            pretrained_path=str(candidate.checkpoint),
            preprocessor_overrides=_processor_overrides(candidate),
        )
    return LeRobotPolicyRuntime(policy, preprocessor, postprocessor)


def _public_lerobot_api() -> LeRobotPublicApi:
    """Import public LeRobot entry points only when a real load is requested."""

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    return LeRobotPublicApi(
        config_from_pretrained=PreTrainedConfig.from_pretrained,
        dataset_metadata=LeRobotDatasetMetadata,
        make_policy=make_policy,
        make_pre_post_processors=make_pre_post_processors,
    )


def load_replay_dataset(candidate: AcceptedPolicyCandidate) -> Any:
    """Open the exact accepted held-out data through LeRobot's public loader."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        with _huggingface_offline(candidate):
            dataset = LeRobotDataset(
                str(candidate.replay_manifest["dataset_repo_id"]),
                root=candidate.replay_dataset,
                episodes=list(EXPECTED_HELDOUT_EPISODES),
                video_backend="pyav",
                return_uint8=True,
            )
    except Exception as exc:
        raise ValidationError(
            "could not load the accepted replay dataset through LeRobot: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if tuple(getattr(dataset, "episodes", ())) != EXPECTED_HELDOUT_EPISODES:
        raise ValidationError("LeRobot did not preserve held-out episodes 27 through 33")
    if len(dataset) != EXPECTED_HELDOUT_FRAMES:
        raise ValidationError(
            f"accepted replay dataset has {len(dataset)} frames, expected {EXPECTED_HELDOUT_FRAMES}"
        )
    return dataset


def load_verification_observation(candidate: AcceptedPolicyCandidate) -> Mapping[str, Any]:
    """Decode one real accepted frame for runtime verification; never synthesize evidence."""

    dataset = load_replay_dataset(candidate)
    try:
        observation = dataset[0]
    except Exception as exc:
        raise ValidationError(
            f"could not decode the first accepted replay frame: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(observation, Mapping):
        raise ValidationError("LeRobot replay frame is not a mapping")
    return observation


def verify_policy(
    candidate: AcceptedPolicyCandidate,
    observation: Mapping[str, Any],
    *,
    runtime_factory: Callable[[AcceptedPolicyCandidate], PolicyRuntime] = load_lerobot_runtime,
    clock: Callable[[], float] = time.perf_counter,
) -> RuntimeVerification:
    """Fresh-load, sample, and benchmark one candidate without hardware access."""

    try:
        runtime = runtime_factory(candidate)
    except Exception as exc:
        return _exception_verification("runtime_load_failed", "runtime_load", exc)

    try:
        prepared = prepare_observation(candidate.spec, observation)
    except Exception as exc:
        return _exception_verification("observation_contract_failed", "synthetic_observation", exc)

    try:
        runtime.reset()
        sample = finite_action(runtime.infer(prepared))
    except Exception as exc:
        return _exception_verification("inference_failed", "sample_inference", exc)

    try:
        for _ in range(WARMUP_COUNT):
            finite_action(runtime.infer(prepared))
        samples: list[float] = []
        for _ in range(TIMED_COUNT):
            started = clock()
            finite_action(runtime.infer(prepared))
            elapsed_ms = (clock() - started) * 1_000.0
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                raise ValidationError("latency clock returned a negative or non-finite duration")
            samples.append(elapsed_ms)
    except Exception as exc:
        return _exception_verification("benchmark_failed", "latency_benchmark", exc)

    latency = {
        "warmups": WARMUP_COUNT,
        "trials": TIMED_COUNT,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "threshold_ms": LATENCY_THRESHOLD_MS,
        "samples_ms": samples,
    }
    if latency["p95_ms"] >= LATENCY_THRESHOLD_MS:
        reason = (
            f"runtime p95 latency {latency['p95_ms']:.6f} ms is not below "
            f"{LATENCY_THRESHOLD_MS:.1f} ms"
        )
        failure = {
            "code": "latency_exceeded",
            "recoverable": False,
            "evidence": {
                "p95_ms": latency["p95_ms"],
                "threshold_ms": LATENCY_THRESHOLD_MS,
                "samples_sha256": hashlib.sha256(
                    canonical_json_bytes(samples)
                ).hexdigest(),
            },
        }
        return RuntimeVerification("ineligible_pc_runtime", sample, latency, reason, failure)
    return RuntimeVerification("eligible", sample, latency, None, None)


def build_verification_payload(
    candidate: AcceptedPolicyCandidate,
    verification: RuntimeVerification,
    *,
    repo: Mapping[str, Any],
    wandb: Mapping[str, str],
    verified_at: str | None = None,
) -> dict[str, Any]:
    """Render the exact verification object consumed by Repo B."""

    return {
        "schema_version": 1,
        "kind": "pc_runtime_verification",
        "status": verification.status,
        "bundle_id": candidate.bundle_id,
        "content_id": candidate.content_id,
        "policy": candidate.policy,
        "checkpoint": CHECKPOINT_ARTIFACT,
        "queue_actions": QUEUE_ACTIONS,
        "task": CANONICAL_TASK,
        "feature_map": dict(candidate.spec.feature_map),
        "runtime_binding": _mutable_json(candidate.runtime_binding),
        "sample_action": (
            None if verification.sample_action is None else list(verification.sample_action)
        ),
        "latency": None if verification.latency is None else _mutable_json(verification.latency),
        "failure_reason": verification.failure_reason,
        "failure": None if verification.failure is None else _mutable_json(verification.failure),
        "repo": dict(repo),
        "wandb": dict(wandb),
        "verified_at": verified_at or datetime.now(UTC).isoformat(),
    }


def verification_sync_binding(
    candidate: AcceptedPolicyCandidate,
    verification_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return Repo B's exact immutable binding for a verification W&B run."""

    return {
        "bundle_id": candidate.bundle_id,
        "content_id": candidate.content_id,
        "policy": candidate.policy,
        "status": verification_payload["status"],
        "repo_commit": verification_payload["repo"]["commit"],
        "runtime_binding": _mutable_json(candidate.runtime_binding),
    }


def build_sync_receipt(
    *,
    operation: str,
    evidence_file: str,
    evidence_sha256: str,
    wandb: Mapping[str, str],
    binding: Mapping[str, Any],
    synced_at: str | None = None,
) -> dict[str, Any]:
    """Build the canonical sidecar proving an evidence file finished online sync."""

    _sha256(evidence_sha256, "evidence_sha256")
    if Path(evidence_file).name != evidence_file:
        raise ValidationError("evidence_file must be a basename")
    return {
        "schema_version": 1,
        "operation": _text(operation, "sync operation"),
        "evidence_file": evidence_file,
        "evidence_sha256": evidence_sha256,
        "wandb": dict(wandb),
        "binding": _mutable_json(binding),
        "synced_at": synced_at or datetime.now(UTC).isoformat(),
    }


def build_runtime_terminal_payload(
    candidate: AcceptedPolicyCandidate,
    verification_payload: Mapping[str, Any],
    *,
    verification_sha256: str,
) -> dict[str, Any]:
    """Build a typed runtime terminal that can never grant shadow or motion."""

    if verification_payload.get("status") != "ineligible_pc_runtime":
        raise ValidationError("runtime terminal requires ineligible_pc_runtime verification")
    if verification_payload.get("bundle_id") != candidate.bundle_id:
        raise ValidationError("runtime terminal verification belongs to another candidate")
    _sha256(verification_sha256, "runtime verification sha256")
    wandb = _object(verification_payload.get("wandb"), "verification W&B identity")
    return {
        "schema_version": 1,
        "kind": "shadow_evidence",
        "status": "ineligible_pc_runtime",
        "bundle_id": candidate.bundle_id,
        "content_id": candidate.content_id,
        "policy": candidate.policy,
        "setup_hashes": None,
        "frozen_state_capture": None,
        "runtime_binding": _mutable_json(candidate.runtime_binding),
        "failure": _mutable_json(verification_payload["failure"]),
        "verification": {
            "sha256": verification_sha256,
            "run_id": wandb["run_id"],
            "url": wandb["url"],
        },
        "summary": None,
        "trace": None,
        "trace_file_sha256": None,
        "videos": {},
        "repo": dict(verification_payload["repo"]),
        "wandb": dict(wandb),
        "completed_at": verification_payload["verified_at"],
    }


def build_runtime_terminal_lineage(
    candidate: AcceptedPolicyCandidate,
    terminal_payload: Mapping[str, Any],
    *,
    verification_sha256: str,
    verification_sync_sha256: str,
) -> dict[str, Any]:
    """Return Repo B's exact manifest lineage for a runtime terminal bundle."""

    _sha256(verification_sha256, "runtime verification sha256")
    _sha256(verification_sync_sha256, "runtime verification sync sha256")
    return {
        "policy_candidate_bundle_id": candidate.bundle_id,
        "policy_candidate_content_id": candidate.content_id,
        "policy": candidate.policy,
        "mode": "runtime-verification",
        "status": "ineligible_pc_runtime",
        "repo_a_commit": terminal_payload["repo"]["commit"],
        "wandb_run_id": terminal_payload["wandb"]["run_id"],
        "runtime_binding": _mutable_json(candidate.runtime_binding),
        "failure_code": terminal_payload["failure"]["code"],
        "verification_sha256": verification_sha256,
        "verification_sync_sha256": verification_sync_sha256,
        "shadow_schema_sha256": SHADOW_SCHEMA_SHA256,
    }


def prepare_observation(spec: PolicySpec, observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate canonical inputs and omit VQ-BeT's evidence-only up image."""

    state = finite_action(observation.get("observation.state"), label="observation.state")
    if observation.get("task") != CANONICAL_TASK:
        raise ValidationError("observation task differs from the benchmark task")
    prepared: dict[str, Any] = {
        "observation.state": observation["observation.state"],
        "task": CANONICAL_TASK,
    }
    for camera in spec.inference_cameras:
        key = f"observation.images.{camera}"
        if key not in observation:
            raise ValidationError(f"observation is missing {key}")
        prepared[key] = observation[key]
    del state
    return prepared


def finite_action(value: Any, *, label: str = "policy action") -> tuple[float, ...]:
    """Convert a tensor/array/sequence to exactly seven finite Python floats."""

    try:
        value = value.detach().cpu()
    except AttributeError:
        pass
    try:
        value = value.numpy()
    except AttributeError:
        pass
    try:
        flattened = value.reshape(-1).tolist()
    except AttributeError:
        flattened = list(value) if value is not None else []
    if len(flattened) != 7:
        raise ValidationError(f"{label} must contain exactly seven values, found {len(flattened)}")
    result: list[float] = []
    for index, item in enumerate(flattened):
        if isinstance(item, bool):
            raise ValidationError(f"{label}[{index}] must be numeric, not boolean")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}[{index}] is not numeric") from exc
        if not math.isfinite(number):
            raise ValidationError(f"{label}[{index}] is not finite")
        result.append(number)
    return tuple(result)


def _exception_verification(
    code: str,
    stage: str,
    exc: Exception,
) -> RuntimeVerification:
    reason = f"{type(exc).__name__}: {exc}"
    return RuntimeVerification(
        status="ineligible_pc_runtime",
        sample_action=None,
        latency=None,
        failure_reason=reason,
        failure={
            "code": code,
            "recoverable": False,
            "evidence": {
                "stage": stage,
                "exception_type": type(exc).__name__,
                "detail_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
            },
        },
    )


def _accepted_dependencies(
    bundle: VerifiedBundle,
    raw_dependencies: Any,
    *,
    spec: PolicySpec,
    artifact_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    dependencies = _object(raw_dependencies, "dependency_artifacts")
    if set(dependencies) != set(spec.dependency_ids):
        raise ValidationError(
            f"{spec.token} dependency IDs differ; expected={list(spec.dependency_ids)}, "
            f"actual={sorted(dependencies)}"
        )
    result: dict[str, Path] = {}
    for dependency_id in spec.dependency_ids:
        item = _object(dependencies[dependency_id], f"dependency {dependency_id}")
        _exact_keys(
            item,
            {"artifact", "inventory_sha256", "relative_path"},
            f"dependency {dependency_id}",
        )
        artifact_name = _text(item["artifact"], f"dependency {dependency_id} artifact")
        declared = artifact_by_name.get(artifact_name)
        if declared is None:
            raise ValidationError(f"dependency {dependency_id} names an absent artifact")
        if item["inventory_sha256"] != declared["inventory_sha256"]:
            raise ValidationError(f"dependency {dependency_id} inventory differs")
        relative_text = _text(item["relative_path"], f"dependency {dependency_id} path")
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError(f"dependency {dependency_id} path escapes its artifact")
        root = _accepted_artifact(bundle, artifact_name)
        resolved = root.joinpath(*relative.parts).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValidationError(f"dependency {dependency_id} path escapes its artifact") from exc
        if resolved.is_symlink() or not resolved.is_dir():
            raise ValidationError(f"dependency {dependency_id} is not an accepted directory")
        result[dependency_id] = resolved
    return result


def _accepted_artifact(bundle: VerifiedBundle, name: str) -> Path:
    try:
        return bundle.accepted_artifact_root(name).resolve()
    except HandoffError as exc:
        raise ValidationError(
            f"candidate artifact {name!r} is not receiver-local; run viola-handoff accept first"
        ) from exc


def _require_pc_a_acceptance(bundle: VerifiedBundle) -> None:
    expected = {item["name"] for item in bundle.manifest["artifacts"]}
    receipts = [
        receipt
        for receipt in bundle.receipts
        if receipt["status"] == "accepted" and receipt["actor"]["role"] == "pc_a"
    ]
    if not receipts:
        raise ValidationError("policy candidate lacks an accepted receipt from pc_a")
    if any(set(receipt["accepted_artifacts"]) != expected for receipt in receipts):
        raise ValidationError("PC-A acceptance receipt does not bind every artifact")


def _require_exact_payload_files(bundle: VerifiedBundle, expected: set[str]) -> None:
    actual = {item["path"] for item in bundle.manifest["payload"]["files"]}
    if actual != expected:
        raise ValidationError(
            f"policy candidate payload differs; expected={sorted(expected)}, actual={sorted(actual)}"
        )


def _bind_dependency_paths(config: Any, candidate: AcceptedPolicyCandidate) -> None:
    paths = candidate.dependencies
    if candidate.policy == "smolvla":
        config.vlm_model_name = str(paths["smolvlm_model"])
    elif candidate.policy == "pi0_fast":
        config.text_tokenizer_name = str(paths["paligemma_tokenizer"])
        config.action_tokenizer_name = str(paths["fast_action_tokenizer"])
    elif candidate.policy == "groot":
        config.base_model_path = str(paths["groot_base_model"])


def _processor_overrides(candidate: AcceptedPolicyCandidate) -> dict[str, dict[str, str]]:
    paths = candidate.dependencies
    if candidate.policy == "smolvla":
        return {"tokenizer_processor": {"tokenizer_name": str(paths["smolvlm_model"])}}
    if candidate.policy in {"pi0", "pi05"}:
        return {
            "tokenizer_processor": {
                "tokenizer_name": str(paths["paligemma_tokenizer"]),
            }
        }
    if candidate.policy == "pi0_fast":
        paligemma = str(paths["paligemma_tokenizer"])
        return {
            "tokenizer_processor": {"tokenizer_name": paligemma},
            "action_tokenizer_processor": {
                "action_tokenizer_name": str(paths["fast_action_tokenizer"]),
                "paligemma_tokenizer_name": paligemma,
            },
        }
    if candidate.policy == "groot":
        return {
            "groot_n1_7_vlm_encode_v1": {
                "model_name": str(paths["cosmos_reason2_processor"]),
            }
        }
    return {}


@contextmanager
def _huggingface_offline(candidate: AcceptedPolicyCandidate):
    updates = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if candidate.policy == "groot":
        updates["HF_HOME"] = str(candidate.dependencies["hf_hub_cache"])
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _front_camera_metadata(metadata: Any) -> Any:
    """Present VQ-BeT's reviewed one-camera feature set to the public factory."""

    features = {
        key: value
        for key, value in metadata.features.items()
        if key != "observation.images.up"
    }
    values: dict[str, Any] = {"features": features}
    if hasattr(metadata, "stats"):
        values["stats"] = {
            key: value
            for key, value in metadata.stats.items()
            if key != "observation.images.up"
        }
    return SimpleNamespace(**values)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_evaluation_sha256(checkpoint_artifact: Mapping[str, Any]) -> str:
    """Recreate Repo B's portable checkpoint snapshot digest.

    Repo B evaluates a map from each relative file path to its byte count and
    SHA-256.  The accepted handoff inventory contains exactly those signed
    values under different field names, so no producer-local path is needed.
    """

    files = {
        entry["path"]: {
            "bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
        }
        for entry in checkpoint_artifact["files"]
    }
    return hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _decode_json_object(raw, label)
    if raw != canonical_json_bytes(value):
        raise ValidationError(f"{label} is not canonical JSON")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    return _decode_json_object(path.read_bytes(), label)


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValidationError(f"{label} contains non-finite JSON token {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid JSON: {exc}") from exc
    return _object(value, label)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValidationError(
            f"{label} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a nonempty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _wandb_url(value: Any, label: str) -> str:
    text = _text(value, label)
    parsed = urlsplit(text)
    parts = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc != "wandb.ai"
        or len(parts) != 5
        or parts[0] != ""
        or not parts[1]
        or not parts[2]
        or parts[3] != "runs"
        or not parts[4]
    ):
        raise ValidationError(f"{label} must identify one W&B run")
    return parts[4]


def _mutable_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(dict(value)))


__all__ = [
    "CHECKPOINT_ARTIFACT",
    "EXPECTED_HELDOUT_EPISODES",
    "EXPECTED_HELDOUT_FRAMES",
    "LATENCY_THRESHOLD_MS",
    "POLICY_CANDIDATE_FILENAME",
    "REPLAY_DATASET_ARTIFACT",
    "REPLAY_MANIFEST_FILENAME",
    "SHADOW_SCHEMA_SHA256",
    "TIMED_COUNT",
    "WARMUP_COUNT",
    "AcceptedPolicyCandidate",
    "LeRobotPublicApi",
    "LeRobotPolicyRuntime",
    "PolicyRuntime",
    "RuntimeVerification",
    "build_runtime_terminal_lineage",
    "build_runtime_terminal_payload",
    "build_sync_receipt",
    "build_verification_payload",
    "finite_action",
    "inspect_candidate",
    "load_lerobot_runtime",
    "load_replay_dataset",
    "load_verification_observation",
    "prepare_observation",
    "verify_policy",
    "verification_sync_binding",
]
