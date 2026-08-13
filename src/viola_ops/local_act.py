"""Repo-A-local runtime for the known, previously deployed ACT checkpoint.

This module is deliberately narrower than the cross-PC policy-candidate path.
It recognizes one exact local ACT checkpoint and the frozen Viola dataset that
trained it.  It does not create motion authority, open hardware, or contact a
model hub.  A caller must still obtain a reviewed setup, current E-stop
evidence, and an operator-issued motion permit before constructing a robot.

The loader uses only public LeRobot configuration, dataset-metadata, policy,
processor, and observation-preparation APIs.  The source bytes are copied into
a private, read-only runtime directory before a live caller loads them.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .dataset import CURRENT_DATASET, DATASET_REPO_ID, DEFAULT_DATASET_ROOT, RELEASE_ID
from .errors import ValidationError
from .jsonutil import read_json_object, sha256_file, sha256_json
from .policies import CANONICAL_TASK, QUEUE_ACTIONS, PolicySpec, get_policy_spec
from .policy_runtime import finite_action
from .publication_guard import TreeSnapshot, snapshot_tree

DEFAULT_ACT_CHECKPOINT: Final = Path("~/models/act_viola_val20_step080000").expanduser()
LEGACY_ACT_MODEL_SHA256: Final = (
    "1093aaeddfb902e7e596425d87676baba58cb8ab617a52c954ec11940726b886"
)

_LEGACY_CHECKPOINT_INVENTORY: Final = {
    "directories": [],
    "files": [
        {
            "path": "config.json",
            "sha256": "6cff214bbffe4786a3a6479017d6d85f2b87099409e7f855e74aa6919a33277b",
            "size_bytes": 1_699,
        },
        {
            "path": "model.safetensors",
            "sha256": LEGACY_ACT_MODEL_SHA256,
            "size_bytes": 206_707_932,
        },
        {
            "path": "policy_postprocessor.json",
            "sha256": "c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5",
            "size_bytes": 660,
        },
        {
            "path": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "sha256": "429ce85a115a9019dc9a4c3f4a7d49ed5cbc0875881080ca55cb29a1fcd14caf",
            "size_bytes": 7_592,
        },
        {
            "path": "policy_preprocessor.json",
            "sha256": "32c4df30be63d115123a5e982429f66218738c7c08b5ef9ad129c66090872ea6",
            "size_bytes": 1_325,
        },
        {
            "path": "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "sha256": "8de4c964e054997cd3c06e819e0e01f510d449bceb9ab07ed0d9b65dce61470b",
            "size_bytes": 7_600,
        },
        {
            "path": "train_config.json",
            "sha256": "ea6e806a82873fcb2532d2810e1e1233028fdf4d04ee5020c751c990e22bc021",
            "size_bytes": 6_328,
        },
    ],
    "file_count": 7,
    "byte_count": 206_733_136,
    "inventory_sha256": "c3764c47677c8b33adbcf5f288429b28180bdb64542cd46e7a92aba7dbdca214",
}

_LEGACY_DATASET_METADATA_INVENTORY: Final = {
    "directories": ["episodes", "episodes/chunk-000"],
    "files": [
        {
            "path": "episodes/chunk-000/file-000.parquet",
            "sha256": "871572eccc2fdee6632d4892a7afba0f046085b0481b40d09e999b0493892cca",
            "size_bytes": 185_509,
        },
        {
            "path": "filter_manifest.json",
            "sha256": "a687c948e4edb3cc5bf6d9a461a63ad587fafc33b4eba7620ba6b5b9c44466ab",
            "size_bytes": 5_683,
        },
        {
            "path": "info.json",
            "sha256": "674ce072d5ef30dcacb908fa18acb88e294853884f6b0f66e0b55e0863903e52",
            "size_bytes": 3_247,
        },
        {
            "path": "stats.json",
            "sha256": "d407701cc9ad1116652cfed6b8a4314509c5f1f3dadbe5d6cedddc00f0161c6b",
            "size_bytes": 13_494,
        },
        {
            "path": "tasks.parquet",
            "sha256": "a083774b7f5afe34e4f7c0b567771b9e87be7dbe287e49e6c651a0afb1f0d482",
            "size_bytes": 2_605,
        },
        {
            "path": "validation_report.json",
            "sha256": "fc93aa7ee34d0a519344b2b20b48930844f7c1c3a042853d1ccc38cc8393fd62",
            "size_bytes": 1_517,
        },
    ],
    "file_count": 6,
    "byte_count": 212_055,
    "inventory_sha256": "6249baf912973f6fed78d9822e81206fbfdd2ed740e71dd89c4f30b3a9b1bad7",
}


@dataclass(frozen=True, slots=True)
class LocalActExpectation:
    """Exact bytes and identities allowed by the local ACT compatibility path."""

    checkpoint_inventory: Mapping[str, Any]
    dataset_inventory: Mapping[str, Any]
    dataset_metadata_inventory: Mapping[str, Any]
    model_sha256: str
    dataset_repo_id: str
    dataset_release_id: str


LEGACY_ACT_EXPECTATION: Final = LocalActExpectation(
    checkpoint_inventory=MappingProxyType(_LEGACY_CHECKPOINT_INVENTORY),
    dataset_inventory=MappingProxyType(CURRENT_DATASET.expected_inventory()),
    dataset_metadata_inventory=MappingProxyType(_LEGACY_DATASET_METADATA_INVENTORY),
    model_sha256=LEGACY_ACT_MODEL_SHA256,
    dataset_repo_id=DATASET_REPO_ID,
    dataset_release_id=RELEASE_ID,
)


@dataclass(frozen=True, slots=True)
class LocalActCandidate:
    """One exact local ACT checkpoint plus its frozen training metadata."""

    checkpoint: Path
    dataset_root: Path
    checkpoint_inventory: Mapping[str, Any]
    dataset_inventory: Mapping[str, Any]
    dataset_metadata_inventory: Mapping[str, Any]
    model_sha256: str
    dataset_repo_id: str
    dataset_release_id: str
    candidate_id: str
    spec: PolicySpec

    @property
    def policy(self) -> str:
        return "act"

    @property
    def bundle_id(self) -> str:
        """Structural compatibility with the guarded execution state machine."""

        return self.candidate_id

    @property
    def content_id(self) -> str:
        return self.candidate_id


@dataclass(frozen=True, slots=True)
class LocalActRuntimeSnapshot:
    """Private, read-only bytes retained for the lifetime of one runtime."""

    candidate: LocalActCandidate
    root: Path
    tree: TreeSnapshot

    def verify(self) -> None:
        current = snapshot_tree(self.root, label="private local ACT runtime")
        if (current.device, current.inode) != (self.tree.device, self.tree.inode):
            raise ValidationError("private local ACT runtime directory was replaced")
        if current.inventory != self.tree.inventory:
            raise ValidationError("private local ACT runtime bytes changed")


@dataclass(frozen=True, slots=True)
class LocalActPublicApi:
    """Injectable handles to the public LeRobot APIs used by this module."""

    config_from_pretrained: Callable[..., Any]
    dataset_metadata: Callable[..., Any]
    make_policy: Callable[..., Any]
    make_pre_post_processors: Callable[..., Any]
    prepare_observation_for_inference: Callable[..., Mapping[str, Any]]


class LocalActRuntime:
    """The loaded policy and its real robot-observation conversion pipeline."""

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        prepare_observation: Callable[..., Mapping[str, Any]],
        *,
        device: Any,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._prepare_observation = prepare_observation
        self._device = device

    def reset(self) -> None:
        for component in (self._policy, self._preprocessor, self._postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def infer(self, observation: Mapping[str, Any]) -> tuple[float, ...]:
        """Prepare one HWC uint8 camera observation and return one finite 7-D action."""

        import torch

        raw = prepare_local_act_observation(observation)
        prepared = self._prepare_observation(
            raw,
            self._device,
            task=CANONICAL_TASK,
            robot_type="starai_viola",
        )
        with torch.inference_mode():
            processed = self._preprocessor(dict(prepared))
            action = self._policy.select_action(processed)
            action = self._postprocessor(action)
        return finite_action(action)


def inspect_local_act_candidate(
    checkpoint: str | Path = DEFAULT_ACT_CHECKPOINT,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    expectation: LocalActExpectation = LEGACY_ACT_EXPECTATION,
) -> LocalActCandidate:
    """Recognize the exact historic ACT deployment without opening hardware."""

    checkpoint_tree = snapshot_tree(checkpoint, label="local ACT checkpoint")
    dataset_tree = snapshot_tree(dataset_root, label="local ACT training dataset")
    metadata_tree = snapshot_tree(
        dataset_tree.path / "meta", label="local ACT dataset metadata"
    )
    _require_inventory(
        checkpoint_tree.inventory,
        expectation.checkpoint_inventory,
        label="local ACT checkpoint",
    )
    _require_inventory(
        dataset_tree.inventory,
        expectation.dataset_inventory,
        label="local ACT training dataset",
    )
    _require_inventory(
        metadata_tree.inventory,
        expectation.dataset_metadata_inventory,
        label="local ACT dataset metadata",
    )

    model_sha256 = sha256_file(checkpoint_tree.path / "model.safetensors")
    if model_sha256 != expectation.model_sha256:
        raise ValidationError("local ACT model hash differs from the reviewed deployment")

    config = read_json_object(checkpoint_tree.path / "config.json", label="ACT config")
    training = read_json_object(
        checkpoint_tree.path / "train_config.json", label="ACT training config"
    )
    info = read_json_object(metadata_tree.path / "info.json", label="ACT dataset info")
    validation = read_json_object(
        metadata_tree.path / "validation_report.json", label="ACT dataset validation"
    )
    _validate_checkpoint_semantics(config, training, expectation)
    _validate_dataset_semantics(info, validation, expectation)

    _require_same_tree(checkpoint_tree, boundary="while inspecting the ACT checkpoint")
    _require_same_tree(dataset_tree, boundary="while inspecting the ACT dataset")
    _require_same_tree(metadata_tree, boundary="while inspecting ACT dataset metadata")
    candidate_id = sha256_json(
        {
            "policy": "act",
            "model_sha256": model_sha256,
            "checkpoint_inventory_sha256": checkpoint_tree.inventory["inventory_sha256"],
            "dataset_release_id": expectation.dataset_release_id,
            "dataset_inventory_sha256": dataset_tree.inventory["inventory_sha256"],
            "dataset_metadata_inventory_sha256": metadata_tree.inventory[
                "inventory_sha256"
            ],
            "deployment_queue_actions": QUEUE_ACTIONS,
        }
    )
    return LocalActCandidate(
        checkpoint=checkpoint_tree.path,
        dataset_root=dataset_tree.path,
        checkpoint_inventory=MappingProxyType(dict(checkpoint_tree.inventory)),
        dataset_inventory=MappingProxyType(dict(dataset_tree.inventory)),
        dataset_metadata_inventory=MappingProxyType(dict(metadata_tree.inventory)),
        model_sha256=model_sha256,
        dataset_repo_id=expectation.dataset_repo_id,
        dataset_release_id=expectation.dataset_release_id,
        candidate_id=candidate_id,
        spec=get_policy_spec("act"),
    )


@contextmanager
def snapshot_local_act_runtime(
    candidate: LocalActCandidate,
) -> Iterator[LocalActRuntimeSnapshot]:
    """Copy the exact checkpoint and dataset metadata to private read-only storage."""

    checkpoint_source = snapshot_tree(candidate.checkpoint, label="local ACT checkpoint")
    dataset_source = snapshot_tree(candidate.dataset_root, label="local ACT training dataset")
    metadata_source = snapshot_tree(
        candidate.dataset_root / "meta", label="local ACT dataset metadata"
    )
    _require_candidate_inventories(
        candidate, checkpoint_source, dataset_source, metadata_source
    )

    root = Path(tempfile.mkdtemp(prefix="viola-local-act-"))
    try:
        checkpoint_copy = root / "checkpoint"
        dataset_copy = root / "dataset"
        dataset_copy.mkdir()
        shutil.copytree(
            checkpoint_source.path,
            checkpoint_copy,
            symlinks=True,
            copy_function=shutil.copyfile,
        )
        shutil.copytree(
            metadata_source.path,
            dataset_copy / "meta",
            symlinks=True,
            copy_function=shutil.copyfile,
        )
        copied_checkpoint = snapshot_tree(
            checkpoint_copy, label="private local ACT checkpoint"
        )
        copied_metadata = snapshot_tree(
            dataset_copy / "meta", label="private local ACT dataset metadata"
        )
        _require_inventory(
            copied_checkpoint.inventory,
            candidate.checkpoint_inventory,
            label="private local ACT checkpoint",
        )
        _require_inventory(
            copied_metadata.inventory,
            candidate.dataset_metadata_inventory,
            label="private local ACT dataset metadata",
        )

        _require_same_tree(checkpoint_source, boundary="while copying the ACT checkpoint")
        _require_same_tree(dataset_source, boundary="while copying the ACT dataset binding")
        _require_same_tree(metadata_source, boundary="while copying ACT dataset metadata")
        runtime_candidate = replace(
            candidate,
            checkpoint=checkpoint_copy,
            dataset_root=dataset_copy,
        )
        runtime_tree = snapshot_tree(root, label="private local ACT runtime")
        _make_read_only(root)
        snapshot = LocalActRuntimeSnapshot(runtime_candidate, root, runtime_tree)
        snapshot.verify()
        yield snapshot
        snapshot.verify()
    finally:
        _remove_private_tree(root)


def load_local_act_runtime(
    candidate: LocalActCandidate,
    *,
    api: LocalActPublicApi | None = None,
) -> LocalActRuntime:
    """Load the exact ACT weights with a ten-action deployment queue."""

    public_api = api or _public_lerobot_api()
    checkpoint_before = snapshot_tree(candidate.checkpoint, label="ACT runtime checkpoint")
    metadata_before = snapshot_tree(
        candidate.dataset_root / "meta", label="ACT runtime dataset metadata"
    )
    _require_inventory(
        checkpoint_before.inventory,
        candidate.checkpoint_inventory,
        label="ACT runtime checkpoint",
    )
    _require_inventory(
        metadata_before.inventory,
        candidate.dataset_metadata_inventory,
        label="ACT runtime dataset metadata",
    )

    with _local_model_only():
        config = public_api.config_from_pretrained(
            candidate.checkpoint,
            local_files_only=True,
        )
        _validate_loaded_config(config, before_overlay=True)
        config.pretrained_path = str(candidate.checkpoint)
        config.pretrained_revision = None
        config.n_action_steps = QUEUE_ACTIONS

        metadata = public_api.dataset_metadata(
            candidate.dataset_repo_id,
            root=candidate.dataset_root,
            force_cache_sync=False,
        )
        policy = public_api.make_policy(config, ds_meta=metadata, rename_map={})
        policy.eval()
        preprocessor, postprocessor = public_api.make_pre_post_processors(
            config,
            pretrained_path=str(candidate.checkpoint),
        )
    _validate_loaded_config(config, before_overlay=False)
    policy_config = getattr(policy, "config", None)
    if policy_config is not config:
        raise ValidationError("LeRobot ACT policy did not retain the bound deployment config")
    _require_same_tree(checkpoint_before, boundary="while LeRobot loaded ACT weights")
    _require_same_tree(metadata_before, boundary="while LeRobot loaded ACT metadata")

    try:
        import torch

        device = torch.device(config.device)
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise ValidationError(f"LeRobot ACT config has an invalid device: {exc}") from exc
    return LocalActRuntime(
        policy,
        preprocessor,
        postprocessor,
        public_api.prepare_observation_for_inference,
        device=device,
    )


def prepare_local_act_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the real NumPy observation shape expected by LeRobot's public helper."""

    import numpy as np

    expected = {
        "observation.state",
        "observation.images.front",
        "observation.images.up",
        "task",
    }
    if set(observation) != expected:
        raise ValidationError(
            "local ACT observation fields differ; "
            f"missing={sorted(expected - set(observation))}, "
            f"unknown={sorted(set(observation) - expected)}"
        )
    if observation["task"] != CANONICAL_TASK:
        raise ValidationError("local ACT observation task differs from the reviewed task")
    state = np.asarray(finite_action(observation["observation.state"]), dtype=np.float32)
    result: dict[str, Any] = {"observation.state": state}
    for camera in ("front", "up"):
        key = f"observation.images.{camera}"
        frame = observation[key]
        if not isinstance(frame, np.ndarray):
            raise ValidationError(f"local ACT {camera} image must be a NumPy array")
        if frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
            raise ValidationError(
                f"local ACT {camera} image must be uint8 HWC with shape 480x640x3"
            )
        result[key] = np.ascontiguousarray(frame)
    return result


def _validate_checkpoint_semantics(
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    expectation: LocalActExpectation,
) -> None:
    expected_inputs = {
        "observation.images.front": ("VISUAL", [3, 480, 640]),
        "observation.images.up": ("VISUAL", [3, 480, 640]),
        "observation.state": ("STATE", [7]),
    }
    inputs = config.get("input_features")
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_inputs):
        raise ValidationError("local ACT config has different input features")
    for name, (feature_type, shape) in expected_inputs.items():
        value = inputs[name]
        if not isinstance(value, Mapping) or value.get("type") != feature_type or value.get(
            "shape"
        ) != shape:
            raise ValidationError(f"local ACT config has a different {name} feature")
    output = config.get("output_features")
    if (
        not isinstance(output, Mapping)
        or set(output) != {"action"}
        or not isinstance(output["action"], Mapping)
        or output["action"].get("type") != "ACTION"
        or output["action"].get("shape") != [7]
    ):
        raise ValidationError("local ACT config must produce exactly one seven-axis action")
    if (
        config.get("type") != "act"
        or config.get("chunk_size") != 100
        or config.get("n_action_steps") != 100
        or config.get("device") != "cuda"
        or config.get("use_amp") is not False
    ):
        raise ValidationError("local ACT checkpoint architecture differs from the reviewed model")
    dataset = training.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("repo_id") != expectation.dataset_repo_id:
        raise ValidationError("local ACT training config names a different dataset")
    if training.get("policy") != config:
        raise ValidationError("local ACT training and deployment configs differ")


def _validate_dataset_semantics(
    info: Mapping[str, Any],
    validation: Mapping[str, Any],
    expectation: LocalActExpectation,
) -> None:
    if (
        info.get("total_episodes") != 34
        or info.get("total_frames") != 28_306
        or info.get("fps") != 30
        or info.get("robot_type") != "starai_viola"
    ):
        raise ValidationError("local ACT dataset dimensions differ from the frozen release")
    features = info.get("features")
    expected = {
        "action": ("float32", [7]),
        "observation.state": ("float32", [7]),
        "observation.images.front": ("video", [480, 640, 3]),
        "observation.images.up": ("video", [480, 640, 3]),
    }
    if not isinstance(features, Mapping):
        raise ValidationError("local ACT dataset lacks feature metadata")
    for name, (dtype, shape) in expected.items():
        value = features.get(name)
        if not isinstance(value, Mapping) or value.get("dtype") != dtype or value.get(
            "shape"
        ) != shape:
            raise ValidationError(f"local ACT dataset has a different {name} feature")
    if (
        validation.get("repo_id") != expectation.dataset_repo_id
        or validation.get("episodes") != 34
        or validation.get("frames") != 28_306
    ):
        raise ValidationError("local ACT dataset validation belongs to another release")


def _validate_loaded_config(config: Any, *, before_overlay: bool) -> None:
    expected_actions = 100 if before_overlay else QUEUE_ACTIONS
    if (
        getattr(config, "type", None) != "act"
        or getattr(config, "chunk_size", None) != 100
        or getattr(config, "n_action_steps", None) != expected_actions
    ):
        stage = "checkpoint" if before_overlay else "deployment overlay"
        raise ValidationError(f"LeRobot ACT config differs from the reviewed {stage}")
    if not before_overlay and not getattr(config, "pretrained_path", None):
        raise ValidationError("LeRobot ACT config is not bound to pretrained weights")


def _public_lerobot_api() -> LocalActPublicApi:
    """Import public LeRobot entry points only when a real load is requested."""

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import prepare_observation_for_inference
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    # Importing the public ACT config registers the public ``act`` policy token.
    del ACTConfig
    return LocalActPublicApi(
        config_from_pretrained=PreTrainedConfig.from_pretrained,
        dataset_metadata=LeRobotDatasetMetadata,
        make_policy=make_policy,
        make_pre_post_processors=make_pre_post_processors,
        prepare_observation_for_inference=prepare_observation_for_inference,
    )


@contextmanager
def _local_model_only() -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_candidate_inventories(
    candidate: LocalActCandidate,
    checkpoint: TreeSnapshot,
    dataset: TreeSnapshot,
    metadata: TreeSnapshot,
) -> None:
    _require_inventory(
        checkpoint.inventory, candidate.checkpoint_inventory, label="local ACT checkpoint"
    )
    _require_inventory(
        dataset.inventory, candidate.dataset_inventory, label="local ACT training dataset"
    )
    _require_inventory(
        metadata.inventory,
        candidate.dataset_metadata_inventory,
        label="local ACT dataset metadata",
    )
    if sha256_file(checkpoint.path / "model.safetensors") != candidate.model_sha256:
        raise ValidationError("local ACT model differs from its inspected identity")


def _require_inventory(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    if dict(actual) != dict(expected):
        raise ValidationError(f"{label} inventory differs from the reviewed bytes")


def _require_same_tree(expected: TreeSnapshot, *, boundary: str) -> None:
    current = snapshot_tree(expected.path, label=expected.label)
    if (current.device, current.inode) != (expected.device, expected.inode):
        raise ValidationError(f"{expected.label} was replaced {boundary}")
    if current.inventory != expected.inventory:
        raise ValidationError(f"{expected.label} bytes changed {boundary}")


def _make_read_only(root: Path) -> None:
    try:
        for current, directories, files in os.walk(root, topdown=False, followlinks=False):
            directory = Path(current)
            for name in files:
                (directory / name).chmod(0o400, follow_symlinks=False)
            for name in directories:
                (directory / name).chmod(0o500, follow_symlinks=False)
            directory.chmod(0o500, follow_symlinks=False)
    except OSError as exc:
        raise ValidationError(f"cannot protect private local ACT runtime: {exc}") from exc


def _remove_private_tree(root: Path) -> None:
    try:
        for current, directories, _files in os.walk(root, topdown=False, followlinks=False):
            directory = Path(current)
            for name in directories:
                (directory / name).chmod(0o700, follow_symlinks=False)
            directory.chmod(0o700, follow_symlinks=False)
        shutil.rmtree(root)
    except OSError:
        pass


__all__ = [
    "DEFAULT_ACT_CHECKPOINT",
    "LEGACY_ACT_EXPECTATION",
    "LEGACY_ACT_MODEL_SHA256",
    "LocalActCandidate",
    "LocalActExpectation",
    "LocalActPublicApi",
    "LocalActRuntime",
    "LocalActRuntimeSnapshot",
    "inspect_local_act_candidate",
    "load_local_act_runtime",
    "prepare_local_act_observation",
    "snapshot_local_act_runtime",
]
