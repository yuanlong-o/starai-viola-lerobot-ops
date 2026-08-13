"""The small, explicit registry for the eight Viola benchmark policies.

The registry intentionally contains data, not callbacks.  Loading and running a
policy belongs in :mod:`viola_ops.policy_runtime`; keeping the table here makes
the cross-PC contract easy for an operator to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

from .errors import ValidationError

PolicyToken = Literal[
    "act",
    "diffusion",
    "vqbet",
    "smolvla",
    "pi0",
    "pi0_fast",
    "pi05",
    "groot",
]

POLICY_TOKENS: Final[tuple[PolicyToken, ...]] = (
    "act",
    "diffusion",
    "vqbet",
    "smolvla",
    "pi0",
    "pi0_fast",
    "pi05",
    "groot",
)

CANONICAL_TASK: Final = (
    "Move the blue cube, then the red cube, from the white pad on the right "
    "to the gray platform on the left."
)
JOINT_NAMES: Final = (
    "Motor_0",
    "Motor_1",
    "Motor_2",
    "Motor_3",
    "Motor_4",
    "Motor_5",
    "gripper",
)
QUEUE_ACTIONS: Final = 10


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Human-readable deployment facts that Repo A and Repo B must share."""

    token: PolicyToken
    queue_config_field: str
    feature_map: Mapping[str, str]
    dependency_ids: tuple[str, ...]
    inference_cameras: tuple[str, ...]
    evidence_cameras: tuple[str, ...] = ("front", "up")


def _spec(
    token: PolicyToken,
    queue_config_field: str = "n_action_steps",
    *,
    feature_map: Mapping[str, str] | None = None,
    dependency_ids: tuple[str, ...] = (),
    inference_cameras: tuple[str, ...] = ("front", "up"),
) -> PolicySpec:
    return PolicySpec(
        token=token,
        queue_config_field=queue_config_field,
        feature_map=MappingProxyType(dict(feature_map or {})),
        dependency_ids=dependency_ids,
        inference_cameras=inference_cameras,
    )


_VLA_FEATURE_MAP = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.up": "observation.images.left_wrist_0_rgb",
}

POLICIES: Final[Mapping[PolicyToken, PolicySpec]] = MappingProxyType(
    {
        "act": _spec("act"),
        "diffusion": _spec("diffusion"),
        "vqbet": _spec(
            "vqbet",
            "action_chunk_size",
            inference_cameras=("front",),
        ),
        "smolvla": _spec(
            "smolvla",
            feature_map={
                "observation.images.front": "observation.images.camera1",
                "observation.images.up": "observation.images.camera2",
            },
            dependency_ids=("smolvlm_model",),
        ),
        "pi0": _spec(
            "pi0",
            feature_map=_VLA_FEATURE_MAP,
            dependency_ids=("paligemma_tokenizer",),
        ),
        "pi0_fast": _spec(
            "pi0_fast",
            feature_map=_VLA_FEATURE_MAP,
            dependency_ids=("fast_action_tokenizer", "paligemma_tokenizer"),
        ),
        "pi05": _spec(
            "pi05",
            feature_map=_VLA_FEATURE_MAP,
            dependency_ids=("paligemma_tokenizer",),
        ),
        "groot": _spec(
            "groot",
            dependency_ids=("cosmos_reason2_processor", "groot_base_model", "hf_hub_cache"),
        ),
    }
)


def get_policy_spec(token: str) -> PolicySpec:
    """Return a policy specification with an actionable error for bad input."""

    try:
        return POLICIES[token]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(POLICY_TOKENS)
        raise ValidationError(f"unsupported policy {token!r}; expected one of: {choices}") from exc


__all__ = [
    "CANONICAL_TASK",
    "JOINT_NAMES",
    "POLICIES",
    "POLICY_TOKENS",
    "QUEUE_ACTIONS",
    "PolicySpec",
    "PolicyToken",
    "get_policy_spec",
]
