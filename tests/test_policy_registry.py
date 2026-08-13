from __future__ import annotations

import pytest

from viola_ops.errors import ValidationError
from viola_ops.policies import POLICIES, POLICY_TOKENS, QUEUE_ACTIONS, get_policy_spec


def test_registry_has_one_readable_entry_for_every_policy() -> None:
    assert tuple(POLICIES) == POLICY_TOKENS == (
        "act",
        "diffusion",
        "vqbet",
        "smolvla",
        "pi0",
        "pi0_fast",
        "pi05",
        "groot",
    )
    assert QUEUE_ACTIONS == 10


@pytest.mark.parametrize("token", POLICY_TOKENS)
def test_registry_matches_repo_b_candidate_contract(token: str) -> None:
    spec = get_policy_spec(token)
    assert spec.token == token
    assert spec.queue_config_field == ("action_chunk_size" if token == "vqbet" else "n_action_steps")
    assert spec.evidence_cameras == ("front", "up")
    assert spec.inference_cameras == (("front",) if token == "vqbet" else ("front", "up"))

    expected_dependencies = {
        "act": (),
        "diffusion": (),
        "vqbet": (),
        "smolvla": ("smolvlm_model",),
        "pi0": ("paligemma_tokenizer",),
        "pi0_fast": ("fast_action_tokenizer", "paligemma_tokenizer"),
        "pi05": ("paligemma_tokenizer",),
        "groot": ("cosmos_reason2_processor", "groot_base_model", "hf_hub_cache"),
    }
    assert spec.dependency_ids == expected_dependencies[token]


def test_unknown_policy_error_lists_the_supported_choices() -> None:
    with pytest.raises(ValidationError, match="unsupported policy.*act.*groot"):
        get_policy_spec("unknown")
