from __future__ import annotations

import hashlib
from pathlib import Path

from viola_handoff import CONTRACT_SHA256


REPO_B_CONTRACT_SHA256 = "fbfef2f214ff320f03891f9694056a4377c1405e0362bb5e9cf226ef1b82e99e"
REPO_B_SCHEMA_SHA256 = {
    "dataset_release_v2.schema.json": (
        "64e1c1fdea32117c6b1a26b93f8cc2825c05a202de36e958b46c3198944df6db"
    ),
    "rollout_evidence.schema.json": (
        "f39530be3a279c893941d0fb2dcf7721bc16fd1e6c9880db7375ca9ab577f9ba"
    ),
    "rollout_session.schema.json": (
        "31c24505b03fcb75acdd649e0ed3492795cf8246267915879ce3e0708b6f80e5"
    ),
    "shadow_evidence.schema.json": (
        "778a3cbb73355764626a2f1b2bf915d347951d118287cc846e7a30e9a09cfd90"
    ),
}


def test_shared_contract_matches_repo_b() -> None:
    assert CONTRACT_SHA256 == REPO_B_CONTRACT_SHA256


def test_shared_schemas_match_repo_b() -> None:
    contracts = Path(__file__).resolve().parents[1] / "contracts"

    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in contracts.glob("*.schema.json")
    }

    assert {name: actual[name] for name in REPO_B_SCHEMA_SHA256} == REPO_B_SCHEMA_SHA256
