from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

import viola_ops.report as report_module
from viola_handoff import RuntimeIdentity, SealRequest, accept_bundle, canonical_json_bytes, seal_bundle
from viola_ops.report import (
    EXPECTED_BENCHMARK_CONFIG_SHA256,
    EXPECTED_DATASET_RELEASE_ID,
    EXPECTED_DATASET_SHA256,
    EXPECTED_EXPERIMENT,
    EXPECTED_WANDB_PROJECT,
    POLICIES,
    REPORT_EXECUTION_KIND,
    REPORT_EXECUTION_SCHEMA_VERSION,
    REPORT_RUN_PREFIX,
    REPORT_SYNC_KIND,
    REPORT_SYNC_SCHEMA_VERSION,
    REVIEWED_REPO_B_COMMITS,
    ReportInspectionError,
    inspect_report,
    wilson_interval,
)


class FakeHandoffEvidence:
    def record(self, **event: Any) -> str:
        return f"https://wandb.ai/tester/{event['project']}/runs/{event['run_id']}"


def _identity(role: str, commit: str) -> RuntimeIdentity:
    repository_commit = commit if len(commit) == 40 else commit * 40
    return RuntimeIdentity(
        role=role,
        repository_commit=repository_commit,
        repository_clean=True,
        hostname=role,
        python_version="3.12.13",
        lerobot_version="0.6.1",
        conda_environment="lerobot",
    )


def _row(policy: str, track: str, successes: int, *, safety_events: int = 0) -> dict[str, Any]:
    low, high = wilson_interval(successes, 10)
    return {
        "policy": policy,
        "track": track,
        "rank": 1,
        "successes": successes,
        "trials": 10,
        "success_rate": successes / 10,
        "wilson_low": low,
        "wilson_high": high,
        "safety_events": safety_events,
        "collisions": safety_events,
        "interventions": 0,
        "safety_aborts": 0,
        "malformed_actions": 0,
        "deadline_misses": 0,
        "successful_completion_mean_sec": 24.5 if successes else None,
        "successful_completion_median_sec": 24.0 if successes else None,
        "inference_p95_ms": 21.5,
        "gpu_hours": 2.25,
        "peak_vram_gb": 12.0,
        "training_throughput_samples_per_sec": 43.5,
        "training_lineage_qualification": "current_release_bound",
        "normalization_scope": "training_only",
        "training_deviations": [],
        "checkpoint_bytes": 1024,
        "first_action_mae": 0.1,
        "first_action_rmse": 0.2,
        "failure_counts": {} if successes == 10 else {"timeout": 10 - successes},
    }


def _valid_report() -> dict[str, Any]:
    statuses = {
        "act": "scored",
        "vqbet": "scored",
    }
    outcomes = []
    for policy in POLICIES:
        status = statuses.get(policy, "blocked_setup")
        outcomes.append(
            {
                "policy": policy,
                "status": status,
                "stage": report_module.STATUS_STAGE[status],
                "reason": "validated scored evidence" if status == "scored" else "setup blocked",
                "outcome_sha256": hashlib.sha256(f"outcome:{policy}".encode()).hexdigest(),
                "evidence_sha256": [
                    hashlib.sha256(f"evidence:{policy}".encode()).hexdigest()
                ],
                "wandb_urls": [
                    f"https://wandb.ai/entity/{EXPECTED_WANDB_PROJECT}/runs/outcome-{policy}"
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at_utc": "2026-08-12T00:00:00+00:00",
        "complete": True,
        "methodology": copy.deepcopy(report_module.METHODOLOGY),
        "leaderboards": {
            "two_camera": [_row("act", "two_camera", 8, safety_events=1)],
            "vqbet_auxiliary": [_row("vqbet", "vqbet_auxiliary", 7)],
        },
        "terminal_outcomes": outcomes,
        "blockers": [],
    }


def _build_bundle(
    tmp_path: Path,
    *,
    benchmark_config_sha256: str = EXPECTED_BENCHMARK_CONFIG_SHA256,
    report_mutation=None,
    rendered_mutation: str | None = None,
    sync_mutation=None,
    extra_payload: bool = False,
    accept: bool = True,
    producer_commit: str = next(iter(REVIEWED_REPO_B_COMMITS)),
) -> Path:
    payload = tmp_path / "payload"
    payload.mkdir()
    report = _valid_report()
    validated = report_module._parse_report(report)
    if report_mutation is not None:
        report_mutation(report)

    (payload / "benchmark_report.json").write_bytes(canonical_json_bytes(report))
    csv_bytes = report_module._render_csv(validated)
    markdown_bytes = report_module._render_markdown(validated)
    if rendered_mutation == "benchmark_leaderboards.csv":
        csv_bytes += b"tampered"
    if rendered_mutation == "benchmark_report.md":
        markdown_bytes += b"tampered"
    (payload / "benchmark_leaderboards.csv").write_bytes(csv_bytes)
    (payload / "benchmark_report.md").write_bytes(markdown_bytes)
    if extra_payload:
        (payload / "unexpected.txt").write_text("not part of the report", encoding="utf-8")

    report_hashes = {
        "json": report_module.sha256_file(payload / "benchmark_report.json"),
        "csv": report_module.sha256_file(payload / "benchmark_leaderboards.csv"),
        "markdown": report_module.sha256_file(payload / "benchmark_report.md"),
    }
    terminal_hashes = {
        item["policy"]: item["outcome_sha256"] for item in report["terminal_outcomes"]
    }
    core = {
        "schema_version": REPORT_EXECUTION_SCHEMA_VERSION,
        "kind": REPORT_EXECUTION_KIND,
        "contract_sha256": report_module.AGGREGATE_REPORT_SCHEMA_SHA256,
        "experiment": EXPECTED_EXPERIMENT,
        "benchmark_config_sha256": benchmark_config_sha256,
        "dataset_release_id": EXPECTED_DATASET_RELEASE_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "terminal_outcome_sha256": terminal_hashes,
        "report_sha256": report_hashes,
    }
    report_content_id = report_module.sha256_json(core)
    run_id = f"{REPORT_RUN_PREFIX}{report_content_id[:24]}"
    wandb = {
        "entity": "entity",
        "project": EXPECTED_WANDB_PROJECT,
        "run_id": run_id,
        "url": f"https://wandb.ai/entity/{EXPECTED_WANDB_PROJECT}/runs/{run_id}",
    }
    sync = {
        "schema_version": REPORT_SYNC_SCHEMA_VERSION,
        "kind": REPORT_SYNC_KIND,
        "report_content_id": report_content_id,
        "report_core": core,
        "report_sha256": report_hashes,
        "wandb": wandb,
        "completed_at_utc": "2026-08-12T00:00:01+00:00",
    }
    if sync_mutation is not None:
        sync_mutation(sync)
    (payload / "wandb_sync.complete.json").write_bytes(canonical_json_bytes(sync))

    lineage_wandb = dict(sync["wandb"])
    lineage = {
        "benchmark_config_sha256": benchmark_config_sha256,
        "dataset_release_id": EXPECTED_DATASET_RELEASE_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "aggregate_report_contract_sha256": report_module.AGGREGATE_REPORT_SCHEMA_SHA256,
        "report_content_id": sync["report_content_id"],
        "report_sha256": dict(sync["report_sha256"]),
        "terminal_outcome_sha256": terminal_hashes,
        "terminal_statuses": {
            item["policy"]: item["status"] for item in report["terminal_outcomes"]
        },
        "wandb": lineage_wandb,
    }
    source = seal_bundle(
        SealRequest(
            root=tmp_path / "nas",
            kind="report",
            experiment=EXPECTED_EXPERIMENT,
            subject="all-policies",
            producer=_identity("pc_b", producer_commit),
            consumer_role="pc_a",
            permission="report_only",
            lineage=lineage,
            wandb_project=EXPECTED_WANDB_PROJECT,
            payload_dir=payload,
            created_at="2026-08-12T00:00:02Z",
        ),
        evidence_logger=FakeHandoffEvidence(),
    )
    if not accept:
        return source.path
    accepted = accept_bundle(
        source.path,
        tmp_path / "accepted",
        receiver=_identity("pc_a", "a"),
        evidence_logger=FakeHandoffEvidence(),
    )
    return accepted.path


def test_inspection_returns_human_readable_verified_summary(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    calls: list[tuple[object, object]] = []

    def verify(evidence, expected_config) -> None:
        calls.append((evidence, expected_config))

    summary = inspect_report(bundle, wandb_verifier=verify)

    assert summary.bundle_id == bundle.name
    assert summary.scored_policies == ("act", "vqbet")
    assert tuple(item.policy for item in summary.outcomes) == POLICIES
    assert summary.two_camera[0].successes == 8
    assert summary.vqbet_auxiliary[0].successes == 7
    assert len(calls) == 1
    assert calls[0][1] == {
        "report_content_id": summary.report_content_id,
        "report_core": calls[0][1]["report_core"],
    }
    text = summary.render_text()
    assert "act: scored [rollout]" in text
    assert "#1 act: 8/10 (80.0%)" in text
    assert "accepted, complete, and blocker-free" in text
    assert "checkout-path-dependent; not a portable readiness proof" in text
    assert "does not authorize robot motion" in text
    assert "robot ready" not in text.lower()


def test_report_accepts_repo_b_path_bound_config_hash_when_fully_bound(
    tmp_path: Path,
) -> None:
    # Repo B currently hashes resolved local config paths, so the same committed
    # config has a different digest when its checkout moves.  The accepted
    # bundle, report core, content-derived W&B run, and lineage must all agree.
    repo_b_main_hash = "11ffd25e262bc86ab8082857bec331ea9a84349b840703ced2be7cd0b32b42ba"
    bundle = _build_bundle(tmp_path, benchmark_config_sha256=repo_b_main_hash)

    summary = inspect_report(bundle, wandb_verifier=lambda *_args: None)

    assert summary.bundle_id == bundle.name
    assert summary.scored_policies == ("act", "vqbet")


def test_report_rejects_an_unreviewed_path_bound_config_hash(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, benchmark_config_sha256="a" * 64)

    with pytest.raises(ReportInspectionError, match="reviewed path-bound set"):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


def test_report_rejects_an_unreviewed_repo_b_producer_revision(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, producer_commit="b" * 40)

    with pytest.raises(ReportInspectionError, match="reviewed Repo-B commit set"):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


def test_default_path_requires_remote_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build_bundle(tmp_path)
    calls = []

    def strict_default(evidence, expected_config) -> None:
        calls.append((evidence, expected_config))

    monkeypatch.setattr(report_module, "verify_remote_wandb", strict_default)
    inspect_report(bundle)
    assert len(calls) == 1


def test_unaccepted_report_is_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, accept=False)
    with pytest.raises(ReportInspectionError, match="no receiver acceptance receipt"):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


def test_payload_must_contain_exactly_four_report_files(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, extra_payload=True)
    with pytest.raises(ReportInspectionError, match="exactly these files"):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["leaderboards"]["two_camera"][0].__setitem__(
                "success_rate", 0.7
            ),
            "success_rate differs",
        ),
        (
            lambda report: report["leaderboards"]["two_camera"][0].__setitem__(
                "wilson_low", 0.0
            ),
            "Wilson interval",
        ),
        (
            lambda report: report["leaderboards"]["two_camera"][0].__setitem__(
                "safety_events", 2
            ),
            "component counts",
        ),
        (
            lambda report: report["terminal_outcomes"].reverse(),
            "approved policy order",
        ),
        (lambda report: report.__setitem__("complete", False), "not complete"),
        (
            lambda report: report.__setitem__(
                "blockers",
                [
                    {
                        "code": "pending",
                        "stage": "report",
                        "message": "not complete",
                        "recoverable": True,
                        "details": {},
                        "recorded_at_utc": "2026-08-12T00:00:00+00:00",
                    }
                ],
            ),
            "contains blockers",
        ),
    ],
)
def test_report_semantics_are_recomputed(tmp_path: Path, mutation, message: str) -> None:
    bundle = _build_bundle(tmp_path, report_mutation=mutation)
    with pytest.raises(ReportInspectionError, match=message):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


@pytest.mark.parametrize(
    "name", ["benchmark_leaderboards.csv", "benchmark_report.md"]
)
def test_rendered_artifacts_must_match_json(tmp_path: Path, name: str) -> None:
    bundle = _build_bundle(tmp_path, rendered_mutation=name)
    with pytest.raises(ReportInspectionError, match="inconsistent"):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


def test_wandb_sync_run_id_must_be_content_derived(tmp_path: Path) -> None:
    def mutate(sync: dict[str, Any]) -> None:
        sync["wandb"]["run_id"] = "report-arbitrary"
        sync["wandb"]["url"] = (
            f"https://wandb.ai/entity/{EXPECTED_WANDB_PROJECT}/runs/report-arbitrary"
        )

    bundle = _build_bundle(tmp_path, sync_mutation=mutate)
    with pytest.raises(ReportInspectionError, match="not content-derived"):
        inspect_report(bundle, wandb_verifier=lambda *_args: None)


def test_remote_verifier_failure_is_fail_closed(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)

    def fail(*_args) -> None:
        raise RuntimeError("offline")

    with pytest.raises(ReportInspectionError, match="W&B verification failed: offline"):
        inspect_report(bundle, wandb_verifier=fail)
