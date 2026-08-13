"""Read-only inspection of Repo-B aggregate benchmark reports.

The report bundle is machine-verifiable, but operators should not need to
read its JSON.  This module keeps the boundary checks strict and returns a
small, human-readable summary.  It never writes a receipt or changes a bundle.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from viola_handoff import HandoffError, VerifiedBundle, canonical_json_bytes, inspect_bundle

from .errors import ValidationError
from .jsonutil import sha256_file, sha256_json

POLICIES: Final = (
    "act",
    "diffusion",
    "vqbet",
    "smolvla",
    "pi0",
    "pi0_fast",
    "pi05",
    "groot",
)
REPORT_FILES: Final = (
    "benchmark_report.json",
    "benchmark_leaderboards.csv",
    "benchmark_report.md",
    "wandb_sync.complete.json",
)

# These values come from the benchmark configuration and aggregate-report
# schema merged on Repo-B main@6fcf643.
EXPECTED_EXPERIMENT: Final = "viola-cubes-v1"
EXPECTED_BENCHMARK_CONFIG_SHA256: Final = (
    "0f88094bfd70446f5a731e140bdcb7f0bcdea3bd3d4ae92ac10b02a86ff3e67e"
)
EXPECTED_DATASET_RELEASE_ID: Final = (
    "viola-cubes-right-to-left-blue-then-red-v1--31cf41385cd9e183"
)
EXPECTED_DATASET_SHA256: Final = (
    "79f603b7c03c5da61309d0052b187aafa7c93f1a2687a852ed5d4fbf8a991373"
)
EXPECTED_WANDB_PROJECT: Final = "starai-viola-policy-benchmark"
AGGREGATE_REPORT_SCHEMA_SHA256: Final = (
    "55c35315ec4447a2c4f12e18557eb109bd92d79896235319f5f2dce8f90b2ec2"
)

REPORT_EXECUTION_SCHEMA_VERSION: Final = 1
REPORT_SYNC_SCHEMA_VERSION: Final = 2
REPORT_EXECUTION_KIND: Final = "aggregate_report_execution"
REPORT_SYNC_KIND: Final = "aggregate_report_wandb_sync"
REPORT_RUN_PREFIX: Final = "report-"

METHODOLOGY: Final[dict[str, Any]] = {
    "primary_metric": "exact real-robot task success",
    "ranking": [
        "success_rate_desc",
        "safety_events_asc",
        "successful_completion_mean_sec_asc",
        "inference_p95_ms_asc",
    ],
    "confidence_interval": "95% Wilson score interval",
    "training_seed_note": "One training seed supports a descriptive comparison only.",
    "loss_note": (
        "Validation loss is a within-policy diagnostic and is never ranked across policies."
    ),
    "track_note": (
        "The one-camera VQ-BeT auxiliary track is not comparable to the two-camera track."
    ),
    "legacy_note": (
        "legacy_hash_bound, normalization-leakage, dirty-worktree, or shared-GPU runs "
        "preserve outcome evidence but are excluded from training-speed comparisons."
    ),
}

STATUS_STAGE: Final = {
    "scored": "rollout",
    "ineligible_training": "training",
    "ineligible_offline": "offline_evaluation",
    "ineligible_pc_runtime": "pc_runtime",
    "unsafe_shadow": "disconnected_shadow",
    "unsafe_shakedown": "shakedown",
    "blocked_setup": "session_preparation",
}
ALLOWED_DEVIATIONS: Final = frozenset(
    {"legacy_hash_bound", "normalization_leakage", "shared_gpu", "dirty_worktree"}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WANDB_URL = re.compile(
    r"^https://wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/"
    r"(?P<run_id>[A-Za-z0-9_-]+)$"
)
_REPORT_FIELDS = {
    "schema_version",
    "generated_at_utc",
    "complete",
    "methodology",
    "leaderboards",
    "terminal_outcomes",
    "blockers",
}
_ROW_FIELDS = {
    "policy",
    "track",
    "rank",
    "successes",
    "trials",
    "success_rate",
    "wilson_low",
    "wilson_high",
    "safety_events",
    "collisions",
    "interventions",
    "safety_aborts",
    "malformed_actions",
    "deadline_misses",
    "successful_completion_mean_sec",
    "successful_completion_median_sec",
    "inference_p95_ms",
    "gpu_hours",
    "peak_vram_gb",
    "training_throughput_samples_per_sec",
    "training_lineage_qualification",
    "normalization_scope",
    "training_deviations",
    "checkpoint_bytes",
    "first_action_mae",
    "first_action_rmse",
    "failure_counts",
}
_OUTCOME_FIELDS = {
    "policy",
    "status",
    "stage",
    "reason",
    "outcome_sha256",
    "evidence_sha256",
    "wandb_urls",
}


class ReportInspectionError(ValidationError):
    """A report bundle is incomplete, inconsistent, or not trustworthy."""


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """One policy's terminal benchmark classification."""

    policy: str
    status: str
    stage: str
    reason: str
    outcome_sha256: str
    evidence_sha256: tuple[str, ...]
    wandb_urls: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "outcome_sha256": self.outcome_sha256,
            "evidence_sha256": list(self.evidence_sha256),
            "wandb_urls": list(self.wandb_urls),
        }


@dataclass(frozen=True, slots=True)
class LeaderboardResult:
    """A recomputed and validated scored-policy result."""

    policy: str
    track: str
    rank: int
    successes: int
    trials: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    safety_events: int
    collisions: int
    interventions: int
    safety_aborts: int
    malformed_actions: int
    deadline_misses: int
    successful_completion_mean_sec: float | None
    successful_completion_median_sec: float | None
    inference_p95_ms: float
    gpu_hours: float
    peak_vram_gb: float
    training_throughput_samples_per_sec: float | None
    training_lineage_qualification: str
    normalization_scope: str
    training_deviations: tuple[str, ...]
    checkpoint_bytes: int
    first_action_mae: float
    first_action_rmse: float
    failure_counts: Mapping[str, int]

    def ranking_key(self) -> tuple[float, int, float, float]:
        completion = self.successful_completion_mean_sec
        return (
            -self.success_rate,
            self.safety_events,
            completion if completion is not None else math.inf,
            self.inference_p95_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "track": self.track,
            "rank": self.rank,
            "successes": self.successes,
            "trials": self.trials,
            "success_rate": self.success_rate,
            "wilson_low": self.wilson_low,
            "wilson_high": self.wilson_high,
            "safety_events": self.safety_events,
            "collisions": self.collisions,
            "interventions": self.interventions,
            "safety_aborts": self.safety_aborts,
            "malformed_actions": self.malformed_actions,
            "deadline_misses": self.deadline_misses,
            "successful_completion_mean_sec": self.successful_completion_mean_sec,
            "successful_completion_median_sec": self.successful_completion_median_sec,
            "inference_p95_ms": self.inference_p95_ms,
            "gpu_hours": self.gpu_hours,
            "peak_vram_gb": self.peak_vram_gb,
            "training_throughput_samples_per_sec": (
                self.training_throughput_samples_per_sec
            ),
            "training_lineage_qualification": self.training_lineage_qualification,
            "normalization_scope": self.normalization_scope,
            "training_deviations": list(self.training_deviations),
            "checkpoint_bytes": self.checkpoint_bytes,
            "first_action_mae": self.first_action_mae,
            "first_action_rmse": self.first_action_rmse,
            "failure_counts": dict(self.failure_counts),
        }


@dataclass(frozen=True, slots=True)
class WandbRunEvidence:
    entity: str
    project: str
    run_id: str
    url: str


class WandbVerifier(Protocol):
    """Injectable remote verifier; production uses the online implementation."""

    def __call__(
        self,
        evidence: WandbRunEvidence,
        expected_config: Mapping[str, Any],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """Operator-facing result of a successful strict inspection."""

    bundle_id: str
    generated_at_utc: str
    experiment: str
    dataset_release_id: str
    report_content_id: str
    wandb: WandbRunEvidence
    outcomes: tuple[TerminalOutcome, ...]
    two_camera: tuple[LeaderboardResult, ...]
    vqbet_auxiliary: tuple[LeaderboardResult, ...]
    remote_wandb_verified: bool = True

    @property
    def scored_policies(self) -> tuple[str, ...]:
        return tuple(outcome.policy for outcome in self.outcomes if outcome.status == "scored")

    def render_text(self) -> str:
        return render_text(self)


@dataclass(frozen=True, slots=True)
class _ValidatedReport:
    generated_at_utc: str
    outcomes: tuple[TerminalOutcome, ...]
    two_camera: tuple[LeaderboardResult, ...]
    vqbet_auxiliary: tuple[LeaderboardResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generated_at_utc": self.generated_at_utc,
            "complete": True,
            "methodology": METHODOLOGY,
            "leaderboards": {
                "two_camera": [row.to_dict() for row in self.two_camera],
                "vqbet_auxiliary": [row.to_dict() for row in self.vqbet_auxiliary],
            },
            "terminal_outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "blockers": [],
        }


def inspect_report(
    bundle_path: str | Path,
    *,
    wandb_verifier: WandbVerifier | None = None,
) -> ReportSummary:
    """Inspect one accepted report bundle without changing local or remote state.

    The default verifier performs a real W&B API read and requires a finished,
    content-bound run.  Tests can inject an equivalent local verifier; omitting
    verification is intentionally not a production option.
    """

    bundle = _accepted_report_bundle(bundle_path)
    paths = _report_paths(bundle)
    raw_report = _read_canonical_object(paths["benchmark_report.json"], "benchmark report")
    report = _parse_report(raw_report)

    expected_json = canonical_json_bytes(report.to_dict())
    if paths["benchmark_report.json"].read_bytes() != expected_json:
        raise ReportInspectionError(
            "benchmark_report.json differs from Repo B's typed canonical representation"
        )
    if paths["benchmark_leaderboards.csv"].read_bytes() != _render_csv(report):
        raise ReportInspectionError(
            "benchmark_leaderboards.csv is inconsistent with benchmark_report.json"
        )
    if paths["benchmark_report.md"].read_bytes() != _render_markdown(report):
        raise ReportInspectionError(
            "benchmark_report.md is inconsistent with benchmark_report.json"
        )

    sync = _read_canonical_object(
        paths["wandb_sync.complete.json"], "report W&B sync receipt"
    )
    core, content_id, wandb = _validate_sync(bundle, paths, report, sync)
    expected_config = {"report_content_id": content_id, "report_core": core}
    verifier = wandb_verifier or verify_remote_wandb
    try:
        verifier(wandb, expected_config)
    except ReportInspectionError:
        raise
    except Exception as exc:
        raise ReportInspectionError(f"report W&B verification failed: {exc}") from exc

    return ReportSummary(
        bundle_id=bundle.bundle_id,
        generated_at_utc=report.generated_at_utc,
        experiment=bundle.manifest["experiment"],
        dataset_release_id=EXPECTED_DATASET_RELEASE_ID,
        report_content_id=content_id,
        wandb=wandb,
        outcomes=report.outcomes,
        two_camera=report.two_camera,
        vqbet_auxiliary=report.vqbet_auxiliary,
    )


def verify_remote_wandb(
    evidence: WandbRunEvidence,
    expected_config: Mapping[str, Any],
) -> None:
    """Require the exact content-bound aggregate run to be finished online."""

    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    if mode in {"disabled", "dryrun", "offline"}:
        raise ReportInspectionError(f"WANDB_MODE={mode!r} cannot verify report evidence")
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise ReportInspectionError("WANDB_DISABLED prevents report evidence verification")
    try:
        import wandb

        remote = wandb.Api(timeout=20).run(
            f"{evidence.entity}/{evidence.project}/{evidence.run_id}"
        )
    except Exception as exc:  # pragma: no cover - exercised only with live W&B
        raise ReportInspectionError(f"could not read the report run from W&B: {exc}") from exc

    resolved = WandbRunEvidence(
        entity=str(getattr(remote, "entity", "") or "").strip(),
        project=str(getattr(remote, "project", "") or "").strip(),
        run_id=str(getattr(remote, "id", "") or "").strip(),
        url=str(getattr(remote, "url", "") or "").strip().rstrip("/"),
    )
    if resolved != evidence:
        raise ReportInspectionError("W&B resolved a different aggregate report run")
    if str(getattr(remote, "state", "") or "").strip().lower() != "finished":
        raise ReportInspectionError("the aggregate report W&B run is not finished")
    config = getattr(remote, "config", None)
    if not isinstance(config, Mapping):
        raise ReportInspectionError("the aggregate report W&B run has no readable config")
    mismatches = sorted(
        key for key, expected in expected_config.items() if config.get(key) != expected
    )
    if mismatches:
        raise ReportInspectionError(
            f"aggregate report W&B config differs in fields: {mismatches}"
        )


def render_text(summary: ReportSummary) -> str:
    """Render a concise operator view without implying robot readiness."""

    lines = [
        f"Viola benchmark report {summary.bundle_id[:12]}",
        f"Generated by Repo B: {summary.generated_at_utc}",
        f"Experiment: {summary.experiment}",
        f"Dataset release: {summary.dataset_release_id}",
        f"W&B evidence: finished and content-verified ({summary.wandb.url})",
        "",
        "Terminal outcomes:",
    ]
    lines.extend(
        f"  {outcome.policy}: {outcome.status} [{outcome.stage}] — {outcome.reason}"
        for outcome in summary.outcomes
    )
    lines.extend(("", "Two-camera leaderboard:"))
    lines.extend(_human_rows(summary.two_camera))
    lines.extend(("", "VQ-BeT auxiliary leaderboard:"))
    lines.extend(_human_rows(summary.vqbet_auxiliary))
    lines.extend(("", "Report state: accepted, complete, and blocker-free."))
    return "\n".join(lines)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return Repo B's two-sided 95% Wilson score interval."""

    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or not isinstance(trials, int)
        or isinstance(trials, bool)
        or trials <= 0
        or successes < 0
        or successes > trials
    ):
        raise ValueError("successes/trials must satisfy 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == trials else min(1.0, center + margin)
    return low, high


def _accepted_report_bundle(path: str | Path) -> VerifiedBundle:
    try:
        bundle = inspect_bundle(
            path,
            verify_artifacts=True,
            expected_kind="report",
            required_permission="report_only",
        )
    except (HandoffError, OSError) as exc:
        raise ReportInspectionError(f"report bundle verification failed: {exc}") from exc
    if bundle.manifest["consumer"] != {"role": "pc_a"}:
        raise ReportInspectionError("Repo A can inspect only a report addressed to pc_a")
    accepted = [
        receipt
        for receipt in bundle.receipts
        if receipt["status"] == "accepted" and receipt["actor"]["role"] == "pc_a"
    ]
    if not accepted:
        raise ReportInspectionError(
            "report has no receiver acceptance receipt; run viola-handoff accept first"
        )
    if bundle.manifest["artifacts"] != []:
        raise ReportInspectionError("report bundles must not reference external artifacts")
    if bundle.manifest["experiment"] != EXPECTED_EXPERIMENT:
        raise ReportInspectionError("report experiment differs from the approved benchmark")
    if bundle.manifest["subject"] != "all-policies":
        raise ReportInspectionError("report subject must be 'all-policies'")
    if bundle.manifest["wandb"]["project"] != EXPECTED_WANDB_PROJECT:
        raise ReportInspectionError("handoff W&B project differs from the approved benchmark")
    return bundle


def _report_paths(bundle: VerifiedBundle) -> dict[str, Path]:
    payload = bundle.manifest["payload"]
    declared = [entry["path"] for entry in payload["files"]]
    if len(declared) != len(REPORT_FILES) or set(declared) != set(REPORT_FILES):
        raise ReportInspectionError(
            f"report payload must contain exactly these files: {list(REPORT_FILES)}"
        )
    if payload["directories"] != []:
        raise ReportInspectionError("report payload must not contain subdirectories")
    return {name: bundle.payload_file(name) for name in REPORT_FILES}


def _parse_report(raw: Mapping[str, Any]) -> _ValidatedReport:
    _exact_keys(raw, _REPORT_FIELDS, "aggregate report")
    if raw["schema_version"] != "1.0":
        raise ReportInspectionError("aggregate report schema_version must be '1.0'")
    generated = _utc_timestamp(raw["generated_at_utc"], "report generated_at_utc")
    if raw["complete"] is not True:
        raise ReportInspectionError("aggregate report is not complete")
    if raw["blockers"] != []:
        raise ReportInspectionError("aggregate report contains blockers")
    if raw["methodology"] != METHODOLOGY:
        raise ReportInspectionError("aggregate report methodology differs from Repo B")

    leaderboards = _object(raw["leaderboards"], "leaderboards")
    _exact_keys(leaderboards, {"two_camera", "vqbet_auxiliary"}, "leaderboards")
    two_camera = _parse_rows(leaderboards["two_camera"], "two_camera")
    auxiliary = _parse_rows(leaderboards["vqbet_auxiliary"], "vqbet_auxiliary")
    outcomes = _parse_outcomes(raw["terminal_outcomes"])

    rows = (*two_camera, *auxiliary)
    if len({row.policy for row in rows}) != len(rows):
        raise ReportInspectionError("a policy appears more than once in the leaderboards")
    scored = {outcome.policy for outcome in outcomes if outcome.status == "scored"}
    if {row.policy for row in rows} != scored:
        raise ReportInspectionError("leaderboard policies differ from scored outcomes")
    for row in rows:
        if (row.policy == "vqbet") != (row.track == "vqbet_auxiliary"):
            raise ReportInspectionError("VQ-BeT is the only auxiliary-track policy")
    _validate_rank_order(two_camera)
    _validate_rank_order(auxiliary)
    return _ValidatedReport(generated, outcomes, two_camera, auxiliary)


def _parse_rows(value: Any, track: str) -> tuple[LeaderboardResult, ...]:
    if not isinstance(value, list):
        raise ReportInspectionError(f"{track} leaderboard must be a list")
    rows: list[LeaderboardResult] = []
    for index, item in enumerate(value):
        raw = _object(item, f"{track} row {index}")
        _exact_keys(raw, _ROW_FIELDS, f"{track} row {index}")
        if raw["track"] != track:
            raise ReportInspectionError(f"{track} row {index} is in the wrong track")
        policy = _policy(raw["policy"], f"{track} row {index} policy")
        deviations = _string_list(raw["training_deviations"], "training_deviations")
        if len(deviations) != len(set(deviations)) or set(deviations) - ALLOWED_DEVIATIONS:
            raise ReportInspectionError("leaderboard training deviations are invalid")
        failures = _positive_counts(raw["failure_counts"])
        row = LeaderboardResult(
            policy=policy,
            track=track,
            rank=_integer(raw["rank"], "rank", minimum=1),
            successes=_integer(raw["successes"], "successes", minimum=0, maximum=10),
            trials=_integer(raw["trials"], "trials", minimum=1),
            success_rate=_number(raw["success_rate"], "success_rate", minimum=0, maximum=1),
            wilson_low=_number(raw["wilson_low"], "wilson_low", minimum=0, maximum=1),
            wilson_high=_number(raw["wilson_high"], "wilson_high", minimum=0, maximum=1),
            safety_events=_integer(raw["safety_events"], "safety_events", minimum=0),
            collisions=_integer(raw["collisions"], "collisions", minimum=0),
            interventions=_integer(raw["interventions"], "interventions", minimum=0),
            safety_aborts=_integer(raw["safety_aborts"], "safety_aborts", minimum=0),
            malformed_actions=_integer(
                raw["malformed_actions"], "malformed_actions", minimum=0
            ),
            deadline_misses=_integer(
                raw["deadline_misses"], "deadline_misses", minimum=0
            ),
            successful_completion_mean_sec=_optional_number(
                raw["successful_completion_mean_sec"], "completion mean", minimum=0
            ),
            successful_completion_median_sec=_optional_number(
                raw["successful_completion_median_sec"], "completion median", minimum=0
            ),
            inference_p95_ms=_number(
                raw["inference_p95_ms"],
                "inference p95",
                minimum=0,
                maximum=300,
                exclusive_maximum=True,
            ),
            gpu_hours=_number(raw["gpu_hours"], "gpu_hours", minimum=0),
            peak_vram_gb=_number(raw["peak_vram_gb"], "peak_vram_gb", minimum=0),
            training_throughput_samples_per_sec=_optional_number(
                raw["training_throughput_samples_per_sec"],
                "training throughput",
                minimum=0,
            ),
            training_lineage_qualification=_choice(
                raw["training_lineage_qualification"],
                {"current_release_bound", "legacy_hash_bound"},
                "training lineage qualification",
            ),
            normalization_scope=_choice(
                raw["normalization_scope"],
                {"training_only", "legacy_dataset_wide"},
                "normalization scope",
            ),
            training_deviations=deviations,
            checkpoint_bytes=_integer(
                raw["checkpoint_bytes"], "checkpoint_bytes", minimum=1
            ),
            first_action_mae=_number(raw["first_action_mae"], "first_action_mae", minimum=0),
            first_action_rmse=_number(
                raw["first_action_rmse"], "first_action_rmse", minimum=0
            ),
            failure_counts=failures,
        )
        _validate_row(row)
        rows.append(row)
    return tuple(rows)


def _validate_row(row: LeaderboardResult) -> None:
    if row.trials != 10 or row.successes > row.trials:
        raise ReportInspectionError("scored rows require successes within ten trials")
    if not math.isclose(row.success_rate, row.successes / row.trials, abs_tol=1e-15):
        raise ReportInspectionError("success_rate differs from successes/trials")
    low, high = wilson_interval(row.successes, row.trials)
    if not math.isclose(row.wilson_low, low, abs_tol=1e-15) or not math.isclose(
        row.wilson_high, high, abs_tol=1e-15
    ):
        raise ReportInspectionError("Wilson interval is inconsistent")
    safety = (
        row.collisions
        + row.interventions
        + row.safety_aborts
        + row.malformed_actions
        + row.deadline_misses
    )
    if row.safety_events != safety:
        raise ReportInspectionError("safety_events differs from its component counts")
    if sum(row.failure_counts.values()) != row.trials - row.successes:
        raise ReportInspectionError("failure counts differ from failed trials")
    if (row.successes == 0) != (row.successful_completion_mean_sec is None):
        raise ReportInspectionError("completion mean is inconsistent with successes")
    if (row.successes == 0) != (row.successful_completion_median_sec is None):
        raise ReportInspectionError("completion median is inconsistent with successes")
    if (
        row.training_lineage_qualification == "legacy_hash_bound"
        and "legacy_hash_bound" not in row.training_deviations
    ):
        raise ReportInspectionError("legacy lineage must disclose legacy_hash_bound")
    if (
        row.normalization_scope == "legacy_dataset_wide"
        and "normalization_leakage" not in row.training_deviations
    ):
        raise ReportInspectionError("legacy normalization must disclose normalization_leakage")
    if bool(row.training_deviations) == (
        row.training_throughput_samples_per_sec is not None
    ):
        raise ReportInspectionError(
            "training speed must be omitted exactly when deviations are declared"
        )


def _parse_outcomes(value: Any) -> tuple[TerminalOutcome, ...]:
    if not isinstance(value, list) or len(value) != len(POLICIES):
        raise ReportInspectionError("report requires exactly eight terminal outcomes")
    outcomes: list[TerminalOutcome] = []
    for index, item in enumerate(value):
        raw = _object(item, f"terminal outcome {index}")
        _exact_keys(raw, _OUTCOME_FIELDS, f"terminal outcome {index}")
        policy = _policy(raw["policy"], f"terminal outcome {index} policy")
        status = _choice(raw["status"], set(STATUS_STAGE), "terminal status")
        stage = _nonempty(raw["stage"], "terminal stage")
        if stage != STATUS_STAGE[status]:
            raise ReportInspectionError("terminal outcome stage differs from its status")
        evidence = _sha256_list(raw["evidence_sha256"], "terminal evidence")
        urls = _wandb_urls(raw["wandb_urls"])
        outcomes.append(
            TerminalOutcome(
                policy=policy,
                status=status,
                stage=stage,
                reason=_nonempty(raw["reason"], "terminal reason"),
                outcome_sha256=_digest(raw["outcome_sha256"], "terminal outcome hash"),
                evidence_sha256=evidence,
                wandb_urls=urls,
            )
        )
    if tuple(outcome.policy for outcome in outcomes) != POLICIES:
        raise ReportInspectionError("terminal outcomes are not in the approved policy order")
    if len({outcome.outcome_sha256 for outcome in outcomes}) != len(outcomes):
        raise ReportInspectionError("terminal outcome source hashes must be unique")
    return tuple(outcomes)


def _validate_rank_order(rows: Sequence[LeaderboardResult]) -> None:
    ordered = sorted(rows, key=lambda row: (*row.ranking_key(), row.policy))
    if list(rows) != ordered:
        raise ReportInspectionError("leaderboard order differs from the preregistered ranking")
    previous_key: tuple[float, int, float, float] | None = None
    previous_rank = 0
    for position, row in enumerate(rows, start=1):
        key = row.ranking_key()
        expected_rank = previous_rank if key == previous_key else position
        if row.rank != expected_rank:
            raise ReportInspectionError("leaderboard rank differs from the ranking")
        previous_key = key
        previous_rank = expected_rank


def _validate_sync(
    bundle: VerifiedBundle,
    paths: Mapping[str, Path],
    report: _ValidatedReport,
    sync: Mapping[str, Any],
) -> tuple[dict[str, Any], str, WandbRunEvidence]:
    _exact_keys(
        sync,
        {
            "schema_version",
            "kind",
            "report_content_id",
            "report_core",
            "report_sha256",
            "wandb",
            "completed_at_utc",
        },
        "report W&B sync receipt",
    )
    if sync["schema_version"] != REPORT_SYNC_SCHEMA_VERSION or sync["kind"] != REPORT_SYNC_KIND:
        raise ReportInspectionError("report W&B sync schema/kind is unsupported")

    report_hashes = {
        "json": sha256_file(paths["benchmark_report.json"]),
        "csv": sha256_file(paths["benchmark_leaderboards.csv"]),
        "markdown": sha256_file(paths["benchmark_report.md"]),
    }
    terminal_hashes = {
        outcome.policy: outcome.outcome_sha256 for outcome in report.outcomes
    }
    core = {
        "schema_version": REPORT_EXECUTION_SCHEMA_VERSION,
        "kind": REPORT_EXECUTION_KIND,
        "contract_sha256": AGGREGATE_REPORT_SCHEMA_SHA256,
        "experiment": EXPECTED_EXPERIMENT,
        "benchmark_config_sha256": EXPECTED_BENCHMARK_CONFIG_SHA256,
        "dataset_release_id": EXPECTED_DATASET_RELEASE_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "terminal_outcome_sha256": terminal_hashes,
        "report_sha256": report_hashes,
    }
    content_id = sha256_json(core)
    if sync["report_core"] != core:
        raise ReportInspectionError("report W&B sync does not bind the approved execution core")
    if sync["report_content_id"] != content_id or sync["report_sha256"] != report_hashes:
        raise ReportInspectionError("report W&B sync hashes do not bind the report files")

    wandb_raw = _object(sync["wandb"], "report W&B lineage")
    _exact_keys(wandb_raw, {"entity", "project", "run_id", "url"}, "report W&B lineage")
    wandb = WandbRunEvidence(
        entity=_nonempty(wandb_raw["entity"], "W&B entity"),
        project=_nonempty(wandb_raw["project"], "W&B project"),
        run_id=_nonempty(wandb_raw["run_id"], "W&B run ID"),
        url=_nonempty(wandb_raw["url"], "W&B URL"),
    )
    if wandb.project != EXPECTED_WANDB_PROJECT:
        raise ReportInspectionError("report W&B project differs from the benchmark")
    if wandb.run_id != f"{REPORT_RUN_PREFIX}{content_id[:24]}":
        raise ReportInspectionError("report W&B run ID is not content-derived")
    expected_url = (
        f"https://wandb.ai/{wandb.entity}/{wandb.project}/runs/{wandb.run_id}"
    )
    if wandb.url != expected_url or _WANDB_URL.fullmatch(wandb.url) is None:
        raise ReportInspectionError("report W&B URL differs from its run identity")
    _utc_timestamp(sync["completed_at_utc"], "report W&B completion timestamp")

    lineage = bundle.manifest["lineage"]
    expected_lineage = {
        "benchmark_config_sha256": EXPECTED_BENCHMARK_CONFIG_SHA256,
        "dataset_release_id": EXPECTED_DATASET_RELEASE_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "aggregate_report_contract_sha256": AGGREGATE_REPORT_SCHEMA_SHA256,
        "report_content_id": content_id,
        "report_sha256": report_hashes,
        "terminal_outcome_sha256": terminal_hashes,
        "terminal_statuses": {
            outcome.policy: outcome.status for outcome in report.outcomes
        },
        "wandb": {
            "entity": wandb.entity,
            "project": wandb.project,
            "run_id": wandb.run_id,
            "url": wandb.url,
        },
    }
    if lineage != expected_lineage:
        raise ReportInspectionError("handoff lineage differs from the report execution evidence")
    return core, content_id, wandb


def _render_csv(report: _ValidatedReport) -> bytes:
    fields = (
        "track",
        "rank",
        "policy",
        "successes",
        "trials",
        "success_rate",
        "wilson_low",
        "wilson_high",
        "safety_events",
        "collisions",
        "interventions",
        "safety_aborts",
        "malformed_actions",
        "deadline_misses",
        "successful_completion_mean_sec",
        "successful_completion_median_sec",
        "inference_p95_ms",
        "gpu_hours",
        "peak_vram_gb",
        "training_throughput_samples_per_sec",
        "training_lineage_qualification",
        "normalization_scope",
        "training_deviations_json",
        "checkpoint_bytes",
        "first_action_mae",
        "first_action_rmse",
        "failure_counts_json",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in (*report.two_camera, *report.vqbet_auxiliary):
        data = row.to_dict()
        data["failure_counts_json"] = json.dumps(
            data.pop("failure_counts"), sort_keys=True
        )
        data["training_deviations_json"] = json.dumps(
            data.pop("training_deviations"), sort_keys=True
        )
        writer.writerow(data)
    return handle.getvalue().encode("utf-8")


def _render_markdown(report: _ValidatedReport) -> bytes:
    lines = [
        "# StarAI Viola Policy Benchmark",
        "",
        f"Generated: {report.generated_at_utc}",
        "",
        "> One training seed supports a descriptive comparison only. Validation loss is never ranked across policy families.",
        "",
        "## Terminal outcomes",
        "",
        "| Policy | Terminal status | Stage | Evidence |",
        "|---|---|---|---:|",
        *[
            f"| {outcome.policy} | {outcome.status} | {outcome.stage} | "
            f"{len(outcome.evidence_sha256)} |"
            for outcome in report.outcomes
        ],
        "",
        "## Two-camera leaderboard",
        "",
        *_markdown_table(report.two_camera),
        "",
        "## One-camera VQ-BeT auxiliary leaderboard",
        "",
        "> This auxiliary result is not directly comparable to the two-camera policies.",
        "",
        *_markdown_table(report.vqbet_auxiliary),
        "",
        "## Offline and compute diagnostics",
        "",
        "First-action errors are diagnostics only; they do not influence rank. A dash in Samples/s means the run has declared deviations and is excluded from speed comparison.",
        "",
        "| Track | Policy | Lineage | Normalization | Deviations | First-action MAE | First-action RMSE | GPU-hours | Peak VRAM (GB) | Samples/s | Checkpoint (GiB) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (*report.two_camera, *report.vqbet_auxiliary):
        lines.append(
            f"| {row.track} | {row.policy} | {row.training_lineage_qualification} | "
            f"{row.normalization_scope} | {', '.join(row.training_deviations) or 'none'} | "
            f"{_fmt(row.first_action_mae)} | {_fmt(row.first_action_rmse)} | "
            f"{_fmt(row.gpu_hours)} | {_fmt(row.peak_vram_gb)} | "
            f"{_fmt(row.training_throughput_samples_per_sec)} | "
            f"{row.checkpoint_bytes / (1024**3):.3f} |"
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _markdown_table(rows: Sequence[LeaderboardResult]) -> list[str]:
    lines = [
        "| Rank | Policy | Success | 95% Wilson CI | Safety events | Mean completion (s) | p95 latency (ms) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        lines.append("| — | No complete results | — | — | — | — | — |")
    for row in rows:
        lines.append(
            f"| {row.rank} | {row.policy} | {row.successes}/{row.trials} "
            f"({row.success_rate:.1%}) | [{row.wilson_low:.1%}, {row.wilson_high:.1%}] | "
            f"{row.safety_events} | {_fmt(row.successful_completion_mean_sec)} | "
            f"{_fmt(row.inference_p95_ms)} |"
        )
    return lines


def _human_rows(rows: Sequence[LeaderboardResult]) -> list[str]:
    if not rows:
        return ["  No scored policies."]
    return [
        f"  #{row.rank} {row.policy}: {row.successes}/{row.trials} "
        f"({row.success_rate:.1%}), Wilson [{row.wilson_low:.1%}, "
        f"{row.wilson_high:.1%}], safety events {row.safety_events}"
        for row in rows
    ]


def _read_canonical_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ReportInspectionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportInspectionError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInspectionError(f"{label} must contain one JSON object")
    if raw != canonical_json_bytes(value):
        raise ReportInspectionError(f"{label} is not canonical sorted compact JSON")
    return value


def _reject_constant(token: str) -> None:
    raise ReportInspectionError(f"non-finite JSON constant is forbidden: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReportInspectionError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ReportInspectionError(
            f"{label} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportInspectionError(f"{label} must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportInspectionError(f"{label} must be a nonempty string")
    return value


def _choice(value: Any, choices: set[str], label: str) -> str:
    text = _nonempty(value, label)
    if text not in choices:
        raise ReportInspectionError(f"{label} is unsupported: {text!r}")
    return text


def _policy(value: Any, label: str) -> str:
    return _choice(value, set(POLICIES), label)


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportInspectionError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ReportInspectionError(f"{label} must be <= {maximum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive_maximum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportInspectionError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ReportInspectionError(f"{label} must be finite and >= {minimum}")
    above_maximum = maximum is not None and number > maximum
    at_exclusive_maximum = (
        maximum is not None and exclusive_maximum and number == maximum
    )
    if above_maximum or at_exclusive_maximum:
        comparator = "<" if exclusive_maximum else "<="
        raise ReportInspectionError(f"{label} must be {comparator} {maximum}")
    return number


def _optional_number(value: Any, label: str, *, minimum: float) -> float | None:
    return None if value is None else _number(value, label, minimum=minimum)


def _positive_counts(value: Any) -> dict[str, int]:
    raw = _object(value, "failure_counts")
    result: dict[str, int] = {}
    for key, count in raw.items():
        if not isinstance(key, str) or not key:
            raise ReportInspectionError("failure_counts keys must be nonempty strings")
        result[key] = _integer(count, f"failure_counts.{key}", minimum=1)
    return result


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ReportInspectionError(f"{label} must be a string list")
    return tuple(value)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReportInspectionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ReportInspectionError(f"{label} must be a nonempty list")
    return tuple(_digest(item, label) for item in value)


def _wandb_urls(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ReportInspectionError("terminal W&B URLs must be a nonempty list")
    if not all(isinstance(url, str) and _WANDB_URL.fullmatch(url) for url in value):
        raise ReportInspectionError("terminal W&B URLs must be canonical run URLs")
    if len(value) != len(set(value)):
        raise ReportInspectionError("terminal W&B URLs must be unique")
    return tuple(value)


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ReportInspectionError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReportInspectionError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ReportInspectionError(f"{label} must use UTC")
    return value


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


__all__ = [
    "AGGREGATE_REPORT_SCHEMA_SHA256",
    "LeaderboardResult",
    "POLICIES",
    "REPORT_FILES",
    "ReportInspectionError",
    "ReportSummary",
    "TerminalOutcome",
    "WandbRunEvidence",
    "WandbVerifier",
    "inspect_report",
    "render_text",
    "verify_remote_wandb",
    "wilson_interval",
]
