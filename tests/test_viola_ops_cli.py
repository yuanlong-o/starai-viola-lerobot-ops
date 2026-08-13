from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import viola_ops.cli as cli
from viola_ops.errors import ValidationError


class Backends:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.confirmation = lambda *_args, **_kwargs: True
        bundle = SimpleNamespace(
            bundle_id="b" * 64,
            path=Path("/accepted/bundle"),
            permission="data_only",
            ready={"wandb_url": "https://wandb.ai/entity/project/runs/report"},
        )

        def validate_dataset(*args, **kwargs):
            self.calls.append(("validate_dataset", args, kwargs))
            return SimpleNamespace(
                release_id="dataset-v1",
                root=Path("/dataset"),
                episodes=34,
                frames=28_306,
                inventory_sha256="d" * 64,
                full_decode=kwargs["full_decode"],
            )

        def release_dataset(*args, **kwargs):
            self.calls.append(("release_dataset", args, kwargs))
            return SimpleNamespace(bundle=bundle)

        def seal_session_inputs(*args, **kwargs):
            self.calls.append(("seal_session_inputs", args, kwargs))
            return SimpleNamespace(
                bundle=SimpleNamespace(
                    **{**bundle.__dict__, "permission": "planning_only"}
                ),
                receiver_destination=Path("/receiver"),
            )

        def capture_frozen_state(*args, **kwargs):
            self.calls.append(("capture_frozen_state", args, kwargs))
            return {"status": "captured", "positions": 7}

        def verify_command(*args, **kwargs):
            self.calls.append(("verify_command", args, kwargs))
            return {"policy": "act", "status": "eligible"}

        def shadow_command(*args, **kwargs):
            self.calls.append(("shadow_command", args, kwargs))
            return {"policy": "act", "mode": "replay", "status": "passed"}

        def execute_command(*args, **kwargs):
            self.calls.append(("execute_command", args, kwargs))
            return {"policy": "act", "phase": kwargs["phase"], "status": "completed"}

        class ReportSummary:
            def render_text(_self) -> str:
                return "Eight-policy report verified"

        def inspect_report(*args, **kwargs):
            self.calls.append(("inspect_report", args, kwargs))
            return ReportSummary()

        self.modules = {
            "dataset": SimpleNamespace(
                validate_dataset=validate_dataset,
                release_dataset=release_dataset,
            ),
            "session_inputs": SimpleNamespace(seal_session_inputs=seal_session_inputs),
            "setup": SimpleNamespace(
                capture_frozen_state=capture_frozen_state,
                interactive_confirmation=self.confirmation,
            ),
            "policy_ops": SimpleNamespace(
                verify_command=verify_command,
                shadow_command=shadow_command,
            ),
            "execute_ops": SimpleNamespace(execute_command=execute_command),
            "report": SimpleNamespace(inspect_report=inspect_report),
        }

    def importer(self, qualified_name: str):
        return self.modules[qualified_name.rsplit(".", 1)[-1]]


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> Backends:
    value = Backends()
    monkeypatch.setattr(cli, "import_module", value.importer)
    return value


def test_top_level_help_lists_every_operator_area(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    for area in ("dataset", "session-inputs", "setup", "policy", "report"):
        assert area in output
    assert "No physical command bypasses accepted-session safety gates" in output


def test_execute_help_explains_where_evidence_may_be_written(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["policy", "execute", "--help"])
    assert stopped.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "shared NAS base visible to PCs A and B" in output
    assert "outside the Git worktree" in output
    assert "immutable nested path" in output


def test_cross_pc_artifact_defaults_use_shared_nas(capsys) -> None:
    assert cli.DEFAULT_MATERIAL_ROOT == Path(
        "/mnt/nas02/yz/starai/producer-materials/v1"
    )
    assert cli.DEFAULT_OUTPUT_ROOT == Path("/mnt/nas02/yz/starai/evidence/v1")

    for arguments, expected in (
        (["session-inputs", "produce", "--help"], "setup_record readable by PC B"),
        (["policy", "shadow", "--help"], "shadow_record readable by PC B"),
    ):
        with pytest.raises(SystemExit) as stopped:
            cli.main(arguments)
        assert stopped.value.code == 0
        output = " ".join(capsys.readouterr().out.split())
        assert expected in output
        assert "/mnt/nas02/yz/starai/" in output


def test_operator_docs_keep_cross_pc_artifacts_on_shared_storage() -> None:
    repository = Path(__file__).resolve().parents[1]
    for relative in ("README.md", "docs/COMMAND_REFERENCE.md"):
        text = (repository / relative).read_text(encoding="utf-8")
        prose = " ".join(text.split())
        assert "/mnt/nas02/yz/starai/producer-materials/v1" in text
        assert "/mnt/nas02/yz/starai/evidence/v1" in text
        assert "policy_specific_act_blocker" in prose
        assert "shared hardware, control, safety, evidence, and W&B" in prose


@pytest.mark.parametrize(
    ("arguments", "phrase"),
    [
        (["dataset", "validate", "--help"], "both videos"),
        (["dataset", "release", "--help"], "data-only bundle"),
        (["session-inputs", "produce", "--help"], "planning-only bundle"),
        (["setup", "capture-frozen-state", "--help"], "without issuing a motor command"),
        (["policy", "verify", "--help"], "without connecting hardware"),
        (["policy", "shadow", "--help"], "without motor commands"),
        (["policy", "execute", "--help"], "interactive operator arming"),
        (["report", "inspect", "--help"], "readable summary"),
    ],
)
def test_every_named_command_has_human_help(arguments, phrase: str, capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 0
    assert phrase in capsys.readouterr().out


def test_dataset_validate_delegates_without_video_when_requested(
    backends: Backends, capsys
) -> None:
    assert cli.main(["dataset", "validate", "--root", "/data", "--numeric-only"]) == 0
    assert backends.calls == [
        ("validate_dataset", (Path("/data"),), {"full_decode": False})
    ]
    output = capsys.readouterr().out
    assert "Dataset validation passed" in output
    assert "34" in output
    assert "diagnostic only" in output


def test_dataset_release_passes_explicit_roots(backends: Backends, capsys) -> None:
    assert (
        cli.main(
            [
                "dataset",
                "release",
                "--root",
                "/data",
                "--experiment",
                "exp",
                "--handoff-root",
                "/handoffs",
                "--material-root",
                "/materials",
                "--repo-root",
                "/repo",
                "--wandb-project",
                "project",
            ]
        )
        == 0
    )
    name, positional, keywords = backends.calls[0]
    assert name == "release_dataset"
    assert positional == (Path("/data"),)
    assert keywords == {
        "experiment": "exp",
        "handoff_root": Path("/handoffs"),
        "material_root": Path("/materials"),
        "wandb_project": "project",
        "repo_root": Path("/repo"),
    }
    assert "Dataset release sealed" in capsys.readouterr().out


def test_session_inputs_produce_delegates_exact_setup(backends: Backends, capsys) -> None:
    assert (
        cli.main(
            [
                "session-inputs",
                "produce",
                "--setup",
                "/reviewed.json",
                "--subject",
                "setup-01",
                "--repo-root",
                "/repo",
            ]
        )
        == 0
    )
    name, positional, keywords = backends.calls[0]
    assert name == "seal_session_inputs"
    assert positional == (Path("/reviewed.json"),)
    assert keywords["subject"] == "setup-01"
    assert keywords["repo_root"] == Path("/repo")
    assert keywords["material_root"] == cli.DEFAULT_MATERIAL_ROOT
    assert "Session inputs sealed" in capsys.readouterr().out


def test_capture_passes_only_backend_interactive_confirmation(
    backends: Backends, capsys
) -> None:
    assert (
        cli.main(
            [
                "setup",
                "capture-frozen-state",
                "--setup",
                "/setup.json",
                "--output-root",
                "/evidence",
                "--operator",
                "operator-a",
                "--repo-root",
                "/repo",
                "--wandb-entity",
                "entity",
            ]
        )
        == 0
    )
    name, positional, keywords = backends.calls[0]
    assert name == "capture_frozen_state"
    assert positional == (Path("/setup.json"),)
    assert keywords["confirm"] is backends.confirmation
    assert keywords["operator"] == "operator-a"
    assert keywords["wandb_entity"] == "entity"
    assert "Frozen state captured" in capsys.readouterr().out


def test_policy_verify_delegates_to_disconnected_backend(backends: Backends, capsys) -> None:
    assert (
        cli.main(
            [
                "policy",
                "verify",
                "--bundle",
                "/candidate",
                "--output-root",
                "/evidence",
                "--repo-root",
                "/repo",
                "--handoff-root",
                "/handoffs",
                "--wandb-entity",
                "entity",
            ]
        )
        == 0
    )
    name, positional, keywords = backends.calls[0]
    assert name == "verify_command"
    assert positional == (
        Path("/candidate"),
        Path("/evidence"),
        Path("/repo"),
        cli.DEFAULT_WANDB_PROJECT,
        "entity",
    )
    assert keywords == {"handoff_root": Path("/handoffs")}
    assert "Eligible: eligible" not in capsys.readouterr().out


def test_replay_shadow_delegates_without_setup_or_cameras(backends: Backends, capsys) -> None:
    assert (
        cli.main(
            [
                "policy",
                "shadow",
                "--bundle",
                "/candidate",
                "--mode",
                "replay",
                "--verification",
                "/verification.json",
                "--wandb-entity",
                "entity",
            ]
        )
        == 0
    )
    name, _positional, keywords = backends.calls[0]
    assert name == "shadow_command"
    assert backends.calls[0][1][1] == "replay"
    assert backends.calls[0][1][3] == cli.DEFAULT_OUTPUT_ROOT
    assert backends.calls[0][1][-2:] == (None, None)
    assert keywords == {}
    assert "Policy: act" in capsys.readouterr().out


def test_live_soak_delegates_only_with_reviewed_setup_and_frozen_state(
    backends: Backends, capsys
) -> None:
    assert (
        cli.main(
            [
                "policy",
                "shadow",
                "--bundle",
                "/candidate",
                "--mode",
                "live-soak",
                "--verification",
                "/verification.json",
                "--setup",
                "/setup.json",
                "--frozen-state",
                "/state.json",
                "--wandb-entity",
                "entity",
            ]
        )
        == 0
    )
    name, positional, keywords = backends.calls[0]
    assert name == "shadow_command"
    assert positional[1] == "live-soak"
    assert positional[-2:] == (Path("/setup.json"), Path("/state.json"))
    assert keywords == {}
    assert "Policy: act" in capsys.readouterr().out


def test_scored_execute_requires_both_predecessors_before_backend_import(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    imported = False

    def fail_import(_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("backend must not import")

    monkeypatch.setattr(cli, "import_module", fail_import)
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "policy",
                "execute",
                "--session",
                "/session",
                "--candidate",
                "/candidate",
                "--phase",
                "scored",
                "--trial",
                "scored-set",
                "--evidence-root",
                "/evidence",
                "--wandb-entity",
                "entity",
            ]
        )
    assert stopped.value.code == 2
    assert imported is False
    assert "requires --prior-hold and --prior-shakedown" in capsys.readouterr().err


def test_execute_delegates_only_after_phase_evidence_is_present(
    backends: Backends, capsys
) -> None:
    assert (
        cli.main(
            [
                "policy",
                "execute",
                "--session",
                "/session",
                "--candidate",
                "/candidate",
                "--phase",
                "scored",
                "--trial",
                "scored-set",
                "--evidence-root",
                "/evidence",
                "--wandb-entity",
                "entity",
                "--prior-hold",
                "/hold",
                "--prior-shakedown",
                "/shakedown",
            ]
        )
        == 0
    )
    name, positional, keywords = backends.calls[0]
    assert name == "execute_command"
    assert positional == (Path("/session"),)
    assert keywords["candidate"] == Path("/candidate")
    assert keywords["prior_hold_bundle"] == Path("/hold")
    assert keywords["prior_shakedown_bundle"] == Path("/shakedown")
    assert keywords["wandb_entity"] == "entity"
    assert "Phase: scored" in capsys.readouterr().out


def test_report_inspect_uses_human_renderer(backends: Backends, capsys) -> None:
    assert cli.main(["report", "inspect", "--bundle", "/report"]) == 0
    assert backends.calls == [("inspect_report", (Path("/report"),), {})]
    assert capsys.readouterr().out == "Eight-policy report verified\n"


def test_backend_errors_are_concise_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def reject(*_args, **_kwargs):
        raise ValidationError("candidate has no acceptance receipt")

    monkeypatch.setattr(
        cli,
        "import_module",
        lambda _name: SimpleNamespace(verify_command=reject),
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "policy",
                "verify",
                "--bundle",
                "/candidate",
                "--wandb-entity",
                "entity",
            ]
        )
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert error == "viola-ops: error: candidate has no acceptance receipt\n"
    assert "Traceback" not in error
