from __future__ import annotations

import json
from pathlib import Path

import pytest

from viola_handoff import RuntimeIdentity, SealRequest, seal_bundle
from viola_handoff.cli import main


class FakeEvidence:
    def record(self, **event: object) -> str:
        return f"https://wandb.ai/tester/{event['project']}/runs/{event['run_id']}"


def test_inspect_cli_reports_verified_bundle(tmp_path: Path, capsys) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "evidence.json").write_bytes(b"{}")
    producer = RuntimeIdentity("pc_a", "a" * 40, True, "a", "3.12.13", "0.6.1", "lerobot")
    bundle = seal_bundle(
        SealRequest(
            root=tmp_path / "nas",
            kind="dataset_release",
            experiment="exp",
            subject="dataset",
            producer=producer,
            lineage={},
            wandb_project="project",
            payload_dir=payload,
        ),
        evidence_logger=FakeEvidence(),
    )

    assert main(["inspect", str(bundle.path), "--expected-kind", "dataset_release"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["bundle_id"] == bundle.bundle_id
    assert output["artifacts_verified"] is True
    assert output["wandb_run_id"].startswith("ho-")


def test_generic_cli_cannot_seal_rollout_session(tmp_path: Path, capsys) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "rollout_session.json").write_bytes(b"{}")
    lineage = tmp_path / "lineage.json"
    lineage.write_text('{"blockers":[]}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "seal",
                "--root",
                str(tmp_path / "nas"),
                "--kind",
                "rollout_session",
                "--experiment",
                "exp",
                "--subject",
                "session",
                "--payload-dir",
                str(payload),
                "--lineage",
                str(lineage),
                "--wandb-project",
                "project",
                "--repo-root",
                str(tmp_path),
            ]
        )
    assert "generic sealing cannot grant rollout_session authority" in capsys.readouterr().err
    assert not (tmp_path / "nas").exists()
