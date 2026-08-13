from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from viola_ops.errors import ValidationError
from viola_ops.wandb_ops import planned_run, publish_finished_run


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _LocalRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.finished_with: int | None = None

    def finish(self, *, exit_code: int) -> None:
        self.finished_with = exit_code


class _Api:
    api_key = "authenticated"

    def __init__(self, remotes: list[Any]) -> None:
        self.remotes = remotes
        self.calls = 0

    def run(self, path: str) -> Any:
        assert path == "entity/project/evidence-1"
        remote = self.remotes[min(self.calls, len(self.remotes) - 1)]
        self.calls += 1
        if isinstance(remote, Exception):
            raise remote
        return remote


class _Wandb:
    def __init__(self, remotes: list[Any]) -> None:
        self.api = _Api(remotes)
        self.local = _LocalRun()
        self.init_calls: list[dict[str, Any]] = []

    def Api(self, *, timeout: float) -> _Api:  # noqa: N802 - mirrors W&B's public API
        assert timeout >= 0
        return self.api

    def init(self, **kwargs: Any) -> _LocalRun:
        self.init_calls.append(kwargs)
        return self.local


def _remote(**changes: Any) -> Any:
    values = {
        "state": "finished",
        "path": ["entity", "project", "evidence-1"],
        "entity": "entity",
        "id": "evidence-1",
        "name": "evidence-1",
        "url": "https://wandb.ai/entity/project/runs/evidence-1",
        "job_type": "viola-test",
        "config": {"operation": "verify", "nested": {"values": [1, 2]}, "_wandb": {}},
        "summary": {"ready": True, "count": 7, "_step": 0},
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _publish(module: _Wandb, *, timeout_s: float = 2.0) -> None:
    clock = _Clock()
    result = publish_finished_run(
        planned_run("entity", "project", "evidence-1"),
        job_type="viola-test",
        config={"operation": "verify", "nested": {"values": [1, 2]}},
        summary={"ready": True, "count": 7},
        timeout_s=timeout_s,
        wandb_module=module,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.run_id == "evidence-1"


def test_publish_proves_exact_remote_binding_and_allows_system_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([_remote()])

    _publish(module)

    assert module.local.finished_with == 0
    assert module.local.summary == {"ready": True, "count": 7}
    assert module.init_calls == [
        {
            "entity": "entity",
            "project": "project",
            "id": "evidence-1",
            "name": "evidence-1",
            "job_type": "viola-test",
            "config": {"operation": "verify", "nested": {"values": [1, 2]}},
            "mode": "online",
            "resume": "allow",
            "reinit": True,
        }
    ]


def test_publish_waits_for_finished_remote_config_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb(
        [
            _remote(state="running"),
            _remote(config={"operation": "old"}),
            _remote(summary={"ready": False, "count": 7}),
            _remote(),
        ]
    )

    _publish(module, timeout_s=4.0)

    assert module.api.calls == 4


def test_publish_retries_transient_remote_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([RuntimeError("not visible yet"), _remote()])

    _publish(module)

    assert module.api.calls == 2


def test_publish_bounds_persistent_remote_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([RuntimeError("temporary API failure")])

    with pytest.raises(ValidationError, match="remote lookup failed: RuntimeError"):
        _publish(module, timeout_s=2.0)

    assert module.api.calls == 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"path": ["entity", "other", "evidence-1"]}, "entity/project/run ID"),
        ({"entity": "other"}, "remote entity differs"),
        ({"id": "other"}, "remote run ID differs"),
        ({"name": "other"}, "remote run name differs"),
        ({"url": "https://wandb.ai/entity/project/runs/other"}, "remote URL differs"),
        ({"job_type": "other"}, "remote job type differs"),
        ({"config": {"operation": "old"}}, "config field 'operation' differs"),
        (
            {"config": {"nested": {"values": [1, 2]}}},
            "config is missing 'operation'",
        ),
        ({"summary": {"ready": False, "count": 7}}, "summary field 'ready' differs"),
        ({"summary": {"ready": True}}, "summary is missing 'count'"),
    ],
)
def test_publish_rejects_finished_remote_with_another_binding(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([_remote(**changes)])

    with pytest.raises(ValidationError, match=message):
        _publish(module, timeout_s=0.0)


def test_same_content_retry_reopens_the_same_remote_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([_remote()])

    _publish(module)
    _publish(module)

    assert len(module.init_calls) == 2
    assert {call["id"] for call in module.init_calls} == {"evidence-1"}


def test_non_json_evidence_is_rejected_before_wandb_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([_remote()])

    with pytest.raises(ValidationError, match="not canonical JSON data"):
        publish_finished_run(
            planned_run("entity", "project", "evidence-1"),
            job_type="viola-test",
            config={"bad": float("nan")},
            summary={"ready": True},
            wandb_module=module,
        )

    assert module.init_calls == []


@pytest.mark.parametrize("timeout_s", [math.nan, math.inf, -1.0, True])
def test_publish_rejects_unbounded_or_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
    timeout_s: float,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    module = _Wandb([_remote()])

    with pytest.raises(ValidationError, match="finite nonnegative"):
        _publish(module, timeout_s=timeout_s)

    assert module.init_calls == []
