from __future__ import annotations

import pytest

from scripts import record_episodes_keep_pose


def test_retired_recorder_exits_with_migration_message(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        record_episodes_keep_pose.main()

    assert stopped.value.code == record_episodes_keep_pose.EXIT_RETIRED
    message = capsys.readouterr().err
    assert "episode recording is disabled" in message
    assert "not supported by the live-session contract" in message


def test_retired_recorder_has_no_runtime_dependencies() -> None:
    module_names = {
        value.__module__
        for value in vars(record_episodes_keep_pose).values()
        if callable(value) and getattr(value, "__module__", None)
    }

    assert module_names == {"scripts.record_episodes_keep_pose"}
