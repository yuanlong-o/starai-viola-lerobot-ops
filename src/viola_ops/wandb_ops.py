"""Small online W&B helper for Repo-A evidence runs."""

from __future__ import annotations

import importlib.metadata
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from viola_handoff import HandoffError, canonical_json_bytes

from .errors import ValidationError


@dataclass(frozen=True, slots=True)
class WandbRunIdentity:
    entity: str
    project: str
    run_id: str
    url: str

    def binding(self) -> dict[str, str]:
        return {"run_id": self.run_id, "url": self.url}


def planned_run(entity: str, project: str, run_id: str) -> WandbRunIdentity:
    """Return a stable identity before evidence bytes are finalized."""

    for label, value in (("entity", entity), ("project", project), ("run_id", run_id)):
        if not isinstance(value, str) or not value.strip() or "/" in value:
            raise ValidationError(f"W&B {label} must be a nonempty path-safe string")
    return WandbRunIdentity(
        entity,
        project,
        run_id,
        f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
    )


def publish_finished_run(
    identity: WandbRunIdentity,
    *,
    job_type: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    timeout_s: float = 60.0,
    wandb_module: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WandbRunIdentity:
    """Publish scalar lineage and prove the finished remote run contains it."""

    _require_online_environment()
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s < 0
    ):
        raise ValidationError("W&B timeout must be a finite nonnegative number")
    if importlib.metadata.version("wandb") != "0.27.2":
        raise ValidationError("evidence publishing requires W&B 0.27.2")
    expected_config = _json_mapping(config, "W&B config")
    expected_summary = _json_mapping(summary, "W&B summary")
    wandb = wandb_module
    if wandb is None:
        import wandb as imported_wandb

        wandb = imported_wandb
    try:
        api = wandb.Api(timeout=timeout_s)
        if not getattr(api, "api_key", None):
            raise ValidationError("W&B is not authenticated; run `wandb login`")
        run = wandb.init(
            entity=identity.entity,
            project=identity.project,
            id=identity.run_id,
            name=identity.run_id,
            job_type=job_type,
            config=expected_config,
            mode="online",
            resume="allow",
            reinit=True,
        )
        if run is None:
            raise ValidationError("W&B did not create an online evidence run")
        run.summary.update(expected_summary)
        run.finish(exit_code=0)
        deadline = clock() + timeout_s
        path = f"{identity.entity}/{identity.project}/{identity.run_id}"
        last_mismatch = "remote run has not reached state=finished"
        while True:
            try:
                remote = api.run(path)
            except Exception as exc:
                # A newly finished run can briefly be unavailable through the
                # public API. Retry ordinary lookup failures until the same
                # bounded deadline; process-control exceptions still escape.
                last_mismatch = f"remote lookup failed: {type(exc).__name__}"
            else:
                if str(remote.state).lower() == "finished":
                    mismatch = _remote_mismatch(
                        remote,
                        identity=identity,
                        job_type=job_type,
                        config=expected_config,
                        summary=expected_summary,
                    )
                    if mismatch is None:
                        return identity
                    last_mismatch = mismatch
            if clock() >= deadline:
                raise ValidationError(
                    "W&B run did not become exact, finished evidence within "
                    f"{timeout_s:g} seconds: {last_mismatch}"
                )
            sleep(1.0)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"online W&B evidence failed: {type(exc).__name__}: {exc}") from exc


def _require_online_environment() -> None:
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        raise ValidationError("W&B is disabled")
    if os.environ.get("WANDB_MODE", "online").strip().lower() != "online":
        raise ValidationError("evidence requires WANDB_MODE=online")


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Return a plain mapping after the shared canonical encoder accepts it."""

    result = dict(value)
    try:
        canonical_json_bytes(result)
    except (HandoffError, TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not canonical JSON data: {exc}") from exc
    return result


def _remote_mismatch(
    remote: Any,
    *,
    identity: WandbRunIdentity,
    job_type: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str | None:
    """Describe the first remote mismatch, allowing unrelated W&B fields."""

    expected_path = [identity.entity, identity.project, identity.run_id]
    try:
        remote_path = list(remote.path)
    except (AttributeError, TypeError):
        return "remote path is unavailable"
    if remote_path != expected_path:
        return "remote entity/project/run ID differs from the planned identity"
    if str(getattr(remote, "entity", "")) != identity.entity:
        return "remote entity differs from the planned identity"
    if str(getattr(remote, "id", "")) != identity.run_id:
        return "remote run ID differs from the planned identity"
    if str(getattr(remote, "name", "")) != identity.run_id:
        return "remote run name differs from the planned identity"
    if str(getattr(remote, "url", "")).rstrip("/") != identity.url:
        return "remote URL differs from the planned identity"
    if str(getattr(remote, "job_type", "")) != job_type:
        return "remote job type differs from the submitted evidence"
    mismatch = _mapping_mismatch(getattr(remote, "config", None), config, "config")
    if mismatch is not None:
        return mismatch
    return _mapping_mismatch(getattr(remote, "summary", None), summary, "summary")


def _mapping_mismatch(remote: Any, expected: Mapping[str, Any], label: str) -> str | None:
    try:
        actual = dict(remote)
    except (TypeError, ValueError):
        return f"remote {label} is unavailable"
    for key, expected_value in expected.items():
        if key not in actual:
            return f"remote {label} is missing {key!r}"
        try:
            exact = canonical_json_bytes(actual[key]) == canonical_json_bytes(expected_value)
        except (HandoffError, TypeError, ValueError):
            return f"remote {label} field {key!r} is not canonical JSON data"
        if not exact:
            return f"remote {label} field {key!r} differs from the submitted evidence"
    return None


__all__ = ["WandbRunIdentity", "planned_run", "publish_finished_run"]
