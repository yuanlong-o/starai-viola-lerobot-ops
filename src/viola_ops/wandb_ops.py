"""Small online W&B helper for Repo-A evidence runs."""

from __future__ import annotations

import importlib.metadata
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

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
    """Publish scalar lineage and prove the remote run reached ``finished``."""

    _require_online_environment()
    if importlib.metadata.version("wandb") != "0.27.2":
        raise ValidationError("evidence publishing requires W&B 0.27.2")
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
            config=dict(config),
            mode="online",
            resume="allow",
            reinit=True,
        )
        if run is None:
            raise ValidationError("W&B did not create an online evidence run")
        run.summary.update(dict(summary))
        run.finish(exit_code=0)
        deadline = clock() + timeout_s
        path = f"{identity.entity}/{identity.project}/{identity.run_id}"
        while True:
            remote = api.run(path)
            if str(remote.state).lower() == "finished":
                remote_url = str(remote.url).rstrip("/")
                if remote_url != identity.url:
                    raise ValidationError("W&B run URL resolved to another identity")
                return identity
            if clock() >= deadline:
                raise ValidationError("W&B run did not reach remote state=finished within 60 seconds")
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


__all__ = ["WandbRunIdentity", "planned_run", "publish_finished_run"]
