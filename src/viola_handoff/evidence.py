"""Online W&B evidence logging for immutable handoffs.

This module deliberately has a very small surface.  It sends scalar lineage
metadata only: it never creates an Artifact, calls ``save``, or uploads a
payload/checkpoint/dataset file.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Protocol

from .errors import EvidenceError


class EvidenceLogger(Protocol):
    """Injectable evidence sink used by seal, accept, and acknowledgement."""

    def record(
        self,
        *,
        project: str,
        run_id: str,
        event: str,
        metadata: Mapping[str, str | int | float | bool | None],
    ) -> str:
        """Record one online event and return its public/private run URL."""


class WandbEvidenceLogger:
    """Mandatory online W&B implementation used outside tests."""

    def __init__(self, *, entity: str | None = None) -> None:
        self.entity = entity

    def record(
        self,
        *,
        project: str,
        run_id: str,
        event: str,
        metadata: Mapping[str, str | int | float | bool | None],
    ) -> str:
        if os.environ.get("WANDB_MODE", "online").lower() != "online":
            raise EvidenceError("WANDB_MODE must be 'online' for handoff writes")
        if os.environ.get("WANDB_DISABLED", "false").lower() in {"1", "true", "yes"}:
            raise EvidenceError("W&B cannot be disabled for handoff writes")

        finished = False
        try:
            import wandb

            if not wandb.login(relogin=False):
                raise EvidenceError("W&B authentication is required")
            settings = wandb.Settings(disable_git=True)
            run = wandb.init(
                project=project,
                entity=self.entity,
                id=run_id,
                resume="allow",
                mode="online",
                job_type="viola-handoff",
                save_code=False,
                settings=settings,
            )
        except EvidenceError:
            raise
        except Exception as exc:  # pragma: no cover - exercised against W&B in integration
            raise EvidenceError(f"could not start mandatory online W&B run: {exc}") from exc

        if run is None:  # pragma: no cover - defensive against third-party behavior
            raise EvidenceError("wandb.init returned no online run")

        try:
            if bool(getattr(run, "offline", False)):
                raise EvidenceError("W&B initialized an offline run")
            mode = getattr(getattr(run, "settings", None), "mode", "online")
            if mode != "online":
                raise EvidenceError(f"W&B initialized in unsupported mode {mode!r}")

            # Only bounded scalar metadata enters W&B.  Raw payload/artifact bytes,
            # file contents, and per-file inventories remain on the filesystem.
            event_data: dict[str, Any] = {"handoff/event": event}
            for key, value in metadata.items():
                if not isinstance(key, str) or not isinstance(
                    value, (str, int, float, bool, type(None))
                ):
                    raise EvidenceError("W&B handoff metadata must contain scalars only")
                event_data[f"handoff/{key}"] = value
            run.log(event_data)
            run.summary["handoff_last_event"] = event
            run.summary["handoff_contract"] = metadata.get("contract_sha256")
            url = getattr(run, "url", None)
            if not isinstance(url, str) or not url.startswith("https://"):
                raise EvidenceError("online W&B handoff run did not expose a URL")
            run.finish(exit_code=0)
            finished = True
            _require_remote_finished(wandb, run, run_id=run_id)
            return url
        except EvidenceError:
            if not finished:
                try:
                    run.finish(exit_code=1)
                finally:
                    raise
            raise
        except Exception as exc:  # pragma: no cover - exercised against W&B in integration
            if not finished:
                try:
                    run.finish(exit_code=1)
                finally:
                    raise EvidenceError(f"could not persist mandatory W&B evidence: {exc}") from exc
            raise EvidenceError(f"could not verify mandatory W&B evidence: {exc}") from exc


def _require_remote_finished(wandb: Any, run: Any, *, run_id: str) -> None:
    """Boundedly prove that the just-finished online run is visible as finished."""

    raw_path = getattr(run, "path", None)
    if isinstance(raw_path, (list, tuple)) and len(raw_path) == 3:
        run_path = "/".join(str(part) for part in raw_path)
    elif isinstance(raw_path, str) and raw_path.count("/") == 2:
        run_path = raw_path
    else:
        entity = getattr(run, "entity", None)
        project = getattr(run, "project", None)
        if not isinstance(entity, str) or not entity or not isinstance(project, str) or not project:
            raise EvidenceError("online W&B run did not expose a canonical entity/project path")
        run_path = f"{entity}/{project}/{run_id}"

    deadline = time.monotonic() + 60.0
    last_state: str | None = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            remote = wandb.Api(timeout=20).run(run_path)
            state = getattr(remote, "state", None)
            last_state = state if isinstance(state, str) else None
            if last_state == "finished":
                return
            if last_state in {"failed", "crashed", "killed"}:
                raise EvidenceError(f"online W&B handoff run ended as {last_state!r}")
        except EvidenceError:
            raise
        except Exception as exc:  # eventual consistency/network retries are bounded
            last_error = exc
        time.sleep(2.0)
    detail = f"state={last_state!r}"
    if last_error is not None:
        detail += f", last_error={last_error}"
    raise EvidenceError(f"online W&B handoff run did not become finished within 60s ({detail})")


def required_evidence_logger(logger: EvidenceLogger | None) -> EvidenceLogger:
    """Return the injected test sink or the mandatory online implementation."""

    return logger if logger is not None else WandbEvidenceLogger()
