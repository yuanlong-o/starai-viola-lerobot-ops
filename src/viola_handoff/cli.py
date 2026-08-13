"""Command-line interface for the shared Viola handoff contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .contract import (
    DEFAULT_ACCEPT_ROOT,
    DEFAULT_HANDOFF_ROOT,
    KIND_RULES,
    SUPPORTED_KINDS,
    RuntimeIdentity,
    SealRequest,
    accept_bundle,
    ack_bundle,
    copy_paste_message,
    inspect_bundle,
    seal_bundle,
)
from .errors import HandoffError


def _read_json_object(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"JSON input must be an object: {path}")
    return value


def _artifact_pairs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise HandoffError("--artifact must use NAME=PATH")
        if name in result:
            raise HandoffError(f"duplicate artifact name: {name}")
        result[name] = Path(path)
    return result


def _bundle_summary(bundle: Any) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "content_id": bundle.content_id,
        "kind": bundle.kind,
        "path": str(bundle.path),
        "permission": bundle.permission,
        "consumer_role": bundle.manifest["consumer"]["role"],
        "manifest_sha256": bundle.ready["manifest_sha256"],
        "inventory_sha256": bundle.ready["inventory_sha256"],
        "wandb_project": bundle.manifest["wandb"]["project"],
        "wandb_run_id": bundle.manifest["wandb"]["run_id"],
        "wandb_url": bundle.ready["wandb_url"],
        "artifacts_verified": bundle.artifacts_verified,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viola-handoff",
        description="Seal, inspect, accept, and acknowledge immutable Viola handoffs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal", help="seal and atomically publish a handoff bundle")
    seal.add_argument("--root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    seal.add_argument("--kind", required=True, choices=SUPPORTED_KINDS)
    seal.add_argument("--experiment", required=True)
    seal.add_argument("--subject", required=True)
    seal.add_argument("--payload-dir", type=Path)
    seal.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    seal.add_argument("--lineage", metavar="JSON_FILE")
    seal.add_argument("--producer-role")
    seal.add_argument("--consumer-role")
    seal.add_argument("--permission")
    seal.add_argument("--wandb-project", required=True)
    seal.add_argument("--repo-root", type=Path, default=Path.cwd())
    seal.add_argument("--destination-root", type=Path, default=DEFAULT_ACCEPT_ROOT)

    inspect = subparsers.add_parser("inspect", help="verify an existing READY bundle")
    inspect.add_argument("bundle", type=Path)
    inspect.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip external artifact bytes; diagnostic use only",
    )
    inspect.add_argument("--expected-kind", choices=SUPPORTED_KINDS)
    inspect.add_argument("--required-permission")
    inspect.add_argument(
        "--allow-inactive",
        action="store_true",
        help="inspect rejected/revoked metadata for diagnostics; never consume it",
    )

    accept = subparsers.add_parser("accept", help="verify and atomically accept a bundle")
    accept.add_argument("bundle", type=Path)
    accept.add_argument("--destination-root", type=Path, default=DEFAULT_ACCEPT_ROOT)
    accept.add_argument("--repo-root", type=Path, default=Path.cwd())
    accept.add_argument("--receiver-role")
    accept.add_argument("--note", default="")

    ack = subparsers.add_parser("ack", help="append an immutable receipt")
    ack.add_argument("bundle", type=Path)
    ack.add_argument("--status", required=True, choices=("rejected", "revoked"))
    ack.add_argument("--repo-root", type=Path, default=Path.cwd())
    ack.add_argument("--actor-role")
    ack.add_argument("--note", default="")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "seal":
        if args.kind == "rollout_session":
            raise HandoffError(
                "generic sealing cannot grant rollout_session authority; "
                "use the typed Repo-B viola-bench handoff rollout-session command"
            )
        producer_role = args.producer_role or KIND_RULES[args.kind]["producer"]
        identity = RuntimeIdentity.capture(role=producer_role, repo_root=args.repo_root)
        bundle = seal_bundle(
            SealRequest(
                root=args.root,
                kind=args.kind,
                experiment=args.experiment,
                subject=args.subject,
                producer=identity,
                lineage=_read_json_object(args.lineage),
                wandb_project=args.wandb_project,
                payload_dir=args.payload_dir,
                artifact_roots=_artifact_pairs(args.artifact),
                consumer_role=args.consumer_role,
                permission=args.permission,
            )
        )
        print(copy_paste_message(bundle, destination_root=args.destination_root))
        return 0

    if args.command == "inspect":
        bundle = inspect_bundle(
            args.bundle,
            verify_artifacts=not args.metadata_only,
            expected_kind=args.expected_kind,
            required_permission=args.required_permission,
            allow_inactive=args.allow_inactive,
        )
        print(json.dumps(_bundle_summary(bundle), indent=2, sort_keys=True))
        return 0

    if args.command == "accept":
        incoming = inspect_bundle(args.bundle, verify_artifacts=True)
        receiver_role = args.receiver_role or incoming.manifest["consumer"]["role"]
        identity = RuntimeIdentity.capture(role=receiver_role, repo_root=args.repo_root)
        bundle = accept_bundle(
            incoming.path,
            args.destination_root,
            receiver=identity,
            note=args.note,
        )
        print(json.dumps(_bundle_summary(bundle), indent=2, sort_keys=True))
        return 0

    if args.command == "ack":
        bundle = inspect_bundle(args.bundle, verify_artifacts=True, allow_inactive=True)
        actor_role = args.actor_role or bundle.manifest["consumer"]["role"]
        identity = RuntimeIdentity.capture(role=actor_role, repo_root=args.repo_root)
        receipt = ack_bundle(
            bundle.path,
            status=args.status,
            actor=identity,
            note=args.note,
        )
        print(json.dumps({"bundle_id": bundle.bundle_id, "receipt": str(receipt)}, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        return _run(parser.parse_args(argv))
    except HandoffError as exc:
        parser.exit(2, f"viola-handoff: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
