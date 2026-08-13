"""Human-readable command line for Repo-A Viola operations.

Subcommands import their implementation only after argument validation.  In
particular, asking for help or running a disconnected command cannot import a
camera, serial, motor, or robot backend by accident.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from viola_handoff import HandoffError

from .errors import ValidationError, ViolaOpsError

DEFAULT_DATASET_ROOT = Path(
    "/mnt/nas02/yz/starai/datasets/bourn117/"
    "viola_cubes_right_to_left_blue_then_red_train_v1"
)
DEFAULT_HANDOFF_ROOT = Path("/mnt/nas02/yz/starai/handoffs/v1")
DEFAULT_ACCEPT_ROOT = Path("~/.local/share/viola/handoffs/v1").expanduser()
DEFAULT_MATERIAL_ROOT = Path("/mnt/nas02/yz/starai/producer-materials/v1")
DEFAULT_OUTPUT_ROOT = Path("/mnt/nas02/yz/starai/evidence/v1")
DEFAULT_EXPERIMENT = "viola-cubes-v1"
DEFAULT_WANDB_PROJECT = "starai-viola-policy-benchmark"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viola-ops",
        description=(
            "Readable Repo-A tools for the eight-policy Viola benchmark. "
            "Physical commands require either the shared session gate or the "
            "explicit Repo-A-local ACT safety gate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="area", required=True)

    dataset = commands.add_parser(
        "dataset", help="validate or release the frozen 34-episode dataset"
    )
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_commands.add_parser(
        "validate",
        help="fully validate dataset bytes, numeric data, and both videos",
        description="Fully validate dataset bytes, numeric data, and both videos.",
    )
    dataset_validate.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_validate.add_argument(
        "--numeric-only",
        action="store_true",
        help="skip video decoding for diagnostics; insufficient for a release",
    )

    dataset_release = dataset_commands.add_parser(
        "release",
        help="fully validate and seal a data-only bundle for Repo B",
        description="Fully validate and seal a data-only bundle for Repo B.",
    )
    dataset_release.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_release.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    dataset_release.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    dataset_release.add_argument(
        "--material-root",
        type=Path,
        default=DEFAULT_MATERIAL_ROOT,
        help="shared NAS producer workspace (default: %(default)s)",
    )
    dataset_release.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    dataset_release.add_argument("--repo-root", type=Path, default=Path.cwd())

    session_inputs = commands.add_parser(
        "session-inputs", help="produce reviewed planning inputs for Repo B"
    )
    session_commands = session_inputs.add_subparsers(dest="session_command", required=True)
    session_produce = session_commands.add_parser(
        "produce",
        help="validate setup/E-stop review and seal a planning-only bundle",
        description="Validate setup/E-stop review and seal a planning-only bundle.",
    )
    session_produce.add_argument("--setup", type=Path, required=True)
    session_produce.add_argument(
        "--subject",
        required=True,
        help="reviewed setup ID; must exactly match the setup document",
    )
    session_produce.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    session_produce.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    session_produce.add_argument(
        "--material-root",
        type=Path,
        default=DEFAULT_MATERIAL_ROOT,
        help=(
            "shared NAS workspace that keeps setup_record readable by PC B "
            "(default: %(default)s)"
        ),
    )
    session_produce.add_argument("--destination-root", type=Path, default=DEFAULT_ACCEPT_ROOT)
    session_produce.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    session_produce.add_argument("--repo-root", type=Path, default=Path.cwd())

    setup = commands.add_parser(
        "setup", help="perform narrowly reviewed setup evidence operations"
    )
    setup_commands = setup.add_subparsers(dest="setup_command", required=True)
    capture = setup_commands.add_parser(
        "capture-frozen-state",
        help="read and record seven positions without issuing a motor command",
        description="Read and record seven positions without issuing a motor command.",
    )
    capture.add_argument("--setup", type=Path, required=True, help="reviewed setup document")
    capture.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "new immutable evidence directory, or the existing capture directory "
            "with --upload-only"
        ),
    )
    capture.add_argument(
        "--operator", required=True, help="operator named by the reviewed setup evidence"
    )
    capture.add_argument("--repo-root", type=Path, default=Path.cwd())
    capture.add_argument("--wandb-entity", required=True)
    capture.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    capture.add_argument(
        "--upload-only",
        action="store_true",
        help=(
            "validate and publish an existing immutable capture after an upload failure; "
            "never confirm or open serial hardware"
        ),
    )

    policy = commands.add_parser(
        "policy", help="verify, shadow, or explicitly authorize one accepted policy"
    )
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    verify = policy_commands.add_parser(
        "verify",
        help="fresh-load and benchmark a candidate without connecting hardware",
        description="Fresh-load and benchmark a candidate without connecting hardware.",
    )
    verify.add_argument("--bundle", type=Path, required=True, help="accepted policy candidate")
    verify.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="evidence workspace on shared NAS (default: %(default)s)",
    )
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    verify.add_argument("--wandb-entity", required=True)
    verify.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)

    shadow = policy_commands.add_parser(
        "shadow",
        help="run replay or camera-only live-soak inference without motor commands",
        description="Run replay or camera-only live-soak inference without motor commands.",
    )
    shadow.add_argument("--bundle", type=Path, required=True, help="accepted policy candidate")
    shadow.add_argument("--mode", choices=("replay", "live-soak"), required=True)
    shadow.add_argument(
        "--verification",
        type=Path,
        required=True,
        help="eligible runtime-verification evidence for this candidate",
    )
    shadow.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "shared NAS workspace that keeps shadow_record readable by PC B "
            "(default: %(default)s)"
        ),
    )
    shadow.add_argument("--repo-root", type=Path, default=Path.cwd())
    shadow.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    shadow.add_argument("--wandb-entity", required=True)
    shadow.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    shadow.add_argument(
        "--setup", type=Path, help="reviewed setup document; required only for live-soak"
    )
    shadow.add_argument(
        "--frozen-state",
        type=Path,
        help="frozen seven-axis capture; required only for live-soak",
    )

    execute = policy_commands.add_parser(
        "execute",
        help="run one accepted live-session phase after interactive operator arming",
        description="Run one accepted live-session phase after interactive operator arming.",
    )
    execute.add_argument("--session", type=Path, required=True, help="accepted live session")
    execute.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="accepted candidate; its ID must match the session",
    )
    execute.add_argument("--phase", choices=("hold", "shakedown", "scored"), required=True)
    execute.add_argument(
        "--trial",
        required=True,
        help="exact phase/trial label included in the interactive ARM challenge",
    )
    execute.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help=(
            "shared NAS base visible to PCs A and B and outside the Git worktree; "
            "each attempt uses a new immutable nested path"
        ),
    )
    execute.add_argument("--repo-root", type=Path, default=Path.cwd())
    execute.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    execute.add_argument("--wandb-entity", required=True)
    execute.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    execute.add_argument("--prior-hold", type=Path, help="accepted hold evidence bundle")
    execute.add_argument(
        "--prior-shakedown", type=Path, help="accepted shakedown evidence bundle"
    )

    act = commands.add_parser(
        "act",
        help="run the exact previously deployed ACT checkpoint from Repo A",
    )
    act_commands = act.add_subparsers(dest="act_command", required=True)
    act_run = act_commands.add_parser(
        "run",
        help="run supervised ACT inference without any Repo-B handoff",
        description=(
            "Run the exact reviewed local step-80,000 ACT checkpoint. This path "
            "does not require Repo-B candidates, receipts, or rollout sessions; "
            "it still requires online W&B and real-TTY E-stop/ARM/START actions."
        ),
    )
    act_run.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("~/models/act_viola_val20_step080000").expanduser(),
        help="exact previously deployed ACT checkpoint (default: %(default)s)",
    )
    act_run.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="frozen 34-episode dataset used to bind the ACT model (default: %(default)s)",
    )
    act_run.add_argument(
        "--setup",
        type=Path,
        default=Path("config/local_act_setup.json"),
        help="versioned Repo-A local hardware setup (default: %(default)s)",
    )
    act_run.add_argument(
        "--evidence-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "local-act",
        help="immutable local ACT evidence base outside Git (default: %(default)s)",
    )
    act_run.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="bounded inference duration, greater than 0 and at most 60 (default: %(default)s)",
    )
    act_run.add_argument(
        "--operator",
        help="physical E-stop owner (default: current operating-system user)",
    )
    act_run.add_argument(
        "--trial",
        help="immutable attempt label (default: current UTC timestamp)",
    )
    act_run.add_argument("--repo-root", type=Path, default=Path.cwd())
    act_run.add_argument("--wandb-entity", default="yuanlongzhang94")
    act_run.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    act_recover = act_commands.add_parser(
        "recover-failure",
        help="publish retained ACT failure evidence without opening hardware",
        description=(
            "Validate and publish one immutable failed ACT attempt. This recovery "
            "does not prompt, load a policy, or open cameras, serial, or motors."
        ),
    )
    act_recover.add_argument(
        "--attempt",
        type=Path,
        required=True,
        help="failed attempt directory printed by act run",
    )
    act_recover.add_argument("--repo-root", type=Path, default=Path.cwd())
    act_recover.add_argument("--wandb-entity", default="yuanlongzhang94")
    act_recover.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)

    report = commands.add_parser("report", help="inspect Repo B's final benchmark report")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    report_inspect = report_commands.add_parser(
        "inspect",
        help="verify an accepted complete report and print a readable summary",
        description="Verify an accepted complete report and print a readable summary.",
    )
    report_inspect.add_argument("--bundle", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.area == "dataset":
        return _run_dataset(args)
    if args.area == "session-inputs":
        return _run_session_inputs(args)
    if args.area == "setup":
        return _run_setup(args)
    if args.area == "policy":
        return _run_policy(args)
    if args.area == "act":
        return _run_act(args)
    if args.area == "report":
        return _run_report(args)
    raise AssertionError(f"unhandled command area: {args.area}")


def _run_dataset(args: argparse.Namespace) -> int:
    backend = _backend("dataset")
    if args.dataset_command == "validate":
        result = backend.validate_dataset(args.root, full_decode=not args.numeric_only)
        print(_dataset_validation_text(result))
        return 0
    if args.dataset_command == "release":
        result = backend.release_dataset(
            args.root,
            experiment=args.experiment,
            handoff_root=args.handoff_root,
            material_root=args.material_root,
            wandb_project=args.wandb_project,
            repo_root=args.repo_root,
        )
        print(_release_text("Dataset release sealed", result))
        return 0
    raise AssertionError(f"unhandled dataset command: {args.dataset_command}")


def _run_session_inputs(args: argparse.Namespace) -> int:
    if args.session_command != "produce":
        raise AssertionError(f"unhandled session-inputs command: {args.session_command}")
    backend = _backend("session_inputs")
    result = backend.seal_session_inputs(
        args.setup,
        experiment=args.experiment,
        subject=args.subject,
        handoff_root=args.handoff_root,
        material_root=args.material_root,
        destination_root=args.destination_root,
        wandb_project=args.wandb_project,
        repo_root=args.repo_root,
    )
    print(_release_text("Session inputs sealed", result))
    return 0


def _run_setup(args: argparse.Namespace) -> int:
    if args.setup_command != "capture-frozen-state":
        raise AssertionError(f"unhandled setup command: {args.setup_command}")
    backend = _backend("setup")
    confirmation = None
    if not args.upload_only:
        confirmation = getattr(backend, "interactive_confirmation", None)
        if not callable(confirmation):
            raise ValidationError("setup backend has no interactive confirmation gate")
    result = backend.capture_frozen_state(
        args.setup,
        output_root=args.output_root,
        operator=args.operator,
        repo_root=args.repo_root,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        confirm=confirmation,
        upload_only=args.upload_only,
    )
    print(_backend_text("Frozen state captured", result))
    return 0


def _run_policy(args: argparse.Namespace) -> int:
    if args.policy_command == "verify":
        backend = _backend("policy_ops")
        result = backend.verify_command(
            args.bundle,
            args.output_root,
            args.repo_root,
            args.wandb_project,
            args.wandb_entity,
            handoff_root=args.handoff_root,
        )
        print(_backend_text("Policy verification complete", result))
        return 0

    if args.policy_command == "shadow":
        _validate_shadow_arguments(args)
        backend = _backend("policy_ops")
        result = backend.shadow_command(
            args.bundle,
            args.mode,
            args.verification,
            args.output_root,
            args.repo_root,
            args.handoff_root,
            args.wandb_project,
            args.wandb_entity,
            args.setup,
            args.frozen_state,
        )
        print(_backend_text("Policy shadow complete", result))
        return 0

    if args.policy_command == "execute":
        _validate_execute_arguments(args)
        backend = _backend("execute_ops")
        result = backend.execute_command(
            args.session,
            candidate=args.candidate,
            phase=args.phase,
            trial=args.trial,
            repo_root=args.repo_root,
            handoff_root=args.handoff_root,
            evidence_root=args.evidence_root,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            prior_hold_bundle=args.prior_hold,
            prior_shakedown_bundle=args.prior_shakedown,
        )
        print(_backend_text("Policy execution phase complete", result))
        return 0
    raise AssertionError(f"unhandled policy command: {args.policy_command}")


def _run_act(args: argparse.Namespace) -> int:
    backend = _backend("local_act_ops")
    if args.act_command == "run":
        result = backend.run_command(
            repo_root=args.repo_root,
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            setup_path=args.setup,
            evidence_root=args.evidence_root,
            operator=args.operator,
            trial=args.trial,
            duration_s=args.duration_seconds,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
        print(_backend_text("Local ACT inference complete", result))
        return 0
    if args.act_command == "recover-failure":
        result = backend.recover_failure_command(
            args.attempt,
            repo_root=args.repo_root,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
        print(_backend_text("Local ACT failure synchronized", result))
        return 0
    raise AssertionError(f"unhandled ACT command: {args.act_command}")


def _run_report(args: argparse.Namespace) -> int:
    if args.report_command != "inspect":
        raise AssertionError(f"unhandled report command: {args.report_command}")
    summary = _backend("report").inspect_report(args.bundle)
    renderer = getattr(summary, "render_text", None)
    if not callable(renderer):
        raise ValidationError("report inspector returned no human-readable summary")
    print(renderer())
    return 0


def _validate_shadow_arguments(args: argparse.Namespace) -> None:
    has_setup = args.setup is not None
    has_state = args.frozen_state is not None
    if args.mode == "live-soak" and not (has_setup and has_state):
        raise ValidationError("live-soak requires both --setup and --frozen-state")
    if args.mode == "replay" and (has_setup or has_state):
        raise ValidationError("replay shadow does not accept --setup or --frozen-state")


def _validate_execute_arguments(args: argparse.Namespace) -> None:
    if args.phase == "hold" and (args.prior_hold is not None or args.prior_shakedown is not None):
        raise ValidationError("hold does not accept predecessor evidence")
    if args.phase == "shakedown":
        if args.prior_hold is None:
            raise ValidationError("shakedown requires --prior-hold")
        if args.prior_shakedown is not None:
            raise ValidationError("shakedown does not accept --prior-shakedown")
    if args.phase == "scored" and (
        args.prior_hold is None or args.prior_shakedown is None
    ):
        raise ValidationError("scored execution requires --prior-hold and --prior-shakedown")


def _backend(name: str) -> Any:
    try:
        return import_module(f"{__package__}.{name}")
    except ImportError as exc:
        raise ValidationError(f"{name.replace('_', '-')} backend is unavailable: {exc}") from exc


def _dataset_validation_text(result: Any) -> str:
    full_decode = bool(getattr(result, "full_decode", False))
    lines = [
        "Dataset validation passed",
        f"  Release: {getattr(result, 'release_id', 'unknown')}",
        f"  Root: {getattr(result, 'root', 'unknown')}",
        f"  Episodes: {getattr(result, 'episodes', 'unknown')}",
        f"  Frames: {getattr(result, 'frames', 'unknown')}",
        f"  Inventory SHA-256: {getattr(result, 'inventory_sha256', 'unknown')}",
        f"  Full two-camera decode: {'yes' if full_decode else 'no (diagnostic only)'}",
    ]
    return "\n".join(lines)


def _release_text(title: str, result: Any) -> str:
    bundle = getattr(result, "bundle", None)
    if bundle is None:
        raise ValidationError("operation returned no sealed bundle")
    lines = [
        title,
        f"  Bundle: {getattr(bundle, 'bundle_id', 'unknown')}",
        f"  Path: {getattr(bundle, 'path', 'unknown')}",
        f"  Permission: {getattr(bundle, 'permission', 'unknown')}",
    ]
    ready = getattr(bundle, "ready", None)
    if isinstance(ready, Mapping) and ready.get("wandb_url"):
        lines.append(f"  W&B: {ready['wandb_url']}")
    destination = getattr(result, "receiver_destination", None)
    if destination is not None:
        lines.append(f"  Receiver destination: {destination}")
    return "\n".join(lines)


def _backend_text(title: str, result: Any) -> str:
    renderer = getattr(result, "render_text", None)
    if callable(renderer):
        rendered = renderer()
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValidationError("operation returned an empty human-readable summary")
        return rendered
    if isinstance(result, str):
        if not result.strip():
            raise ValidationError("operation returned an empty summary")
        return result
    if isinstance(result, Mapping):
        lines = [title]
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"  {_human_label(str(key))}: {value}")
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                lines.append(f"  {_human_label(str(key))}: {', '.join(map(str, value))}")
            else:
                lines.append(f"  {_human_label(str(key))}: {value}")
        return "\n".join(lines)

    lines = [title]
    for name in (
        "status",
        "policy",
        "mode",
        "phase",
        "bundle_id",
        "path",
        "output_root",
        "state_path",
        "evidence_path",
        "sync_path",
    ):
        value = getattr(result, name, None)
        if value is not None:
            lines.append(f"  {_human_label(name)}: {value}")
    if len(lines) == 1:
        raise ValidationError("operation returned no human-readable result")
    return "\n".join(lines)


def _human_label(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        return _run(parser.parse_args(argv))
    except (ViolaOpsError, HandoffError, OSError) as exc:
        parser.exit(2, f"viola-ops: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["main"]
