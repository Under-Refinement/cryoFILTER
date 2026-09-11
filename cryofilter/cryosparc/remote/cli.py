# Publication classifier UI integration.
"""Engaging-side CLI commands for CryoSPARC remote integration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID, uuid4

from cryofilter import __version__ as CRYOFILTER_VERSION
from cryofilter.cryosparc import PROTOCOL_VERSION
from cryofilter.cryosparc.protocol.manifests import sha256_file
from cryofilter.cryosparc.protocol.validation import (
    parse_output_ref,
    validate_job_uid,
    validate_project_uid,
    validate_workspace_uid,
)
from cryofilter.cryosparc.remote import predict as predict_helpers
from cryofilter.cryosparc.remote.config import (
    BridgeConfig,
    CryoSPARCIntegrationConfig,
    load_config,
)
from cryofilter.cryosparc.remote.transport import (
    LocalTransport,
    RemoteCommandError,
    RemoteTransport,
    SSHRsyncTransport,
)

DEFAULT_PUBLIC_CHECKPOINT_RELATIVE = Path("pretrained_models") / "cryoFILTER_FULL.pt"


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    cryosparc = subparsers.add_parser(
        "cryosparc",
        help="Remote CryoSPARC integration commands.",
    )
    cryosparc.set_defaults(func=_run_cryosparc)
    cryosparc.add_argument("--config", default=None, help="Path to cryosparc.toml.")
    cryosparc.add_argument(
        "--host",
        default=None,
        help="Override configured bridge host. Use 'local' to run the bridge on this machine without SSH.",
    )
    cryosparc.add_argument("--bridge-command", default=None, help="Override remote bridge command.")
    cryosparc.add_argument(
        "--remote-source-root",
        default=None,
        help="Override the remote directory where the bridge source is deployed.",
    )
    cryosparc.add_argument(
        "--remote-work-root",
        default=None,
        help="Override the remote directory used for staging and run state.",
    )
    cryosparc.add_argument(
        "--cryosparc-base-url",
        dest="api_base_url",
        default=None,
        help="Override the CryoSPARC API base URL passed to the remote bridge.",
    )
    cryosparc.add_argument(
        "--cryosparc-host",
        dest="api_host",
        default=None,
        help="Override the CryoSPARC master host passed to the remote bridge.",
    )
    cryosparc.add_argument(
        "--cryosparc-base-port",
        dest="api_base_port",
        type=int,
        default=None,
        help="Override the CryoSPARC base port passed to the remote bridge.",
    )
    cryosparc.add_argument(
        "--cryosparc-email",
        dest="api_email",
        default=None,
        help="Override the CryoSPARC username/email passed to the remote bridge.",
    )
    cryosparc.add_argument(
        "--ssh-option",
        action="append",
        default=None,
        help="Extra option passed to ssh and rsync's ssh transport. Repeat for multiple options.",
    )
    cryosparc.add_argument("--json", action="store_true", help="Write valid JSON to stdout.")
    sub = cryosparc.add_subparsers(dest="cryosparc_command", required=True)

    sub.add_parser("doctor", help="Check local machine, SSH, bridge, and CryoSPARC health.")

    test_transport = sub.add_parser("test-transport", help="Exercise SSH and rsync without CryoSPARC.")
    test_transport.add_argument("--keep-remote-temp", action="store_true")
    test_transport.add_argument("--timeout", type=float, default=60.0)

    deploy = sub.add_parser("deploy-bridge", help="Rsync the minimal bridge source bundle to the worker host.")
    deploy.add_argument("--timeout", type=float, default=120.0)

    inspect = sub.add_parser("inspect", help="Inspect CryoSPARC metadata through the remote bridge.")
    inspect.add_argument("--project", default=None, help="Project UID, for example P42.")
    inspect.add_argument("--workspace", default=None, help="Workspace UID, for example W5.")
    inspect.add_argument("--job", default=None, help="Job UID, for example J20.")

    roundtrip = sub.add_parser(
        "test-roundtrip",
        help="Create a particle External Job split without MRC transfer or inference.",
    )
    roundtrip.add_argument("--project", required=True, help="Project UID, for example P42.")
    roundtrip.add_argument("--workspace", required=True, help="Workspace UID, for example W5.")
    roundtrip.add_argument("--particles", required=True, help="Particle output ref, for example J20:particles.")
    roundtrip.add_argument("--title", default="cryoFILTER particle round-trip test")

    stage = sub.add_parser(
        "stage-test",
        help="Stage and optionally pull a small CryoSPARC micrograph batch.",
    )
    stage.add_argument("--project", required=True, help="Project UID, for example P42.")
    stage.add_argument("--workspace", required=True, help="Workspace UID, for example W5.")
    stage.add_argument("--micrographs", required=True, help="Micrograph output ref, for example J20:micrographs.")
    stage.add_argument("--particles", default=None, help="Optional particle output ref, for example J21:particles.")
    stage.add_argument("--run-id", default=None, help="Optional UUID for traceability.")
    stage.add_argument("--model-id", default="stage-test", help="Model label stored in the prepare request.")
    stage.add_argument("--threshold", type=float, default=0.6, help="Mask threshold stored in the prepare request.")
    stage.add_argument("--title", default="cryoFILTER staging test", help="External Job title prefix.")
    stage.add_argument(
        "--limit-micrographs",
        type=int,
        default=1,
        help="Stage at most this many micrographs. Default: 1 for CLI smoke tests.",
    )
    stage.add_argument(
        "--all-micrographs",
        action="store_true",
        help="Stage the full selected output instead of the smoke-test limit.",
    )
    stage.add_argument("--micrograph-path-field", default=None, help="Override the dataset field containing micrograph paths.")
    stage.add_argument("--local-run-root", default=None, help="Local directory for staged runs.")
    stage.add_argument("--timeout", type=float, default=300.0, help="Timeout for prepare and transfer steps.")
    stage.add_argument(
        "--max-transfer-gb",
        type=float,
        default=25.0,
        help="Abort before pulling if staged micrographs exceed this many GiB.",
    )
    stage.add_argument("--no-pull", action="store_true", help="Create the remote manifest but skip rsync pull.")
    stage.add_argument(
        "--no-stage-micrographs",
        action="store_true",
        help="For manifest-only source inspection, skip creating remote micrograph symlinks.",
    )
    stage.add_argument(
        "--create-external-job",
        action="store_true",
        help="Create a CryoSPARC External Job during staging. Default stage-test mode is manifest-only.",
    )

    diagnostics = sub.add_parser(
        "build-diagnostics",
        help="Build CryoSPARC-ready diagnostic PNGs from cryoFILTER output summaries.",
    )
    diagnostics.add_argument("--inference-summary", required=True, help="Path to inference_summary.json.")
    diagnostics.add_argument("--typing-summary", default=None, help="Optional contamination typing summary.json.")
    diagnostics.add_argument("--output-dir", default=None, help="Directory for diagnostic PNGs and manifest.")
    diagnostics.add_argument("--max-overlay-images", type=int, default=10, help="Maximum individual OTF overlays to attach.")
    diagnostics.add_argument("--max-bar-items", type=int, default=30, help="Maximum micrographs to show in the ranked bar chart.")

    attach_diagnostics = sub.add_parser(
        "attach-diagnostics",
        help="Push diagnostic PNGs to the bridge host and attach them to a CryoSPARC job.",
    )
    attach_diagnostics.add_argument("--project", required=True, help="Project UID, for example P42.")
    attach_diagnostics.add_argument("--job", required=True, help="Target job UID, for example J20.")
    attach_diagnostics.add_argument("--diagnostics-manifest", required=True, help="Local cryosparc_diagnostics_manifest.json.")
    attach_diagnostics.add_argument("--remote-diagnostics-dir", default=None, help="Remote directory for uploaded diagnostic PNGs.")
    attach_diagnostics.add_argument("--timeout", type=float, default=300.0)

    predict = sub.add_parser(
        "predict",
        help="Run cryoFILTER on CryoSPARC micrographs and write accepted/rejected particles back.",
    )
    predict.add_argument("--project", required=True, help="Project UID, for example P42.")
    predict.add_argument("--workspace", required=True, help="Workspace UID, for example W5.")
    predict.add_argument("--micrographs", required=True, help="Micrograph output ref, for example J20:micrographs.")
    predict.add_argument("--particles", required=True, help="Particle output ref, for example J21:particles.")
    predict.add_argument("--run-id", default=None, help="Optional UUID for traceability.")
    predict.add_argument("--model-id", default="cryoFILTER", help="Model label stored in run manifests.")
    predict.add_argument("--checkpoint", default=None, help="Optional cryoFILTER checkpoint path for local inference.")
    predict.add_argument("--threshold", type=float, default=0.6, help="Mask threshold for cryoFILTER inference.")
    predict.add_argument("--title", default="cryoFILTER prediction", help="External Job title prefix.")
    predict.add_argument("--limit-micrographs", type=int, default=None, help="Stage at most this many micrographs.")
    predict.add_argument("--micrograph-path-field", default=None, help="Override the dataset field containing micrograph paths.")
    predict.add_argument("--local-run-root", default=None, help="Local directory for CryoSPARC-backed runs.")
    predict.add_argument("--timeout", type=float, default=600.0, help="Timeout for remote prepare/finalize and transfers.")
    predict.add_argument("--inference-timeout", type=float, default=None, help="Optional timeout for local cryoFILTER inference.")
    predict.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Optional CPU thread count for local inference and typing subprocesses.",
    )
    predict.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Optional number of visible CUDA GPUs for local inference subprocesses.",
    )
    predict.add_argument(
        "--max-transfer-gb",
        type=float,
        default=25.0,
        help="Abort before pulling if staged micrographs exceed this many GiB.",
    )
    predict.add_argument(
        "--particle-exclusion-distance-angstrom",
        type=float,
        default=100.0,
        help="Reject particles at or within this distance from the final contamination mask.",
    )
    predict.add_argument("--typing-summary", default=None, help="Optional local contamination typing summary.json.")
    predict.add_argument(
        "--run-typing",
        dest="run_typing",
        action="store_true",
        default=None,
        help="Run cryoFILTER contamination typing after inference. This is the default when no --typing-summary is supplied.",
    )
    predict.add_argument(
        "--no-run-typing",
        dest="run_typing",
        action="store_false",
        default=None,
        help="Skip automatic contamination typing. A provided --typing-summary is still used for diagnostics.",
    )
    predict.add_argument(
        "--typing-normalization-method",
        default=None,
        help="Optional normalization method forwarded to cryofilter type.",
    )
    predict.add_argument(
        "--typing-min-component-area-px",
        type=int,
        default=None,
        help="Optional minimum component area forwarded to cryofilter type.",
    )
    predict.add_argument(
        "--typing-pixel-size-angstrom",
        type=float,
        default=None,
        help="Optional global pixel-size override forwarded to cryofilter type.",
    )
    predict.add_argument("--live-typing", choices=("background", "final-only"), default="background", help="Update types in the background during inference, or type only at finalization.")
    predict.add_argument("--typing-workers", type=int, default=None, help="Publication typing CPU threads; auto uses up to 4 CPUs within the CPU budget.")
    predict.add_argument("--typing-timeout", type=float, default=None, help="Optional timeout for contamination typing.")
    predict.add_argument("--max-overlay-images", type=int, default=10, help="Maximum individual OTF overlays to attach.")
    predict.add_argument("--max-bar-items", type=int, default=30, help="Maximum micrographs to show in the ranked bar chart.")
    predict.add_argument(
        "infer_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments after -- are forwarded to cryofilter infer.",
    )

    finalize_run = sub.add_parser(
        "finalize-run",
        aliases=["resume", "finalize-from-run"],
        help="Finalize an existing local CryoSPARC-backed cryoFILTER run.",
    )
    finalize_run.add_argument("--local-run-dir", required=True, help="Existing local run directory.")
    finalize_run.add_argument("--transfer-manifest", default=None, help="Local transfer_manifest.json path.")
    finalize_run.add_argument("--local-transfer-dir", default=None, help="Local staged transfer directory.")
    finalize_run.add_argument("--inference-dir", default=None, help="Directory containing inference_summary.json and final masks.")
    finalize_run.add_argument("--inference-summary", default=None, help="Path to inference_summary.json.")
    finalize_run.add_argument("--project", default=None, help="Override project UID inferred from transfer manifest.")
    finalize_run.add_argument("--workspace", default=None, help="Override workspace UID inferred from transfer manifest.")
    finalize_run.add_argument("--job", default=None, help="Override External Job UID inferred from transfer manifest.")
    finalize_run.add_argument("--remote-run-dir", default=None, help="Remote bridge run directory.")
    finalize_run.add_argument("--remote-transfer-manifest", default=None, help="Remote transfer manifest path.")
    finalize_run.add_argument("--model-id", default="cryoFILTER", help="Model label stored in result manifest.")
    finalize_run.add_argument("--checkpoint", default=None, help="Optional checkpoint path stored in result manifest.")
    finalize_run.add_argument("--threshold", type=float, default=0.6, help="Mask threshold stored in result manifest.")
    finalize_run.add_argument("--timeout", type=float, default=600.0, help="Timeout for remote uploads and finalization.")
    finalize_run.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Optional CPU thread count for contamination typing subprocesses.",
    )
    finalize_run.add_argument(
        "--particle-exclusion-distance-angstrom",
        type=float,
        default=100.0,
        help="Reject particles at or within this distance from the final contamination mask.",
    )
    finalize_run.add_argument("--typing-summary", default=None, help="Optional local contamination typing summary.json.")
    finalize_run.add_argument(
        "--run-typing",
        dest="run_typing",
        action="store_true",
        default=None,
        help="Run cryoFILTER contamination typing before finalization. This is the default when no typing summary exists.",
    )
    finalize_run.add_argument(
        "--no-run-typing",
        dest="run_typing",
        action="store_false",
        default=None,
        help="Skip automatic contamination typing. A provided or existing typing summary is still used for diagnostics.",
    )
    finalize_run.add_argument(
        "--typing-normalization-method",
        default=None,
        help="Optional normalization method forwarded to cryofilter type.",
    )
    finalize_run.add_argument(
        "--typing-min-component-area-px",
        type=int,
        default=None,
        help="Optional minimum component area forwarded to cryofilter type.",
    )
    finalize_run.add_argument(
        "--typing-pixel-size-angstrom",
        type=float,
        default=None,
        help="Optional global pixel-size override forwarded to cryofilter type.",
    )
    finalize_run.add_argument("--typing-workers", type=int, default=None, help="Publication typing CPU threads; auto uses up to 4 CPUs.")
    finalize_run.add_argument("--typing-timeout", type=float, default=None, help="Optional timeout for contamination typing.")
    finalize_run.add_argument("--max-overlay-images", type=int, default=10, help="Maximum individual OTF overlays to attach.")
    finalize_run.add_argument("--max-bar-items", type=int, default=30, help="Maximum micrographs to show in the ranked bar chart.")


def _apply_cli_overrides(
    config: CryoSPARCIntegrationConfig,
    args: argparse.Namespace,
) -> CryoSPARCIntegrationConfig:
    if (
        not args.host
        and not args.bridge_command
        and not args.remote_source_root
        and not args.remote_work_root
        and not args.ssh_option
        and not args.api_base_url
        and not args.api_host
        and args.api_base_port is None
        and not args.api_email
    ):
        return config
    from dataclasses import replace

    if args.host:
        config = replace(
            config,
            cryosparc=replace(config.cryosparc, host=str(args.host)),
        )
    if args.ssh_option:
        config = replace(
            config,
            cryosparc=replace(
                config.cryosparc,
                ssh_options=tuple(str(option) for option in args.ssh_option),
            ),
        )
    if args.bridge_command:
        config = replace(
            config,
            bridge=replace(config.bridge, command=str(args.bridge_command)),
        )
    if args.remote_source_root or args.remote_work_root:
        config = replace(
            config,
            bridge=replace(
                config.bridge,
                remote_source_root=(
                    str(args.remote_source_root)
                    if args.remote_source_root
                    else config.bridge.remote_source_root
                ),
                remote_work_root=(
                    str(args.remote_work_root)
                    if args.remote_work_root
                    else config.bridge.remote_work_root
                ),
            ),
        )
    if args.api_base_url or args.api_host or args.api_base_port is not None or args.api_email:
        base_url = str(args.api_base_url) if args.api_base_url else config.api.base_url
        host_value = str(args.api_host) if args.api_host else config.api.host
        base_port_value = (
            int(args.api_base_port)
            if args.api_base_port is not None
            else config.api.base_port
        )
        if args.api_base_url:
            host_value = None
            base_port_value = None
        config = replace(
            config,
            api=replace(
                config.api,
                base_url=base_url,
                host=host_value,
                base_port=base_port_value,
                email=str(args.api_email) if args.api_email else config.api.email,
            ),
        )
    return config


def _transport(config: CryoSPARCIntegrationConfig) -> RemoteTransport:
    if str(config.host).lower() in {"local", "localhost", "127.0.0.1", "::1"}:
        return LocalTransport()
    return SSHRsyncTransport(
        host=config.host,
        ssh_options=config.cryosparc.ssh_options,
        rsync_options=config.transfer.rsync_options,
    )


def _bridge_argv(config: CryoSPARCIntegrationConfig, *args: str) -> list[str]:
    use_local_module = (
        str(config.host).lower() in {"local", "localhost", "127.0.0.1", "::1"}
        and config.bridge.command == BridgeConfig().command
    )
    if use_local_module:
        argv = [sys.executable, "-m", "cryofilter.cryosparc.bridge.cli", *args]
    else:
        argv = [config.bridge.command, *args]
    if config.bridge.connection_file and args and not str(args[0]).startswith("--"):
        argv.extend(["--connection-file", config.bridge.connection_file])
    bridge_env = _bridge_environment(config)
    if not use_local_module and config.bridge.python != BridgeConfig().python:
        bridge_env["CRYOFILTER_BRIDGE_PYTHON"] = config.bridge.python
    if bridge_env:
        argv = ["env", *[f"{key}={value}" for key, value in bridge_env.items()], *argv]
    return argv


def _bridge_environment(config: CryoSPARCIntegrationConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    if config.api.base_url:
        env["CRYOSPARC_BASE_URL"] = config.api.base_url
        env["CRYOSPARC_HOST"] = ""
        env["CRYOSPARC_BASE_PORT"] = ""
    elif config.api.host:
        env["CRYOSPARC_BASE_URL"] = ""
        env["CRYOSPARC_HOST"] = config.api.host
        if config.api.base_port is not None:
            env["CRYOSPARC_BASE_PORT"] = str(config.api.base_port)
    if config.api.email:
        env["CRYOSPARC_EMAIL"] = config.api.email
    return env


def _load_remote_json(result_stdout: str) -> dict[str, Any]:
    text = result_stdout.strip()
    if not text:
        raise ValueError("Remote command produced no JSON on stdout")
    return json.loads(text)


def _emit(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_human(payload)


def _print_human(payload: dict[str, Any]) -> None:
    command = payload.get("command")
    if command == "doctor":
        for section in payload.get("sections", []):
            print(str(section["name"]).upper())
            for check in section.get("checks", []):
                status = "PASS" if check.get("ok") else "FAIL"
                detail = check.get("detail")
                suffix = f" - {detail}" if detail else ""
                print(f"[{status}] {check.get('name')}{suffix}")
            print()
        print(payload.get("summary", ""))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def _local_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _local_doctor_checks(config: CryoSPARCIntegrationConfig) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {
            "name": "cryoFILTER",
            "ok": True,
            "detail": f"version {CRYOFILTER_VERSION}, protocol {PROTOCOL_VERSION}",
        },
        {
            "name": "config",
            "ok": True,
            "detail": (
                str(config.config_path)
                if config.config_path is not None
                else "using defaults/env overrides"
            ),
        },
        {
            "name": "ssh",
            "ok": shutil.which("ssh") is not None,
            "detail": shutil.which("ssh") or "not found",
        },
        {
            "name": "rsync",
            "ok": shutil.which("rsync") is not None,
            "detail": shutil.which("rsync") or "not found",
        },
        {
            "name": "model",
            "ok": (Path.cwd() / "pretrained_models" / "cryoFILTER_FULL.pt").exists(),
            "detail": "pretrained_models/cryoFILTER_FULL.pt",
        },
    ]
    try:
        import torch

        checks.append(
            {
                "name": "GPU",
                "ok": bool(torch.cuda.is_available()),
                "detail": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "torch available, CUDA unavailable"
                ),
            }
        )
    except Exception as exc:
        checks.append({"name": "GPU", "ok": False, "detail": f"torch unavailable: {exc}"})
    return checks


def _run_remote_bridge_json(
    config: CryoSPARCIntegrationConfig,
    bridge_args: Sequence[str],
    *,
    timeout: float | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    transport = _transport(config)
    try:
        result = transport.run(
            _bridge_argv(config, *bridge_args, "--json"),
            timeout=timeout,
        )
        return _load_remote_json(result.stdout), None
    except RemoteCommandError as exc:
        stderr = exc.result.stderr.strip()
        stdout = exc.result.stdout.strip()
        if stdout:
            try:
                return _load_remote_json(stdout), None
            except Exception:
                pass
        detail = stderr or stdout or str(exc)
        return None, detail
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_doctor(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    local_checks = _local_doctor_checks(config)
    remote_payload, remote_error = _run_remote_bridge_json(config, ["doctor"], timeout=60.0)

    remote_checks = [
        {
            "name": "SSH -> worker",
            "ok": remote_payload is not None,
            "detail": remote_error or config.host,
        },
    ]
    if remote_payload is not None:
        remote_checks.extend(
            [
                {
                    "name": "bridge installed",
                    "ok": bool(remote_payload.get("bridge_version")),
                    "detail": remote_payload.get("bridge_version"),
                },
                {
                    "name": "protocol version",
                    "ok": remote_payload.get("protocol_version") == PROTOCOL_VERSION,
                    "detail": str(remote_payload.get("protocol_version")),
                },
                {
                    "name": "cryosparc-tools",
                    "ok": bool(remote_payload.get("cryosparc_tools_version")),
                    "detail": remote_payload.get("cryosparc_tools_version"),
                },
                {
                    "name": "CryoSPARC authentication",
                    "ok": bool(remote_payload.get("authenticated")),
                    "detail": remote_payload.get("cryosparc_server_version"),
                },
            ]
        )
    ok = all(check["ok"] for check in local_checks) and all(
        check["ok"] for check in remote_checks
    )
    payload = {
        "ok": ok,
        "command": "doctor",
        "local_git_commit": _local_git_commit(),
        "remote": remote_payload,
        "sections": [
            {"name": "engaging", "checks": local_checks},
            {"name": "worker", "checks": remote_checks},
        ],
        "summary": (
            "Remote integration ready."
            if ok
            else "Remote integration is not ready; inspect failed checks above."
        ),
    }
    _emit(payload, as_json=bool(args.json))
    return 0 if ok else 1


def _run_inspect(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    if (args.workspace or args.job) and not args.project:
        raise ValueError("--workspace and --job require --project")
    bridge_args = ["inspect"]
    if args.project:
        bridge_args.extend(["--project", validate_project_uid(args.project)])
    if args.workspace:
        bridge_args.extend(["--workspace", validate_workspace_uid(args.workspace)])
    if args.job:
        bridge_args.extend(["--job", validate_job_uid(args.job)])
    payload, error = _run_remote_bridge_json(config, bridge_args, timeout=60.0)
    if payload is None:
        payload = {"ok": False, "command": "inspect", "errors": [error]}
    else:
        payload["command"] = "inspect"
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _normalize_output_ref(value: str, *, default_output_name: str) -> str:
    text = str(value).strip()
    if ":" in text:
        return text
    if text.startswith("J") and text[1:].isdigit():
        return f"{text}:{default_output_name}"
    return text


def _infer_arg_value(tokens: Sequence[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, token in enumerate(tokens):
        text = str(token)
        if text.startswith(prefix):
            return text[len(prefix):]
        if text == option and index + 1 < len(tokens):
            return str(tokens[index + 1])
    return None


def _preflight_local_inference_inputs(args: argparse.Namespace) -> None:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
        missing_message = (
            f"cryoFILTER checkpoint was not found before staging: {checkpoint_path}. "
            "Update the Weights field or copy cryoFILTER_FULL.pt to that path."
        )
    else:
        checkpoint_path = (Path.cwd() / DEFAULT_PUBLIC_CHECKPOINT_RELATIVE).resolve()
        missing_message = (
            "No checkpoint was provided and the default public checkpoint was not found "
            f"before staging at {checkpoint_path}. Set the Weights field or copy "
            "cryoFILTER_FULL.pt into pretrained_models/."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(missing_message)

    num_cpus = getattr(args, "num_cpus", None)
    if num_cpus is not None and int(num_cpus) < 1:
        raise ValueError("--num-cpus must be at least 1 when supplied")
    num_gpus = getattr(args, "num_gpus", None)
    if num_gpus is not None and int(num_gpus) < 0:
        raise ValueError("--num-gpus must be non-negative when supplied")

    infer_args = [str(token) for token in (getattr(args, "infer_args", []) or [])]
    device = (_infer_arg_value(infer_args, "--device") or "").strip().lower()
    if device.startswith("cuda") and num_gpus == 0:
        raise RuntimeError("CUDA inference was requested, but --num-gpus is 0.")
    if device.startswith("cuda"):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "CUDA inference was requested, but PyTorch could not be imported in "
                "the Python environment running cryoFILTER."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA inference was requested, but the active Python environment cannot "
                "see CUDA. Set Device to cpu for a smoke test or use a CUDA-enabled "
                "PyTorch environment."
            )


def _first_visible_cuda_devices(count: int) -> str:
    if count <= 0:
        return ""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        if devices and len(devices) >= count:
            return ",".join(devices[:count])
    return ",".join(str(index) for index in range(count))


def _local_resource_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    num_cpus = getattr(args, "num_cpus", None)
    if num_cpus is not None:
        value = str(int(num_cpus))
        env.update(
            {
                "CRYOFILTER_NUM_CPUS": value,
                "OMP_NUM_THREADS": value,
                "MKL_NUM_THREADS": value,
                "OPENBLAS_NUM_THREADS": value,
                "NUMEXPR_NUM_THREADS": value,
            }
        )
    num_gpus = getattr(args, "num_gpus", None)
    if num_gpus is not None:
        count = int(num_gpus)
        env["CRYOFILTER_NUM_GPUS"] = str(count)
        env["CUDA_VISIBLE_DEVICES"] = _first_visible_cuda_devices(count)
    return env


def _run_roundtrip(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    project_uid = validate_project_uid(args.project)
    validate_workspace_uid(args.workspace)
    particles_ref = _normalize_output_ref(args.particles, default_output_name="particles")
    parse_output_ref(particles_ref, project_uid=project_uid)
    bridge_args = [
        "test-roundtrip",
        "--project",
        project_uid,
        "--workspace",
        args.workspace,
        "--particles",
        particles_ref,
        "--title",
        args.title,
    ]
    payload, error = _run_remote_bridge_json(config, bridge_args, timeout=None)
    if payload is None:
        payload = {"ok": False, "command": "test-roundtrip", "errors": [error]}
    else:
        payload["command"] = "test-roundtrip"
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _local_run_root(config: CryoSPARCIntegrationConfig, value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if config.local.run_root is not None:
        return Path(config.local.run_root).expanduser().resolve()
    return (Path.cwd() / "cryofilter_runs" / "cryosparc").resolve()


def _directory_size(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if not path.exists():
        return count, total
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            total += int(item.stat().st_size)
    return count, total


def _run_stage_test(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    project_uid = validate_project_uid(args.project)
    workspace_uid = validate_workspace_uid(args.workspace)
    micrographs_ref = _normalize_output_ref(args.micrographs, default_output_name="micrographs")
    particles_ref = (
        _normalize_output_ref(args.particles, default_output_name="particles")
        if args.particles
        else None
    )
    parse_output_ref(micrographs_ref, project_uid=project_uid)
    if args.particles:
        parse_output_ref(particles_ref, project_uid=project_uid)
    limit_micrographs = None if bool(getattr(args, "all_micrographs", False)) else int(args.limit_micrographs)
    if limit_micrographs is not None and limit_micrographs < 1:
        raise ValueError("--limit-micrographs must be at least 1")
    no_stage_micrographs = bool(getattr(args, "no_stage_micrographs", False))
    if no_stage_micrographs and not args.no_pull:
        raise ValueError("--no-stage-micrographs requires --no-pull")
    if float(args.max_transfer_gb) <= 0:
        raise ValueError("--max-transfer-gb must be positive")
    run_id = str(UUID(args.run_id)) if args.run_id else str(uuid4())
    local_run_dir = _local_run_root(config, args.local_run_root) / run_id
    if local_run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing local run directory: {local_run_dir}")

    bridge_args = [
        "prepare",
        "--run-id",
        run_id,
        "--project",
        project_uid,
        "--workspace",
        workspace_uid,
        "--micrographs",
        micrographs_ref,
        "--model-id",
        str(args.model_id),
        "--threshold",
        str(float(args.threshold)),
        "--remote-work-root",
        config.bridge.remote_work_root,
        "--title",
        str(args.title),
    ]
    if limit_micrographs is not None:
        bridge_args.extend(["--limit-micrographs", str(limit_micrographs)])
    if particles_ref:
        bridge_args.extend(["--particles", particles_ref])
    if args.micrograph_path_field:
        bridge_args.extend(["--micrograph-path-field", str(args.micrograph_path_field)])
    if no_stage_micrographs:
        bridge_args.append("--no-stage-micrographs")
    if not args.create_external_job:
        bridge_args.append("--no-create-external-job")

    payload, error = _run_remote_bridge_json(config, bridge_args, timeout=args.timeout)
    if payload is None:
        payload = {"ok": False, "command": "stage-test", "errors": [error]}
        _emit(payload, as_json=bool(args.json))
        return 1

    payload["command"] = "stage-test"
    if not payload.get("ok"):
        _emit(payload, as_json=bool(args.json))
        return 1

    local_manifest_file = local_run_dir / "transfer_manifest.json"
    local_transfer_dir = local_run_dir / "transfer"
    payload["local_run_dir"] = str(local_run_dir)
    payload["local_manifest_file"] = str(local_manifest_file)
    payload["local_transfer_dir"] = str(local_transfer_dir)

    transport = _transport(config)
    try:
        local_run_dir.mkdir(parents=True, exist_ok=True)
        transport.pull(str(payload["manifest_file"]), local_manifest_file, timeout=args.timeout)

        total_bytes = int(payload.get("total_micrograph_bytes") or 0)
        max_bytes = int(float(args.max_transfer_gb) * 1024**3)
        payload["max_transfer_bytes"] = max_bytes
        if args.no_pull:
            payload["transfer_skipped"] = True
            payload["pulled_file_count"] = 0
            payload["pulled_bytes"] = 0
        elif total_bytes > max_bytes:
            payload["ok"] = False
            payload.setdefault("errors", []).append(
                f"Staged micrographs require {total_bytes} bytes, above --max-transfer-gb={args.max_transfer_gb}"
            )
        else:
            local_transfer_dir.mkdir(parents=True, exist_ok=True)
            remote_transfer_dir = str(payload["transfer_dir"]).rstrip("/") + "/"
            transport.pull(
                remote_transfer_dir,
                local_transfer_dir,
                timeout=args.timeout,
                dereference=True,
            )
            file_count, pulled_bytes = _directory_size(local_transfer_dir)
            payload["transfer_skipped"] = False
            payload["pulled_file_count"] = file_count
            payload["pulled_bytes"] = pulled_bytes
    except Exception as exc:
        payload["ok"] = False
        payload.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")

    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _run_build_diagnostics(args: argparse.Namespace) -> int:
    from cryofilter.cryosparc.diagnostics import build_diagnostics_manifest

    manifest = build_diagnostics_manifest(
        inference_summary_path=args.inference_summary,
        output_dir=args.output_dir,
        typing_summary_path=args.typing_summary,
        max_overlay_images=int(args.max_overlay_images),
        max_bar_items=int(args.max_bar_items),
    )
    payload = {
        "ok": True,
        "command": "build-diagnostics",
        "manifest_file": str(
            (
                Path(args.output_dir).expanduser().resolve()
                if args.output_dir
                else Path(args.inference_summary).expanduser().resolve().parent
                / "cryosparc_diagnostics"
            )
            / "cryosparc_diagnostics_manifest.json"
        ),
        "assets": [asset.model_dump() for asset in manifest.assets],
        "summary": manifest.summary,
    }
    _emit(payload, as_json=bool(args.json))
    return 0


def _remote_diagnostics_dir(
    config: CryoSPARCIntegrationConfig,
    *,
    requested: str | None,
    run_id: object,
) -> str:
    if requested:
        return str(requested).rstrip("/")
    suffix = str(run_id) if run_id else uuid4().hex
    return f"{config.bridge.remote_work_root.rstrip('/')}/diagnostics/{suffix}"


def _upload_diagnostics_manifest(
    *,
    config: CryoSPARCIntegrationConfig,
    transport: SSHRsyncTransport,
    manifest_path: str | Path,
    remote_dir: str,
    timeout: float | None,
) -> dict[str, Any]:
    from cryofilter.cryosparc.protocol.manifests import read_json_model, write_json_model
    from cryofilter.cryosparc.protocol.models import DiagnosticsManifest

    del config
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = read_json_model(DiagnosticsManifest, manifest_path)
    remote_dir = str(remote_dir).rstrip("/")
    remote_asset_dir = f"{remote_dir}/assets"
    remote_manifest_file = f"{remote_dir}/cryosparc_diagnostics_manifest.json"

    transport.run(["mkdir", "-p", remote_asset_dir], timeout=timeout)
    patched_assets = []
    for index, asset in enumerate(manifest.assets, start=1):
        local_path = Path(asset.local_path).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Diagnostic asset does not exist locally: {local_path}")
        if local_path.suffix.lower() != ".png":
            raise ValueError(f"Diagnostic asset must be a PNG: {local_path}")
        remote_path = f"{remote_asset_dir}/{index:02d}_{local_path.name}"
        transport.push(local_path, remote_path, timeout=timeout)
        patched_assets.append(asset.model_copy(update={"remote_path": remote_path}))
    patched_manifest = manifest.model_copy(update={"assets": patched_assets})
    tmp_parent = Path.cwd() / ".codex_tmp" / "cryosparc_diagnostics"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    local_patched_manifest = tmp_parent / f"{uuid4().hex}_diagnostics_manifest.json"
    write_json_model(patched_manifest, local_patched_manifest)
    transport.push(local_patched_manifest, remote_manifest_file, timeout=timeout)
    return {
        "remote_manifest_file": remote_manifest_file,
        "assets_uploaded": len(patched_assets),
        "assets": [asset.model_dump() for asset in patched_assets],
        "summary": patched_manifest.summary,
    }


def _run_attach_diagnostics(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    project_uid = validate_project_uid(args.project)
    job_uid = validate_job_uid(args.job)
    manifest_path = Path(args.diagnostics_manifest).expanduser().resolve()
    manifest_run_id = None
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest_data, dict):
            manifest_run_id = manifest_data.get("run_id")
    except Exception:
        manifest_run_id = None
    remote_dir = _remote_diagnostics_dir(
        config,
        requested=args.remote_diagnostics_dir,
        run_id=manifest_run_id,
    )

    payload: dict[str, Any] = {
        "ok": False,
        "command": "attach-diagnostics",
        "project_uid": project_uid,
        "job_uid": job_uid,
        "local_manifest_file": str(manifest_path),
        "remote_diagnostics_dir": remote_dir,
        "remote_manifest_file": f"{remote_dir.rstrip('/')}/cryosparc_diagnostics_manifest.json",
        "assets_uploaded": 0,
        "errors": [],
    }
    transport = _transport(config)
    try:
        upload = _upload_diagnostics_manifest(
            config=config,
            transport=transport,
            manifest_path=manifest_path,
            remote_dir=remote_dir,
            timeout=args.timeout,
        )
        payload["remote_manifest_file"] = upload["remote_manifest_file"]
        payload["assets_uploaded"] = upload["assets_uploaded"]

        remote_payload, error = _run_remote_bridge_json(
            config,
            [
                "attach-diagnostics",
                "--project",
                project_uid,
                "--job",
                job_uid,
                "--diagnostics-manifest",
                str(upload["remote_manifest_file"]),
            ],
            timeout=args.timeout,
        )
        if remote_payload is None:
            payload["errors"].append(str(error))
        else:
            payload["remote"] = remote_payload
            payload["ok"] = bool(remote_payload.get("ok"))
            if not payload["ok"]:
                payload["errors"].extend(remote_payload.get("errors", []))
    except Exception as exc:
        payload["errors"].append(f"{type(exc).__name__}: {exc}")

    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _build_local_diagnostics_manifest(
    *,
    inference_summary_path: str | Path,
    output_dir: str | Path,
    typing_summary_path: str | Path | None,
    max_overlay_images: int,
    max_bar_items: int,
):
    from cryofilter.cryosparc.diagnostics import build_diagnostics_manifest

    return build_diagnostics_manifest(
        inference_summary_path=inference_summary_path,
        output_dir=output_dir,
        typing_summary_path=typing_summary_path,
        max_overlay_images=max_overlay_images,
        max_bar_items=max_bar_items,
    )


def _remote_result_dir(remote_run_dir: object) -> str:
    return f"{str(remote_run_dir).rstrip('/')}/results"


def _remote_run_path(remote_run_dir: object, *parts: str) -> str:
    base = str(remote_run_dir).rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    return f"{base}/{suffix}" if suffix else base


def _local_typing_result_files(summary_path: str | Path) -> dict[str, Path]:
    summary_file = Path(summary_path).expanduser().resolve()
    files: dict[str, Path] = {"typing_summary_json": summary_file}
    try:
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        return {key: path for key, path in files.items() if path.exists()}
    if not isinstance(payload, dict):
        return {key: path for key, path in files.items() if path.exists()}

    for key in (
        "component_type_assignments_csv",
        "image_contamination_summary_csv",
        "dataset_contamination_summary_csv",
    ):
        raw = payload.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = (summary_file.parent / path).resolve()
        if path.exists():
            files[f"typing_{key}"] = path
    return {key: path for key, path in files.items() if path.exists()}


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_local_result_path(raw: object, *, base: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_mrc_2d(path: Path):
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data)
    if data.ndim == 2:
        return np.asarray(data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] > 0:
        return np.asarray(data[0], dtype=np.float32)
    raise ValueError(f"Expected 2D MRC data or a stack with at least one plane: {path}")


def _typing_mask_lookup(typing_summary_path: Path) -> dict[str, Path]:
    summary = _load_json_object(typing_summary_path)
    typed_dir = _resolve_local_result_path(summary.get("typed_mask_dir"), base=typing_summary_path.parent)
    if typed_dir is None:
        return {}
    image_csv = _resolve_local_result_path(
        summary.get("image_contamination_summary_csv"),
        base=typing_summary_path.parent,
    )
    typed_masks: dict[str, Path] = {}
    if image_csv is not None and image_csv.exists():
        with image_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset_id = str(row.get("dataset_id") or "").strip()
                stem = str(row.get("stem") or "").strip()
                if not dataset_id or not stem:
                    continue
                path = typed_dir / f"{dataset_id}__{stem}_typed_mask.npy"
                if path.exists():
                    typed_masks[f"{dataset_id}__{stem}"] = path
                    typed_masks.setdefault(stem, path)
    if not typed_masks and typed_dir.exists():
        for path in sorted(typed_dir.glob("*_typed_mask.npy")):
            stem = path.name[: -len("_typed_mask.npy")]
            typed_masks[stem] = path
            if "__" in stem:
                typed_masks.setdefault(stem.rsplit("__", 1)[-1], path)
    return typed_masks


def _inference_summary_rows_by_stem(inference_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in inference_summary.get("inputs", []):
        if not isinstance(row, dict):
            continue
        raw_mrc = row.get("input_mrc")
        if not raw_mrc:
            continue
        rows.setdefault(Path(str(raw_mrc)).stem, row)
    return rows


def _live_inference_summary_from_workers(
    *,
    local_manifest_file: Path,
    local_transfer_dir: Path,
    local_inference_dir: Path,
) -> dict[str, Any] | None:
    worker_paths = sorted(local_inference_dir.glob(".cryofilter_worker_*_summary.json"))
    if not worker_paths:
        return None

    worker_summaries: list[dict[str, Any]] = []
    for path in worker_paths:
        try:
            payload = _load_json_object(path)
        except Exception:
            continue
        if isinstance(payload.get("inputs"), list) and payload.get("inputs"):
            worker_summaries.append(payload)
    if not worker_summaries:
        return None

    manifest = predict_helpers.load_transfer_manifest(local_manifest_file)
    micrographs = manifest.get("micrographs")
    order_by_stem: dict[str, int] = {}
    if isinstance(micrographs, list):
        for index, entry in enumerate(micrographs):
            if not isinstance(entry, dict):
                continue
            order_by_stem[predict_helpers._manifest_micrograph_path(local_transfer_dir, entry).stem] = index

    rows_by_stem: dict[str, dict[str, Any]] = {}
    overlays: list[dict[str, Any]] = []
    for summary in worker_summaries:
        overlay = summary.get("particle_overlay_rendering")
        if isinstance(overlay, dict) and overlay.get("enabled"):
            overlays.append(overlay)
        for row in summary.get("inputs", []):
            if not isinstance(row, dict) or not row.get("input_mrc"):
                continue
            rows_by_stem[Path(str(row["input_mrc"])).stem] = dict(row)
    if not rows_by_stem:
        return None

    ordered_rows = sorted(
        rows_by_stem.values(),
        key=lambda row: order_by_stem.get(Path(str(row.get("input_mrc") or "")).stem, len(order_by_stem)),
    )
    live_summary = dict(worker_summaries[0])
    live_summary["inputs"] = ordered_rows
    if order_by_stem:
        live_summary["n_images_expected"] = int(len(order_by_stem))
    if overlays:
        merged_overlay = dict(overlays[0])
        frames_by_key: dict[str, dict[str, Any]] = {}
        for overlay in overlays:
            for frame in overlay.get("frames", []):
                if not isinstance(frame, dict):
                    continue
                key = str(frame.get("output_png") or frame.get("micrograph") or "")
                if key:
                    frames_by_key[key] = dict(frame)
        ordered_frames: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in ordered_rows:
            particle_overlay = row.get("particle_overlay")
            if not isinstance(particle_overlay, dict):
                continue
            frame = {"micrograph": row.get("input_mrc"), **particle_overlay}
            key = str(frame.get("output_png") or frame.get("micrograph") or "")
            if not key or key in seen:
                continue
            ordered_frames.append({**frames_by_key.get(key, {}), **frame})
            seen.add(key)
        for key, frame in frames_by_key.items():
            if key not in seen:
                ordered_frames.append(frame)
        merged_overlay["frames"] = ordered_frames
        live_summary["particle_overlay_rendering"] = merged_overlay
    return live_summary


def _load_inference_summary_for_typed_overlays(
    *,
    local_manifest_file: Path,
    local_transfer_dir: Path,
    local_inference_dir: Path,
    inference_summary_file: Path,
) -> dict[str, Any] | None:
    file_summary: dict[str, Any] | None = None
    if inference_summary_file.exists():
        try:
            file_summary = _load_json_object(inference_summary_file)
        except Exception:
            pass
    worker_summary = _live_inference_summary_from_workers(
        local_manifest_file=local_manifest_file,
        local_transfer_dir=local_transfer_dir,
        local_inference_dir=local_inference_dir,
    )
    if worker_summary is None:
        return file_summary
    if file_summary is None:
        return worker_summary
    file_rows = file_summary.get("inputs")
    worker_rows = worker_summary.get("inputs")
    if isinstance(worker_rows, list) and (
        not isinstance(file_rows, list) or len(worker_rows) > len(file_rows)
    ):
        return worker_summary
    return file_summary


def _refresh_typed_particle_overlays(
    *,
    local_manifest_file: Path,
    local_transfer_dir: Path,
    local_inference_dir: Path,
    inference_summary_file: Path,
    typing_summary_path: Path | None,
    particle_exclusion_distance_angstrom: float,
    create_if_missing: bool = False,
    summary_output_file: Path | None = None,
) -> dict[str, Any]:
    if typing_summary_path is None or not typing_summary_path.exists():
        return {"enabled": False, "refreshed": 0, "reason": "missing typing or inference summary"}

    import numpy as np
    from cryofilter.typing_cli import _digest, _file_stamp, _read_cache, _write_json_atomic

    from cryofilter.particle_filtering import classify_particle_coordinates_by_mask
    from cryofilter.qc_render import render_particle_overlay_png

    inference_summary = _load_inference_summary_for_typed_overlays(
        local_manifest_file=local_manifest_file,
        local_transfer_dir=local_transfer_dir,
        local_inference_dir=local_inference_dir,
        inference_summary_file=inference_summary_file,
    )
    if inference_summary is None:
        return {"enabled": False, "refreshed": 0, "reason": "missing typing or inference summary"}
    overlay = inference_summary.get("particle_overlay_rendering")
    if not isinstance(overlay, dict) or not overlay.get("enabled"):
        if not create_if_missing:
            return {"enabled": False, "refreshed": 0, "reason": "particle overlays were not enabled"}
        overlay = {
            "enabled": True,
            "output_dir": str(local_inference_dir / "OTF_images"),
            "contact_sheet": False,
            "max_display_dim": 1400,
            "particle_diameter_px": 34.0,
            "mask_alpha": 0.35,
            "raw_reference_panel": True,
            "typed_mask_panel": False,
            "probability_panel": True,
            "frames": [],
        }
        inference_summary["particle_overlay_rendering"] = overlay

    typed_masks = _typing_mask_lookup(typing_summary_path)
    if not typed_masks:
        return {"enabled": True, "refreshed": 0, "missing_typed_masks": True, "reason": "no typed masks found"}

    manifest = predict_helpers.load_transfer_manifest(local_manifest_file)
    micrographs = manifest.get("micrographs")
    particles = manifest.get("particles")
    if not isinstance(micrographs, list) or not isinstance(particles, list):
        return {"enabled": True, "refreshed": 0, "missing_manifest_rows": True, "reason": "transfer manifest missing micrographs or particles"}

    particles_by_micrograph: dict[int, list[dict[str, Any]]] = {}
    for particle in particles:
        if not isinstance(particle, dict):
            continue
        particles_by_micrograph.setdefault(int(particle["micrograph_uid"]), []).append(particle)

    rows_by_stem = _inference_summary_rows_by_stem(inference_summary)
    refreshed = 0
    rendered = 0
    cache_path = local_inference_dir / ".typed_overlay_cache.json"
    cache = _read_cache(cache_path)
    manifest_stamp = _file_stamp(local_manifest_file)
    skipped: list[str] = []
    skipped_reasons: dict[str, str] = {}
    overlay_dir = _resolve_local_result_path(overlay.get("output_dir"), base=local_inference_dir)
    if overlay_dir is None:
        overlay_dir = local_inference_dir / "OTF_images"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    frames = overlay.get("frames")
    if not isinstance(frames, list):
        frames = []
        overlay["frames"] = frames
    for entry in micrographs:
        if not isinstance(entry, dict):
            continue
        local_path = predict_helpers._manifest_micrograph_path(local_transfer_dir, entry)
        stem = local_path.stem
        row = rows_by_stem.get(stem)
        typed_path = typed_masks.get(stem)
        if typed_path is None:
            for key, path in typed_masks.items():
                if key.endswith(f"__{stem}"):
                    typed_path = path
                    break
        mask_path = _resolve_local_result_path(
            row.get("output_mask_npy") if isinstance(row, dict) else None,
            base=local_inference_dir,
        ) or predict_helpers._mask_output_path(local_inference_dir, stem)
        prob_path = _resolve_local_result_path(
            row.get("output_prob_npy") if isinstance(row, dict) else None,
            base=local_inference_dir,
        )
        particle_overlay = row.get("particle_overlay") if isinstance(row, dict) else None
        output_png = None
        if isinstance(particle_overlay, dict):
            output_png = _resolve_local_result_path(particle_overlay.get("output_png"), base=local_inference_dir)
        if output_png is None:
            output_png = overlay_dir / f"{stem}_particle_overlay.png"
        if row is None:
            skipped.append(stem)
            skipped_reasons[stem] = "missing inference summary row"
            continue
        if typed_path is None or prob_path is None or output_png is None:
            skipped.append(stem)
            skipped_reasons[stem] = "missing typed mask, probability map, or output path"
            continue
        if not local_path.exists() or not mask_path.exists() or not prob_path.exists() or not typed_path.exists():
            skipped.append(stem)
            missing = []
            if not local_path.exists():
                missing.append("micrograph")
            if not mask_path.exists():
                missing.append("mask")
            if not prob_path.exists():
                missing.append("probability")
            if not typed_path.exists():
                missing.append("typed mask")
            skipped_reasons[stem] = "missing " + ", ".join(missing)
            continue

        signature = _digest([
            [_file_stamp(path) for path in (local_path, mask_path, prob_path, typed_path)],
            manifest_stamp, particle_exclusion_distance_angstrom,
            {key: overlay.get(key) for key in ("max_display_dim", "particle_diameter_px", "mask_alpha")},
        ])
        cached = cache.get(stem, {})
        if (cached.get("signature") == signature and output_png.exists()
                and cached.get("output_stamp") == _file_stamp(output_png)):
            frame_record = cached["frame"]
        else:
            height, width = predict_helpers._shape_yx(entry, local_path)
            pixel_size = predict_helpers._pixel_size_angstrom(entry, local_path)
            micrograph_particles = particles_by_micrograph.get(int(entry["uid"]), [])
            coords = np.asarray(
                [
                    (
                        float(particle["center_x_frac"]) * float(width),
                        float(particle["center_y_frac"]) * float(height),
                    )
                    for particle in micrograph_particles
                    if isinstance(particle, dict)
                ],
                dtype=np.float32,
            ).reshape(-1, 2)
            mask = np.asarray(np.load(mask_path), dtype=bool)
            keep, _distances = classify_particle_coordinates_by_mask(
                coords,
                bad_mask=mask,
                input_shape=(int(height), int(width)),
                pixel_size_angstrom=float(pixel_size),
                exclusion_distance_angstrom=float(particle_exclusion_distance_angstrom),
            )
            image = _load_mrc_2d(local_path)
            render_particle_overlay_png(
                image=image,
                mask=mask,
                coords_xy=coords,
                keep=keep,
                output_path=output_png,
                probability_map=np.load(prob_path),
                typed_mask=np.load(typed_path),
                label=(
                    f"{local_path.name}   kept {int(np.count_nonzero(keep))} / "
                    f"rejected {int(len(keep) - np.count_nonzero(keep))}"
                ),
                max_display_dim=int(overlay.get("max_display_dim") or 1400),
                particle_diameter_px=float(overlay.get("particle_diameter_px") or 34.0),
                mask_alpha=float(overlay.get("mask_alpha") or 0.35),
                include_raw_panel=True,
                include_typed_mask_panel=True,
                include_probability_panel=True,
            )
            n_kept = int(np.count_nonzero(keep))
            n_removed = int(len(keep) - n_kept)
            frame_record = {
                "micrograph": str(local_path),
                "output_png": str(output_png),
                "particles": int(len(keep)),
                "particles_kept": n_kept,
                "particles_removed": n_removed,
                "typed_mask_npy": str(typed_path),
                "typed_mask_panel": True,
            }
            rendered += 1
            cache[stem] = {"signature": signature, "output_stamp": _file_stamp(output_png), "frame": frame_record}
        if isinstance(particle_overlay, dict):
            particle_overlay.update(frame_record)
        else:
            row["particle_overlay"] = frame_record
        replaced = False
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            if str(frame.get("micrograph") or "") == str(local_path) or str(frame.get("output_png") or "") == str(output_png):
                frames[index] = {**frame, **frame_record}
                replaced = True
                break
        if not replaced:
            frames.append(frame_record)
        refreshed += 1

    overlay["typed_mask_panel"] = refreshed > 0
    overlay["raw_reference_panel"] = refreshed > 0
    overlay["probability_panel"] = refreshed > 0
    overlay["panel_layout"] = [
        "raw_micrograph",
        "mask_and_particles",
        "typed_mask",
        "probability_map",
    ]
    overlay["typed_mask_refresh"] = {
        "refreshed": int(refreshed),
        "skipped": int(len(skipped)),
        "skipped_stems": skipped[:10],
        "skipped_reasons": {stem: skipped_reasons.get(stem, "skipped") for stem in skipped[:10]},
        "typing_summary": str(typing_summary_path),
    }
    _write_json_atomic(cache_path, cache)
    _write_json_atomic(summary_output_file or inference_summary_file, inference_summary)
    return {"enabled": True, "refreshed": int(refreshed), "rendered": rendered, "skipped": int(len(skipped))}


def _require_typed_particle_overlay_refresh(refresh: dict[str, Any], *, run_typing: bool) -> None:
    if not run_typing or not bool(refresh.get("enabled")):
        return
    if int(refresh.get("refreshed") or 0) > 0:
        return
    reason = str(refresh.get("reason") or "no typed particle overlays were refreshed")
    skipped = refresh.get("skipped_reasons")
    if isinstance(skipped, dict) and skipped:
        examples = "; ".join(f"{stem}: {why}" for stem, why in list(skipped.items())[:3])
        reason = f"{reason}; {examples}"
    raise RuntimeError(
        "Typing completed, but cryoFILTER could not create the 4-panel typed OTF images. "
        f"{reason}"
    )


def _finalize_prediction_remote(
    *,
    config: CryoSPARCIntegrationConfig,
    project_uid: str,
    workspace_uid: str,
    external_job_uid: str,
    remote_transfer_manifest: str,
    remote_result_manifest: str,
    timeout: float | None,
) -> tuple[dict[str, Any] | None, str | None]:
    return _run_remote_bridge_json(
        config,
        [
            "finalize-prediction",
            "--project",
            project_uid,
            "--workspace",
            workspace_uid,
            "--job",
            external_job_uid,
            "--transfer-manifest",
            remote_transfer_manifest,
            "--result-manifest",
            remote_result_manifest,
        ],
        timeout=timeout,
    )


def _resolve_auto_typing(args: argparse.Namespace, *, typing_summary_path: Path | None) -> bool:
    raw = getattr(args, "run_typing", None)
    if raw is None:
        return typing_summary_path is None
    return bool(raw)


def _render_initial_particle_overlays(*, run_typing: bool, typing_summary_path: Path | None) -> bool:
    return True


def _validate_typing_options(args: argparse.Namespace) -> None:
    if getattr(args, "typing_workers", None) is not None and int(args.typing_workers) < 1:
        raise ValueError("--typing-workers must be at least 1")
    if getattr(args, "run_typing", None) is True and getattr(args, "typing_summary", None):
        raise ValueError("--run-typing and --typing-summary cannot be combined")
    if (
        getattr(args, "typing_min_component_area_px", None) is not None
        and int(args.typing_min_component_area_px) < 1
    ):
        raise ValueError("--typing-min-component-area-px must be at least 1")
    if (
        getattr(args, "typing_pixel_size_angstrom", None) is not None
        and float(args.typing_pixel_size_angstrom) <= 0
    ):
        raise ValueError("--typing-pixel-size-angstrom must be positive")


def _local_cpu_budget(resource_env: dict[str, str] | None) -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    env = {**os.environ, **(resource_env or {})}
    for key in ("CRYOFILTER_NUM_CPUS", "SLURM_CPUS_PER_TASK"):
        try:
            available = min(available, max(1, int(env[key])))
        except (KeyError, ValueError):
            pass
    return max(1, available)


def _typing_workers(resource_env: dict[str, str] | None, requested: int | None, *, live: bool = False) -> int:
    available = _local_cpu_budget(resource_env)
    budget = max(1, available // 2) if live else available
    return max(1, min(budget, int(requested) if requested is not None else 4))


class _LiveTypingUpdater:
    def __init__(
        self,
        *,
        local_manifest_file: Path,
        local_transfer_dir: Path,
        local_inference_dir: Path,
        inference_summary_file: Path,
        local_run_dir: Path,
        local_typing_dir: Path,
        project_uid: str,
        workspace_uid: str,
        particle_exclusion_distance_angstrom: float,
        typing_normalization_method: str | None,
        typing_min_component_area_px: int | None,
        typing_pixel_size_angstrom: float | None,
        typing_timeout: float | None,
        resource_env: dict[str, str] | None,
        typing_workers: int = 1,
    ) -> None:
        manifest = predict_helpers.load_transfer_manifest(local_manifest_file)
        micrographs = manifest.get("micrographs")
        self.expected_images = len(micrographs) if isinstance(micrographs, list) else 0
        self.local_manifest_file = local_manifest_file
        self.local_transfer_dir = local_transfer_dir
        self.local_inference_dir = local_inference_dir
        self.inference_summary_file = inference_summary_file
        self.typing_manifest_file = local_run_dir / "contamination_typing_manifest.csv"
        self.local_typing_dir = local_typing_dir
        self.dataset_id = f"{project_uid}_{workspace_uid}"
        self.particle_exclusion_distance_angstrom = float(particle_exclusion_distance_angstrom)
        self.typing_normalization_method = typing_normalization_method
        self.typing_min_component_area_px = typing_min_component_area_px
        self.typing_pixel_size_angstrom = typing_pixel_size_angstrom
        self.typing_timeout = typing_timeout
        self.resource_env = resource_env
        self.typing_workers = typing_workers
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self.last_images = 0
        self.last_refreshed_images = 0
        self.last_attempt = 0.0
        try:
            self.min_interval_seconds = max(
                5.0,
                float(os.environ.get("CRYOFILTER_LIVE_TYPING_INTERVAL_SECONDS", "20")),
            )
        except ValueError:
            self.min_interval_seconds = 20.0

    def _refresh_available_otfs(self) -> int:
        typing_summary_path = self.local_typing_dir / "summary.json"
        refresh = _refresh_typed_particle_overlays(
            local_manifest_file=self.local_manifest_file,
            local_transfer_dir=self.local_transfer_dir,
            local_inference_dir=self.local_inference_dir,
            inference_summary_file=self.inference_summary_file,
            typing_summary_path=typing_summary_path,
            particle_exclusion_distance_angstrom=self.particle_exclusion_distance_angstrom,
            create_if_missing=True,
            summary_output_file=self.local_inference_dir / ".live_typed_overlay_summary.json",
        )
        refreshed = int(refresh.get("refreshed") or 0)
        if refreshed > self.last_refreshed_images:
            self.last_refreshed_images = refreshed
            print(f"Live typed OTF refresh: {refreshed} four-panel image(s) available.", flush=True)
        return refreshed

    def __call__(self) -> None:
        # The inference supervisor only schedules work; scans, typing and PNG I/O
        # belong to this single background owner of the typing output directory.
        if self._cancel.is_set() or (self._thread is not None and self._thread.is_alive()):
            return
        now = time.monotonic()
        if self.last_attempt and now - self.last_attempt < self.min_interval_seconds:
            return
        self.last_attempt = now
        self._finished.clear()
        self._thread = threading.Thread(target=self._update_safely, name="cryofilter-live-typing", daemon=True)
        self._thread.start()

    def close(self, *, cancel: bool = False) -> None:
        if cancel:
            self._cancel.set()
        if self._thread is not None:
            try:
                self._finished.wait()
            except BaseException:
                # Cancellation can arrive after inference, while waiting for
                # live typing to release the output directory for finalization.
                self._cancel.set()
                self._finished.wait()
                raise
            self._thread.join()

    def _update_safely(self) -> None:
        try:
            self._update()
        except Exception as exc:
            print(f"Warning: background typing update failed; finalization will retry: {type(exc).__name__}: {exc}", flush=True)
        finally:
            self._finished.set()

    def _update(self) -> None:
        now = time.monotonic()
        try:
            typing_manifest = predict_helpers.write_typing_manifest_from_transfer(
                transfer_manifest_file=self.local_manifest_file,
                local_transfer_dir=self.local_transfer_dir,
                inference_dir=self.local_inference_dir,
                manifest_path=self.typing_manifest_file,
                dataset_id=self.dataset_id,
                require_all_masks=False,
            )
        except ValueError as exc:
            if "No rows were written" in str(exc):
                return
            raise
        images = int(typing_manifest.get("images") or 0)
        if images <= 0:
            return
        if images <= self.last_images:
            if self.last_refreshed_images < images:
                self.last_attempt = now
                self._refresh_available_otfs()
            return
        self.last_attempt = now
        print(
            f"Live typing update: {images}/{self.expected_images or images} completed mask(s) ready.",
            flush=True,
        )
        predict_helpers.run_local_typing(
            manifest_path=self.typing_manifest_file,
            output_dir=self.local_typing_dir,
            normalization_method=self.typing_normalization_method,
            min_component_area_px=self.typing_min_component_area_px,
            pixel_size_angstrom=self.typing_pixel_size_angstrom,
            expected_images=self.expected_images or images,
            timeout=self.typing_timeout,
            env_overrides=self.resource_env,
            poll_callback=self._refresh_available_otfs,
            poll_interval=5.0,
            workers=self.typing_workers,
            incremental=True,
            cancel_event=self._cancel,
        )
        self._refresh_available_otfs()
        self.last_images = images


def _resolve_existing_inference_paths(
    *,
    local_run_dir: Path,
    inference_dir: str | Path | None,
    inference_summary: str | Path | None,
) -> tuple[Path, Path]:
    if inference_summary:
        summary_path = Path(inference_summary).expanduser().resolve()
        resolved_dir = (
            Path(inference_dir).expanduser().resolve()
            if inference_dir
            else summary_path.parent
        )
        return resolved_dir, summary_path

    if inference_dir:
        resolved_dir = Path(inference_dir).expanduser().resolve()
        return resolved_dir, resolved_dir / "inference_summary.json"

    preferred = [
        local_run_dir / "inference" / "inference_summary.json",
        local_run_dir / "inference_gpu" / "inference_summary.json",
        local_run_dir / "inference_summary.json",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate.parent, candidate

    candidates = sorted(local_run_dir.glob("*/inference_summary.json"))
    if len(candidates) == 1:
        return candidates[0].parent, candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not infer inference output. Supply --inference-dir or --inference-summary."
        )
    examples = ", ".join(str(path) for path in candidates[:5])
    raise ValueError(
        "Multiple inference summaries found; supply --inference-summary. "
        f"Candidates: {examples}"
    )


def _finalize_local_run_outputs(
    *,
    config: CryoSPARCIntegrationConfig,
    transport: SSHRsyncTransport,
    local_run_dir: Path,
    local_manifest_file: Path,
    local_transfer_dir: Path,
    local_inference_dir: Path,
    inference_summary_file: Path,
    project_uid: str,
    workspace_uid: str,
    external_job_uid: str,
    remote_transfer_manifest: str,
    remote_run_dir: str,
    model: dict[str, Any],
    particle_exclusion_distance_angstrom: float,
    typing_summary_path: Path | None,
    run_typing: bool,
    typing_normalization_method: str | None,
    typing_min_component_area_px: int | None,
    typing_pixel_size_angstrom: float | None,
    typing_timeout: float | None,
    max_overlay_images: int,
    max_bar_items: int,
    resource_env: dict[str, str] | None,
    timeout: float | None,
    typing_workers: int | None = None,
) -> dict[str, Any]:
    manifest = predict_helpers.load_transfer_manifest(local_manifest_file)
    run_id = str(UUID(str(manifest.get("run_id"))))
    local_results_dir = local_run_dir / "results"
    local_diagnostics_dir = local_run_dir / "diagnostics"
    local_typing_dir = local_run_dir / "typing"
    local_results_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"errors": []}
    typing_manifest_file: Path | None = None
    if run_typing:
        typing_manifest_file = local_run_dir / "contamination_typing_manifest.csv"
        typing_manifest = predict_helpers.write_typing_manifest_from_transfer(
            transfer_manifest_file=local_manifest_file,
            local_transfer_dir=local_transfer_dir,
            inference_dir=local_inference_dir,
            manifest_path=typing_manifest_file,
            dataset_id=f"{project_uid}_{workspace_uid}",
        )
        print("Phase: final typing/finalization (GPU inference complete).", flush=True)
        typing_command = predict_helpers.run_local_typing(
            manifest_path=typing_manifest_file,
            output_dir=local_typing_dir,
            normalization_method=typing_normalization_method,
            min_component_area_px=typing_min_component_area_px,
            pixel_size_angstrom=typing_pixel_size_angstrom,
            expected_images=int(typing_manifest.get("images_expected") or typing_manifest.get("images") or 0),
            timeout=typing_timeout,
            env_overrides=resource_env,
            workers=_typing_workers(resource_env, typing_workers),
            incremental=True,
        )
        typing_summary_path = local_typing_dir / "summary.json"
        if not typing_summary_path.exists():
            raise FileNotFoundError(f"Typing summary was not written: {typing_summary_path}")
        payload["typing"] = {
            "source": "run",
            "manifest": typing_manifest,
            "command": typing_command,
            "summary_file": str(typing_summary_path),
        }
    elif typing_summary_path is not None:
        payload["typing"] = {
            "source": "provided",
            "summary_file": str(typing_summary_path),
        }

    if typing_summary_path is not None:
        payload["typed_otf_overlays"] = _refresh_typed_particle_overlays(
            local_manifest_file=local_manifest_file,
            local_transfer_dir=local_transfer_dir,
            local_inference_dir=local_inference_dir,
            inference_summary_file=inference_summary_file,
            typing_summary_path=typing_summary_path,
            particle_exclusion_distance_angstrom=float(particle_exclusion_distance_angstrom),
            create_if_missing=True,
        )
        _require_typed_particle_overlay_refresh(
            payload["typed_otf_overlays"],
            run_typing=run_typing,
        )

    split = predict_helpers.classify_particles_from_manifest(
        transfer_manifest_file=local_manifest_file,
        local_transfer_dir=local_transfer_dir,
        inference_dir=local_inference_dir,
        exclusion_distance_angstrom=float(particle_exclusion_distance_angstrom),
    )
    accepted_uid_file = predict_helpers.write_uid_file(
        path=local_results_dir / "accepted_uids.json",
        run_id=run_id,
        kind="accepted",
        uids=split["accepted_uids"],
    )
    rejected_uid_file = predict_helpers.write_uid_file(
        path=local_results_dir / "rejected_uids.json",
        run_id=run_id,
        kind="rejected",
        uids=split["rejected_uids"],
    )
    split_summary_file = local_results_dir / "particle_uid_split_summary.json"
    split_summary_file.write_text(json.dumps(split, indent=2), encoding="utf-8")

    diagnostics_manifest = _build_local_diagnostics_manifest(
        inference_summary_path=inference_summary_file,
        output_dir=local_diagnostics_dir,
        typing_summary_path=typing_summary_path,
        max_overlay_images=int(max_overlay_images),
        max_bar_items=int(max_bar_items),
    )
    diagnostics_manifest_file = local_diagnostics_dir / "cryosparc_diagnostics_manifest.json"

    print("Phase: upload/register results to CryoSPARC.", flush=True)
    remote_result_dir = _remote_result_dir(remote_run_dir)
    remote_accepted_uid_file = f"{remote_result_dir}/accepted_uids.json"
    remote_rejected_uid_file = f"{remote_result_dir}/rejected_uids.json"
    remote_split_summary_file = f"{remote_result_dir}/particle_uid_split_summary.json"
    remote_inference_summary_file = f"{remote_result_dir}/inference_summary.json"
    remote_result_manifest = f"{remote_result_dir}/result_manifest.json"
    transport.run(["mkdir", "-p", remote_result_dir], timeout=timeout)
    transport.push(accepted_uid_file, remote_accepted_uid_file, timeout=timeout)
    transport.push(rejected_uid_file, remote_rejected_uid_file, timeout=timeout)
    transport.push(split_summary_file, remote_split_summary_file, timeout=timeout)
    transport.push(inference_summary_file, remote_inference_summary_file, timeout=timeout)

    remote_typing_files: dict[str, str] = {}
    if typing_summary_path is not None:
        remote_typing_dir = f"{remote_result_dir}/typing"
        local_typing_files = _local_typing_result_files(typing_summary_path)
        if typing_manifest_file is None:
            candidate = local_run_dir / "contamination_typing_manifest.csv"
            if candidate.exists():
                typing_manifest_file = candidate
        if typing_manifest_file is not None and typing_manifest_file.exists():
            local_typing_files["typing_manifest_csv"] = typing_manifest_file
        if local_typing_files:
            transport.run(["mkdir", "-p", remote_typing_dir], timeout=timeout)
            for key, path in local_typing_files.items():
                remote_path = f"{remote_typing_dir}/{path.name}"
                transport.push(path, remote_path, timeout=timeout)
                remote_typing_files[key] = remote_path
            payload["typing_upload"] = {
                "remote_dir": remote_typing_dir,
                "files": remote_typing_files,
            }

    diagnostic_upload = _upload_diagnostics_manifest(
        config=config,
        transport=transport,
        manifest_path=diagnostics_manifest_file,
        remote_dir=_remote_run_path(remote_run_dir, "diagnostics"),
        timeout=timeout,
    )
    diagnostics_assets = diagnostic_upload.get("assets") or [
        asset.model_dump() for asset in diagnostics_manifest.assets
    ]

    local_result_manifest = local_results_dir / "result_manifest.json"
    result_files = {
        "accepted_uids_json": remote_accepted_uid_file,
        "rejected_uids_json": remote_rejected_uid_file,
        "particle_uid_split_summary_json": remote_split_summary_file,
        "inference_summary_json": remote_inference_summary_file,
        "diagnostics_manifest_json": str(diagnostic_upload["remote_manifest_file"]),
    }
    result_files.update(remote_typing_files)
    model_payload = dict(model)
    model_payload["typing"] = typing_summary_path is not None
    predict_helpers.write_result_manifest_payload(
        path=local_result_manifest,
        run_id=run_id,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        external_job_uid=external_job_uid,
        transfer_manifest_file=remote_transfer_manifest,
        status="completed",
        error=None,
        model=model_payload,
        micrographs_processed=int(len(manifest.get("micrographs") or [])),
        particles_processed=int(split["particles_processed"]),
        accepted_particles=int(split["accepted_particles"]),
        rejected_particles=int(split["rejected_particles"]),
        files=result_files,
        diagnostics=diagnostics_assets,
    )
    transport.push(local_result_manifest, remote_result_manifest, timeout=timeout)

    remote_payload, finalize_error = _finalize_prediction_remote(
        config=config,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        external_job_uid=external_job_uid,
        remote_transfer_manifest=remote_transfer_manifest,
        remote_result_manifest=remote_result_manifest,
        timeout=timeout,
    )
    payload["remote_result_manifest"] = remote_result_manifest
    payload["diagnostics_upload"] = diagnostic_upload
    payload["split"] = {
        key: value
        for key, value in split.items()
        if key not in {"accepted_uids", "rejected_uids"}
    }
    if remote_payload is None:
        payload["ok"] = False
        payload["errors"].append(str(finalize_error))
    else:
        payload["finalize"] = remote_payload
        payload["ok"] = bool(remote_payload.get("ok"))
        if not payload["ok"]:
            payload["errors"].extend(remote_payload.get("errors", []))
    return payload


def _write_and_push_failure_result(
    *,
    config: CryoSPARCIntegrationConfig,
    transport: SSHRsyncTransport,
    local_run_dir: Path,
    prepare_payload: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
    error: str,
    timeout: float | None,
) -> dict[str, Any] | None:
    run_id = str(prepare_payload.get("run_id") or uuid4())
    external_job_uid = str(prepare_payload.get("external_job_uid") or "")
    remote_transfer_manifest = str(prepare_payload.get("manifest_file") or "")
    remote_run_dir = str(prepare_payload.get("remote_run_dir") or "")
    if not external_job_uid or not remote_transfer_manifest or not remote_run_dir:
        return None

    local_result_dir = local_run_dir / "results"
    local_result_manifest = local_result_dir / "result_manifest.failed.json"
    remote_dir = _remote_result_dir(remote_run_dir)
    remote_result_manifest = f"{remote_dir}/result_manifest.failed.json"
    predict_helpers.write_result_manifest_payload(
        path=local_result_manifest,
        run_id=run_id,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        external_job_uid=external_job_uid,
        transfer_manifest_file=remote_transfer_manifest,
        status="failed",
        error=error,
        model={"cryoFILTER_version": CRYOFILTER_VERSION},
        micrographs_processed=0,
        particles_processed=0,
        accepted_particles=0,
        rejected_particles=0,
        files={},
        diagnostics=[],
    )
    try:
        transport.run(["mkdir", "-p", remote_dir], timeout=timeout)
        transport.push(local_result_manifest, remote_result_manifest, timeout=timeout)
        remote_payload, remote_error = _finalize_prediction_remote(
            config=config,
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            external_job_uid=external_job_uid,
            remote_transfer_manifest=remote_transfer_manifest,
            remote_result_manifest=remote_result_manifest,
            timeout=timeout,
        )
    except Exception as exc:
        return {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    if remote_payload is None:
        return {"ok": False, "errors": [str(remote_error)]}
    return remote_payload


def _run_predict(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    project_uid = validate_project_uid(args.project)
    workspace_uid = validate_workspace_uid(args.workspace)
    micrographs_ref = _normalize_output_ref(args.micrographs, default_output_name="micrographs")
    particles_ref = _normalize_output_ref(args.particles, default_output_name="particles")
    parse_output_ref(micrographs_ref, project_uid=project_uid)
    parse_output_ref(particles_ref, project_uid=project_uid)
    if args.limit_micrographs is not None and int(args.limit_micrographs) < 1:
        raise ValueError("--limit-micrographs must be at least 1 when supplied")
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if float(args.max_transfer_gb) <= 0:
        raise ValueError("--max-transfer-gb must be positive")
    if float(args.particle_exclusion_distance_angstrom) < 0:
        raise ValueError("--particle-exclusion-distance-angstrom must be >= 0")
    _validate_typing_options(args)
    typing_summary_path: Path | None = (
        Path(args.typing_summary).expanduser().resolve()
        if getattr(args, "typing_summary", None)
        else None
    )
    run_typing = _resolve_auto_typing(args, typing_summary_path=typing_summary_path)

    run_id = str(UUID(args.run_id)) if args.run_id else str(uuid4())
    local_run_dir = _local_run_root(config, args.local_run_root) / run_id
    if local_run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing local run directory: {local_run_dir}")

    payload: dict[str, Any] = {
        "ok": False,
        "command": "predict",
        "run_id": run_id,
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "local_run_dir": str(local_run_dir),
        "resources": {
            "num_cpus": getattr(args, "num_cpus", None),
            "num_gpus": getattr(args, "num_gpus", None),
        },
        "errors": [],
    }

    try:
        _preflight_local_inference_inputs(args)
    except Exception as exc:
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        _emit(payload, as_json=bool(args.json))
        return 1
    resource_env = _local_resource_env(args)

    transport = _transport(config)
    prepare_payload: dict[str, Any] | None = None

    try:
        bridge_args = [
            "prepare",
            "--run-id",
            run_id,
            "--project",
            project_uid,
            "--workspace",
            workspace_uid,
            "--micrographs",
            micrographs_ref,
            "--particles",
            particles_ref,
            "--model-id",
            str(args.model_id),
            "--threshold",
            str(float(args.threshold)),
            "--remote-work-root",
            config.bridge.remote_work_root,
            "--title",
            str(args.title),
        ]
        if args.limit_micrographs is not None:
            bridge_args.extend(["--limit-micrographs", str(int(args.limit_micrographs))])
        if args.micrograph_path_field:
            bridge_args.extend(["--micrograph-path-field", str(args.micrograph_path_field)])

        prepare_payload, error = _run_remote_bridge_json(config, bridge_args, timeout=args.timeout)
        if prepare_payload is None:
            payload["errors"].append(str(error))
            _emit(payload, as_json=bool(args.json))
            return 1
        payload["prepare"] = prepare_payload
        if not prepare_payload.get("ok"):
            payload["errors"].extend(prepare_payload.get("errors", []))
            _emit(payload, as_json=bool(args.json))
            return 1
        if not prepare_payload.get("external_job_uid"):
            raise ValueError("Remote prepare did not create an External Job")

        local_manifest_file = local_run_dir / "transfer_manifest.json"
        local_transfer_dir = local_run_dir / "transfer"
        local_inference_dir = local_run_dir / "inference"
        local_results_dir = local_run_dir / "results"
        local_diagnostics_dir = local_run_dir / "diagnostics"
        local_typing_dir = local_run_dir / "typing"
        payload.update(
            {
                "external_job_uid": prepare_payload["external_job_uid"],
                "local_manifest_file": str(local_manifest_file),
                "local_transfer_dir": str(local_transfer_dir),
                "local_inference_dir": str(local_inference_dir),
                "local_diagnostics_dir": str(local_diagnostics_dir),
                "local_typing_dir": str(local_typing_dir),
            }
        )

        local_run_dir.mkdir(parents=True, exist_ok=True)
        transport.pull(str(prepare_payload["manifest_file"]), local_manifest_file, timeout=args.timeout)

        total_bytes = int(prepare_payload.get("total_micrograph_bytes") or 0)
        max_bytes = int(float(args.max_transfer_gb) * 1024**3)
        payload["max_transfer_bytes"] = max_bytes
        if total_bytes > max_bytes:
            raise ValueError(
                f"Staged micrographs require {total_bytes} bytes, above "
                f"--max-transfer-gb={args.max_transfer_gb}"
            )

        local_transfer_dir.mkdir(parents=True, exist_ok=True)
        remote_transfer_dir = str(prepare_payload["transfer_dir"]).rstrip("/") + "/"
        transport.pull(
            remote_transfer_dir,
            local_transfer_dir,
            timeout=args.timeout,
            dereference=True,
        )
        file_count, pulled_bytes = _directory_size(local_transfer_dir)
        payload["pulled_file_count"] = file_count
        payload["pulled_bytes"] = pulled_bytes

        particle_star = local_run_dir / "particles_from_cryosparc.star"
        star_info = predict_helpers.write_particle_star_from_manifest(
            transfer_manifest_file=local_manifest_file,
            local_transfer_dir=local_transfer_dir,
            star_path=particle_star,
        )
        payload["particle_star"] = star_info

        infer_args = list(getattr(args, "infer_args", []) or [])
        live_typing_callback = None
        live_enabled = run_typing and typing_summary_path is None and getattr(args, "live_typing", "background") == "background"
        cpu_budget = _local_cpu_budget(resource_env)
        if cpu_budget == 1:
            live_enabled = False
        inference_env = resource_env.copy()
        inference_cpus = getattr(args, "num_cpus", None)
        if live_enabled:
            live_workers = _typing_workers(resource_env, getattr(args, "typing_workers", None), live=True)
            if inference_cpus is not None or os.environ.get("CRYOFILTER_NUM_CPUS") or os.environ.get("SLURM_CPUS_PER_TASK"):
                inference_cpus = max(1, cpu_budget - live_workers)
                for key in ("CRYOFILTER_NUM_CPUS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    inference_env[key] = str(inference_cpus)
            print(f"CPU allocation: inference {inference_cpus or 'auto'}, background typing {live_workers} worker(s).", flush=True)
            live_typing_callback = _LiveTypingUpdater(
                local_manifest_file=local_manifest_file,
                local_transfer_dir=local_transfer_dir,
                local_inference_dir=local_inference_dir,
                inference_summary_file=local_inference_dir / "inference_summary.json",
                local_run_dir=local_run_dir,
                local_typing_dir=local_typing_dir,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particle_exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
                typing_normalization_method=getattr(args, "typing_normalization_method", None),
                typing_min_component_area_px=getattr(args, "typing_min_component_area_px", None),
                typing_pixel_size_angstrom=getattr(args, "typing_pixel_size_angstrom", None),
                typing_timeout=getattr(args, "typing_timeout", None),
                resource_env=resource_env,
                typing_workers=live_workers,
            )
        print("Phase: GPU inference.", flush=True)
        inference_ok = False
        inference_started = time.monotonic()
        try:
            inference_command = predict_helpers.run_local_inference(
                input_dir=local_transfer_dir / "micrographs",
                output_dir=local_inference_dir,
                checkpoint=args.checkpoint,
                threshold=float(args.threshold),
                particle_star=particle_star,
                exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
                extra_args=infer_args,
                timeout=args.inference_timeout,
                env_overrides=inference_env,
                num_cpus=inference_cpus,
                num_gpus=getattr(args, "num_gpus", None),
                render_particle_overlays=_render_initial_particle_overlays(
                    run_typing=run_typing,
                    typing_summary_path=typing_summary_path,
                ),
                poll_callback=live_typing_callback,
                poll_interval=5.0,
            )
            inference_ok = True
            print(f"Phase: GPU inference complete in {time.monotonic() - inference_started:.3f}s; final typing/finalization.", flush=True)
        finally:
            if live_typing_callback is not None:
                live_typing_callback.close(cancel=not inference_ok)
        payload["inference_command"] = inference_command

        finalize_payload = _finalize_local_run_outputs(
            config=config,
            transport=transport,
            local_run_dir=local_run_dir,
            local_manifest_file=local_manifest_file,
            local_transfer_dir=local_transfer_dir,
            local_inference_dir=local_inference_dir,
            inference_summary_file=local_inference_dir / "inference_summary.json",
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            external_job_uid=str(prepare_payload["external_job_uid"]),
            remote_transfer_manifest=str(prepare_payload["manifest_file"]),
            remote_run_dir=str(prepare_payload["remote_run_dir"]),
            model={
                "model_id": str(args.model_id),
                "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                "threshold": float(args.threshold),
                "cryoFILTER_version": CRYOFILTER_VERSION,
            },
            particle_exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
            typing_summary_path=typing_summary_path,
            run_typing=run_typing,
            typing_normalization_method=getattr(args, "typing_normalization_method", None),
            typing_min_component_area_px=getattr(args, "typing_min_component_area_px", None),
            typing_pixel_size_angstrom=getattr(args, "typing_pixel_size_angstrom", None),
            typing_timeout=getattr(args, "typing_timeout", None),
            typing_workers=getattr(args, "typing_workers", None),
            max_overlay_images=int(args.max_overlay_images),
            max_bar_items=int(args.max_bar_items),
            resource_env=resource_env,
            timeout=args.timeout,
        )
        payload.update(finalize_payload)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        payload["errors"].append(error)
        if prepare_payload and prepare_payload.get("external_job_uid"):
            failure = _write_and_push_failure_result(
                config=config,
                transport=transport,
                local_run_dir=local_run_dir,
                prepare_payload=prepare_payload,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                error=error,
                timeout=args.timeout,
            )
            if failure is not None:
                payload["failure_finalize"] = failure

    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _run_finalize_run(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    local_run_dir = Path(args.local_run_dir).expanduser().resolve()
    payload: dict[str, Any] = {
        "ok": False,
        "command": "finalize-run",
        "local_run_dir": str(local_run_dir),
        "errors": [],
    }
    try:
        if not local_run_dir.exists():
            raise FileNotFoundError(f"Local run directory does not exist: {local_run_dir}")
        local_manifest_file = (
            Path(args.transfer_manifest).expanduser().resolve()
            if getattr(args, "transfer_manifest", None)
            else local_run_dir / "transfer_manifest.json"
        )
        manifest = predict_helpers.load_transfer_manifest(local_manifest_file)
        run_id = str(UUID(str(manifest.get("run_id"))))
        project_uid = validate_project_uid(str(args.project or manifest.get("project_uid") or ""))
        workspace_uid = validate_workspace_uid(
            str(args.workspace or manifest.get("workspace_uid") or "")
        )
        external_job_uid = validate_job_uid(str(args.job or manifest.get("external_job_uid") or ""))
        local_transfer_dir = (
            Path(args.local_transfer_dir).expanduser().resolve()
            if getattr(args, "local_transfer_dir", None)
            else local_run_dir / "transfer"
        )
        local_inference_dir, inference_summary_file = _resolve_existing_inference_paths(
            local_run_dir=local_run_dir,
            inference_dir=getattr(args, "inference_dir", None),
            inference_summary=getattr(args, "inference_summary", None),
        )
        if not inference_summary_file.exists():
            raise FileNotFoundError(f"Inference summary does not exist: {inference_summary_file}")
        _validate_typing_options(args)
        typing_summary_path: Path | None = (
            Path(args.typing_summary).expanduser().resolve()
            if getattr(args, "typing_summary", None)
            else None
        )
        existing_typing_summary = local_run_dir / "typing" / "summary.json"
        if (
            typing_summary_path is None
            and getattr(args, "run_typing", None) is not True
            and existing_typing_summary.exists()
        ):
            typing_summary_path = existing_typing_summary.resolve()
        run_typing = _resolve_auto_typing(args, typing_summary_path=typing_summary_path)
        resource_env = _local_resource_env(args)
        remote_run_dir = str(
            args.remote_run_dir
            or f"{config.bridge.remote_work_root.rstrip('/')}/{run_id}"
        )
        remote_transfer_manifest = str(
            args.remote_transfer_manifest
            or _remote_run_path(remote_run_dir, "transfer_manifest.json")
        )
        payload.update(
            {
                "run_id": run_id,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "external_job_uid": external_job_uid,
                "local_manifest_file": str(local_manifest_file),
                "local_transfer_dir": str(local_transfer_dir),
                "local_inference_dir": str(local_inference_dir),
                "inference_summary_file": str(inference_summary_file),
                "remote_run_dir": remote_run_dir,
                "remote_transfer_manifest": remote_transfer_manifest,
                "run_typing": run_typing,
                "typing_summary_file": str(typing_summary_path) if typing_summary_path else None,
                "resources": {
                    "num_cpus": getattr(args, "num_cpus", None),
                    "num_gpus": None,
                },
            }
        )

        transport = _transport(config)
        finalize_payload = _finalize_local_run_outputs(
            config=config,
            transport=transport,
            local_run_dir=local_run_dir,
            local_manifest_file=local_manifest_file,
            local_transfer_dir=local_transfer_dir,
            local_inference_dir=local_inference_dir,
            inference_summary_file=inference_summary_file,
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            external_job_uid=external_job_uid,
            remote_transfer_manifest=remote_transfer_manifest,
            remote_run_dir=remote_run_dir,
            model={
                "model_id": str(args.model_id),
                "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                "threshold": float(args.threshold),
                "cryoFILTER_version": CRYOFILTER_VERSION,
            },
            particle_exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
            typing_summary_path=typing_summary_path,
            run_typing=run_typing,
            typing_normalization_method=getattr(args, "typing_normalization_method", None),
            typing_min_component_area_px=getattr(args, "typing_min_component_area_px", None),
            typing_pixel_size_angstrom=getattr(args, "typing_pixel_size_angstrom", None),
            typing_timeout=getattr(args, "typing_timeout", None),
            typing_workers=getattr(args, "typing_workers", None),
            max_overlay_images=int(args.max_overlay_images),
            max_bar_items=int(args.max_bar_items),
            resource_env=resource_env,
            timeout=args.timeout,
        )
        payload.update(finalize_payload)
    except Exception as exc:
        payload["errors"].append(f"{type(exc).__name__}: {exc}")

    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _write_bridge_bundle(bundle_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = bundle_root / "cryofilter"
    cryosparc_package = package_root / "cryosparc"
    cryosparc_package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "cryofilter" / "__init__.py", package_root / "__init__.py")
    shutil.copy2(
        repo_root / "cryofilter" / "cryosparc" / "__init__.py",
        cryosparc_package / "__init__.py",
    )
    shutil.copy2(
        repo_root / "cryofilter" / "cryosparc" / "diagnostics.py",
        cryosparc_package / "diagnostics.py",
    )
    for subpackage in ("protocol", "bridge"):
        shutil.copytree(
            repo_root / "cryofilter" / "cryosparc" / subpackage,
            cryosparc_package / subpackage,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    bin_dir = bundle_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / "cryofilter-bridge"
    launcher.write_text(
        "#!/usr/bin/env sh\n"
        "SCRIPT_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "ROOT=$(CDPATH= cd -- \"$SCRIPT_DIR/..\" && pwd)\n"
        "PYDEPS=\"$ROOT/../cryofilter_bridge_pydeps\"\n"
        "PYTHON=${CRYOFILTER_BRIDGE_PYTHON:-python3}\n"
        "if [ -d \"$PYDEPS\" ]; then\n"
        "  PYTHONPATH=\"$ROOT:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}\"\n"
        "else\n"
        "  PYTHONPATH=\"$ROOT${PYTHONPATH:+:$PYTHONPATH}\"\n"
        "fi\n"
        "export PYTHONPATH\n"
        "exec \"$PYTHON\" -m cryofilter.cryosparc.bridge.cli \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_deploy_bridge(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    transport = _transport(config)
    payload: dict[str, Any] = {
        "ok": False,
        "command": "deploy-bridge",
        "host": config.host,
        "remote_source_root": config.bridge.remote_source_root,
        "remote_command": config.bridge.command,
        "errors": [],
    }
    try:
        tmp_parent = Path.cwd() / ".codex_tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cryofilter_bridge_bundle_", dir=tmp_parent) as tmp:
            bundle_root = Path(tmp)
            _write_bridge_bundle(bundle_root)
            transport.run(["mkdir", "-p", config.bridge.remote_source_root], timeout=args.timeout)
            transport.push(
                bundle_root,
                config.bridge.remote_source_root,
                contents=True,
                timeout=args.timeout,
            )
        remote_version, error = _run_remote_bridge_json(config, ["--version"], timeout=30.0)
        if remote_version is None:
            payload["errors"].append(error)
        else:
            payload["remote_version"] = remote_version
            payload["ok"] = (
                remote_version.get("protocol_version") == PROTOCOL_VERSION
                and bool(remote_version.get("bridge_version"))
            )
    except Exception as exc:
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("ok") else 1


def _run_test_transport(args: argparse.Namespace, config: CryoSPARCIntegrationConfig) -> int:
    transport = _transport(config)
    run_id = uuid4().hex
    remote_dir = f"{config.bridge.remote_work_root.rstrip('/')}/transport_tests/{run_id}"
    payload: dict[str, Any] = {
        "ok": False,
        "command": "test-transport",
        "run_id": run_id,
        "host": config.host,
        "remote_dir": remote_dir,
        "checks": [],
        "errors": [],
    }

    def add_check(name: str, ok: bool, detail: str | None = None) -> None:
        payload["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        result = transport.run(["hostname"], timeout=args.timeout)
        add_check("SSH local -> worker", result.ok, result.stdout.strip())
        transport.run(["mkdir", "-p", remote_dir], timeout=args.timeout)
        add_check("remote command", True, "mkdir")
        tmp_parent = Path.cwd() / ".codex_tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cryofilter_transport_", dir=tmp_parent) as tmp:
            tmp_path = Path(tmp)
            local_payload = tmp_path / "payload.bin"
            pulled_payload = tmp_path / "payload.pulled.bin"
            local_payload.write_bytes(os.urandom(1024))
            expected = sha256_file(local_payload)
            transport.push(local_payload, f"{remote_dir}/payload.bin", timeout=args.timeout)
            add_check("push file", True, "1024 bytes")
            transport.pull(f"{remote_dir}/payload.bin", pulled_payload, timeout=args.timeout)
            add_check("pull file", pulled_payload.exists(), str(pulled_payload))
            actual = sha256_file(pulled_payload)
            add_check("SHA256 identical", expected == actual, actual)
        if args.keep_remote_temp:
            add_check("remote cleanup", True, "kept by request")
        else:
            transport.run(["rm", "-f", f"{remote_dir}/payload.bin"], timeout=args.timeout)
            transport.run(["rmdir", remote_dir], timeout=args.timeout)
            add_check("remote cleanup", True, remote_dir)
    except Exception as exc:
        payload["errors"].append(f"{type(exc).__name__}: {exc}")

    payload["ok"] = not payload["errors"] and all(
        bool(check.get("ok")) for check in payload["checks"]
    )
    _emit(payload, as_json=bool(args.json))
    return 0 if payload["ok"] else 1


def _run_cryosparc(args: argparse.Namespace) -> int:
    config = _apply_cli_overrides(load_config(args.config), args)
    command = args.cryosparc_command
    if command == "doctor":
        return _run_doctor(args, config)
    if command == "test-transport":
        return _run_test_transport(args, config)
    if command == "deploy-bridge":
        return _run_deploy_bridge(args, config)
    if command == "inspect":
        return _run_inspect(args, config)
    if command == "test-roundtrip":
        return _run_roundtrip(args, config)
    if command == "stage-test":
        return _run_stage_test(args, config)
    if command == "build-diagnostics":
        return _run_build_diagnostics(args)
    if command == "attach-diagnostics":
        return _run_attach_diagnostics(args, config)
    if command in {"predict", "finalize-run", "resume", "finalize-from-run"}:
        # App cancellation sends SIGTERM. Unwind local subprocess supervisors
        # so their inference/typing worker groups are stopped and reaped.
        def terminate(_signum, _frame):
            raise SystemExit(143)

        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = signal.signal(signal.SIGTERM, terminate)
        try:
            return _run_predict(args, config) if command == "predict" else _run_finalize_run(args, config)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
    raise ValueError(f"Unknown cryosparc command: {command}")


__all__ = ["add_subparser"]
