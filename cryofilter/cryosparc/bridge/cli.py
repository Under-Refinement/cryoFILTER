"""Console entry point for the cryoFILTER CryoSPARC bridge."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID, uuid4

from cryofilter.cryosparc import PROTOCOL_VERSION
from cryofilter.cryosparc.bridge import BRIDGE_VERSION
from cryofilter.cryosparc.bridge.doctor import run_doctor
from cryofilter.cryosparc.bridge.inspect import inspect_cryosparc
from cryofilter.cryosparc.bridge.roundtrip import run_particle_roundtrip
from cryofilter.cryosparc.protocol.models import (
    AttachDiagnosticsRequest,
    FinalizePredictionRequest,
    InspectRequest,
    PredictPrepareRequest,
    TestRoundtripRequest,
    load_model_json,
)
from cryofilter.cryosparc.protocol.validation import (
    parse_output_ref,
    validate_job_uid,
    validate_project_uid,
    validate_workspace_uid,
)


def _emit(payload: Any, *, as_json: bool) -> None:
    if hasattr(payload, "model_dump"):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    if as_json:
        print(text)
    else:
        print(text)


def _call_quietly_for_json(callback, *, as_json: bool):
    if not as_json:
        return callback()

    noise = io.StringIO()
    try:
        with contextlib.redirect_stdout(noise):
            return callback()
    finally:
        captured = noise.getvalue()
        if captured:
            print(captured, end="", file=sys.stderr)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Write valid JSON to stdout.")
    parser.add_argument(
        "--connection-file",
        default=None,
        help="Optional JSON file with CryoSPARC Tools connection arguments.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryofilter-bridge",
        description="Remote bridge for cryoFILTER CryoSPARC integration.",
    )
    parser.add_argument("--version", action="store_true", help="Print bridge version.")
    parser.add_argument("--json", action="store_true", help="Write valid JSON to stdout.")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Report bridge and CryoSPARC health.")
    _add_common(doctor)
    doctor.add_argument("--project", default=None, help="Optional project UID to validate.")

    inspect = subparsers.add_parser("inspect", help="Inspect project/workspace/job metadata.")
    _add_common(inspect)
    inspect.add_argument("--project", default=None, help="Project UID, for example P42.")
    inspect.add_argument("--workspace", default=None, help="Workspace UID, for example W5.")
    inspect.add_argument("--job", default=None, help="Job UID, for example J20.")

    inspect_projects = subparsers.add_parser("inspect-projects", help="List CryoSPARC projects.")
    _add_common(inspect_projects)

    roundtrip = subparsers.add_parser(
        "test-roundtrip",
        help="Create an External Job that splits one particle output by UID.",
    )
    _add_common(roundtrip)
    roundtrip.add_argument("--request", default=None, help="Optional JSON request file.")
    roundtrip.add_argument("--run-id", default=None, help="Optional UUID for traceability.")
    roundtrip.add_argument("--project", default=None, help="Project UID, for example P42.")
    roundtrip.add_argument("--workspace", default=None, help="Workspace UID, for example W5.")
    roundtrip.add_argument("--particles", default=None, help="Particle output ref, for example J20:particles.")
    roundtrip.add_argument("--title", default="cryoFILTER particle round-trip test")

    prepare = subparsers.add_parser(
        "prepare",
        help="Create a cryoFILTER staging manifest and optional External Job.",
    )
    _add_common(prepare)
    prepare.add_argument("--request", default=None, help="Optional JSON request file.")
    prepare.add_argument("--run-id", default=None, help="Optional UUID for traceability.")
    prepare.add_argument("--project", default=None, help="Project UID, for example P42.")
    prepare.add_argument("--workspace", default=None, help="Workspace UID, for example W5.")
    prepare.add_argument("--micrographs", default=None, help="Micrograph output ref, for example J20:micrographs.")
    prepare.add_argument("--particles", default=None, help="Optional particle output ref, for example J21:particles.")
    prepare.add_argument("--model-id", default="cryoFILTER", help="Model label stored with the run request.")
    prepare.add_argument("--threshold", type=float, default=0.6, help="Mask threshold stored with the run request.")
    prepare.add_argument("--remote-work-root", default="~/cryofilter_bridge_work", help="Remote staging root.")
    prepare.add_argument("--title", default="cryoFILTER prediction", help="External Job title prefix.")
    prepare.add_argument("--limit-micrographs", type=int, default=None, help="Stage at most this many micrographs.")
    prepare.add_argument("--micrograph-path-field", default=None, help="Override the dataset field containing micrograph paths.")
    prepare.add_argument(
        "--no-stage-micrographs",
        action="store_true",
        help="Write source-path manifest entries without creating transfer symlinks.",
    )
    prepare.add_argument(
        "--no-create-external-job",
        action="store_true",
        help="Only create the staging manifest; do not create a CryoSPARC External Job.",
    )

    attach_diagnostics = subparsers.add_parser(
        "attach-diagnostics",
        help="Attach diagnostic PNGs from a diagnostics manifest to a CryoSPARC job.",
    )
    _add_common(attach_diagnostics)
    attach_diagnostics.add_argument("--request", default=None, help="Optional JSON request file.")
    attach_diagnostics.add_argument("--project", default=None, help="Project UID, for example P42.")
    attach_diagnostics.add_argument("--job", default=None, help="Target job UID, for example J20.")
    attach_diagnostics.add_argument("--diagnostics-manifest", default=None, help="Diagnostics manifest path on the bridge host.")

    finalize = subparsers.add_parser(
        "finalize-prediction",
        help="Save cryoFILTER particle outputs and diagnostics to a prepared External Job.",
    )
    _add_common(finalize)
    finalize.add_argument("--request", default=None, help="Optional JSON request file.")
    finalize.add_argument("--project", default=None, help="Project UID, for example P42.")
    finalize.add_argument("--workspace", default=None, help="Workspace UID, for example W5.")
    finalize.add_argument("--job", default=None, help="Prepared External Job UID, for example J20.")
    finalize.add_argument("--transfer-manifest", default=None, help="Transfer manifest path on the bridge host.")
    finalize.add_argument("--result-manifest", default=None, help="Result manifest path on the bridge host.")

    return parser


def _version_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "bridge_version": BRIDGE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }


def _roundtrip_request_from_args(args: argparse.Namespace) -> TestRoundtripRequest:
    if args.request:
        return load_model_json(TestRoundtripRequest, Path(args.request))  # type: ignore[return-value]
    if not args.project or not args.workspace or not args.particles:
        raise ValueError("--project, --workspace, and --particles are required without --request")
    project_uid = validate_project_uid(args.project)
    particles = parse_output_ref(args.particles, project_uid=project_uid)
    run_id = UUID(args.run_id) if args.run_id else uuid4()
    return TestRoundtripRequest(
        run_id=run_id,
        project_uid=project_uid,
        workspace_uid=validate_workspace_uid(args.workspace),
        particles={
            "project_uid": particles.project_uid,
            "job_uid": particles.job_uid,
            "output_name": particles.output_name,
        },
        title=str(args.title),
    )


def _prepare_request_from_args(args: argparse.Namespace) -> PredictPrepareRequest:
    if args.request:
        return load_model_json(PredictPrepareRequest, Path(args.request))  # type: ignore[return-value]
    if not args.project or not args.workspace or not args.micrographs:
        raise ValueError("--project, --workspace, and --micrographs are required without --request")
    project_uid = validate_project_uid(args.project)
    micrographs = parse_output_ref(args.micrographs, project_uid=project_uid)
    particles = parse_output_ref(args.particles, project_uid=project_uid) if args.particles else None
    run_id = UUID(args.run_id) if args.run_id else uuid4()
    return PredictPrepareRequest(
        run_id=run_id,
        project_uid=project_uid,
        workspace_uid=validate_workspace_uid(args.workspace),
        micrographs={
            "project_uid": micrographs.project_uid,
            "job_uid": micrographs.job_uid,
            "output_name": micrographs.output_name,
        },
        particles=(
            None
            if particles is None
            else {
                "project_uid": particles.project_uid,
                "job_uid": particles.job_uid,
                "output_name": particles.output_name,
            }
        ),
        model_id=str(args.model_id),
        threshold=float(args.threshold),
        remote_work_root=str(args.remote_work_root),
        title=str(args.title),
        limit_micrographs=args.limit_micrographs,
        micrograph_path_field=args.micrograph_path_field,
        create_external_job=not bool(args.no_create_external_job),
        stage_micrographs=not bool(args.no_stage_micrographs),
    )


def _attach_diagnostics_request_from_args(args: argparse.Namespace) -> AttachDiagnosticsRequest:
    if args.request:
        return load_model_json(AttachDiagnosticsRequest, Path(args.request))  # type: ignore[return-value]
    if not args.project or not args.job or not args.diagnostics_manifest:
        raise ValueError("--project, --job, and --diagnostics-manifest are required without --request")
    return AttachDiagnosticsRequest(
        project_uid=validate_project_uid(args.project),
        job_uid=validate_job_uid(args.job),
        diagnostics_manifest_file=str(args.diagnostics_manifest),
    )


def _finalize_prediction_request_from_args(args: argparse.Namespace) -> FinalizePredictionRequest:
    if args.request:
        return load_model_json(FinalizePredictionRequest, Path(args.request))  # type: ignore[return-value]
    if not args.project or not args.workspace or not args.job or not args.transfer_manifest or not args.result_manifest:
        raise ValueError(
            "--project, --workspace, --job, --transfer-manifest, and --result-manifest "
            "are required without --request"
        )
    return FinalizePredictionRequest(
        project_uid=validate_project_uid(args.project),
        workspace_uid=validate_workspace_uid(args.workspace),
        job_uid=validate_job_uid(args.job),
        transfer_manifest_file=str(args.transfer_manifest),
        result_manifest_file=str(args.result_manifest),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.version or args.command is None:
            _emit(_version_payload(), as_json=bool(args.json))
            return 0

        if args.command == "doctor":
            project_uid = validate_project_uid(args.project) if args.project else None
            response = _call_quietly_for_json(
                lambda: run_doctor(
                    connection_file=args.connection_file,
                    project_uid=project_uid,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        if args.command == "inspect-projects":
            response = _call_quietly_for_json(
                lambda: inspect_cryosparc(
                    InspectRequest(),
                    connection_file=args.connection_file,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        if args.command == "inspect":
            if (args.workspace or args.job) and not args.project:
                raise ValueError("--workspace and --job require --project")
            response = _call_quietly_for_json(
                lambda: inspect_cryosparc(
                    InspectRequest(
                        project_uid=validate_project_uid(args.project) if args.project else None,
                        workspace_uid=validate_workspace_uid(args.workspace) if args.workspace else None,
                        job_uid=validate_job_uid(args.job) if args.job else None,
                    ),
                    connection_file=args.connection_file,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        if args.command == "test-roundtrip":
            response = _call_quietly_for_json(
                lambda: run_particle_roundtrip(
                    _roundtrip_request_from_args(args),
                    connection_file=args.connection_file,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        if args.command == "prepare":
            from cryofilter.cryosparc.bridge.prepare import run_prepare

            response = _call_quietly_for_json(
                lambda: run_prepare(
                    _prepare_request_from_args(args),
                    connection_file=args.connection_file,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        if args.command == "attach-diagnostics":
            from cryofilter.cryosparc.bridge.diagnostics import attach_diagnostics

            response = _call_quietly_for_json(
                lambda: attach_diagnostics(
                    _attach_diagnostics_request_from_args(args),
                    connection_file=args.connection_file,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        if args.command == "finalize-prediction":
            from cryofilter.cryosparc.bridge.finalize import finalize_prediction

            response = _call_quietly_for_json(
                lambda: finalize_prediction(
                    _finalize_prediction_request_from_args(args),
                    connection_file=args.connection_file,
                ),
                as_json=bool(args.json),
            )
            _emit(response, as_json=bool(args.json))
            return 0 if response.ok else 1

        parser.error(f"Unknown command: {args.command}")
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if bool(getattr(args, "json", False)):
            print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
