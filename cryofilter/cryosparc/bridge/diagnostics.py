"""Attach cryoFILTER diagnostic plots to a CryoSPARC job."""

from __future__ import annotations

from pathlib import Path

from cryofilter.cryosparc.bridge.client import make_client
from cryofilter.cryosparc.bridge.compat import find_job, find_project, log_plot
from cryofilter.cryosparc.protocol.manifests import read_json_model
from cryofilter.cryosparc.protocol.models import (
    AttachDiagnosticsRequest,
    AttachDiagnosticsResponse,
    DiagnosticsManifest,
)


def log_diagnostic_assets(job: object, diagnostics_manifest_file: str | Path) -> tuple[int, list[str], list[str]]:
    """Log all PNG assets from a diagnostics manifest to an existing job."""

    errors: list[str] = []
    event_ids: list[str] = []
    manifest = read_json_model(DiagnosticsManifest, diagnostics_manifest_file)

    for asset in manifest.assets:
        raw_path = asset.remote_path or asset.local_path
        asset_path = Path(raw_path).expanduser()
        if not asset_path.exists():
            errors.append(f"Diagnostic asset does not exist on bridge host: {asset_path}")
            continue
        if asset_path.suffix.lower() != ".png":
            errors.append(f"Diagnostic asset must be a PNG for CryoSPARC logging: {asset_path}")
            continue
        event_id = log_plot(
            job,
            figure=asset_path,
            text=f"{asset.title}\n\n{asset.caption}",
            formats=("png",),
        )
        if event_id is not None:
            event_ids.append(str(event_id))

    return len(event_ids), event_ids, errors


def attach_diagnostics(
    request: AttachDiagnosticsRequest,
    *,
    connection_file: str | None = None,
) -> AttachDiagnosticsResponse:
    """Log PNG diagnostics to the target CryoSPARC job."""

    errors: list[str] = []
    event_ids: list[str] = []
    attached = 0

    try:
        client = make_client(connection_file)
        project = find_project(client, request.project_uid)
        job = find_job(client, project, request.project_uid, request.job_uid)
        attached, event_ids, asset_errors = log_diagnostic_assets(
            job,
            request.diagnostics_manifest_file,
        )
        errors.extend(asset_errors)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return AttachDiagnosticsResponse(
        ok=not errors,
        project_uid=request.project_uid,
        job_uid=request.job_uid,
        diagnostics_manifest_file=request.diagnostics_manifest_file,
        attached_assets=attached,
        event_ids=event_ids,
        errors=errors,
    )


__all__ = ["attach_diagnostics", "log_diagnostic_assets"]
