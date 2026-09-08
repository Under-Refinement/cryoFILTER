"""Finalize cryoFILTER predictions into a CryoSPARC External Job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from cryofilter.cryosparc.bridge.client import make_client
from cryofilter.cryosparc.bridge.compat import (
    add_particle_output,
    dataset_filter_prefixes,
    dataset_prefixes,
    dataset_take,
    dataset_uids,
    find_external_job,
    find_job,
    find_project,
    run_external_job,
    save_output,
    stop_external_job,
)
from cryofilter.cryosparc.bridge.diagnostics import log_diagnostic_assets
from cryofilter.cryosparc.protocol.manifests import read_json_model
from cryofilter.cryosparc.protocol.models import (
    FinalizePredictionRequest,
    FinalizePredictionResponse,
    ResultManifest,
    TransferManifest,
)
from cryofilter.cryosparc.protocol.validation import validate_uid_partition

ACCEPTED_UID_KEYS = ("accepted_uids_json", "accepted_particle_uids_json", "accepted_uids")
REJECTED_UID_KEYS = ("rejected_uids_json", "rejected_particle_uids_json", "rejected_uids")
DIAGNOSTICS_KEYS = ("diagnostics_manifest_json", "diagnostics_manifest")


def _read_uid_file(path: str | Path) -> list[int]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = None
        for key in ("uids", "accepted_uids", "rejected_uids"):
            if key in data:
                values = data[key]
                break
    elif isinstance(data, list):
        values = data
    else:
        values = None
    if values is None or not isinstance(values, list):
        raise ValueError(f"UID file must be a list or contain a 'uids' list: {path}")
    return [int(value) for value in values]


def _required_file(result: ResultManifest, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = result.files.get(key)
        if value:
            return str(value)
    raise ValueError(f"Result manifest is missing one of these file entries: {keys}")


def _optional_file(result: ResultManifest, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = result.files.get(key)
        if value:
            return str(value)
    return None


def _validate_identity(
    request: FinalizePredictionRequest,
    transfer: TransferManifest,
    result: ResultManifest,
) -> None:
    if transfer.project_uid != request.project_uid:
        raise ValueError(
            f"Transfer manifest project {transfer.project_uid} does not match {request.project_uid}"
        )
    if transfer.workspace_uid != request.workspace_uid:
        raise ValueError(
            f"Transfer manifest workspace {transfer.workspace_uid} does not match {request.workspace_uid}"
        )
    if transfer.external_job_uid != request.job_uid:
        raise ValueError(
            f"Transfer manifest external job {transfer.external_job_uid} does not match {request.job_uid}"
        )
    if result.run_id != transfer.run_id:
        raise ValueError(f"Result manifest run_id {result.run_id} does not match {transfer.run_id}")
    if result.project_uid is not None and result.project_uid != request.project_uid:
        raise ValueError(f"Result manifest project {result.project_uid} does not match {request.project_uid}")
    if result.workspace_uid is not None and result.workspace_uid != request.workspace_uid:
        raise ValueError(
            f"Result manifest workspace {result.workspace_uid} does not match {request.workspace_uid}"
        )
    if result.external_job_uid is not None and result.external_job_uid != request.job_uid:
        raise ValueError(
            f"Result manifest external job {result.external_job_uid} does not match {request.job_uid}"
        )


def _python_ints(values: Iterable[object]) -> list[int]:
    return [int(value) for value in values]


def _indices_for_uids(
    source_uids: Iterable[object],
    requested_uids: Iterable[object],
    *,
    label: str,
) -> np.ndarray:
    source_list = _python_ints(source_uids)
    requested_list = _python_ints(requested_uids)
    requested_set = set(requested_list)
    indices = [index for index, value in enumerate(source_list) if value in requested_set]
    if len(indices) != len(requested_set):
        found = {source_list[index] for index in indices}
        missing = sorted(requested_set - found)
        extra = "" if not missing else f"; missing from source dataset: {missing[:10]}"
        raise ValueError(f"Could not map every {label} UID to source particles{extra}")
    return np.asarray(indices, dtype=np.int64)


def _finish_failed_job(
    request: FinalizePredictionRequest,
    result: ResultManifest,
    *,
    connection_file: str | None,
) -> FinalizePredictionResponse:
    errors: list[str] = []
    try:
        client = make_client(connection_file)
        project = find_project(client, request.project_uid)
        external_job = find_external_job(client, project, request.project_uid, request.job_uid)
        stop_external_job(
            external_job,
            error=result.error or "cryoFILTER prediction failed before CryoSPARC finalization.",
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    if result.error:
        errors.insert(0, result.error)
    return FinalizePredictionResponse(
        ok=False,
        run_id=result.run_id,
        project_uid=request.project_uid,
        workspace_uid=request.workspace_uid,
        job_uid=request.job_uid,
        status="failed",
        input_particles=int(result.particles_processed),
        accepted_particles=int(result.accepted_particles),
        rejected_particles=int(result.rejected_particles),
        validated=False,
        errors=errors,
    )


def finalize_prediction(
    request: FinalizePredictionRequest,
    *,
    connection_file: str | None = None,
) -> FinalizePredictionResponse:
    """Save cryoFILTER accepted/rejected particles and diagnostics to CryoSPARC."""

    errors: list[str] = []
    event_ids: list[str] = []
    attached_assets = 0
    saved_outputs: dict[str, int] = {}
    run_id = None
    status = "failed"
    input_count = 0
    accepted_count = 0
    rejected_count = 0
    validated = False

    try:
        transfer = read_json_model(TransferManifest, request.transfer_manifest_file)
        result = read_json_model(ResultManifest, request.result_manifest_file)
        run_id = result.run_id
        status = result.status
        _validate_identity(request, transfer, result)

        if result.status == "failed":
            return _finish_failed_job(request, result, connection_file=connection_file)

        if transfer.particles_ref is None:
            raise ValueError("Prediction finalization requires a particle input reference")
        if not transfer.particles:
            raise ValueError("Transfer manifest contains no staged particles to finalize")

        accepted_uids = _read_uid_file(_required_file(result, ACCEPTED_UID_KEYS))
        rejected_uids = _read_uid_file(_required_file(result, REJECTED_UID_KEYS))
        staged_uids = [int(entry.uid) for entry in transfer.particles]
        input_count = int(len(staged_uids))
        accepted_count = int(len(accepted_uids))
        rejected_count = int(len(rejected_uids))
        if int(result.particles_processed) != input_count:
            raise ValueError(
                f"Result manifest particles_processed={result.particles_processed} "
                f"does not match staged particle count {input_count}"
            )
        if int(result.accepted_particles) != accepted_count:
            raise ValueError(
                f"Result manifest accepted_particles={result.accepted_particles} "
                f"does not match UID file count {accepted_count}"
            )
        if int(result.rejected_particles) != rejected_count:
            raise ValueError(
                f"Result manifest rejected_particles={result.rejected_particles} "
                f"does not match UID file count {rejected_count}"
            )
        report = validate_uid_partition(
            input_uids=staged_uids,
            accepted_uids=accepted_uids,
            rejected_uids=rejected_uids,
        )
        report.require_ok()
        validated = True

        client = make_client(connection_file)
        project = find_project(client, request.project_uid)
        source_job = find_job(
            client,
            project,
            request.project_uid,
            transfer.particles_ref.job_uid,
        )
        source_particles = source_job.load_output(transfer.particles_ref.output_name)
        source_uids = _python_ints(dataset_uids(source_particles))
        accepted_indices = _indices_for_uids(source_uids, accepted_uids, label="accepted")
        rejected_indices = _indices_for_uids(source_uids, rejected_uids, label="rejected")

        accepted_particles = dataset_take(source_particles, accepted_indices)
        rejected_particles = dataset_take(source_particles, rejected_indices)
        slots = dataset_prefixes(source_particles)
        if not slots:
            raise ValueError("Could not determine output slots from particle dataset")
        output_slots = ["location"] if "location" in slots else [slots[0]]
        output_accepted = dataset_filter_prefixes(accepted_particles, output_slots)
        output_rejected = dataset_filter_prefixes(rejected_particles, output_slots)

        external_job = find_external_job(client, project, request.project_uid, request.job_uid)
        add_particle_output(
            external_job,
            name="particles_accepted",
            passthrough="input_particles",
            slots=output_slots,
            title="cryoFILTER accepted particles",
            alloc=accepted_particles,
        )
        add_particle_output(
            external_job,
            name="particles_rejected",
            passthrough="input_particles",
            slots=output_slots,
            title="cryoFILTER rejected particles",
            alloc=rejected_particles,
        )

        diagnostics_manifest_file = _optional_file(result, DIAGNOSTICS_KEYS)
        with run_external_job(external_job):
            save_output(external_job, "particles_accepted", output_accepted)
            save_output(external_job, "particles_rejected", output_rejected)
            saved_outputs["particles_accepted"] = int(len(accepted_particles))
            saved_outputs["particles_rejected"] = int(len(rejected_particles))
            if diagnostics_manifest_file:
                attached_assets, event_ids, diagnostic_errors = log_diagnostic_assets(
                    external_job,
                    diagnostics_manifest_file,
                )
                errors.extend(diagnostic_errors)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return FinalizePredictionResponse(
        ok=validated and not errors and bool(saved_outputs),
        run_id=run_id,
        project_uid=request.project_uid,
        workspace_uid=request.workspace_uid,
        job_uid=request.job_uid,
        status=status,
        input_particles=input_count,
        accepted_particles=accepted_count,
        rejected_particles=rejected_count,
        saved_outputs=saved_outputs,
        validated=validated,
        attached_assets=attached_assets,
        event_ids=event_ids,
        errors=errors,
    )


__all__ = ["finalize_prediction"]
