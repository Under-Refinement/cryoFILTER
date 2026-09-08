"""Deterministic particle External Job round-trip test."""

from __future__ import annotations

import numpy as np

from cryofilter.cryosparc.bridge.client import make_client
from cryofilter.cryosparc.bridge.compat import (
    add_particle_output,
    connect_particle_input,
    create_external_job,
    dataset_filter_prefixes,
    dataset_prefixes,
    dataset_take,
    dataset_uids,
    find_job,
    find_project,
    object_uid,
    run_external_job,
    save_output,
)
from cryofilter.cryosparc.protocol.models import (
    TestRoundtripRequest,
    TestRoundtripResponse,
)
from cryofilter.cryosparc.protocol.validation import validate_uid_partition


def run_particle_roundtrip(
    request: TestRoundtripRequest,
    *,
    connection_file: str | None = None,
) -> TestRoundtripResponse:
    """Create an External Job that splits a particle set by UID and reloads it."""

    errors: list[str] = []
    external_job_uid: str | None = None
    input_count = 0
    count_a = 0
    count_b = 0
    validated = False

    try:
        client = make_client(connection_file)
        project = find_project(client, request.project_uid)
        source_job = find_job(
            client,
            project,
            request.project_uid,
            request.particles.job_uid,
        )
        particles = source_job.load_output(request.particles.output_name)
        input_uids = dataset_uids(particles)
        input_count = int(len(input_uids))
        a_indices = np.arange(0, input_count, 2, dtype=np.int64)
        b_indices = np.arange(1, input_count, 2, dtype=np.int64)
        a_uids = input_uids[a_indices]
        b_uids = input_uids[b_indices]
        report = validate_uid_partition(
            input_uids=input_uids,
            accepted_uids=a_uids,
            rejected_uids=b_uids,
        )
        report.require_ok()

        particles_a = dataset_take(particles, a_indices)
        particles_b = dataset_take(particles, b_indices)
        count_a = int(len(particles_a))
        count_b = int(len(particles_b))
        slots = dataset_prefixes(particles)
        if not slots:
            raise ValueError("Could not determine output slots from particle dataset")
        output_slots = ["location"] if "location" in slots else [slots[0]]
        output_particles_a = dataset_filter_prefixes(particles_a, output_slots)
        output_particles_b = dataset_filter_prefixes(particles_b, output_slots)

        title = f"{request.title} ({request.run_id})"
        external_job = create_external_job(
            project,
            request.workspace_uid,
            title=title,
            desc=(
                "cryoFILTER bridge validation: deterministic every-other-particle "
                "round-trip with no MRC transfer and no model inference."
            ),
        )
        external_job_uid = object_uid(external_job)
        connect_particle_input(
            external_job,
            input_name="input_particles",
            source_job_uid=request.particles.job_uid,
            source_output_name=request.particles.output_name,
            slots=slots,
        )
        add_particle_output(
            external_job,
            name="particles_a",
            passthrough="input_particles",
            slots=output_slots,
            title="cryoFILTER round-trip particles A",
            alloc=particles_a,
        )
        add_particle_output(
            external_job,
            name="particles_b",
            passthrough="input_particles",
            slots=output_slots,
            title="cryoFILTER round-trip particles B",
            alloc=particles_b,
        )

        with run_external_job(external_job):
            save_output(external_job, "particles_a", output_particles_a)
            save_output(external_job, "particles_b", output_particles_b)

        reloaded_job = find_job(client, project, request.project_uid, external_job_uid)
        reloaded_a = reloaded_job.load_output("particles_a")
        reloaded_b = reloaded_job.load_output("particles_b")
        reload_report = validate_uid_partition(
            input_uids=input_uids,
            accepted_uids=dataset_uids(reloaded_a),
            rejected_uids=dataset_uids(reloaded_b),
        )
        reload_report.require_ok()
        validated = True
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return TestRoundtripResponse(
        ok=validated and not errors,
        run_id=request.run_id,
        project_uid=request.project_uid,
        workspace_uid=request.workspace_uid,
        source_job_uid=request.particles.job_uid,
        source_output_name=request.particles.output_name,
        external_job_uid=external_job_uid,
        input_particles=input_count,
        particles_a=count_a,
        particles_b=count_b,
        validated=validated,
        errors=errors,
    )


__all__ = ["run_particle_roundtrip"]
