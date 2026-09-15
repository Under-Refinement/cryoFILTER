"""CPU-only, vectorized particle filtering using the OTF card's UID index."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .bridge.compat import (
    add_particle_output, connect_particle_input,
    create_external_job, dataset_filter_prefixes, dataset_prefixes, dataset_take,
    find_external_job, find_job, find_project, object_uid, run_external_job, save_output,
    stop_external_job,
)
from .bridge.prepare import _dataset_fields, _object_dir, _load_output_with_aliases
from .otf_state import Index, file_stamp, load_mask, read_card, write_json
from .protocol.models import CryoSPARCOutputRef
from .protocol.validation import parse_output_ref, validate_job_uid, validate_project_uid, validate_workspace_uid


def keep_from_mask(fractions, mask, *, shape, pixel_size, exclusion):
    """Exact pixel-center rule without normally allocating a full distance map.

    A clean pixel's nearest contaminated pixel lies on the mask boundary. Typical
    cryoFILTER masks have relatively short boundaries; query those with a KD-tree.
    Fragmented masks use the original distance transform to bound tree memory.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D contamination mask")
    height, width = shape
    mh, mw = mask.shape
    points = np.asarray(fractions, dtype=np.float64) * [width, height]
    # Preserve the legacy scalar filter's float64 arithmetic and edge clipping.
    x = np.clip(np.floor(points[:, 0] * mw / width).astype(np.int64), 0, mw - 1)
    y = np.clip(np.floor(points[:, 1] * mh / height).astype(np.int64), 0, mh - 1)
    keep = ~mask[y, x]
    if exclusion == 0 or not keep.any() or not mask.any():
        return keep
    sy, sx = pixel_size * height / mh, pixel_size * width / mw
    boundary = mask & ~ndi.binary_erosion(mask)
    boundary_count = np.count_nonzero(boundary)
    if boundary_count > min(250000, mask.size // 8):
        distances = ndi.distance_transform_edt(~mask, sampling=(sy, sx)).astype(np.float32)
        keep[keep] = distances[y[keep], x[keep]] > exclusion
    else:
        by, bx = np.nonzero(boundary)
        tree = cKDTree(np.column_stack((by * sy, bx * sx)))
        distances, _ = tree.query(np.column_stack((y[keep] * sy, x[keep] * sx)), workers=1)
        # The original filter stores its EDT as float32 before comparing distance.
        keep[keep] = distances.astype(np.float32) > exclusion
    return keep


def filter_by_index(particles, rows, *, exclusion_angstrom=100.0, missing_masks="error"):
    if not math.isfinite(exclusion_angstrom) or exclusion_angstrom < 0:
        raise ValueError("Exclusion distance must be finite and nonnegative")
    if missing_masks not in {"error", "pending"}:
        raise ValueError("Missing-mask policy must be error or pending")
    required = {"uid", "location/micrograph_uid", "location/center_x_frac", "location/center_y_frac"}
    missing = required - set(_dataset_fields(particles))
    if missing:
        raise ValueError(f"Particle output lacks required fields: {sorted(missing)}")
    uids = np.asarray(particles["location/micrograph_uid"], dtype=np.uint64)
    coords = np.column_stack((particles["location/center_x_frac"], particles["location/center_y_frac"]))
    if not np.all(np.isfinite(coords)) or np.any((coords < 0) | (coords > 1)):
        raise ValueError("Particle coordinates must be finite fractions between 0 and 1")
    lookup = {entry["uid"]: (entry, inference) for entry, inference, _ in rows if inference}
    # Stable groups preserve original particle order in the final boolean selection.
    order = np.argsort(uids, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(uids[order]) != 0) + 1, len(order)]
    decisions = np.full(len(uids), 2, dtype=np.uint8)  # 0 accepted, 1 rejected, 2 pending
    missing_uids = []
    counts = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start == end:
            continue
        indices = order[start:end]
        uid = str(int(uids[indices[0]]))
        match = lookup.get(uid)
        ready = False
        if match is not None:
            entry, inference = match
            try:
                ready = (file_stamp(entry["path"]) == entry["stamp"] and
                         file_stamp(inference["output_mask_npy"]) == inference["output_stamps"][0])
            except (OSError, KeyError):
                ready = False
        if not ready:
            missing_uids.append(uid)
            continue
        mask = load_mask(inference["output_mask_npy"])
        if mask.ndim != 2 or list(mask.shape) != list(inference["output_image_shape"]):
            raise ValueError(f"Mask geometry changed for micrograph UID {uid}")
        keep = keep_from_mask(coords[indices], mask, shape=entry["shape"],
                              pixel_size=entry["pixel_size_angstrom"], exclusion=exclusion_angstrom)
        decisions[indices] = np.where(keep, 0, 1)
        counts.append({"micrograph_uid": uid, "particles": len(indices),
                       "accepted": int(keep.sum()), "rejected": int((~keep).sum())})
        # Each distance transform is released before loading the next mask.
    if missing_uids and missing_masks == "error":
        raise ValueError(f"Masks are missing, unfinished or changed for {len(missing_uids)} micrograph(s) "
                         f"({int(np.count_nonzero(decisions == 2))} particles). Example UIDs: {', '.join(missing_uids[:5])}. "
                         "Wait for OTF, or choose pending to publish unprocessed particles separately.")
    return decisions, {"particles": len(uids), "accepted": int(np.count_nonzero(decisions == 0)),
                       "rejected": int(np.count_nonzero(decisions == 1)), "pending": int(np.count_nonzero(decisions == 2)),
                       "missing_micrograph_uids": missing_uids, "per_micrograph": counts}


def run_filter(args, config):
    from .otf import _connect
    from .remote.cli import _normalize_output_ref
    project_uid = validate_project_uid(args.project)
    workspace_uid = validate_workspace_uid(args.workspace)
    otf_uid = validate_job_uid(args.otf_job)
    ref = parse_output_ref(_normalize_output_ref(args.particles, default_output_name="particles"), project_uid=project_uid)
    ref = CryoSPARCOutputRef(project_uid=ref.project_uid, job_uid=ref.job_uid, output_name=ref.output_name)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new filtering output directory: {output}")
    client = _connect(config)
    project = find_project(client, project_uid)
    otf_job = find_external_job(client, project, project_uid, otf_uid)
    card, path = read_card(_object_dir(otf_job, "job_dir"), project=project_uid, job_uid=otf_uid)
    index = Index(path, readonly=True)
    try:
        rows = index.rows()  # One committed snapshot; unfinished masks stay pending.
    finally:
        index.close()
    particle_job = find_job(client, project, project_uid, ref.job_uid)
    if str(particle_job.status) != "completed":
        raise ValueError("Choose a completed particle-picking job; the OTF mask job may still be running")
    particles, ref = _load_output_with_aliases(particle_job, ref, kind="particles")
    started = time.monotonic()
    decisions, summary = filter_by_index(particles, rows,
        exclusion_angstrom=args.particle_exclusion_distance_angstrom, missing_masks=args.missing_masks)
    summary.update(elapsed_seconds=time.monotonic() - started, otf_job=otf_uid, project=project_uid,
                   particle_job=ref.job_uid, exclusion_distance_angstrom=args.particle_exclusion_distance_angstrom,
                   index_path=str(path), masks_snapshot_time=time.time())
    output.mkdir(parents=True, exist_ok=False)
    # Persist decisions and provenance before publishing the separate output card.
    from cryofilter.typing_cli import _save_npy_atomic
    _save_npy_atomic(output / "particle_uids.npy", np.asarray(particles["uid"], dtype=np.uint64))
    _save_npy_atomic(output / "decisions.npy", decisions)
    write_json(output / "particle_filter_summary.json", summary)
    external = create_external_job(project, workspace_uid, title=args.title,
        desc=f"Particle masks from {otf_uid}; {summary['accepted']} accepted, {summary['rejected']} rejected, {summary['pending']} pending.")
    try:
        slots = dataset_prefixes(particles)
        connect_particle_input(external, input_name="input_particles", source_job_uid=ref.job_uid,
                               source_output_name=ref.output_name, slots=slots)
        # Reference the mask card in provenance without requiring it to finish.
        output_slots = ["location"]
        with run_external_job(external):
            for value, name in ((0, "particles_accepted"), (1, "particles_rejected"), (2, "particles_pending")):
                if value == 2 and args.missing_masks != "pending":
                    continue
                subset = dataset_take(particles, np.flatnonzero(decisions == value).tolist())
                add_particle_output(external, name=name, slots=output_slots,
                    title=name.replace("_", " ").capitalize(), passthrough="input_particles", alloc=subset)
                save_output(external, name, dataset_filter_prefixes(subset, output_slots))
            external.log(f"Masks: {project_uid}/{otf_uid}. Accepted {summary['accepted']}, rejected {summary['rejected']}, pending {summary['pending']}.")
    except BaseException as exc:
        stop_external_job(external, error=str(exc))
        raise
    summary["external_job_uid"] = object_uid(external)
    write_json(output / "particle_filter_summary.json", summary)
    print(f"Filtered picks: {project_uid}/{object_uid(external)} — {summary['accepted']} accepted, "
          f"{summary['rejected']} rejected, {summary['pending']} pending; {summary['elapsed_seconds']:.2f}s filtering.", flush=True)
    return 0
