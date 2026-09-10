"""Local helpers for orchestrating CryoSPARC-backed cryoFILTER predictions."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cryofilter.cryosparc import PROTOCOL_VERSION
from cryofilter.cryosparc.protocol.validation import validate_uid_partition

INTERNAL_MAP_OUTPUT_SUBDIR = ".cryofilter_internal_maps"


def load_transfer_manifest(path: str | Path) -> dict[str, Any]:
    """Load a transfer manifest without requiring Pydantic in lightweight test shells."""

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Transfer manifest must be a JSON object: {path}")
    return data


def _manifest_micrograph_path(local_transfer_dir: Path, entry: dict[str, Any]) -> Path:
    return (local_transfer_dir / str(entry["transfer_filename"])).expanduser().resolve()


def _mask_output_path(mask_dir: Path, stem: str) -> Path:
    direct_path = mask_dir / f"{stem}_mask.npy"
    if direct_path.exists():
        return direct_path
    internal_path = mask_dir / INTERNAL_MAP_OUTPUT_SUBDIR / f"{stem}_mask.npy"
    if internal_path.exists():
        return internal_path
    return direct_path


def _read_mrc_geometry(path: Path) -> tuple[tuple[int, int], float | None]:
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as mrc:
        shape = tuple(int(value) for value in mrc.data.shape)
        try:
            pixel_size = float(mrc.voxel_size.x)
        except Exception:
            pixel_size = float("nan")
    if len(shape) == 2:
        shape_yx = (int(shape[0]), int(shape[1]))
    elif len(shape) == 3 and shape[0] > 0:
        shape_yx = (int(shape[-2]), int(shape[-1]))
    else:
        raise ValueError(f"Unsupported MRC shape for {path}: {shape}")
    if pixel_size is None or not np.isfinite(float(pixel_size)) or float(pixel_size) <= 0:
        pixel_size_value = None
    else:
        pixel_size_value = float(pixel_size)
    return shape_yx, pixel_size_value


def _shape_yx(entry: dict[str, Any], local_path: Path) -> tuple[int, int]:
    raw_shape = entry.get("shape_yx")
    if isinstance(raw_shape, (list, tuple)) and len(raw_shape) >= 2:
        return (int(raw_shape[0]), int(raw_shape[1]))
    shape_yx, _ = _read_mrc_geometry(local_path)
    return shape_yx


def _pixel_size_angstrom(entry: dict[str, Any], local_path: Path) -> float:
    raw_pixel_size = entry.get("pixel_size_angstrom")
    if raw_pixel_size is not None:
        value = float(raw_pixel_size)
        if value > 0:
            return value
    _, pixel_size = _read_mrc_geometry(local_path)
    if pixel_size is None:
        raise ValueError(f"Could not determine pixel size for staged micrograph: {local_path}")
    return float(pixel_size)


def _star_value(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    if any(char.isspace() for char in text) or text.startswith("#") or text.startswith("_"):
        return "'" + text.replace("'", "\\'") + "'"
    return text


def write_particle_star_from_manifest(
    *,
    transfer_manifest_file: str | Path,
    local_transfer_dir: str | Path,
    star_path: str | Path,
) -> dict[str, Any]:
    """Write a RELION-style coordinate STAR file from staged CryoSPARC particles."""

    manifest = load_transfer_manifest(transfer_manifest_file)
    transfer_dir = Path(local_transfer_dir).expanduser().resolve()
    output_path = Path(star_path).expanduser().resolve()
    micrographs = manifest.get("micrographs")
    particles = manifest.get("particles")
    if not isinstance(micrographs, list) or not micrographs:
        raise ValueError("Transfer manifest contains no micrographs")
    if not isinstance(particles, list) or not particles:
        raise ValueError("Transfer manifest contains no particles")

    micrograph_by_uid: dict[int, dict[str, Any]] = {}
    for entry in micrographs:
        if not isinstance(entry, dict):
            continue
        micrograph_by_uid[int(entry["uid"])] = entry

    lines = [
        "data_\n",
        "\n",
        "loop_\n",
        "_rlnMicrographName #1\n",
        "_rlnCoordinateX #2\n",
        "_rlnCoordinateY #3\n",
    ]
    written = 0
    missing_micrographs: set[int] = set()
    for particle in particles:
        if not isinstance(particle, dict):
            continue
        micrograph_uid = int(particle["micrograph_uid"])
        entry = micrograph_by_uid.get(micrograph_uid)
        if entry is None:
            missing_micrographs.add(micrograph_uid)
            continue
        local_path = _manifest_micrograph_path(transfer_dir, entry)
        height, width = _shape_yx(entry, local_path)
        x = float(particle["center_x_frac"]) * float(width) + 1.0
        y = float(particle["center_y_frac"]) * float(height) + 1.0
        lines.append(
            f"{_star_value(local_path)} {x:.6f} {y:.6f}\n"
        )
        written += 1

    if missing_micrographs:
        examples = ", ".join(str(uid) for uid in sorted(missing_micrographs)[:10])
        raise ValueError(f"Particles refer to unstaged micrograph UID(s): {examples}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return {
        "particle_star": str(output_path),
        "particles_written": int(written),
        "micrographs": int(len(micrograph_by_uid)),
    }


def run_local_inference(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path | None,
    threshold: float,
    particle_star: str | Path,
    exclusion_distance_angstrom: float,
    extra_args: Sequence[str] = (),
    timeout: float | None = None,
    env_overrides: Mapping[str, str] | None = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    render_particle_overlays: bool = True,
    poll_callback: Callable[[], None] | None = None,
    poll_interval: float = 5.0,
) -> list[str]:
    """Run cryoFILTER inference, optionally deferring OTF overlays until typing finishes."""

    command = [
        sys.executable,
        "-m",
        "cryofilter.cli",
        "infer",
        "--input",
        str(Path(input_dir).expanduser().resolve()),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--threshold",
        str(float(threshold)),
        "--particle-file",
        str(Path(particle_star).expanduser().resolve()),
        "--overwrite-filtered-particles",
        "--particle-exclusion-distance-angstrom",
        str(float(exclusion_distance_angstrom)),
    ]
    if render_particle_overlays:
        command.extend(["--render-particle-overlays", "--no-particle-overlay-contact-sheet"])
    else:
        command.append("--no-render-particle-overlays")
    if checkpoint is not None:
        command.extend(["--checkpoint", str(Path(checkpoint).expanduser())])
    if num_cpus is not None:
        command.extend(["--num-cpus", str(int(num_cpus))])
    if num_gpus is not None:
        command.extend(["--num-gpus", str(int(num_gpus))])
    forwarded = list(extra_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command.extend(str(value) for value in forwarded)
    print("Executing:", " ".join(shlex.quote(part) for part in command), flush=True)
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    if poll_callback is None:
        completed = subprocess.run(command, timeout=timeout, env=env)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        return command

    process = subprocess.Popen(command, env=env)
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    last_poll = 0.0
    try:
        while True:
            returncode = process.poll()
            now = time.monotonic()
            if now - last_poll >= max(0.5, float(poll_interval)):
                try:
                    poll_callback()
                except Exception as exc:
                    print(f"Warning: live typing update skipped: {type(exc).__name__}: {exc}", flush=True)
                last_poll = now
            if returncode is not None:
                break
            if deadline is not None and now >= deadline:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(0.5)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    try:
        poll_callback()
    except Exception as exc:
        print(f"Warning: final live typing update skipped: {type(exc).__name__}: {exc}", flush=True)
    return command


def write_typing_manifest_from_transfer(
    *,
    transfer_manifest_file: str | Path,
    local_transfer_dir: str | Path,
    inference_dir: str | Path,
    manifest_path: str | Path,
    dataset_id: str | None = None,
    require_all_masks: bool = True,
) -> dict[str, Any]:
    """Write a contamination typing CSV manifest from staged micrographs and masks."""

    manifest = load_transfer_manifest(transfer_manifest_file)
    transfer_dir = Path(local_transfer_dir).expanduser().resolve()
    mask_dir = Path(inference_dir).expanduser().resolve()
    output_path = Path(manifest_path).expanduser().resolve()
    micrographs = manifest.get("micrographs")
    if not isinstance(micrographs, list) or not micrographs:
        raise ValueError("Transfer manifest contains no micrographs")

    rows: list[dict[str, object]] = []
    missing_masks: list[str] = []
    resolved_dataset_id = dataset_id or str(manifest.get("workspace_uid") or "cryosparc")
    for entry in micrographs:
        if not isinstance(entry, dict):
            continue
        local_path = _manifest_micrograph_path(transfer_dir, entry)
        mask_path = _mask_output_path(mask_dir, local_path.stem)
        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue
        rows.append(
            {
                "dataset_id": resolved_dataset_id,
                "stem": local_path.stem,
                "micrograph_path": str(local_path),
                "binary_mask_path": str(mask_path),
                "pixel_size_angstrom": _pixel_size_angstrom(entry, local_path),
            }
        )

    if missing_masks and require_all_masks:
        examples = ", ".join(missing_masks[:3])
        raise FileNotFoundError(f"Typing masks were not found for staged micrographs: {examples}")
    if not rows:
        raise ValueError("No rows were written to the contamination typing manifest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "stem",
                "micrograph_path",
                "binary_mask_path",
                "pixel_size_angstrom",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "typing_manifest": str(output_path),
        "images": int(len(rows)),
        "images_expected": int(len(micrographs)),
        "missing_masks": int(len(missing_masks)),
        "complete": int(len(missing_masks)) == 0,
        "dataset_id": resolved_dataset_id,
    }


def run_local_typing(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    normalization_method: str | None = None,
    min_component_area_px: int | None = None,
    pixel_size_angstrom: float | None = None,
    expected_images: int | None = None,
    timeout: float | None = None,
    env_overrides: Mapping[str, str] | None = None,
    poll_callback: Callable[[], None] | None = None,
    poll_interval: float = 5.0,
) -> list[str]:
    """Run cryoFILTER contamination typing on inference masks."""

    command = [
        sys.executable,
        "-m",
        "cryofilter.cli",
        "type",
        "--manifest",
        str(Path(manifest_path).expanduser().resolve()),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
    ]
    if normalization_method:
        command.extend(["--normalization-method", str(normalization_method)])
    if min_component_area_px is not None:
        command.extend(["--min-component-area-px", str(int(min_component_area_px))])
    if pixel_size_angstrom is not None:
        command.extend(["--pixel-size-angstrom", str(float(pixel_size_angstrom))])
    if expected_images is not None:
        command.extend(["--expected-images", str(int(expected_images))])
    print(
        "Running contamination typing; 4-panel OTF images will refresh as typing outputs are written.",
        flush=True,
    )
    print("Executing:", " ".join(shlex.quote(part) for part in command), flush=True)
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    if poll_callback is None:
        completed = subprocess.run(command, timeout=timeout, env=env)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        return command

    process = subprocess.Popen(command, env=env)
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    last_poll = 0.0
    try:
        while True:
            returncode = process.poll()
            now = time.monotonic()
            if now - last_poll >= max(0.5, float(poll_interval)):
                try:
                    poll_callback()
                except Exception as exc:
                    print(f"Warning: live typed OTF refresh skipped: {type(exc).__name__}: {exc}", flush=True)
                last_poll = now
            if returncode is not None:
                break
            if deadline is not None and now >= deadline:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(0.5)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    try:
        poll_callback()
    except Exception as exc:
        print(f"Warning: final live typed OTF refresh skipped: {type(exc).__name__}: {exc}", flush=True)
    return command


def classify_particles_from_manifest(
    *,
    transfer_manifest_file: str | Path,
    local_transfer_dir: str | Path,
    inference_dir: str | Path,
    exclusion_distance_angstrom: float,
) -> dict[str, Any]:
    """Classify staged CryoSPARC particle UIDs from cryoFILTER mask outputs."""

    import numpy as np

    from cryofilter.particle_filtering import classify_particle_coordinates_by_mask

    manifest = load_transfer_manifest(transfer_manifest_file)
    transfer_dir = Path(local_transfer_dir).expanduser().resolve()
    mask_dir = Path(inference_dir).expanduser().resolve()
    micrographs = manifest.get("micrographs")
    particles = manifest.get("particles")
    if not isinstance(micrographs, list) or not micrographs:
        raise ValueError("Transfer manifest contains no micrographs")
    if not isinstance(particles, list) or not particles:
        raise ValueError("Transfer manifest contains no particles")

    micrograph_by_uid: dict[int, dict[str, Any]] = {
        int(entry["uid"]): entry for entry in micrographs if isinstance(entry, dict)
    }
    particles_by_micrograph: dict[int, list[dict[str, Any]]] = {}
    for particle in particles:
        if not isinstance(particle, dict):
            continue
        particles_by_micrograph.setdefault(int(particle["micrograph_uid"]), []).append(particle)

    accepted_uids: list[int] = []
    rejected_uids: list[int] = []
    per_micrograph: list[dict[str, Any]] = []
    distance_values: list[float] = []

    for micrograph_uid, entry in sorted(micrograph_by_uid.items()):
        micrograph_particles = particles_by_micrograph.get(micrograph_uid, [])
        if not micrograph_particles:
            continue
        local_path = _manifest_micrograph_path(transfer_dir, entry)
        mask_path = _mask_output_path(mask_dir, local_path.stem)
        if not mask_path.exists():
            raise FileNotFoundError(f"Final cryoFILTER mask not found: {mask_path}")
        height, width = _shape_yx(entry, local_path)
        pixel_size = _pixel_size_angstrom(entry, local_path)
        coords = np.asarray(
            [
                (
                    float(particle["center_x_frac"]) * float(width),
                    float(particle["center_y_frac"]) * float(height),
                )
                for particle in micrograph_particles
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        mask = np.asarray(np.load(mask_path), dtype=bool)
        keep, distances = classify_particle_coordinates_by_mask(
            coords,
            bad_mask=mask,
            input_shape=(int(height), int(width)),
            pixel_size_angstrom=float(pixel_size),
            exclusion_distance_angstrom=float(exclusion_distance_angstrom),
        )
        kept = 0
        removed = 0
        for particle, keep_value, distance in zip(micrograph_particles, keep, distances):
            uid = int(particle["uid"])
            distance_values.append(float(distance))
            if bool(keep_value):
                accepted_uids.append(uid)
                kept += 1
            else:
                rejected_uids.append(uid)
                removed += 1
        per_micrograph.append(
            {
                "micrograph_uid": int(micrograph_uid),
                "micrograph": str(local_path),
                "mask": str(mask_path),
                "particles": int(len(micrograph_particles)),
                "accepted_particles": int(kept),
                "rejected_particles": int(removed),
                "rejected_fraction": float(removed / max(len(micrograph_particles), 1)),
            }
        )

    staged_uids = [int(particle["uid"]) for particle in particles if isinstance(particle, dict)]
    report = validate_uid_partition(
        input_uids=staged_uids,
        accepted_uids=accepted_uids,
        rejected_uids=rejected_uids,
    )
    report.require_ok()
    finite_distances = [value for value in distance_values if np.isfinite(value)]
    return {
        "particles_processed": int(len(staged_uids)),
        "accepted_uids": accepted_uids,
        "rejected_uids": rejected_uids,
        "accepted_particles": int(len(accepted_uids)),
        "rejected_particles": int(len(rejected_uids)),
        "per_micrograph": per_micrograph,
        "min_distance_angstrom": min(finite_distances) if finite_distances else None,
        "median_distance_angstrom": (
            float(np.median(np.asarray(finite_distances, dtype=np.float32)))
            if finite_distances
            else None
        ),
    }


def write_uid_file(
    *,
    path: str | Path,
    run_id: str,
    kind: str,
    uids: Sequence[int],
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": str(run_id),
        "kind": str(kind),
        "uids": [int(uid) for uid in uids],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_result_manifest_payload(
    *,
    path: str | Path,
    run_id: str,
    project_uid: str,
    workspace_uid: str,
    external_job_uid: str,
    transfer_manifest_file: str,
    status: str,
    error: str | None,
    model: dict[str, Any],
    micrographs_processed: int,
    particles_processed: int,
    accepted_particles: int,
    rejected_particles: int,
    files: dict[str, str],
    diagnostics: Sequence[dict[str, Any]] = (),
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": str(run_id),
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "external_job_uid": external_job_uid,
        "transfer_manifest_file": transfer_manifest_file,
        "status": status,
        "error": error,
        "model": dict(model),
        "micrographs_processed": int(micrographs_processed),
        "particles_processed": int(particles_processed),
        "accepted_particles": int(accepted_particles),
        "rejected_particles": int(rejected_particles),
        "files": dict(files),
        "diagnostics": [dict(asset) for asset in diagnostics],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "classify_particles_from_manifest",
    "load_transfer_manifest",
    "run_local_inference",
    "run_local_typing",
    "write_particle_star_from_manifest",
    "write_result_manifest_payload",
    "write_typing_manifest_from_transfer",
    "write_uid_file",
]
