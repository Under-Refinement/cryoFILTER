"""Prepare CryoSPARC inputs for cryoFILTER transfer and prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

from cryofilter.cryosparc.bridge.client import make_client
from cryofilter.cryosparc.bridge.compat import (
    connect_exposure_input,
    connect_particle_input,
    create_external_job,
    dataset_prefixes,
    find_job,
    find_project,
    object_uid,
    stop_external_job,
)
from cryofilter.cryosparc.protocol.manifests import write_json_model
from cryofilter.cryosparc.protocol.models import (
    CryoSPARCOutputRef,
    MicrographManifestEntry,
    ParticleManifestEntry,
    PredictPrepareRequest,
    PredictPrepareResponse,
    TransferManifest,
)

MICROGRAPH_PATH_FIELDS = (
    "micrograph_blob/path",
    "movie_blob/path",
    "micrograph_blob_non_dw/path",
    "blob/path",
    "location/micrograph_path",
)

MICROGRAPH_SHAPE_SUFFIXES = ("shape", "micrograph_shape")
MICROGRAPH_PIXEL_SIZE_SUFFIXES = ("psize_A", "micrograph_psize_A")
MICROGRAPH_OUTPUT_ALIASES = (
    "micrographs",
    "imported_micrographs",
    "exposures",
)
PARTICLE_OUTPUT_ALIASES = (
    "particles",
    "imported_particles",
    "picked_particles",
    "extracted_particles",
)


def _dataset_fields(dataset: object) -> list[str]:
    fields = getattr(dataset, "fields", None)
    if callable(fields):
        return [str(value) for value in fields()]
    dtype = getattr(dataset, "dtype", None)
    names = getattr(dtype, "names", None) or ()
    return [str(value) for value in names]


def _column(dataset: object, field: str):
    try:
        return dataset[field]
    except Exception as exc:
        raise ValueError(f"CryoSPARC dataset does not contain readable field {field!r}") from exc


def _value_at(dataset: object, field: str, index: int):
    value = _column(dataset, field)[index]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        try:
            value = tolist()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _first_existing_field(fields: Iterable[str], candidates: Iterable[str]) -> str | None:
    field_set = set(fields)
    for candidate in candidates:
        if candidate in field_set:
            return candidate
    return None


def _infer_path_field(fields: list[str], requested: str | None) -> str:
    if requested:
        if requested not in fields:
            raise ValueError(f"Requested micrograph path field is not present: {requested}")
        return requested
    field = _first_existing_field(fields, MICROGRAPH_PATH_FIELDS)
    if field is not None:
        return field
    for candidate in fields:
        lower = candidate.lower()
        if lower.endswith("/path") and "micrograph" in lower:
            return candidate
    raise ValueError("Could not find a micrograph path field in the CryoSPARC dataset")


def _prefix_for_field(field: str) -> str | None:
    if "/" not in field:
        return None
    return field.rsplit("/", 1)[0]


def _field_for_suffixes(fields: list[str], prefix: str | None, suffixes: Iterable[str]) -> str | None:
    candidates: list[str] = []
    if prefix:
        candidates.extend(f"{prefix}/{suffix}" for suffix in suffixes)
    candidates.extend(f"location/{suffix}" for suffix in suffixes)
    return _first_existing_field(fields, candidates)


def _shape_yx(dataset: object, field: str | None, index: int) -> tuple[int, int] | None:
    if field is None:
        return None
    value = _value_at(dataset, field, index)
    if value in (None, ""):
        return None
    values = list(value)
    if len(values) < 2:
        return None
    return (int(values[0]), int(values[1]))


def _optional_float(dataset: object, field: str | None, index: int) -> float | None:
    if field is None:
        return None
    value = _value_at(dataset, field, index)
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(dataset: object, field: str | None, index: int) -> int | None:
    if field is None:
        return None
    value = _value_at(dataset, field, index)
    if value in (None, ""):
        return None
    return int(value)


def _object_dir(obj: object, key: str) -> Path | None:
    value = getattr(obj, "dir", None)
    if value:
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if value:
            return Path(str(value)).expanduser()

    for attr_name in ("model", "doc"):
        attr = getattr(obj, attr_name, None)
        if attr is None:
            continue
        model_dump = getattr(attr, "model_dump", None)
        if callable(model_dump):
            attr = model_dump()
        if isinstance(attr, dict) and attr.get(key):
            return Path(str(attr[key])).expanduser()
    return None


def _job_output_names(job: object) -> list[str]:
    names: list[str] = []
    for attr_name in ("model", "doc"):
        doc = getattr(job, attr_name, None)
        if doc is None:
            continue
        model_dump = getattr(doc, "model_dump", None)
        if callable(model_dump):
            doc = model_dump()
        if not isinstance(doc, dict):
            continue
        for key in ("output_result_groups", "output_results", "outputs"):
            value = doc.get(key)
            if isinstance(value, dict):
                names.extend(str(name) for name in value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("output_name")
                        if name:
                            names.append(str(name))
    return sorted(dict.fromkeys(names))


def _output_name_matches_kind(name: str, kind: Literal["micrographs", "particles"]) -> bool:
    lower = name.lower()
    if kind == "micrographs":
        return "micrograph" in lower or "exposure" in lower
    return "particle" in lower


def _candidate_output_names(
    requested: str,
    *,
    kind: Literal["micrographs", "particles"],
    available: Iterable[str],
) -> list[str]:
    aliases = MICROGRAPH_OUTPUT_ALIASES if kind == "micrographs" else PARTICLE_OUTPUT_ALIASES
    candidates = [requested, *aliases]
    candidates.extend(name for name in available if _output_name_matches_kind(name, kind))
    return list(dict.fromkeys(candidates))


def _ref_with_output_name(ref: CryoSPARCOutputRef, output_name: str) -> CryoSPARCOutputRef:
    return ref.model_copy(update={"output_name": output_name})


def _load_output_with_aliases(
    job: object,
    ref: CryoSPARCOutputRef,
    *,
    kind: Literal["micrographs", "particles"],
) -> tuple[object, CryoSPARCOutputRef]:
    available = _job_output_names(job)
    attempts: list[str] = []
    errors: list[str] = []
    for output_name in _candidate_output_names(ref.output_name, kind=kind, available=available):
        attempts.append(output_name)
        try:
            return job.load_output(output_name), _ref_with_output_name(ref, output_name)
        except Exception as exc:
            errors.append(f"{output_name}: {type(exc).__name__}: {exc}")

    available_text = ", ".join(available) if available else "not discoverable"
    last_error = errors[-1] if errors else "none"
    raise ValueError(
        f"Could not load CryoSPARC {kind} output {ref.job_uid}:{ref.output_name}. "
        f"Tried: {', '.join(attempts)}. Available output names: {available_text}. "
        f"Last error: {last_error}"
    )


def _resolve_project_file(project_dir: Path, dataset_path: str) -> tuple[Path, str | None]:
    raw_path = str(dataset_path)
    path = Path(raw_path)
    if path.is_absolute():
        return path, None
    return project_dir / raw_path, raw_path


def _safe_transfer_name(uid: int, source_path: Path) -> str:
    basename = source_path.name or f"{uid}.mrc"
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename)
    return f"{uid}_{basename}"


def _stage_micrographs(
    *,
    project_dir: Path,
    dataset: object,
    request: PredictPrepareRequest,
    transfer_dir: Path,
) -> list[MicrographManifestEntry]:
    fields = _dataset_fields(dataset)
    path_field = _infer_path_field(fields, request.micrograph_path_field)
    blob_prefix = _prefix_for_field(path_field)
    shape_field = _field_for_suffixes(fields, blob_prefix, MICROGRAPH_SHAPE_SUFFIXES)
    psize_field = _field_for_suffixes(fields, blob_prefix, MICROGRAPH_PIXEL_SIZE_SUFFIXES)
    idx_field = f"{blob_prefix}/idx" if blob_prefix and f"{blob_prefix}/idx" in fields else None
    total = len(dataset)  # type: ignore[arg-type]
    limit = total if request.limit_micrographs is None else min(total, request.limit_micrographs)
    micrograph_dir = transfer_dir / "micrographs"
    if request.stage_micrographs:
        micrograph_dir.mkdir(parents=True, exist_ok=True)

    entries: list[MicrographManifestEntry] = []
    for index in range(limit):
        uid = int(_value_at(dataset, "uid", index))
        dataset_path = str(_value_at(dataset, path_field, index))
        source_path, relative_path = _resolve_project_file(project_dir, dataset_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Staged micrograph source does not exist: {source_path}")
        transfer_filename = str(Path("micrographs") / _safe_transfer_name(uid, source_path))
        staged_path = transfer_dir / transfer_filename
        remote_transfer_path = None
        if request.stage_micrographs:
            if staged_path.exists() or staged_path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing staged path: {staged_path}")
            staged_path.symlink_to(source_path)
            remote_transfer_path = str(staged_path)
        file_size = int(source_path.stat().st_size)
        entries.append(
            MicrographManifestEntry(
                uid=uid,
                transfer_filename=transfer_filename,
                source_path=str(source_path),
                source_relative_path=relative_path,
                remote_transfer_path=remote_transfer_path,
                source_index=_optional_int(dataset, idx_field, index),
                blob_prefix=blob_prefix,
                shape_yx=_shape_yx(dataset, shape_field, index),
                pixel_size_angstrom=_optional_float(dataset, psize_field, index),
                file_size_bytes=file_size,
            )
        )
    return entries


def _stage_particles(
    *,
    dataset: object,
    micrograph_uids: Iterable[int],
) -> list[ParticleManifestEntry]:
    fields = _dataset_fields(dataset)
    required = {
        "uid": "uid",
        "micrograph_uid": "location/micrograph_uid",
        "center_x_frac": "location/center_x_frac",
        "center_y_frac": "location/center_y_frac",
    }
    missing = [field for field in required.values() if field not in fields]
    if missing:
        raise ValueError(f"Particle dataset is missing required location fields: {missing}")

    selected = set(int(uid) for uid in micrograph_uids)
    entries: list[ParticleManifestEntry] = []
    for index in range(len(dataset)):  # type: ignore[arg-type]
        micrograph_uid = int(_value_at(dataset, required["micrograph_uid"], index))
        if micrograph_uid not in selected:
            continue
        entries.append(
            ParticleManifestEntry(
                uid=int(_value_at(dataset, required["uid"], index)),
                micrograph_uid=micrograph_uid,
                center_x_frac=float(_value_at(dataset, required["center_x_frac"], index)),
                center_y_frac=float(_value_at(dataset, required["center_y_frac"], index)),
            )
        )
    return entries


def run_prepare(
    request: PredictPrepareRequest,
    *,
    connection_file: str | None = None,
) -> PredictPrepareResponse:
    """Create a staging area and optional External Job for a cryoFILTER run."""

    client = make_client(connection_file)
    project = find_project(client, request.project_uid)
    project_dir = _object_dir(project, "project_dir")
    if project_dir is None:
        raise ValueError(f"Could not determine project directory for {request.project_uid}")

    micrograph_job = find_job(
        client,
        project,
        request.project_uid,
        request.micrographs.job_uid,
    )
    micrographs, micrographs_ref = _load_output_with_aliases(
        micrograph_job,
        request.micrographs,
        kind="micrographs",
    )
    n_micrographs_total = int(len(micrographs))

    remote_run_dir = Path(request.remote_work_root).expanduser() / str(request.run_id)
    transfer_dir = remote_run_dir / "transfer"
    if remote_run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing remote run directory: {remote_run_dir}")
    transfer_dir.mkdir(parents=True, exist_ok=True)
    micrograph_entries = _stage_micrographs(
        project_dir=project_dir,
        dataset=micrographs,
        request=request,
        transfer_dir=transfer_dir,
    )

    particle_entries: list[ParticleManifestEntry] = []
    particle_dataset = None
    particles_ref = request.particles
    if request.particles is not None:
        particle_job = find_job(
            client,
            project,
            request.project_uid,
            request.particles.job_uid,
        )
        particle_dataset, particles_ref = _load_output_with_aliases(
            particle_job,
            request.particles,
            kind="particles",
        )
        particle_entries = _stage_particles(
            dataset=particle_dataset,
            micrograph_uids=[entry.uid for entry in micrograph_entries],
        )

    external_job_uid: str | None = None
    external_job = None
    try:
        if request.create_external_job:
            external_job = create_external_job(
                project,
                request.workspace_uid,
                title=f"{request.title} ({request.run_id})",
                desc=(
                    "cryoFILTER remote prediction job prepared by the bridge. "
                    "Inputs are staged for external inference and finalization."
                ),
            )
            external_job_uid = object_uid(external_job)
            connect_exposure_input(
                external_job,
                input_name="input_micrographs",
                source_job_uid=micrographs_ref.job_uid,
                source_output_name=micrographs_ref.output_name,
                slots=dataset_prefixes(micrographs),
            )
            if request.particles is not None and particle_dataset is not None:
                connect_particle_input(
                    external_job,
                    input_name="input_particles",
                    source_job_uid=particles_ref.job_uid,
                    source_output_name=particles_ref.output_name,
                    slots=dataset_prefixes(particle_dataset),
                )
    except Exception as exc:
        if external_job is not None:
            stop_external_job(external_job, error=str(exc))
        raise

    manifest_file = remote_run_dir / "transfer_manifest.json"
    manifest = TransferManifest(
        run_id=request.run_id,
        project_uid=request.project_uid,
        workspace_uid=request.workspace_uid,
        external_job_uid=external_job_uid,
        micrographs_ref=micrographs_ref,
        particles_ref=particles_ref,
        micrographs=micrograph_entries,
        particles=particle_entries,
    )
    write_json_model(manifest, manifest_file)

    return PredictPrepareResponse(
        ok=True,
        run_id=request.run_id,
        external_job_uid=external_job_uid,
        remote_run_dir=str(remote_run_dir),
        transfer_dir=str(transfer_dir),
        manifest_file=str(manifest_file),
        n_micrographs=len(micrograph_entries),
        n_micrographs_total=n_micrographs_total,
        n_particles=len(particle_entries) if request.particles is not None else None,
        total_micrograph_bytes=sum(
            int(entry.file_size_bytes or 0) for entry in micrograph_entries
        ),
        errors=[],
    )


__all__ = ["run_prepare"]
