"""Filter particle coordinates against final cryoFILTER contamination masks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import mrcfile
import numpy as np
import yaml
from scipy import ndimage as ndi

from utils.file_io import load_cs, save_cs
from utils.star_import import _find_micrograph_loop, _format_star_value, strip_cryosparc_uid_prefix


SUPPORTED_PARTICLE_SUFFIXES = (".cs", ".star")
REQUIRED_CRYOSPARC_LOCATION_FIELDS = {
    "location/micrograph_path",
    "location/center_x_frac",
    "location/center_y_frac",
}


@dataclass(frozen=True)
class _MaskSource:
    micrograph_path: Path
    mask_path: Path
    input_shape: tuple[int, int]
    mask_shape: tuple[int, int]
    pixel_size_angstrom: float


@dataclass(frozen=True)
class _MicrographSource:
    micrograph_path: Path
    input_shape: tuple[int, int]
    pixel_size_angstrom: float


def _load_micrograph_geometry(
    path: Path,
    *,
    pixel_size_override_angstrom: Optional[float],
) -> tuple[tuple[int, int], float]:
    with mrcfile.open(path, permissive=True) as mrc:
        shape = tuple(int(value) for value in mrc.data.shape)
        try:
            header_pixel_size = float(mrc.voxel_size.x)
        except Exception:
            header_pixel_size = float("nan")

    if len(shape) == 2:
        image_shape = shape
    elif len(shape) == 3 and shape[0] > 0:
        image_shape = shape[-2:]
    else:
        raise ValueError(f"Unsupported MRC dimensionality for particle filtering: {path} shape={shape}")

    pixel_size = pixel_size_override_angstrom
    if pixel_size is None:
        pixel_size = header_pixel_size
    if pixel_size is None or not np.isfinite(float(pixel_size)) or float(pixel_size) <= 0:
        raise ValueError(
            f"Could not determine a valid pixel size for {path}. "
            "Repair the MRC header or pass --pixel-size-angstrom."
        )
    return (int(image_shape[0]), int(image_shape[1])), float(pixel_size)


def _path_aliases(raw_path: str | Path) -> set[str]:
    basename = Path(str(raw_path)).name
    stripped, _ = strip_cryosparc_uid_prefix(basename)
    aliases = {basename, stripped, Path(basename).stem, Path(stripped).stem}
    for name in tuple(aliases):
        stem = Path(name).stem
        if stem.endswith("_particles"):
            aliases.add(stem[: -len("_particles")])
            aliases.add(stem[: -len("_particles")] + ".mrc")
    return {alias for alias in aliases if alias}


def _build_mask_sources(
    *,
    micrograph_paths: Sequence[str | Path],
    mask_dir: str | Path,
    pixel_size_override_angstrom: Optional[float],
) -> tuple[list[_MaskSource], dict[str, _MaskSource], set[str]]:
    mask_root = Path(mask_dir)
    sources: list[_MaskSource] = []
    aliases: dict[str, _MaskSource] = {}
    ambiguous: set[str] = set()

    for raw_path in micrograph_paths:
        micrograph_path = Path(raw_path).expanduser().resolve()
        mask_path = mask_root / f"{micrograph_path.stem}_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Final contamination mask not found for particle filtering: {mask_path}"
            )
        input_shape, pixel_size = _load_micrograph_geometry(
            micrograph_path,
            pixel_size_override_angstrom=pixel_size_override_angstrom,
        )
        mask = np.load(mask_path, mmap_mode="r")
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask at {mask_path}, got shape={mask.shape}")
        source = _MaskSource(
            micrograph_path=micrograph_path,
            mask_path=mask_path.resolve(),
            input_shape=input_shape,
            mask_shape=(int(mask.shape[0]), int(mask.shape[1])),
            pixel_size_angstrom=pixel_size,
        )
        sources.append(source)

        for alias in _path_aliases(micrograph_path):
            previous = aliases.get(alias)
            if previous is not None and previous.micrograph_path != source.micrograph_path:
                ambiguous.add(alias)
            else:
                aliases[alias] = source

    for alias in ambiguous:
        aliases.pop(alias, None)
    return sources, aliases, ambiguous


def _build_micrograph_sources(
    *,
    micrograph_paths: Sequence[str | Path],
    pixel_size_override_angstrom: Optional[float],
) -> tuple[list[_MicrographSource], dict[str, _MicrographSource], set[str]]:
    sources: list[_MicrographSource] = []
    aliases: dict[str, _MicrographSource] = {}
    ambiguous: set[str] = set()

    for raw_path in micrograph_paths:
        micrograph_path = Path(raw_path).expanduser().resolve()
        input_shape, pixel_size = _load_micrograph_geometry(
            micrograph_path,
            pixel_size_override_angstrom=pixel_size_override_angstrom,
        )
        source = _MicrographSource(
            micrograph_path=micrograph_path,
            input_shape=input_shape,
            pixel_size_angstrom=pixel_size,
        )
        sources.append(source)

        for alias in _path_aliases(micrograph_path):
            previous = aliases.get(alias)
            if previous is not None and previous.micrograph_path != source.micrograph_path:
                ambiguous.add(alias)
            else:
                aliases[alias] = source

    for alias in ambiguous:
        aliases.pop(alias, None)
    return sources, aliases, ambiguous


def _resolve_source(
    raw_micrograph_path: str,
    aliases: dict[str, Any],
    ambiguous_aliases: set[str],
) -> Optional[Any]:
    candidates = _path_aliases(raw_micrograph_path)
    if candidates & ambiguous_aliases:
        return None
    resolved = {aliases[alias] for alias in candidates if alias in aliases}
    if len(resolved) == 1:
        return next(iter(resolved))
    return None


def _distance_map_angstrom(source: _MaskSource) -> np.ndarray:
    mask = np.asarray(np.load(source.mask_path), dtype=bool)
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    input_h, input_w = source.input_shape
    mask_h, mask_w = source.mask_shape
    sampling_y = float(source.pixel_size_angstrom) * float(input_h) / float(mask_h)
    sampling_x = float(source.pixel_size_angstrom) * float(input_w) / float(mask_w)
    return ndi.distance_transform_edt(~mask, sampling=(sampling_y, sampling_x)).astype(
        np.float32,
        copy=False,
    )


def _distance_for_input_coordinate(
    distance_map: np.ndarray,
    *,
    x_input_px: float,
    y_input_px: float,
    input_shape: tuple[int, int],
) -> float:
    input_h, input_w = input_shape
    mask_h, mask_w = distance_map.shape
    x_mask = int(np.clip(np.floor(float(x_input_px) * mask_w / max(input_w, 1)), 0, mask_w - 1))
    y_mask = int(np.clip(np.floor(float(y_input_px) * mask_h / max(input_h, 1)), 0, mask_h - 1))
    return float(distance_map[y_mask, x_mask])


def _new_counts(sources: Iterable[_MaskSource]) -> dict[Path, dict[str, int]]:
    return {
        source.micrograph_path: {"particles": 0, "removed": 0, "kept": 0}
        for source in sources
    }


def _counts_payload(counts: dict[Path, dict[str, int]]) -> list[dict[str, Any]]:
    return [
        {"micrograph": str(path), **values}
        for path, values in sorted(counts.items(), key=lambda item: str(item[0]))
        if int(values["particles"]) > 0
    ]


def _strip_csg_metafile_prefix(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith(">"):
        text = text[1:]
    return text.strip()


def _strip_cryosparc_export_prefix(filename: str) -> str:
    return re.sub(r"^cryosparc_P\d+_", "", str(filename))


def _resolve_csg_metafile_path(
    csg_path: Path,
    metafile: object,
    *,
    known_paths: Sequence[Path] = (),
) -> Path:
    raw = _strip_csg_metafile_prefix(metafile)
    if not raw:
        raise ValueError(f"Empty metafile entry in CryoSPARC .csg file: {csg_path}")

    raw_path = Path(raw).expanduser()
    candidate = raw_path if raw_path.is_absolute() else csg_path.parent / raw_path
    if candidate.exists():
        return candidate.resolve()

    basename = raw_path.name
    for known_path in known_paths:
        known = Path(known_path).expanduser().resolve()
        if known.name == basename or _strip_cryosparc_export_prefix(known.name) == basename:
            return known

    matches = [
        path.resolve()
        for path in csg_path.parent.glob("*.cs")
        if path.name == basename or path.name.endswith(basename)
    ]
    exact_or_export_matches = [
        path
        for path in matches
        if path.name == basename or _strip_cryosparc_export_prefix(path.name) == basename
    ]
    if len(exact_or_export_matches) == 1:
        return exact_or_export_matches[0]
    if len(exact_or_export_matches) > 1:
        matches = exact_or_export_matches
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"CryoSPARC .csg metafile {basename!r} is ambiguous in {csg_path.parent}: "
            + ", ".join(str(path.name) for path in matches[:5])
        )

    raise FileNotFoundError(
        f"CryoSPARC .csg metafile was not found: {candidate}. "
        "If your exported .cs files were renamed, pass one of the matching .cs files with "
        "--particle-file and keep the other exported .cs files next to the .csg."
    )


def _load_csg_group_metafiles(csg_path: Path, *, known_paths: Sequence[Path] = ()) -> dict[str, Path]:
    csg_data = yaml.safe_load(csg_path.read_text(encoding="utf-8"))
    if not isinstance(csg_data, dict):
        raise ValueError(f"Expected a mapping in CryoSPARC .csg file: {csg_path}")
    results = csg_data.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"CryoSPARC .csg file does not contain result groups: {csg_path}")

    group_to_file: dict[str, Path] = {}
    for group_name, group_data in results.items():
        if not isinstance(group_data, dict) or "metafile" not in group_data:
            continue
        path = _resolve_csg_metafile_path(
            csg_path,
            group_data["metafile"],
            known_paths=known_paths,
        )
        if path.suffix.lower() == ".cs":
            group_to_file[str(group_name)] = path

    if not group_to_file:
        raise ValueError(f"CryoSPARC .csg file does not reference any .cs metafiles: {csg_path}")
    return group_to_file


def _cs_has_location_fields(cs_data: np.ndarray) -> bool:
    field_names = set(cs_data.dtype.names or ())
    return REQUIRED_CRYOSPARC_LOCATION_FIELDS.issubset(field_names)


def _missing_location_fields(cs_data: np.ndarray) -> list[str]:
    field_names = set(cs_data.dtype.names or ())
    return sorted(REQUIRED_CRYOSPARC_LOCATION_FIELDS - field_names)


def _csg_member_output_paths(
    *,
    group_to_input_file: dict[str, Path],
    primary_input_path: Path,
    primary_output_path: Path,
) -> dict[Path, Path]:
    input_to_output: dict[Path, Path] = {}
    for member_path in dict.fromkeys(group_to_input_file.values()):
        if member_path == primary_input_path:
            output_path = primary_output_path
        else:
            output_path = primary_output_path.parent / f"{member_path.stem}_cryofiltered{member_path.suffix.lower()}"
        previous = input_to_output.get(member_path)
        if previous is not None and previous != output_path:
            raise ValueError(f"Conflicting filtered output paths for {member_path}: {previous} and {output_path}")
        input_to_output[member_path] = output_path

    if primary_input_path not in input_to_output:
        input_to_output[primary_input_path] = primary_output_path
    return input_to_output


def _relativize_for_csg(file_path: Path, output_csg_path: Path) -> str:
    file_abs_path = Path(file_path).expanduser().resolve()
    output_csg_dir = output_csg_path.parent.resolve()
    try:
        return os.path.relpath(file_abs_path, output_csg_dir)
    except ValueError:
        return str(file_abs_path)


def _write_filtered_csg(
    *,
    original_csg_path: Path,
    output_csg_path: Path,
    group_to_output_file: dict[str, Path],
    num_items: int,
) -> None:
    csg_data = yaml.safe_load(original_csg_path.read_text(encoding="utf-8"))
    if not isinstance(csg_data, dict):
        raise ValueError(f"Expected a mapping in CryoSPARC .csg file: {original_csg_path}")
    results = csg_data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"CryoSPARC .csg file does not contain result groups: {original_csg_path}")

    for group_name, group_data in results.items():
        if not isinstance(group_data, dict):
            continue
        output_file = group_to_output_file.get(str(group_name))
        if output_file is None:
            continue
        group_data["metafile"] = ">" + _relativize_for_csg(output_file, output_csg_path)
        group_data["num_items"] = int(num_items)

    output_csg_path.write_text(
        yaml.dump(csg_data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _filter_cs_array_by_reference(
    data: np.ndarray,
    *,
    keep: np.ndarray,
    reference: np.ndarray,
    kept_uids: Optional[np.ndarray],
    data_path: Path,
) -> np.ndarray:
    if kept_uids is not None and data.dtype.names and "uid" in data.dtype.names:
        return data[np.isin(data["uid"], kept_uids)]
    if len(data) != len(reference):
        raise ValueError(
            f"Cannot filter {data_path} positionally because it has {len(data)} row(s), "
            f"but the location source has {len(reference)} row(s), and no uid field was available."
        )
    return data[keep]


def collect_particle_coordinates_by_micrograph(
    *,
    particle_file: str | Path,
    micrograph_paths: Sequence[str | Path],
    pixel_size_override_angstrom: Optional[float] = None,
    csg_file: str | Path | None = None,
    allow_unmatched: bool = False,
) -> dict[str, Any]:
    """Collect particle center coordinates in original input-micrograph pixels."""

    input_path = Path(particle_file).expanduser().resolve()
    original_input_path = input_path
    location_input_path = input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_PARTICLE_SUFFIXES:
        raise ValueError(
            f"Unsupported particle file {input_path}. Expected one of {SUPPORTED_PARTICLE_SUFFIXES}."
        )

    sources, aliases, ambiguous_aliases = _build_micrograph_sources(
        micrograph_paths=micrograph_paths,
        pixel_size_override_angstrom=pixel_size_override_angstrom,
    )
    coords_by_micrograph: dict[Path, list[tuple[float, float]]] = {
        source.micrograph_path: [] for source in sources
    }
    unmatched_names: set[str] = set()
    unmatched_particles = 0
    particles_input = 0

    if suffix == ".cs":
        particles = load_cs(str(input_path))
        missing = _missing_location_fields(particles)
        if missing:
            resolved_csg: Optional[Path] = None
            if csg_file is not None:
                resolved_csg = Path(csg_file).expanduser().resolve()
                if not resolved_csg.exists():
                    raise FileNotFoundError(resolved_csg)
            else:
                sibling = input_path.with_suffix(".csg")
                if sibling.exists():
                    resolved_csg = sibling
            if resolved_csg is not None:
                csg_group_inputs = _load_csg_group_metafiles(resolved_csg, known_paths=(input_path,))
                candidate_paths = []
                preferred_location_path = csg_group_inputs.get("location")
                if preferred_location_path is not None:
                    candidate_paths.append(preferred_location_path)
                candidate_paths.extend(
                    path for path in dict.fromkeys(csg_group_inputs.values()) if path not in candidate_paths
                )
                for candidate_path in candidate_paths:
                    candidate_particles = load_cs(str(candidate_path))
                    if _cs_has_location_fields(candidate_particles):
                        particles = candidate_particles
                        missing = []
                        location_input_path = candidate_path
                        break
            if missing:
                hint = (
                    " If this is a split CryoSPARC export, also pass the matching .csg with "
                    "--particle-csg or use the passthrough/location .cs file."
                )
                raise ValueError(f"CryoSPARC particle file is missing required fields: {missing}.{hint}")
        particles_input = int(len(particles))
        for particle in particles:
            raw_name = particle["location/micrograph_path"]
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode("utf-8", errors="replace")
            raw_name = str(raw_name)
            source = _resolve_source(raw_name, aliases, ambiguous_aliases)
            if source is None:
                unmatched_names.add(raw_name)
                unmatched_particles += 1
                continue
            input_h, input_w = source.input_shape
            x_input = float(particle["location/center_x_frac"]) * float(input_w)
            y_input = float(particle["location/center_y_frac"]) * float(input_h)
            coords_by_micrograph[source.micrograph_path].append((x_input, y_input))
        source_format = "cryosparc_cs"
    else:
        lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        loop = _find_micrograph_loop(lines)
        lower_columns = {column.lower(): idx for idx, column in enumerate(loop.columns)}
        required = ("_rlnmicrographname", "_rlncoordinatex", "_rlncoordinatey")
        missing = [column for column in required if column not in lower_columns]
        if missing:
            raise ValueError(f"RELION STAR particle loop is missing required columns: {missing}")
        mic_idx = lower_columns["_rlnmicrographname"]
        x_idx = lower_columns["_rlncoordinatex"]
        y_idx = lower_columns["_rlncoordinatey"]
        particles_input = int(len(loop.rows))
        for row in loop.rows:
            raw_name = str(row[mic_idx])
            source = _resolve_source(raw_name, aliases, ambiguous_aliases)
            if source is None:
                unmatched_names.add(raw_name)
                unmatched_particles += 1
                continue
            coords_by_micrograph[source.micrograph_path].append(
                (float(row[x_idx]) - 1.0, float(row[y_idx]) - 1.0)
            )
        source_format = "relion_star"

    if unmatched_particles and not allow_unmatched:
        examples = ", ".join(sorted(unmatched_names)[:5])
        raise ValueError(
            f"Could not match {unmatched_particles} particle(s) to inferred micrographs "
            f"({examples}). Include those micrographs or pass --allow-unmatched-particles "
            "to preserve unmatched particles unchanged."
        )

    arrays_by_micrograph = {
        path: (
            np.asarray(coords, dtype=np.float32).reshape(-1, 2)
            if coords
            else np.empty((0, 2), dtype=np.float32)
        )
        for path, coords in coords_by_micrograph.items()
    }
    return {
        "format": source_format,
        "input_particle_file": str(original_input_path),
        "location_particle_file": str(location_input_path) if suffix == ".cs" else None,
        "particles_input": int(particles_input),
        "particles_matched": int(sum(len(coords) for coords in arrays_by_micrograph.values())),
        "unmatched_particles_preserved": int(unmatched_particles),
        "unmatched_micrograph_names": sorted(unmatched_names),
        "per_micrograph": [
            {"micrograph": str(path), "particles": int(len(coords))}
            for path, coords in sorted(arrays_by_micrograph.items(), key=lambda item: str(item[0]))
            if len(coords) > 0
        ],
        "coordinates_by_micrograph": arrays_by_micrograph,
    }


def classify_particle_coordinates_by_mask(
    coords_xy: np.ndarray,
    *,
    bad_mask: np.ndarray,
    input_shape: tuple[int, int],
    pixel_size_angstrom: float,
    exclusion_distance_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(keep, distance_angstrom)`` for particle centers and an in-memory mask."""

    coords = np.asarray(coords_xy, dtype=np.float32)
    if coords.size:
        coords = coords.reshape(-1, 2)
    else:
        coords = np.empty((0, 2), dtype=np.float32)
    if len(coords) == 0:
        return np.zeros(0, dtype=bool), np.empty(0, dtype=np.float32)

    mask = np.asarray(bad_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D bad_mask, got shape={mask.shape}")

    if not np.any(mask):
        distances = np.full(len(coords), np.inf, dtype=np.float32)
        return np.ones(len(coords), dtype=bool), distances

    input_h, input_w = int(input_shape[0]), int(input_shape[1])
    mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
    sampling_y = float(pixel_size_angstrom) * float(input_h) / float(max(mask_h, 1))
    sampling_x = float(pixel_size_angstrom) * float(input_w) / float(max(mask_w, 1))
    distance_map = ndi.distance_transform_edt(~mask, sampling=(sampling_y, sampling_x)).astype(
        np.float32,
        copy=False,
    )
    x_mask = np.clip(
        np.floor(coords[:, 0] * float(mask_w) / float(max(input_w, 1))).astype(np.int64),
        0,
        mask_w - 1,
    )
    y_mask = np.clip(
        np.floor(coords[:, 1] * float(mask_h) / float(max(input_h, 1))).astype(np.int64),
        0,
        mask_h - 1,
    )
    distances = np.asarray(distance_map[y_mask, x_mask], dtype=np.float32)
    keep = distances > float(exclusion_distance_angstrom)
    return keep.astype(bool, copy=False), distances


def _filter_cs(
    *,
    input_path: Path,
    output_path: Path,
    sources: Sequence[_MaskSource],
    aliases: dict[str, _MaskSource],
    ambiguous_aliases: set[str],
    exclusion_distance_angstrom: float,
    allow_unmatched: bool,
    csg_path: Optional[Path],
    csg_group_inputs: dict[str, Path],
) -> dict[str, Any]:
    loaded_cs: dict[Path, np.ndarray] = {}

    def _load(path: Path) -> np.ndarray:
        resolved = Path(path).expanduser().resolve()
        data = loaded_cs.get(resolved)
        if data is None:
            data = load_cs(str(resolved))
            loaded_cs[resolved] = data
        return data

    input_particles = _load(input_path)
    location_source_path = input_path if _cs_has_location_fields(input_particles) else None
    if location_source_path is None and csg_path is not None:
        preferred_location_path = csg_group_inputs.get("location")
        if preferred_location_path is not None and _cs_has_location_fields(_load(preferred_location_path)):
            location_source_path = preferred_location_path
        else:
            for member_path in dict.fromkeys(csg_group_inputs.values()):
                if _cs_has_location_fields(_load(member_path)):
                    location_source_path = member_path
                    break

    if location_source_path is None:
        missing = _missing_location_fields(input_particles)
        hint = (
            " If this is a split CryoSPARC export, also pass the matching .csg with "
            "--particle-csg or use the passthrough/location .cs file."
        )
        raise ValueError(f"CryoSPARC particle file is missing required fields: {missing}.{hint}")

    particles = _load(location_source_path)

    counts = _new_counts(sources)
    keep = np.ones(len(particles), dtype=bool)
    distance_maps: dict[Path, np.ndarray] = {}
    unmatched_names: set[str] = set()
    unmatched_particles = 0

    for idx, particle in enumerate(particles):
        raw_name = particle["location/micrograph_path"]
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="replace")
        raw_name = str(raw_name)
        source = _resolve_source(raw_name, aliases, ambiguous_aliases)
        if source is None:
            unmatched_names.add(raw_name)
            unmatched_particles += 1
            continue

        counts[source.micrograph_path]["particles"] += 1
        input_h, input_w = source.input_shape
        x_input = float(particle["location/center_x_frac"]) * float(input_w)
        y_input = float(particle["location/center_y_frac"]) * float(input_h)
        distance_map = distance_maps.get(source.mask_path)
        if distance_map is None:
            distance_map = _distance_map_angstrom(source)
            distance_maps[source.mask_path] = distance_map
        distance = _distance_for_input_coordinate(
            distance_map,
            x_input_px=x_input,
            y_input_px=y_input,
            input_shape=source.input_shape,
        )
        remove = distance <= float(exclusion_distance_angstrom)
        keep[idx] = not remove
        counts[source.micrograph_path]["removed" if remove else "kept"] += 1

    if unmatched_particles and not allow_unmatched:
        examples = ", ".join(sorted(unmatched_names)[:5])
        raise ValueError(
            f"Could not match {unmatched_particles} particle(s) to inferred micrographs "
            f"({examples}). Include those micrographs or pass --allow-unmatched-particles "
            "to preserve unmatched particles unchanged."
        )

    kept_uids: Optional[np.ndarray] = None
    if particles.dtype.names and "uid" in particles.dtype.names:
        kept_uids = np.asarray(particles["uid"][keep])
    kept_count = int(np.sum(keep))

    if csg_path is not None and csg_group_inputs:
        output_cs_files = _csg_member_output_paths(
            group_to_input_file=csg_group_inputs,
            primary_input_path=input_path,
            primary_output_path=output_path,
        )
    else:
        output_cs_files = {input_path: output_path}

    for member_input, member_output in output_cs_files.items():
        member_data = _load(member_input)
        filtered_member = _filter_cs_array_by_reference(
            member_data,
            keep=keep,
            reference=particles,
            kept_uids=kept_uids,
            data_path=member_input,
        )
        if len(filtered_member) != kept_count:
            raise ValueError(
                f"Filtered row count mismatch for {member_input}: wrote {len(filtered_member)} row(s), "
                f"expected {kept_count}. Check that all .cs files in the .csg share particle uids."
            )
        member_output.parent.mkdir(parents=True, exist_ok=True)
        save_cs(filtered_member, str(member_output))

    output_csg: Optional[Path] = None
    if csg_path is not None:
        output_csg = output_path.with_suffix(".csg")
        output_csg.parent.mkdir(parents=True, exist_ok=True)
        group_to_output_file = {
            group_name: output_cs_files[group_input]
            for group_name, group_input in csg_group_inputs.items()
            if group_input in output_cs_files
        }
        _write_filtered_csg(
            original_csg_path=csg_path,
            output_csg_path=output_csg,
            group_to_output_file=group_to_output_file,
            num_items=kept_count,
        )

    output_location_file = output_cs_files.get(location_source_path, output_path)
    output_file_list = [str(path) for path in sorted(set(output_cs_files.values()), key=lambda value: str(value))]

    return {
        "format": "cryosparc_cs",
        "input_particle_file": str(input_path),
        "output_particle_file": str(output_path),
        "location_particle_file": str(location_source_path),
        "output_location_particle_file": str(output_location_file),
        "output_particle_files": output_file_list,
        "output_csg_file": None if output_csg is None else str(output_csg),
        "particles_input": int(len(particles)),
        "particles_kept": int(np.sum(keep)),
        "particles_removed": int(np.sum(~keep)),
        "unmatched_particles_preserved": int(unmatched_particles),
        "unmatched_micrograph_names": sorted(unmatched_names),
        "per_micrograph": _counts_payload(counts),
    }


def _filter_star(
    *,
    input_path: Path,
    output_path: Path,
    sources: Sequence[_MaskSource],
    aliases: dict[str, _MaskSource],
    ambiguous_aliases: set[str],
    exclusion_distance_angstrom: float,
    allow_unmatched: bool,
) -> dict[str, Any]:
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    loop = _find_micrograph_loop(lines)
    lower_columns = {column.lower(): idx for idx, column in enumerate(loop.columns)}
    required = ("_rlnmicrographname", "_rlncoordinatex", "_rlncoordinatey")
    missing = [column for column in required if column not in lower_columns]
    if missing:
        raise ValueError(f"RELION STAR particle loop is missing required columns: {missing}")

    mic_idx = lower_columns["_rlnmicrographname"]
    x_idx = lower_columns["_rlncoordinatex"]
    y_idx = lower_columns["_rlncoordinatey"]
    counts = _new_counts(sources)
    distance_maps: dict[Path, np.ndarray] = {}
    keep_rows: list[list[str]] = []
    removed = 0
    unmatched_names: set[str] = set()
    unmatched_particles = 0

    for row in loop.rows:
        raw_name = str(row[mic_idx])
        source = _resolve_source(raw_name, aliases, ambiguous_aliases)
        if source is None:
            unmatched_names.add(raw_name)
            unmatched_particles += 1
            keep_rows.append(row)
            continue

        counts[source.micrograph_path]["particles"] += 1
        distance_map = distance_maps.get(source.mask_path)
        if distance_map is None:
            distance_map = _distance_map_angstrom(source)
            distance_maps[source.mask_path] = distance_map
        distance = _distance_for_input_coordinate(
            distance_map,
            x_input_px=float(row[x_idx]) - 1.0,
            y_input_px=float(row[y_idx]) - 1.0,
            input_shape=source.input_shape,
        )
        remove = distance <= float(exclusion_distance_angstrom)
        counts[source.micrograph_path]["removed" if remove else "kept"] += 1
        if remove:
            removed += 1
        else:
            keep_rows.append(row)

    if unmatched_particles and not allow_unmatched:
        examples = ", ".join(sorted(unmatched_names)[:5])
        raise ValueError(
            f"Could not match {unmatched_particles} particle(s) to inferred micrographs "
            f"({examples}). Include those micrographs or pass --allow-unmatched-particles "
            "to preserve unmatched particles unchanged."
        )

    row_lines = [" ".join(_format_star_value(token) for token in row) + "\n" for row in keep_rows]
    new_lines = loop.lines[: loop.data_start_idx] + row_lines + loop.lines[loop.data_end_idx :]
    text = "".join(new_lines)
    if text and not text.endswith("\n"):
        text += "\n"
    output_path.write_text(text, encoding="utf-8")

    return {
        "format": "relion_star",
        "input_particle_file": str(input_path),
        "output_particle_file": str(output_path),
        "output_csg_file": None,
        "particles_input": int(len(loop.rows)),
        "particles_kept": int(len(keep_rows)),
        "particles_removed": int(removed),
        "unmatched_particles_preserved": int(unmatched_particles),
        "unmatched_micrograph_names": sorted(unmatched_names),
        "per_micrograph": _counts_payload(counts),
    }


def default_filtered_particle_path(particle_file: str | Path, output_dir: str | Path) -> Path:
    source = Path(particle_file)
    return Path(output_dir) / f"{source.stem}_cryofiltered{source.suffix.lower()}"


def filter_particle_file(
    *,
    particle_file: str | Path,
    output_file: str | Path,
    micrograph_paths: Sequence[str | Path],
    mask_dir: str | Path,
    exclusion_distance_angstrom: float,
    pixel_size_override_angstrom: Optional[float] = None,
    csg_file: str | Path | None = None,
    allow_unmatched: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Filter a CryoSPARC ``.cs`` or RELION coordinate ``.star`` file."""

    input_path = Path(particle_file).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_PARTICLE_SUFFIXES:
        raise ValueError(
            f"Unsupported particle file {input_path}. Expected one of {SUPPORTED_PARTICLE_SUFFIXES}."
        )
    if output_path.suffix.lower() != suffix:
        raise ValueError(
            f"Filtered particle output must keep the {suffix} extension: {output_path}"
        )
    if output_path == input_path:
        raise ValueError("Filtered particle output must not overwrite the input particle file")
    if not np.isfinite(float(exclusion_distance_angstrom)) or float(exclusion_distance_angstrom) < 0:
        raise ValueError("exclusion_distance_angstrom must be a finite value >= 0")

    resolved_csg: Optional[Path] = None
    csg_group_inputs: dict[str, Path] = {}
    if suffix == ".cs":
        if csg_file is not None:
            resolved_csg = Path(csg_file).expanduser().resolve()
            if not resolved_csg.exists():
                raise FileNotFoundError(resolved_csg)
        else:
            sibling = input_path.with_suffix(".csg")
            if sibling.exists():
                resolved_csg = sibling
        if resolved_csg is not None:
            csg_group_inputs = _load_csg_group_metafiles(resolved_csg, known_paths=(input_path,))
    elif csg_file is not None:
        raise ValueError("--particle-csg can only be used with a CryoSPARC .cs particle file")

    possible_outputs = [output_path]
    if resolved_csg is not None:
        possible_outputs.append(output_path.with_suffix(".csg"))
        for member_output in _csg_member_output_paths(
            group_to_input_file=csg_group_inputs,
            primary_input_path=input_path,
            primary_output_path=output_path,
        ).values():
            possible_outputs.append(member_output)
    possible_outputs = sorted(set(possible_outputs), key=lambda path: str(path))
    existing = [path for path in possible_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Filtered particle output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite-filtered-particles to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources, aliases, ambiguous_aliases = _build_mask_sources(
        micrograph_paths=micrograph_paths,
        mask_dir=mask_dir,
        pixel_size_override_angstrom=pixel_size_override_angstrom,
    )
    if not sources:
        raise ValueError("No inferred micrographs were available for particle filtering")

    common = {
        "input_path": input_path,
        "output_path": output_path,
        "sources": sources,
        "aliases": aliases,
        "ambiguous_aliases": ambiguous_aliases,
        "exclusion_distance_angstrom": float(exclusion_distance_angstrom),
        "allow_unmatched": bool(allow_unmatched),
    }
    if suffix == ".cs":
        result = _filter_cs(**common, csg_path=resolved_csg, csg_group_inputs=csg_group_inputs)
    else:
        result = _filter_star(**common)
    result["exclusion_distance_angstrom"] = float(exclusion_distance_angstrom)
    result["exclusion_distance_coordinate_system"] = "original_micrograph"
    result["mask_source_count"] = int(len(sources))
    return result


__all__ = [
    "SUPPORTED_PARTICLE_SUFFIXES",
    "classify_particle_coordinates_by_mask",
    "collect_particle_coordinates_by_micrograph",
    "default_filtered_particle_path",
    "filter_particle_file",
]
