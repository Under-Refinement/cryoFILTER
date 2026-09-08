"""Helpers for preparing coordinate STAR files for cryoSPARC import."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shlex
from typing import Any


UID_PREFIX_MIN_DIGITS = 10


@dataclass
class StarImportSummary:
    input_star: str
    output_star: str
    n_rows: int
    n_micrograph_names_changed: int
    n_uid_prefixes_stripped: int
    added_rln_image_pixel_size: bool
    repaired_rln_image_pixel_size: int
    pixel_size_angstrom: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_star": self.input_star,
            "output_star": self.output_star,
            "n_rows": self.n_rows,
            "n_micrograph_names_changed": self.n_micrograph_names_changed,
            "n_uid_prefixes_stripped": self.n_uid_prefixes_stripped,
            "added_rln_image_pixel_size": self.added_rln_image_pixel_size,
            "repaired_rln_image_pixel_size": self.repaired_rln_image_pixel_size,
            "pixel_size_angstrom": self.pixel_size_angstrom,
        }


@dataclass
class _StarLoop:
    lines: list[str]
    col_start_idx: int
    data_start_idx: int
    data_end_idx: int
    columns: list[str]
    rows: list[list[str]]


def strip_cryosparc_uid_prefix(name: str) -> tuple[str, bool]:
    """Return a basename with a long CryoSPARC UID prefix removed.

    CryoSPARC can prepend decimal uint64-like identifiers to exported exposure
    names, for example ``000012076402221568849_FoilHole_...mrc``. Only long
    numeric prefixes are stripped so date-like names such as
    ``20200224_FoilHole_...mrc`` are preserved.
    """

    basename = Path(str(name)).name
    prefix, sep, rest = basename.partition("_")
    if sep and prefix.isdigit() and len(prefix) >= UID_PREFIX_MIN_DIGITS and rest:
        return rest, True
    return basename, False


def _parse_star_tokens(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return line.strip().split()


def _find_micrograph_loop(lines: list[str]) -> _StarLoop:
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.lower() != "loop_":
            i += 1
            continue

        j = i + 1
        columns: list[str] = []
        col_start_idx = j
        while j < len(lines):
            col_line = lines[j].strip()
            if not col_line:
                j += 1
                continue
            if not col_line.startswith("_"):
                break
            columns.append(col_line.split()[0])
            j += 1

        if not columns:
            i += 1
            continue

        lower_columns = {col.lower(): idx for idx, col in enumerate(columns)}
        if "_rlnmicrographname" not in lower_columns:
            i = j
            continue

        rows: list[list[str]] = []
        k = j
        while k < len(lines):
            row_line = lines[k].strip()
            if not row_line:
                break
            if row_line.startswith("#"):
                break
            row_line_lower = row_line.lower()
            if row_line.startswith("_") or row_line_lower.startswith("loop_") or row_line_lower.startswith("data_"):
                break
            parts = _parse_star_tokens(row_line)
            if len(parts) < len(columns):
                break
            rows.append(parts[: len(columns)])
            k += 1

        if rows:
            return _StarLoop(
                lines=lines,
                col_start_idx=col_start_idx,
                data_start_idx=j,
                data_end_idx=k,
                columns=columns,
                rows=rows,
            )
        i = k

    raise ValueError("No STAR loop containing _rlnMicrographName rows was found")


def _format_star_value(value: str) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in ('"', "'")):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _valid_pixel_size(value: str) -> bool:
    try:
        parsed = float(value)
    except Exception:
        return False
    return math.isfinite(parsed) and parsed > 0.0


def _resolve_pixel_size(pixel_size_angstrom: float | None) -> float | None:
    if pixel_size_angstrom is None:
        return None
    value = float(pixel_size_angstrom)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"pixel_size_angstrom must be a positive finite value, got {pixel_size_angstrom!r}")
    return value


def prepare_star_for_cryosparc_import(
    input_star: str | Path,
    output_star: str | Path,
    *,
    pixel_size_angstrom: float | None = None,
    normalize_micrograph_names: bool = True,
    overwrite: bool = False,
) -> StarImportSummary:
    """Write a CryoSPARC import-safe copy of a coordinate STAR file.

    The output STAR keeps the original coordinate loop structure but can rewrite
    ``_rlnMicrographName`` to bare exposure basenames with long CryoSPARC UID
    prefixes stripped. It also ensures ``_rlnImagePixelSize`` is present when a
    pixel size is supplied.
    """

    src = Path(input_star)
    dst = Path(output_star)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; pass overwrite=True to replace it")

    pixel_size = _resolve_pixel_size(pixel_size_angstrom)
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    loop = _find_micrograph_loop(lines)

    lower_columns = {col.lower(): idx for idx, col in enumerate(loop.columns)}
    micrograph_idx = lower_columns["_rlnmicrographname"]
    pixel_idx = lower_columns.get("_rlnimagepixelsize")
    added_pixel_column = pixel_idx is None
    if added_pixel_column:
        if pixel_size is None:
            raise ValueError(
                "The STAR file does not contain _rlnImagePixelSize; pass pixel_size_angstrom "
                "so cryoSPARC receives particle pixel-size metadata."
            )
        pixel_idx = len(loop.columns)
        loop.columns.append("_rlnImagePixelSize")

    changed_names = 0
    stripped_prefixes = 0
    repaired_pixel_values = 0
    rewritten_rows: list[list[str]] = []
    for row in loop.rows:
        out_row = list(row)
        if normalize_micrograph_names:
            normalized, stripped = strip_cryosparc_uid_prefix(out_row[micrograph_idx])
            if normalized != out_row[micrograph_idx]:
                changed_names += 1
            if stripped:
                stripped_prefixes += 1
            out_row[micrograph_idx] = normalized

        if added_pixel_column:
            out_row.append(f"{float(pixel_size):.6f}")
        else:
            assert pixel_idx is not None
            if not _valid_pixel_size(out_row[pixel_idx]):
                if pixel_size is None:
                    raise ValueError(
                        "The STAR file contains invalid _rlnImagePixelSize values; pass "
                        "pixel_size_angstrom to repair them."
                    )
                out_row[pixel_idx] = f"{float(pixel_size):.6f}"
                repaired_pixel_values += 1
        rewritten_rows.append(out_row)

    column_lines = [f"{col} #{idx}\n" for idx, col in enumerate(loop.columns, start=1)]
    row_lines = [" ".join(_format_star_value(token) for token in row) + "\n" for row in rewritten_rows]
    new_lines = (
        loop.lines[: loop.col_start_idx]
        + column_lines
        + row_lines
        + loop.lines[loop.data_end_idx :]
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(new_lines)
    if text and not text.endswith("\n"):
        text += "\n"
    dst.write_text(text, encoding="utf-8")

    return StarImportSummary(
        input_star=str(src),
        output_star=str(dst),
        n_rows=len(rewritten_rows),
        n_micrograph_names_changed=changed_names,
        n_uid_prefixes_stripped=stripped_prefixes,
        added_rln_image_pixel_size=added_pixel_column,
        repaired_rln_image_pixel_size=repaired_pixel_values,
        pixel_size_angstrom=pixel_size,
    )
