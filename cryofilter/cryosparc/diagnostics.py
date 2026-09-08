"""Build CryoSPARC-facing diagnostic artifacts from cryoFILTER outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from cryofilter.cryosparc.protocol.manifests import write_json_model
from cryofilter.cryosparc.protocol.models import DiagnosticAsset, DiagnosticsManifest

TYPE_COLORS = {
    "Carbon": "#D95F02",
    "Crystalline": "#1B9E77",
    "Aggregate": "#CC79A7",
    "Ethane": "#E6AB02",
}
DEFAULT_MAX_OVERLAY_IMAGES = 10


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _resolve_path(raw: object, *, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _run_id_from_summary(summary: dict[str, Any]) -> UUID | None:
    for key in ("run_id", "cryosparc_run_id"):
        value = summary.get(key)
        if not value:
            continue
        try:
            return UUID(str(value))
        except ValueError:
            continue
    return None


def _asset(
    *,
    kind: str,
    title: str,
    caption: str,
    path: Path,
    source: str | None = None,
) -> DiagnosticAsset:
    return DiagnosticAsset(
        kind=kind,
        title=title,
        caption=caption,
        local_path=str(path),
        source=source,
    )


def _contamination_rows(inference_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(inference_summary.get("inputs", []), start=1):
        if not isinstance(item, dict):
            continue
        mask_meta = item.get("mask_postprocessing")
        if not isinstance(mask_meta, dict):
            mask_meta = {}

        shape = item.get("output_image_shape") or item.get("input_image_shape")
        total_pixels = 0
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            total_pixels = int(shape[0]) * int(shape[1])

        mask_pixels = mask_meta.get("final_mask_pixels")
        mask_fraction = mask_meta.get("final_mask_fraction")
        if mask_pixels is None and mask_fraction is not None and total_pixels > 0:
            mask_pixels = int(round(float(mask_fraction) * total_pixels))
        if mask_fraction is None and mask_pixels is not None and total_pixels > 0:
            mask_fraction = float(mask_pixels) / float(total_pixels)
        if mask_pixels is None or mask_fraction is None:
            continue

        input_mrc = item.get("input_mrc")
        label = Path(str(input_mrc)).name if input_mrc else f"micrograph_{index:03d}"
        rows.append(
            {
                "label": label,
                "input_mrc": None if input_mrc is None else str(input_mrc),
                "contaminated_pixels": int(mask_pixels),
                "total_pixels": int(total_pixels),
                "contamination_fraction": float(mask_fraction),
                "contamination_pct": float(mask_fraction) * 100.0,
            }
        )
    return rows


def _micrograph_lookup_keys(value: object) -> set[str]:
    if value in (None, ""):
        return set()
    text = str(value)
    path = Path(text)
    return {
        text,
        path.name,
        path.stem,
    }


def _rank_overlay_frames(
    frames: list[object],
    *,
    contamination_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None, float | None, int]]:
    row_by_key: dict[str, dict[str, Any]] = {}
    for row in contamination_rows:
        for key in _micrograph_lookup_keys(row.get("input_mrc")) | _micrograph_lookup_keys(row.get("label")):
            row_by_key.setdefault(key, row)

    ranked: list[tuple[dict[str, Any], dict[str, Any] | None, float | None, int]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        row = None
        for key in _micrograph_lookup_keys(frame.get("micrograph")):
            row = row_by_key.get(key)
            if row is not None:
                break
        score = None
        if row is not None:
            score = float(row["contamination_pct"])
        ranked.append((frame, row, score, index))

    if any(score is not None for _, _, score, _ in ranked):
        ranked.sort(key=lambda item: (item[2] is None, -(item[2] or 0.0), item[3]))
    return ranked


def _existing_overlay_assets(
    inference_summary: dict[str, Any],
    *,
    inference_dir: Path,
    diagnostics_dir: Path,
    max_overlay_images: int,
    contamination_rows: list[dict[str, Any]],
) -> tuple[list[DiagnosticAsset], dict[str, Any]]:
    assets: list[DiagnosticAsset] = []
    selection_summary: dict[str, Any] = {}
    overlay = inference_summary.get("particle_overlay_rendering")
    if not isinstance(overlay, dict) or not overlay.get("enabled"):
        return assets, selection_summary

    frame_limit = max(0, int(max_overlay_images))
    frames = overlay.get("frames")
    if not isinstance(frames, list):
        frames = []
    selection_summary = {
        "otf_overlays_available": len([frame for frame in frames if isinstance(frame, dict)]),
        "otf_overlays_max": frame_limit,
        "otf_overlay_selection_strategy": (
            "top_contaminated_micrographs" if contamination_rows else "inference_order"
        ),
        "otf_overlay_ranked_labels": [],
    }
    selected: list[tuple[Path, str, float | None, int]] = []
    selected_count = 0
    if frame_limit > 0:
        for frame, row, score, _index in _rank_overlay_frames(
            frames,
            contamination_rows=contamination_rows,
        ):
            if selected_count >= frame_limit:
                break
            if not frame.get("output_png"):
                continue
            frame_path = _resolve_path(frame["output_png"], base=inference_dir)
            if not frame_path.exists():
                continue
            selected_count += 1
            label = (
                str(row["label"])
                if row is not None and row.get("label")
                else Path(str(frame.get("micrograph") or frame_path.stem)).name
            )
            selected.append((frame_path, label, score, selected_count))
            selection_summary["otf_overlay_ranked_labels"].append(
                {
                    "rank": selected_count,
                    "label": label,
                    "contamination_pct": None if score is None else float(score),
                }
            )
    selection_summary["otf_overlays_selected"] = len(selected)

    sheet = overlay.get("final_contact_sheet_path") or overlay.get("contact_sheet_path")
    original_sheet_path = _resolve_path(sheet, base=inference_dir) if sheet else None
    contact_sheet_path: Path | None = None
    contact_sheet_caption = (
        "On-the-fly particle filtering overlays: raw micrograph, "
        "mask/particle decision overlay, and probability panel when enabled."
    )
    if selected:
        contact_sheet_path = diagnostics_dir / "top_contaminated_otf_contact_sheet.png"
        try:
            from cryofilter.qc_render import montage_from_paths

            montage_from_paths(
                [frame_path for frame_path, _label, _score, _rank in selected],
                output_path=contact_sheet_path,
                cols=min(5, len(selected)),
                tile=(0, 400),
            )
            contact_sheet_caption = (
                f"Top {len(selected)} cryoFILTER OTF overlays ranked by predicted "
                "contaminated area."
            )
        except Exception as exc:
            selection_summary["otf_contact_sheet_warning"] = f"{type(exc).__name__}: {exc}"
            contact_sheet_path = (
                original_sheet_path
                if original_sheet_path is not None and original_sheet_path.exists()
                else None
            )
    elif original_sheet_path is not None and original_sheet_path.exists():
        contact_sheet_path = original_sheet_path

    if contact_sheet_path is not None and contact_sheet_path.exists():
        assets.append(
            _asset(
                kind="otf_contact_sheet",
                title="cryoFILTER OTF Diagnostics",
                caption=contact_sheet_caption,
                path=contact_sheet_path,
                source="inference_summary",
            )
        )

    for frame_path, _label, score, rank in selected:
        contamination_caption = (
            f"rank {rank}, {score:.2f}% contaminated area"
            if score is not None
            else f"selected overlay {rank}"
        )
        assets.append(
            _asset(
                kind="otf_overlay",
                title=Path(frame_path).stem,
                caption=f"Top cryoFILTER OTF particle filtering overlay ({contamination_caption}).",
                path=frame_path,
                source="inference_summary",
            )
        )
    return assets, selection_summary



def _style_axes(ax) -> None:
    ax.set_facecolor("#101418")
    ax.tick_params(colors="#d8dee9", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")


def _new_figure(width: float, height: float):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width, height), facecolor="#101418")
    return fig, plt


def _save_total_pie(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    contaminated = int(sum(row["contaminated_pixels"] for row in rows))
    total = int(sum(row["total_pixels"] for row in rows))
    clean = max(0, total - contaminated)
    pct = 100.0 * float(contaminated) / float(max(total, 1))

    fig, plt = _new_figure(6.0, 4.6)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    values = [contaminated, clean]
    labels = ["Contamination", "Clean"]
    colors = ["#b77cff", "#21c7a8"]
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda value: f"{value:.1f}%",
        pctdistance=0.74,
        textprops={"color": "#f8fafc", "fontsize": 10, "weight": "bold"},
        wedgeprops={"linewidth": 2, "edgecolor": "#101418"},
    )
    ax.set_title("Total Contamination", color="#f8fafc", fontsize=16, weight="bold", pad=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "contaminated_pixels": contaminated,
        "total_pixels": total,
        "contamination_pct": pct,
    }


def _save_micrograph_bar(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    max_bar_items: int,
) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: row["contamination_pct"], reverse=True)
    shown = ranked[: max(1, int(max_bar_items))]
    labels = [str(index) for index in range(1, len(shown) + 1)]
    values = [float(row["contamination_pct"]) for row in shown]

    height = max(3.6, min(12.0, 0.34 * len(shown) + 1.7))
    fig, plt = _new_figure(8.5, height)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    positions = list(range(len(shown)))
    colors = ["#ff7a90" if value >= 25.0 else "#ffc857" if value >= 10.0 else "#33d6a6" for value in values]
    ax.barh(positions, values, color=colors, edgecolor="#101418", linewidth=0.8)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Contaminated area (%)", color="#d8dee9")
    ax.set_ylabel("Rank", color="#d8dee9")
    ax.set_title("Contamination by Micrograph", color="#f8fafc", fontsize=15, weight="bold", pad=12)
    ax.grid(axis="x", color="#344052", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    xmax = max(values) if values else 1.0
    ax.set_xlim(0, max(5.0, xmax * 1.18))
    for y, value in zip(positions, values):
        ax.text(value + max(0.2, xmax * 0.015), y, f"{value:.2f}%", color="#f8fafc", va="center", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "micrographs_plotted": len(shown),
        "micrographs_total": len(rows),
        "micrograph_ranked_labels": [
            {
                "rank": int(index),
                "label": str(row["label"]),
                "contamination_pct": float(row["contamination_pct"]),
            }
            for index, row in enumerate(shown, start=1)
        ],
    }


def _slug_for_type(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def _typing_rows(typing_summary: dict[str, Any]) -> list[dict[str, Any]]:
    overall = typing_summary.get("overall_contamination_summary")
    if not isinstance(overall, dict):
        return []
    order = typing_summary.get("type_order")
    if not isinstance(order, list) or not order:
        order = list(TYPE_COLORS)

    rows: list[dict[str, Any]] = []
    for label_raw in order:
        label = str(label_raw)
        slug = _slug_for_type(label)
        area = int(float(overall.get(f"{slug}_area_px", 0) or 0))
        pct_of_contam = float(overall.get(f"{slug}_pct_of_contamination", 0.0) or 0.0)
        pct_total = float(overall.get(f"{slug}_pct_total_image", 0.0) or 0.0)
        rows.append(
            {
                "label": label,
                "area_px": area,
                "pct_of_contamination": pct_of_contam,
                "pct_total_image": pct_total,
                "color": TYPE_COLORS.get(label, "#8ab4f8"),
            }
        )
    return [row for row in rows if row["area_px"] > 0 or row["pct_of_contamination"] > 0]


def _save_type_pie(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    values = [float(row["area_px"]) for row in rows]
    if sum(values) <= 0:
        values = [float(row["pct_of_contamination"]) for row in rows]
    labels = [str(row["label"]) for row in rows]
    colors = [str(row["color"]) for row in rows]

    fig, plt = _new_figure(6.2, 4.8)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda value: f"{value:.1f}%" if value >= 3.0 else "",
        pctdistance=0.74,
        textprops={"color": "#f8fafc", "fontsize": 10, "weight": "bold"},
        wedgeprops={"linewidth": 2, "edgecolor": "#101418"},
    )
    ax.text(
        0,
        0,
        "typed\nmask",
        color="#f8fafc",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    ax.set_title("Contamination Type Breakdown", color="#f8fafc", fontsize=15, weight="bold", pad=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"types": len(rows)}


def _save_type_bar(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    labels = [str(row["label"]) for row in rows]
    values = [float(row["pct_of_contamination"]) for row in rows]
    colors = [str(row["color"]) for row in rows]

    fig, plt = _new_figure(7.0, 4.2)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    bars = ax.bar(labels, values, color=colors, edgecolor="#101418", linewidth=1.0)
    ax.set_ylabel("Share of contamination (%)", color="#d8dee9")
    ax.set_title("Typed Contamination Share", color="#f8fafc", fontsize=15, weight="bold", pad=12)
    ax.grid(axis="y", color="#344052", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(5.0, max(values) * 1.18 if values else 5.0))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(0.2, max(values) * 0.015 if values else 0.2),
            f"{value:.1f}%",
            color="#f8fafc",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"types": len(rows)}


def _maybe_attach_particle_summary(
    inference_summary: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    particle = inference_summary.get("particle_filtering")
    if not isinstance(particle, dict):
        return
    for key in ("particles_input", "particles_kept", "particles_removed"):
        if key in particle:
            summary[key] = int(particle[key])


def build_diagnostics_manifest(
    *,
    inference_summary_path: str | Path,
    output_dir: str | Path | None = None,
    typing_summary_path: str | Path | None = None,
    max_overlay_images: int = DEFAULT_MAX_OVERLAY_IMAGES,
    max_bar_items: int = 30,
) -> DiagnosticsManifest:
    """Create CryoSPARC-ready PNG diagnostics and a JSON asset manifest."""

    inference_path = Path(inference_summary_path).expanduser().resolve()
    inference_dir = inference_path.parent
    diagnostics_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (inference_dir / "cryosparc_diagnostics").resolve()
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    inference_summary = _load_json(inference_path)
    contam_rows = _contamination_rows(inference_summary)
    assets, overlay_summary = _existing_overlay_assets(
        inference_summary,
        inference_dir=inference_dir,
        diagnostics_dir=diagnostics_dir,
        max_overlay_images=max_overlay_images,
        contamination_rows=contam_rows,
    )
    summary: dict[str, Any] = {
        "inference_summary": str(inference_path),
        "typing_summary": None,
    }
    summary.update(overlay_summary)
    _maybe_attach_particle_summary(inference_summary, summary)

    if contam_rows:
        total_info = _save_total_pie(
            contam_rows,
            diagnostics_dir / "contamination_total_pie.png",
        )
        summary.update(total_info)
        assets.append(
            _asset(
                kind="contamination_total_pie",
                title="Total Contamination",
                caption=(
                    f"Predicted contaminated area across {len(contam_rows)} staged "
                    f"micrograph(s): {total_info['contamination_pct']:.2f}%."
                ),
                path=diagnostics_dir / "contamination_total_pie.png",
                source="inference_summary",
            )
        )
        bar_info = _save_micrograph_bar(
            contam_rows,
            diagnostics_dir / "contamination_by_micrograph_bar.png",
            max_bar_items=max_bar_items,
        )
        summary.update(bar_info)
        assets.append(
            _asset(
                kind="contamination_by_micrograph_bar",
                title="Contamination by Micrograph",
                caption="Per-micrograph contaminated area ranked from highest to lowest.",
                path=diagnostics_dir / "contamination_by_micrograph_bar.png",
                source="inference_summary",
            )
        )

    if typing_summary_path is not None:
        typing_path = Path(typing_summary_path).expanduser().resolve()
        typing_summary = _load_json(typing_path)
        summary["typing_summary"] = str(typing_path)
        type_rows = _typing_rows(typing_summary)
        summary["typed_contamination_types"] = [
            {
                "label": row["label"],
                "area_px": row["area_px"],
                "pct_of_contamination": row["pct_of_contamination"],
                "pct_total_image": row["pct_total_image"],
            }
            for row in type_rows
        ]
        if type_rows:
            _save_type_pie(type_rows, diagnostics_dir / "contamination_type_pie.png")
            assets.append(
                _asset(
                    kind="contamination_type_pie",
                    title="Contamination Types",
                    caption="Area-weighted predicted contamination type breakdown.",
                    path=diagnostics_dir / "contamination_type_pie.png",
                    source="typing_summary",
                )
            )
            _save_type_bar(type_rows, diagnostics_dir / "contamination_type_bar.png")
            assets.append(
                _asset(
                    kind="contamination_type_bar",
                    title="Typed Contamination Share",
                    caption="Predicted contamination type share by contaminated area.",
                    path=diagnostics_dir / "contamination_type_bar.png",
                    source="typing_summary",
                )
            )

    manifest = DiagnosticsManifest(
        run_id=_run_id_from_summary(inference_summary),
        assets=assets,
        summary=summary,
    )
    write_json_model(manifest, diagnostics_dir / "cryosparc_diagnostics_manifest.json")
    return manifest


__all__ = ["build_diagnostics_manifest"]
