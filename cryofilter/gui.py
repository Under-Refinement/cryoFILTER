"""Small public GUI for reviewing and correcting binary contamination masks."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


MICROGRAPH_COLUMNS = (
    "micrograph_path",
    "full_mic_path",
    "mic_path",
    "down_mic_path",
)
MASK_COLUMNS = ("mask_path", "binary_mask_path", "gt_mask_path")
MASK_COLOR = "#8B5CF6"
MASK_EDGE_COLOR = "#C4B5FD"


@dataclass(frozen=True)
class GuiRecord:
    row_index: int
    micrograph_path: Path
    mask_path: Optional[Path]
    dataset_id: str
    stem: str


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _first_value(row: dict[str, str], columns: Sequence[str]) -> tuple[str, str]:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return column, value
    return "", ""


def load_gui_manifest(
    manifest_path: str | Path,
    *,
    micrograph_dir: str | Path | None = None,
) -> tuple[Path, list[dict[str, str]], list[str], list[GuiRecord]]:
    """Load inference or training manifests accepted by the public mask GUI."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"GUI manifest not found: {manifest}")
    mic_root = (
        Path(micrograph_dir).expanduser().resolve()
        if micrograph_dir is not None
        else None
    )

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"GUI manifest contains no rows: {manifest}")
    if not any(column in fieldnames for column in MICROGRAPH_COLUMNS):
        raise ValueError(
            "GUI manifest needs one micrograph path column: "
            + ", ".join(MICROGRAPH_COLUMNS)
        )

    records: list[GuiRecord] = []
    for row_index, row in enumerate(rows):
        _mic_column, mic_value = _first_value(row, MICROGRAPH_COLUMNS)
        if not mic_value:
            raise ValueError(f"GUI manifest row {row_index + 2} has no micrograph path")
        micrograph = _resolve_path(mic_value, manifest.parent)
        if mic_root is not None:
            candidates = (mic_root / Path(mic_value).name, mic_root / f"{Path(mic_value).stem}.mrc")
            micrograph = next((path.resolve() for path in candidates if path.is_file()), micrograph)
        if not micrograph.is_file():
            raise FileNotFoundError(
                f"GUI manifest row {row_index + 2}: micrograph not found: {micrograph}"
            )

        _mask_column, mask_value = _first_value(row, MASK_COLUMNS)
        mask_path = _resolve_path(mask_value, manifest.parent) if mask_value else None
        if mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(
                f"GUI manifest row {row_index + 2}: mask not found: {mask_path}"
            )
        dataset_id = str(row.get("dataset_id", "") or "").strip()
        if not dataset_id:
            dataset_id = micrograph.parent.name or "dataset"
        stem = str(row.get("stem", "") or "").strip() or micrograph.stem
        records.append(
            GuiRecord(
                row_index=row_index,
                micrograph_path=micrograph,
                mask_path=mask_path,
                dataset_id=dataset_id,
                stem=stem,
            )
        )
    return manifest, rows, fieldnames, records


def _load_2d_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix in {".mrc", ".mrcs"}:
        import mrcfile

        with mrcfile.open(path, permissive=True) as mrc:
            array = mrc.data.copy()
    elif suffix in {".tif", ".tiff"}:
        from skimage.io import imread

        array = imread(path)
    else:
        raise ValueError(f"Unsupported GUI image format: {path}")
    array = np.asarray(array).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image at {path}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values found in {path}")
    return np.asarray(array, dtype=np.float32)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    step_y = max(1, source.shape[0] // 1024)
    step_x = max(1, source.shape[1] // 1024)
    sample = source[::step_y, ::step_x]
    low, high = (float(value) for value in np.percentile(sample, (1.0, 99.0)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(source, dtype=np.float32)
    return np.clip((source - low) / (high - low), 0.0, 1.0)


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "micrograph"


def _create_interactive_figure():
    import matplotlib

    errors: list[str] = []
    for backend in ("QtAgg", "TkAgg", "MacOSX"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            plt.switch_backend(backend)
            figure, axis = plt.subplots(figsize=(12.0, 9.0))
            return plt, figure, axis
        except Exception as exc:  # pragma: no cover - host GUI availability varies
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(
        "Could not start an interactive GUI backend. Install the GUI extra with "
        "`pip install -e '.[gui]'`; for a remote server, connect with X forwarding "
        "(`ssh -Y`) and confirm DISPLAY is set. Backend errors: " + "; ".join(errors)
    )


class MaskReviewGui:
    """Interactive polygon editor for inferred or user-supplied binary masks."""

    def __init__(
        self,
        *,
        manifest: Path,
        rows: list[dict[str, str]],
        fieldnames: list[str],
        records: list[GuiRecord],
        output_dir: str | Path,
        max_display_dim: int,
    ) -> None:
        self.manifest = manifest
        self.rows = rows
        self.fieldnames = list(fieldnames)
        for column in ("micrograph_path", "mask_path", "binary_mask_path", "gt_mask_path"):
            if column not in self.fieldnames:
                self.fieldnames.append(column)
        self.records = records
        for record in self.records:
            row = self.rows[record.row_index]
            row["micrograph_path"] = str(record.micrograph_path)
            if record.mask_path is not None:
                row["mask_path"] = str(record.mask_path)
                row["binary_mask_path"] = str(record.mask_path)
                row["gt_mask_path"] = str(record.mask_path)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.mask_dir = self.output_dir / "masks"
        self.output_manifest = self.output_dir / "manifest_edited.csv"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.max_display_dim = max(256, int(max_display_dim))

        self.index = 0
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.display_stride = 1
        self.polygon: list[tuple[float, float]] = []
        self.erase_mode = False
        self.selected_component: Optional[np.ndarray] = None
        self.status = ""
        self._closing = False

        self.plt, self.figure, self.axis = _create_interactive_figure()
        self.figure.subplots_adjust(bottom=0.13, top=0.92, left=0.03, right=0.97)
        try:
            self.figure.canvas.manager.set_window_title("cryoFILTER mask review")
        except Exception:
            pass
        self._add_buttons()
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._load_current()

    def _output_mask_path(self, record: GuiRecord) -> Path:
        return self.mask_dir / f"{record.row_index:05d}_{_safe_stem(record.stem)}_mask.npy"

    def _add_buttons(self) -> None:
        from matplotlib.widgets import Button

        buttons = (
            ("Previous", 0.18, lambda _event: self._move(-1)),
            ("Next", 0.30, lambda _event: self._move(1)),
            ("Finish polygon", 0.42, lambda _event: self._finish_polygon()),
            ("Delete selected", 0.58, lambda _event: self._delete_selected()),
            ("Clear", 0.74, lambda _event: self._clear()),
            ("Save", 0.84, lambda _event: self._save_current()),
        )
        self.button_axes = []
        self.buttons = []
        for label, left, callback in buttons:
            width = 0.14 if label in {"Finish polygon", "Delete selected"} else 0.10
            button_axis = self.figure.add_axes((left, 0.035, width, 0.045))
            button = Button(button_axis, label)
            button.on_clicked(callback)
            self.button_axes.append(button_axis)
            self.buttons.append(button)

    def _load_current(self) -> None:
        record = self.records[self.index]
        self.image = _load_2d_array(record.micrograph_path)
        saved_mask = self._output_mask_path(record)
        source_mask = saved_mask if saved_mask.is_file() else record.mask_path
        if source_mask is None:
            self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        else:
            loaded_mask = _load_2d_array(source_mask)
            if loaded_mask.shape != self.image.shape:
                raise ValueError(
                    f"Micrograph/mask shape mismatch for {record.stem}: "
                    f"{self.image.shape} versus {loaded_mask.shape}"
                )
            self.mask = (loaded_mask > 0).astype(np.uint8)
        self.display_stride = max(1, int(np.ceil(max(self.image.shape) / self.max_display_dim)))
        self.display_image = _normalize_for_display(self.image)[
            :: self.display_stride, :: self.display_stride
        ]
        self.polygon = []
        self.selected_component = None
        self.status = "Left-click polygon points; right-click or Enter applies the polygon. Press x to erase."
        self._redraw()

    def _redraw(self) -> None:
        if self.display_image is None or self.mask is None:
            return
        from matplotlib.colors import ListedColormap

        displayed_mask = self.mask[:: self.display_stride, :: self.display_stride] > 0
        self.axis.clear()
        self.axis.imshow(self.display_image, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        self.axis.imshow(
            np.ma.masked_where(~displayed_mask, displayed_mask),
            cmap=ListedColormap((MASK_COLOR,)),
            alpha=0.42,
            origin="upper",
            interpolation="nearest",
        )
        if np.any(displayed_mask) and np.any(~displayed_mask):
            self.axis.contour(
                displayed_mask.astype(np.float32),
                levels=(0.5,),
                colors=(MASK_EDGE_COLOR,),
                linewidths=0.7,
            )
        if self.selected_component is not None:
            selected = self.selected_component[:: self.display_stride, :: self.display_stride]
            if np.any(selected) and np.any(~selected):
                self.axis.contour(
                    selected.astype(np.float32),
                    levels=(0.5,),
                    colors=("#FDE047",),
                    linewidths=1.2,
                )
        if self.polygon:
            points = np.asarray(self.polygon, dtype=np.float32) / float(self.display_stride)
            polygon_color = "#F97316" if self.erase_mode else "#FDE047"
            self.axis.plot(points[:, 0], points[:, 1], "o-", color=polygon_color, linewidth=1.0, markersize=3)
        record = self.records[self.index]
        self.axis.set_title(
            f"{record.dataset_id} / {record.stem}  ({self.index + 1}/{len(self.records)})",
            fontsize=10,
        )
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.figure.suptitle(
            self.status
            + "  Keys: n/p next/previous, x erase-mode, s save, middle-click + d delete, c clear, q save+quit",
            fontsize=9,
        )
        self.figure.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata) * self.display_stride
        y = float(event.ydata) * self.display_stride
        if event.button == 1:
            self.polygon.append((x, y))
            self.status = f"Polygon: {len(self.polygon)} point(s)"
        elif event.button == 3:
            self._finish_polygon()
            return
        elif event.button == 2:
            self._select_component(x=x, y=y)
            return
        self._redraw()

    def _finish_polygon(self) -> None:
        if self.mask is None or len(self.polygon) < 3:
            self.status = "A contamination polygon needs at least three points."
            self._redraw()
            return
        from skimage.draw import polygon

        previous = int(np.sum(self.mask > 0))
        xs = np.asarray([point[0] for point in self.polygon])
        ys = np.asarray([point[1] for point in self.polygon])
        rr, cc = polygon(ys, xs, shape=self.mask.shape)
        if self.erase_mode:
            self.mask[rr, cc] = 0
            changed = previous - int(np.sum(self.mask > 0))
            action = "Erased"
        else:
            self.mask[rr, cc] = 1
            changed = int(np.sum(self.mask > 0)) - previous
            action = "Added"
        self.polygon = []
        self.selected_component = None
        self.status = f"{action} {changed:,} contamination pixel(s)."
        self._redraw()

    def _select_component(self, *, x: float, y: float) -> None:
        if self.mask is None:
            return
        from scipy import ndimage

        x_index = int(np.clip(round(x), 0, self.mask.shape[1] - 1))
        y_index = int(np.clip(round(y), 0, self.mask.shape[0] - 1))
        labels, _count = ndimage.label(self.mask > 0)
        label = int(labels[y_index, x_index])
        self.selected_component = labels == label if label > 0 else None
        self.status = "Selected component; press d to delete." if label > 0 else "No mask component at cursor."
        self._redraw()

    def _delete_selected(self) -> None:
        if self.mask is None or self.selected_component is None:
            self.status = "Middle-click a mask component before deleting it."
        else:
            removed = int(np.sum(self.selected_component))
            self.mask[self.selected_component] = 0
            self.selected_component = None
            self.status = f"Deleted {removed:,} mask pixel(s)."
        self._redraw()

    def _clear(self) -> None:
        if self.mask is not None:
            self.mask.fill(0)
        self.polygon = []
        self.selected_component = None
        self.status = "Cleared the current mask."
        self._redraw()

    def _save_current(self) -> None:
        if self.mask is None:
            return
        record = self.records[self.index]
        output_mask = self._output_mask_path(record)
        np.save(output_mask, self.mask.astype(np.uint8, copy=False))
        relative_mask = str(output_mask.relative_to(self.output_dir))
        row = self.rows[record.row_index]
        row["mask_path"] = relative_mask
        row["binary_mask_path"] = relative_mask
        row["gt_mask_path"] = relative_mask
        with self.output_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        self.status = f"Saved mask and {self.output_manifest.name}."
        self._redraw()

    def _move(self, delta: int) -> None:
        self._save_current()
        self.index = (self.index + int(delta)) % len(self.records)
        self._load_current()

    def _on_key(self, event) -> None:
        key = str(event.key or "").lower()
        if key == "n":
            self._move(1)
        elif key == "p":
            self._move(-1)
        elif key in {"enter", "return"}:
            self._finish_polygon()
        elif key == "d":
            self._delete_selected()
        elif key == "c":
            self._clear()
        elif key == "s":
            self._save_current()
        elif key == "x":
            self.erase_mode = not self.erase_mode
            mode = "Erase" if self.erase_mode else "Add"
            self.status = f"{mode} polygon mode."
            self._redraw()
        elif key == "escape":
            self.polygon = []
            self.status = "Cancelled polygon."
            self._redraw()
        elif key == "q":
            self._save_current()
            self._closing = True
            self.plt.close(self.figure)

    def _on_close(self, _event) -> None:
        if not self._closing:
            self._save_current()
        self._closing = True

    def show(self) -> None:
        self.plt.show()


def add_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "gui",
        help="Review or correct binary contamination masks interactively.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Inference typing manifest or training manifest containing micrograph paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="./cryofilter_gui",
        help="Directory for corrected .npy masks and manifest_edited.csv.",
    )
    parser.add_argument(
        "--micrograph-dir",
        default=None,
        help="Optional directory override used to locate micrographs by filename.",
    )
    parser.add_argument(
        "--max-display-dim",
        type=int,
        default=1600,
        help="Maximum displayed micrograph dimension; saved masks remain full-size.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    manifest, rows, fieldnames, records = load_gui_manifest(
        args.manifest,
        micrograph_dir=args.micrograph_dir,
    )
    editor = MaskReviewGui(
        manifest=manifest,
        rows=rows,
        fieldnames=fieldnames,
        records=records,
        output_dir=args.output_dir,
        max_display_dim=int(args.max_display_dim),
    )
    editor.show()
    return 0


__all__ = ["GuiRecord", "MaskReviewGui", "add_subparser", "load_gui_manifest", "run"]
