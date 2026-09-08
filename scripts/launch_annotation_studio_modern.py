#!/usr/bin/env python3
"""Thin launcher wrapper for the modern contamination-labeling studio."""

from __future__ import annotations

from typing import Optional

from scripts.gt_mask_editor_modern import SimpleMaskEditor


def launch_annotation_studio(
    *,
    manifest_path: str,
    mic_dir: Optional[str] = None,
    output_root: str = "annotation_exports",
    output_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    title: str = "cryoFILTER Annotation Studio",
    val_manifest: Optional[str] = None,
    initial_split: str = "ALL",
    initial_dataset_filter: str = "",
    initial_row_key: str = "",
    export_only_modified: bool = False,
    prefetch_workers: Optional[int] = None,
    prefetch_radius: int = 0,
    row_cache_size: int = 8,
) -> None:
    editor = SimpleMaskEditor(
        manifest_path=manifest_path,
        mic_dir=mic_dir,
        output_dir=output_dir,
        output_root=output_root,
        output_name=output_name,
        val_manifest=val_manifest,
        initial_split=initial_split,
        initial_dataset_filter=initial_dataset_filter,
        initial_row_key=initial_row_key,
        export_only_modified=export_only_modified,
        prefetch_workers=prefetch_workers,
        prefetch_radius=prefetch_radius,
        row_cache_size=row_cache_size,
    )
    try:
        editor.fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    editor.run()


def main() -> None:
    from scripts.gt_mask_editor_modern import main as editor_main

    editor_main()


if __name__ == "__main__":
    main()
