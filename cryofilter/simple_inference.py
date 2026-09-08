"""Simple external-user launcher for cryoFILTER inference."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from cryofilter.cli import (
    DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS,
    DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY,
    DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE,
    DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX,
    DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM,
    DEFAULT_PARTICLE_OVERLAY_SUBDIR,
    DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM,
    main as cryofilter_main,
)
from utils.small_pixel_policy import DEFAULT_INFERENCE_PROFILE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryofilter-inference",
        description="Simple launcher for the public cryoFILTER FULL checkpoint.",
    )
    parser.add_argument(
        "--input_directory",
        "--input_dir",
        "--input_path",
        "--input_file",
        "--input",
        dest="input_path",
        required=True,
        help="Directory containing .mrc micrographs, or one .mrc file.",
    )
    parser.add_argument(
        "--output_directory",
        "--output_dir",
        "--output",
        dest="output_dir",
        default=None,
        help="Optional output directory. Defaults to <input>_cryofilter_output.",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        dest="checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to the released FULL checkpoint when available.",
    )
    parser.add_argument("--device", default=None, help="Optional device override, for example cuda or cpu.")
    parser.add_argument(
        "--pixel_size_angstrom",
        type=float,
        default=None,
        help="Optional pixel size override in Angstrom per pixel.",
    )
    parser.add_argument(
        "--inference_profile",
        default=DEFAULT_INFERENCE_PROFILE,
        choices=("fast", "balanced", "quality"),
        help="Small-pixel automatic profile. Default: balanced.",
    )
    parser.add_argument(
        "--particle_exclusion_distance_angstrom",
        type=float,
        default=DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM,
        help=(
            "Particle-exclusion distance used when --particle_file is supplied and recorded in summaries. "
            f"Default: {DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM:.0f} A."
        ),
    )
    parser.add_argument(
        "--particle_file",
        "--particle-file",
        default=None,
        help="Optional CryoSPARC .cs or RELION coordinate .star file to filter after inference.",
    )
    parser.add_argument(
        "--filtered_particle_file",
        "--filtered-particle-file",
        default=None,
        help="Optional output path for the filtered .cs or .star file.",
    )
    parser.add_argument(
        "--particle_csg",
        "--particle-csg",
        default=None,
        help="Optional CryoSPARC .csg template paired with --particle_file.",
    )
    parser.add_argument(
        "--allow_unmatched_particles",
        "--allow-unmatched-particles",
        action="store_true",
        help="Preserve particle rows whose micrograph names do not match the inference inputs.",
    )
    parser.add_argument(
        "--overwrite_filtered_particles",
        "--overwrite-filtered-particles",
        action="store_true",
        help="Replace an existing filtered particle output.",
    )
    parser.add_argument(
        "--render_particle_overlays",
        "--render-particle-overlays",
        dest="render_particle_overlays",
        action="store_true",
        default=None,
        help=(
            "With --particle_file, write live QC PNGs showing the predicted mask, "
            "kept particles, rejected particles, and probability map. Default: on when --particle_file is supplied."
        ),
    )
    parser.add_argument(
        "--no_render_particle_overlays",
        "--no-render-particle-overlays",
        dest="render_particle_overlays",
        action="store_false",
        help="Disable live QC PNG generation when filtering a particle file.",
    )
    parser.add_argument(
        "--particle_overlay_dir",
        "--particle-overlay-dir",
        default=None,
        help=f"Directory for live QC PNGs. Defaults to <output-dir>/{DEFAULT_PARTICLE_OVERLAY_SUBDIR}.",
    )
    parser.add_argument(
        "--particle_overlay_max_display_dim",
        "--particle-overlay-max-display-dim",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM,
        help="Longest display edge for particle overlay PNGs.",
    )
    parser.add_argument(
        "--particle_overlay_diameter_px",
        "--particle-overlay-diameter-px",
        type=float,
        default=DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX,
        help="Display-space particle marker diameter.",
    )
    parser.add_argument(
        "--particle_overlay_raw_panel",
        "--particle-overlay-raw-panel",
        dest="particle_overlay_raw_panel",
        action="store_true",
        default=True,
        help="Include the raw micrograph as the left panel in each live particle-overlay PNG. Default: on.",
    )
    parser.add_argument(
        "--no_particle_overlay_raw_panel",
        "--no-particle-overlay-raw-panel",
        dest="particle_overlay_raw_panel",
        action="store_false",
        help="Write compact overlay-only live particle QC PNGs.",
    )
    parser.add_argument(
        "--particle_overlay_probability_panel",
        "--particle-overlay-probability-panel",
        dest="particle_overlay_probability_panel",
        action="store_true",
        default=True,
        help="Include the viridis probability map as the rightmost live particle-overlay panel. Default: on.",
    )
    parser.add_argument(
        "--no_particle_overlay_probability_panel",
        "--no-particle-overlay-probability-panel",
        dest="particle_overlay_probability_panel",
        action="store_false",
        help="Do not include the probability-map panel in live particle QC PNGs.",
    )
    parser.add_argument(
        "--particle_overlay_contact_sheet_cols",
        "--particle-overlay-contact-sheet-cols",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS,
        help="Number of columns in the live contact sheet.",
    )
    parser.add_argument(
        "--particle_overlay_contact_sheet_tile",
        "--particle-overlay-contact-sheet-tile",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE,
        help="Tile size in pixels for the live contact sheet.",
    )
    parser.add_argument(
        "--particle_overlay_contact_sheet_every",
        "--particle-overlay-contact-sheet-every",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY,
        help="Refresh the contact sheet after this many newly rendered frames.",
    )
    parser.add_argument(
        "--no_particle_overlay_contact_sheet",
        "--no-particle-overlay-contact-sheet",
        dest="particle_overlay_contact_sheet",
        action="store_false",
        default=True,
        help="Write per-micrograph overlay PNGs only.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively discover .mrc files under the input directory. Default: on.",
    )
    parser.add_argument(
        "--no-recursive",
        "--no_recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top level of the input directory.",
    )
    parser.add_argument(
        "--save_mrc",
        dest="save_mrc",
        action="store_true",
        default=True,
        help="Write .mrc copies alongside .npy outputs. Default: on.",
    )
    parser.add_argument(
        "--no-save-mrc",
        "--no_save_mrc",
        dest="save_mrc",
        action="store_false",
        help="Write only .npy outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        default=False,
        help="Skip micrographs whose expected outputs already exist.",
    )
    return parser


def _build_forwarded_argv(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "infer",
        "--input",
        str(args.input_path),
        "--inference-profile",
        str(args.inference_profile),
        "--particle-exclusion-distance-angstrom",
        str(float(args.particle_exclusion_distance_angstrom)),
    ]
    if bool(args.recursive):
        forwarded.append("--recursive")
    if args.output_dir:
        forwarded.extend(["--output-dir", str(args.output_dir)])
    if args.checkpoint:
        forwarded.extend(["--checkpoint", str(args.checkpoint)])
    if args.device:
        forwarded.extend(["--device", str(args.device)])
    if args.pixel_size_angstrom is not None:
        forwarded.extend(["--pixel-size-angstrom", str(float(args.pixel_size_angstrom))])
    if args.particle_file:
        forwarded.extend(["--particle-file", str(args.particle_file)])
    if args.filtered_particle_file:
        forwarded.extend(["--filtered-particle-file", str(args.filtered_particle_file)])
    if args.particle_csg:
        forwarded.extend(["--particle-csg", str(args.particle_csg)])
    if bool(args.allow_unmatched_particles):
        forwarded.append("--allow-unmatched-particles")
    if bool(args.overwrite_filtered_particles):
        forwarded.append("--overwrite-filtered-particles")
    if args.render_particle_overlays is True:
        forwarded.append("--render-particle-overlays")
    elif args.render_particle_overlays is False:
        forwarded.append("--no-render-particle-overlays")
    if args.particle_overlay_dir:
        forwarded.extend(["--particle-overlay-dir", str(args.particle_overlay_dir)])
    if int(args.particle_overlay_max_display_dim) != DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM:
        forwarded.extend(["--particle-overlay-max-display-dim", str(int(args.particle_overlay_max_display_dim))])
    if float(args.particle_overlay_diameter_px) != DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX:
        forwarded.extend(["--particle-overlay-diameter-px", str(float(args.particle_overlay_diameter_px))])
    if not bool(args.particle_overlay_raw_panel):
        forwarded.append("--no-particle-overlay-raw-panel")
    if not bool(args.particle_overlay_probability_panel):
        forwarded.append("--no-particle-overlay-probability-panel")
    if int(args.particle_overlay_contact_sheet_cols) != DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS:
        forwarded.extend(["--particle-overlay-contact-sheet-cols", str(int(args.particle_overlay_contact_sheet_cols))])
    if int(args.particle_overlay_contact_sheet_tile) != DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE:
        forwarded.extend(["--particle-overlay-contact-sheet-tile", str(int(args.particle_overlay_contact_sheet_tile))])
    if int(args.particle_overlay_contact_sheet_every) != DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY:
        forwarded.extend(["--particle-overlay-contact-sheet-every", str(int(args.particle_overlay_contact_sheet_every))])
    if not bool(args.particle_overlay_contact_sheet):
        forwarded.append("--no-particle-overlay-contact-sheet")
    if not bool(args.save_mrc):
        forwarded.append("--no-save-mrc")
    if bool(args.skip_existing):
        forwarded.append("--skip-existing")
    return forwarded


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return cryofilter_main(_build_forwarded_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
