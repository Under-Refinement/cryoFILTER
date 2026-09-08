from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import yaml

from cryofilter.cli import _build_parser
from cryofilter.particle_filtering import (
    classify_particle_coordinates_by_mask,
    collect_particle_coordinates_by_micrograph,
    filter_particle_file,
)


def _write_mrc(path: Path, shape: tuple[int, int] = (100, 100), pixel_size: float = 2.0) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.zeros(shape, dtype=np.float32))
        mrc.voxel_size = pixel_size


def _write_mask(output_dir: Path, stem: str) -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    np.save(output_dir / f"{stem}_mask.npy", mask)


def _write_cs(path: Path, micrograph_name: str) -> None:
    dtype = np.dtype(
        [
            ("uid", "<u8"),
            ("location/micrograph_path", "S256"),
            ("location/center_x_frac", "<f4"),
            ("location/center_y_frac", "<f4"),
        ]
    )
    particles = np.zeros(2, dtype=dtype)
    particles["uid"] = [1, 2]
    particles["location/micrograph_path"] = micrograph_name.encode("utf-8")
    particles["location/center_x_frac"] = [0.50, 0.10]
    particles["location/center_y_frac"] = [0.50, 0.10]
    with open(path, "wb") as handle:
        np.save(handle, particles, allow_pickle=False)


def _write_blob_cs(path: Path) -> None:
    dtype = np.dtype(
        [
            ("uid", "<u8"),
            ("blob/path", "S256"),
            ("blob/idx", "<u4"),
        ]
    )
    particles = np.zeros(2, dtype=dtype)
    particles["uid"] = [1, 2]
    particles["blob/path"] = [b"stack.mrc", b"stack.mrc"]
    particles["blob/idx"] = [10, 11]
    with open(path, "wb") as handle:
        np.save(handle, particles, allow_pickle=False)


def _write_split_csg(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "created: '2026-09-03'",
                "group:",
                "  name: particles_selected",
                "  type: particle",
                "results:",
                "  alignments2D:",
                "    metafile: '>particles_selected.cs'",
                "    num_items: 2",
                "    type: particle.alignments2D",
                "  blob:",
                "    metafile: '>particles_selected.cs'",
                "    num_items: 2",
                "    type: particle.blob",
                "  ctf:",
                "    metafile: '>J12_passthrough_particles_selected.cs'",
                "    num_items: 2",
                "    type: particle.ctf",
                "  location:",
                "    metafile: '>J12_passthrough_particles_selected.cs'",
                "    num_items: 2",
                "    type: particle.location",
                "  pick_stats:",
                "    metafile: '>J12_passthrough_particles_selected.cs'",
                "    num_items: 2",
                "    type: particle.pick_stats",
                "version: v4.5.3",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_filter_cryosparc_cs_uses_final_mask(tmp_path: Path) -> None:
    micrograph = tmp_path / "micrograph_001.mrc"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_mrc(micrograph)
    _write_mask(output_dir, micrograph.stem)

    input_cs = tmp_path / "particles.cs"
    output_cs = output_dir / "particles_cryofiltered.cs"
    _write_cs(input_cs, micrograph.name)

    summary = filter_particle_file(
        particle_file=input_cs,
        output_file=output_cs,
        micrograph_paths=[micrograph],
        mask_dir=output_dir,
        exclusion_distance_angstrom=0.0,
    )

    filtered = np.load(output_cs)
    assert filtered["uid"].tolist() == [2]
    assert summary["particles_input"] == 2
    assert summary["particles_removed"] == 1
    assert summary["particles_kept"] == 1
    assert summary["exclusion_distance_coordinate_system"] == "original_micrograph"


def test_filter_split_cryosparc_csg_from_blob_cs_uses_location_metafile(tmp_path: Path) -> None:
    micrograph = tmp_path / "micrograph_001.mrc"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_mrc(micrograph)
    _write_mask(output_dir, micrograph.stem)

    blob_cs = tmp_path / "cryosparc_P280_particles_selected.cs"
    passthrough_cs = tmp_path / "cryosparc_P280_J12_passthrough_particles_selected.cs"
    csg = tmp_path / "J12_particles_selected.csg"
    output_cs = output_dir / "cryosparc_P280_particles_selected_cryofiltered.cs"
    _write_blob_cs(blob_cs)
    _write_cs(passthrough_cs, micrograph.name)
    _write_split_csg(csg)

    particle_index = collect_particle_coordinates_by_micrograph(
        particle_file=blob_cs,
        micrograph_paths=[micrograph],
        csg_file=csg,
    )
    assert particle_index["input_particle_file"] == str(blob_cs.resolve())
    assert particle_index["location_particle_file"] == str(passthrough_cs.resolve())
    assert particle_index["coordinates_by_micrograph"][micrograph.resolve()].tolist() == [
        [50.0, 50.0],
        [10.0, 10.0],
    ]

    summary = filter_particle_file(
        particle_file=blob_cs,
        output_file=output_cs,
        micrograph_paths=[micrograph],
        mask_dir=output_dir,
        exclusion_distance_angstrom=0.0,
        csg_file=csg,
    )

    output_passthrough = output_dir / "cryosparc_P280_J12_passthrough_particles_selected_cryofiltered.cs"
    output_csg = output_cs.with_suffix(".csg")
    filtered_blob = np.load(output_cs)
    filtered_passthrough = np.load(output_passthrough)
    csg_payload = yaml.safe_load(output_csg.read_text(encoding="utf-8"))

    assert filtered_blob["uid"].tolist() == [2]
    assert filtered_passthrough["uid"].tolist() == [2]
    assert summary["location_particle_file"] == str(passthrough_cs.resolve())
    assert summary["output_particle_file"] == str(output_cs.resolve())
    assert summary["output_location_particle_file"] == str(output_passthrough.resolve())
    assert set(summary["output_particle_files"]) == {str(output_cs.resolve()), str(output_passthrough.resolve())}
    assert summary["output_csg_file"] == str(output_csg.resolve())
    assert csg_payload["results"]["blob"]["metafile"] == f">{output_cs.name}"
    assert csg_payload["results"]["alignments2D"]["metafile"] == f">{output_cs.name}"
    assert csg_payload["results"]["location"]["metafile"] == f">{output_passthrough.name}"
    assert csg_payload["results"]["ctf"]["num_items"] == 1
    assert summary["particles_input"] == 2
    assert summary["particles_removed"] == 1
    assert summary["particles_kept"] == 1


def test_collect_particle_coordinates_and_classify_mask(tmp_path: Path) -> None:
    micrograph = tmp_path / "micrograph_001.mrc"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_mrc(micrograph)
    _write_mask(output_dir, micrograph.stem)
    input_cs = tmp_path / "particles.cs"
    _write_cs(input_cs, micrograph.name)

    particle_index = collect_particle_coordinates_by_micrograph(
        particle_file=input_cs,
        micrograph_paths=[micrograph],
    )
    coords = particle_index["coordinates_by_micrograph"][micrograph.resolve()]
    mask = np.load(output_dir / f"{micrograph.stem}_mask.npy")
    keep, distances = classify_particle_coordinates_by_mask(
        coords,
        bad_mask=mask,
        input_shape=(100, 100),
        pixel_size_angstrom=2.0,
        exclusion_distance_angstrom=0.0,
    )

    assert coords.tolist() == [[50.0, 50.0], [10.0, 10.0]]
    assert keep.tolist() == [False, True]
    assert distances[0] == 0.0
    assert distances[1] > 0.0


def test_filter_relion_star_preserves_clean_particle(tmp_path: Path) -> None:
    micrograph = tmp_path / "micrograph_001.mrc"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_mrc(micrograph)
    _write_mask(output_dir, micrograph.stem)

    input_star = tmp_path / "particles.star"
    output_star = output_dir / "particles_cryofiltered.star"
    input_star.write_text(
        "\n".join(
            [
                "data_particles",
                "",
                "loop_",
                "_rlnMicrographName #1",
                "_rlnCoordinateX #2",
                "_rlnCoordinateY #3",
                f"{micrograph.name} 50 50",
                f"{micrograph.name} 10 10",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = filter_particle_file(
        particle_file=input_star,
        output_file=output_star,
        micrograph_paths=[micrograph],
        mask_dir=output_dir,
        exclusion_distance_angstrom=0.0,
    )

    text = output_star.read_text(encoding="utf-8")
    assert f"{micrograph.name} 50 50" not in text
    assert f"{micrograph.name} 10 10" in text
    assert summary["particles_removed"] == 1
    assert summary["particles_kept"] == 1


def test_collect_relion_star_particle_coordinates(tmp_path: Path) -> None:
    micrograph = tmp_path / "micrograph_001.mrc"
    _write_mrc(micrograph)
    input_star = tmp_path / "particles.star"
    input_star.write_text(
        "\n".join(
            [
                "data_particles",
                "",
                "loop_",
                "_rlnMicrographName #1",
                "_rlnCoordinateX #2",
                "_rlnCoordinateY #3",
                f"{micrograph.name} 51 51",
                f"{micrograph.name} 11 11",
                "",
            ]
        ),
        encoding="utf-8",
    )

    particle_index = collect_particle_coordinates_by_micrograph(
        particle_file=input_star,
        micrograph_paths=[micrograph],
    )
    coords = particle_index["coordinates_by_micrograph"][micrograph.resolve()]

    assert particle_index["format"] == "relion_star"
    assert coords.tolist() == [[50.0, 50.0], [10.0, 10.0]]


def test_unmatched_particles_fail_closed_by_default(tmp_path: Path) -> None:
    micrograph = tmp_path / "micrograph_001.mrc"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    _write_mrc(micrograph)
    _write_mask(output_dir, micrograph.stem)
    input_cs = tmp_path / "particles.cs"
    output_cs = output_dir / "particles_cryofiltered.cs"
    _write_cs(input_cs, "different_micrograph.mrc")

    with pytest.raises(ValueError, match="Could not match"):
        filter_particle_file(
            particle_file=input_cs,
            output_file=output_cs,
            micrograph_paths=[micrograph],
            mask_dir=output_dir,
            exclusion_distance_angstrom=100.0,
        )
    assert not output_cs.exists()


def test_infer_parser_accepts_optional_particle_filtering() -> None:
    args = _build_parser().parse_args(
        [
            "infer",
            "--input",
            "micrographs",
            "--particle-file",
            "particles.cs",
            "--filtered-particle-file",
            "filtered.cs",
            "--render-particle-overlays",
        ]
    )

    assert args.particle_file == "particles.cs"
    assert args.filtered_particle_file == "filtered.cs"
    assert args.particle_exclusion_distance_angstrom == 100.0
    assert args.render_particle_overlays is True
    assert args.particle_overlay_raw_panel is True
    assert args.particle_overlay_probability_panel is True

    default_args = _build_parser().parse_args(
        [
            "infer",
            "--input",
            "micrographs",
            "--particle-file",
            "particles.cs",
        ]
    )
    assert default_args.render_particle_overlays is None

    disabled_args = _build_parser().parse_args(
        [
            "infer",
            "--input",
            "micrographs",
            "--particle-file",
            "particles.cs",
            "--no-render-particle-overlays",
        ]
    )
    assert disabled_args.render_particle_overlays is False


def test_filter_particles_parser_accepts_existing_masks_workflow() -> None:
    args = _build_parser().parse_args(
        [
            "filter-particles",
            "--input",
            "micrographs",
            "--mask-dir",
            "cryofilter_output",
            "--particle-file",
            "particles.cs",
            "--particle-csg",
            "particles.csg",
            "--filtered-particle-file",
            "filtered.cs",
            "--recursive",
        ]
    )

    assert args.command == "filter-particles"
    assert args.input == "micrographs"
    assert args.mask_dir == "cryofilter_output"
    assert args.particle_file == "particles.cs"
    assert args.particle_csg == "particles.csg"
    assert args.filtered_particle_file == "filtered.cs"
    assert args.particle_exclusion_distance_angstrom == 100.0
    assert args.recursive is True
