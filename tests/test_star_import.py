from __future__ import annotations

import pytest

from utils.star_import import prepare_star_for_cryosparc_import, strip_cryosparc_uid_prefix


def test_prepare_star_for_cryosparc_import_strips_uid_and_adds_pixel_size(tmp_path) -> None:
    input_star = tmp_path / "particles.star"
    output_star = tmp_path / "particles_cryosparc.star"
    input_star.write_text(
        "\n".join(
            [
                "data_particles",
                "",
                "loop_",
                "_rlnMicrographName #1",
                "_rlnCoordinateX #2",
                "_rlnCoordinateY #3",
                "000012076402221568849_FoilHole_24136930_Data_24136393_patch_aligned.mrc 3328.000 3640.000",
                "000012076402221568850_FoilHole_24136931_Data_24136394_patch_aligned.mrc 100.000 200.000",
                "",
            ]
        )
    )

    summary = prepare_star_for_cryosparc_import(
        input_star,
        output_star,
        pixel_size_angstrom=1.03,
    )

    text = output_star.read_text()
    assert "_rlnImagePixelSize #4" in text
    assert "000012076402221568849_" not in text
    assert "FoilHole_24136930_Data_24136393_patch_aligned.mrc 3328.000 3640.000 1.030000" in text
    assert summary.n_rows == 2
    assert summary.n_micrograph_names_changed == 2
    assert summary.n_uid_prefixes_stripped == 2
    assert summary.added_rln_image_pixel_size is True


def test_strip_cryosparc_uid_prefix_preserves_date_prefixed_exposure_name() -> None:
    normalized, stripped = strip_cryosparc_uid_prefix("20200224_FoilHole_24136930_patch_aligned.mrc")

    assert normalized == "20200224_FoilHole_24136930_patch_aligned.mrc"
    assert stripped is False


def test_prepare_star_for_cryosparc_import_requires_pixel_size_when_missing(tmp_path) -> None:
    input_star = tmp_path / "particles.star"
    output_star = tmp_path / "particles_cryosparc.star"
    input_star.write_text(
        "\n".join(
            [
                "data_particles",
                "",
                "loop_",
                "_rlnMicrographName #1",
                "_rlnCoordinateX #2",
                "_rlnCoordinateY #3",
                "FoilHole_24136930_patch_aligned.mrc 1.0 2.0",
                "",
            ]
        )
    )

    with pytest.raises(ValueError, match="pixel_size_angstrom"):
        prepare_star_for_cryosparc_import(input_star, output_star)
