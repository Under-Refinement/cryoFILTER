import json
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd

from cryofilter.typing_cli import main as typing_main
from utils.contamination_typing import assign_frequency_component_type, categorize_contamination_type


def _blank_mask() -> np.ndarray:
    return np.zeros((256, 256), dtype=np.uint8)


def test_corner_wedge_is_classified_as_edge_artifact() -> None:
    mask = _blank_mask()
    for row in range(140):
        mask[row, : max(0, 140 - row)] = 1

    assert categorize_contamination_type(mask) == "edge_artifact"


def test_long_single_edge_strip_is_classified_as_edge_artifact() -> None:
    mask = _blank_mask()
    mask[:48, 20:236] = 1

    assert categorize_contamination_type(mask) == "edge_artifact"


def test_mid_edge_strip_without_corner_still_counts_as_edge_artifact() -> None:
    mask = _blank_mask()
    mask[:22, 82:178] = 1

    assert categorize_contamination_type(mask) == "edge_artifact"


def test_bottom_right_corner_component_with_discrete_span_still_counts_as_edge_artifact() -> None:
    mask = _blank_mask()
    mask[206:256, 200:256] = 1

    assert categorize_contamination_type(mask) == "edge_artifact"


def test_compact_corner_beam_edge_still_counts_as_edge_artifact() -> None:
    mask = _blank_mask()
    mask[231:256, 203:256] = 1
    mask[203:256, 231:256] = 1

    assert categorize_contamination_type(mask) == "edge_artifact"


def test_border_touch_without_long_span_does_not_trigger_edge_artifact() -> None:
    mask = _blank_mask()
    mask[:18, :24] = 1

    assert categorize_contamination_type(mask) == "clean"


def test_large_interior_blob_near_edge_band_stays_large_contam() -> None:
    mask = _blank_mask()
    mask[18:150, 40:190] = 1

    assert categorize_contamination_type(mask) == "large_contam"


def test_small_interior_component_stays_small_contam() -> None:
    mask = _blank_mask()
    mask[100:130, 100:130] = 1

    assert categorize_contamination_type(mask) == "small_contam"


def test_fragmented_medium_components_count_as_large_contam() -> None:
    mask = _blank_mask()
    mask[40:68, 40:68] = 1
    mask[160:188, 160:188] = 1

    assert categorize_contamination_type(mask) == "large_contam"


def test_fragmented_specks_below_large_burden_stay_small_contam() -> None:
    mask = _blank_mask()
    mask[44:66, 44:66] = 1
    mask[164:186, 164:186] = 1

    assert categorize_contamination_type(mask) == "small_contam"


def test_fragmented_interior_burden_without_edge_support_stays_non_edge() -> None:
    mask = _blank_mask()
    for top, left in ((40, 40), (40, 100), (40, 160), (120, 60), (120, 140)):
        mask[top : top + 22, left : left + 22] = 1

    assert categorize_contamination_type(mask) == "large_contam"


def test_edge_artifact_still_overrides_large_interior_burden() -> None:
    mask = _blank_mask()
    mask[:22, 20:120] = 1
    mask[100:170, 100:170] = 1

    assert categorize_contamination_type(mask) == "edge_artifact"


def test_non_edge_components_keep_three_way_frequency_split() -> None:
    assert assign_frequency_component_type(
        edge_artifact=False,
        crystalline_score=1.8,
        diffuse_score=0.4,
        isolated_score=0.1,
    )[0] == "crystalline"
    assert assign_frequency_component_type(
        edge_artifact=False,
        crystalline_score=0.1,
        diffuse_score=1.6,
        isolated_score=0.8,
    )[0] == "diffuse"
    assert assign_frequency_component_type(
        edge_artifact=False,
        crystalline_score=-0.2,
        diffuse_score=0.4,
        isolated_score=1.2,
    )[0] == "isolated"


def test_edge_components_override_non_edge_scores_only_when_rule_is_hit() -> None:
    assert assign_frequency_component_type(
        edge_artifact=True,
        crystalline_score=0.8,
        diffuse_score=1.2,
        isolated_score=0.1,
    )[0] == "edge/carbon"


def _write_typing_mrc(path: Path) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.zeros((100, 100), dtype=np.float32))
        mrc.voxel_size = 2.0


def test_public_typing_uses_required_names_and_handles_clean_images(
    tmp_path: Path,
    capsys,
) -> None:
    clean_mrc = tmp_path / "clean.mrc"
    ethane_mrc = tmp_path / "ethane.mrc"
    _write_typing_mrc(clean_mrc)
    _write_typing_mrc(ethane_mrc)

    clean_mask = np.zeros((100, 100), dtype=np.uint8)
    ethane_mask = np.zeros((100, 100), dtype=np.uint8)
    ethane_mask[48:52, 48:52] = 1
    clean_mask_path = tmp_path / "clean_mask.npy"
    ethane_mask_path = tmp_path / "ethane_mask.npy"
    np.save(clean_mask_path, clean_mask)
    np.save(ethane_mask_path, ethane_mask)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "stem": "clean",
                "micrograph_path": clean_mrc,
                "binary_mask_path": clean_mask_path,
                "pixel_size_angstrom": 2.0,
            },
            {
                "dataset_id": "demo",
                "stem": "ethane",
                "micrograph_path": ethane_mrc,
                "binary_mask_path": ethane_mask_path,
                "pixel_size_angstrom": 2.0,
            },
        ]
    ).to_csv(manifest, index=False)

    output_dir = tmp_path / "typing"
    assert typing_main(["--manifest", str(manifest), "--output-dir", str(output_dir)]) == 0

    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["type_order"] == ["Carbon", "Crystalline", "Aggregate", "Ethane"]
    assert payload["n_images_expected"] == 2
    assert payload["typing_status"] == "complete"
    assert payload["type_to_id"] == {
        "Carbon": 1,
        "Crystalline": 2,
        "Aggregate": 3,
        "Ethane": 4,
    }
    component_df = pd.read_csv(output_dir / "component_type_assignments.csv")
    assert component_df["type"].tolist() == ["Ethane"]
    assert set(component_df["type"]) <= {"Carbon", "Crystalline", "Aggregate", "Ethane"}
    typed = np.load(output_dir / "typed_masks" / "demo__ethane_typed_mask.npy")
    assert set(np.unique(typed)) == {0, 4}

    stdout = capsys.readouterr().out
    assert "live summary updated" in stdout
    assert "Total contamination:" in stdout
    assert "Carbon:" in stdout
    assert "Crystalline:" in stdout
    assert "Aggregate:" in stdout
    assert "Ethane:" in stdout
