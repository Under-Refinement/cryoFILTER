from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import mrcfile
import numpy as np

from cryofilter.cli import (
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM,
    DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS,
    DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY,
    DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE,
    DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX,
    DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM,
    DEFAULT_PARTICLE_OVERLAY_SUBDIR,
    DEFAULT_PUBLIC_ADAPTIVE_DOWNSAMPLE_METHOD,
    DEFAULT_PUBLIC_BLENDING_EDGE_PX,
    DEFAULT_PUBLIC_BLENDING_WINDOW,
    DEFAULT_PUBLIC_NORMALIZATION_METHOD,
    DEFAULT_PUBLIC_OVERLAP,
    DEFAULT_PUBLIC_TARGET_PIXEL_SIZE,
    _cpu_threads_per_gpu_worker,
    _build_parser,
    _default_output_dir_for_input,
    _merge_worker_summaries,
    _multi_gpu_worker,
    _resolve_auto_batch_forward_size,
    _resolve_inference_devices,
    _run_multi_gpu_infer,
    _shard_mrc_paths,
    _write_typing_manifest,
)
from utils.small_pixel_policy import (
    DEFAULT_INFERENCE_PROFILE,
    DEFAULT_SMALL_PIXEL_CUTOFF_ANGSTROM,
    resolve_small_pixel_inference_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_install_environment_includes_cryosparc_tools() -> None:
    environment_text = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "cryosparc-tools>=4.1" in environment_text
    assert "cryosparc-bridge" in pyproject_text
    assert "cryosparc-tools>=4.1" in pyproject_text
    assert "cryofilter-install-cryosparc-tools" in pyproject_text


def _policy_for_pixel_size(
    pixel_size_angstrom: float,
    *,
    inference_profile: str = DEFAULT_INFERENCE_PROFILE,
    normalization_method: str = DEFAULT_PUBLIC_NORMALIZATION_METHOD,
):
    return resolve_small_pixel_inference_policy(
        pixel_size_angstrom=pixel_size_angstrom,
        standard_inference_recipe="adaptive_resample",
        standard_target_pixel_size=DEFAULT_PUBLIC_TARGET_PIXEL_SIZE,
        standard_multiscale_targets=(DEFAULT_PUBLIC_TARGET_PIXEL_SIZE,),
        standard_normalization_method=normalization_method,
        standard_overlap=DEFAULT_PUBLIC_OVERLAP,
        standard_adaptive_downsample=True,
        standard_adaptive_downsample_method=DEFAULT_PUBLIC_ADAPTIVE_DOWNSAMPLE_METHOD,
        standard_multiscale_psd_source="patch",
        standard_blending_window=DEFAULT_PUBLIC_BLENDING_WINDOW,
        standard_blending_edge_px=DEFAULT_PUBLIC_BLENDING_EDGE_PX,
        standard_use_tta=False,
        inference_profile=inference_profile,
    )


def test_rtx_4090_uses_fast_auto_batch_despite_reported_memory_rounding() -> None:
    assert _resolve_auto_batch_forward_size("NVIDIA GeForce RTX 4090", 23.988) == 32
    assert _resolve_auto_batch_forward_size("Unknown 24GB-ish GPU", 23.988) == 32
    assert _resolve_auto_batch_forward_size("NVIDIA RTX A4000", 16.0) == 16
    assert _resolve_auto_batch_forward_size("small GPU", 12.0) == 8


def test_cpu_threads_are_split_across_gpu_workers() -> None:
    assert _cpu_threads_per_gpu_worker(None, 4) is None
    assert _cpu_threads_per_gpu_worker(32, 4) == 8
    assert _cpu_threads_per_gpu_worker(10, 4) == 2
    assert _cpu_threads_per_gpu_worker(2, 4) == 1


def test_small_pixel_policy_triggers_for_1106_apix() -> None:
    assert DEFAULT_SMALL_PIXEL_CUTOFF_ANGSTROM >= 1.106
    policy = _policy_for_pixel_size(1.106)
    assert policy.policy_name == "small_pixel_auto"
    assert policy.target_pixel_size == 2.0
    assert policy.multiscale_targets == (2.0,)
    assert policy.inference_profile == "balanced"
    assert policy.overlap == 160
    assert policy.use_tta is False
    assert policy.adaptive_downsample_method == "fourier"
    assert policy.multiscale_psd_source == "patch"


def test_standard_policy_keeps_release_recipe_for_non_triggering_pixel_size() -> None:
    policy = _policy_for_pixel_size(1.50)

    assert policy.policy_name == "standard"
    assert policy.target_pixel_size == DEFAULT_PUBLIC_TARGET_PIXEL_SIZE
    assert policy.multiscale_targets == (DEFAULT_PUBLIC_TARGET_PIXEL_SIZE,)
    assert policy.overlap == DEFAULT_PUBLIC_OVERLAP
    assert policy.normalization_method == DEFAULT_PUBLIC_NORMALIZATION_METHOD
    assert policy.adaptive_downsample_method == DEFAULT_PUBLIC_ADAPTIVE_DOWNSAMPLE_METHOD
    assert policy.multiscale_psd_source == "patch"
    assert policy.blending_window == DEFAULT_PUBLIC_BLENDING_WINDOW
    assert policy.blending_edge_px == DEFAULT_PUBLIC_BLENDING_EDGE_PX
    assert policy.use_tta is False


def test_quality_profile_preserves_manuscript_small_pixel_recipe() -> None:
    policy = _policy_for_pixel_size(1.106, inference_profile="quality")

    assert policy.policy_name == "small_pixel_auto"
    assert policy.inference_profile == "quality"
    assert policy.target_pixel_size == 2.0
    assert policy.overlap == 192
    assert policy.use_tta is True
    assert policy.normalization_method == "percentile_extra_wide"


def test_fast_profile_uses_low_overlap_without_tta() -> None:
    policy = _policy_for_pixel_size(1.106, inference_profile="fast")

    assert policy.policy_name == "small_pixel_auto"
    assert policy.inference_profile == "fast"
    assert policy.overlap == 128
    assert policy.use_tta is False


def test_explicit_normalization_is_respected_by_small_pixel_profile() -> None:
    policy = _policy_for_pixel_size(1.106, normalization_method="robust")

    assert policy.policy_name == "small_pixel_auto"
    assert policy.normalization_method == "robust"


def test_infer_parser_aliases_and_safer_defaults() -> None:
    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None) and "infer" in action.choices
    )
    infer_parser = subparsers.choices["infer"]
    args = parser.parse_args(["infer", "--input-directory", "mic_dir", "--tta", "4", "--skip-existing"])

    assert DEFAULT_MASK_THRESHOLD == 0.60
    assert args.threshold == DEFAULT_MASK_THRESHOLD
    assert args.checkpoint is None
    assert args.output_dir is None
    assert args.inference_profile == "balanced"
    assert args.input == "mic_dir"
    assert args.particle_exclusion_distance_angstrom == DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM
    assert args.target_pixel_size == DEFAULT_PUBLIC_TARGET_PIXEL_SIZE
    assert args.overlap == DEFAULT_PUBLIC_OVERLAP
    assert args.normalization_method == DEFAULT_PUBLIC_NORMALIZATION_METHOD
    assert args.adaptive_downsample_method == DEFAULT_PUBLIC_ADAPTIVE_DOWNSAMPLE_METHOD
    assert args.blending_window == DEFAULT_PUBLIC_BLENDING_WINDOW
    assert args.blending_edge_px == DEFAULT_PUBLIC_BLENDING_EDGE_PX
    assert args.tta == 4
    assert args.skip_existing is True
    assert args.small_pixel_auto is True
    assert args.gpus == ""
    assert args.export_masks is True
    assert args.no_resample is False
    assert args.render_images is False
    assert args.image_output_dir is None
    assert args.image_max_dim == 1400
    assert args.image_dpi == 180
    assert args.render_particle_overlays is None
    assert args.particle_overlay_dir is None
    assert DEFAULT_PARTICLE_OVERLAY_SUBDIR == "OTF_images"
    assert args.particle_overlay_max_display_dim == DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM
    assert args.particle_overlay_diameter_px == DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX
    assert args.particle_overlay_raw_panel is True
    assert args.particle_overlay_probability_panel is True
    assert args.particle_overlay_contact_sheet is True
    assert args.particle_overlay_contact_sheet_cols == DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS
    assert args.particle_overlay_contact_sheet_tile == DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE
    assert args.particle_overlay_contact_sheet_every == DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY
    model_type_action = next(action for action in infer_parser._actions if action.dest == "model_type")
    assert set(model_type_action.choices) == {
        "auto",
        "unet_attention",
        "unet",
        "resnet_unet",
        "simple",
    }
    help_text = infer_parser.format_help()
    assert "patch-local PSD" in help_text
    assert "full-image PSD" not in help_text
    assert "four-panel PNG" in help_text
    assert "raw micrograph" in help_text
    assert "probability map" in help_text
    assert "on-the-fly QC PNG" in help_text
    assert "OTF_images" in help_text
    particle_distance_action = next(
        action
        for action in infer_parser._actions
        if action.dest == "particle_exclusion_distance_angstrom"
    )
    assert "original particle-extraction box" in particle_distance_action.help
    assert "Fourier-cropped box" in particle_distance_action.help

    disabled = parser.parse_args(
        [
            "infer",
            "--input-directory",
            "mic_dir",
            "--particle-file",
            "particles.star",
            "--no-render-particle-overlays",
        ]
    )
    assert disabled.render_particle_overlays is False

    no_export = parser.parse_args(["infer", "--input-directory", "mic_dir", "--no-export-masks"])
    assert no_export.export_masks is False


def test_multi_gpu_parser_sharding_and_summary_order(tmp_path: Path, monkeypatch) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["infer", "--input", "mic_dir", "--gpus", "0,1"]
    )
    assert args.gpus == "0,1"
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    assert _resolve_inference_devices(args.gpus) == ["cuda:0", "cuda:1"]
    assert _resolve_inference_devices("0,1,2,3") == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    ]
    assert _resolve_inference_devices("cuda:0,cuda:1") == ["cuda:0", "cuda:1"]
    assert _resolve_inference_devices("all") == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    mrc_paths = [tmp_path / f"mic_{index}.mrc" for index in range(5)]
    shards = _shard_mrc_paths(mrc_paths, 2)
    assert shards == [mrc_paths[0::2], mrc_paths[1::2]]

    worker_summaries = [
        {
            "device": "cuda:0",
            "inputs": [
                {"input_mrc": str(path), "particle_overlay": {"output_png": str(tmp_path / f"{path.stem}.png")}}
                for path in shards[0]
            ],
            "particle_overlay_rendering": {
                "enabled": True,
                "output_dir": str(tmp_path / "OTF_images"),
                "frames": [
                    {"micrograph": str(path), "output_png": str(tmp_path / f"{path.stem}.png")}
                    for path in shards[0]
                ],
            },
        },
        {
            "device": "cuda:1",
            "inputs": [
                {"input_mrc": str(path), "particle_overlay": {"output_png": str(tmp_path / f"{path.stem}.png")}}
                for path in shards[1]
            ],
            "particle_overlay_rendering": {
                "enabled": True,
                "output_dir": str(tmp_path / "OTF_images"),
                "frames": [
                    {"micrograph": str(path), "output_png": str(tmp_path / f"{path.stem}.png")}
                    for path in shards[1]
                ],
            },
        },
    ]
    merged = _merge_worker_summaries(worker_summaries, mrc_paths)
    assert [Path(row["input_mrc"]) for row in merged["inputs"]] == mrc_paths
    assert [
        Path(frame["micrograph"])
        for frame in merged["particle_overlay_rendering"]["frames"]
    ] == mrc_paths


def test_multi_gpu_worker_preserves_live_particle_overlays(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run_infer(args, *, mrc_paths_override, finalize, live_summary_path):
        captured["particle_file"] = args.particle_file
        captured["render_particle_overlays"] = args.render_particle_overlays
        captured["filtered_particle_file"] = args.filtered_particle_file
        captured["allow_unmatched_particles"] = args.allow_unmatched_particles
        captured["particle_overlay_contact_sheet"] = args.particle_overlay_contact_sheet
        captured["finalize"] = finalize
        captured["live_summary_path"] = live_summary_path
        return {"inputs": []}

    monkeypatch.setattr("cryofilter.cli._run_infer", fake_run_infer)

    _multi_gpu_worker(
        {
            "particle_file": "particles.star",
            "filtered_particle_file": "filtered.star",
            "render_particle_overlays": True,
            "particle_overlay_contact_sheet": True,
            "allow_unmatched_particles": False,
        },
        [tmp_path / "mic_001.mrc"],
        tmp_path / "worker_summary.json",
    )

    assert captured == {
        "particle_file": "particles.star",
        "render_particle_overlays": True,
        "filtered_particle_file": None,
        "allow_unmatched_particles": True,
        "particle_overlay_contact_sheet": False,
        "finalize": False,
        "live_summary_path": tmp_path / "worker_summary.json",
    }


def test_single_gpu_wrapper_preserves_live_summary_file(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "micrographs"
    output_dir = tmp_path / "output"
    live_summary = output_dir / ".cryofilter_worker_00_summary.json"
    input_dir.mkdir()
    (input_dir / "mic_001.mrc").write_bytes(b"")
    captured = {}

    def fake_run_infer(args, *, live_summary_path):
        captured["device"] = args.device
        captured["live_summary_path"] = live_summary_path
        return 0

    monkeypatch.setattr("cryofilter.cli._run_infer", fake_run_infer)
    args = _build_parser().parse_args(
        [
            "infer",
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--live-summary-file",
            str(live_summary),
        ]
    )

    assert _run_multi_gpu_infer(args, ["cuda:0"]) == 0
    assert captured == {
        "device": "cuda:0",
        "live_summary_path": live_summary.resolve(),
    }


def test_multi_gpu_infer_treats_num_cpus_as_total_worker_budget(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "micrographs"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for index in range(4):
        (input_dir / f"mic_{index}.mrc").write_bytes(b"")

    worker_payloads = []
    finalized = {}

    class FakeProcess:
        def __init__(self, *, target, args, name):
            self.target = target
            self.args = args
            self.name = name
            self.exitcode = None

        def start(self):
            payload, shard, worker_summary_path = self.args
            worker_payloads.append(payload)
            worker_summary_path.write_text(
                json.dumps(
                    {
                        "inputs": [
                            {
                                "input_mrc": str(path),
                                "output_image_shape": [10, 10],
                                "mask_postprocessing": {"final_mask_pixels": 1},
                            }
                            for path in shard
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.exitcode = 0

        def join(self):
            return None

    class FakeContext:
        def Process(self, *, target, args, name):
            return FakeProcess(target=target, args=args, name=name)

    def fake_finalize(**kwargs):
        finalized.update(kwargs)
        return 0

    monkeypatch.setattr("cryofilter.cli.mp.get_context", lambda _name: FakeContext())
    monkeypatch.setattr("cryofilter.cli._finalize_inference_run", fake_finalize)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    args = _build_parser().parse_args(
        [
            "infer",
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--num-cpus",
            "32",
            "--num-gpus",
            "4",
        ]
    )
    args.num_cpus = 32
    args.num_gpus = 4

    assert _run_multi_gpu_infer(args, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]) == 0

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert [payload["num_cpus"] for payload in worker_payloads] == [8, 8, 8, 8]
    assert finalized["summary"]["multi_gpu"]["assignments"][0]["cpu_threads"] == 8


def test_default_output_dir_is_derived_from_input_path(tmp_path) -> None:
    input_dir = tmp_path / "micrographs"
    input_dir.mkdir()
    input_file = tmp_path / "single_image.mrc"
    input_file.write_bytes(b"")

    assert _default_output_dir_for_input(input_dir) == (tmp_path / "micrographs_cryofilter_output").resolve()
    assert _default_output_dir_for_input(input_file) == (tmp_path / "single_image_cryofilter_output").resolve()


def test_inference_writes_ready_to_use_typing_manifest(tmp_path) -> None:
    input_dir = tmp_path / "micrographs"
    nested_dir = input_dir / "grid_1"
    output_dir = tmp_path / "output"
    nested_dir.mkdir(parents=True)
    output_dir.mkdir()
    first_mrc = input_dir / "first.mrc"
    second_mrc = nested_dir / "second.mrc"
    first_mrc.write_bytes(b"")
    with mrcfile.new(second_mrc) as mrc:
        mrc.set_data(np.zeros((2, 2), dtype=np.float32))
        mrc.voxel_size = 1.25
    first_mask = output_dir / "first_mask.npy"
    second_mask = output_dir / "second_mask.npy"
    first_mask.write_bytes(b"")
    second_mask.write_bytes(b"")

    manifest = _write_typing_manifest(
        output_dir,
        input_dir,
        [
            {
                "input_mrc": str(first_mrc),
                "output_mask_npy": str(first_mask),
                "pixel_size_angstrom": 1.06,
                "generated_outputs": True,
            },
            {
                "input_mrc": str(second_mrc),
                "output_mask_npy": str(second_mask),
                "generated_outputs": True,
                "status": "skipped_existing",
            },
            {
                "input_mrc": str(input_dir / "not_generated.mrc"),
                "output_mask_npy": None,
                "generated_outputs": False,
            },
        ],
    )

    assert manifest == output_dir / "contamination_typing_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "dataset_id": "micrographs",
            "stem": "first",
            "micrograph_path": str(first_mrc.resolve()),
            "binary_mask_path": str(first_mask.resolve()),
            "pixel_size_angstrom": "1.06",
        },
        {
            "dataset_id": "micrographs__grid_1",
            "stem": "second",
            "micrograph_path": str(second_mrc.resolve()),
            "binary_mask_path": str(second_mask.resolve()),
            "pixel_size_angstrom": "1.25",
        },
    ]


def test_public_cli_contains_only_documented_workflows() -> None:
    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    assert set(subparsers.choices) == {
        "app",
        "cryosparc",
        "filter",
        "filter-particles",
        "gui",
        "infer",
        "install-cryosparc-tools",
        "studio",
        "train",
        "type",
    }
