from __future__ import annotations

import json
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryofilter.app.server import (
    ARTIFACT_SUFFIXES,
    AppState,
    build_job_spec,
    runtime_status,
    validate_cryosparc_connection,
)
from cryofilter.cli import _build_parser


def test_app_parser_aliases_are_available() -> None:
    parser = _build_parser()

    args = parser.parse_args(["app", "--port", "9100"])
    assert args.command == "app"
    assert args.port == 9100

    args = parser.parse_args(["studio", "--work-dir", "runs"])
    assert args.command == "studio"
    assert args.work_dir == "runs"


def test_infer_job_spec_builds_cli_command(tmp_path: Path) -> None:
    spec = build_job_spec(
        "infer",
        {
            "input": "/data/micrographs",
            "checkpoint": "pretrained_models/cryoFILTER_FULL.pt",
            "output_dir": str(tmp_path / "out"),
            "device": "cuda",
            "num_cpus": "8",
            "num_gpus": "1",
            "threshold": "0.6",
            "particle_file": "/data/particles.cs",
            "binned_masks": True,
            "recursive": True,
            "render_overlays": True,
            "extra_args": "--batch-forward-size 32",
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert argv[:4] == [sys.executable, "-m", "cryofilter.cli", "infer"]
    assert argv[argv.index("--input") + 1] == "/data/micrographs"
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "out")
    assert argv[argv.index("--num-cpus") + 1] == "8"
    assert argv[argv.index("--num-gpus") + 1] == "1"
    assert "--recursive" in argv
    assert "--render-particle-overlays" in argv
    assert "--no-export-masks" not in argv
    assert "--no-resample" in argv
    assert "--batch-forward-size" in argv
    assert spec.metadata["export_masks"] is True
    assert spec.metadata["binned_masks"] is True
    assert spec.artifact_roots == [tmp_path / "out"]


def test_infer_job_spec_can_disable_visible_mask_exports(tmp_path: Path) -> None:
    spec = build_job_spec(
        "infer",
        {
            "input": "/data/micrographs",
            "checkpoint": "custom_weights.pt",
            "output_dir": str(tmp_path / "out"),
            "export_masks": False,
            "binned_masks": False,
            "render_overlays": False,
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert "--no-export-masks" in argv
    assert "--no-resample" not in argv
    assert "--no-render-particle-overlays" in argv
    assert spec.metadata["export_masks"] is False
    assert spec.metadata["binned_masks"] is False


def test_infer_job_spec_reports_missing_default_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pretrained_models/cryoFILTER_FULL.pt"):
        build_job_spec(
            "infer",
            {
                "input": "/data/micrographs",
                "output_dir": str(tmp_path / "out"),
            },
            work_dir=tmp_path,
        )


def test_infer_job_spec_leaves_binned_masks_off_by_default(tmp_path: Path) -> None:
    spec = build_job_spec(
        "infer",
        {
            "input": "/data/micrographs",
            "checkpoint": "custom_weights.pt",
            "output_dir": str(tmp_path / "out"),
        },
        work_dir=tmp_path,
    )

    assert "--no-resample" not in spec.steps[0].argv
    assert spec.metadata["binned_masks"] is False


def test_cryosparc_predict_job_spec_uses_typing_default(tmp_path: Path) -> None:
    spec = build_job_spec(
        "cryosparc_predict",
        {
            "config": "cryosparc.toml",
            "project": "P280",
            "workspace": "W5",
            "micrographs": "J76:micrographs",
            "particles": "J76:particles",
            "checkpoint": "pretrained_models/cryoFILTER_FULL.pt",
            "cryosparc_base_url": "http://cryosparc.example.edu:39000/browse/P280-W5-J*",
            "local_run_root": str(tmp_path / "runs"),
            "device": "cuda",
            "inference_profile": "balanced",
            "num_cpus": "8",
            "num_gpus": "1",
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert argv[:4] == [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
    assert "predict" in argv
    assert "--run-typing" not in argv
    assert "--no-run-typing" not in argv
    assert argv[argv.index("--config") + 1] == "cryosparc.toml"
    assert "--cryosparc-base-url" not in argv
    assert argv[argv.index("--cryosparc-host") + 1] == "cryosparc.example.edu"
    assert argv[argv.index("--cryosparc-base-port") + 1] == "39000"
    assert argv[argv.index("--micrographs") + 1] == "J76:micrographs"
    assert argv[argv.index("--particles") + 1] == "J76:particles"
    assert "--run-id" in argv
    run_id = argv[argv.index("--run-id") + 1]
    assert argv[argv.index("--num-cpus") + 1] == "8"
    assert argv[argv.index("--num-gpus") + 1] == "1"
    assert "--" in argv
    assert argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--inference-profile") + 1] == "balanced"
    assert "--no-export-masks" not in argv
    assert "--no-resample" not in argv[argv.index("--") + 1 :]
    assert spec.metadata["run_typing"] is True
    assert spec.metadata["num_cpus"] == "8"
    assert spec.metadata["num_gpus"] == "1"
    assert spec.metadata["export_masks"] is True
    assert spec.metadata["binned_masks"] is False
    assert spec.metadata["run_id"] == run_id
    assert spec.metadata["local_run_dir"] == tmp_path / "runs" / run_id
    assert spec.artifact_roots == [tmp_path / "runs" / run_id]


def test_cryosparc_predict_job_spec_forwards_output_options(tmp_path: Path) -> None:
    spec = build_job_spec(
        "cryosparc_predict",
        {
            "project": "P1",
            "workspace": "W2",
            "micrographs": "J3",
            "particles": "J4",
            "checkpoint": "custom_weights.pt",
            "export_masks": False,
            "binned_masks": False,
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert "--" in argv
    forwarded = argv[argv.index("--") + 1 :]
    assert "--no-export-masks" in forwarded
    assert "--no-resample" not in forwarded
    assert spec.metadata["export_masks"] is False
    assert spec.metadata["binned_masks"] is False


def test_filter_particles_job_spec_builds_existing_mask_command(tmp_path: Path) -> None:
    spec = build_job_spec(
        "filter_particles",
        {
            "input": "/data/micrographs",
            "mask_dir": str(tmp_path / "masks"),
            "output_dir": str(tmp_path / "filtered"),
            "particle_file": "/data/particles.cs",
            "particle_csg": "/data/particles.csg",
            "particle_exclusion_distance_angstrom": "120",
            "pixel_size_angstrom": "1.06",
            "recursive": True,
            "allow_unmatched_particles": True,
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert argv[:4] == [sys.executable, "-m", "cryofilter.cli", "filter-particles"]
    assert argv[argv.index("--input") + 1] == "/data/micrographs"
    assert argv[argv.index("--mask-dir") + 1] == str(tmp_path / "masks")
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "filtered")
    assert argv[argv.index("--particle-file") + 1] == "/data/particles.cs"
    assert argv[argv.index("--particle-csg") + 1] == "/data/particles.csg"
    assert argv[argv.index("--particle-exclusion-distance-angstrom") + 1] == "120"
    assert argv[argv.index("--pixel-size-angstrom") + 1] == "1.06"
    assert "--recursive" in argv
    assert "--allow-unmatched-particles" in argv
    assert spec.artifact_roots == [tmp_path / "filtered"]
    assert spec.metadata["mask_dir"] == tmp_path / "masks"
    assert {".cs", ".csg", ".star"}.issubset(ARTIFACT_SUFFIXES)


def test_cryosparc_predict_job_spec_accepts_common_output_ref_shorthand(tmp_path: Path) -> None:
    spec = build_job_spec(
        "cryosparc_predict",
        {
            "project": "P280",
            "workspace": "W5",
            "micrographs": "J54",
            "particles": "J56",
            "checkpoint": "custom_weights.pt",
            "cryosparc_base_url": "cryosparc.example.edu:39000",
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert "--cryosparc-base-url" not in argv
    assert argv[argv.index("--cryosparc-host") + 1] == "cryosparc.example.edu"
    assert argv[argv.index("--cryosparc-base-port") + 1] == "39000"
    assert argv[argv.index("--micrographs") + 1] == "J54:micrographs"
    assert argv[argv.index("--particles") + 1] == "J56:particles"
    assert spec.metadata["micrographs"] == "J54:micrographs"
    assert spec.metadata["particles"] == "J56:particles"


def test_cryosparc_password_is_ephemeral_job_env(tmp_path: Path) -> None:
    spec = build_job_spec(
        "cryosparc_predict",
        {
            "project": "P1",
            "workspace": "W2",
            "micrographs": "J3",
            "particles": "J4",
            "checkpoint": "custom_weights.pt",
            "cryosparc_password": "unit-test-secret",
        },
        work_dir=tmp_path,
    )

    assert spec.env == {"CRYOSPARC_PASSWORD": "unit-test-secret"}
    serialized = spec.as_dict()
    assert "env" not in serialized
    assert "unit-test-secret" not in json.dumps(serialized)


def test_cryosparc_connection_probe_uses_ephemeral_password() -> None:
    seen: dict[str, object] = {}

    def make_client(**kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    result = validate_cryosparc_connection(
        {
            "cryosparc_base_url": "http://cryosparc.example.edu:39000/browse/P1-W2-J*",
            "cryosparc_email": "user@example.edu",
            "cryosparc_password": "unit-test-secret",
        },
        make_client=make_client,
        test_connection=lambda _client: True,
        server_version=lambda _client: "v4.test",
    )

    assert seen == {
        "host": "cryosparc.example.edu",
        "base_port": 39000,
        "email": "user@example.edu",
        "password": "unit-test-secret",
    }
    assert result["ok"] is True
    assert result["display"] == "cryosparc.example.edu:39000"
    assert result["server_version"] == "v4.test"
    assert "password" not in result
    assert "unit-test-secret" not in json.dumps(result)


def test_cryosparc_connection_probe_redacts_password_from_errors() -> None:
    def make_client(**_kwargs: object) -> object:
        raise RuntimeError("login failed for unit-test-secret")

    with pytest.raises(RuntimeError) as exc_info:
        validate_cryosparc_connection(
            {
                "cryosparc_base_url": "cryosparc.example.edu:39000",
                "cryosparc_password": "unit-test-secret",
            },
            make_client=make_client,
            test_connection=lambda _client: True,
            server_version=lambda _client: None,
        )

    message = str(exc_info.value)
    assert "unit-test-secret" not in message
    assert "[hidden]" in message


def test_cryosparc_predict_job_spec_can_opt_out_of_typing(tmp_path: Path) -> None:
    spec = build_job_spec(
        "cryosparc_predict",
        {
            "project": "P1",
            "workspace": "W2",
            "micrographs": "J3:micrographs",
            "particles": "J4:particles",
            "checkpoint": "custom_weights.pt",
            "run_typing": False,
        },
        work_dir=tmp_path,
    )

    assert "--no-run-typing" in spec.steps[0].argv
    assert spec.metadata["run_typing"] is False


def test_cryosparc_predict_job_spec_passes_bridge_controls(tmp_path: Path) -> None:
    spec = build_job_spec(
        "cryosparc_predict",
        {
            "project": "P1",
            "workspace": "W2",
            "micrographs": "J3:micrographs",
            "particles": "J4:particles",
            "checkpoint": "custom_weights.pt",
            "bridge_host": "local",
            "bridge_command": "/opt/cryofilter/bin/cryofilter-bridge",
            "remote_work_root": "/scratch/cryofilter_bridge_work",
            "remote_source_root": "/opt/cryofilter/bridge_src",
            "ssh_options": "-J user@login.example.edu -o ConnectTimeout=10",
        },
        work_dir=tmp_path,
    )

    argv = spec.steps[0].argv
    assert argv[argv.index("--host") + 1] == "local"
    assert argv[argv.index("--bridge-command") + 1] == "/opt/cryofilter/bin/cryofilter-bridge"
    assert argv[argv.index("--remote-work-root") + 1] == "/scratch/cryofilter_bridge_work"
    assert argv[argv.index("--remote-source-root") + 1] == "/opt/cryofilter/bridge_src"
    ssh_option_values = [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--ssh-option"
    ]
    assert ssh_option_values == ["-J", "user@login.example.edu", "-o", "ConnectTimeout=10"]


def test_annotation_job_spec_builds_manifest_from_micrograph_source(tmp_path: Path) -> None:
    spec = build_job_spec(
        "annotation",
        {
            "source_mode": "micrographs",
            "micrograph_source": "/data/motioncorrected",
            "output_root": str(tmp_path / "annotations"),
            "output_name": "session_a",
            "dataset_id": "7",
        },
        work_dir=tmp_path,
    )

    assert len(spec.steps) == 2
    build_argv = spec.steps[0].argv
    editor_argv = spec.steps[1].argv
    assert build_argv[:3] == [sys.executable, "-m", "cryofilter.app.annotation_manifest"]
    assert build_argv[build_argv.index("--micrographs") + 1] == "/data/motioncorrected"
    assert build_argv[build_argv.index("--dataset-id") + 1] == "7"
    assert "--reuse-existing" in build_argv
    assert editor_argv[editor_argv.index("--manifest") + 1].endswith(
        "annotations/session_a/source_manifest.csv"
    )
    assert editor_argv[editor_argv.index("--output_name") + 1] == "session_a"
    assert editor_argv[editor_argv.index("--prefetch_workers") + 1] == "2"
    assert spec.artifact_roots == [tmp_path / "annotations"]


def test_annotation_job_spec_resumes_existing_manifest_in_place(tmp_path: Path) -> None:
    manifest = tmp_path / "annotation_exports" / "session_a" / "manifest_edited.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("dataset_id,stem,micrograph_path\n1,a,/data/a.mrc\n", encoding="utf-8")

    spec = build_job_spec(
        "annotation",
        {
            "source_mode": "manifest",
            "manifest": str(manifest),
            "resume_in_place": True,
        },
        work_dir=tmp_path,
    )

    assert len(spec.steps) == 1
    argv = spec.steps[0].argv
    assert argv[argv.index("--manifest") + 1] == str(manifest)
    assert argv[argv.index("--output_dir") + 1] == str(manifest.parent)
    assert spec.artifact_roots == [manifest.parent]


def test_annotation_job_spec_can_stage_cryosparc_micrographs(tmp_path: Path) -> None:
    spec = build_job_spec(
        "annotation",
        {
            "source_mode": "cryosparc",
            "cryosparc_base_url": "http://cryosparc.example.edu:39000",
            "cryosparc_email": "user@example.edu",
            "cryosparc_password": "unit-test-secret",
            "cryosparc_project": "P306",
            "cryosparc_workspace": "W1",
            "cryosparc_micrographs": "J42",
            "output_root": str(tmp_path / "annotations"),
            "output_name": "from_cs",
        },
        work_dir=tmp_path,
    )

    assert len(spec.steps) == 3
    stage_argv = spec.steps[0].argv
    build_argv = spec.steps[1].argv
    editor_argv = spec.steps[2].argv
    assert stage_argv[:4] == [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
    assert stage_argv[stage_argv.index("--host") + 1] == "local"
    assert stage_argv[stage_argv.index("--cryosparc-host") + 1] == "cryosparc.example.edu"
    assert stage_argv[stage_argv.index("--cryosparc-base-port") + 1] == "39000"
    assert stage_argv[stage_argv.index("--micrographs") + 1] == "J42:micrographs"
    assert "--all-micrographs" in stage_argv
    assert "--limit-micrographs" not in stage_argv
    assert "--no-pull" in stage_argv
    assert "--no-stage-micrographs" in stage_argv
    assert build_argv[build_argv.index("--transfer-manifest") + 1].endswith(
        "transfer_manifest.json"
    )
    assert "--prefer-source-paths" in build_argv
    assert "--mic_dir" not in editor_argv
    assert spec.env == {"CRYOSPARC_PASSWORD": "unit-test-secret"}
    assert "unit-test-secret" not in json.dumps(spec.as_dict())


def test_annotation_job_spec_can_copy_cryosparc_micrographs_when_requested(tmp_path: Path) -> None:
    spec = build_job_spec(
        "annotation",
        {
            "source_mode": "cryosparc",
            "cryosparc_base_url": "http://cryosparc.example.edu:39000",
            "cryosparc_project": "P306",
            "cryosparc_workspace": "W1",
            "cryosparc_micrographs": "J42",
            "copy_cryosparc_micrographs": True,
            "limit_micrographs": "5",
        },
        work_dir=tmp_path,
    )

    stage_argv = spec.steps[0].argv
    build_argv = spec.steps[1].argv
    editor_argv = spec.steps[2].argv
    assert "--no-pull" not in stage_argv
    assert "--no-stage-micrographs" not in stage_argv
    assert stage_argv[stage_argv.index("--limit-micrographs") + 1] == "5"
    assert "--prefer-source-paths" not in build_argv
    assert editor_argv[editor_argv.index("--mic_dir") + 1].endswith("transfer/micrographs")


def test_train_job_spec_wraps_recommended_finetune_recipe(tmp_path: Path) -> None:
    spec = build_job_spec(
        "train",
        {
            "manifest": "annotation_exports/session/manifest_edited.csv",
            "mic_dir": "/data/micrographs",
            "pretrained_model": "pretrained_models/cryoFILTER_FULL.pt",
            "output_dir": str(tmp_path / "finetune"),
            "train_dataset_ids": "1,2",
            "val_dataset_ids": "101",
            "num_epochs": "20",
        },
        work_dir=tmp_path,
    )

    prepare_argv = spec.steps[0].argv
    argv = spec.steps[1].argv
    assert prepare_argv[:3] == [sys.executable, "-m", "cryofilter.app.training_manifest"]
    assert prepare_argv[prepare_argv.index("--manifest") + 1] == str(tmp_path / "annotation_exports/session/manifest_edited.csv")
    assert prepare_argv[prepare_argv.index("--output-manifest") + 1] == str(tmp_path / "finetune" / "cryofilter_training_manifest.csv")
    assert "--preserve-split" in prepare_argv
    assert argv[:2] == [sys.executable, str(Path("scripts/train_patch_based_fast.py").resolve())]
    assert argv[argv.index("--manifest") + 1] == str(tmp_path / "finetune" / "cryofilter_training_manifest.csv")
    assert "--full_micrograph_training" in argv
    assert "--psd_frequency_band_channels" in argv
    assert "--no_region_gt_fill_holes" in argv
    assert argv[argv.index("--num_epochs") + 1] == "20"
    assert argv[argv.index("--batch_size") + 1] == "16"
    assert argv[argv.index("--num_gpus") + 1] == "1"
    assert argv[argv.index("--full_mic_patches_per_mic") + 1] == "64"
    assert argv[argv.index("--stitched_val_batch_forward_size") + 1] == "16"
    assert spec.artifact_roots == [tmp_path / "finetune"]


def test_train_job_spec_infers_micrograph_root_from_manifest(tmp_path: Path) -> None:
    mic_dir = tmp_path / "micrographs"
    mic_dir.mkdir()
    manifest = tmp_path / "annotation_exports" / "session" / "manifest_edited.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "dataset_id,stem,micrograph_path,gt_mask_path,split\n"
        f"1,FoilHole_0001,{mic_dir / 'FoilHole_0001.mrc'},masks/FoilHole_0001.npy,TRAINING\n"
        f"1,FoilHole_0002,{mic_dir / 'FoilHole_0002.mrc'},masks/FoilHole_0002.npy,VALIDATION\n",
        encoding="utf-8",
    )

    spec = build_job_spec(
        "train",
        {
            "manifest": str(manifest.relative_to(tmp_path)),
            "pretrained_model": "pretrained_models/cryoFILTER_FULL.pt",
            "output_dir": str(tmp_path / "finetune"),
        },
        work_dir=tmp_path,
    )

    prepare_argv = spec.steps[0].argv
    argv = spec.steps[1].argv
    assert prepare_argv[prepare_argv.index("--manifest") + 1] == str(manifest)
    assert prepare_argv[prepare_argv.index("--output-manifest") + 1] == str(tmp_path / "finetune" / "cryofilter_training_manifest.csv")
    assert prepare_argv[prepare_argv.index("--train-fraction") + 1] == "0.7"
    assert prepare_argv[prepare_argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--manifest") + 1] == str(tmp_path / "finetune" / "cryofilter_training_manifest.csv")
    assert argv[argv.index("--mic_dir") + 1] == str(mic_dir)
    assert "--train_dataset_ids" not in argv
    assert "--val_dataset_ids" not in argv
    assert spec.metadata["mic_dir"] == str(mic_dir)
    assert spec.metadata["split_counts"] == {"TRAINING": 1, "VALIDATION": 1}


def test_train_job_spec_can_reuse_cryosparc_micrograph_output(tmp_path: Path) -> None:
    manifest = tmp_path / "annotation_exports" / "session" / "manifest_edited.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "dataset_id,stem,micrograph_path,gt_mask_path,source_uid\n"
        "1,FoilHole_0001,/cryosparc/P1/FoilHole_0001.mrc,masks/FoilHole_0001.npy,7\n",
        encoding="utf-8",
    )

    spec = build_job_spec(
        "train",
        {
            "manifest": str(manifest),
            "mic_source_mode": "cryosparc",
            "cryosparc_base_url": "http://cryosparc.example.edu:39000",
            "cryosparc_email": "user@example.edu",
            "cryosparc_password": "unit-test-secret",
            "cryosparc_project": "P306",
            "cryosparc_workspace": "W1",
            "cryosparc_micrographs": "J5",
            "pretrained_model": "pretrained_models/cryoFILTER_FULL.pt",
            "output_dir": str(tmp_path / "finetune"),
        },
        work_dir=tmp_path,
    )

    stage_argv = spec.steps[0].argv
    prepare_argv = spec.steps[1].argv
    train_argv = spec.steps[2].argv
    assert stage_argv[:4] == [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
    assert stage_argv[stage_argv.index("--cryosparc-host") + 1] == "cryosparc.example.edu"
    assert stage_argv[stage_argv.index("--cryosparc-base-port") + 1] == "39000"
    assert stage_argv[stage_argv.index("--project") + 1] == "P306"
    assert stage_argv[stage_argv.index("--workspace") + 1] == "W1"
    assert stage_argv[stage_argv.index("--micrographs") + 1] == "J5:micrographs"
    assert "--no-pull" in stage_argv
    assert "--no-stage-micrographs" in stage_argv
    assert prepare_argv[:3] == [sys.executable, "-m", "cryofilter.app.training_manifest"]
    assert prepare_argv[prepare_argv.index("--transfer-manifest") + 1].endswith("transfer_manifest.json")
    assert "--prefer-source-paths" in prepare_argv
    assert train_argv[train_argv.index("--manifest") + 1] == str(tmp_path / "finetune" / "cryofilter_training_manifest.csv")
    assert spec.env == {"CRYOSPARC_PASSWORD": "unit-test-secret"}
    assert "unit-test-secret" not in json.dumps(spec.as_dict())


def test_browser_cryosparc_annotation_stage_does_not_precreate_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    micrograph = tmp_path / "source.mrc"
    with mrcfile.new(micrograph, overwrite=True) as handle:
        handle.set_data(np.zeros((12, 16), dtype=np.float32))

    def fake_stage(
        self: AppState,
        argv: list[str],
        *,
        log_path: Path,
        secret_env: dict[str, str] | None = None,
    ) -> None:
        run_id = argv[argv.index("--run-id") + 1]
        stage_root = Path(argv[argv.index("--local-run-root") + 1])
        stage_run_dir = stage_root / run_id
        assert log_path.parent == stage_root
        assert log_path.parent != stage_run_dir
        assert not stage_run_dir.exists()
        stage_run_dir.mkdir(parents=True)
        (stage_run_dir / "transfer_manifest.json").write_text(
            json.dumps(
                {
                    "project_uid": "P1",
                    "workspace_uid": "W2",
                    "micrographs_ref": {"project_uid": "P1", "job_uid": "J3", "output_name": "micrographs"},
                    "micrographs": [
                        {
                            "uid": "mic0",
                            "source_path": str(micrograph),
                            "transfer_filename": micrograph.name,
                            "pixel_size_angstrom": "1.0",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(AppState, "_run_inline_stage", fake_stage)

    state = AppState(tmp_path / "work")
    session = state.create_annotation_session(
        {
            "source_mode": "cryosparc",
            "cryosparc_base_url": "http://cryosparc.example.edu:39000",
            "cryosparc_project": "P1",
            "cryosparc_workspace": "W2",
            "cryosparc_micrographs": "J3",
            "output_name": "browser_cryosparc",
        }
    )

    assert session["row_count"] == 1
    assert session["stage_log_file"].endswith(".browser_annotation_stage.log")
    assert Path(session["stage_log_file"]).parent != Path(session["stage_run_dir"])
    assert session["cryosparc_project"] == "P1"
    assert session["cryosparc_workspace"] == "W2"
    assert session["micrographs"] == "J3:micrographs"


def test_app_state_marks_interrupted_jobs_stale(tmp_path: Path) -> None:
    first = AppState(tmp_path)
    job_id = "abc123"
    job_dir = first.jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"id": job_id, "status": "running"}),
        encoding="utf-8",
    )

    second = AppState(tmp_path)

    meta = second.get_job(job_id)
    assert meta["status"] == "unknown"
    assert meta["stale_reason"] == "server_restarted"


def test_app_state_live_summary_aggregates_typing_range(tmp_path: Path) -> None:
    state = AppState(tmp_path)
    job_id = "summary123"
    run_dir = tmp_path / "runs" / "one"
    typing_dir = run_dir / "typing"
    typing_dir.mkdir(parents=True)
    image_csv = typing_dir / "image_contamination_summary.csv"
    image_csv.write_text(
        "\n".join(
            [
                "image_id,dataset_id,stem,total_pixels,contaminated_pixels,carbon_area_px,crystalline_area_px,aggregate_area_px,ethane_area_px",
                "d__a,d,a,100,20,10,10,0,0",
                "d__b,d,b,100,30,20,0,10,0",
                "d__c,d,c,200,50,0,0,0,50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (typing_dir / "summary.json").write_text(
        json.dumps(
            {
                "type_order": ["Carbon", "Crystalline", "Aggregate", "Ethane"],
                "image_contamination_summary_csv": str(image_csv),
            }
        ),
        encoding="utf-8",
    )
    job_dir = state.jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"id": job_id, "kind": "cryosparc_predict", "artifact_roots": [str(run_dir)]}),
        encoding="utf-8",
    )

    summary = state.live_summary(job_id, mode="last", count=2)

    assert summary["available"] is True
    assert summary["source"] == "typing"
    assert summary["n_images"] == 2
    assert summary["n_images_total"] == 3
    assert summary["total_pixels"] == 300
    assert summary["contaminated_pixels"] == 80
    assert summary["clean_pixels"] == 220
    assert {item["label"]: item["area_px"] for item in summary["types"]} == {
        "Carbon": 20,
        "Crystalline": 0,
        "Aggregate": 10,
        "Ethane": 50,
    }


def test_app_state_live_summary_aggregates_multi_gpu_worker_summaries(tmp_path: Path) -> None:
    state = AppState(tmp_path)
    job_id = "workers123"
    run_dir = tmp_path / "runs" / "one"
    inference_dir = run_dir / "inference"
    inference_dir.mkdir(parents=True)
    for worker_index, rows in enumerate(
        [
            [
                ("b.mrc", [10, 10], 20),
                ("d.mrc", [20, 10], 40),
            ],
            [
                ("a.mrc", [10, 10], 10),
                ("c.mrc", [10, 20], 30),
            ],
        ]
    ):
        (inference_dir / f".cryofilter_worker_{worker_index:02d}_summary.json").write_text(
            json.dumps(
                {
                    "inputs": [
                        {
                            "input_mrc": name,
                            "output_image_shape": shape,
                            "mask_postprocessing": {"final_mask_pixels": pixels},
                        }
                        for name, shape, pixels in rows
                    ]
                }
            ),
            encoding="utf-8",
        )
    job_dir = state.jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"id": job_id, "kind": "cryosparc_predict", "artifact_roots": [str(run_dir)]}),
        encoding="utf-8",
    )

    summary = state.live_summary(job_id, mode="last", count=3)

    assert summary["available"] is True
    assert summary["source"] == "multi_gpu_workers"
    assert summary["worker_count"] == 2
    assert summary["n_images"] == 3
    assert summary["n_images_total"] == 4
    assert summary["total_pixels"] == 500
    assert summary["contaminated_pixels"] == 90
    assert summary["clean_pixels"] == 410


def test_runtime_status_reports_display_metadata(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    status = runtime_status()

    assert status["python_version"]
    assert status["node"]
    assert status["gpu"] == "CUDA_VISIBLE_DEVICES=0"
