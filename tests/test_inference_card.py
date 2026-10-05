from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import mrcfile
import numpy as np

from cryofilter.cryosparc import inference_card
from cryofilter.cryosparc.otf_state import (
    Index,
    MASK_CARD_FORMAT,
    monitor_summary,
    read_card,
)
from cryofilter.cryosparc.protocol.manifests import write_json_model
from cryofilter.cryosparc.protocol.models import TransferManifest


class _Dataset:
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return self.array[key]

    def fields(self):
        return self.array.dtype.names

    def prefixes(self):
        return sorted({name.split("/", 1)[0] for name in self.fields() if "/" in name})

    def take(self, indices):
        return _Dataset(self.array[indices])

    def filter_prefixes(self, prefixes, copy=True):
        names = [
            name
            for name in self.fields()
            if name == "uid" or name.split("/", 1)[0] in prefixes
        ]
        return _Dataset(self.array[names].copy())


def _micrograph(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path) as handle:
        handle.set_data(np.zeros((16, 20), dtype=np.float32))
        handle.voxel_size = 1.5
    return path


def _transfer_manifest(local_run_dir: Path, staged: Path) -> TransferManifest:
    transfer = TransferManifest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000401",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "external_job_uid": "J9",
            "micrographs_ref": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "micrographs": [
                {
                    "uid": 7,
                    "transfer_filename": "micrographs/7_example.mrc",
                    "source_path": "/project/J3/example.mrc",
                    "source_relative_path": "J3/example.mrc",
                    "shape_yx": [16, 20],
                    "pixel_size_angstrom": 1.5,
                    "blob_prefix": "micrograph_blob",
                    "file_size_bytes": staged.stat().st_size,
                }
            ],
            "particles": [],
        }
    )
    write_json_model(transfer, local_run_dir / "transfer_manifest.json")
    return transfer


def test_manifest_entries_preserve_uid_and_geometry(tmp_path: Path) -> None:
    local_run_dir = tmp_path / "run"
    staged = _micrograph(local_run_dir / "transfer/micrographs/7_example.mrc")
    transfer = _transfer_manifest(local_run_dir, staged)

    entries = inference_card._manifest_entries(
        transfer,
        local_transfer_dir=local_run_dir / "transfer",
    )

    assert entries[0]["uid"] == "7"
    assert entries[0]["path"] == str(staged.resolve())
    assert entries[0]["shape"] == [16, 20]
    assert entries[0]["pixel_size_angstrom"] == 1.5


def test_inference_card_publishes_live_index_and_micrograph_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    local_run_dir = tmp_path / "run"
    staged = _micrograph(local_run_dir / "transfer/micrographs/7_example.mrc")
    _transfer_manifest(local_run_dir, staged)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[3:5, 6:8] = 1
    mask_path = output_dir / "7_example_mask.npy"
    probability_path = output_dir / "7_example_prob.npy"
    np.save(mask_path, mask)
    np.save(probability_path, mask.astype(np.float32))
    (output_dir / "inference_summary.json").write_text(
        """{
  "inputs": [{
    "input_mrc": "%s",
    "output_mask_npy": "%s",
    "output_prob_npy": "%s",
    "output_image_shape": [16, 20],
    "mask_postprocessing": {"final_mask_pixels": 4}
  }]
}\n"""
        % (staged.resolve(), mask_path.resolve(), probability_path.resolve()),
        encoding="utf-8",
    )

    raw = np.zeros(
        1,
        dtype=[
            ("uid", "<u8"),
            ("micrograph_blob/path", "U128"),
            ("ctf/df1_A", "<f4"),
        ],
    )
    raw["uid"] = 7
    raw["micrograph_blob/path"] = "J3/example.mrc"
    raw["ctf/df1_A"] = 12345
    source = _Dataset(raw)

    class _External:
        uid = "J9"
        project_uid = "P1"

        def __init__(self):
            self.dir = tmp_path / "project/J9"
            self.dir.mkdir(parents=True)
            self.started = False
            self.start_status = None
            self.stopped = False
            self.output_spec = None
            self.saved = None

        def add_output(self, **kwargs):
            self.output_spec = kwargs

        def start(self, status=None):
            self.started = True
            self.start_status = status

        def stop(self, **kwargs):
            assert not kwargs
            self.stopped = True

        def save_output(self, name, dataset):
            self.saved = (name, dataset)

        def log(self, message, id=None):
            return id or "event-1"

    external = _External()
    source_job = SimpleNamespace(load_output=lambda name: source)
    project = SimpleNamespace(
        find_external_job=lambda uid: external,
        find_job=lambda uid: source_job,
    )
    client = SimpleNamespace(find_project=lambda uid: project)
    monkeypatch.setattr(inference_card, "_connect", lambda config: client)
    preview_updates = []
    monkeypatch.setattr(
        inference_card,
        "_publish_previews",
        lambda rows, *args, **kwargs: preview_updates.append(list(rows))
        or {"progress_event_id": "preview-event"},
    )
    monkeypatch.setattr(inference_card.time, "sleep", lambda seconds: None)

    typing_calls = []

    def fake_typing(**kwargs):
        typing_calls.append(kwargs)
        typing_dir = Path(kwargs["output_dir"])
        typed_dir = typing_dir / "typed_masks"
        typed_dir.mkdir(parents=True, exist_ok=True)
        typed_path = typed_dir / "P1_W2__7_example_typed_mask.npy"
        np.save(typed_path, mask.astype(np.uint8))
        image_csv = typing_dir / "image_contamination_summary.csv"
        image_csv.write_text(
            "dataset_id,stem,total_pixels,contaminated_pixels,carbon_area_px,"
            "crystalline_area_px,aggregate_area_px,ethane_area_px\n"
            "P1_W2,7_example,320,4,1,1,1,1\n",
            encoding="utf-8",
        )
        (typing_dir / "summary.json").write_text(
            """{
  "image_contamination_summary_csv": "%s",
  "typed_mask_dir": "%s",
  "n_images": 1,
  "n_images_expected": 1,
  "typing_status": "complete"
}\n"""
            % (image_csv.resolve(), typed_dir.resolve()),
            encoding="utf-8",
        )
        kwargs["poll_callback"]()
        return ["fake-typing"]

    monkeypatch.setattr(inference_card, "run_local_typing", fake_typing)
    monkeypatch.setattr(
        inference_card,
        "_typing_device_and_batch",
        lambda infer_args, **kwargs: ("cuda:1", 32, None),
    )

    commands = []

    class _Process:
        pid = 999999

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                (local_run_dir / "inference_live_summary.json").write_text(
                    (output_dir / "inference_summary.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            return None if self.polls == 1 else 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        inference_card.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append((command, kwargs)) or _Process(),
    )
    args = SimpleNamespace(
        project="P1",
        workspace="W2",
        local_run_dir=str(local_run_dir),
        output_dir=str(output_dir),
        poll_seconds=0.2,
        run_typing=True,
        typing_workers=3,
        typing_sample_stride_px=32,
        infer_args=[
            "--",
            "--input",
            str(local_run_dir / "transfer/micrographs"),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            "weights.pt",
            "--num-gpus",
            "2",
        ],
    )

    assert inference_card.run_inference_card(args, None) == 0
    assert commands[0][0][:4] == [
        inference_card.sys.executable,
        "-m",
        "cryofilter.cli",
        "infer",
    ]
    assert commands[0][1]["start_new_session"] is True
    assert commands[0][0][commands[0][0].index("--gpus") + 1] == "0"
    assert commands[0][0][commands[0][0].index("--num-gpus") + 1] == "1"
    live_summary = commands[0][0][commands[0][0].index("--live-summary-file") + 1]
    assert live_summary == str(local_run_dir / "inference_live_summary.json")
    assert external.started and external.stopped
    assert external.start_status == "running"
    assert external.output_spec["type"] == "exposure"
    assert external.output_spec["name"] == "micrographs"
    assert external.saved[0] == "micrographs"
    assert external.saved[1]["uid"].tolist() == [7]
    assert set(external.saved[1].fields()) == {"uid", "micrograph_blob/path"}
    assert preview_updates and preview_updates[0][0][0]["uid"] == "7"
    assert all(update[0][2] is not None for update in preview_updates)
    assert preview_updates[-1][0][2]["path"].endswith("_typed_mask.npy")
    assert typing_calls[0]["workers"] == 3
    assert typing_calls[0]["typing_device"] == "cuda:1"
    assert typing_calls[0]["sample_stride_px"] == 32
    assert typing_calls[0]["typing_batch_size"] == 32

    card, index_path = read_card(external.dir, project="P1", job_uid="J9")
    assert card["format"] == MASK_CARD_FORMAT
    assert card["kind"] == "batch_inference"
    assert card["source_job"] == "J3"
    index = Index(index_path, readonly=True)
    try:
        rows = index.rows()
        assert rows[0][0]["uid"] == "7"
        assert rows[0][1]["output_mask_npy"] == str(mask_path.resolve())
        assert rows[0][2]["summary"]["carbon_area_px"] == 1
        assert index.get_meta("status")["phase"] == "completed"
        assert index.get_meta("status")["typed"] == 1
    finally:
        index.close()
    live = monitor_summary(index_path)
    assert live["source"] == "typing"
    assert live["n_images_completed"] == 1
    assert live["types"][0]["area_px"] == 1


def test_typing_uses_freest_requested_gpu_and_memory_safe_batch(monkeypatch) -> None:
    result = SimpleNamespace(
        stdout=json.dumps(
            [
                {"index": 0, "free": 1 * 1024**3, "total": 24 * 1024**3},
                {"index": 1, "free": 7 * 1024**3, "total": 24 * 1024**3},
                {"index": 2, "free": 20 * 1024**3, "total": 24 * 1024**3},
            ]
        )
    )
    monkeypatch.setattr(inference_card.subprocess, "run", lambda *args, **kwargs: result)

    device, batch_size, free_bytes = inference_card._typing_device_and_batch(
        ["--device", "cuda", "--num-gpus", "2"]
    )

    assert device == "cuda:1"
    assert batch_size == 16
    assert free_bytes == 7 * 1024**3


def test_live_typing_reserves_half_of_requested_gpus() -> None:
    infer_args, typing_devices = inference_card._reserve_live_typing_gpu(
        ["--device", "cuda", "--gpus", "1,2,3,4", "--num-gpus", "4"]
    )

    assert inference_card._infer_option(infer_args, "--gpus") == "1,2"
    assert inference_card._infer_option(infer_args, "--num-gpus") == "2"
    assert typing_devices == [3, 4]
