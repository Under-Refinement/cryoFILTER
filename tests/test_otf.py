from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import mrcfile
import numpy as np
import pytest

from cryofilter.app.server import build_job_spec, _build_live_summary
from cryofilter.cli import _build_parser
from cryofilter.cryosparc.otf import supervise
from cryofilter.cryosparc.otf_filter import filter_by_index, keep_from_mask
from cryofilter.cryosparc.otf_source import FileReadiness, PatchSource, mrc_geometry
from cryofilter.cryosparc.otf_state import (
    Index, FORMAT, CARD_FILE, file_stamp, generation_key, load_mask, read_card, save_mask, write_json,
)
from cryofilter.cryosparc.otf_workers import allocate_devices


def micrograph(path, shape=(64, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.zeros(shape, dtype=np.float32))
        mrc.voxel_size = 2.0
    return path


def indexed_image(tmp_path, uid="18446744073709550000", mask=None):
    path = micrograph(tmp_path / f"{uid}_mic.mrc")
    stamp = file_stamp(path)
    entry = {"uid": uid, "path": str(path), "stamp": stamp, "key": generation_key(uid, stamp),
             "shape": [64, 80], "pixel_size_angstrom": 2.0}
    if mask is None:
        mask = np.zeros((64, 80), dtype=bool)
        mask[30:35, 38:43] = True
    mask_path = tmp_path / f"{uid}_mask.npz"
    save_mask(mask_path, mask)
    prob_path = tmp_path / f"{uid}_prob.npy"
    np.save(prob_path, mask.astype(np.float16))
    inference = {"input_mrc": str(path), "output_mask_npy": str(mask_path), "output_prob_npy": str(prob_path),
                 "output_stamps": [file_stamp(mask_path), file_stamp(prob_path)],
                 "input_image_shape": [64, 80], "output_image_shape": list(mask.shape),
                 "mask_postprocessing": {"final_mask_pixels": int(mask.sum())}}
    return entry, inference, None


def particles(uids, points):
    arr = np.zeros(len(uids), dtype=[("uid", "<u8"), ("location/micrograph_uid", "<u8"),
        ("location/center_x_frac", "<f4"), ("location/center_y_frac", "<f4"), ("ctf/df1_A", "<f4")])
    arr["uid"] = np.arange(len(arr)) + 100
    arr["location/micrograph_uid"] = uids
    arr["location/center_x_frac"], arr["location/center_y_frac"] = np.asarray(points).T
    arr["ctf/df1_A"] = 12345
    return arr


def test_gpu_allocation_never_types_with_one_gpu():
    assert allocate_devices(["3"], False) == (["3"], [])
    with pytest.raises(ValueError, match="segmentation only"):
        allocate_devices(["3"], True)
    assert allocate_devices(["2", "4", "5"], True) == (["2"], ["4", "5"])
    assert allocate_devices(["2", "4", "5"], False) == (["2", "4", "5"], [])
    with pytest.raises(ValueError, match="distinct"):
        allocate_devices(["2", "2"], False)


def test_readiness_waits_for_complete_stable_payload(tmp_path):
    path = micrograph(tmp_path / "mic.mrc")
    complete = path.read_bytes()
    path.write_bytes(complete[:1100])
    checker = FileReadiness(stable_seconds=4)
    assert checker.ready(path, now=0) is None
    assert checker.ready(path, now=5) is None
    path.write_bytes(complete)
    assert checker.ready(path, now=6) is None
    assert checker.ready(path, now=9) is None
    stamp, shape, px = checker.ready(path, now=10)
    assert shape == [64, 80] and px == 2
    assert stamp == file_stamp(path)


def test_mask_packing_is_bit_exact_including_odd_dimensions(tmp_path):
    mask = np.random.default_rng(8).random((71, 83)) > 0.7
    path = tmp_path / "mask.npz"
    save_mask(path, mask)
    assert np.array_equal(load_mask(path), mask)


class FakeJob:
    type = "patch_motion_correction_multi"
    status = "running"
    uid = "J4"

    def __init__(self, directory, data):
        self.directory = directory
        self.data = data

    def dir(self):
        return self.directory

    def refresh(self):
        return self

    def load_input(self, name, slots):
        assert name == "movies" and slots == ["movie_blob"]
        return self.data

    def load_output(self, name, slots):
        assert name == "micrographs" and slots == ["micrograph_blob"]
        return self.data


def test_patch_source_uid_and_doseweighted_selection(tmp_path):
    uid = 2951270711614884258  # Real Silva output uses this padded UID naming scheme.
    relative = f"J4/motioncorrected/{uid:021d}_movie_patch_aligned_doseweighted.mrc"
    micrograph(tmp_path / relative)
    micrograph(tmp_path / f"J4/motioncorrected/{uid:021d}_movie_patch_aligned.mrc")
    micrograph(tmp_path / "J4/motioncorrected/000000000000000000002_other_patch_aligned_doseweighted.mrc")
    data = np.zeros(1, dtype=[("uid", "<u8"), ("micrograph_blob/path", "U256")])
    data["uid"] = uid
    data["micrograph_blob/path"] = relative
    job = FakeJob(tmp_path / "J4", data)
    source = PatchSource(job, SimpleNamespace(dir=lambda: tmp_path))
    source.readiness.stable_seconds = 0
    source.refresh()
    assert source.scan({})[0] == []
    entries, waiting = source.scan({})
    assert len(entries) == 1 and not waiting
    assert entries[0]["uid"] == str(uid)
    known = {str(uid): entries[0]["key"]}
    assert source.scan(known) == ([], [])
    job.status = "completed"
    source.refresh()
    assert source.expected == 1
    assert source.scan(known) == ([], [])


def test_index_snapshot_only_exposes_committed_masks_and_rejects_stale_worker(tmp_path):
    entry, record, _ = indexed_image(tmp_path)
    index = Index(tmp_path / "index.sqlite")
    reader = Index(index.path, readonly=True)
    try:
        index.discover(entry)
        assert reader.rows()[0][1] is None
        assert index.complete("segmentation", entry, record)
        assert reader.rows()[0][1] == record
        changed = dict(entry, key="new-generation")
        index.discover(changed)
        assert not index.complete("segmentation", entry, record)
        assert reader.rows()[0][1] is None
    finally:
        reader.close()
        index.close()


def test_filter_uses_uint64_uids_and_preserves_source_and_order(tmp_path):
    row = indexed_image(tmp_path)
    uid = int(row[0]["uid"])
    source = particles([uid, uid, uid], [[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
    before = source.copy()
    decisions, summary = filter_by_index(source, [row], exclusion_angstrom=2)
    assert decisions.tolist() == [1, 0, 0]
    assert summary["accepted"] == 2
    assert np.array_equal(source, before)
    assert source["uid"][decisions == 0].tolist() == [101, 102]


def test_filter_matches_coordinate_reference_with_binned_mask(tmp_path):
    from cryofilter.particle_filtering import classify_particle_coordinates_by_mask
    mask = np.zeros((32, 40), dtype=bool)
    mask[14:18, 19:23] = True
    row = indexed_image(tmp_path, mask=mask)
    rng = np.random.default_rng(20)
    points = rng.random((2000, 2)).astype(np.float32)
    data = particles([int(row[0]["uid"])] * len(points), points)
    decisions, _ = filter_by_index(data, [row], exclusion_angstrom=8)
    keep, _ = classify_particle_coordinates_by_mask(points * [80, 64], bad_mask=mask,
        input_shape=(64, 80), pixel_size_angstrom=2, exclusion_distance_angstrom=8)
    assert np.array_equal(decisions == 0, keep)


@pytest.mark.parametrize("kind", ["rectangle", "fragmented", "empty", "full"])
@pytest.mark.parametrize("exclusion", [0, 2, 8.123])
def test_fast_boundary_query_matches_legacy_scalar_distance_rule(kind, exclusion):
    from scipy.ndimage import distance_transform_edt
    from cryofilter.particle_filtering import _distance_for_input_coordinate
    rng = np.random.default_rng(918)
    mask = np.zeros((97, 123), dtype=bool)
    if kind == "rectangle":
        mask[30:60, 28:65] = True
    elif kind == "fragmented":
        mask[:] = rng.random(mask.shape) > 0.5
    elif kind == "full":
        mask[:] = True
    points = np.r_[rng.random((3000, 2)).astype(np.float32), [[0, 0], [1, 1], [0.5, 0.5]]]
    shape, px = (4092, 5760), 1.06
    if mask.any():
        distance = distance_transform_edt(~mask, sampling=(px * shape[0] / mask.shape[0], px * shape[1] / mask.shape[1])).astype(np.float32)
    else:
        distance = np.full(mask.shape, np.inf, dtype=np.float32)
    expected = [_distance_for_input_coordinate(distance, x_input_px=float(x) * shape[1],
                y_input_px=float(y) * shape[0], input_shape=shape) > exclusion for x, y in points]
    actual = keep_from_mask(points, mask, shape=shape, pixel_size=px, exclusion=exclusion)
    assert np.array_equal(actual, expected)


def test_missing_masks_are_errors_or_separate_pending_never_accepted(tmp_path):
    row = indexed_image(tmp_path)
    data = particles([int(row[0]["uid"]), 2], [[0.1, 0.1], [0.1, 0.1]])
    with pytest.raises(ValueError, match="1 micrograph"):
        filter_by_index(data, [row])
    decisions, summary = filter_by_index(data, [row], exclusion_angstrom=0, missing_masks="pending")
    assert decisions.tolist() == [0, 2]
    assert summary["pending"] == 1
    Path(row[0]["path"]).touch()
    assert filter_by_index(data, [row], missing_masks="pending")[0].tolist() == [2, 2]


def test_card_lookup_and_monitor_survive_a_new_index_reader(tmp_path):
    row = indexed_image(tmp_path)
    index = Index(tmp_path / "index.sqlite")
    index.discover(row[0])
    index.complete("segmentation", row[0], row[1])
    index.set_meta("status", {"phase": "waiting_for_motion_correction", "expected": 10, "segmented": 1})
    index.close()
    write_json(tmp_path / CARD_FILE, {"format": FORMAT, "project": "P1", "job_uid": "J5", "index_path": str(tmp_path / "index.sqlite")})
    _, path = read_card(tmp_path, project="P1", job_uid="J5")
    assert path.name == "index.sqlite"
    summary = _build_live_summary([tmp_path], mode="all", count=100)
    assert summary["n_images_completed"] == 1 and summary["n_images_total"] == 10
    assert summary["otf"]["phase"] == "waiting_for_motion_correction"
    with pytest.raises(ValueError, match="matching"):
        read_card(tmp_path, project="P2", job_uid="J5")


def test_ui_commands_and_cli_enforce_gpu_policy_and_hide_credentials(tmp_path):
    payload = {"project": "P1", "workspace": "W2", "micrographs": "J4", "gpu_devices": "1",
               "cryosparc_base_url": "http://silva.mit.edu:39000", "cryosparc_password": "test-secret",
               "checkpoint": str(tmp_path / "weights.pt")}
    spec = build_job_spec("cryosparc_otf", payload, work_dir=tmp_path)
    args = _build_parser().parse_args(spec.steps[0].argv[3:])
    assert args.cryosparc_command == "otf" and not args.run_typing
    assert args.host == "local"
    assert "test-secret" not in str(spec.steps) + str(spec.metadata)
    assert spec.env["CRYOSPARC_PASSWORD"] == "test-secret"
    with pytest.raises(ValueError, match="2 allocated GPUs"):
        build_job_spec("cryosparc_otf", dict(payload, run_typing=True), work_dir=tmp_path)
    spec = build_job_spec("cryosparc_filter_otf", dict(payload, otf_job="J5", particles="J20"), work_dir=tmp_path)
    args = _build_parser().parse_args(spec.steps[0].argv[3:])
    assert args.cryosparc_command == "filter-otf" and args.otf_job == "J5"


def test_supervisor_drains_late_arrivals_and_typing_with_one_model_pool(tmp_path, monkeypatch):
    from cryofilter.cryosparc import otf
    rows = [indexed_image(tmp_path, uid=str(i)) for i in (11, 12)]
    class Source:
        status = "running"
        expected = 2
        final_entries = None
        ticks = 0

        def refresh(self):
            self.ticks += 1
            if self.ticks >= 4:
                self.status = "completed"
                self.final_entries = {r[0]["uid"]: Path(r[0]["path"]) for r in rows}

        def scan(self, known):
            available = rows[:1] if self.ticks < 3 else rows
            return [r[0] for r in available if r[0]["uid"] not in known], []

    class Pool:
        capacity = {"segmentation": 2, "typing": 2}
        pending = []
        submitted = []

        def submit(self, role, entry):
            self.submitted.append((role, entry["uid"]))
            row = next(r for r in rows if r[0]["uid"] == entry["uid"])
            record = row[1] if role == "segmentation" else {"path": row[1]["output_mask_npy"], "summary": {}}
            self.pending.append((role, entry, record, 0.01, None))
            return True

        def collect(self):
            result, self.pending = self.pending, []
            return result

        def check(self):
            pass

    counter = [0.0]
    def tick():
        counter[0] += 1
        return counter[0]
    monkeypatch.setattr(otf.time, "monotonic", tick)
    monkeypatch.setattr(otf.time, "sleep", lambda _: None)
    index = Index(tmp_path / "index.sqlite")
    pool = Pool()
    args = SimpleNamespace(max_output_gb=1, run_typing=True, poll_seconds=1, gpu_devices="1,2")
    card = SimpleNamespace(uid="J8", log=lambda *a, **k: None)
    try:
        supervise(Source(), index, pool, card, args, tmp_path)
        assert sorted(pool.submitted) == [("segmentation", "11"), ("segmentation", "12"), ("typing", "11"), ("typing", "12")]
        assert index.get_meta("status")["phase"] == "completed"
        assert all(i and t for _, i, t in index.rows())
    finally:
        index.close()


def test_stream_reuses_loaded_model_and_matches_batch_masks(tmp_path, monkeypatch):
    import torch
    from cryofilter import cli
    from models import bad_region_detector

    class TinyModel(torch.nn.Module):
        def forward(self, image):
            return 8 * (image[:, :1] - 0.5)

    loads = []
    def make_model(**kwargs):
        loads.append(kwargs)
        return TinyModel()
    monkeypatch.setattr(bad_region_detector, "create_model", make_model)
    weights = tmp_path / "tiny.pt"
    torch.save({"model_state_dict": {"placeholder": torch.tensor(1)}, "model_type": "unet_attention",
                "use_power_spectrum": False, "input_channels": 1, "num_classes": 1}, weights)
    paths = []
    for n in range(2):
        path = micrograph(tmp_path / f"image{n}.mrc")
        with mrcfile.open(path, mode="r+") as mrc:
            mrc.data[:] = np.random.default_rng(n).normal(size=mrc.data.shape)
        paths.append(path)
    common = ["infer", "--input", str(tmp_path), "--checkpoint", str(weights), "--device", "cpu",
              "--patch-size", "48", "--overlap", "24", "--no-small-pixel-auto", "--batch-forward-size", "2"]
    batch_args = _build_parser().parse_args(common + ["--output-dir", str(tmp_path / "batch")])
    batch = cli._run_infer(batch_args, mrc_paths_override=paths, finalize=False)
    stream_args = _build_parser().parse_args(common + ["--output-dir", str(tmp_path / "stream")])
    emitted = []
    def incoming():
        yield {"path": paths[0], "stem": "uid1"}
        assert len(emitted) == 1  # Results become usable before the next image arrives.
        yield {"path": paths[1], "stem": "uid2"}
    result = cli._run_infer(stream_args, input_stream=incoming(), on_result=emitted.append, finalize=False)
    assert len(loads) == 2  # One load for the batch, one for the entire stream.
    assert result["inputs"] == []  # No ever-growing per-worker manifest.
    for baseline, streamed in zip(batch["inputs"], emitted):
        assert np.array_equal(np.load(baseline["output_mask_npy"]), load_mask(streamed["output_mask_npy"]))
        assert streamed["probability_scope"] == "preview_only"


def test_resume_cannot_change_status_of_an_active_runner(tmp_path, monkeypatch):
    import fcntl
    from cryofilter.cryosparc import otf
    run_dir = tmp_path / "run"
    job_dir = tmp_path / "J8"
    job_dir.mkdir()
    index = Index(run_dir / "index.sqlite")
    index.set_meta("status", {"phase": "processing"})
    index.close()
    write_json(job_dir / CARD_FILE, {"format": FORMAT, "project": "P1", "job_uid": "J8", "index_path": str(run_dir / "index.sqlite")})
    stopped = []
    external = SimpleNamespace(dir=lambda: job_dir, stop=lambda **kwargs: stopped.append(kwargs))
    project = SimpleNamespace(find_external_job=lambda _: external)
    source = SimpleNamespace(project_dir=tmp_path, refresh=lambda: None)
    monkeypatch.setattr(otf, "_options", lambda args: (["1"], {}, {}))
    monkeypatch.setattr(otf, "_validate_devices", lambda _: None)
    monkeypatch.setattr(otf, "_connect", lambda _: None)
    monkeypatch.setattr(otf, "find_project", lambda *args: project)
    monkeypatch.setattr(otf, "find_job", lambda *args: None)
    monkeypatch.setattr(otf, "PatchSource", lambda *args: source)
    args = _build_parser().parse_args(["cryosparc", "otf", "--project", "P1", "--workspace", "W1",
        "--micrographs", "J4", "--resume-job", "J8", "--gpu-devices", "1", "--checkpoint", "unused.pt"])
    with (run_dir / "runner.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active runner"):
            otf.run_otf(args, None)
    index = Index(run_dir / "index.sqlite", readonly=True)
    try:
        assert index.get_meta("status")["phase"] == "processing"
        assert not stopped
    finally:
        index.close()


@pytest.mark.parametrize("changed", ["source", "mask", "probability", "typing"])
def test_restart_invalidates_changed_cached_files(tmp_path, changed):
    from cryofilter.cryosparc.otf import _validate_cached_outputs
    import os
    entry, inference, _ = indexed_image(tmp_path)
    typed_path = tmp_path / "typed.npy"
    np.save(typed_path, np.zeros(entry["shape"], dtype=np.uint8))
    typed = {"path": str(typed_path), "output_stamp": file_stamp(typed_path), "summary": {}}
    index = Index(tmp_path / "index.sqlite")
    try:
        index.discover(entry)
        index.complete("segmentation", entry, inference)
        index.complete("typing", entry, typed)
        path = Path({"source": entry["path"], "mask": inference["output_mask_npy"],
                     "probability": inference["output_prob_npy"], "typing": typed["path"]}[changed])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        _validate_cached_outputs(index)
        _, cached_inference, cached_typing = index.rows()[0]
        assert (cached_inference is not None) == (changed == "typing")
        assert cached_typing is None
    finally:
        index.close()


def test_resume_reserves_space_before_unfinished_typing(tmp_path):
    row = indexed_image(tmp_path)
    index = Index(tmp_path / "index.sqlite")
    try:
        index.discover(row[0])
        index.complete("segmentation", row[0], row[1])
        source = SimpleNamespace(refresh=lambda: None, scan=lambda _: ([], []))
        def unexpected_submit(*args):
            pytest.fail("Typing must not start when its output exceeds the budget")
        pool = SimpleNamespace(collect=lambda: [], check=lambda: None,
                               capacity={"segmentation": 2, "typing": 2}, submit=unexpected_submit)
        args = SimpleNamespace(max_output_gb=0.0001, run_typing=True, poll_seconds=1)
        with pytest.raises(ValueError, match="output budget"):
            supervise(source, index, pool, None, args, tmp_path)
        assert index.rows()[0][1] == row[1]
    finally:
        index.close()


def test_invalidated_image_waits_for_source_readiness_before_resubmission(tmp_path, monkeypatch):
    from cryofilter.cryosparc import otf
    entry, inference, _ = indexed_image(tmp_path)
    index = Index(tmp_path / "index.sqlite")
    index.discover(entry)
    ticks = [0]
    pending = []
    def refresh():
        ticks[0] += 1
    def submit(role, task):
        assert ticks[0] >= 3
        pending.append((role, task, inference, 0.1, None))
        return True
    def collect():
        result = pending[:]
        pending.clear()
        return result
    source = SimpleNamespace(refresh=refresh, status="completed", expected=1,
        final_entries={entry["uid"]: entry["path"]},
        scan=lambda known: (([entry], []) if ticks[0] >= 3 else ([], [entry["uid"]])))
    pool = SimpleNamespace(collect=collect, check=lambda: None, submit=submit,
                           capacity={"segmentation": 2, "typing": 0})
    counter = [0]
    def clock():
        counter[0] += 1
        return counter[0]
    monkeypatch.setattr(otf.time, "monotonic", clock)
    monkeypatch.setattr(otf.time, "sleep", lambda _: None)
    args = SimpleNamespace(max_output_gb=1, run_typing=False, poll_seconds=1, gpu_devices="1")
    try:
        supervise(source, index, pool, SimpleNamespace(uid="J8", log=lambda *a, **k: None), args, tmp_path)
        assert index.get_meta("status")["phase"] == "completed"
    finally:
        index.close()


def test_patch_source_retries_delayed_final_publication(tmp_path):
    class DelayedJob(FakeJob):
        status = "completed"
        calls = 0
        def load_output(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Output not yet published")
            return self.data
    data = np.zeros(1, dtype=[("uid", "<u8"), ("micrograph_blob/path", "U32")])
    data["uid"] = 1
    data["micrograph_blob/path"] = "J4/motioncorrected/mic.mrc"
    job = DelayedJob(tmp_path, data)
    source = PatchSource(job, SimpleNamespace(dir=lambda: tmp_path))
    with pytest.raises(ConnectionError, match="final micrograph output"):
        source.refresh()
    source.refresh()
    assert source.final_entries == {"1": (tmp_path / data["micrograph_blob/path"][0]).resolve()}


def test_filter_card_roundtrip_preserves_uids_and_metadata(tmp_path, monkeypatch):
    from cryofilter.cryosparc import otf, otf_filter
    row = indexed_image(tmp_path)
    index = Index(tmp_path / "index.sqlite")
    index.discover(row[0])
    index.complete("segmentation", row[0], row[1])
    index.close()
    write_json(tmp_path / CARD_FILE, {"format": FORMAT, "project": "P1", "job_uid": "J8",
                                    "index_path": str(tmp_path / "index.sqlite")})
    raw = particles([int(row[0]["uid"]), int(row[0]["uid"]), 12],
                    [[0.1, 0.1], [0.5, 0.5], [0.1, 0.1]])
    before = raw.copy()
    class Dataset:
        def __init__(self, array):
            self.array = array
        def __getitem__(self, key):
            return self.array[key]
        def fields(self):
            return self.array.dtype.names
        def prefixes(self):
            return ["location", "ctf"]
        def take(self, indices):
            return Dataset(self.array[indices])
        def filter_prefixes(self, prefixes, copy):
            return Dataset(self.array[[n for n in self.fields() if n == "uid" or n.split("/")[0] in prefixes]].copy())
    class External:
        uid = "J30"
        outputs = {}
        allocations = {}
        stopped = False
        def add_input(self, **kwargs):
            assert set(kwargs["slots"]) == {"location", "ctf"}
        def connect(self, **kwargs):
            assert kwargs["source_job_uid"] == "J20"
        def start(self):
            pass
        def stop(self, **kwargs):
            assert not kwargs
            self.stopped = True
        def add_output(self, **kwargs):
            assert kwargs["passthrough"] == "input_particles"
            self.allocations[kwargs["name"]] = kwargs["alloc"]
        def save_output(self, name, dataset):
            assert set(dataset.fields()) == {n for n in raw.dtype.names if n == "uid" or n.startswith("location/")}
            self.outputs[name] = dataset
        def log(self, message):
            pass
    external = External()
    picker = SimpleNamespace(status="completed", load_output=lambda _: Dataset(raw))
    card = SimpleNamespace(dir=lambda: tmp_path)
    project = SimpleNamespace(find_external_job=lambda _: card, find_job=lambda _: picker,
                              create_external_job=lambda **kwargs: external)
    monkeypatch.setattr(otf, "_connect", lambda _: SimpleNamespace(find_project=lambda _: project))
    args = _build_parser().parse_args(["cryosparc", "filter-otf", "--project", "P1", "--workspace", "W1",
        "--otf-job", "J8", "--particles", "J20", "--missing-masks", "pending",
        "--particle-exclusion-distance-angstrom", "0", "--output-dir", str(tmp_path / "filtered")])
    assert otf_filter.run_filter(args, None) == 0
    assert external.stopped
    for name, uid in (("particles_accepted", 100), ("particles_rejected", 101), ("particles_pending", 102)):
        assert external.outputs[name]["uid"].tolist() == [uid]
        assert external.allocations[name]["ctf/df1_A"].tolist() == [12345]
    assert np.array_equal(raw, before)
    summary = json.loads((tmp_path / "filtered/particle_filter_summary.json").read_text())
    assert summary["external_job_uid"] == "J30" and summary["pending"] == 1


def test_event_gallery_caps_at_ten_and_reuses_cached_previews(tmp_path, monkeypatch):
    from PIL import Image
    from cryofilter import cli
    from cryofilter.cryosparc import otf
    rows = [indexed_image(tmp_path, uid=str(uid)) for uid in range(12, 0, -1)]
    output = tmp_path / "OTF_images"
    output.mkdir()
    first = otf._preview_gallery(rows[:1], output)
    assert first.name == "latest_micrographs.png"
    with Image.open(first) as picture:
        first_height = picture.height
        assert picture.format == "PNG" and picture.width == 1600
    final = otf._preview_gallery(rows, output)
    with Image.open(final) as picture:
        assert picture.width == 1600 and first_height < picture.height <= 6124
    assert len(list(output.glob("*_600px_particle_overlay.png"))) == 10
    assert first == final  # One current local gallery, even as the window grows.
    before = final.read_bytes()
    def unexpected_read(*args):
        pytest.fail("Cached previews must not reread micrographs")
    monkeypatch.setattr(cli, "_load_mrc_2d", unexpected_read)
    otf._preview_gallery(rows, output)
    assert final.read_bytes() == before


def test_preview_publishes_png_to_event_log_even_if_tile_update_fails(tmp_path):
    from cryofilter.cryosparc import otf
    from PIL import Image
    rows = [indexed_image(tmp_path, uid=str(uid)) for uid in (2, 1)]
    output = tmp_path / "OTF_images"
    output.mkdir()
    uploads = []
    def tile(path):
        assert Path(path).suffix == ".png"
        assert Path(path).parent == output / ".cryosparc"
        raise ConnectionError("Temporary tile error")
    def log_plot(*, figure, text, formats, flags):
        assert Path(figure).suffix == ".png"
        with Image.open(figure) as picture:
            assert picture.format == "PNG"
            assert picture.getpixel((0, 0)) == (255, 255, 255)
        uploads.append((text, formats))
        assert "cryofilter_otf_gallery" in flags
        return "image-event-1"
    external = SimpleNamespace(project_uid="P1", uid="J8", set_tile_image=tile, log_plot=log_plot,
        log_checkpoint=lambda **k: "checkpoint-1", log=lambda *a, **k: "progress-1",
        cs=SimpleNamespace(api=SimpleNamespace(jobs=SimpleNamespace(get_event_logs=lambda *a: []))))
    assert otf._publish_previews(rows, output, external)["image_event_id"] == "image-event-1"
    assert len(uploads) == 1
    assert "latest 2 micrographs (newest first)" in uploads[0][0]
    assert uploads[0][1] == ["png"]


class PreviewExternal:
    """Model public SDK event ordering and lost responses, without admin APIs."""
    project_uid, uid = "P1", "J8"
    def __init__(self, events=None, fail=None):
        self.events = events if events is not None else []
        self.fail, self.history_reads = fail, 0
        self.cs = SimpleNamespace(api=SimpleNamespace(jobs=SimpleNamespace(get_event_logs=self.get_events)))
    def get_events(self, *args):
        self.history_reads += 1
        return self.events
    def add(self, kind, **kwargs):
        event = SimpleNamespace(id=f"event-{len(self.events)}", created_at=len(self.events), type=kind, **kwargs)
        self.events.append(event)
        if self.fail == kind:
            self.fail = None
            raise ConnectionError("Response lost")
        return event.id
    def log_checkpoint(self, *, meta):
        return self.add("checkpoint", meta=meta)
    def log(self, text, *, id=None):
        if id:
            next(e for e in self.events if e.id == id).text = text
            return id
        return self.add("text", text=text)
    def log_plot(self, **kwargs):
        return self.add("image", **kwargs)


def test_each_changed_snapshot_has_one_gallery_after_its_checkpoint(tmp_path):
    from cryofilter.cryosparc.otf_preview_log import publish_gallery
    external = PreviewExternal()
    rows = [({"key": str(uid)}, {}, None) for uid in range(12, 0, -1)]
    for count in range(1, 13):
        result = publish_gallery(external, tmp_path / "gallery.png", rows[:count], "OTF processing")
        checkpoint = next(i for i, e in enumerate(external.events) if e.id == result["checkpoint_id"])
        visible = external.events[checkpoint + 1:]
        assert [e.type for e in visible] == ["text", "image"]
        assert f"latest {min(count, 10)} micrographs" in visible[-1].text
    # Once capped at ten, identical visible content adds no checkpoints or plots.
    assert sum(e.type == "checkpoint" for e in external.events) == 10
    assert external.history_reads == 1
    before = len(external.events)
    result = publish_gallery(external, tmp_path / "gallery.png", rows, "OTF complete")
    assert len(external.events) == before
    assert next(e for e in external.events if e.id == result["progress_event_id"]).text == "OTF complete"
    # Late typing changes the gallery without changing the micrograph order.
    typed = [(entry, inference, {"output_stamp": [1, 2]}) for entry, inference, _ in rows]
    publish_gallery(external, tmp_path / "gallery.png", typed, "OTF typing complete")
    assert sum(e.type == "checkpoint" for e in external.events) == 11


@pytest.mark.parametrize("failed_stage", ["checkpoint", "text", "image"])
def test_checkpoint_gallery_recovers_lost_responses_and_resume(tmp_path, failed_stage):
    from cryofilter.cryosparc.otf_preview_log import publish_gallery
    rows = [({"key": "1"}, {}, None)]
    external = PreviewExternal(fail=failed_stage)
    with pytest.raises(ConnectionError, match="Response lost"):
        publish_gallery(external, tmp_path / "gallery.png", rows, "OTF processing")
    resumed = PreviewExternal(events=external.events)
    publish_gallery(resumed, tmp_path / "gallery.png", rows, "OTF resumed")
    assert [e.type for e in resumed.events] == ["checkpoint", "text", "image"]
    assert resumed.events[1].text == "OTF resumed"
    publish_gallery(resumed, tmp_path / "gallery.png", rows, "OTF complete")
    assert len(resumed.events) == 3
    assert resumed.history_reads == 1


def test_white_cryosparc_exports_keep_ui_cache_and_style(tmp_path):
    from PIL import Image
    from cryofilter.app.server import _is_visible_artifact_path
    from cryofilter.cryosparc import otf
    rows = [indexed_image(tmp_path, uid=str(uid)) for uid in (2, 1)]
    output = tmp_path / "OTF_images"
    output.mkdir()
    ui_gallery = otf._preview_gallery(rows, output)
    ui_tile = otf._preview(*rows[0], output)
    original = {p: p.read_bytes() for p in output.glob("*.png")}
    white_gallery = otf._preview_gallery(rows, output, white_background=True)
    white_tile = otf._preview(*rows[0], output, white_background=True)
    assert ui_gallery != white_gallery and ui_tile != white_tile
    assert all(p.read_bytes() == data for p, data in original.items())
    for path in (ui_gallery, ui_tile):
        assert _is_visible_artifact_path(output, path)
    for path in (white_gallery, white_tile):
        assert not _is_visible_artifact_path(output, path)
        with Image.open(path) as picture:
            assert picture.getpixel((0, 0)) == (255, 255, 255)
    with Image.open(ui_gallery) as picture:
        assert picture.getpixel((0, 0)) == (25, 25, 25)


@pytest.mark.parametrize("deferred", [False, True])
def test_supervisor_latest_ten_galleries_and_slow_upload_backpressure(tmp_path, monkeypatch, deferred):
    from cryofilter.cryosparc import otf
    rows = [indexed_image(tmp_path, uid=str(uid)) for uid in range(1, 13)]
    events = [("segmentation", row[0], row[1], 0.1, None) for row in rows]
    # Typing of the oldest images finishes after they have left the latest-ten
    # window. It must not bring those images back into the current gallery.
    events += [("typing", row[0], {"path": row[1]["output_mask_npy"], "summary": {}}, 0.1, None) for row in rows]
    class Pool:
        capacity = {"segmentation": 20, "typing": 20}
        first = True
        def collect(self):
            if self.first:
                self.first = False
                return []
            return [events.pop(0)] if events else []
        def submit(self, *args):
            return True
        def check(self):
            pass
    class Source:
        status = "running"
        expected = 12
        final_entries = None
        def refresh(self):
            if not events:
                self.status = "completed"
                self.final_entries = {row[0]["uid"]: row[0]["path"] for row in rows}
        def scan(self, known):
            return [row[0] for row in rows if row[0]["uid"] not in known], []
    published = []
    def publish(snapshot, output, external, **kwargs):
        published.append([(entry["uid"], typed is not None) for entry, _, typed in snapshot])
    monkeypatch.setattr(otf, "_publish_previews", publish)
    class Future:
        def __init__(self, callback, args, kwargs):
            self.callback, self.args, self.kwargs = callback, args, kwargs
            self.finished = False
            if not deferred:
                self.result()
        def done(self):
            return self.finished
        def result(self):
            if not self.finished:
                self.callback(*self.args, **self.kwargs)
                self.finished = True
    class Executor:
        futures = []
        def submit(self, callback, *args, **kwargs):
            assert not any(not future.done() for future in self.futures), "Only one pending preview allowed"
            future = Future(callback, args, kwargs)
            self.futures.append(future)
            return future
    counter = [0]
    def clock():
        counter[0] += 16
        return counter[0]
    monkeypatch.setattr(otf.time, "monotonic", clock)
    monkeypatch.setattr(otf.time, "sleep", lambda _: None)
    index = Index(tmp_path / "index.sqlite")
    args = SimpleNamespace(max_output_gb=1, run_typing=True, poll_seconds=1, gpu_devices="0,1")
    try:
        executor = Executor()
        supervise(Source(), index, Pool(), SimpleNamespace(uid="J8", log=lambda *a, **k: None),
                  args, tmp_path, previews=executor, preview_dir=tmp_path)
        assert index.get_meta("status")["phase"] == "completed"
        assert index.get_meta("preview_uids") == [str(uid) for uid in range(3, 13)]
        assert all(len(snapshot) <= 10 for snapshot in published)
        assert published[0] == [("1", False)]
        assert published[-1] == [(str(uid), True) for uid in range(12, 2, -1)]
        if deferred:
            assert len(published) == 2  # Coalesce updates while the upload is slow.
            assert not events  # Segmentation and typing drained before upload finished.
        else:
            assert set(range(1, 11)).issubset({len(snapshot) for snapshot in published})
    finally:
        index.close()


@pytest.mark.parametrize("fail_first", [False, True])
def test_resumed_gallery_restores_completion_order_and_retries_final_upload(tmp_path, monkeypatch, fail_first):
    from cryofilter.cryosparc import otf
    rows = [indexed_image(tmp_path, uid=str(uid)) for uid in (1, 2, 3)]
    index = Index(tmp_path / "index.sqlite")
    for entry, inference, _ in rows:
        index.discover(entry)
        index.complete("segmentation", entry, inference)
    index.set_meta("preview_uids", ["3", "1", "2"])
    published = []
    def publish(snapshot, *args, progress_message=""):
        published.append([entry["uid"] for entry, _, _ in snapshot])
        return {"progress_event_id": "checkpoint-progress"}
    monkeypatch.setattr(otf, "_publish_previews", publish)
    class Executor:
        attempts = 0
        def submit(self, callback, *args, **kwargs):
            self.attempts += 1
            attempt = self.attempts
            def result():
                if fail_first and attempt == 1:
                    raise ConnectionError("Temporary upload failure")
                return callback(*args, **kwargs)
            return SimpleNamespace(done=lambda: False, result=result)
    source = SimpleNamespace(status="completed", expected=3, refresh=lambda: None,
        scan=lambda _: ([], []), final_entries={row[0]["uid"]: row[0]["path"] for row in rows})
    pool = SimpleNamespace(collect=lambda: [], check=lambda: None, capacity={"segmentation": 2, "typing": 0})
    args = SimpleNamespace(max_output_gb=1, run_typing=False, poll_seconds=1, gpu_devices="0")
    try:
        executor = Executor()
        supervise(source, index, pool, SimpleNamespace(uid="J8", log=lambda *a, **k: None), args, tmp_path,
                  previews=executor, preview_dir=tmp_path)
        assert index.get_meta("status")["phase"] == "completed"
        assert published == [["2", "1", "3"]] * (1 if fail_first else 2)
        assert executor.attempts == 2
        assert index.get_meta("progress_event_id") == "checkpoint-progress"
    finally:
        index.close()


def test_preview_averages_pixels_and_reduces_sampling_noise():
    from cryofilter.cryosparc.otf import _average_preview_pixels
    noise = np.random.default_rng(712).normal(size=(400, 560)).astype(np.float32)
    before = noise.copy()
    averaged = _average_preview_pixels(noise, max_display_dim=70)
    reference = noise.reshape(50, 8, 70, 8).mean(axis=(1, 3))
    np.testing.assert_allclose(averaged, reference, atol=1e-6)
    assert averaged.std() < noise[::8, ::8].std() * 0.2
    np.testing.assert_array_equal(noise, before)
    assert _average_preview_pixels(noise, max_display_dim=600) is noise


def test_preview_averaging_retains_odd_sized_image_edges():
    from cryofilter.cryosparc.otf import _average_preview_pixels
    raw = np.zeros((19, 23), dtype=np.float32)
    raw[-1, :] = 100
    raw[:, -1] = 100
    averaged = _average_preview_pixels(raw, max_display_dim=7)
    assert averaged.shape == (5, 6)
    assert np.all(averaged[-1, :] > 0) and np.all(averaged[:, -1] > 0)


def test_preview_refreshes_old_cache_and_keeps_discrete_overlays_aligned(tmp_path, monkeypatch):
    from PIL import Image
    from cryofilter import qc_render
    from cryofilter.cryosparc import otf
    mask = np.zeros((64, 80), dtype=bool)
    mask[:, 40:] = True
    entry, inference, _ = indexed_image(tmp_path, mask=mask)
    with mrcfile.open(entry["path"], mode="r+") as mrc:
        mrc.data[:] = mask * 2
    typed_path = tmp_path / "typed.npy"
    np.save(typed_path, np.where(mask, 3, 0).astype(np.uint8))
    inputs = [Path(entry["path"]), Path(inference["output_mask_npy"]), typed_path]
    originals = [path.read_bytes() for path in inputs]
    output = tmp_path / "OTF_images"
    output.mkdir()
    old_cache = output / (entry["key"] + "_typed_20px_particle_overlay.png")
    Image.new("RGB", (1, 1), "red").save(old_cache)
    captured = []
    def render(**kwargs):
        assert kwargs["image"].shape == kwargs["mask"].shape == kwargs["typed_mask"].shape == (16, 20)
        np.testing.assert_array_equal(kwargs["image"] > 1, kwargs["mask"])
        np.testing.assert_array_equal(kwargs["typed_mask"], kwargs["mask"] * 3)
        captured.append(kwargs["image"])
        Image.new("RGB", (20, 16)).save(kwargs["output_path"])
    monkeypatch.setattr(qc_render, "render_particle_overlay_png", render)
    path = otf._preview(entry, inference, {"path": str(typed_path)}, output, max_display_dim=20)
    assert len(captured) == 1 and path != old_cache
    assert old_cache.is_file() and path.is_file()
    assert [path.read_bytes() for path in inputs] == originals
