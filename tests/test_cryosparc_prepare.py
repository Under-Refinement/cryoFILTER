from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from cryofilter.cryosparc.bridge import prepare as prepare_bridge
from cryofilter.cryosparc.protocol.models import PredictPrepareRequest, TransferManifest


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def fields(self) -> list[str]:
        return list(self._rows[0])

    def prefixes(self) -> list[str]:
        return sorted({field.split("/", 1)[0] for field in self.fields() if "/" in field})

    def __getitem__(self, field: str) -> list[object]:
        return [row[field] for row in self._rows]


class _FakeJob:
    def __init__(self, outputs: dict[str, object], *, job_dir: Path | None = None):
        self.model = {
            "output_result_groups": [
                {"name": name}
                for name in outputs
            ]
        }
        self.outputs = outputs
        self.load_attempts: list[str] = []
        self.dir = job_dir

    def load_output(self, name: str) -> object:
        self.load_attempts.append(name)
        if name not in self.outputs:
            raise RuntimeError(f"output {name!r} does not exist")
        return self.outputs[name]


class _FakeProject:
    def __init__(self, project_dir: Path, jobs: dict[str, _FakeJob]):
        self.model = {"project_dir": str(project_dir)}
        self.jobs = jobs

    def find_job(self, job_uid: str) -> _FakeJob:
        return self.jobs[job_uid]


class _FakeClient:
    def __init__(self, project: _FakeProject):
        self.project = project

    def find_project(self, project_uid: str) -> _FakeProject:
        return self.project


class _DirectReadClient:
    def __init__(self, jobs: dict[str, _FakeJob]):
        self.jobs = jobs
        self.job_lookups: list[tuple[str, str]] = []

    def find_job(self, project_uid: str, job_uid: str) -> _FakeJob:
        self.job_lookups.append((project_uid, job_uid))
        return self.jobs[job_uid]

    def find_project(self, project_uid: str) -> _FakeProject:
        raise AssertionError("read-only preparation must not load the project controller")


def _output_ref(output_name: str = "micrographs"):
    return prepare_bridge.CryoSPARCOutputRef.model_validate(
        {
            "project_uid": "P1",
            "job_uid": "J3",
            "output_name": output_name,
        }
    )


def test_job_output_names_reads_current_outputs_property() -> None:
    job = _FakeJob({"split_0": object(), "split_1": object(), "remainder": object()})
    job.model = {"spec": {"outputs": {"root": {"split_0": {}, "split_1": {}}}}}

    assert prepare_bridge._job_output_names(job) == ["remainder", "split_0", "split_1"]


def test_curate_exposures_prefers_accepted_micrographs() -> None:
    accepted = object()
    job = _FakeJob(
        {
            "micrographs_accepted": accepted,
            "micrographs_rejected": object(),
        }
    )

    dataset, resolved = prepare_bridge._load_output_with_aliases(
        job,
        _output_ref(),
        kind="micrographs",
    )

    assert dataset is accepted
    assert resolved.output_name == "micrographs_accepted"
    assert job.load_attempts == ["micrographs", "micrographs_accepted"]


def test_multi_output_exposure_set_requires_explicit_output() -> None:
    job = _FakeJob({"split_0": object(), "split_1": object(), "remainder": object()})

    with pytest.raises(ValueError) as error:
        prepare_bridge._load_output_with_aliases(job, _output_ref(), kind="micrographs")

    message = str(error.value)
    assert "Available output names: remainder, split_0, split_1" in message
    assert "J3:<output_name>" in message
    assert job.load_attempts == ["micrographs"]


def test_single_nonstandard_micrograph_output_is_resolved() -> None:
    split = object()
    job = _FakeJob({"split_0": split})

    dataset, resolved = prepare_bridge._load_output_with_aliases(
        job,
        _output_ref(),
        kind="micrographs",
    )

    assert dataset is split
    assert resolved.output_name == "split_0"
    assert job.load_attempts == ["micrographs", "split_0"]


def test_prepare_resolves_import_micrographs_output_alias(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "micrograph_001.mrc").write_bytes(b"mrc")
    micrographs = _FakeDataset(
        [
            {
                "uid": 7,
                "micrograph_blob/path": "micrograph_001.mrc",
                "micrograph_blob/idx": 0,
                "micrograph_blob/shape": (100, 200),
                "micrograph_blob/psize_A": 1.2,
            }
        ]
    )
    job = _FakeJob({"imported_micrographs": micrographs})
    project = _FakeProject(project_dir, {"J3": job})
    monkeypatch.setattr(
        prepare_bridge,
        "make_client",
        lambda connection_file=None: _FakeClient(project),
    )
    request = PredictPrepareRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000321",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "micrographs": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "model_id": "stage-test",
            "threshold": 0.6,
            "remote_work_root": str(tmp_path / "remote"),
            "limit_micrographs": 1,
            "create_external_job": False,
        }
    )

    response = prepare_bridge.run_prepare(request)
    manifest_path = Path(response.manifest_file)
    transfer = TransferManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    assert response.ok
    assert job.load_attempts == ["micrographs", "imported_micrographs"]
    assert transfer.micrographs_ref.output_name == "imported_micrographs"
    assert transfer.micrographs[0].transfer_filename == "micrographs/7_micrograph_001.mrc"


def test_prepare_can_write_manifest_without_staging_symlinks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source = project_dir / "micrograph_001.mrc"
    source.write_bytes(b"mrc")
    micrographs = _FakeDataset(
        [
            {
                "uid": 7,
                "micrograph_blob/path": "micrograph_001.mrc",
                "micrograph_blob/shape": (100, 200),
                "micrograph_blob/psize_A": 1.2,
            }
        ]
    )
    project = _FakeProject(project_dir, {"J3": _FakeJob({"micrographs": micrographs})})
    monkeypatch.setattr(
        prepare_bridge,
        "make_client",
        lambda connection_file=None: _FakeClient(project),
    )
    request = PredictPrepareRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000322",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "micrographs": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "model_id": "annotation",
            "threshold": 0.6,
            "remote_work_root": str(tmp_path / "remote"),
            "create_external_job": False,
            "stage_micrographs": False,
        }
    )

    response = prepare_bridge.run_prepare(request)
    transfer = TransferManifest.model_validate_json(
        Path(response.manifest_file).read_text(encoding="utf-8")
    )

    assert response.ok
    assert response.n_micrographs == 1
    assert transfer.micrographs[0].source_path == str(source)
    assert transfer.micrographs[0].remote_transfer_path is None
    assert not (Path(response.transfer_dir) / "micrographs" / "7_micrograph_001.mrc").exists()


def test_read_only_prepare_uses_direct_job_lookup_without_project_owner_access(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    job_dir = project_dir / "J3"
    job_dir.mkdir(parents=True)
    source = project_dir / "micrograph_001.mrc"
    source.write_bytes(b"mrc")
    micrographs = _FakeDataset(
        [
            {
                "uid": 7,
                "micrograph_blob/path": "micrograph_001.mrc",
                "micrograph_blob/shape": (100, 200),
                "micrograph_blob/psize_A": 1.2,
            }
        ]
    )
    client = _DirectReadClient(
        {"J3": _FakeJob({"micrographs": micrographs}, job_dir=job_dir)}
    )
    monkeypatch.setattr(
        prepare_bridge,
        "make_client",
        lambda connection_file=None: client,
    )
    request = PredictPrepareRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000325",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "micrographs": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "model_id": "annotation",
            "threshold": 0.6,
            "remote_work_root": str(tmp_path / "remote"),
            "create_external_job": False,
            "stage_micrographs": False,
        }
    )

    response = prepare_bridge.run_prepare(request)
    transfer = TransferManifest.model_validate_json(
        Path(response.manifest_file).read_text(encoding="utf-8")
    )

    assert response.ok
    assert client.job_lookups == [("P1", "J3")]
    assert transfer.micrographs[0].source_path == str(source)


def test_prepare_falls_back_to_existing_non_dw_micrograph_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source = project_dir / "micrograph_001_non_dw.mrc"
    source.write_bytes(b"mrc")
    micrographs = _FakeDataset(
        [
            {
                "uid": 7,
                "micrograph_blob/path": "missing_micrograph_001.mrc",
                "micrograph_blob_non_dw/path": "micrograph_001_non_dw.mrc",
                "micrograph_blob_non_dw/shape": (100, 200),
                "micrograph_blob_non_dw/psize_A": 1.2,
            }
        ]
    )
    project = _FakeProject(project_dir, {"J3": _FakeJob({"micrographs": micrographs})})
    monkeypatch.setattr(
        prepare_bridge,
        "make_client",
        lambda connection_file=None: _FakeClient(project),
    )
    request = PredictPrepareRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000323",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "micrographs": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "model_id": "stage-test",
            "threshold": 0.6,
            "remote_work_root": str(tmp_path / "remote"),
            "create_external_job": False,
        }
    )

    response = prepare_bridge.run_prepare(request)
    transfer = TransferManifest.model_validate_json(
        Path(response.manifest_file).read_text(encoding="utf-8")
    )

    assert response.ok
    assert transfer.micrographs[0].source_path == str(source)
    assert transfer.micrographs[0].blob_prefix == "micrograph_blob_non_dw"
    assert transfer.micrographs[0].shape_yx == (100, 200)


def test_prepare_missing_source_lists_all_candidate_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    micrographs = _FakeDataset(
        [
            {
                "uid": 7,
                "micrograph_blob/path": "missing_dw.mrc",
                "micrograph_blob_non_dw/path": "missing_non_dw.mrc",
            }
        ]
    )
    project = _FakeProject(project_dir, {"J3": _FakeJob({"micrographs": micrographs})})
    monkeypatch.setattr(
        prepare_bridge,
        "make_client",
        lambda connection_file=None: _FakeClient(project),
    )
    request = PredictPrepareRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000324",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "micrographs": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "model_id": "stage-test",
            "threshold": 0.6,
            "remote_work_root": str(tmp_path / "remote"),
            "create_external_job": False,
        }
    )

    with pytest.raises(FileNotFoundError) as error:
        prepare_bridge.run_prepare(request)

    message = str(error.value)
    assert "micrograph_blob/path=" in message
    assert "missing_dw.mrc" in message
    assert "micrograph_blob_non_dw/path=" in message
    assert "missing_non_dw.mrc" in message
