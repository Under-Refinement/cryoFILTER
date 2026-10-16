from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")

from cryofilter.cryosparc.bridge import finalize as finalize_bridge


class _FakeDataset:
    def __init__(self, columns: dict[str, list[object] | np.ndarray]):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}

    def __len__(self) -> int:
        return len(self._columns["uid"])

    def fields(self) -> list[str]:
        return list(self._columns)

    def prefixes(self) -> list[str]:
        return sorted({name.split("/", 1)[0] for name in self._columns if "/" in name})

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]

    def take(self, indices: np.ndarray) -> "_FakeDataset":
        return _FakeDataset({name: values[indices] for name, values in self._columns.items()})


class _FakeSourceJob:
    def __init__(self, particles: _FakeDataset):
        self.particles = particles

    def load_output(self, name: str) -> _FakeDataset:
        assert name == "particles"
        return self.particles


class _FakeExternalJob:
    def __init__(self):
        self.output_specs: list[dict[str, object]] = []
        self.saved: dict[str, _FakeDataset] = {}
        self.started = False
        self.stop_error: str | None = None

    def add_output(self, **kwargs):
        self.output_specs.append(kwargs)
        return kwargs["name"]

    def save_output(self, *, name: str, dataset: _FakeDataset) -> None:
        self.saved[name] = dataset

    def start(self) -> None:
        self.started = True

    def stop(self, error: str | None = None) -> None:
        self.stop_error = error


class _FakeProject:
    def __init__(self, source_job: _FakeSourceJob, external_job: _FakeExternalJob):
        self.source_job = source_job
        self.external_job = external_job

    def find_job(self, uid: str):
        assert uid == "J4"
        return self.source_job

    def find_external_job(self, uid: str):
        assert uid == "J9"
        return self.external_job


class _FakeClient:
    def __init__(self, project: _FakeProject):
        self.project = project

    def find_project(self, uid: str) -> _FakeProject:
        assert uid == "P1"
        return self.project


def test_finalize_writes_self_contained_particle_metadata_without_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "00000000-0000-0000-0000-000000000901"
    transfer_file = tmp_path / "transfer_manifest.json"
    result_file = tmp_path / "result_manifest.json"
    accepted_file = tmp_path / "accepted.json"
    rejected_file = tmp_path / "rejected.json"
    accepted_file.write_text(json.dumps({"uids": [11, 13]}), encoding="utf-8")
    rejected_file.write_text(json.dumps({"uids": [12]}), encoding="utf-8")
    transfer_file.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "run_id": run_id,
                "project_uid": "P1",
                "workspace_uid": "W2",
                "external_job_uid": "J9",
                "micrographs_ref": {
                    "project_uid": "P1",
                    "job_uid": "J3",
                    "output_name": "micrographs",
                },
                "particles_ref": {
                    "project_uid": "P1",
                    "job_uid": "J4",
                    "output_name": "particles",
                },
                "micrographs": [],
                "particles": [
                    {
                        "uid": uid,
                        "micrograph_uid": 7,
                        "center_x_frac": 0.25,
                        "center_y_frac": 0.5,
                    }
                    for uid in (11, 12, 13)
                ],
            }
        ),
        encoding="utf-8",
    )
    result_file.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "run_id": run_id,
                "project_uid": "P1",
                "workspace_uid": "W2",
                "external_job_uid": "J9",
                "status": "completed",
                "micrographs_processed": 1,
                "particles_processed": 3,
                "accepted_particles": 2,
                "rejected_particles": 1,
                "files": {
                    "accepted_uids_json": str(accepted_file),
                    "rejected_uids_json": str(rejected_file),
                },
            }
        ),
        encoding="utf-8",
    )

    source = _FakeDataset(
        {
            "uid": [11, 12, 13],
            "location/micrograph_uid": [7, 7, 7],
            "location/center_x_frac": [0.1, 0.2, 0.3],
            "pick_stats/ncc_score": [0.8, 0.4, 0.9],
            "ctf/df1_A": [10000.0, 11000.0, 12000.0],
        }
    )
    external = _FakeExternalJob()
    project = _FakeProject(_FakeSourceJob(source), external)
    monkeypatch.setattr(finalize_bridge, "make_client", lambda _connection_file=None: _FakeClient(project))

    response = finalize_bridge.finalize_prediction(
        finalize_bridge.FinalizePredictionRequest(
            project_uid="P1",
            workspace_uid="W2",
            job_uid="J9",
            transfer_manifest_file=str(transfer_file),
            result_manifest_file=str(result_file),
        )
    )

    assert response.ok
    assert external.started
    assert external.stop_error is None
    assert len(external.output_specs) == 2
    assert all(spec.get("passthrough") is None for spec in external.output_specs)
    assert all(spec["slots"] == ["ctf", "location", "pick_stats"] for spec in external.output_specs)
    assert external.saved["particles_accepted"].fields() == source.fields()
    assert external.saved["particles_rejected"].fields() == source.fields()
    assert external.saved["particles_accepted"]["pick_stats/ncc_score"].tolist() == [0.8, 0.9]
    assert external.saved["particles_rejected"]["pick_stats/ncc_score"].tolist() == [0.4]
