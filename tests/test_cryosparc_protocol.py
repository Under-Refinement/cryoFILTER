from __future__ import annotations

import json
from uuid import UUID

import pytest

pytest.importorskip("pydantic")

from cryofilter.cryosparc.protocol.models import (
    FinalizePredictionRequest,
    PredictPrepareRequest,
    ResultManifest,
    TestRoundtripRequest,
    TransferManifest,
)
from cryofilter.cryosparc.protocol.validation import (
    parse_output_ref,
    validate_uid_partition,
)


def test_output_ref_requires_job_and_output_name() -> None:
    ref = parse_output_ref("J20:particles", project_uid="P42")

    assert ref.project_uid == "P42"
    assert ref.job_uid == "J20"
    assert ref.output_name == "particles"

    with pytest.raises(ValueError):
        parse_output_ref("J20", project_uid="P42")
    with pytest.raises(ValueError):
        parse_output_ref("J20:bad name", project_uid="P42")


def test_protocol_version_rejects_mismatch() -> None:
    payload = {
        "protocol_version": 999,
        "run_id": str(UUID(int=1)),
        "project_uid": "P1",
        "workspace_uid": "W2",
        "particles": {"project_uid": "P1", "job_uid": "J3", "output_name": "particles"},
    }

    with pytest.raises(ValueError):
        TestRoundtripRequest.model_validate(payload)


def test_uid_partition_accepts_lossless_split() -> None:
    report = validate_uid_partition(
        input_uids=[1, 2, 3, 4],
        accepted_uids=[1, 3],
        rejected_uids=[2, 4],
    )

    assert report.ok
    assert report.accepted_count == 2
    assert report.rejected_count == 2


def test_uid_partition_rejects_missing_unknown_and_overlap() -> None:
    report = validate_uid_partition(
        input_uids=[1, 2, 3],
        accepted_uids=[1, 2],
        rejected_uids=[2, 99],
    )

    assert not report.ok
    assert report.missing_uids == (3,)
    assert report.unknown_uids == (99,)
    assert report.overlapping_uids == (2,)


def test_finalize_uid_helpers_preserve_unsigned_cryosparc_ids(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from cryofilter.cryosparc.bridge.finalize import _indices_for_uids, _read_uid_file

    high_uid = 2**63 + 123
    uid_file = tmp_path / "accepted_uids.json"
    uid_file.write_text(json.dumps({"uids": [high_uid]}), encoding="utf-8")

    assert _read_uid_file(uid_file) == [high_uid]

    indices = _indices_for_uids(
        np.asarray([17, high_uid], dtype=np.uint64),
        [high_uid],
        label="accepted",
    )

    assert indices.dtype == np.int64
    assert indices.tolist() == [1]


def test_prepare_manifest_allows_manifest_only_staging() -> None:
    request = PredictPrepareRequest.model_validate(
        {
            "run_id": "00000000-0000-0000-0000-000000000123",
            "project_uid": "P1",
            "workspace_uid": "W2",
            "micrographs": {
                "project_uid": "P1",
                "job_uid": "J3",
                "output_name": "micrographs",
            },
            "model_id": "stage-test",
            "threshold": 0.6,
            "remote_work_root": "/remote/work",
            "limit_micrographs": 1,
            "create_external_job": False,
        }
    )

    manifest = TransferManifest(
        run_id=request.run_id,
        project_uid=request.project_uid,
        workspace_uid=request.workspace_uid,
        external_job_uid=None,
        micrographs_ref=request.micrographs,
        micrographs=[
            {
                "uid": 123,
                "transfer_filename": "micrographs/123_example.mrc",
                "source_path": "/project/J1/imported/example.mrc",
                "remote_transfer_path": "/remote/work/run/transfer/micrographs/123_example.mrc",
                "file_size_bytes": 4,
            }
        ],
    )

    assert not request.create_external_job
    assert manifest.external_job_uid is None
    assert manifest.micrographs[0].transfer_filename.endswith(".mrc")


def test_finalize_prediction_contracts_reference_remote_result_files() -> None:
    request = FinalizePredictionRequest(
        project_uid="P1",
        workspace_uid="W2",
        job_uid="J5",
        transfer_manifest_file="/remote/run/transfer_manifest.json",
        result_manifest_file="/remote/run/results/result_manifest.json",
    )
    result = ResultManifest(
        run_id=UUID(int=10),
        project_uid="P1",
        workspace_uid="W2",
        external_job_uid="J5",
        transfer_manifest_file="/remote/run/transfer_manifest.json",
        status="completed",
        micrographs_processed=1,
        particles_processed=2,
        accepted_particles=1,
        rejected_particles=1,
        files={
            "accepted_uids_json": "/remote/run/results/accepted_uids.json",
            "rejected_uids_json": "/remote/run/results/rejected_uids.json",
            "diagnostics_manifest_json": "/remote/run/diagnostics/cryosparc_diagnostics_manifest.json",
        },
    )

    assert request.job_uid == result.external_job_uid
    assert result.files["accepted_uids_json"].endswith("accepted_uids.json")
