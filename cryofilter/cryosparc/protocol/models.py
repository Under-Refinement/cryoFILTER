"""Pydantic models for the cryoFILTER CryoSPARC remote protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from cryofilter.cryosparc import PROTOCOL_VERSION

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise RuntimeError(
        "The cryoFILTER CryoSPARC integration requires pydantic>=2. "
        "Install the bridge dependencies with: pip install pydantic"
    ) from exc

PROJECT_UID_RE = r"^P[0-9]+$"
WORKSPACE_UID_RE = r"^W[0-9]+$"
JOB_UID_RE = r"^J[0-9]+$"
OUTPUT_NAME_RE = r"^[A-Za-z0-9_.:-]+$"


class ProtocolModel(BaseModel):
    """Base class that rejects unknown JSON fields and version mismatches."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: int = Field(default=PROTOCOL_VERSION)

    @field_validator("protocol_version")
    @classmethod
    def _validate_protocol_version(cls, value: int) -> int:
        if int(value) != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported protocol_version={value}; expected {PROTOCOL_VERSION}"
            )
        return int(value)


class CryoSPARCOutputRef(BaseModel):
    """Reference to a CryoSPARC job output group."""

    model_config = ConfigDict(extra="forbid")

    project_uid: str = Field(pattern=PROJECT_UID_RE)
    job_uid: str = Field(pattern=JOB_UID_RE)
    output_name: str = Field(pattern=OUTPUT_NAME_RE, min_length=1)


class BridgeCapabilities(BaseModel):
    """Feature flags detected by the remote bridge."""

    model_config = ConfigDict(extra="forbid")

    external_jobs: bool = False
    job_run_context: bool = False
    tile_images: bool = False


class BridgeDoctorResponse(ProtocolModel):
    """Machine-readable bridge health report."""

    ok: bool
    hostname: str
    bridge_version: str
    git_commit: str | None = None
    python_version: str
    cryosparc_tools_version: str | None = None
    cryosparc_server_version: str | None = None
    connection: bool = False
    authenticated: bool = False
    project_access: bool = False
    capabilities: BridgeCapabilities = Field(default_factory=BridgeCapabilities)
    errors: list[str] = Field(default_factory=list)


class InspectRequest(ProtocolModel):
    """Request normalized CryoSPARC metadata."""

    project_uid: str | None = Field(default=None, pattern=PROJECT_UID_RE)
    workspace_uid: str | None = Field(default=None, pattern=WORKSPACE_UID_RE)
    job_uid: str | None = Field(default=None, pattern=JOB_UID_RE)


class InspectResponse(ProtocolModel):
    """Normalized CryoSPARC metadata response."""

    ok: bool
    project_uid: str | None = None
    workspace_uid: str | None = None
    job_uid: str | None = None
    projects: list[dict[str, Any]] = Field(default_factory=list)
    workspaces: list[dict[str, Any]] = Field(default_factory=list)
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PredictPrepareRequest(ProtocolModel):
    """Initial prediction request sent from the local orchestrator to the bridge."""

    run_id: UUID = Field(default_factory=uuid4)
    project_uid: str = Field(pattern=PROJECT_UID_RE)
    workspace_uid: str = Field(pattern=WORKSPACE_UID_RE)
    micrographs: CryoSPARCOutputRef
    particles: CryoSPARCOutputRef | None = None
    model_id: str = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)
    remote_work_root: str = Field(min_length=1)
    title: str = "cryoFILTER prediction"
    limit_micrographs: int | None = Field(default=None, ge=1)
    micrograph_path_field: str | None = Field(default=None, min_length=1)
    create_external_job: bool = True
    stage_micrographs: bool = True


class PredictPrepareResponse(ProtocolModel):
    """Bridge response after creating the remote prediction staging area."""

    ok: bool
    run_id: UUID
    external_job_uid: str | None = Field(default=None, pattern=JOB_UID_RE)
    remote_run_dir: str
    transfer_dir: str
    manifest_file: str
    n_micrographs: int = Field(ge=0)
    n_micrographs_total: int = Field(ge=0)
    n_particles: int | None = Field(default=None, ge=0)
    total_micrograph_bytes: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class MicrographManifestEntry(BaseModel):
    """One micrograph staged by the bridge for transfer to engaging."""

    model_config = ConfigDict(extra="forbid")

    uid: int
    transfer_filename: str
    source_path: str
    source_relative_path: str | None = None
    remote_transfer_path: str | None = None
    source_index: int | None = None
    blob_prefix: str | None = None
    shape_yx: tuple[int, int] | None = None
    pixel_size_angstrom: float | None = None
    file_size_bytes: int | None = Field(default=None, ge=0)


class ParticleManifestEntry(BaseModel):
    """Minimal particle metadata needed by the compute worker."""

    model_config = ConfigDict(extra="forbid")

    uid: int
    micrograph_uid: int
    center_x_frac: float = Field(ge=0.0, le=1.0)
    center_y_frac: float = Field(ge=0.0, le=1.0)


class TransferManifest(ProtocolModel):
    """Manifest written by the bridge and pulled to the local orchestrator."""

    run_id: UUID
    project_uid: str = Field(pattern=PROJECT_UID_RE)
    workspace_uid: str = Field(pattern=WORKSPACE_UID_RE)
    external_job_uid: str | None = Field(default=None, pattern=JOB_UID_RE)
    micrographs_ref: CryoSPARCOutputRef
    particles_ref: CryoSPARCOutputRef | None = None
    micrographs: list[MicrographManifestEntry] = Field(default_factory=list)
    particles: list[ParticleManifestEntry] = Field(default_factory=list)


class DiagnosticAsset(BaseModel):
    """One diagnostic artifact intended for the CryoSPARC job event/card."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "otf_contact_sheet",
        "otf_overlay",
        "contamination_total_pie",
        "contamination_by_micrograph_bar",
        "contamination_type_pie",
        "contamination_type_bar",
    ]
    title: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    local_path: str
    remote_path: str | None = None
    mime_type: str = "image/png"
    source: str | None = None


class DiagnosticsManifest(ProtocolModel):
    """Local diagnostics selected for CryoSPARC write-back."""

    run_id: UUID | None = None
    assets: list[DiagnosticAsset] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class AttachDiagnosticsRequest(ProtocolModel):
    """Request bridge-side diagnostic image attachment to a CryoSPARC job."""

    project_uid: str = Field(pattern=PROJECT_UID_RE)
    job_uid: str = Field(pattern=JOB_UID_RE)
    diagnostics_manifest_file: str = Field(min_length=1)


class AttachDiagnosticsResponse(ProtocolModel):
    """Bridge-side response after logging diagnostic plots to a CryoSPARC job."""

    ok: bool
    project_uid: str
    job_uid: str
    diagnostics_manifest_file: str
    attached_assets: int = Field(ge=0)
    event_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ResultManifest(ProtocolModel):
    """Manifest pushed from the local orchestrator back to the bridge before finalization."""

    run_id: UUID
    project_uid: str | None = Field(default=None, pattern=PROJECT_UID_RE)
    workspace_uid: str | None = Field(default=None, pattern=WORKSPACE_UID_RE)
    external_job_uid: str | None = Field(default=None, pattern=JOB_UID_RE)
    transfer_manifest_file: str | None = None
    status: Literal["completed", "failed"]
    error: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    micrographs_processed: int = Field(ge=0)
    particles_processed: int = Field(ge=0)
    accepted_particles: int = Field(ge=0)
    rejected_particles: int = Field(ge=0)
    files: dict[str, str] = Field(default_factory=dict)
    diagnostics: list[DiagnosticAsset] = Field(default_factory=list)


class FinalizePredictionRequest(ProtocolModel):
    """Request bridge-side CryoSPARC output write-back for a completed prediction."""

    project_uid: str = Field(pattern=PROJECT_UID_RE)
    workspace_uid: str = Field(pattern=WORKSPACE_UID_RE)
    job_uid: str = Field(pattern=JOB_UID_RE)
    transfer_manifest_file: str = Field(min_length=1)
    result_manifest_file: str = Field(min_length=1)


class FinalizePredictionResponse(ProtocolModel):
    """Bridge-side response after saving particle outputs and attaching diagnostics."""

    ok: bool
    run_id: UUID | None = None
    project_uid: str
    workspace_uid: str
    job_uid: str
    status: Literal["completed", "failed"]
    input_particles: int = Field(default=0, ge=0)
    accepted_particles: int = Field(default=0, ge=0)
    rejected_particles: int = Field(default=0, ge=0)
    saved_outputs: dict[str, int] = Field(default_factory=dict)
    validated: bool = False
    attached_assets: int = Field(default=0, ge=0)
    event_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class TestRoundtripRequest(ProtocolModel):
    """Request a deterministic particle split inside a CryoSPARC External Job."""

    run_id: UUID = Field(default_factory=uuid4)
    project_uid: str = Field(pattern=PROJECT_UID_RE)
    workspace_uid: str = Field(pattern=WORKSPACE_UID_RE)
    particles: CryoSPARCOutputRef
    title: str = "cryoFILTER particle round-trip test"


class TestRoundtripResponse(ProtocolModel):
    """Result of the no-MRC External Job particle split test."""

    ok: bool
    run_id: UUID
    project_uid: str
    workspace_uid: str
    source_job_uid: str
    source_output_name: str
    external_job_uid: str | None = None
    input_particles: int = 0
    particles_a: int = 0
    particles_b: int = 0
    validated: bool = False
    errors: list[str] = Field(default_factory=list)


def model_to_json(model: BaseModel) -> str:
    """Serialize a pydantic model as compact JSON."""

    return model.model_dump_json(exclude_none=False)


def model_to_pretty_json(model: BaseModel) -> str:
    """Serialize a pydantic model as readable JSON."""

    return model.model_dump_json(indent=2, exclude_none=False)


def load_model_json(model_type: type[BaseModel], path: str | Path) -> BaseModel:
    """Load a protocol model from a JSON file."""

    return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
