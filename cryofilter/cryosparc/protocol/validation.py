"""Validation helpers shared by the local orchestrator and remote bridge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

PROJECT_UID_RE = r"^P[0-9]+$"
WORKSPACE_UID_RE = r"^W[0-9]+$"
JOB_UID_RE = r"^J[0-9]+$"
OUTPUT_NAME_RE = r"^[A-Za-z0-9_.:-]+$"

_PROJECT_UID_PATTERN = re.compile(PROJECT_UID_RE)
_WORKSPACE_UID_PATTERN = re.compile(WORKSPACE_UID_RE)
_JOB_UID_PATTERN = re.compile(JOB_UID_RE)
_OUTPUT_NAME_PATTERN = re.compile(OUTPUT_NAME_RE)


@dataclass(frozen=True)
class ParsedOutputRef:
    project_uid: str
    job_uid: str
    output_name: str


def validate_project_uid(value: str) -> str:
    if not _PROJECT_UID_PATTERN.fullmatch(str(value)):
        raise ValueError(f"Invalid CryoSPARC project UID: {value!r}")
    return str(value)


def validate_workspace_uid(value: str) -> str:
    if not _WORKSPACE_UID_PATTERN.fullmatch(str(value)):
        raise ValueError(f"Invalid CryoSPARC workspace UID: {value!r}")
    return str(value)


def validate_job_uid(value: str) -> str:
    if not _JOB_UID_PATTERN.fullmatch(str(value)):
        raise ValueError(f"Invalid CryoSPARC job UID: {value!r}")
    return str(value)


def validate_output_name(value: str) -> str:
    if not _OUTPUT_NAME_PATTERN.fullmatch(str(value)):
        raise ValueError(f"Invalid CryoSPARC output name: {value!r}")
    return str(value)


def parse_output_ref(value: str, *, project_uid: str) -> ParsedOutputRef:
    """Parse a CLI value like ``J20:particles`` into an output reference."""

    text = str(value).strip()
    if ":" not in text:
        raise ValueError("Output references must have the form J123:output_name")
    job_uid, output_name = text.split(":", 1)
    return ParsedOutputRef(
        project_uid=validate_project_uid(project_uid),
        job_uid=validate_job_uid(job_uid),
        output_name=validate_output_name(output_name),
    )


@dataclass(frozen=True)
class UidPartitionReport:
    """Detailed validation report for UID-based accepted/rejected partitions."""

    ok: bool
    input_count: int
    accepted_count: int
    rejected_count: int
    duplicate_input_uids: tuple[int, ...]
    duplicate_accepted_uids: tuple[int, ...]
    duplicate_rejected_uids: tuple[int, ...]
    missing_uids: tuple[int, ...]
    unknown_uids: tuple[int, ...]
    overlapping_uids: tuple[int, ...]

    def require_ok(self) -> None:
        if not self.ok:
            raise ValueError(f"Invalid UID partition: {self}")


def _as_int_tuple(values: Iterable[object]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _duplicates(values: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    dupes: set[int] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return tuple(sorted(dupes))


def validate_uid_partition(
    *,
    input_uids: Iterable[object],
    accepted_uids: Iterable[object],
    rejected_uids: Iterable[object],
) -> UidPartitionReport:
    """Validate the lossless UID partition required before CryoSPARC finalization."""

    input_values = _as_int_tuple(input_uids)
    accepted_values = _as_int_tuple(accepted_uids)
    rejected_values = _as_int_tuple(rejected_uids)

    input_set = set(input_values)
    accepted_set = set(accepted_values)
    rejected_set = set(rejected_values)
    result_set = accepted_set | rejected_set

    report = UidPartitionReport(
        ok=False,
        input_count=len(input_values),
        accepted_count=len(accepted_values),
        rejected_count=len(rejected_values),
        duplicate_input_uids=_duplicates(input_values),
        duplicate_accepted_uids=_duplicates(accepted_values),
        duplicate_rejected_uids=_duplicates(rejected_values),
        missing_uids=tuple(sorted(input_set - result_set)),
        unknown_uids=tuple(sorted(result_set - input_set)),
        overlapping_uids=tuple(sorted(accepted_set & rejected_set)),
    )

    ok = (
        not report.duplicate_input_uids
        and not report.duplicate_accepted_uids
        and not report.duplicate_rejected_uids
        and not report.missing_uids
        and not report.unknown_uids
        and not report.overlapping_uids
        and len(accepted_values) + len(rejected_values) == len(input_values)
    )
    return UidPartitionReport(ok=ok, **{k: v for k, v in report.__dict__.items() if k != "ok"})


__all__ = [
    "ParsedOutputRef",
    "UidPartitionReport",
    "parse_output_ref",
    "validate_job_uid",
    "validate_output_name",
    "validate_project_uid",
    "validate_uid_partition",
    "validate_workspace_uid",
]
