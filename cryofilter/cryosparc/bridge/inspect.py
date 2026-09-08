"""Normalized CryoSPARC inspection commands for the remote bridge."""

from __future__ import annotations

import hashlib
from typing import Any

from cryofilter.cryosparc.bridge.client import make_client
from cryofilter.cryosparc.bridge.compat import find_job, find_project, find_workspace
from cryofilter.cryosparc.protocol.models import InspectRequest, InspectResponse


def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        if len(value) <= 64:
            return {"type": "bytes", "hex": value.hex()}
        return {
            "type": "bytes",
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass
    return str(value)


def _model_to_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        return _json_safe(payload) if isinstance(payload, dict) else {"value": str(payload)}
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        payload = dict_method()
        return _json_safe(payload) if isinstance(payload, dict) else {"value": str(payload)}
    if isinstance(value, dict):
        return _json_safe(value)
    return {"value": str(value)}


def _doc(obj: object) -> dict[str, Any]:
    model = getattr(obj, "model", None)
    if model is not None:
        return _model_to_dict(model)
    return _model_to_dict(getattr(obj, "doc", None))


def _list_from_candidate(obj: object, names: tuple[str, ...]) -> list[dict[str, Any]]:
    for name in names:
        candidate = getattr(obj, name, None)
        if callable(candidate):
            try:
                values = candidate()
            except TypeError:
                continue
            if values is None:
                return []
            return [_doc(value) if not isinstance(value, dict) else _json_safe(value) for value in values]
    return []


def _project_summaries(client: object) -> list[dict[str, Any]]:
    for name in ("list_projects", "find_projects"):
        candidate = getattr(client, name, None)
        if callable(candidate):
            try:
                values = candidate()
            except TypeError:
                values = candidate({})
            return [_doc(value) if not isinstance(value, dict) else _json_safe(value) for value in values]
    api = getattr(client, "api", None)
    projects = getattr(api, "projects", None)
    finder = getattr(projects, "find", None)
    if callable(finder):
        return [_json_safe(value) for value in finder({})]
    return []


def _job_outputs(job: object) -> list[dict[str, Any]]:
    doc = _doc(job)
    outputs: list[dict[str, Any]] = []
    for key in ("output_result_groups", "output_results", "outputs"):
        value = doc.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    outputs.append(_json_safe(item))
        elif isinstance(value, dict):
            for name, item in value.items():
                if isinstance(item, dict):
                    outputs.append(_json_safe({"name": name, **item}))
                else:
                    outputs.append(_json_safe({"name": name, "value": item}))
    return outputs


def inspect_cryosparc(
    request: InspectRequest,
    *,
    connection_file: str | None = None,
) -> InspectResponse:
    """Inspect projects, workspaces, jobs, or outputs via CryoSPARC Tools."""

    errors: list[str] = []
    projects: list[dict[str, Any]] = []
    workspaces: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    try:
        client = make_client(connection_file)
        if request.project_uid is None:
            projects = _project_summaries(client)
        else:
            project = find_project(client, request.project_uid)
            if request.workspace_uid is None and request.job_uid is None:
                workspaces = _list_from_candidate(
                    project,
                    ("list_workspaces", "find_workspaces", "workspaces"),
                )
                jobs = _list_from_candidate(project, ("list_jobs", "find_jobs", "jobs"))
            if request.workspace_uid is not None:
                workspace = find_workspace(
                    client,
                    project,
                    request.project_uid,
                    request.workspace_uid,
                )
                if workspace is not None:
                    workspaces = [_doc(workspace)]
                    jobs = _list_from_candidate(workspace, ("list_jobs", "find_jobs", "jobs"))
            if request.job_uid is not None:
                job = find_job(client, project, request.project_uid, request.job_uid)
                jobs = [_doc(job)]
                outputs = _job_outputs(job)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return InspectResponse(
        ok=not errors,
        project_uid=request.project_uid,
        workspace_uid=request.workspace_uid,
        job_uid=request.job_uid,
        projects=projects,
        workspaces=workspaces,
        jobs=jobs,
        outputs=outputs,
        errors=errors,
    )


__all__ = ["inspect_cryosparc"]
