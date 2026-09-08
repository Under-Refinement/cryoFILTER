"""Remote bridge health checks."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys

from cryofilter.cryosparc import PROTOCOL_VERSION
from cryofilter.cryosparc.bridge import BRIDGE_VERSION
from cryofilter.cryosparc.bridge.client import (
    cryosparc_tools_version,
    make_client,
    server_version,
    test_connection,
)
from cryofilter.cryosparc.bridge.compat import (
    create_external_job,
    find_project,
    supports_job_plot_logs,
)
from cryofilter.cryosparc.protocol.models import (
    BridgeCapabilities,
    BridgeDoctorResponse,
)
from cryofilter.cryosparc.protocol.validation import validate_project_uid


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def run_doctor(
    *,
    connection_file: str | None = None,
    project_uid: str | None = None,
) -> BridgeDoctorResponse:
    """Return a JSON-safe health report for the remote bridge."""

    errors: list[str] = []
    tools_version = cryosparc_tools_version()
    connection = False
    authenticated = False
    project_access = False
    version = None
    tile_images = supports_job_plot_logs()
    capabilities = BridgeCapabilities()

    if project_uid is not None:
        validate_project_uid(project_uid)

    try:
        client = make_client(connection_file)
        connection = test_connection(client)
        authenticated = bool(connection)
        version = server_version(client)

        project = None
        if project_uid is not None:
            project = find_project(client, project_uid)
            project_access = project is not None
        if project is not None:
            capabilities = BridgeCapabilities(
                external_jobs=hasattr(project, "create_external_job")
                or callable(getattr(create_external_job, "__call__", None)),
                job_run_context=True,
                tile_images=tile_images,
            )
        else:
            capabilities = BridgeCapabilities(
                external_jobs=tools_version is not None,
                job_run_context=tools_version is not None,
                tile_images=tile_images,
            )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    ok = bool(tools_version and connection and authenticated and not errors)
    if project_uid is not None:
        ok = ok and project_access

    return BridgeDoctorResponse(
        ok=ok,
        hostname=socket.gethostname(),
        bridge_version=BRIDGE_VERSION,
        git_commit=_git_commit(),
        protocol_version=PROTOCOL_VERSION,
        python_version=f"{platform.python_version()} ({sys.executable})",
        cryosparc_tools_version=tools_version,
        cryosparc_server_version=version,
        connection=connection,
        authenticated=authenticated,
        project_access=project_access,
        capabilities=capabilities,
        errors=errors,
    )


__all__ = ["run_doctor"]
