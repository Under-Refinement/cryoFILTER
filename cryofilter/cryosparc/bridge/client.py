"""CryoSPARC Tools client construction for the remote bridge."""

from __future__ import annotations

import json
import os
from importlib import metadata
from pathlib import Path
from typing import Any


def cryosparc_tools_version() -> str | None:
    """Return the installed cryosparc-tools package version when available."""

    for package_name in ("cryosparc-tools", "cryosparc_tools"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def import_cryosparc_class():
    """Import CryoSPARC from supported package layouts."""

    first_missing: ModuleNotFoundError | None = None
    try:
        from cryosparc.tools import CryoSPARC

        return CryoSPARC
    except ModuleNotFoundError as first_exc:
        if first_exc.name != "cryosparc":
            raise
        first_missing = first_exc

    try:
        from cryosparc_tools.cryosparc.tools import CryoSPARC

        return CryoSPARC
    except ModuleNotFoundError as second_exc:
        if second_exc.name != "cryosparc_tools":
            raise
        raise ModuleNotFoundError(
            "cryoFILTER CryoSPARC integration requires cryosparc-tools in this "
            "Python environment. Run "
            "`cryoFILTER install-cryosparc-tools --cryosparc-version X.Y` "
            "with your CryoSPARC server minor version, or install the "
            "noninteractive extra with "
            '`python -m pip install -e ".[cryosparc-bridge]"`.'
        ) from first_missing


def _read_connection_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser()
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"CryoSPARC connection file must contain a JSON object: {file_path}")
    return data


def connection_kwargs(connection_file: str | Path | None = None) -> dict[str, Any]:
    """Resolve CryoSPARC Tools connection arguments from file or environment."""

    env_file = (
        connection_file
        or os.environ.get("CRYOFILTER_CRYOSPARC_CONNECTION_FILE")
        or os.environ.get("CRYOSPARC_INSTANCE_INFO")
    )
    if env_file:
        return _read_connection_file(env_file)

    kwargs: dict[str, Any] = {}
    base_url = os.environ.get("CRYOSPARC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    else:
        kwargs["host"] = (
            os.environ.get("CRYOSPARC_HOST")
            or os.environ.get("CRYOSPARC_MASTER_HOSTNAME")
            or "localhost"
        )
        base_port = os.environ.get("CRYOSPARC_BASE_PORT")
        if base_port:
            kwargs["base_port"] = int(base_port)

    email = os.environ.get("CRYOSPARC_EMAIL")
    if email:
        kwargs["email"] = email
    password = os.environ.get("CRYOSPARC_PASSWORD")
    if password:
        kwargs["password"] = password
    license_id = os.environ.get("CRYOSPARC_LICENSE") or os.environ.get("CRYOSPARC_LICENSE_ID")
    if license_id:
        kwargs["license"] = license_id
    return kwargs


def make_client(connection_file: str | Path | None = None):
    """Create a CryoSPARC Tools client without importing GPU dependencies."""

    cryosparc_cls = import_cryosparc_class()
    return cryosparc_cls(**connection_kwargs(connection_file))


def test_connection(client: object) -> bool:
    test = getattr(client, "test_connection", None)
    if callable(test):
        return bool(test())
    return True


def server_version(client: object) -> str | None:
    """Best-effort CryoSPARC server version discovery."""

    for attr in ("get_version", "version"):
        candidate = getattr(client, attr, None)
        if callable(candidate):
            value = candidate()
            return None if value is None else str(value)
        if candidate:
            return str(candidate)

    cli = getattr(client, "cli", None)
    if cli is not None:
        for name in ("get_version", "get_server_version"):
            candidate = getattr(cli, name, None)
            if callable(candidate):
                value = candidate()
                return None if value is None else str(value)

    api = getattr(client, "api", None)
    if api is not None:
        system = getattr(api, "system", None)
        if system is not None:
            candidate = getattr(system, "version", None)
            if callable(candidate):
                value = candidate()
                return None if value is None else str(value)
    return None


__all__ = [
    "connection_kwargs",
    "cryosparc_tools_version",
    "import_cryosparc_class",
    "make_client",
    "server_version",
    "test_connection",
]
