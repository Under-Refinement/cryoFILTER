"""Configuration for engaging-side CryoSPARC remote integration."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CryoSPARCConnectionConfig:
    host: str = "worker"
    ssh_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class CryoSPARCApiConfig:
    base_url: str | None = None
    host: str | None = None
    base_port: int | None = None
    email: str | None = None


@dataclass(frozen=True)
class BridgeConfig:
    command: str = "~/venvs/cryofilter-bridge/bin/cryofilter-bridge"
    python: str = "~/venvs/cryofilter-bridge/bin/python"
    remote_work_root: str = "~/cryofilter_bridge_work"
    remote_source_root: str = "~/cryofilter_bridge_src"
    connection_file: str | None = None


@dataclass(frozen=True)
class TransferConfig:
    method: str = "rsync"
    batch_micrographs: int = 64
    rsync_options: tuple[str, ...] = ("-a", "--partial", "--protect-args")


@dataclass(frozen=True)
class LocalConfig:
    run_root: Path | None = None
    model_cache: Path | None = None


@dataclass(frozen=True)
class CryoSPARCIntegrationConfig:
    cryosparc: CryoSPARCConnectionConfig = field(default_factory=CryoSPARCConnectionConfig)
    api: CryoSPARCApiConfig = field(default_factory=CryoSPARCApiConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    config_path: Path | None = None

    @property
    def host(self) -> str:
        return self.cryosparc.host


def default_config_path() -> Path:
    env_path = os.environ.get("CRYOFILTER_CRYOSPARC_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return Path("~/.config/cryofilter/cryosparc.toml").expanduser()


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return text


def _parse_simple_toml(text: str) -> dict[str, dict[str, Any]]:
    """Fallback parser for the small static config used by this integration."""

    data: dict[str, dict[str, Any]] = {}
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data.setdefault(section, {})[key.strip()] = _parse_value(value)
    return data


def _load_toml(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:
        return _parse_simple_toml(text)
    return tomllib.loads(text)


def _section(data: dict[str, Any], dotted_name: str) -> dict[str, Any]:
    if dotted_name in data and isinstance(data[dotted_name], dict):
        return data[dotted_name]

    current: Any = data
    for part in dotted_name.split("."):
        if not isinstance(current, dict):
            return {}
        current = current.get(part)
    return current if isinstance(current, dict) else {}


def _optional_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _option_tuple(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError(f"Expected SSH options as a string or list, got {type(value).__name__}")


def load_config(path: str | Path | None = None) -> CryoSPARCIntegrationConfig:
    """Load ``~/.config/cryofilter/cryosparc.toml`` when present."""

    config_path = Path(path).expanduser() if path is not None else default_config_path()
    config = CryoSPARCIntegrationConfig(config_path=config_path if config_path.exists() else None)
    if not config_path.exists():
        return _apply_env_overrides(config)

    data = _load_toml(config_path)
    cryosparc_data = _section(data, "cryosparc")
    ssh_data = _section(data, "cryosparc.ssh")
    api_data = _section(data, "cryosparc.api")
    bridge_data = _section(data, "cryosparc.bridge")
    transfer_data = _section(data, "cryosparc.transfer")
    local_data = _section(data, "local")

    config = CryoSPARCIntegrationConfig(
        cryosparc=CryoSPARCConnectionConfig(
            host=str(cryosparc_data.get("host", config.cryosparc.host)),
            ssh_options=_option_tuple(ssh_data.get("options")),
        ),
        api=CryoSPARCApiConfig(
            base_url=_optional_str(api_data.get("base_url")),
            host=_optional_str(api_data.get("host")),
            base_port=_optional_int(api_data.get("base_port")),
            email=_optional_str(api_data.get("email") or api_data.get("username")),
        ),
        bridge=BridgeConfig(
            command=str(bridge_data.get("command", config.bridge.command)),
            python=str(bridge_data.get("python", config.bridge.python)),
            remote_work_root=str(
                bridge_data.get("remote_work_root", config.bridge.remote_work_root)
            ),
            remote_source_root=str(
                bridge_data.get("remote_source_root", config.bridge.remote_source_root)
            ),
            connection_file=(
                None
                if bridge_data.get("connection_file") in (None, "")
                else str(bridge_data.get("connection_file"))
            ),
        ),
        transfer=TransferConfig(
            method=str(transfer_data.get("method", config.transfer.method)),
            batch_micrographs=int(
                transfer_data.get("batch_micrographs", config.transfer.batch_micrographs)
            ),
            rsync_options=config.transfer.rsync_options,
        ),
        local=LocalConfig(
            run_root=_optional_path(local_data.get("run_root")),
            model_cache=_optional_path(local_data.get("model_cache")),
        ),
        config_path=config_path,
    )
    return _apply_env_overrides(config)


def _apply_env_overrides(config: CryoSPARCIntegrationConfig) -> CryoSPARCIntegrationConfig:
    host = os.environ.get("CRYOFILTER_CRYOSPARC_HOST")
    ssh_options = os.environ.get("CRYOFILTER_CRYOSPARC_SSH_OPTIONS")
    api_base_url = os.environ.get("CRYOFILTER_CRYOSPARC_BASE_URL")
    api_host = os.environ.get("CRYOFILTER_CRYOSPARC_API_HOST")
    api_base_port = os.environ.get("CRYOFILTER_CRYOSPARC_BASE_PORT")
    api_email = os.environ.get("CRYOFILTER_CRYOSPARC_EMAIL")
    command = os.environ.get("CRYOFILTER_CRYOSPARC_BRIDGE_COMMAND")
    remote_work_root = os.environ.get("CRYOFILTER_CRYOSPARC_REMOTE_WORK_ROOT")
    remote_source_root = os.environ.get("CRYOFILTER_CRYOSPARC_REMOTE_SOURCE_ROOT")
    connection_file = os.environ.get("CRYOFILTER_CRYOSPARC_CONNECTION_FILE")

    if host:
        config = replace(config, cryosparc=replace(config.cryosparc, host=host))
    if ssh_options:
        config = replace(
            config,
            cryosparc=replace(config.cryosparc, ssh_options=_option_tuple(ssh_options)),
        )
    if api_base_url or api_host or api_base_port or api_email:
        base_url = api_base_url or config.api.base_url
        host_value = api_host or config.api.host
        base_port_value = (
            int(api_base_port)
            if api_base_port
            else config.api.base_port
        )
        if api_base_url:
            host_value = None
            base_port_value = None
        config = replace(
            config,
            api=replace(
                config.api,
                base_url=base_url,
                host=host_value,
                base_port=base_port_value,
                email=api_email or config.api.email,
            ),
        )
    if command or remote_work_root or remote_source_root or connection_file:
        config = replace(
            config,
            bridge=replace(
                config.bridge,
                command=command or config.bridge.command,
                remote_work_root=remote_work_root or config.bridge.remote_work_root,
                remote_source_root=remote_source_root or config.bridge.remote_source_root,
                connection_file=connection_file or config.bridge.connection_file,
            ),
        )
    return config


__all__ = [
    "BridgeConfig",
    "CryoSPARCApiConfig",
    "CryoSPARCConnectionConfig",
    "CryoSPARCIntegrationConfig",
    "LocalConfig",
    "TransferConfig",
    "default_config_path",
    "load_config",
]
