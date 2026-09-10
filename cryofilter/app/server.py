"""Small dependency-light local web app for cryoFILTER workflows."""

from __future__ import annotations

import argparse
import csv
import errno
import http.client
import json
import mimetypes
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse

from cryofilter.app import browser_annotation
from cryofilter.app.annotation_manifest import (
    collect_micrographs,
    rows_from_micrographs,
    rows_from_transfer_manifest,
    write_manifest,
)


APP_DIRNAME = ".cryofilter_app"
MAX_LOG_BYTES_DEFAULT = 256_000
APP_PORT_SEARCH_LIMIT = 50
LOCALHOST_NAMES = {"127.0.0.1", "localhost", "::1"}
ARTIFACT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".json",
    ".csv",
    ".cs",
    ".csg",
    ".star",
    ".txt",
    ".log",
}
DEFAULT_PUBLIC_CHECKPOINT_RELATIVE = Path("pretrained_models") / "cryoFILTER_FULL.pt"
CONTAMINATION_TYPE_LABELS = ("Carbon", "Crystalline", "Aggregate", "Ethane")
CONTAMINATION_TYPE_COLORS = {
    "Carbon": "#226F54",
    "Support": "#226F54",
    "Crystalline": "#003D5B",
    "Aggregate": "#DE541E",
    "Ethane": "#96BBBB",
}


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    app = subparsers.add_parser(
        "app",
        aliases=["studio"],
        help="Launch the local cryoFILTER web wrapper.",
    )
    app.add_argument("--host", default="127.0.0.1", help="Bind address. Default: localhost only.")
    app.add_argument("--port", type=int, default=8765, help="Port for the local web app.")
    app.add_argument(
        "--reclaim-port",
        dest="reclaim_port",
        action="store_true",
        default=True,
        help="Stop an existing cryoFILTER app on the requested localhost port before binding. Default: on.",
    )
    app.add_argument(
        "--no-reclaim-port",
        dest="reclaim_port",
        action="store_false",
        help="Do not stop an existing cryoFILTER app before binding; use the next available port instead.",
    )
    app.add_argument(
        "--work-dir",
        default=".",
        help="Directory where app state and relative workflow outputs are written.",
    )
    app.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    state = AppState(Path(args.work_dir).expanduser().resolve())
    host = str(args.host)
    requested_port = int(args.port)
    reclaimed_pids = (
        _reclaim_cryofilter_app_port(host=host, port=requested_port)
        if bool(getattr(args, "reclaim_port", True))
        else []
    )
    server, bound_port = _bind_app_server(
        host=host,
        port=requested_port,
        handler_cls=make_handler(state),
    )
    if reclaimed_pids:
        pids = ", ".join(str(pid) for pid in reclaimed_pids)
        print(f"Stopped existing cryoFILTER app on port {requested_port} (pid(s): {pids}).", flush=True)
    if requested_port != 0 and bound_port != requested_port:
        print(f"Port {requested_port} is in use; using {bound_port} instead.", flush=True)
    for line in _app_startup_lines(
        host=host,
        requested_port=requested_port,
        bound_port=bound_port,
        state_dir=state.data_dir,
    ):
        print(line, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping cryoFILTER app.", flush=True)
    finally:
        server.server_close()
    return 0


def _reclaim_cryofilter_app_port(
    *,
    host: str,
    port: int,
    timeout_s: float = 3.0,
) -> list[int]:
    if port <= 0 or host not in LOCALHOST_NAMES:
        return []
    pids = _listening_pids_for_port(port)
    if not pids:
        return []
    owned_pids = [pid for pid in pids if pid != os.getpid() and _process_owned_by_current_user(pid)]
    if not owned_pids:
        return []
    is_cryofilter = _is_cryofilter_app_port(host, port) or any(
        _process_looks_like_cryofilter_app(pid) for pid in owned_pids
    )
    if not is_cryofilter:
        return []
    _terminate_pids(owned_pids, timeout_s=timeout_s)
    remaining = set(_listening_pids_for_port(port))
    return [pid for pid in owned_pids if pid not in remaining]


def _is_cryofilter_app_port(host: str, port: int, *, timeout_s: float = 0.35) -> bool:
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        server_header = response.getheader("Server", "")
        response.read(512)
        return "cryoFILTERApp/" in server_header
    except Exception:
        return False
    finally:
        if connection is not None:
            connection.close()


def _listening_pids_for_port(port: int) -> list[int]:
    inodes = _listening_socket_inodes_for_port(port)
    if not inodes:
        return []
    pids: set[int] = set()
    proc = Path("/proc")
    if not proc.exists():
        return []
    for fd_dir in proc.glob("[0-9]*/fd"):
        try:
            pid = int(fd_dir.parent.name)
        except ValueError:
            continue
        try:
            fd_paths = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd_path in fd_paths:
            try:
                target = os.readlink(fd_path)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inode = target.removeprefix("socket:[").removesuffix("]")
                if inode in inodes:
                    pids.add(pid)
                    break
    return sorted(pids)


def _listening_socket_inodes_for_port(port: int) -> set[str]:
    inodes: set[str] = set()
    target_port = f"{int(port):04X}"
    for proc_file in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = proc_file.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            local_address = fields[1]
            _, _, local_port = local_address.rpartition(":")
            if local_port.upper() == target_port:
                inodes.add(fields[9])
    return inodes


def _process_owned_by_current_user(pid: int) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").stat().st_uid == os.getuid()
    except OSError:
        return False


def _process_looks_like_cryofilter_app(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return False
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").lower()
    return (
        "cryofilter" in command
        and (" app" in command or " studio" in command)
    )


def _terminate_pids(pids: Sequence[int], *, timeout_s: float) -> None:
    unique_pids = sorted({int(pid) for pid in pids if int(pid) > 0 and int(pid) != os.getpid()})
    for pid in unique_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while time.monotonic() < deadline:
        if all(not _pid_exists(pid) for pid in unique_pids):
            return
        time.sleep(0.05)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _app_startup_lines(
    *,
    host: str,
    requested_port: int,
    bound_port: int,
    state_dir: Path,
    remote_name: str | None = None,
) -> list[str]:
    lines = [f"cryoFILTER app listening at http://{host}:{bound_port}/"]
    if host in LOCALHOST_NAMES:
        lines.append(f"Laptop browser URL with SSH forwarding: http://127.0.0.1:{bound_port}/")
        if requested_port != 0 and bound_port != requested_port:
            lines.append(
                f"Note: an existing SSH tunnel for {requested_port} still opens whatever is "
                f"running on the server's {requested_port}; forward {bound_port} before "
                f"opening the {bound_port} URL on your laptop."
            )
        ssh_target = remote_name or socket.gethostname() or "their-server"
        lines.append(f"SSH example: ssh -L {bound_port}:127.0.0.1:{bound_port} user@{ssh_target}")
    lines.append(f"State directory: {state_dir}")
    return lines


def _bind_app_server(
    *,
    host: str,
    port: int,
    handler_cls: type[BaseHTTPRequestHandler],
    search_limit: int = APP_PORT_SEARCH_LIMIT,
) -> tuple["CryoFilterHTTPServer", int]:
    if port < 0 or port > 65535:
        raise ValueError("--port must be between 0 and 65535")
    if port == 0:
        server = CryoFilterHTTPServer((host, port), handler_cls)
        return server, int(server.server_address[1])

    last_port = min(65535, port + max(0, int(search_limit)))
    last_error: OSError | None = None
    for candidate in range(port, last_port + 1):
        try:
            server = CryoFilterHTTPServer((host, candidate), handler_cls)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_error = exc
            continue
        return server, candidate
    raise OSError(
        errno.EADDRINUSE,
        f"No available port found from {port} through {last_port}",
    ) from last_error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def runtime_status() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "node": socket.gethostname(),
        "gpu": _gpu_status(),
    }


def _gpu_status() -> str:
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL"):
        value = os.environ.get(name)
        if value:
            return f"{name}={value}"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return "not detected"
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and names:
        first = names[0]
        suffix = f" +{len(names) - 1}" if len(names) > 1 else ""
        return f"{first}{suffix}"
    return "not detected"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    text = _optional_str(value)
    return int(text) if text is not None else None


def _optional_float(value: object, default: float) -> float:
    text = _optional_str(value)
    return float(text) if text is not None else float(default)


def _instance_url_host_port(value: object) -> tuple[str | None, int | None]:
    text = _optional_str(value)
    if text is None:
        return None, None
    candidate = text if "://" in text else f"http://{text}"
    parsed = urlparse(candidate)
    if not parsed.hostname:
        return None, None
    try:
        port = parsed.port
    except ValueError:
        port = None
    return parsed.hostname, port


def _normalize_output_ref(value: str, *, default_output_name: str) -> str:
    text = str(value).strip()
    if ":" in text:
        return text
    if text.startswith("J") and text[1:].isdigit():
        return f"{text}:{default_output_name}"
    return text


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return shlex.split(text)


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = _optional_str(payload.get(key))
    if not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def _add_option(argv: list[str], flag: str, value: object | None) -> None:
    text = _optional_str(value)
    if text is not None:
        argv.extend([flag, text])


def _redact_text(text: str, secrets: Sequence[object | None]) -> str:
    redacted = str(text)
    for secret in secrets:
        value = _optional_str(secret)
        if value:
            redacted = redacted.replace(value, "[hidden]")
    return redacted


def _make_cryosparc_probe_client(
    *,
    host: str,
    base_port: int | None,
    email: str | None,
    password: str | None,
) -> object:
    from cryofilter.cryosparc.bridge.client import import_cryosparc_class

    kwargs: dict[str, Any] = {"host": host}
    if base_port is not None:
        kwargs["base_port"] = int(base_port)
    if email is not None:
        kwargs["email"] = email
    if password is not None:
        kwargs["password"] = password
    return import_cryosparc_class()(**kwargs)


def validate_cryosparc_connection(
    payload: dict[str, Any],
    *,
    make_client: Callable[..., object] | None = None,
    test_connection: Callable[[object], bool] | None = None,
    server_version: Callable[[object], str | None] | None = None,
) -> dict[str, Any]:
    instance_url = _require_text(payload, "cryosparc_base_url")
    host, base_port = _instance_url_host_port(instance_url)
    if host is None:
        raise ValueError("Enter a valid CryoSPARC instance URL.")

    email = _optional_str(payload.get("cryosparc_email"))
    password = _optional_str(payload.get("cryosparc_password"))
    try:
        if make_client is None:
            make_client = _make_cryosparc_probe_client
        if test_connection is None or server_version is None:
            from cryofilter.cryosparc.bridge import client as bridge_client

            if test_connection is None:
                test_connection = bridge_client.test_connection
            if server_version is None:
                server_version = bridge_client.server_version
        client = make_client(
            host=host,
            base_port=base_port,
            email=email,
            password=password,
        )
        if not test_connection(client):
            raise RuntimeError("CryoSPARC did not accept the connection test.")
        version = server_version(client)
    except ModuleNotFoundError as exc:
        if str(exc).find("cryosparc") >= 0:
            raise RuntimeError(
                "cryosparc-tools is not installed in this Python environment. "
                "Install with `pip install cryosparc-tools` or "
                '`pip install -e ".[cryosparc-bridge]"`.'
            ) from None
        raise
    except Exception as exc:
        detail = _redact_text(f"{type(exc).__name__}: {exc}", [password])
        raise RuntimeError(f"CryoSPARC connection failed: {detail}") from None

    display = host if base_port is None else f"{host}:{base_port}"
    return {
        "ok": True,
        "connected": True,
        "host": host,
        "base_port": base_port,
        "display": display,
        "email": email,
        "server_version": version,
    }


def _metadata_value(value: object | None) -> object | None:
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_work_path(value: object, *, work_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (work_dir / path).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_checkpoint_path() -> Path:
    return _repo_root() / DEFAULT_PUBLIC_CHECKPOINT_RELATIVE


def _resolve_checkpoint_option(value: object) -> str:
    text = _optional_str(value)
    if text is not None:
        return text
    default_path = _default_checkpoint_path()
    if default_path.is_file():
        return str(default_path)
    raise FileNotFoundError(
        "Weights not found: pretrained_models/cryoFILTER_FULL.pt. "
        f"Place the default checkpoint at {default_path} or specify a weights path."
    )


def _annotation_session_name(payload: dict[str, Any]) -> str:
    return _optional_str(payload.get("output_name")) or (
        "annotation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )


def _add_editor_common_options(argv: list[str], payload: dict[str, Any]) -> None:
    _add_option(argv, "--initial_split", payload.get("initial_split"))
    _add_option(argv, "--initial_dataset_filter", payload.get("initial_dataset_filter"))
    _add_option(argv, "--initial_row_key", payload.get("initial_row_key"))
    _add_option(argv, "--val_manifest", payload.get("val_manifest"))
    _add_option(argv, "--split_seed", payload.get("split_seed"))
    _add_option(argv, "--split_train_frac", payload.get("train_fraction"))
    _add_option(argv, "--prefetch_workers", payload.get("prefetch_workers") or "2")
    _add_option(argv, "--prefetch_radius", payload.get("prefetch_radius") or "3")
    _add_option(argv, "--row_cache_size", payload.get("row_cache_size") or "18")
    if _as_bool(payload.get("export_only_modified"), default=False):
        argv.append("--export_only_modified")


def _annotation_editor_step(
    *,
    manifest: Path,
    payload: dict[str, Any],
    output_root: Path,
    output_name: str | None = None,
    output_dir: Path | None = None,
    mic_dir: Path | str | None = None,
) -> JobStep:
    script = Path(__file__).resolve().parents[2] / "scripts" / "gt_mask_editor_modern.py"
    argv = [sys.executable, str(script), "--manifest", str(manifest)]
    if mic_dir is not None:
        argv.extend(["--mic_dir", str(mic_dir)])
    if output_dir is not None:
        argv.extend(["--output_dir", str(output_dir)])
    else:
        argv.extend(["--output_root", str(output_root)])
        if output_name:
            argv.extend(["--output_name", output_name])
    _add_editor_common_options(argv, payload)
    return JobStep("Launch annotation studio", argv)


def _manifest_path_value(
    value: object,
    *,
    manifest_dir: Path,
    work_dir: Path,
) -> Path | None:
    text = _optional_str(value)
    if text is None or text.lower() in {"nan", "none"}:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    for candidate in (manifest_dir / path, work_dir / path, path):
        if candidate.exists():
            return candidate.resolve()
    return None


def _infer_manifest_micrograph_root(manifest: Path, *, work_dir: Path) -> Path | None:
    if not manifest.exists():
        return None
    manifest_dir = manifest.resolve().parent
    path_columns = (
        "micrograph_path",
        "full_mic_path",
        "resolved_mic_path",
        "staged_mic_path",
        "mic_path",
    )
    paths: list[Path] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in path_columns:
                path = _manifest_path_value(
                    row.get(column),
                    manifest_dir=manifest_dir,
                    work_dir=work_dir,
                )
                if path is not None:
                    paths.append(path)
                    break
    if not paths:
        return None
    parents = [path if path.is_dir() else path.parent for path in paths]
    try:
        return Path(os.path.commonpath([str(parent) for parent in parents]))
    except ValueError:
        return None


def _manifest_split_counts(manifest: Path) -> dict[str, int]:
    if not manifest.exists():
        return {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        split_column = next(
            (column for column in ("split", "split_label") if column in reader.fieldnames),
            None,
        )
        if split_column is None:
            return {}
        counts = {"TRAINING": 0, "VALIDATION": 0}
        for row in reader:
            text = _optional_str(row.get(split_column))
            label = "" if text is None else text.upper()
            if label in {"TRAIN", "TRAINING"}:
                counts["TRAINING"] += 1
            elif label in {"VAL", "VALID", "VALIDATION"}:
                counts["VALIDATION"] += 1
        return counts if counts["TRAINING"] and counts["VALIDATION"] else {}


@dataclass(frozen=True)
class JobStep:
    name: str
    argv: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv": list(self.argv)}


@dataclass(frozen=True)
class JobSpec:
    kind: str
    title: str
    steps: list[JobStep]
    artifact_roots: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "steps": [step.as_dict() for step in self.steps],
            "artifact_roots": [str(path) for path in self.artifact_roots],
            "metadata": {key: _metadata_value(value) for key, value in self.metadata.items()},
        }


def build_job_spec(kind: str, payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    if kind == "infer":
        return _build_infer_spec(payload, work_dir=work_dir)
    if kind == "type":
        return _build_type_spec(payload, work_dir=work_dir)
    if kind == "filter_particles":
        return _build_filter_particles_spec(payload, work_dir=work_dir)
    if kind == "cryosparc_predict":
        return _build_cryosparc_predict_spec(payload, work_dir=work_dir)
    if kind == "annotation":
        return _build_annotation_spec(payload, work_dir=work_dir)
    if kind == "train":
        return _build_train_spec(payload, work_dir=work_dir)
    raise ValueError(f"Unknown job kind: {kind}")


def _build_infer_spec(payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    input_path = _require_text(payload, "input")
    output_dir = Path(
        _optional_str(payload.get("output_dir")) or str(work_dir / "cryofilter_output")
    ).expanduser()
    export_masks = _as_bool(payload.get("export_masks"), default=True)
    binned_masks = _as_bool(payload.get("binned_masks"), default=False)
    checkpoint = _resolve_checkpoint_option(payload.get("checkpoint"))
    argv = [
        sys.executable,
        "-m",
        "cryofilter.cli",
        "infer",
        "--input",
        input_path,
        "--output-dir",
        str(output_dir),
    ]
    _add_option(argv, "--checkpoint", checkpoint)
    _add_option(argv, "--device", payload.get("device"))
    _add_option(argv, "--num-cpus", payload.get("num_cpus"))
    _add_option(argv, "--num-gpus", payload.get("num_gpus"))
    _add_option(argv, "--threshold", payload.get("threshold"))
    _add_option(argv, "--particle-file", payload.get("particle_file"))
    _add_option(argv, "--particle-csg", payload.get("particle_csg"))
    _add_option(argv, "--pixel-size-angstrom", payload.get("pixel_size_angstrom"))
    _add_option(argv, "--inference-profile", payload.get("inference_profile"))
    if _as_bool(payload.get("recursive"), default=True):
        argv.append("--recursive")
    if _as_bool(payload.get("skip_existing"), default=False):
        argv.append("--skip-existing")
    if _as_bool(payload.get("render_overlays"), default=True):
        argv.append("--render-particle-overlays")
    else:
        argv.append("--no-render-particle-overlays")
    if not export_masks:
        argv.append("--no-export-masks")
    if binned_masks:
        argv.append("--no-resample")
    argv.extend(_as_tokens(payload.get("extra_args")))
    return JobSpec(
        kind="infer",
        title=_optional_str(payload.get("title")) or f"Inference: {Path(input_path).name}",
        steps=[JobStep("cryoFILTER inference", argv)],
        artifact_roots=[output_dir],
        metadata={
            "input": input_path,
            "checkpoint": checkpoint,
            "output_dir": output_dir,
            "num_cpus": payload.get("num_cpus"),
            "num_gpus": payload.get("num_gpus"),
            "export_masks": export_masks,
            "binned_masks": binned_masks,
        },
    )


def _build_type_spec(payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    manifest = _require_text(payload, "manifest")
    output_dir = Path(
        _optional_str(payload.get("output_dir")) or str(work_dir / "contamination_typing")
    ).expanduser()
    argv = [
        sys.executable,
        "-m",
        "cryofilter.cli",
        "type",
        "--manifest",
        manifest,
        "--output-dir",
        str(output_dir),
    ]
    _add_option(argv, "--normalization-method", payload.get("normalization_method"))
    _add_option(argv, "--min-component-area-px", payload.get("min_component_area_px"))
    _add_option(argv, "--pixel-size-angstrom", payload.get("pixel_size_angstrom"))
    argv.extend(_as_tokens(payload.get("extra_args")))
    return JobSpec(
        kind="type",
        title=_optional_str(payload.get("title")) or "Contamination typing",
        steps=[JobStep("cryoFILTER typing", argv)],
        artifact_roots=[output_dir],
        metadata={"manifest": manifest, "output_dir": output_dir},
    )


def _build_filter_particles_spec(payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    input_path = _require_text(payload, "input")
    mask_dir = Path(_require_text(payload, "mask_dir")).expanduser()
    output_dir = Path(
        _optional_str(payload.get("output_dir")) or str(mask_dir)
    ).expanduser()
    particle_file = _require_text(payload, "particle_file")
    argv = [
        sys.executable,
        "-m",
        "cryofilter.cli",
        "filter-particles",
        "--input",
        input_path,
        "--mask-dir",
        str(mask_dir),
        "--particle-file",
        particle_file,
    ]
    _add_option(argv, "--output-dir", payload.get("output_dir"))
    _add_option(argv, "--filtered-particle-file", payload.get("filtered_particle_file"))
    _add_option(argv, "--particle-csg", payload.get("particle_csg"))
    _add_option(
        argv,
        "--particle-exclusion-distance-angstrom",
        payload.get("particle_exclusion_distance_angstrom"),
    )
    _add_option(argv, "--pixel-size-angstrom", payload.get("pixel_size_angstrom"))
    _add_option(argv, "--summary-file", payload.get("summary_file"))
    if _as_bool(payload.get("recursive"), default=True):
        argv.append("--recursive")
    if _as_bool(payload.get("allow_unmatched_particles"), default=False):
        argv.append("--allow-unmatched-particles")
    if _as_bool(payload.get("overwrite_filtered_particles"), default=False):
        argv.append("--overwrite-filtered-particles")
    return JobSpec(
        kind="filter_particles",
        title=_optional_str(payload.get("title")) or f"Filter picks: {Path(particle_file).name}",
        steps=[JobStep("Filter existing picks", argv)],
        artifact_roots=[output_dir],
        metadata={
            "input": input_path,
            "mask_dir": mask_dir,
            "output_dir": output_dir,
            "particle_file": particle_file,
        },
    )


def _build_cryosparc_predict_spec(payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    micrographs_ref = _normalize_output_ref(
        _require_text(payload, "micrographs"),
        default_output_name="micrographs",
    )
    particles_ref = _normalize_output_ref(
        _require_text(payload, "particles"),
        default_output_name="particles",
    )
    instance_host, instance_port = _instance_url_host_port(
        payload.get("cryosparc_base_url") or payload.get("cryosparc_url")
    )
    run_root = Path(
        _optional_str(payload.get("local_run_root")) or str(work_dir / "cryofilter_runs" / "cryosparc")
    ).expanduser()
    run_id = _optional_str(payload.get("run_id")) or str(uuid.uuid4())
    run_dir = run_root / run_id
    checkpoint = _resolve_checkpoint_option(payload.get("checkpoint"))
    argv = [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
    _add_option(argv, "--config", payload.get("config"))
    _add_option(argv, "--host", payload.get("bridge_host"))
    _add_option(argv, "--bridge-command", payload.get("bridge_command"))
    _add_option(argv, "--remote-source-root", payload.get("remote_source_root"))
    _add_option(argv, "--remote-work-root", payload.get("remote_work_root"))
    _add_option(argv, "--cryosparc-host", payload.get("cryosparc_host") or instance_host)
    _add_option(argv, "--cryosparc-base-port", payload.get("cryosparc_base_port") or instance_port)
    _add_option(argv, "--cryosparc-email", payload.get("cryosparc_email"))
    for option in _as_tokens(payload.get("ssh_options")):
        argv.extend(["--ssh-option", option])
    argv.extend(
        [
            "predict",
            "--project",
            _require_text(payload, "project"),
            "--workspace",
            _require_text(payload, "workspace"),
            "--micrographs",
            micrographs_ref,
            "--particles",
            particles_ref,
            "--run-id",
            run_id,
        ]
    )
    _add_option(argv, "--checkpoint", checkpoint)
    _add_option(argv, "--threshold", payload.get("threshold"))
    _add_option(argv, "--title", payload.get("title"))
    _add_option(argv, "--limit-micrographs", payload.get("limit_micrographs"))
    _add_option(argv, "--local-run-root", str(run_root))
    _add_option(argv, "--num-cpus", payload.get("num_cpus"))
    _add_option(argv, "--num-gpus", payload.get("num_gpus"))
    _add_option(argv, "--max-transfer-gb", payload.get("max_transfer_gb"))
    _add_option(argv, "--typing-summary", payload.get("typing_summary"))
    if not _as_bool(payload.get("run_typing"), default=True):
        argv.append("--no-run-typing")

    infer_args = _as_tokens(payload.get("infer_args"))
    device = _optional_str(payload.get("device"))
    profile = _optional_str(payload.get("inference_profile"))
    batch_forward_size = _optional_str(payload.get("batch_forward_size"))
    export_masks = _as_bool(payload.get("export_masks"), default=True)
    binned_masks = _as_bool(payload.get("binned_masks"), default=False)
    if device or profile or batch_forward_size or infer_args or not export_masks or binned_masks:
        argv.append("--")
        if device:
            argv.extend(["--device", device])
        if profile:
            argv.extend(["--inference-profile", profile])
        if batch_forward_size:
            argv.extend(["--batch-forward-size", batch_forward_size])
        if not export_masks:
            argv.append("--no-export-masks")
        if binned_masks:
            argv.append("--no-resample")
        argv.extend(infer_args)

    secret_env: dict[str, str] = {}
    cryosparc_password = _optional_str(payload.get("cryosparc_password"))
    if cryosparc_password is not None:
        secret_env["CRYOSPARC_PASSWORD"] = cryosparc_password
    return JobSpec(
        kind="cryosparc_predict",
        title=_optional_str(payload.get("title")) or "CryoSPARC prediction",
        steps=[JobStep("CryoSPARC prediction", argv)],
        artifact_roots=[run_dir],
        metadata={
            "project": payload.get("project"),
            "workspace": payload.get("workspace"),
            "micrographs": micrographs_ref,
            "particles": particles_ref,
            "checkpoint": checkpoint,
            "run_id": run_id,
            "local_run_root": run_root,
            "local_run_dir": run_dir,
            "run_typing": _as_bool(payload.get("run_typing"), default=True),
            "num_cpus": payload.get("num_cpus"),
            "num_gpus": payload.get("num_gpus"),
            "export_masks": export_masks,
            "binned_masks": binned_masks,
        },
        env=secret_env,
    )


def _build_annotation_spec(payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    source_mode = (_optional_str(payload.get("source_mode")) or "micrographs").lower()
    output_root = _resolve_work_path(
        _optional_str(payload.get("output_root")) or "annotation_exports",
        work_dir=work_dir,
    )
    dataset_id = _optional_str(payload.get("dataset_id")) or "1"
    steps: list[JobStep] = []
    artifact_roots = [output_root]
    metadata: dict[str, Any] = {"source_mode": source_mode, "output_root": output_root}
    secret_env: dict[str, str] = {}

    if source_mode == "manifest":
        manifest = _resolve_work_path(_require_text(payload, "manifest"), work_dir=work_dir)
        output_name = _optional_str(payload.get("output_name"))
        resume_in_place = _as_bool(payload.get("resume_in_place"), default=True)
        if resume_in_place and output_name is None:
            output_dir = manifest.parent
            artifact_roots = [output_dir]
            steps.append(
                _annotation_editor_step(
                    manifest=manifest,
                    payload=payload,
                    output_root=output_root,
                    output_dir=output_dir,
                    mic_dir=payload.get("mic_dir"),
                )
            )
        else:
            steps.append(
                _annotation_editor_step(
                    manifest=manifest,
                    payload=payload,
                    output_root=output_root,
                    output_name=output_name,
                    mic_dir=payload.get("mic_dir"),
                )
            )
        metadata["manifest"] = manifest
        return JobSpec(
            kind="annotation",
            title=_optional_str(payload.get("title")) or "Resume annotation studio",
            steps=steps,
            artifact_roots=artifact_roots,
            metadata=metadata,
        )

    session_name = _annotation_session_name(payload)
    session_dir = output_root / session_name
    source_manifest = session_dir / "source_manifest.csv"
    metadata.update({"manifest": source_manifest, "session_dir": session_dir})

    if source_mode == "micrographs":
        micrograph_source = _require_text(payload, "micrograph_source")
        build_argv = [
            sys.executable,
            "-m",
            "cryofilter.app.annotation_manifest",
            "--micrographs",
            micrograph_source,
            "--output-manifest",
            str(source_manifest),
            "--dataset-id",
            dataset_id,
            "--reuse-existing",
        ]
        _add_option(build_argv, "--pixel-size-angstrom", payload.get("pixel_size_angstrom"))
        _add_option(build_argv, "--limit", payload.get("limit_micrographs"))
        _add_option(build_argv, "--train-fraction", payload.get("train_fraction"))
        _add_option(build_argv, "--seed", payload.get("split_seed"))
        if not _as_bool(payload.get("recursive"), default=True):
            build_argv.append("--no-recursive")
        steps.append(JobStep("Build annotation manifest", build_argv))
        steps.append(
            _annotation_editor_step(
                manifest=source_manifest,
                payload=payload,
                output_root=output_root,
                output_name=session_name,
            )
        )
        metadata["micrograph_source"] = micrograph_source
        return JobSpec(
            kind="annotation",
            title=_optional_str(payload.get("title")) or f"Annotation: {Path(micrograph_source).name}",
            steps=steps,
            artifact_roots=artifact_roots,
            metadata=metadata,
        )

    if source_mode != "cryosparc":
        raise ValueError("Annotation source must be micrographs, cryosparc, or manifest.")

    if not _optional_str(payload.get("cryosparc_base_url")):
        raise ValueError("Connect to CryoSPARC before launching annotation from a workspace output.")
    micrographs_ref = _normalize_output_ref(
        _require_text(payload, "cryosparc_micrographs"),
        default_output_name="micrographs",
    )
    instance_host, instance_port = _instance_url_host_port(payload.get("cryosparc_base_url"))
    run_id = str(uuid.uuid4())
    stage_root = _resolve_work_path(
        _optional_str(payload.get("local_run_root")) or "cryofilter_runs/annotation_staging",
        work_dir=work_dir,
    )
    stage_run_dir = stage_root / run_id
    transfer_manifest = stage_run_dir / "transfer_manifest.json"
    transfer_mic_dir = stage_run_dir / "transfer" / "micrographs"
    copy_cryosparc_micrographs = _as_bool(
        payload.get("copy_cryosparc_micrographs"),
        default=False,
    )
    stage_argv = [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
    _add_option(stage_argv, "--config", payload.get("config"))
    _add_option(stage_argv, "--host", payload.get("bridge_host") or "local")
    _add_option(stage_argv, "--bridge-command", payload.get("bridge_command"))
    _add_option(stage_argv, "--remote-source-root", payload.get("remote_source_root"))
    _add_option(stage_argv, "--remote-work-root", payload.get("remote_work_root"))
    _add_option(stage_argv, "--cryosparc-host", payload.get("cryosparc_host") or instance_host)
    _add_option(stage_argv, "--cryosparc-base-port", payload.get("cryosparc_base_port") or instance_port)
    _add_option(stage_argv, "--cryosparc-email", payload.get("cryosparc_email"))
    for option in _as_tokens(payload.get("ssh_options")):
        stage_argv.extend(["--ssh-option", option])
    stage_argv.extend(
        [
            "stage-test",
            "--project",
            _require_text(payload, "cryosparc_project"),
            "--workspace",
            _require_text(payload, "cryosparc_workspace"),
            "--micrographs",
            micrographs_ref,
            "--run-id",
            run_id,
            "--local-run-root",
            str(stage_root),
            "--model-id",
            "annotation",
            "--title",
            "cryoFILTER annotation source",
        ]
    )
    if _optional_str(payload.get("limit_micrographs")):
        _add_option(stage_argv, "--limit-micrographs", payload.get("limit_micrographs"))
    else:
        stage_argv.append("--all-micrographs")
    _add_option(stage_argv, "--max-transfer-gb", payload.get("max_transfer_gb") or "25")
    if not copy_cryosparc_micrographs:
        stage_argv.extend(["--no-pull", "--no-stage-micrographs"])
    steps.append(JobStep("Stage CryoSPARC micrographs", stage_argv))

    build_argv = [
        sys.executable,
        "-m",
        "cryofilter.app.annotation_manifest",
        "--transfer-manifest",
        str(transfer_manifest),
        "--output-manifest",
        str(source_manifest),
        "--dataset-id",
        dataset_id,
        "--reuse-existing",
    ]
    if not copy_cryosparc_micrographs:
        build_argv.append("--prefer-source-paths")
    _add_option(build_argv, "--limit", payload.get("limit_micrographs"))
    _add_option(build_argv, "--train-fraction", payload.get("train_fraction"))
    _add_option(build_argv, "--seed", payload.get("split_seed"))
    steps.append(JobStep("Build annotation manifest", build_argv))
    steps.append(
        _annotation_editor_step(
            manifest=source_manifest,
            payload=payload,
            output_root=output_root,
            output_name=session_name,
            mic_dir=transfer_mic_dir if copy_cryosparc_micrographs else None,
        )
    )
    cryosparc_password = _optional_str(payload.get("cryosparc_password"))
    if cryosparc_password is not None:
        secret_env["CRYOSPARC_PASSWORD"] = cryosparc_password
    metadata.update(
        {
            "micrographs": micrographs_ref,
            "stage_run_dir": stage_run_dir,
            "transfer_manifest": transfer_manifest,
            "copy_cryosparc_micrographs": copy_cryosparc_micrographs,
        }
    )
    return JobSpec(
        kind="annotation",
        title=_optional_str(payload.get("title")) or "Annotation from CryoSPARC",
        steps=steps,
        artifact_roots=artifact_roots + [stage_run_dir],
        metadata=metadata,
        env=secret_env,
    )


def _build_train_spec(payload: dict[str, Any], *, work_dir: Path) -> JobSpec:
    manifest = _require_text(payload, "manifest")
    manifest_path = _resolve_work_path(manifest, work_dir=work_dir)
    output_dir = Path(
        _optional_str(payload.get("output_dir")) or str(work_dir / "finetune_cryofilter")
    ).expanduser()
    mic_dir_text = _optional_str(payload.get("mic_dir"))
    mic_source_mode = _optional_str(payload.get("mic_source_mode"))
    if mic_source_mode is None:
        mic_source_mode = "local" if mic_dir_text is not None else "manifest"
    mic_source_mode = mic_source_mode.lower()
    if mic_source_mode in {"auto", "manifest_paths"}:
        mic_source_mode = "manifest"
    if mic_source_mode not in {"manifest", "local", "cryosparc"}:
        raise ValueError("Fine-tune Micrograph source must be Auto from manifest, Local directory, or CryoSPARC output.")

    train_ids = _optional_str(payload.get("train_dataset_ids"))
    val_ids = _optional_str(payload.get("val_dataset_ids"))
    if bool(train_ids) != bool(val_ids):
        raise ValueError("Provide both Train IDs and Val IDs, or leave both blank for an automatic 70/30 row split.")

    steps: list[JobStep] = []
    artifact_roots = [output_dir]
    secret_env: dict[str, str] = {}
    training_manifest = output_dir / "cryofilter_training_manifest.csv"
    prepare_manifest_argv = [
        sys.executable,
        "-m",
        "cryofilter.app.training_manifest",
        "--manifest",
        str(manifest_path),
        "--output-manifest",
        str(training_manifest),
    ]
    if train_ids is None:
        prepare_manifest_argv.extend(
            [
                "--train-fraction",
                str(payload.get("train_fraction") or "0.7"),
                "--seed",
                str(payload.get("split_seed") or "42"),
            ]
        )
    else:
        prepare_manifest_argv.append("--preserve-split")

    if mic_source_mode == "local":
        mic_dir = str(_resolve_work_path(_require_text(payload, "mic_dir"), work_dir=work_dir))
    elif mic_source_mode == "manifest":
        inferred_mic_dir = _infer_manifest_micrograph_root(manifest_path, work_dir=work_dir)
        if inferred_mic_dir is None:
            raise ValueError(
                "Could not infer Micrographs from the edited manifest. Choose Local directory "
                "or CryoSPARC output in the Fine-tune Micrograph source."
            )
        mic_dir = str(inferred_mic_dir)
    else:
        if not _optional_str(payload.get("cryosparc_base_url")):
            raise ValueError("Connect to CryoSPARC before using a CryoSPARC Fine-tune micrograph source.")
        micrographs_ref = _normalize_output_ref(
            _require_text(payload, "cryosparc_micrographs"),
            default_output_name="micrographs",
        )
        instance_host, instance_port = _instance_url_host_port(payload.get("cryosparc_base_url"))
        run_id = str(uuid.uuid4())
        stage_root = output_dir / "cryosparc_micrograph_source"
        stage_run_dir = stage_root / run_id
        transfer_manifest = stage_run_dir / "transfer_manifest.json"
        stage_argv = [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
        _add_option(stage_argv, "--config", payload.get("config"))
        _add_option(stage_argv, "--host", payload.get("bridge_host") or "local")
        _add_option(stage_argv, "--bridge-command", payload.get("bridge_command"))
        _add_option(stage_argv, "--remote-source-root", payload.get("remote_source_root"))
        _add_option(stage_argv, "--remote-work-root", payload.get("remote_work_root"))
        _add_option(stage_argv, "--cryosparc-host", payload.get("cryosparc_host") or instance_host)
        _add_option(stage_argv, "--cryosparc-base-port", payload.get("cryosparc_base_port") or instance_port)
        _add_option(stage_argv, "--cryosparc-email", payload.get("cryosparc_email"))
        for option in _as_tokens(payload.get("ssh_options")):
            stage_argv.extend(["--ssh-option", option])
        stage_argv.extend(
            [
                "stage-test",
                "--project",
                _require_text(payload, "cryosparc_project"),
                "--workspace",
                _require_text(payload, "cryosparc_workspace"),
                "--micrographs",
                micrographs_ref,
                "--run-id",
                run_id,
                "--local-run-root",
                str(stage_root),
                "--model-id",
                "fine-tune",
                "--title",
                "cryoFILTER fine-tune micrograph source",
                "--all-micrographs",
            ]
        )
        _add_option(stage_argv, "--max-transfer-gb", payload.get("max_transfer_gb") or "25")
        stage_argv.extend(["--no-pull", "--no-stage-micrographs"])
        steps.append(JobStep("Stage CryoSPARC micrograph paths", stage_argv))
        prepare_manifest_argv.extend(
            [
                "--transfer-manifest",
                str(transfer_manifest),
                "--prefer-source-paths",
            ]
        )
        inferred_mic_dir = _infer_manifest_micrograph_root(manifest_path, work_dir=work_dir)
        mic_dir = str(_resolve_work_path(mic_dir_text, work_dir=work_dir)) if mic_dir_text else str(inferred_mic_dir or work_dir)
        artifact_roots.append(stage_run_dir)
        cryosparc_password = _optional_str(payload.get("cryosparc_password"))
        if cryosparc_password is not None:
            secret_env["CRYOSPARC_PASSWORD"] = cryosparc_password

    pretrained_model = _require_text(payload, "pretrained_model")
    split_counts = _manifest_split_counts(manifest_path)
    steps.append(JobStep("Prepare training manifest", prepare_manifest_argv))
    train_script = Path(__file__).resolve().parents[2] / "scripts" / "train_patch_based_fast.py"
    argv = [
        sys.executable,
        str(train_script),
        "--manifest",
        str(training_manifest),
        "--mic_dir",
        mic_dir,
        "--pretrained_model",
        pretrained_model,
        "--output_dir",
        str(output_dir),
        "--model_type",
        "unet_attention",
        "--attention_type",
        "global",
        "--use_psd",
        "true",
        "--adaptive_downsample",
        "--adaptive_downsample_method",
        "fourier",
        "--target_pixel_size",
        str(payload.get("target_pixel_size") or "2.0"),
        "--working_patch_size",
        str(payload.get("working_patch_size") or "256"),
        "--working_stride",
        str(payload.get("working_stride") or "96"),
        "--normalization_method",
        str(payload.get("normalization_method") or "percentile_extra_wide"),
        "--norm_type",
        "group",
        "--batch_size",
        str(payload.get("batch_size") or "16"),
        "--num_gpus",
        str(payload.get("num_gpus") or "1"),
        "--num_epochs",
        str(payload.get("num_epochs") or "120"),
        "--lr",
        str(payload.get("learning_rate") or "2e-5"),
        "--warmup_steps",
        str(payload.get("warmup_steps") or "80"),
        "--weight_decay",
        str(payload.get("weight_decay") or "0.01"),
        "--decoder_dropout",
        "0.1",
        "--label_smoothing",
        "0.0",
        "--full_micrograph_training",
        "--full_mic_patches_per_mic",
        str(payload.get("full_mic_patches_per_mic") or "64"),
        "--full_mic_overlap",
        "0.625",
        "--full_mic_prefetch_workers",
        "1",
        "--skip_patch_cache_in_full_mic",
        "--psd_frequency_band_channels",
        "--psd_frequency_bands",
        "0.03:0.05,0.09:0.11,0.14:0.16,0.22:0.24",
        "--use_dice_loss",
        "--lambda_dice",
        "2.0",
        "--lambda_focal",
        "1.0",
        "--lambda_edge",
        "0.3",
        "--stitched_val_every_n_epochs",
        str(payload.get("stitched_val_every_n_epochs") or "3"),
        "--stitched_val_mics_per_dataset",
        "1",
        "--stitched_val_overlap",
        "160",
        "--stitched_val_batch_forward_size",
        str(payload.get("stitched_val_batch_forward_size") or "16"),
        "--stitched_val_blending_window",
        "tukey",
        "--stitched_val_blending_edge_px",
        "16",
        "--stitched_val_thresholds",
        "0.05,0.15,0.25,0.35,0.45,0.55,0.70",
        "--stitched_val_metrics_max_samples",
        "3000000",
        "--early_stop_patience",
        str(payload.get("early_stop_patience") or "14"),
        "--early_stop_min_epochs",
        "30",
        "--early_stop_min_stitched_cycles",
        "10",
        "--num_workers",
        str(payload.get("num_workers") or "4"),
    ]
    if train_ids is not None and val_ids is not None:
        argv.extend(["--train_dataset_ids", train_ids, "--val_dataset_ids", val_ids])
    if not _as_bool(payload.get("fill_region_holes"), default=False):
        argv.append("--no_region_gt_fill_holes")
    argv.extend(_as_tokens(payload.get("extra_args")))
    return JobSpec(
        kind="train",
        title=_optional_str(payload.get("title")) or "Fine-tune cryoFILTER",
        steps=steps + [JobStep("Fine-tune model", argv)],
        artifact_roots=artifact_roots,
        metadata={
            "manifest": manifest_path,
            "training_manifest": training_manifest,
            "mic_source_mode": mic_source_mode,
            "mic_dir": mic_dir,
            "output_dir": output_dir,
            "train_dataset_ids": train_ids,
            "val_dataset_ids": val_ids,
            "train_fraction": payload.get("train_fraction") or "0.7",
            "split_seed": payload.get("split_seed") or "42",
            "split_counts": split_counts,
        },
        env=secret_env,
    )


def _live_summary_slug(label: str) -> str:
    slug = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return "carbon" if slug == "support" else slug


def _csv_number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _select_summary_rows(
    rows: list[dict[str, str]],
    *,
    mode: str,
    count: int,
) -> list[dict[str, str]]:
    if mode == "first":
        return rows[:count]
    if mode == "last":
        return rows[-count:]
    return rows


def _normalize_live_summary_mode(value: object) -> str:
    mode = (_optional_str(value) or "all").lower()
    return mode if mode in {"all", "first", "last"} else "all"


def _normalize_live_summary_count(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 100
    return max(1, min(count, 10000))


def _resolve_summary_path(raw: object, *, base: Path) -> Path | None:
    text = _optional_str(raw)
    if text is None:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _candidate_typing_summary_files(roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend([root / "typing" / "summary.json", root / "summary.json"])
        if root.exists():
            candidates.extend(root.glob("*/typing/summary.json"))
            candidates.extend(root.glob("*/summary.json"))
    unique = {path.resolve(): path for path in candidates if path.exists() and path.is_file()}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)


def _candidate_inference_summary_files(roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend([root / "inference" / "inference_summary.json", root / "inference_summary.json"])
        if root.exists():
            candidates.extend(root.glob("*/inference_summary.json"))
            candidates.extend(root.glob("*/inference/inference_summary.json"))
    unique = {path.resolve(): path for path in candidates if path.exists() and path.is_file()}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)


def _candidate_worker_inference_summary_files(roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.glob(".cryofilter_worker_*_summary.json") if root.exists() else [])
        candidates.extend(root.glob("*/.cryofilter_worker_*_summary.json") if root.exists() else [])
        candidates.extend(root.glob("*/*/.cryofilter_worker_*_summary.json") if root.exists() else [])
    unique = {path.resolve(): path for path in candidates if path.exists() and path.is_file()}
    return sorted(unique.values(), key=lambda path: str(path))


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _empty_live_summary(message: str) -> dict[str, Any]:
    return {"ok": True, "available": False, "message": message}


def _typing_live_summary(
    summary_path: Path,
    *,
    mode: str,
    count: int,
) -> dict[str, Any] | None:
    summary = _load_json_object(summary_path)
    csv_path = _resolve_summary_path(
        summary.get("image_contamination_summary_csv"),
        base=summary_path.parent,
    )
    if csv_path is None or not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    selected = _select_summary_rows(rows, mode=mode, count=count)
    type_order_raw = summary.get("type_order")
    type_order = [
        str(label)
        for label in type_order_raw
        if str(label).strip()
    ] if isinstance(type_order_raw, list) else list(CONTAMINATION_TYPE_LABELS)
    total_pixels = int(round(sum(_csv_number(row, "total_pixels") for row in selected)))
    contaminated_pixels = int(round(sum(_csv_number(row, "contaminated_pixels") for row in selected)))
    clean_pixels = max(0, total_pixels - contaminated_pixels)
    type_rows = []
    for label in type_order:
        slug = _live_summary_slug(label)
        area = int(round(sum(_csv_number(row, f"{slug}_area_px") for row in selected)))
        type_rows.append(
            {
                "label": label,
                "area_px": area,
                "color": CONTAMINATION_TYPE_COLORS.get(label, "#8DA0AE"),
            }
        )
    return {
        "ok": True,
        "available": True,
        "source": "typing",
        "source_path": str(summary_path),
        "mode": mode,
        "count": count,
        "n_images": int(len(selected)),
        "n_images_total": int(len(rows)),
        "total_pixels": total_pixels,
        "contaminated_pixels": contaminated_pixels,
        "clean_pixels": clean_pixels,
        "contamination_fraction": float(contaminated_pixels / max(total_pixels, 1)),
        "types": type_rows,
    }


def _inference_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, item in enumerate(summary.get("inputs", []), start=1):
        if not isinstance(item, dict):
            continue
        mask_meta = item.get("mask_postprocessing")
        if not isinstance(mask_meta, dict):
            mask_meta = {}
        shape = item.get("output_image_shape") or item.get("input_image_shape")
        total_pixels = 0
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            total_pixels = int(shape[0]) * int(shape[1])
        mask_pixels = mask_meta.get("final_mask_pixels")
        mask_fraction = mask_meta.get("final_mask_fraction")
        if mask_pixels is None and mask_fraction is not None and total_pixels > 0:
            mask_pixels = int(round(float(mask_fraction) * total_pixels))
        if mask_pixels is None:
            continue
        rows.append(
            {
                "image_id": str(item.get("input_mrc") or f"micrograph_{index:03d}"),
                "total_pixels": str(total_pixels),
                "contaminated_pixels": str(int(mask_pixels)),
            }
        )
    return rows


def _inference_live_summary(
    summary_path: Path,
    *,
    mode: str,
    count: int,
) -> dict[str, Any] | None:
    rows = _inference_rows(_load_json_object(summary_path))
    if not rows:
        return None
    return _inference_live_summary_from_rows(
        rows,
        mode=mode,
        count=count,
        source="inference",
        source_path=summary_path,
    )


def _inference_live_summary_from_rows(
    rows: list[dict[str, str]],
    *,
    mode: str,
    count: int,
    source: str,
    source_path: Path | None = None,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    selected = _select_summary_rows(rows, mode=mode, count=count)
    total_pixels = int(round(sum(_csv_number(row, "total_pixels") for row in selected)))
    contaminated_pixels = int(round(sum(_csv_number(row, "contaminated_pixels") for row in selected)))
    payload: dict[str, Any] = {
        "ok": True,
        "available": True,
        "source": source,
        "mode": mode,
        "count": count,
        "n_images": int(len(selected)),
        "n_images_total": int(len(rows)),
        "total_pixels": total_pixels,
        "contaminated_pixels": contaminated_pixels,
        "clean_pixels": max(0, total_pixels - contaminated_pixels),
        "contamination_fraction": float(contaminated_pixels / max(total_pixels, 1)),
        "types": [],
    }
    if source_path is not None:
        payload["source_path"] = str(source_path)
    if source_paths is not None:
        payload["source_paths"] = [str(path) for path in source_paths]
        payload["worker_count"] = int(len(source_paths))
    return payload


def _worker_inference_live_summary(
    summary_paths: Sequence[Path],
    *,
    mode: str,
    count: int,
) -> dict[str, Any] | None:
    rows: list[dict[str, str]] = []
    loaded_paths: list[Path] = []
    for summary_path in summary_paths:
        try:
            worker_rows = _inference_rows(_load_json_object(summary_path))
        except Exception:
            continue
        if not worker_rows:
            continue
        rows.extend(worker_rows)
        loaded_paths.append(summary_path)
    if not rows:
        return None
    rows.sort(key=lambda row: row.get("image_id", ""))
    return _inference_live_summary_from_rows(
        rows,
        mode=mode,
        count=count,
        source="multi_gpu_workers",
        source_paths=loaded_paths,
    )


def _build_live_summary(
    roots: Sequence[Path],
    *,
    mode: str,
    count: int,
) -> dict[str, Any]:
    for summary_path in _candidate_typing_summary_files(roots):
        try:
            summary = _typing_live_summary(summary_path, mode=mode, count=count)
        except Exception:
            continue
        if summary is not None:
            return summary
    for summary_path in _candidate_inference_summary_files(roots):
        try:
            summary = _inference_live_summary(summary_path, mode=mode, count=count)
        except Exception:
            continue
        if summary is not None:
            return summary
    worker_summary_paths = _candidate_worker_inference_summary_files(roots)
    if worker_summary_paths:
        summary = _worker_inference_live_summary(worker_summary_paths, mode=mode, count=count)
        if summary is not None:
            return summary
    return _empty_live_summary("Waiting for inference or typing summary data.")


class AppState:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir.resolve()
        self.data_dir = self.work_dir / APP_DIRNAME
        self.jobs_dir = self.data_dir / "jobs"
        self.annotation_sessions_dir = self.data_dir / "annotation_sessions"
        self.lock = threading.RLock()
        self.active: dict[str, subprocess.Popen[str]] = {}
        self.secret_env: dict[str, dict[str, str]] = {}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_sessions_dir.mkdir(parents=True, exist_ok=True)
        self._mark_stale_jobs()

    def build_job_spec(self, kind: str, payload: dict[str, Any]) -> JobSpec:
        return build_job_spec(kind, payload, work_dir=self.work_dir)

    def create_annotation_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_mode = (_optional_str(payload.get("source_mode")) or "micrographs").lower()
        output_root = _resolve_work_path(
            _optional_str(payload.get("output_root")) or "annotation_exports",
            work_dir=self.work_dir,
        )
        dataset_id = _optional_str(payload.get("dataset_id")) or "1"
        train_fraction = _optional_float(payload.get("train_fraction"), 0.7)
        split_seed = _optional_int(payload.get("split_seed")) or 42
        limit = _optional_int(payload.get("limit_micrographs"))
        session_id = uuid.uuid4().hex[:12]
        title = _optional_str(payload.get("title"))
        mic_dir = (
            _resolve_work_path(payload.get("mic_dir"), work_dir=self.work_dir)
            if _optional_str(payload.get("mic_dir"))
            else None
        )

        if source_mode == "manifest":
            manifest = _resolve_work_path(_require_text(payload, "manifest"), work_dir=self.work_dir)
            output_name = _optional_str(payload.get("output_name"))
            if _as_bool(payload.get("resume_in_place"), default=True) and output_name is None:
                run_dir = manifest.parent
            else:
                run_dir = output_root / _annotation_session_name(payload)
            summary = browser_annotation.create_browser_session(
                session_id=session_id,
                manifest_path=manifest,
                run_dir=run_dir,
                source_mode=source_mode,
                title=title or "Resume annotation",
                mic_dir=mic_dir,
            )
            self._write_annotation_meta(session_id, summary)
            return summary

        session_name = _annotation_session_name(payload)
        session_dir = output_root / session_name
        source_manifest = session_dir / "source_manifest.csv"

        if source_mode == "micrographs":
            micrograph_source = _require_text(payload, "micrograph_source")
            if not source_manifest.exists():
                rows = rows_from_micrographs(
                    collect_micrographs(
                        micrograph_source,
                        recursive=_as_bool(payload.get("recursive"), default=True),
                        limit=limit,
                    ),
                    dataset_id=dataset_id,
                    pixel_size_angstrom=_optional_str(payload.get("pixel_size_angstrom")),
                    train_fraction=train_fraction,
                    seed=split_seed,
                )
                write_manifest(rows, source_manifest, reuse_existing=True)
            summary = browser_annotation.create_browser_session(
                session_id=session_id,
                manifest_path=source_manifest,
                run_dir=session_dir,
                source_mode=source_mode,
                title=title or f"Annotation: {Path(micrograph_source).name}",
                mic_dir=mic_dir,
            )
            self._write_annotation_meta(session_id, summary)
            return summary

        if source_mode != "cryosparc":
            raise ValueError("Annotation source must be micrographs, cryosparc, or manifest.")
        if not _optional_str(payload.get("cryosparc_base_url")):
            raise ValueError("Connect to CryoSPARC before creating an annotation session.")

        micrographs_ref = _normalize_output_ref(
            _require_text(payload, "cryosparc_micrographs"),
            default_output_name="micrographs",
        )
        instance_host, instance_port = _instance_url_host_port(payload.get("cryosparc_base_url"))
        run_id = str(uuid.uuid4())
        stage_root = _resolve_work_path(
            _optional_str(payload.get("local_run_root")) or "cryofilter_runs/annotation_staging",
            work_dir=self.work_dir,
        )
        stage_run_dir = stage_root / run_id
        stage_log_path = stage_root / f"{run_id}.browser_annotation_stage.log"
        transfer_manifest = stage_run_dir / "transfer_manifest.json"
        transfer_mic_dir = stage_run_dir / "transfer" / "micrographs"
        copy_micrographs = _as_bool(payload.get("copy_cryosparc_micrographs"), default=False)
        stage_argv = [sys.executable, "-m", "cryofilter.cli", "cryosparc"]
        _add_option(stage_argv, "--config", payload.get("config"))
        _add_option(stage_argv, "--host", payload.get("bridge_host") or "local")
        _add_option(stage_argv, "--bridge-command", payload.get("bridge_command"))
        _add_option(stage_argv, "--remote-source-root", payload.get("remote_source_root"))
        _add_option(stage_argv, "--remote-work-root", payload.get("remote_work_root"))
        _add_option(stage_argv, "--cryosparc-host", payload.get("cryosparc_host") or instance_host)
        _add_option(stage_argv, "--cryosparc-base-port", payload.get("cryosparc_base_port") or instance_port)
        _add_option(stage_argv, "--cryosparc-email", payload.get("cryosparc_email"))
        for option in _as_tokens(payload.get("ssh_options")):
            stage_argv.extend(["--ssh-option", option])
        stage_argv.extend(
            [
                "stage-test",
                "--project",
                _require_text(payload, "cryosparc_project"),
                "--workspace",
                _require_text(payload, "cryosparc_workspace"),
                "--micrographs",
                micrographs_ref,
                "--run-id",
                run_id,
                "--local-run-root",
                str(stage_root),
                "--model-id",
                "annotation",
                "--title",
                "cryoFILTER annotation source",
            ]
        )
        if limit is None:
            stage_argv.append("--all-micrographs")
        else:
            stage_argv.extend(["--limit-micrographs", str(limit)])
        _add_option(stage_argv, "--max-transfer-gb", payload.get("max_transfer_gb") or "25")
        if not copy_micrographs:
            stage_argv.extend(["--no-pull", "--no-stage-micrographs"])
        self._run_inline_stage(
            stage_argv,
            log_path=stage_log_path,
            secret_env={
                "CRYOSPARC_PASSWORD": _optional_str(payload.get("cryosparc_password")) or "",
            },
        )
        rows = rows_from_transfer_manifest(
            transfer_manifest,
            dataset_id=dataset_id,
            prefer_source_paths=not copy_micrographs,
            train_fraction=train_fraction,
            seed=split_seed,
        )
        if limit is not None:
            rows = rows[:limit]
        write_manifest(rows, source_manifest, reuse_existing=True)
        summary = browser_annotation.create_browser_session(
            session_id=session_id,
            manifest_path=source_manifest,
            run_dir=session_dir,
            source_mode=source_mode,
            title=title or "Annotation from CryoSPARC",
            mic_dir=transfer_mic_dir if copy_micrographs else mic_dir,
        )
        summary.update(
            {
                "cryosparc_project": _require_text(payload, "cryosparc_project"),
                "cryosparc_workspace": _require_text(payload, "cryosparc_workspace"),
                "micrographs": micrographs_ref,
                "stage_run_dir": str(stage_run_dir),
                "stage_log_file": str(stage_log_path),
                "transfer_manifest": str(transfer_manifest),
                "copy_cryosparc_micrographs": copy_micrographs,
            }
        )
        self._write_annotation_meta(session_id, summary)
        return summary

    def list_annotation_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in sorted(self.annotation_sessions_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                sessions.append(browser_annotation.session_summary(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return sessions

    def get_annotation_session(self, session_id: str) -> dict[str, Any]:
        meta = self._read_annotation_meta(session_id)
        return browser_annotation.session_summary(meta)

    def get_annotation_row(self, session_id: str, index: int) -> dict[str, Any]:
        return browser_annotation.get_row_payload(self._read_annotation_meta(session_id), index)

    def get_annotation_preview(self, session_id: str, index: int, *, max_dim: int = 1600) -> tuple[bytes, dict[str, Any]]:
        return browser_annotation.get_preview_png(
            self._read_annotation_meta(session_id),
            index,
            max_dim=max_dim,
        )

    def save_annotation_row(self, session_id: str, index: int, payload: dict[str, Any]) -> dict[str, Any]:
        meta = self._read_annotation_meta(session_id)
        summary = browser_annotation.save_row_annotations(meta, index, payload)
        self._write_annotation_meta(session_id, browser_annotation.session_summary(meta))
        return summary

    def finalize_annotation_session(self, session_id: str) -> dict[str, Any]:
        meta = self._read_annotation_meta(session_id)
        summary = browser_annotation.finalize_session(meta)
        self._write_annotation_meta(session_id, summary)
        return summary

    def start_job(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = self.build_job_spec(kind, payload)
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        log_path = job_dir / "job.log"
        meta = {
            "id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "ended_at": None,
            "returncode": None,
            "work_dir": str(self.work_dir),
            "log_file": str(log_path),
            "cancel_requested": False,
            **spec.as_dict(),
        }
        self._write_meta(job_id, meta)
        if spec.env:
            with self.lock:
                self.secret_env[job_id] = dict(spec.env)
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return meta

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*/meta.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any]:
        meta_path = self._meta_path(job_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"Unknown job: {job_id}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self.lock:
            meta = self.get_job(job_id)
            meta.update(updates)
            self._write_meta(job_id, meta)
            return meta

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            meta = self.update_job(job_id, cancel_requested=True)
            process = self.active.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
            return meta

    def read_log(self, job_id: str, *, offset: int | None = None, tail: int = MAX_LOG_BYTES_DEFAULT) -> dict[str, Any]:
        meta = self.get_job(job_id)
        path = Path(str(meta["log_file"]))
        if not path.exists():
            return {"text": "", "size": 0, "offset": 0}
        size = path.stat().st_size
        start = int(offset) if offset is not None and int(offset) >= 0 else max(0, size - int(tail))
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read()
        return {
            "text": raw.decode("utf-8", errors="replace"),
            "size": size,
            "offset": start,
        }

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        meta = self.get_job(job_id)
        roots = [Path(path).expanduser() for path in meta.get("artifact_roots", [])]
        artifacts: list[dict[str, Any]] = []
        for root_index, root in enumerate(roots):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
                if not path.is_file() or path.suffix.lower() not in ARTIFACT_SUFFIXES:
                    continue
                rel = path.relative_to(root).as_posix()
                artifacts.append(
                    {
                        "root_index": root_index,
                        "root": str(root),
                        "relative_path": rel,
                        "name": path.name,
                        "suffix": path.suffix.lower(),
                        "size": path.stat().st_size,
                        "mtime": int(path.stat().st_mtime),
                        "url": f"/api/jobs/{job_id}/artifact/{root_index}/{quote(rel)}",
                    }
                )
                if len(artifacts) >= 200:
                    return artifacts
        return artifacts

    def live_summary(self, job_id: str, *, mode: str = "all", count: object = 100) -> dict[str, Any]:
        meta = self.get_job(job_id)
        roots = [Path(path).expanduser() for path in meta.get("artifact_roots", [])]
        return _build_live_summary(
            roots,
            mode=_normalize_live_summary_mode(mode),
            count=_normalize_live_summary_count(count),
        )

    def resolve_artifact(self, job_id: str, root_index: int, relative_path: str) -> Path:
        meta = self.get_job(job_id)
        roots = [Path(path).expanduser().resolve() for path in meta.get("artifact_roots", [])]
        if root_index < 0 or root_index >= len(roots):
            raise FileNotFoundError("Unknown artifact root")
        root = roots[root_index]
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError("Artifact path escapes the job output root") from exc
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {relative_path}")
        return candidate

    def _run_job(self, job_id: str) -> None:
        meta = self.update_job(job_id, status="running", started_at=utc_now())
        log_path = Path(str(meta["log_file"]))
        try:
            for step in meta["steps"]:
                argv = [str(item) for item in step["argv"]]
                self._append_log(log_path, f"\n$ {shlex.join(argv)}\n")
                completed = self._run_step(job_id, argv, log_path)
                if completed != 0:
                    latest = self.get_job(job_id)
                    status = "canceled" if latest.get("cancel_requested") else "failed"
                    self.update_job(
                        job_id,
                        status=status,
                        returncode=int(completed),
                        ended_at=utc_now(),
                    )
                    return
            self.update_job(job_id, status="succeeded", returncode=0, ended_at=utc_now())
        except Exception as exc:
            self._append_log(log_path, f"\n{type(exc).__name__}: {exc}\n")
            self.update_job(job_id, status="failed", returncode=1, ended_at=utc_now())
        finally:
            with self.lock:
                self.active.pop(job_id, None)
                self.secret_env.pop(job_id, None)

    def _run_step(self, job_id: str, argv: Sequence[str], log_path: Path) -> int:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        with self.lock:
            env.update(self.secret_env.get(job_id, {}))
        process = subprocess.Popen(
            list(argv),
            cwd=str(self.work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with self.lock:
            self.active[job_id] = process
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
        return int(process.wait())

    def _run_inline_stage(
        self,
        argv: Sequence[str],
        *,
        log_path: Path,
        secret_env: dict[str, str] | None = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if secret_env:
            env.update({key: value for key, value in secret_env.items() if value})
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\n$ {shlex.join([str(item) for item in argv])}\n")
            handle.flush()
            completed = subprocess.run(
                list(argv),
                cwd=str(self.work_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output = completed.stdout or ""
            handle.write(output)
            handle.flush()
        if completed.returncode != 0:
            tail = "\n".join(output.splitlines()[-8:])
            detail = f" See {log_path} for the full staging log."
            if tail:
                detail = f"{tail}\n{detail}"
            raise RuntimeError(
                f"CryoSPARC annotation source staging failed with exit code {completed.returncode}. {detail}"
            )

    def _mark_stale_jobs(self) -> None:
        for job in self.list_jobs():
            if job.get("status") in {"queued", "running"}:
                self.update_job(
                    str(job["id"]),
                    status="unknown",
                    ended_at=utc_now(),
                    returncode=None,
                    stale_reason="server_restarted",
                )

    def _meta_path(self, job_id: str) -> Path:
        safe = "".join(char for char in str(job_id) if char.isalnum() or char in {"-", "_"})
        return self.jobs_dir / safe / "meta.json"

    def _annotation_meta_path(self, session_id: str) -> Path:
        safe = "".join(char for char in str(session_id) if char.isalnum() or char in {"-", "_"})
        return self.annotation_sessions_dir / f"{safe}.json"

    def _read_annotation_meta(self, session_id: str) -> dict[str, Any]:
        path = self._annotation_meta_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown annotation session: {session_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid annotation session metadata: {session_id}")
        return payload

    def _write_annotation_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        self._annotation_meta_path(session_id).write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_meta(self, job_id: str, meta: dict[str, Any]) -> None:
        self._meta_path(job_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def _append_log(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)


class CryoFilterHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def make_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    static_dir = Path(__file__).resolve().parent / "static"

    class Handler(BaseHTTPRequestHandler):
        server_version = "cryoFILTERApp/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/":
                    self._send_file(static_dir / "index.html")
                    return
                if path.startswith("/static/"):
                    self._send_file(static_dir / path.removeprefix("/static/"))
                    return
                if path == "/api/status":
                    self._send_json(
                        {
                            "ok": True,
                            "work_dir": str(state.work_dir),
                            "data_dir": str(state.data_dir),
                            "python": sys.executable,
                            "pid": os.getpid(),
                            "time": utc_now(),
                            **runtime_status(),
                        }
                    )
                    return
                if path == "/api/jobs":
                    self._send_json({"jobs": state.list_jobs()})
                    return
                if path == "/api/annotation/sessions":
                    self._send_json({"sessions": state.list_annotation_sessions()})
                    return
                if path.startswith("/api/annotation/sessions/"):
                    parts = [unquote(part) for part in path.split("/") if part]
                    if len(parts) == 4:
                        self._send_json(state.get_annotation_session(parts[3]))
                        return
                    if len(parts) == 6 and parts[4] == "rows":
                        self._send_json(state.get_annotation_row(parts[3], int(parts[5])))
                        return
                    if len(parts) == 7 and parts[4] == "rows" and parts[6] == "image":
                        query = parse_qs(parsed.query)
                        max_dim = int(query.get("max_dim", ["1600"])[0])
                        data, _info = state.get_annotation_preview(parts[3], int(parts[5]), max_dim=max_dim)
                        self._send_bytes(data, content_type="image/png")
                        return
                if path.startswith("/api/jobs/"):
                    parts = [unquote(part) for part in path.split("/") if part]
                    if len(parts) == 3:
                        self._send_json(state.get_job(parts[2]))
                        return
                    if len(parts) == 4 and parts[3] == "log":
                        query = parse_qs(parsed.query)
                        offset = query.get("offset", [None])[0]
                        tail = int(query.get("tail", [str(MAX_LOG_BYTES_DEFAULT)])[0])
                        self._send_json(
                            state.read_log(
                                parts[2],
                                offset=None if offset is None else int(offset),
                                tail=tail,
                            )
                        )
                        return
                    if len(parts) == 4 and parts[3] == "artifacts":
                        self._send_json({"artifacts": state.list_artifacts(parts[2])})
                        return
                    if len(parts) == 4 and parts[3] == "live-summary":
                        query = parse_qs(parsed.query)
                        mode = query.get("mode", ["all"])[0]
                        count = query.get("count", ["100"])[0]
                        self._send_json(state.live_summary(parts[2], mode=mode, count=count))
                        return
                    if len(parts) >= 6 and parts[3] == "artifact":
                        rel = "/".join(parts[5:])
                        self._send_file(state.resolve_artifact(parts[2], int(parts[4]), rel))
                        return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/api/cryosparc/connect":
                    self._send_json(validate_cryosparc_connection(self._read_json()))
                    return
                if path == "/api/annotation/sessions":
                    self._send_json(
                        state.create_annotation_session(self._read_json()),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if path.startswith("/api/annotation/sessions/"):
                    parts = [unquote(part) for part in path.split("/") if part]
                    if len(parts) == 6 and parts[4] == "rows":
                        self._send_json(state.save_annotation_row(parts[3], int(parts[5]), self._read_json()))
                        return
                    if len(parts) == 5 and parts[4] == "finalize":
                        self._send_json(state.finalize_annotation_session(parts[3]))
                        return
                if path == "/api/jobs":
                    payload = self._read_json()
                    kind = _require_text(payload, "kind")
                    data = payload.get("payload")
                    if data is None:
                        data = {key: value for key, value in payload.items() if key not in {"kind"}}
                    if not isinstance(data, dict):
                        raise ValueError("payload must be a JSON object")
                    self._send_json(state.start_job(kind, data), status=HTTPStatus.CREATED)
                    return
                if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    parts = [unquote(part) for part in path.split("/") if part]
                    if len(parts) == 4:
                        self._send_json(state.cancel_job(parts[2]))
                        return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(
            self,
            data: bytes,
            *,
            content_type: str = "application/octet-stream",
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            resolved = path.resolve()
            if str(path).startswith(str(static_dir)):
                try:
                    resolved.relative_to(static_dir.resolve())
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
            if not resolved.exists() or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            data = resolved.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

    return Handler


__all__ = [
    "AppState",
    "JobSpec",
    "JobStep",
    "add_subparser",
    "build_job_spec",
    "runtime_status",
    "run",
    "validate_cryosparc_connection",
]
