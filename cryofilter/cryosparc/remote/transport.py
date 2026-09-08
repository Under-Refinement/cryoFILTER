"""SSH and rsync transport primitives for engaging-controlled workflows."""

from __future__ import annotations

import json
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RemoteCommandResult:
    """Captured result of an SSH or rsync command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class RemoteCommandError(RuntimeError):
    """Raised when a remote command exits unsuccessfully."""

    def __init__(self, result: RemoteCommandResult):
        self.result = result
        super().__init__(
            f"Remote command failed with exit code {result.returncode}: "
            + " ".join(result.argv)
        )


class RemoteTransport(Protocol):
    """Abstract remote command and transfer transport."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> RemoteCommandResult:
        ...

    def pull(
        self,
        remote_path: str,
        local_path: Path,
        *,
        timeout: float | None = None,
        dereference: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        ...

    def push(
        self,
        local_path: Path,
        remote_path: str,
        *,
        timeout: float | None = None,
        contents: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        ...


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv:
        raise ValueError("Remote command argv must not be empty")
    values = tuple(str(value) for value in argv)
    for value in values:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"Unsafe remote argument: {value!r}")
    return values


def _validate_remote_path(path: str) -> str:
    value = str(path)
    if not value:
        raise ValueError("Remote path must not be empty")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"Unsafe remote path: {value!r}")
    return value


@dataclass
class SSHRsyncTransport:
    """RemoteTransport implementation using OpenSSH and rsync."""

    host: str
    ssh_options: tuple[str, ...] = ()
    rsync_options: tuple[str, ...] = ("-a", "--partial", "--protect-args")
    log_path: Path | None = None

    def _record(self, result: RemoteCommandResult) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_unix": time.time(),
            "argv": list(result.argv),
            "returncode": int(result.returncode),
            "duration_s": float(result.duration_s),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _run_local(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None,
        check: bool,
    ) -> RemoteCommandResult:
        command = _validate_argv(argv)
        started = time.monotonic()
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = RemoteCommandResult(
            argv=command,
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=float(time.monotonic() - started),
        )
        self._record(result)
        if check and not result.ok:
            raise RemoteCommandError(result)
        return result

    def _rsync_command_prefix(self, extra_options: Sequence[str] = ()) -> list[str]:
        options = list(self.rsync_options)
        for option in extra_options:
            if option not in options:
                options.append(option)
        if self.ssh_options:
            remote_shell = " ".join(shlex.quote(part) for part in ("ssh", *self.ssh_options))
            options.extend(["-e", remote_shell])
        return ["rsync", *options]

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> RemoteCommandResult:
        remote_argv = _validate_argv(argv)
        remote_command = " ".join(shlex.quote(part) for part in remote_argv)
        command = ["ssh", *self.ssh_options, self.host]
        if remote_command:
            command.append(remote_command)
        return self._run_local(command, timeout=timeout, check=check)

    def pull(
        self,
        remote_path: str,
        local_path: Path,
        *,
        timeout: float | None = None,
        dereference: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        source = f"{self.host}:{_validate_remote_path(remote_path)}"
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        extra = ("-L",) if dereference else ()
        return self._run_local(
            [*self._rsync_command_prefix(extra), source, str(target)],
            timeout=timeout,
            check=check,
        )

    def push(
        self,
        local_path: Path,
        remote_path: str,
        *,
        timeout: float | None = None,
        contents: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        source_path = Path(local_path)
        source = str(source_path)
        if contents and source_path.is_dir() and not source.endswith("/"):
            source += "/"
        target = f"{self.host}:{_validate_remote_path(remote_path)}"
        return self._run_local(
            [*self._rsync_command_prefix(), source, target],
            timeout=timeout,
            check=check,
        )


@dataclass
class LocalTransport:
    """RemoteTransport implementation for a bridge running on this machine."""

    log_path: Path | None = None

    def _record(self, result: RemoteCommandResult) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_unix": time.time(),
            "argv": list(result.argv),
            "returncode": int(result.returncode),
            "duration_s": float(result.duration_s),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> RemoteCommandResult:
        command = _validate_argv(argv)
        started = time.monotonic()
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result = RemoteCommandResult(
            argv=command,
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=float(time.monotonic() - started),
        )
        self._record(result)
        if check and not result.ok:
            raise RemoteCommandError(result)
        return result

    def pull(
        self,
        remote_path: str,
        local_path: Path,
        *,
        timeout: float | None = None,
        dereference: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        return self._copy(
            _validate_remote_path(remote_path),
            str(local_path),
            timeout=timeout,
            dereference=dereference,
            check=check,
        )

    def push(
        self,
        local_path: Path,
        remote_path: str,
        *,
        timeout: float | None = None,
        contents: bool = False,
        check: bool = True,
    ) -> RemoteCommandResult:
        source = str(local_path)
        if contents and Path(local_path).is_dir():
            source = source.rstrip("/") + "/"
        return self._copy(
            source,
            _validate_remote_path(remote_path),
            timeout=timeout,
            dereference=False,
            check=check,
        )

    def _copy(
        self,
        source_text: str,
        target_text: str,
        *,
        timeout: float | None,
        dereference: bool,
        check: bool,
    ) -> RemoteCommandResult:
        if timeout is not None and timeout <= 0:
            raise TimeoutError("Copy timeout must be positive")
        started = time.monotonic()
        command = ("local-copy", source_text, target_text)
        try:
            _copy_path(source_text, target_text, dereference=dereference)
            result = RemoteCommandResult(
                argv=command,
                returncode=0,
                stdout="",
                stderr="",
                duration_s=float(time.monotonic() - started),
            )
        except Exception as exc:
            result = RemoteCommandResult(
                argv=command,
                returncode=1,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                duration_s=float(time.monotonic() - started),
            )
            self._record(result)
            if check:
                raise RemoteCommandError(result) from exc
            return result
        self._record(result)
        return result


def _copy_path(source_text: str, target_text: str, *, dereference: bool) -> None:
    copy_contents = source_text.endswith("/")
    source = Path(source_text.rstrip("/")).expanduser()
    target = Path(target_text).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        if copy_contents:
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                _copy_path(str(child), str(target / child.name), dereference=dereference)
            return
        if target.exists() and target.is_dir():
            target = target / source.name
        shutil.copytree(source, target, symlinks=not dereference, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        target = target / source.name
    if source.is_symlink() and not dereference:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.symlink_to(source.readlink())
        return
    shutil.copy2(source, target, follow_symlinks=dereference)


__all__ = [
    "LocalTransport",
    "RemoteCommandError",
    "RemoteCommandResult",
    "RemoteTransport",
    "SSHRsyncTransport",
]
