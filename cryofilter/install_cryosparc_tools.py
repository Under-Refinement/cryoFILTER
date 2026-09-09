"""Install a CryoSPARC-version-matched cryosparc-tools package."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TextIO

PYPI_JSON_URL = "https://pypi.org/pypi/cryosparc-tools/json"
LATEST_WARNING = (
    "NOTE: the cryosparc-tools version must match the CryoSPARC server minor "
    "version. If your CryoSPARC server is not up to date, reinstall a matching "
    'version, for example: python -m pip install -U "cryosparc-tools~=4.7.0"'
)


def _parse_release_version(version: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", str(version).strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def cryosparc_minor_from_version(version: str) -> str:
    """Return ``X.Y`` from a CryoSPARC version such as ``v5.0.3`` or ``5.0``."""

    text = str(version).strip()
    match = re.fullmatch(r"v?(\d+)\.(\d+)(?:\.(?:\d+|x))?", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            "CryoSPARC version must look like 5.0, 5.0.3, v5.0.3, or 4.7.x"
        )
    return f"{int(match.group(1))}.{int(match.group(2))}"


def cryosparc_tools_spec_for_server_version(version: str) -> str:
    minor = cryosparc_minor_from_version(version)
    return f"cryosparc-tools~={minor}.0"


def available_minor_versions_from_releases(
    releases: Mapping[str, object] | Iterable[str],
) -> list[str]:
    if isinstance(releases, Mapping):
        version_texts = releases.keys()
    else:
        version_texts = releases

    minors: set[tuple[int, int]] = set()
    for version_text in version_texts:
        parsed = _parse_release_version(str(version_text))
        if parsed is None:
            continue
        major, minor, _patch = parsed
        minors.add((major, minor))
    return [f"{major}.{minor}" for major, minor in sorted(minors, reverse=True)]


def fetch_available_minor_versions(timeout: float = 10.0) -> list[str]:
    with urllib.request.urlopen(PYPI_JSON_URL, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        return []
    return available_minor_versions_from_releases(releases)


def _print_minor_menu(minors: Sequence[str], *, stdout: TextIO) -> None:
    if minors:
        print("Available cryosparc-tools minor-version families:", file=stdout)
        for index, minor in enumerate(minors, start=1):
            print(f"  {index}. CryoSPARC {minor}.x -> cryosparc-tools~={minor}.0", file=stdout)
    else:
        print("Could not load the cryosparc-tools version list from PyPI.", file=stdout)
    print("Press Enter to install the latest cryosparc-tools release instead.", file=stdout)


def _select_minor_from_answer(answer: str, minors: Sequence[str]) -> str | None:
    text = str(answer).strip()
    if not text:
        return None
    if text.isdigit() and minors:
        index = int(text)
        if 1 <= index <= len(minors):
            return minors[index - 1]
    return cryosparc_minor_from_version(text)


def select_cryosparc_tools_spec(
    *,
    cryosparc_version: str | None = None,
    latest: bool = False,
    prompt: bool = True,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    input_func: Callable[[str], str] = input,
    fetch_versions: Callable[[], list[str]] = fetch_available_minor_versions,
) -> tuple[str, bool]:
    """Return ``(pip_requirement, used_latest_default)``."""

    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    if cryosparc_version:
        return cryosparc_tools_spec_for_server_version(cryosparc_version), False

    if latest or not prompt or not stdin.isatty():
        print(LATEST_WARNING, file=stderr)
        return "cryosparc-tools", True

    try:
        minors = fetch_versions()
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Could not fetch cryosparc-tools versions: {exc}", file=stderr)
        minors = []

    _print_minor_menu(minors, stdout=stdout)
    answer = input_func("CryoSPARC server version or menu number: ")
    minor = _select_minor_from_answer(answer, minors)
    if minor is None:
        print(LATEST_WARNING, file=stderr)
        return "cryosparc-tools", True
    return f"cryosparc-tools~={minor}.0", False


def _pip_command(
    requirement: str,
    *,
    python_executable: str,
    upgrade: bool,
) -> list[str]:
    command = [python_executable, "-m", "pip", "install"]
    if upgrade:
        command.append("-U")
    command.append(requirement)
    return command


def run(args: argparse.Namespace) -> int:
    requirement, _used_latest = select_cryosparc_tools_spec(
        cryosparc_version=args.cryosparc_version,
        latest=bool(args.latest),
        prompt=not bool(args.no_prompt),
    )
    command = _pip_command(
        requirement,
        python_executable=str(args.python),
        upgrade=not bool(args.no_upgrade),
    )
    if bool(args.dry_run):
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    return subprocess.run(command, check=False).returncode


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cryosparc-version",
        "--server-version",
        default=None,
        help="CryoSPARC server version, e.g. 5.0, v5.0.3, or 4.7.x.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Skip version matching and install the latest cryosparc-tools package with a compatibility warning.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt; install latest unless --cryosparc-version is provided.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable whose environment should receive cryosparc-tools.",
    )
    parser.add_argument(
        "--no-upgrade",
        action="store_true",
        help="Omit pip's -U flag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command without running it.",
    )


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "install-cryosparc-tools",
        help="Install cryosparc-tools, optionally matched to a CryoSPARC server minor version.",
    )
    _add_arguments(parser)
    parser.set_defaults(func=run)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cryofilter-install-cryosparc-tools",
        description="Install cryosparc-tools for cryoFILTER CryoSPARC integration.",
    )
    _add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
