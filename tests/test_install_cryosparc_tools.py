from __future__ import annotations

import argparse
import io
import sys

from cryofilter.install_cryosparc_tools import (
    LATEST_WARNING,
    available_minor_versions_from_releases,
    cryosparc_minor_from_version,
    cryosparc_tools_spec_for_server_version,
    run,
    select_cryosparc_tools_spec,
)


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_server_version_maps_to_matching_tools_minor() -> None:
    assert cryosparc_minor_from_version("v5.0.3") == "5.0"
    assert cryosparc_minor_from_version("4.7.x") == "4.7"
    assert cryosparc_tools_spec_for_server_version("4.7.1") == "cryosparc-tools~=4.7.0"


def test_available_minor_versions_are_unique_and_newest_first() -> None:
    assert available_minor_versions_from_releases(
        {
            "4.7.0": [],
            "4.7.1": [],
            "5.0.3": [],
            "5.0.0rc1": [],
            "bad": [],
            "4.5.0": [],
        }
    ) == ["5.0", "4.7", "4.5"]


def test_prompt_menu_selection_returns_matching_spec() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    spec, used_latest = select_cryosparc_tools_spec(
        stdin=_TTYStringIO(),
        stdout=stdout,
        stderr=stderr,
        input_func=lambda _prompt: "2",
        fetch_versions=lambda: ["5.0", "4.7"],
    )

    assert spec == "cryosparc-tools~=4.7.0"
    assert used_latest is False
    assert "CryoSPARC 4.7.x" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_blank_prompt_installs_latest_with_warning() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    spec, used_latest = select_cryosparc_tools_spec(
        stdin=_TTYStringIO(),
        stdout=stdout,
        stderr=stderr,
        input_func=lambda _prompt: "",
        fetch_versions=lambda: ["5.0", "4.7"],
    )

    assert spec == "cryosparc-tools"
    assert used_latest is True
    assert LATEST_WARNING in stderr.getvalue()


def test_noninteractive_dry_run_uses_latest_warning(capsys) -> None:
    args = argparse.Namespace(
        cryosparc_version=None,
        latest=False,
        no_prompt=False,
        python=sys.executable,
        no_upgrade=False,
        dry_run=True,
    )

    assert run(args) == 0

    captured = capsys.readouterr()
    assert "cryosparc-tools" in captured.out
    assert LATEST_WARNING in captured.err


def test_dry_run_with_server_version_uses_matching_spec(capsys) -> None:
    args = argparse.Namespace(
        cryosparc_version="v5.0.3",
        latest=False,
        no_prompt=False,
        python=sys.executable,
        no_upgrade=False,
        dry_run=True,
    )

    assert run(args) == 0

    captured = capsys.readouterr()
    assert "cryosparc-tools~=5.0.0" in captured.out
