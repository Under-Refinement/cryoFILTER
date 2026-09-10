from __future__ import annotations

import argparse
import builtins
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cryofilter.cryosparc.bridge.client import connection_kwargs
from cryofilter.cryosparc.bridge.compat import find_external_job
from cryofilter.cryosparc.remote import cli as remote_cli
from cryofilter.cryosparc.remote import predict as predict_helpers
from cryofilter.cryosparc.remote.cli import _bridge_argv
from cryofilter.cryosparc.remote.config import (
    BridgeConfig,
    CryoSPARCApiConfig,
    CryoSPARCConnectionConfig,
    CryoSPARCIntegrationConfig,
    LocalConfig,
    load_config,
)
from cryofilter.cryosparc.remote.transport import (
    LocalTransport,
    RemoteCommandError,
    RemoteCommandResult,
    SSHRsyncTransport,
)


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_find_external_job_uses_external_accessor() -> None:
    class Project:
        def find_external_job(self, job_uid: str):
            return ("project-external", job_uid)

    class Client:
        def find_external_job(self, project_uid: str, job_uid: str):
            return ("client-external", project_uid, job_uid)

    assert find_external_job(Client(), Project(), "P1", "J5") == ("project-external", "J5")

    class ProjectWithoutExternal:
        pass

    assert find_external_job(Client(), ProjectWithoutExternal(), "P1", "J6") == (
        "client-external",
        "P1",
        "J6",
    )


def test_config_loads_cryosparc_api_section(tmp_path: Path) -> None:
    config_path = tmp_path / "cryosparc.toml"
    config_path.write_text(
        "\n".join(
            [
                "[cryosparc]",
                'host = "user@worker.example.edu"',
                "[cryosparc.api]",
                'host = "cryosparc-master.example.edu"',
                'email = "user@example.edu"',
                "base_port = 39000",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.host == "user@worker.example.edu"
    assert config.api.host == "cryosparc-master.example.edu"
    assert config.api.email == "user@example.edu"
    assert config.api.base_port == 39000


def test_bridge_argv_passes_api_environment_without_password() -> None:
    config = CryoSPARCIntegrationConfig(
        api=CryoSPARCApiConfig(
            host="cryosparc-master.example.edu",
            base_port=39000,
            email="user@example.edu",
        ),
        bridge=BridgeConfig(command="/nfs/bridge/bin/cryofilter-bridge"),
    )

    argv = _bridge_argv(config, "doctor", "--json")

    assert argv == [
        "env",
        "CRYOSPARC_BASE_URL=",
        "CRYOSPARC_HOST=cryosparc-master.example.edu",
        "CRYOSPARC_BASE_PORT=39000",
        "CRYOSPARC_EMAIL=user@example.edu",
        "/nfs/bridge/bin/cryofilter-bridge",
        "doctor",
        "--json",
    ]


def test_bridge_argv_uses_module_for_local_default_bridge() -> None:
    config = CryoSPARCIntegrationConfig(
        cryosparc=CryoSPARCConnectionConfig(host="local"),
        api=CryoSPARCApiConfig(base_url="http://cryosparc.example.edu:39000"),
    )

    argv = _bridge_argv(config, "doctor", "--json")

    assert argv == [
        "env",
        "CRYOSPARC_BASE_URL=http://cryosparc.example.edu:39000",
        "CRYOSPARC_HOST=",
        "CRYOSPARC_BASE_PORT=",
        sys.executable,
        "-m",
        "cryofilter.cryosparc.bridge.cli",
        "doctor",
        "--json",
    ]


def test_bridge_argv_passes_configured_bridge_python_to_deployed_launcher() -> None:
    config = CryoSPARCIntegrationConfig(
        api=CryoSPARCApiConfig(base_url="http://cryosparc.example.edu:39000"),
        bridge=BridgeConfig(
            command="/shared/path/cryofilter_bridge_src/bin/cryofilter-bridge",
            python="/shared/path/venvs/cryofilter-bridge/bin/python",
        ),
    )

    argv = _bridge_argv(config, "--version")

    assert "CRYOFILTER_BRIDGE_PYTHON=/shared/path/venvs/cryofilter-bridge/bin/python" in argv
    assert argv[-2:] == [
        "/shared/path/cryofilter_bridge_src/bin/cryofilter-bridge",
        "--version",
    ]


def test_bridge_environment_base_url_excludes_split_host_and_port() -> None:
    config = CryoSPARCIntegrationConfig(
        api=CryoSPARCApiConfig(
            base_url="http://cryosparc.example.edu:39000",
            host="stale-host.example.edu",
            base_port=39001,
            email="user@example.edu",
        ),
    )

    assert remote_cli._bridge_environment(config) == {
        "CRYOSPARC_BASE_URL": "http://cryosparc.example.edu:39000",
        "CRYOSPARC_HOST": "",
        "CRYOSPARC_BASE_PORT": "",
        "CRYOSPARC_EMAIL": "user@example.edu",
    }


def test_bridge_environment_split_api_blanks_inherited_base_url() -> None:
    config = CryoSPARCIntegrationConfig(
        api=CryoSPARCApiConfig(
            host="cryosparc-master.example.edu",
            base_port=39000,
            email="user@example.edu",
        ),
    )

    assert remote_cli._bridge_environment(config) == {
        "CRYOSPARC_BASE_URL": "",
        "CRYOSPARC_HOST": "cryosparc-master.example.edu",
        "CRYOSPARC_BASE_PORT": "39000",
        "CRYOSPARC_EMAIL": "user@example.edu",
    }


def test_connection_kwargs_base_url_ignores_inherited_split_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRYOFILTER_CRYOSPARC_CONNECTION_FILE", raising=False)
    monkeypatch.delenv("CRYOSPARC_INSTANCE_INFO", raising=False)
    monkeypatch.setenv("CRYOSPARC_BASE_URL", "http://cryosparc.example.edu:39000")
    monkeypatch.setenv("CRYOSPARC_HOST", "stale-host.example.edu")
    monkeypatch.setenv("CRYOSPARC_BASE_PORT", "39001")
    monkeypatch.setenv("CRYOSPARC_EMAIL", "user@example.edu")

    kwargs = connection_kwargs()

    assert kwargs["base_url"] == "http://cryosparc.example.edu:39000"
    assert kwargs["email"] == "user@example.edu"
    assert "host" not in kwargs
    assert "base_port" not in kwargs


def test_missing_cryosparc_tools_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cryofilter.cryosparc.bridge import client as bridge_client

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"cryosparc", "cryosparc_tools"} or name.startswith(
            ("cryosparc.", "cryosparc_tools.")
        ):
            missing_root = name.split(".", 1)[0]
            raise ModuleNotFoundError(
                f"No module named {missing_root!r}",
                name=missing_root,
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="cryosparc-bridge"):
        bridge_client.import_cryosparc_class()


def test_parser_keeps_run_typing_unset_until_explicit(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    remote_cli.add_subparser(subparsers)

    args = parser.parse_args(
        [
            "cryosparc",
            "predict",
            "--project",
            "P1",
            "--workspace",
            "W2",
            "--micrographs",
            "J3:micrographs",
            "--particles",
            "J4:particles",
            "--typing-summary",
            str(tmp_path / "typing" / "summary.json"),
        ]
    )

    assert args.run_typing is None

    args = parser.parse_args(
        [
            "cryosparc",
            "finalize-run",
            "--local-run-dir",
            str(tmp_path / "run"),
            "--typing-summary",
            str(tmp_path / "typing" / "summary.json"),
        ]
    )

    assert args.run_typing is None

    args = parser.parse_args(
        [
            "cryosparc",
            "predict",
            "--project",
            "P1",
            "--workspace",
            "W2",
            "--micrographs",
            "J3:micrographs",
            "--particles",
            "J4:particles",
            "--no-run-typing",
        ]
    )

    assert args.run_typing is False


def test_cli_overrides_remote_roots() -> None:
    args = argparse.Namespace(
        host=None,
        bridge_command=None,
        remote_source_root="/shared/cryofilter_bridge_src",
        remote_work_root="/scratch/cryofilter_bridge_work",
        ssh_option=None,
        api_base_url=None,
        api_host=None,
        api_base_port=None,
        api_email=None,
    )

    config = remote_cli._apply_cli_overrides(CryoSPARCIntegrationConfig(), args)

    assert config.bridge.remote_source_root == "/shared/cryofilter_bridge_src"
    assert config.bridge.remote_work_root == "/scratch/cryofilter_bridge_work"


def test_preflight_rejects_cuda_when_torch_has_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    args = argparse.Namespace(checkpoint=str(checkpoint), infer_args=["--device", "cuda"])

    with pytest.raises(RuntimeError, match="CUDA inference was requested"):
        remote_cli._preflight_local_inference_inputs(args)


def test_predict_missing_checkpoint_fails_before_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_transport(config):
        raise AssertionError("transport should not be created before checkpoint preflight")

    monkeypatch.setattr(remote_cli, "_transport", fail_transport)
    run_id = "00000000-0000-0000-0000-000000000404"
    args = argparse.Namespace(
        project="P1",
        workspace="W2",
        micrographs="J3:micrographs",
        particles="J4:particles",
        run_id=run_id,
        model_id="model-a",
        checkpoint=str(tmp_path / "missing.pt"),
        threshold=0.6,
        title="predict",
        limit_micrographs=1,
        micrograph_path_field=None,
        local_run_root=str(tmp_path / "runs"),
        timeout=30.0,
        inference_timeout=None,
        max_transfer_gb=1.0,
        particle_exclusion_distance_angstrom=100.0,
        typing_summary=None,
        run_typing=None,
        typing_normalization_method=None,
        typing_min_component_area_px=None,
        typing_pixel_size_angstrom=None,
        typing_timeout=None,
        max_overlay_images=8,
        max_bar_items=30,
        infer_args=["--device", "cpu"],
        json=True,
    )

    assert remote_cli._run_predict(args, CryoSPARCIntegrationConfig()) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "checkpoint was not found before staging" in payload["errors"][0]
    assert not (tmp_path / "runs" / run_id).exists()


def test_cli_base_url_override_clears_stale_split_api_values() -> None:
    args = argparse.Namespace(
        host=None,
        bridge_command=None,
        remote_source_root=None,
        remote_work_root=None,
        ssh_option=None,
        api_base_url="http://cryosparc.example.edu:39000",
        api_host=None,
        api_base_port=None,
        api_email="user@example.edu",
    )
    config = CryoSPARCIntegrationConfig(
        api=CryoSPARCApiConfig(host="stale-host.example.edu", base_port=39001),
    )

    updated = remote_cli._apply_cli_overrides(config, args)

    assert updated.api.base_url == "http://cryosparc.example.edu:39000"
    assert updated.api.host is None
    assert updated.api.base_port is None
    assert updated.api.email == "user@example.edu"


def test_env_base_url_override_clears_stale_split_api_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRYOFILTER_CRYOSPARC_BASE_URL", "http://cryosparc.example.edu:39000")
    monkeypatch.setenv("CRYOFILTER_CRYOSPARC_API_HOST", "stale-host.example.edu")
    monkeypatch.setenv("CRYOFILTER_CRYOSPARC_BASE_PORT", "39001")

    config = load_config("/does/not/exist/cryosparc.toml")

    assert config.api.base_url == "http://cryosparc.example.edu:39000"
    assert config.api.host is None
    assert config.api.base_port is None


def test_remote_json_is_parsed_even_when_bridge_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTransport:
        def run(self, argv, *, timeout=None):
            raise RemoteCommandError(
                RemoteCommandResult(
                    argv=tuple(argv),
                    returncode=1,
                    stdout='{"ok": false, "errors": ["roundtrip failed"]}',
                    stderr="",
                    duration_s=0.1,
                )
            )

    monkeypatch.setattr(remote_cli, "_transport", lambda config: FakeTransport())

    payload, error = remote_cli._run_remote_bridge_json(
        CryoSPARCIntegrationConfig(),
        ["test-roundtrip", "--json"],
    )

    assert payload == {"ok": False, "errors": ["roundtrip failed"]}
    assert error is None


def test_ssh_transport_uses_argv_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(stdout="worker\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transport = SSHRsyncTransport(host="worker")

    result = transport.run(["cryofilter-bridge", "doctor", "--json"])

    assert result.ok
    assert calls[0][0] == ["ssh", "worker", "cryofilter-bridge doctor --json"]
    assert "shell" not in calls[0][1]


def test_ssh_transport_quotes_remote_arguments_with_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(stdout="worker\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transport = SSHRsyncTransport(host="worker")

    transport.run(["cryofilter-bridge", "test-roundtrip", "--title", "hello world"])

    assert calls[0][0] == [
        "ssh",
        "worker",
        "cryofilter-bridge test-roundtrip --title 'hello world'",
    ]
    assert "shell" not in calls[0][1]


def test_rsync_pull_can_dereference_symlink_farm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    transport = SSHRsyncTransport(host="worker")
    transport.pull("/remote/run/transfer_in/", tmp_path / "input", dereference=True)

    argv = calls[0][0]
    assert argv[0] == "rsync"
    assert "-L" in argv
    assert "worker:/remote/run/transfer_in/" in argv
    assert "shell" not in calls[0][1]


def test_rsync_uses_configured_ssh_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    transport = SSHRsyncTransport(
        host="user@worker.example.edu",
        ssh_options=("-S", ".codex_tmp/worker_mux"),
    )
    transport.push(tmp_path, "~/cryofilter_bridge_src", contents=True)

    argv = calls[0][0]
    assert argv[:2] == ["rsync", "-a"]
    assert "-e" in argv
    assert "ssh -S .codex_tmp/worker_mux" in argv
    assert "user@worker.example.edu:~/cryofilter_bridge_src" in argv


def test_failed_remote_command_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        return _Completed(returncode=255, stderr="ssh failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transport = SSHRsyncTransport(host="worker")

    with pytest.raises(RemoteCommandError) as caught:
        transport.run(["hostname"])

    assert caught.value.result.returncode == 255
    assert "ssh failed" in caught.value.result.stderr


def test_local_transport_runs_without_ssh() -> None:
    transport = LocalTransport()

    result = transport.run([sys.executable, "-c", "print('local bridge')"])

    assert result.ok
    assert result.stdout.strip() == "local bridge"
    assert result.argv[:2] == (sys.executable, "-c")


def test_local_transport_pull_and_push_copy_files(tmp_path: Path) -> None:
    remote_file = tmp_path / "remote" / "payload.txt"
    remote_file.parent.mkdir()
    remote_file.write_text("payload", encoding="utf-8")
    local_file = tmp_path / "local" / "payload.txt"
    pushed_file = tmp_path / "remote" / "pushed.txt"
    transport = LocalTransport()

    pull_result = transport.pull(str(remote_file), local_file)
    push_result = transport.push(local_file, str(pushed_file))

    assert pull_result.ok
    assert push_result.ok
    assert local_file.read_text(encoding="utf-8") == "payload"
    assert pushed_file.read_text(encoding="utf-8") == "payload"


def test_local_host_selects_local_transport() -> None:
    config = CryoSPARCIntegrationConfig(
        cryosparc=CryoSPARCConnectionConfig(host="local"),
    )

    assert isinstance(remote_cli._transport(config), LocalTransport)


def test_remote_arguments_reject_newlines() -> None:
    transport = SSHRsyncTransport(host="worker")

    with pytest.raises(ValueError):
        transport.run(["cryofilter-bridge", "doctor\nwhoops"])


def test_stage_test_uses_generic_prepare_and_manifest_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    remote_manifest = tmp_path / "remote_manifest.json"
    remote_manifest.write_text('{"ok": true}', encoding="utf-8")

    class FakeTransport:
        def run(self, argv, *, timeout=None):
            calls.append(("run", tuple(argv), {"timeout": timeout}))
            return _Completed(
                stdout=(
                    '{"ok": true, "protocol_version": 1, '
                    '"run_id": "00000000-0000-0000-0000-000000000001", '
                    '"external_job_uid": null, "remote_run_dir": "/remote/run", '
                    '"transfer_dir": "/remote/run/transfer", '
                    '"manifest_file": "/remote/run/transfer_manifest.json", '
                    '"n_micrographs": 1, "n_micrographs_total": 60, "n_particles": 12, '
                    '"total_micrograph_bytes": 1024, "errors": []}'
                )
            )

        def pull(self, remote_path, local_path, *, timeout=None, dereference=False, check=True):
            calls.append(
                (
                    "pull",
                    (remote_path, Path(local_path)),
                    {"timeout": timeout, "dereference": dereference, "check": check},
                )
            )
            target = Path(local_path)
            if str(remote_path).endswith("transfer_manifest.json"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(remote_manifest.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                target.mkdir(parents=True, exist_ok=True)
                (target / "micrograph.mrc").write_bytes(b"data")
            return _Completed()

    monkeypatch.setattr(remote_cli, "_transport", lambda config: FakeTransport())
    args = argparse.Namespace(
        project="P1",
        workspace="W5",
        micrographs="J3:micrographs",
        particles="J3:particles",
        run_id="00000000-0000-0000-0000-000000000001",
        model_id="stage-test",
        threshold=0.6,
        title="stage",
        limit_micrographs=1,
        micrograph_path_field=None,
        local_run_root=str(tmp_path / "runs"),
        timeout=30.0,
        max_transfer_gb=1.0,
        no_pull=False,
        create_external_job=False,
        json=True,
    )
    config = CryoSPARCIntegrationConfig(
        cryosparc=CryoSPARCConnectionConfig(host="worker.example.org"),
        bridge=BridgeConfig(
            command="/opt/cryofilter/bin/cryofilter-bridge",
            remote_work_root="/remote/work",
        ),
        local=LocalConfig(run_root=tmp_path / "unused"),
    )

    assert remote_cli._run_stage_test(args, config) == 0

    run_call = calls[0]
    assert run_call[0] == "run"
    assert "--remote-work-root" in run_call[1]
    assert "/remote/work" in run_call[1]
    assert "--no-create-external-job" in run_call[1]
    assert calls[2][1][0] == "/remote/run/transfer/"
    assert calls[2][2]["dereference"] is True


def test_stage_test_aborts_before_large_pull(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeTransport:
        def run(self, argv, *, timeout=None):
            return _Completed(
                stdout=(
                    '{"ok": true, "protocol_version": 1, '
                    '"run_id": "00000000-0000-0000-0000-000000000002", '
                    '"external_job_uid": null, "remote_run_dir": "/remote/run", '
                    '"transfer_dir": "/remote/run/transfer", '
                    '"manifest_file": "/remote/run/transfer_manifest.json", '
                    '"n_micrographs": 1, "n_micrographs_total": 60, "n_particles": null, '
                    '"total_micrograph_bytes": 10737418240, "errors": []}'
                )
            )

        def pull(self, remote_path, local_path, *, timeout=None, dereference=False, check=True):
            calls.append(("pull", str(remote_path)))
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_text("{}", encoding="utf-8")
            return _Completed()

    monkeypatch.setattr(remote_cli, "_transport", lambda config: FakeTransport())
    args = argparse.Namespace(
        project="P1",
        workspace="W5",
        micrographs="J3:micrographs",
        particles=None,
        run_id="00000000-0000-0000-0000-000000000002",
        model_id="stage-test",
        threshold=0.6,
        title="stage",
        limit_micrographs=1,
        micrograph_path_field=None,
        local_run_root=str(tmp_path / "runs"),
        timeout=30.0,
        max_transfer_gb=1.0,
        no_pull=False,
        create_external_job=False,
        json=True,
    )

    assert remote_cli._run_stage_test(args, CryoSPARCIntegrationConfig()) == 1
    assert calls == [("pull", "/remote/run/transfer_manifest.json")]


def test_write_particle_star_from_manifest_uses_relion_one_based_coordinates(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-000000000009"
    transfer = tmp_path / "transfer"
    (transfer / "micrographs").mkdir(parents=True)
    (transfer / "micrographs" / "7_example.mrc").write_bytes(b"not-used-because-shape-is-in-manifest")
    manifest_path = tmp_path / "transfer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "run_id": run_id,
                "project_uid": "P1",
                "workspace_uid": "W2",
                "external_job_uid": "J9",
                "micrographs_ref": {"project_uid": "P1", "job_uid": "J3", "output_name": "micrographs"},
                "particles_ref": {"project_uid": "P1", "job_uid": "J4", "output_name": "particles"},
                "micrographs": [
                    {
                        "uid": 7,
                        "transfer_filename": "micrographs/7_example.mrc",
                        "source_path": "/project/example.mrc",
                        "shape_yx": [100, 200],
                        "pixel_size_angstrom": 1.5,
                    }
                ],
                "particles": [
                    {
                        "uid": 11,
                        "micrograph_uid": 7,
                        "center_x_frac": 0.25,
                        "center_y_frac": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    star_path = tmp_path / "particles.star"
    info = predict_helpers.write_particle_star_from_manifest(
        transfer_manifest_file=manifest_path,
        local_transfer_dir=transfer,
        star_path=star_path,
    )

    assert info["particles_written"] == 1
    text = star_path.read_text(encoding="utf-8")
    assert "_rlnMicrographName #1" in text
    assert "51.000000 51.000000" in text


def test_run_local_inference_builds_required_input_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(returncode=0)

    monkeypatch.setattr(predict_helpers.subprocess, "run", fake_run)

    command = predict_helpers.run_local_inference(
        input_dir=tmp_path / "mics",
        output_dir=tmp_path / "infer",
        checkpoint=tmp_path / "model.pt",
        threshold=0.6,
        particle_star=tmp_path / "particles.star",
        exclusion_distance_angstrom=100.0,
        extra_args=["--", "--device", "cpu"],
        env_overrides={"OMP_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "0"},
        num_cpus=8,
        num_gpus=1,
    )

    assert "--input" in command
    assert command[command.index("--input") + 1].endswith("/mics")
    assert command[command.index("--num-cpus") + 1] == "8"
    assert command[command.index("--num-gpus") + 1] == "1"
    assert "--device" in command
    assert "--no-particle-overlay-contact-sheet" in command
    assert "--no-render-particle-overlays" not in command
    assert calls[0][0] == command
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "8"
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_run_local_inference_can_defer_particle_overlay_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(returncode=0)

    monkeypatch.setattr(predict_helpers.subprocess, "run", fake_run)

    command = predict_helpers.run_local_inference(
        input_dir=tmp_path / "mics",
        output_dir=tmp_path / "infer",
        checkpoint=tmp_path / "model.pt",
        threshold=0.6,
        particle_star=tmp_path / "particles.star",
        exclusion_distance_angstrom=100.0,
        render_particle_overlays=False,
    )

    assert "--no-render-particle-overlays" in command
    assert "--render-particle-overlays" not in command
    assert "--no-particle-overlay-contact-sheet" not in command
    assert calls[0][0] == command


def test_write_typing_manifest_from_transfer_pairs_micrographs_and_masks(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-000000000012"
    transfer = tmp_path / "transfer"
    (transfer / "micrographs").mkdir(parents=True)
    (transfer / "micrographs" / "7_example.mrc").write_bytes(b"not-used-because-pixel-size-is-in-manifest")
    inference = tmp_path / "inference"
    inference.mkdir()
    (inference / "7_example_mask.npy").write_bytes(b"mask")
    manifest_path = tmp_path / "transfer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "run_id": run_id,
                "project_uid": "P1",
                "workspace_uid": "W2",
                "external_job_uid": "J9",
                "micrographs": [
                    {
                        "uid": 7,
                        "transfer_filename": "micrographs/7_example.mrc",
                        "source_path": "/project/example.mrc",
                        "shape_yx": [100, 200],
                        "pixel_size_angstrom": 1.5,
                    }
                ],
                "particles": [],
            }
        ),
        encoding="utf-8",
    )

    output_csv = tmp_path / "typing_manifest.csv"
    info = predict_helpers.write_typing_manifest_from_transfer(
        transfer_manifest_file=manifest_path,
        local_transfer_dir=transfer,
        inference_dir=inference,
        manifest_path=output_csv,
        dataset_id="P1_W2",
    )

    assert info["images"] == 1
    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset_id"] == "P1_W2"
    assert rows[0]["stem"] == "7_example"
    assert rows[0]["micrograph_path"].endswith("/transfer/micrographs/7_example.mrc")
    assert rows[0]["binary_mask_path"].endswith("/inference/7_example_mask.npy")
    assert rows[0]["pixel_size_angstrom"] == "1.5"


def test_cryosparc_helpers_find_internal_masks_when_exports_are_disabled(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-000000000014"
    transfer = tmp_path / "transfer"
    (transfer / "micrographs").mkdir(parents=True)
    (transfer / "micrographs" / "7_example.mrc").write_bytes(b"not-used-because-geometry-is-in-manifest")
    inference = tmp_path / "inference"
    internal = inference / ".cryofilter_internal_maps"
    internal.mkdir(parents=True)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 2] = True
    np.save(internal / "7_example_mask.npy", mask)
    manifest_path = tmp_path / "transfer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "run_id": run_id,
                "project_uid": "P1",
                "workspace_uid": "W2",
                "external_job_uid": "J9",
                "micrographs": [
                    {
                        "uid": 7,
                        "transfer_filename": "micrographs/7_example.mrc",
                        "source_path": "/project/example.mrc",
                        "shape_yx": [10, 10],
                        "pixel_size_angstrom": 1.0,
                    }
                ],
                "particles": [
                    {"uid": 11, "micrograph_uid": 7, "center_x_frac": 0.2, "center_y_frac": 0.2},
                    {"uid": 12, "micrograph_uid": 7, "center_x_frac": 0.8, "center_y_frac": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )

    output_csv = tmp_path / "typing_manifest.csv"
    info = predict_helpers.write_typing_manifest_from_transfer(
        transfer_manifest_file=manifest_path,
        local_transfer_dir=transfer,
        inference_dir=inference,
        manifest_path=output_csv,
        dataset_id="P1_W2",
    )
    split = predict_helpers.classify_particles_from_manifest(
        transfer_manifest_file=manifest_path,
        local_transfer_dir=transfer,
        inference_dir=inference,
        exclusion_distance_angstrom=0.0,
    )

    assert info["images"] == 1
    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["binary_mask_path"].endswith("/inference/.cryofilter_internal_maps/7_example_mask.npy")
    assert split["accepted_uids"] == [12]
    assert split["rejected_uids"] == [11]


def test_run_local_typing_builds_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(returncode=0)

    monkeypatch.setattr(predict_helpers.subprocess, "run", fake_run)

    command = predict_helpers.run_local_typing(
        manifest_path=tmp_path / "typing_manifest.csv",
        output_dir=tmp_path / "typing",
        normalization_method="percentile_extra_wide",
        min_component_area_px=75,
        pixel_size_angstrom=1.35,
        timeout=12.0,
        env_overrides={"OMP_NUM_THREADS": "8"},
    )

    assert command[:4] == [predict_helpers.sys.executable, "-m", "cryofilter.cli", "type"]
    assert "--manifest" in command
    assert "--output-dir" in command
    assert "--min-component-area-px" in command
    assert calls[0][0] == command
    assert calls[0][1]["timeout"] == 12.0
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "8"


def test_refresh_typed_particle_overlays_writes_four_panel_png(tmp_path: Path) -> None:
    import mrcfile
    from PIL import Image

    run_dir = tmp_path / "runs" / "00000000-0000-0000-0000-000000000099"
    transfer_dir = run_dir / "transfer"
    micrograph_dir = transfer_dir / "micrographs"
    inference_dir = run_dir / "inference"
    typing_dir = run_dir / "typing"
    overlay_dir = inference_dir / "OTF_images"
    micrograph_dir.mkdir(parents=True)
    inference_dir.mkdir(parents=True)
    typing_dir.mkdir(parents=True)
    overlay_dir.mkdir(parents=True)

    micrograph = micrograph_dir / "mic_001.mrc"
    yy, xx = np.mgrid[0:48, 0:64]
    with mrcfile.new(micrograph, overwrite=True) as handle:
        handle.set_data((np.sin(xx / 7.0) + np.cos(yy / 9.0)).astype(np.float32))
        handle.voxel_size = 1.5
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:36, 28:52] = 1
    probability = mask.astype(np.float32) * 0.8
    typed_mask = np.zeros((48, 64), dtype=np.uint8)
    typed_mask[12:24, 28:52] = 1
    typed_mask[24:36, 28:52] = 4
    mask_path = inference_dir / "mic_001_mask.npy"
    prob_path = inference_dir / "mic_001_prob.npy"
    typed_dir = typing_dir / "typed_masks"
    typed_dir.mkdir()
    typed_path = typed_dir / "P1_W2__mic_001_typed_mask.npy"
    np.save(mask_path, mask)
    np.save(prob_path, probability)
    np.save(typed_path, typed_mask)

    overlay_path = overlay_dir / "mic_001_particle_overlay.png"
    (run_dir / "transfer_manifest.json").write_text(
        json.dumps(
            {
                "micrographs": [
                    {
                        "uid": 7,
                        "transfer_filename": "micrographs/mic_001.mrc",
                        "shape_yx": [48, 64],
                        "pixel_size_angstrom": 1.5,
                    }
                ],
                "particles": [
                    {"uid": 11, "micrograph_uid": 7, "center_x_frac": 0.5, "center_y_frac": 0.5},
                    {"uid": 12, "micrograph_uid": 7, "center_x_frac": 0.1, "center_y_frac": 0.1},
                ],
            }
        ),
        encoding="utf-8",
    )
    (inference_dir / "inference_summary.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "input_mrc": str(micrograph),
                        "output_mask_npy": str(mask_path),
                        "output_prob_npy": str(prob_path),
                        "particle_overlay": {"output_png": str(overlay_path)},
                    }
                ],
                "particle_overlay_rendering": {
                    "enabled": True,
                    "output_dir": str(overlay_dir),
                    "max_display_dim": 64,
                    "particle_diameter_px": 10,
                    "mask_alpha": 0.35,
                    "frames": [{"micrograph": str(micrograph), "output_png": str(overlay_path)}],
                },
            }
        ),
        encoding="utf-8",
    )
    image_csv = typing_dir / "image_contamination_summary.csv"
    image_csv.write_text(
        "image_id,dataset_id,stem,total_pixels,contaminated_pixels,carbon_area_px,crystalline_area_px,aggregate_area_px,ethane_area_px\n"
        "P1_W2__mic_001,P1_W2,mic_001,3072,576,288,0,0,288\n",
        encoding="utf-8",
    )
    (typing_dir / "summary.json").write_text(
        json.dumps(
            {
                "typed_mask_dir": str(typed_dir),
                "image_contamination_summary_csv": str(image_csv),
            }
        ),
        encoding="utf-8",
    )

    refreshed = remote_cli._refresh_typed_particle_overlays(
        local_manifest_file=run_dir / "transfer_manifest.json",
        local_transfer_dir=transfer_dir,
        local_inference_dir=inference_dir,
        inference_summary_file=inference_dir / "inference_summary.json",
        typing_summary_path=typing_dir / "summary.json",
        particle_exclusion_distance_angstrom=0.0,
    )

    assert refreshed["refreshed"] == 1
    frame = Image.open(overlay_path)
    assert frame.size == (64 * 4 + 16 * 3, 48)
    summary = json.loads((inference_dir / "inference_summary.json").read_text(encoding="utf-8"))
    assert summary["particle_overlay_rendering"]["typed_mask_panel"] is True


def test_refresh_typed_particle_overlays_creates_four_panel_otf_when_inference_deferred(
    tmp_path: Path,
) -> None:
    import mrcfile
    from PIL import Image

    run_dir = tmp_path / "runs" / "00000000-0000-0000-0000-000000000100"
    transfer_dir = run_dir / "transfer"
    micrograph_dir = transfer_dir / "micrographs"
    inference_dir = run_dir / "inference"
    typing_dir = run_dir / "typing"
    micrograph_dir.mkdir(parents=True)
    inference_dir.mkdir(parents=True)
    typing_dir.mkdir(parents=True)

    micrograph = micrograph_dir / "mic_002.mrc"
    yy, xx = np.mgrid[0:48, 0:64]
    with mrcfile.new(micrograph, overwrite=True) as handle:
        handle.set_data((np.sin(xx / 5.0) + np.cos(yy / 11.0)).astype(np.float32))
        handle.voxel_size = 1.5
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:34, 20:44] = 1
    probability = mask.astype(np.float32) * 0.9
    typed_mask = np.zeros((48, 64), dtype=np.uint8)
    typed_mask[10:22, 20:44] = 2
    typed_mask[22:34, 20:44] = 3
    mask_path = inference_dir / "mic_002_mask.npy"
    prob_path = inference_dir / "mic_002_prob.npy"
    typed_dir = typing_dir / "typed_masks"
    typed_dir.mkdir()
    typed_path = typed_dir / "P1_W2__mic_002_typed_mask.npy"
    np.save(mask_path, mask)
    np.save(prob_path, probability)
    np.save(typed_path, typed_mask)
    (run_dir / "transfer_manifest.json").write_text(
        json.dumps(
            {
                "micrographs": [
                    {
                        "uid": 7,
                        "transfer_filename": "micrographs/mic_002.mrc",
                        "shape_yx": [48, 64],
                        "pixel_size_angstrom": 1.5,
                    }
                ],
                "particles": [
                    {"uid": 11, "micrograph_uid": 7, "center_x_frac": 0.5, "center_y_frac": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    (inference_dir / "inference_summary.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "input_mrc": str(micrograph),
                        "output_mask_npy": str(mask_path),
                        "output_prob_npy": str(prob_path),
                    }
                ],
                "particle_overlay_rendering": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    image_csv = typing_dir / "image_contamination_summary.csv"
    image_csv.write_text(
        "image_id,dataset_id,stem,total_pixels,contaminated_pixels,carbon_area_px,crystalline_area_px,aggregate_area_px,ethane_area_px\n"
        "P1_W2__mic_002,P1_W2,mic_002,3072,576,0,288,288,0\n",
        encoding="utf-8",
    )
    (typing_dir / "summary.json").write_text(
        json.dumps(
            {
                "typed_mask_dir": str(typed_dir),
                "image_contamination_summary_csv": str(image_csv),
            }
        ),
        encoding="utf-8",
    )

    refreshed = remote_cli._refresh_typed_particle_overlays(
        local_manifest_file=run_dir / "transfer_manifest.json",
        local_transfer_dir=transfer_dir,
        local_inference_dir=inference_dir,
        inference_summary_file=inference_dir / "inference_summary.json",
        typing_summary_path=typing_dir / "summary.json",
        particle_exclusion_distance_angstrom=0.0,
        create_if_missing=True,
    )

    assert refreshed["refreshed"] == 1
    output_path = inference_dir / "OTF_images" / "mic_002_particle_overlay.png"
    frame = Image.open(output_path)
    assert frame.size == (64 * 4 + 16 * 3, 48)
    summary = json.loads((inference_dir / "inference_summary.json").read_text(encoding="utf-8"))
    overlay = summary["particle_overlay_rendering"]
    assert overlay["typed_mask_panel"] is True
    assert overlay["frames"][0]["typed_mask_panel"] is True
    assert summary["inputs"][0]["particle_overlay"]["output_png"] == str(output_path)


def test_required_typed_particle_overlay_refresh_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="4-panel typed OTF"):
        remote_cli._require_typed_particle_overlay_refresh(
            {"enabled": True, "refreshed": 0, "reason": "no typed masks found"},
            run_typing=True,
        )


def test_auto_typing_defaults_to_enabled_without_summary(tmp_path: Path) -> None:
    assert remote_cli._resolve_auto_typing(
        argparse.Namespace(run_typing=None),
        typing_summary_path=None,
    )
    assert not remote_cli._resolve_auto_typing(
        argparse.Namespace(run_typing=None),
        typing_summary_path=tmp_path / "summary.json",
    )
    assert not remote_cli._resolve_auto_typing(
        argparse.Namespace(run_typing=False),
        typing_summary_path=None,
    )
    assert remote_cli._resolve_auto_typing(
        argparse.Namespace(run_typing=True),
        typing_summary_path=None,
    )


def test_predict_orchestrates_prepare_infer_push_and_finalize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run_id = "00000000-0000-0000-0000-000000000010"
    transfer_manifest = {
        "protocol_version": 1,
        "run_id": run_id,
        "project_uid": "P1",
        "workspace_uid": "W2",
        "external_job_uid": "J99",
        "micrographs_ref": {"project_uid": "P1", "job_uid": "J3", "output_name": "micrographs"},
        "particles_ref": {"project_uid": "P1", "job_uid": "J4", "output_name": "particles"},
        "micrographs": [
            {
                "uid": 7,
                "transfer_filename": "micrographs/7_example.mrc",
                "source_path": "/project/example.mrc",
                "remote_transfer_path": "/remote/run/transfer/micrographs/7_example.mrc",
                "shape_yx": [100, 200],
                "pixel_size_angstrom": 1.5,
            }
        ],
        "particles": [
            {"uid": 11, "micrograph_uid": 7, "center_x_frac": 0.25, "center_y_frac": 0.5},
            {"uid": 12, "micrograph_uid": 7, "center_x_frac": 0.75, "center_y_frac": 0.5},
        ],
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeDiagnosticsManifest:
        assets: list[object] = []

    captured_diagnostics: dict[str, object] = {}
    captured_inference: dict[str, object] = {}
    captured_typing: dict[str, object] = {}
    captured_typed_overlay_refresh: dict[str, object] = {}

    class FakeTransport:
        def run(self, argv, *, timeout=None):
            calls.append(("run", tuple(argv), {"timeout": timeout}))
            if "prepare" in argv:
                return _Completed(
                    stdout=(
                        '{"ok": true, "protocol_version": 1, '
                        f'"run_id": "{run_id}", '
                        '"external_job_uid": "J99", "remote_run_dir": "/remote/run", '
                        '"transfer_dir": "/remote/run/transfer", '
                        '"manifest_file": "/remote/run/transfer_manifest.json", '
                        '"n_micrographs": 1, "n_micrographs_total": 1, "n_particles": 2, '
                        '"total_micrograph_bytes": 1024, "errors": []}'
                    )
                )
            if "finalize-prediction" in argv:
                return _Completed(
                    stdout=(
                        '{"ok": true, "protocol_version": 1, '
                        f'"run_id": "{run_id}", "project_uid": "P1", '
                        '"workspace_uid": "W2", "job_uid": "J99", "status": "completed", '
                        '"input_particles": 2, "accepted_particles": 1, "rejected_particles": 1, '
                        '"saved_outputs": {"particles_accepted": 1, "particles_rejected": 1}, '
                        '"validated": true, "attached_assets": 1, "event_ids": ["event_1"], "errors": []}'
                    )
                )
            return _Completed()

        def pull(self, remote_path, local_path, *, timeout=None, dereference=False, check=True):
            calls.append(
                (
                    "pull",
                    (remote_path, Path(local_path)),
                    {"timeout": timeout, "dereference": dereference, "check": check},
                )
            )
            target = Path(local_path)
            if str(remote_path).endswith("transfer_manifest.json"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(transfer_manifest), encoding="utf-8")
            else:
                (target / "micrographs").mkdir(parents=True, exist_ok=True)
                (target / "micrographs" / "7_example.mrc").write_bytes(b"mrc")
            return _Completed()

        def push(self, local_path, remote_path, *, timeout=None, contents=False, check=True):
            calls.append(
                (
                    "push",
                    (Path(local_path), remote_path),
                    {"timeout": timeout, "contents": contents, "check": check},
                )
            )
            return _Completed()

    def fake_run_local_inference(**kwargs):
        captured_inference.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "inference_summary.json").write_text(
            json.dumps({"run_id": run_id, "inputs": [], "particle_filtering": {}}),
            encoding="utf-8",
        )
        return ["cryofilter", "infer", "--device", "cpu"]

    def fake_write_typing_manifest_from_transfer(**kwargs):
        manifest_path = Path(kwargs["manifest_path"])
        manifest_path.write_text(
            "dataset_id,stem,micrograph_path,binary_mask_path,pixel_size_angstrom\n",
            encoding="utf-8",
        )
        return {"typing_manifest": str(manifest_path), "images": 1, "dataset_id": "P1_W2"}

    def fake_run_local_typing(**kwargs):
        captured_typing.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "component_type_assignments.csv",
            "image_contamination_summary.csv",
            "dataset_contamination_summary.csv",
        ):
            (output_dir / name).write_text("demo\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "component_type_assignments_csv": str(output_dir / "component_type_assignments.csv"),
                    "image_contamination_summary_csv": str(output_dir / "image_contamination_summary.csv"),
                    "dataset_contamination_summary_csv": str(output_dir / "dataset_contamination_summary.csv"),
                    "overall_contamination_summary": {},
                }
            ),
            encoding="utf-8",
        )
        return ["cryofilter", "type"]

    def fake_build_diagnostics_manifest(**kwargs):
        captured_diagnostics.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cryosparc_diagnostics_manifest.json").write_text(
            '{"protocol_version": 1, "assets": [], "summary": {}}',
            encoding="utf-8",
        )
        return FakeDiagnosticsManifest()

    def fake_upload_diagnostics_manifest(**kwargs):
        return {
            "remote_manifest_file": "/remote/run/diagnostics/cryosparc_diagnostics_manifest.json",
            "assets_uploaded": 1,
            "assets": [
                {
                    "kind": "contamination_total_pie",
                    "title": "Total Contamination",
                    "caption": "summary",
                    "local_path": str(tmp_path / "diagnostics" / "chart.png"),
                    "remote_path": "/remote/run/diagnostics/assets/01_chart.png",
                    "mime_type": "image/png",
                    "source": "test",
                }
            ],
            "summary": {},
        }

    fake_transport = FakeTransport()
    monkeypatch.setattr(remote_cli, "_transport", lambda config: fake_transport)
    monkeypatch.setattr(predict_helpers, "run_local_inference", fake_run_local_inference)
    monkeypatch.setattr(
        predict_helpers,
        "write_typing_manifest_from_transfer",
        fake_write_typing_manifest_from_transfer,
    )
    monkeypatch.setattr(predict_helpers, "run_local_typing", fake_run_local_typing)
    monkeypatch.setattr(
        remote_cli,
        "_refresh_typed_particle_overlays",
        lambda **kwargs: captured_typed_overlay_refresh.update(kwargs)
        or {"enabled": True, "refreshed": 1, "skipped": 0},
    )
    monkeypatch.setattr(
        predict_helpers,
        "classify_particles_from_manifest",
        lambda **kwargs: {
            "particles_processed": 2,
            "accepted_uids": [11],
            "rejected_uids": [12],
            "accepted_particles": 1,
            "rejected_particles": 1,
            "per_micrograph": [],
        },
    )
    monkeypatch.setattr(remote_cli, "_build_local_diagnostics_manifest", fake_build_diagnostics_manifest)
    monkeypatch.setattr(remote_cli, "_upload_diagnostics_manifest", fake_upload_diagnostics_manifest)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = argparse.Namespace(
        project="P1",
        workspace="W2",
        micrographs="J3",
        particles="J4",
        run_id=run_id,
        model_id="model-a",
        checkpoint=str(checkpoint),
        threshold=0.6,
        title="predict",
        limit_micrographs=1,
        micrograph_path_field=None,
        local_run_root=str(tmp_path / "runs"),
        timeout=30.0,
        inference_timeout=None,
        num_cpus=8,
        num_gpus=1,
        max_transfer_gb=1.0,
        particle_exclusion_distance_angstrom=100.0,
        typing_summary=None,
        run_typing=None,
        typing_normalization_method=None,
        typing_min_component_area_px=None,
        typing_pixel_size_angstrom=None,
        typing_timeout=None,
        max_overlay_images=8,
        max_bar_items=30,
        infer_args=["--", "--device", "cpu"],
        json=True,
    )
    config = CryoSPARCIntegrationConfig(
        bridge=BridgeConfig(
            command="/opt/cryofilter/bin/cryofilter-bridge",
            remote_work_root="/remote/work",
        ),
        local=LocalConfig(run_root=tmp_path / "unused"),
    )

    assert remote_cli._run_predict(args, config) == 0

    assert captured_inference["num_cpus"] == 8
    assert captured_inference["num_gpus"] == 1
    assert captured_inference["render_particle_overlays"] is False
    assert captured_inference["env_overrides"]["OMP_NUM_THREADS"] == "8"
    assert captured_inference["env_overrides"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured_typing["env_overrides"]["OMP_NUM_THREADS"] == "8"
    assert captured_typed_overlay_refresh["create_if_missing"] is True

    prepare_call = next(call for call in calls if call[0] == "run" and "prepare" in call[1])
    assert "--no-create-external-job" not in prepare_call[1]
    assert prepare_call[1][prepare_call[1].index("--micrographs") + 1] == "J3:micrographs"
    assert prepare_call[1][prepare_call[1].index("--particles") + 1] == "J4:particles"
    assert ("pull", ("/remote/run/transfer/", tmp_path / "runs" / run_id / "transfer"), {"timeout": 30.0, "dereference": True, "check": True}) in calls
    finalize_call = next(call for call in calls if call[0] == "run" and "finalize-prediction" in call[1])
    assert "/remote/run/results/result_manifest.json" in finalize_call[1]

    result_push = next(
        call
        for call in calls
        if call[0] == "push" and call[1][1] == "/remote/run/results/result_manifest.json"
    )
    result_payload = json.loads(Path(result_push[1][0]).read_text(encoding="utf-8"))
    assert result_payload["status"] == "completed"
    assert result_payload["accepted_particles"] == 1
    assert result_payload["rejected_particles"] == 1
    assert result_payload["model"]["typing"] is True
    assert result_payload["files"]["accepted_uids_json"] == "/remote/run/results/accepted_uids.json"
    assert result_payload["files"]["diagnostics_manifest_json"].endswith(
        "cryosparc_diagnostics_manifest.json"
    )
    assert result_payload["files"]["typing_summary_json"].endswith("/typing/summary.json")
    assert result_payload["files"]["typing_manifest_csv"].endswith(
        "/typing/contamination_typing_manifest.csv"
    )
    assert result_payload["diagnostics"][0]["kind"] == "contamination_total_pie"
    assert str(captured_diagnostics["typing_summary_path"]).endswith("/typing/summary.json")


def test_finalize_run_reuses_local_run_and_runs_typing_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "00000000-0000-0000-0000-000000000013"
    local_run_dir = tmp_path / "runs" / run_id
    transfer_dir = local_run_dir / "transfer"
    inference_dir = local_run_dir / "inference"
    (transfer_dir / "micrographs").mkdir(parents=True)
    inference_dir.mkdir(parents=True)
    (transfer_dir / "micrographs" / "7_example.mrc").write_bytes(b"mrc")
    (inference_dir / "inference_summary.json").write_text(
        json.dumps({"run_id": run_id, "inputs": [], "particle_filtering": {}}),
        encoding="utf-8",
    )
    transfer_manifest = {
        "protocol_version": 1,
        "run_id": run_id,
        "project_uid": "P1",
        "workspace_uid": "W2",
        "external_job_uid": "J99",
        "micrographs_ref": {"project_uid": "P1", "job_uid": "J3", "output_name": "micrographs"},
        "particles_ref": {"project_uid": "P1", "job_uid": "J4", "output_name": "particles"},
        "micrographs": [
            {
                "uid": 7,
                "transfer_filename": "micrographs/7_example.mrc",
                "source_path": "/project/example.mrc",
                "shape_yx": [100, 200],
                "pixel_size_angstrom": 1.5,
            }
        ],
        "particles": [
            {"uid": 11, "micrograph_uid": 7, "center_x_frac": 0.25, "center_y_frac": 0.5},
            {"uid": 12, "micrograph_uid": 7, "center_x_frac": 0.75, "center_y_frac": 0.5},
        ],
    }
    (local_run_dir / "transfer_manifest.json").write_text(
        json.dumps(transfer_manifest),
        encoding="utf-8",
    )
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeDiagnosticsManifest:
        assets: list[object] = []

    captured_diagnostics: dict[str, object] = {}
    captured_typing: dict[str, object] = {}
    captured_typed_overlay_refresh: dict[str, object] = {}

    class FakeTransport:
        def run(self, argv, *, timeout=None):
            calls.append(("run", tuple(argv), {"timeout": timeout}))
            if "finalize-prediction" in argv:
                return _Completed(
                    stdout=(
                        '{"ok": true, "protocol_version": 1, '
                        f'"run_id": "{run_id}", "project_uid": "P1", '
                        '"workspace_uid": "W2", "job_uid": "J99", "status": "completed", '
                        '"input_particles": 2, "accepted_particles": 1, "rejected_particles": 1, '
                        '"saved_outputs": {"particles_accepted": 1, "particles_rejected": 1}, '
                        '"validated": true, "attached_assets": 1, "event_ids": ["event_1"], "errors": []}'
                    )
                )
            return _Completed()

        def push(self, local_path, remote_path, *, timeout=None, contents=False, check=True):
            calls.append(
                (
                    "push",
                    (Path(local_path), remote_path),
                    {"timeout": timeout, "contents": contents, "check": check},
                )
            )
            return _Completed()

    def fake_write_typing_manifest_from_transfer(**kwargs):
        manifest_path = Path(kwargs["manifest_path"])
        captured_typing["manifest_path"] = manifest_path
        manifest_path.write_text(
            "dataset_id,stem,micrograph_path,binary_mask_path,pixel_size_angstrom\n",
            encoding="utf-8",
        )
        return {"typing_manifest": str(manifest_path), "images": 1, "dataset_id": "P1_W2"}

    def fake_run_local_typing(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        captured_typing["output_dir"] = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "component_type_assignments.csv",
            "image_contamination_summary.csv",
            "dataset_contamination_summary.csv",
        ):
            (output_dir / name).write_text("demo\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "component_type_assignments_csv": str(output_dir / "component_type_assignments.csv"),
                    "image_contamination_summary_csv": str(output_dir / "image_contamination_summary.csv"),
                    "dataset_contamination_summary_csv": str(output_dir / "dataset_contamination_summary.csv"),
                    "overall_contamination_summary": {},
                }
            ),
            encoding="utf-8",
        )
        return ["cryofilter", "type"]

    def fake_build_diagnostics_manifest(**kwargs):
        captured_diagnostics.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cryosparc_diagnostics_manifest.json").write_text(
            '{"protocol_version": 1, "assets": [], "summary": {}}',
            encoding="utf-8",
        )
        return FakeDiagnosticsManifest()

    def fake_upload_diagnostics_manifest(**kwargs):
        return {
            "remote_manifest_file": (
                "/remote/work/00000000-0000-0000-0000-000000000013/"
                "diagnostics/cryosparc_diagnostics_manifest.json"
            ),
            "assets_uploaded": 1,
            "assets": [
                {
                    "kind": "contamination_total_pie",
                    "title": "Total Contamination",
                    "caption": "summary",
                    "local_path": str(tmp_path / "diagnostics" / "chart.png"),
                    "remote_path": (
                        "/remote/work/00000000-0000-0000-0000-000000000013/"
                        "diagnostics/assets/01_chart.png"
                    ),
                    "mime_type": "image/png",
                    "source": "test",
                }
            ],
            "summary": {},
        }

    monkeypatch.setattr(remote_cli, "_transport", lambda config: FakeTransport())
    monkeypatch.setattr(
        predict_helpers,
        "write_typing_manifest_from_transfer",
        fake_write_typing_manifest_from_transfer,
    )
    monkeypatch.setattr(predict_helpers, "run_local_typing", fake_run_local_typing)
    monkeypatch.setattr(
        remote_cli,
        "_refresh_typed_particle_overlays",
        lambda **kwargs: captured_typed_overlay_refresh.update(kwargs)
        or {"enabled": True, "refreshed": 1, "skipped": 0},
    )
    monkeypatch.setattr(
        predict_helpers,
        "classify_particles_from_manifest",
        lambda **kwargs: {
            "particles_processed": 2,
            "accepted_uids": [11],
            "rejected_uids": [12],
            "accepted_particles": 1,
            "rejected_particles": 1,
            "per_micrograph": [],
        },
    )
    monkeypatch.setattr(remote_cli, "_build_local_diagnostics_manifest", fake_build_diagnostics_manifest)
    monkeypatch.setattr(remote_cli, "_upload_diagnostics_manifest", fake_upload_diagnostics_manifest)
    args = argparse.Namespace(
        cryosparc_command="finalize-run",
        local_run_dir=str(local_run_dir),
        transfer_manifest=None,
        local_transfer_dir=None,
        inference_dir=None,
        inference_summary=None,
        project=None,
        workspace=None,
        job=None,
        remote_run_dir=None,
        remote_transfer_manifest=None,
        model_id="model-a",
        checkpoint="/models/cryoFILTER.pt",
        threshold=0.6,
        timeout=30.0,
        particle_exclusion_distance_angstrom=100.0,
        typing_summary=None,
        run_typing=None,
        typing_normalization_method=None,
        typing_min_component_area_px=None,
        typing_pixel_size_angstrom=None,
        typing_timeout=None,
        max_overlay_images=10,
        max_bar_items=30,
        json=True,
    )
    config = CryoSPARCIntegrationConfig(
        bridge=BridgeConfig(
            command="/opt/cryofilter/bin/cryofilter-bridge",
            remote_work_root="/remote/work",
        )
    )

    assert remote_cli._run_finalize_run(args, config) == 0

    assert captured_typing["manifest_path"] == local_run_dir / "contamination_typing_manifest.csv"
    assert captured_typing["output_dir"] == local_run_dir / "typing"
    assert captured_typed_overlay_refresh["create_if_missing"] is True
    assert str(captured_diagnostics["typing_summary_path"]).endswith("/typing/summary.json")
    finalize_call = next(call for call in calls if call[0] == "run" and "finalize-prediction" in call[1])
    assert "/remote/work/00000000-0000-0000-0000-000000000013/transfer_manifest.json" in finalize_call[1]
    assert "/remote/work/00000000-0000-0000-0000-000000000013/results/result_manifest.json" in finalize_call[1]
    result_push = next(
        call
        for call in calls
        if call[0] == "push"
        and call[1][1]
        == "/remote/work/00000000-0000-0000-0000-000000000013/results/result_manifest.json"
    )
    result_payload = json.loads(Path(result_push[1][0]).read_text(encoding="utf-8"))
    assert result_payload["model"]["typing"] is True
    assert result_payload["accepted_particles"] == 1
    assert result_payload["rejected_particles"] == 1


def test_predict_pushes_failed_result_when_local_inference_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "00000000-0000-0000-0000-000000000011"
    calls: list[tuple[str, tuple[object, ...]]] = []
    transfer_manifest = {
        "protocol_version": 1,
        "run_id": run_id,
        "project_uid": "P1",
        "workspace_uid": "W2",
        "external_job_uid": "J99",
        "micrographs_ref": {"project_uid": "P1", "job_uid": "J3", "output_name": "micrographs"},
        "particles_ref": {"project_uid": "P1", "job_uid": "J4", "output_name": "particles"},
        "micrographs": [
            {
                "uid": 7,
                "transfer_filename": "micrographs/7_example.mrc",
                "source_path": "/project/example.mrc",
                "shape_yx": [100, 200],
                "pixel_size_angstrom": 1.5,
            }
        ],
        "particles": [
            {"uid": 11, "micrograph_uid": 7, "center_x_frac": 0.25, "center_y_frac": 0.5}
        ],
    }

    class FakeTransport:
        def run(self, argv, *, timeout=None):
            calls.append(("run", tuple(argv)))
            if "prepare" in argv:
                return _Completed(
                    stdout=(
                        '{"ok": true, "protocol_version": 1, '
                        f'"run_id": "{run_id}", '
                        '"external_job_uid": "J99", "remote_run_dir": "/remote/run", '
                        '"transfer_dir": "/remote/run/transfer", '
                        '"manifest_file": "/remote/run/transfer_manifest.json", '
                        '"n_micrographs": 1, "n_micrographs_total": 1, "n_particles": 1, '
                        '"total_micrograph_bytes": 1024, "errors": []}'
                    )
                )
            if "finalize-prediction" in argv:
                return _Completed(
                    stdout=(
                        '{"ok": false, "protocol_version": 1, '
                        f'"run_id": "{run_id}", "project_uid": "P1", '
                        '"workspace_uid": "W2", "job_uid": "J99", "status": "failed", '
                        '"input_particles": 0, "accepted_particles": 0, "rejected_particles": 0, '
                        '"saved_outputs": {}, "validated": false, "attached_assets": 0, '
                        '"event_ids": [], "errors": ["inference failed"]}'
                    )
                )
            return _Completed()

        def pull(self, remote_path, local_path, *, timeout=None, dereference=False, check=True):
            calls.append(("pull", (remote_path, Path(local_path))))
            target = Path(local_path)
            if str(remote_path).endswith("transfer_manifest.json"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(transfer_manifest), encoding="utf-8")
            else:
                (target / "micrographs").mkdir(parents=True, exist_ok=True)
                (target / "micrographs" / "7_example.mrc").write_bytes(b"mrc")
            return _Completed()

        def push(self, local_path, remote_path, *, timeout=None, contents=False, check=True):
            calls.append(("push", (Path(local_path), remote_path)))
            return _Completed()

    monkeypatch.setattr(remote_cli, "_transport", lambda config: FakeTransport())
    monkeypatch.setattr(
        predict_helpers,
        "run_local_inference",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = argparse.Namespace(
        project="P1",
        workspace="W2",
        micrographs="J3:micrographs",
        particles="J4:particles",
        run_id=run_id,
        model_id="model-a",
        checkpoint=str(checkpoint),
        threshold=0.6,
        title="predict",
        limit_micrographs=1,
        micrograph_path_field=None,
        local_run_root=str(tmp_path / "runs"),
        timeout=30.0,
        inference_timeout=None,
        max_transfer_gb=1.0,
        particle_exclusion_distance_angstrom=100.0,
        typing_summary=None,
        max_overlay_images=8,
        max_bar_items=30,
        infer_args=[],
        json=True,
    )

    assert remote_cli._run_predict(args, CryoSPARCIntegrationConfig()) == 1
    failed_push = next(
        call
        for call in calls
        if call[0] == "push" and call[1][1] == "/remote/run/results/result_manifest.failed.json"
    )
    failed_payload = json.loads(Path(failed_push[1][0]).read_text(encoding="utf-8"))
    assert failed_payload["status"] == "failed"
    assert "inference failed" in failed_payload["error"]
    assert any(call[0] == "run" and "finalize-prediction" in call[1] for call in calls)
