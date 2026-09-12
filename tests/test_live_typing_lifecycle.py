"""Background supervision must preserve cancellation and retry behavior."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from cryofilter.cryosparc.remote import cli, predict


def _updater(tmp_path):
    manifest = tmp_path / "transfer_manifest.json"
    manifest.write_text(json.dumps({"micrographs": [{"uid": 1}]}))
    return cli._LiveTypingUpdater(
        local_manifest_file=manifest, local_transfer_dir=tmp_path,
        local_inference_dir=tmp_path, inference_summary_file=tmp_path / "inference_summary.json",
        local_run_dir=tmp_path, local_typing_dir=tmp_path / "typing",
        project_uid="P1", workspace_uid="W1", particle_exclusion_distance_angstrom=100,
        typing_normalization_method=None, typing_min_component_area_px=None,
        typing_pixel_size_angstrom=None, typing_timeout=None, resource_env={},
    )


def test_failed_live_typing_is_retried_and_close_stops_future_work(tmp_path, monkeypatch):
    updater = _updater(tmp_path)
    monkeypatch.setattr(predict, "write_typing_manifest_from_transfer", lambda **kwargs: {"images": 1})
    calls = []

    def type_images(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("transient incomplete mask")

    monkeypatch.setattr(predict, "run_local_typing", type_images)
    monkeypatch.setattr(updater, "_refresh_available_otfs", lambda: 1)
    updater()
    updater.close()
    assert updater.last_images == 0
    updater.last_attempt = 0
    updater()
    updater.close(cancel=True)
    assert updater.last_images == 1
    updater.last_attempt = 0
    updater()
    assert len(calls) == 2


def test_interrupt_while_waiting_for_live_typing_cancels_and_waits_for_cleanup(tmp_path, monkeypatch):
    updater = _updater(tmp_path)
    started = threading.Event()

    def update():
        started.set()
        assert updater._cancel.wait(timeout=5)

    monkeypatch.setattr(updater, "_update", update)
    original_wait = updater._finished.wait
    interrupted = False

    def wait(timeout=None):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_wait(timeout=5)

    monkeypatch.setattr(updater._finished, "wait", wait)
    updater()
    assert started.wait(timeout=5)
    with pytest.raises(KeyboardInterrupt):
        updater.close()
    assert updater._cancel.is_set()
    assert updater._finished.is_set()


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker-group cleanup")
def test_cancel_stops_typing_parent_and_descendant(tmp_path):
    pid_file = tmp_path / "pids.json"
    source = (
        "import os, subprocess, sys, time, json; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"open({str(pid_file)!r}, 'w').write(json.dumps([os.getpid(), child.pid])); "
        "time.sleep(30)"
    )
    cancel = threading.Event()

    def poll():
        if pid_file.exists():
            cancel.set()

    with pytest.raises(RuntimeError, match="canceled"):
        predict._run_polled_command([sys.executable, "-c", source], env=os.environ.copy(), timeout=10,
                                    poll_callback=poll, poll_interval=0.5, cancel_event=cancel)
    for pid in json.loads(pid_file.read_text()):
        status = Path(f"/proc/{pid}/status")
        if status.exists():
            assert any(line.startswith("State:") and "Z" in line for line in status.read_text().splitlines())


def test_typing_timeout_is_propagated():
    with pytest.raises(subprocess.TimeoutExpired):
        predict._run_polled_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                    env=os.environ.copy(), timeout=0.1, poll_callback=None, poll_interval=1)


def test_worker_budget_respects_cpu_allocation(monkeypatch):
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _: set(range(32)), raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "8"}, None, live=True) == 4
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "8"}, 20, live=True) == 4
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "8"}, 6) == 6
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "1"}, None) == 1
    with pytest.raises(ValueError, match="typing-workers"):
        cli._validate_typing_options(argparse.Namespace(typing_workers=0))


def test_auto_typing_workers_scale_with_cpu_budget_instead_of_fixed_cap(monkeypatch):
    # A large CryoSPARC CPU allocation should let typing use far more than the old
    # hardcoded 4-worker cap when --typing-workers is left unset.
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "24"}, None, live=False) == 24
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "24"}, None, live=True) == 12
    # An explicit request is still capped by the available budget, live or not.
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "24"}, 100, live=False) == 24
    assert cli._typing_workers({"CRYOFILTER_NUM_CPUS": "24"}, 100, live=True) == 12


def test_default_num_cpus_leaves_two_cores_free(monkeypatch):
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _: set(range(24)), raising=False)
    monkeypatch.delenv("CRYOFILTER_NUM_CPUS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

    args = argparse.Namespace(num_cpus=None)
    cli._default_num_cpus_if_unset(args)
    assert args.num_cpus == 22

    # An explicit value is left untouched.
    args_explicit = argparse.Namespace(num_cpus=8)
    cli._default_num_cpus_if_unset(args_explicit)
    assert args_explicit.num_cpus == 8

    # The reserve never drives the budget below one CPU.
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _: {0}, raising=False)
    args_single = argparse.Namespace(num_cpus=None)
    cli._default_num_cpus_if_unset(args_single)
    assert args_single.num_cpus == 1
