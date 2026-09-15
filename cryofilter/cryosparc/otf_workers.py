"""Persistent OTF model workers. Queues carry paths/metadata, never image arrays."""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path


def _worker(role, tasks, results, stop, options):
    # Set before importing numerical libraries (also set before spawn by WorkerPool).
    for key, value in options["env"].items():
        os.environ[key] = value
    import torch
    torch.set_num_threads(options["cpus"])
    current = None
    started = 0.0

    def receive():
        nonlocal current, started
        while not stop.is_set():
            try:
                current = tasks.get(timeout=0.5)
            except queue.Empty:
                continue
            if current is None:
                return
            started = time.monotonic()
            yield current

    try:
        if role == "segmentation":
            from cryofilter.cli import _build_parser, _run_infer
            args = _build_parser().parse_args([
                "infer", "--input", options["output"], "--output-dir", options["output"],
                "--checkpoint", options["checkpoint"], "--device", options["device"],
                "--threshold", str(options["threshold"]),
                "--inference-profile", options["profile"],
                "--batch-forward-size", str(options["batch_size"]),
            ])

            def inputs():
                for entry in receive():
                    yield {"path": entry["path"], "stem": entry["key"],
                           "pixel_size_angstrom": entry["pixel_size_angstrom"]}

            def complete(record):
                results.put((role, current, record, time.monotonic() - started, None))

            _run_infer(args, finalize=False, input_stream=inputs(), on_result=complete)
        else:
            import numpy as np
            from cryofilter.cli import _load_mrc_2d
            from cryofilter.publication_typing import PublicationClassifier
            from cryofilter.publication_runner import _summaries
            from cryofilter.typing_cli import _save_npy_atomic
            from cryofilter.cryosparc.otf_state import load_mask
            classifier = PublicationClassifier(
                checkpoint=options["typing_checkpoint"], device=options["device"],
                batch_size=32, cpu_workers=options["cpus"],
            )
            classifier.kwargs["sample_stride_px"] = options["typing_stride"]
            for entry in receive():
                image, _ = _load_mrc_2d(Path(entry["path"]))
                mask = load_mask(entry["mask_path"]).astype(bool)
                prediction = classifier.predict(image, mask, entry["pixel_size_angstrom"])
                path = Path(options["output"]) / (entry["key"] + "_typed_mask.npy")
                _save_npy_atomic(path, prediction["pred_map"].astype(np.uint8))
                summary, _ = _summaries({"dataset_id": "otf", "stem": entry["uid"]}, mask, prediction)
                results.put((role, entry, {"path": str(path), "summary": summary},
                             time.monotonic() - started, None))
    except BaseException as exc:
        if not stop.is_set():
            traceback.print_exc()
            results.put((role, current, None, 0.0, f"{type(exc).__name__}: {exc}"))
        raise


def allocate_devices(devices: list[str], typing: bool):
    """Device IDs refer to the visible allocation, not an assumption of idle GPUs."""
    if not devices or len(set(devices)) != len(devices) or any(not d.strip() for d in devices):
        raise ValueError("Select at least one distinct GPU device")
    if typing and len(devices) < 2:
        raise ValueError("OTF typing requires at least 2 allocated GPUs; 1 GPU supports segmentation only")
    cut = len(devices) // 2 if typing else len(devices)
    return devices[:cut], devices[cut:]


class WorkerPool:
    def __init__(self, devices, typing, cpus, options):
        segmentation, type_devices = allocate_devices(devices, typing)
        if cpus < len(devices):
            raise ValueError("Allocate at least one CPU per GPU worker")
        ctx = mp.get_context("spawn")
        self.stop = ctx.Event()
        self.results = ctx.Queue()
        self.tasks = {"segmentation": ctx.Queue(maxsize=2 * len(segmentation))}
        if type_devices:
            self.tasks["typing"] = ctx.Queue(maxsize=2 * len(type_devices))
        self.processes = []
        self.capacity = {"segmentation": 2 * len(segmentation), "typing": 2 * len(type_devices)}
        try:
            for index, device in enumerate(devices):
                role = "segmentation" if index < len(segmentation) else "typing"
                threads = cpus // len(devices) + (index < cpus % len(devices))
                env = {key: str(threads) for key in (
                    "CRYOFILTER_NUM_CPUS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
                env["CUDA_VISIBLE_DEVICES"] = device
                env["CRYOFILTER_NUM_GPUS"] = "1"
                opts = dict(options, cpus=threads, env=env, device="cuda:0")
                opts["output"] = str(Path(options["output"]) / role)
                Path(opts["output"]).mkdir(parents=True, exist_ok=True)
                process = ctx.Process(target=_worker, args=(role, self.tasks[role], self.results, self.stop, opts))
                old = {key: os.environ.get(key) for key in env}
                try:
                    os.environ.update(env)
                    process.start()
                finally:
                    for key, value in old.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def submit(self, role, entry):
        try:
            self.tasks[role].put_nowait(entry)
            return True
        except queue.Full:
            return False

    def collect(self):
        while True:
            try:
                yield self.results.get_nowait()
            except queue.Empty:
                return

    def check(self):
        for process in self.processes:
            if process.exitcode is not None:
                raise RuntimeError(f"OTF GPU worker exited unexpectedly (code {process.exitcode}); resume the run after correcting the error")

    def close(self):
        self.stop.set()
        deadline = time.monotonic() + 5
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for channel in [*self.tasks.values(), self.results]:
            channel.cancel_join_thread()
            channel.close()
