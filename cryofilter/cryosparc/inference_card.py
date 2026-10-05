"""Publish batch inference as a live, reusable CryoSPARC mask job."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bridge.compat import (
    add_exposure_output,
    dataset_filter_prefixes,
    dataset_prefixes,
    dataset_take,
    dataset_uids,
    find_external_job,
    find_job,
    find_project,
    object_uid,
    save_output,
    start_external_job,
    stop_external_job,
)
from .otf import _connect, _publish_previews
from .otf_source import mrc_geometry
from .otf_state import (
    CARD_FILE,
    MASK_CARD_FORMAT,
    Index,
    file_stamp,
    generation_key,
    load_mask,
    write_json,
)
from .protocol.manifests import read_json_model
from .protocol.models import TransferManifest
from .protocol.validation import validate_project_uid, validate_workspace_uid
from .remote.predict import run_local_typing, write_typing_manifest_from_transfer


def _manifest_entries(
    transfer: TransferManifest,
    *,
    local_transfer_dir: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in transfer.micrographs:
        path = (local_transfer_dir / item.transfer_filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Staged CryoSPARC micrograph is missing: {path}")
        shape = list(item.shape_yx) if item.shape_yx is not None else None
        pixel_size = item.pixel_size_angstrom
        if shape is None or pixel_size is None:
            header_shape, header_pixel_size = mrc_geometry(path)
            shape = shape or header_shape
            pixel_size = pixel_size or header_pixel_size
        stamp = file_stamp(path)
        uid = str(int(item.uid))
        entries.append(
            {
                "uid": uid,
                "key": generation_key(uid, stamp),
                "path": str(path),
                "source_path": item.source_path,
                "source_relative_path": item.source_relative_path,
                "stamp": stamp,
                "shape": [int(shape[0]), int(shape[1])],
                "pixel_size_angstrom": float(pixel_size),
            }
        )
    return entries


def _summary_paths(output_dir: Path) -> list[Path]:
    paths = [output_dir / "inference_summary.json"]
    paths.extend(sorted(output_dir.glob(".cryofilter_worker_*_summary.json")))
    return paths


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    rows = payload.get("inputs") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _complete_record(entry: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any] | None:
    mask_value = raw.get("output_mask_npy")
    probability_value = raw.get("output_prob_npy")
    if not mask_value or not probability_value:
        return None
    mask_path = Path(str(mask_value)).expanduser().resolve()
    probability_path = Path(str(probability_value)).expanduser().resolve()
    if not mask_path.is_file() or not probability_path.is_file():
        return None

    record = dict(raw)
    mask = load_mask(mask_path)
    record.setdefault("input_image_shape", list(entry["shape"]))
    record.setdefault("output_image_shape", list(mask.shape))
    postprocessing = dict(record.get("mask_postprocessing") or {})
    postprocessing.setdefault("final_mask_pixels", int(np.count_nonzero(mask)))
    record["mask_postprocessing"] = postprocessing
    record["output_mask_npy"] = str(mask_path)
    record["output_prob_npy"] = str(probability_path)
    record["output_stamps"] = [file_stamp(mask_path), file_stamp(probability_path)]
    return record


def _sync_index(
    index: Index,
    *,
    entries_by_path: dict[Path, dict[str, Any]],
    output_dir: Path,
    completed_uids: list[str],
    summary_paths: Iterable[Path] | None = None,
    ignored_summary_stamps: dict[Path, object] | None = None,
) -> int:
    known = set(completed_uids)
    added = 0
    for summary_path in summary_paths or _summary_paths(output_dir):
        if (
            ignored_summary_stamps
            and summary_path in ignored_summary_stamps
            and summary_path.is_file()
            and file_stamp(summary_path) == ignored_summary_stamps[summary_path]
        ):
            continue
        for raw in _load_summary_rows(summary_path):
            input_value = raw.get("input_mrc")
            if not input_value:
                continue
            entry = entries_by_path.get(Path(str(input_value)).expanduser().resolve())
            if entry is None or entry["uid"] in known:
                continue
            record = _complete_record(entry, raw)
            if record is None:
                continue
            if index.complete("segmentation", entry, record):
                completed_uids.append(entry["uid"])
                known.add(entry["uid"])
                added += 1
    if added:
        index.set_meta("preview_uids", completed_uids[-10:])
    return added


def _resolved_summary_path(value: object, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sync_typing_index(
    index: Index,
    *,
    entries_by_stem: dict[str, dict[str, Any]],
    typing_dir: Path,
    typed_uids: list[str],
) -> int:
    summary_path = typing_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return 0
    if not isinstance(summary, dict):
        return 0
    csv_path = _resolved_summary_path(
        summary.get("image_contamination_summary_csv"),
        base=summary_path.parent,
    )
    typed_dir = _resolved_summary_path(summary.get("typed_mask_dir"), base=summary_path.parent)
    if csv_path is None or typed_dir is None:
        return 0
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return 0

    known = set(typed_uids)
    added = 0
    area_keys = ("carbon", "crystalline", "aggregate", "ethane")
    for raw in rows:
        entry = entries_by_stem.get(str(raw.get("stem") or ""))
        if entry is None or entry["uid"] in known:
            continue
        dataset_id = str(raw.get("dataset_id") or "").strip()
        stem = str(raw.get("stem") or "").strip()
        if not dataset_id or not stem:
            continue
        typed_path = typed_dir / f"{dataset_id}__{stem}_typed_mask.npy"
        if not typed_path.is_file():
            continue
        typed_summary = {
            f"{key}_area_px": int(float(raw.get(f"{key}_area_px") or 0))
            for key in area_keys
        }
        record = {
            "path": str(typed_path.resolve()),
            "output_stamp": file_stamp(typed_path),
            "summary": typed_summary,
        }
        if index.complete("typing", entry, record):
            typed_uids.append(entry["uid"])
            known.add(entry["uid"])
            added += 1
    return added


def _selected_source_output(
    client: object,
    project: object,
    transfer: TransferManifest,
) -> tuple[object, list[str]]:
    source_job = find_job(
        client,
        project,
        transfer.project_uid,
        transfer.micrographs_ref.job_uid,
    )
    source = source_job.load_output(transfer.micrographs_ref.output_name)
    source_uids = [int(value) for value in dataset_uids(source)]
    positions = {uid: index for index, uid in enumerate(source_uids)}
    requested = [int(entry.uid) for entry in transfer.micrographs]
    missing = [uid for uid in requested if uid not in positions]
    if missing:
        raise ValueError(
            "Could not map staged micrographs back to the CryoSPARC source output; "
            f"missing UIDs: {missing[:10]}"
        )
    selected = dataset_take(source, [positions[uid] for uid in requested])
    slots = dataset_prefixes(source)
    preferred = [
        str(entry.blob_prefix)
        for entry in transfer.micrographs
        if entry.blob_prefix and str(entry.blob_prefix) in slots
    ]
    preferred.extend(
        slot
        for slot in slots
        if any(token in slot.lower() for token in ("micrograph", "exposure", "movie", "blob"))
    )
    output_slots = list(dict.fromkeys(preferred))[:1]
    if not output_slots:
        if not slots:
            raise ValueError("CryoSPARC micrograph output has no result-group slots")
        output_slots = [slots[0]]
    return selected, output_slots


def _preview_rows(
    index: Index,
    completed_uids: Iterable[str],
) -> list[tuple[dict, dict, dict | None]]:
    records = {entry["uid"]: (entry, inference, typed) for entry, inference, typed in index.rows()}
    return [
        records[uid]
        for uid in reversed(list(completed_uids)[-10:])
        if uid in records and records[uid][1] is not None
    ]


def _infer_option(infer_args: list[str], name: str) -> str | None:
    for index, token in enumerate(infer_args):
        if token == name and index + 1 < len(infer_args):
            return str(infer_args[index + 1])
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _typing_worker_count(args, infer_args: list[str], *, live: bool = False) -> int:
    infer_cpus = _infer_option(infer_args, "--num-cpus")
    available = int(infer_cpus) if infer_cpus is not None else (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    budget = max(1, available // 2) if live else max(1, available - 2)
    requested = getattr(args, "typing_workers", None)
    if requested is not None:
        if int(requested) < 1:
            raise ValueError("--typing-workers must be at least 1")
        return min(budget, int(requested))
    return budget


def _set_infer_option(infer_args: list[str], name: str, value: object) -> list[str]:
    """Return infer arguments with one scalar option replaced or appended."""

    updated: list[str] = []
    index = 0
    while index < len(infer_args):
        token = infer_args[index]
        if token == name:
            index += 2
            continue
        if token.startswith(name + "="):
            index += 1
            continue
        updated.append(token)
        index += 1
    updated.extend([name, str(value)])
    return updated


def _requested_cuda_devices(infer_args: list[str]) -> list[int]:
    device_value = (_infer_option(infer_args, "--device") or "cuda").lower()
    if device_value == "cpu":
        return []
    raw_count = _infer_option(infer_args, "--num-gpus")
    count = int(raw_count) if raw_count is not None else None
    raw_gpus = _infer_option(infer_args, "--gpus")
    if raw_gpus and raw_gpus.lower() != "all":
        devices: list[int] = []
        for token in raw_gpus.split(","):
            value = token.strip().lower().removeprefix("cuda:")
            if value.isdigit() and int(value) not in devices:
                devices.append(int(value))
        return devices if count is None else devices[:count]
    if count is not None:
        return list(range(max(0, count)))
    if device_value.startswith("cuda:") and device_value.split(":", 1)[1].isdigit():
        return [int(device_value.split(":", 1)[1])]
    return [0]


def _reserve_live_typing_gpu(infer_args: list[str]) -> tuple[list[str], list[int]]:
    """Match OTF allocation by reserving half the requested GPUs for typing."""

    devices = _requested_cuda_devices(infer_args)
    if len(devices) < 2:
        return list(infer_args), []
    segmentation_count = len(devices) // 2
    segmentation_devices = devices[:segmentation_count]
    typing_devices = devices[segmentation_count:]
    updated = _set_infer_option(
        infer_args,
        "--gpus",
        ",".join(map(str, segmentation_devices)),
    )
    updated = _set_infer_option(updated, "--num-gpus", len(segmentation_devices))
    return updated, typing_devices


def _typing_device_and_batch(
    infer_args: list[str],
    *,
    candidate_devices: Iterable[int] | None = None,
) -> tuple[str, int, int | None]:
    """Pick the least-loaded inference GPU without retaining a CUDA context."""

    device_value = (_infer_option(infer_args, "--device") or "cuda").lower()
    if device_value == "cpu":
        return "cpu", 32, None

    candidates_requested = (
        list(candidate_devices)
        if candidate_devices is not None
        else _requested_cuda_devices(infer_args)
    )
    requested = {int(value) for value in candidates_requested} or None

    probe = (
        "import json, torch; print(json.dumps(["
        "{'index': i, 'free': torch.cuda.mem_get_info(i)[0], "
        "'total': torch.cuda.mem_get_info(i)[1]} "
        "for i in range(torch.cuda.device_count())]))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        rows = json.loads(completed.stdout.strip().splitlines()[-1])
        candidates = [
            row for row in rows
            if requested is None or int(row["index"]) in requested
        ]
        selected = max(candidates, key=lambda row: int(row["free"]))
        free_bytes = int(selected["free"])
        free_gib = free_bytes / 1024**3
        if free_gib >= 10:
            batch_size = 32
        elif free_gib >= 6:
            batch_size = 16
        elif free_gib >= 4:
            batch_size = 8
        elif free_gib >= 2.5:
            batch_size = 4
        else:
            batch_size = 1
        return f"cuda:{int(selected['index'])}", batch_size, free_bytes
    except (IndexError, KeyError, OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        fallback = min(requested) if requested else 0
        return f"cuda:{fallback}", 32, None


def _log_progress(external: object, index: Index, message: str) -> None:
    try:
        event_id = external.log(message, id=index.get_meta("progress_event_id"))
    except TypeError:
        event_id = external.log(message)
    if event_id:
        index.set_meta("progress_event_id", str(event_id))


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_inference_card(args, config) -> int:
    project_uid = validate_project_uid(args.project)
    workspace_uid = validate_workspace_uid(args.workspace)
    local_run_dir = Path(args.local_run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    transfer_path = local_run_dir / "transfer_manifest.json"
    transfer_dir = local_run_dir / "transfer"
    transfer = read_json_model(TransferManifest, transfer_path)
    if transfer.project_uid != project_uid or transfer.workspace_uid != workspace_uid:
        raise ValueError("Inference card project/workspace does not match the staging manifest")
    if not transfer.external_job_uid:
        raise ValueError("Staging did not create a CryoSPARC External Job")

    infer_args = list(args.infer_args or [])
    if infer_args and infer_args[0] == "--":
        infer_args = infer_args[1:]
    if not infer_args:
        raise ValueError("Inference card requires cryofilter infer arguments after --")
    run_typing = bool(getattr(args, "run_typing", False))
    original_infer_args = list(infer_args)
    live_typing_devices: list[int] = []
    live_typing_workers: int | None = None
    if run_typing:
        infer_args, live_typing_devices = _reserve_live_typing_gpu(infer_args)
        if live_typing_devices:
            live_typing_workers = _typing_worker_count(args, original_infer_args, live=True)
            # A two-GPU typed run becomes one segmentation worker plus one
            # typing worker. Give the current segmentation run a known live
            # summary so the card never consumes stale worker summaries left
            # by an earlier run in the same output directory.
            infer_args = _set_infer_option(
                infer_args,
                "--live-summary-file",
                local_run_dir / "inference_live_summary.json",
            )
            raw_cpus = _infer_option(original_infer_args, "--num-cpus")
            if raw_cpus is not None:
                infer_args = _set_infer_option(
                    infer_args,
                    "--num-cpus",
                    max(1, int(raw_cpus) - live_typing_workers),
                )
    command = [sys.executable, "-m", "cryofilter.cli", "infer", *infer_args]
    segmentation_devices = _requested_cuda_devices(infer_args)
    live_summary_value = _infer_option(infer_args, "--live-summary-file")
    if len(segmentation_devices) > 1:
        current_summary_paths = [
            output_dir / f".cryofilter_worker_{index:02d}_summary.json"
            for index in range(len(segmentation_devices))
        ]
    elif live_summary_value:
        current_summary_paths = [Path(live_summary_value).expanduser().resolve()]
    else:
        current_summary_paths = [output_dir / "inference_summary.json"]

    client = _connect(config)
    project = find_project(client, project_uid)
    external = find_external_job(client, project, project_uid, transfer.external_job_uid)
    external_uid = str(object_uid(external) or transfer.external_job_uid)
    raw_job_dir = getattr(external, "dir", None)
    if callable(raw_job_dir):
        raw_job_dir = raw_job_dir()
    job_dir = Path(str(raw_job_dir)).expanduser() if raw_job_dir else None
    if job_dir is None or not job_dir.is_dir():
        from .bridge.prepare import _object_dir

        resolved_job_dir = _object_dir(external, "job_dir")
        if resolved_job_dir is None or not resolved_job_dir.is_dir():
            raise ValueError("CryoSPARC External Job directory is not accessible on this machine")
        job_dir = resolved_job_dir

    entries = _manifest_entries(transfer, local_transfer_dir=transfer_dir)
    entries_by_path = {Path(entry["path"]).resolve(): entry for entry in entries}
    entries_by_stem = {Path(entry["path"]).stem: entry for entry in entries}
    entries_by_uid = {entry["uid"]: entry for entry in entries}
    index = Index(local_run_dir / "index.sqlite")
    process: subprocess.Popen | None = None
    preview_dir = local_run_dir / "card_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    completed_uids: list[str] = []
    typed_uids: list[str] = []
    try:
        for entry in entries:
            index.discover(entry)
        index.set_meta("card_kind", "batch_inference")
        index.set_meta(
            "status",
            {
                "phase": "staging",
                "expected": len(entries),
                "segmented": 0,
                "typed": 0,
                "segmentation_pending": len(entries),
                "typing_pending": 0,
                "run_typing": run_typing,
                "job_uid": external_uid,
                "updated_at": time.time(),
            },
        )
        index.set_meta("preview_uids", [])

        card = {
            "format": MASK_CARD_FORMAT,
            "kind": "batch_inference",
            "project": project_uid,
            "workspace": workspace_uid,
            "job_uid": external_uid,
            "source_job": transfer.micrographs_ref.job_uid,
            "source_output": transfer.micrographs_ref.output_name,
            "index_path": str(index.path),
            "run_dir": str(local_run_dir),
            "output_dir": str(output_dir),
            "run_id": str(transfer.run_id),
        }
        card_path = job_dir / CARD_FILE
        if card_path.exists():
            raise FileExistsError(f"cryoFILTER card metadata already exists: {card_path}")
        write_json(card_path, card)
        write_json(local_run_dir / CARD_FILE, card)

        source_output, output_slots = _selected_source_output(client, project, transfer)
        add_exposure_output(
            external,
            name="micrographs",
            passthrough="input_micrographs",
            slots=output_slots,
            title="cryoFILTER processed micrographs",
            alloc=source_output,
        )

        start_external_job(external, status="running")
        print(
            f"cryoFILTER inference card: {project_uid}/{external_uid}; "
            f"source {transfer.micrographs_ref.job_uid}:{transfer.micrographs_ref.output_name}",
            flush=True,
        )
        print(f"Mask index: {index.path}", flush=True)
        _log_progress(external, index, f"Inference starting: 0/{len(entries)} micrographs complete.")
        index.set_meta(
            "status",
            {
                "phase": "processing",
                "expected": len(entries),
                "segmented": 0,
                "typed": 0,
                "segmentation_pending": len(entries),
                "typing_pending": 0,
                "run_typing": run_typing,
                "job_uid": external_uid,
                "updated_at": time.time(),
            },
        )

        typing_manifest_path = local_run_dir / "contamination_typing_manifest.csv"
        typing_dir = output_dir / "typing"
        expected_typed = len(entries)
        checkpoint_value = _infer_option(original_infer_args, "--checkpoint")
        typing_device: str | None = None
        typing_batch_size: int | None = None
        next_typing_preview = 0.0
        typing_preview_pending = False
        live_manifest_images = 0

        if run_typing and live_typing_devices:
            typing_device, typing_batch_size, free_bytes = _typing_device_and_batch(
                original_infer_args,
                candidate_devices=live_typing_devices,
            )
            free_text = (
                "free-memory probe unavailable"
                if free_bytes is None
                else f"{free_bytes / 1024**3:.1f} GiB free"
            )
            print(
                "Live typing allocation: segmentation GPU(s) "
                f"{','.join(map(str, segmentation_devices))}; typing {typing_device} "
                f"({free_text}, batch {typing_batch_size}, {live_typing_workers} CPU workers).",
                flush=True,
            )
        elif run_typing:
            print(
                "Live typed previews require at least 2 requested GPUs; typing will run "
                "uncontended after segmentation.",
                flush=True,
            )

        def refresh_typing(*, phase: str) -> None:
            nonlocal next_typing_preview, typing_preview_pending
            if not run_typing:
                return
            changed = _sync_typing_index(
                index,
                entries_by_stem=entries_by_stem,
                typing_dir=typing_dir,
                typed_uids=typed_uids,
            )
            if not changed and not typing_preview_pending:
                return
            typing_target = expected_typed if phase == "typing" else len(completed_uids)
            typing_message = (
                f"Inference typing: {len(typed_uids)}/{expected_typed} micrographs typed; "
                f"{len(completed_uids)} segmented."
            )
            if changed:
                typing_preview_pending = True
                print(typing_message, flush=True)
                _log_progress(external, index, typing_message)
                index.set_meta(
                    "status",
                    {
                        "phase": phase,
                        "expected": len(entries),
                        "segmented": len(completed_uids),
                        "typed": len(typed_uids),
                        "segmentation_pending": len(entries) - len(completed_uids),
                        "typing_pending": max(0, typing_target - len(typed_uids)),
                        "run_typing": True,
                        "job_uid": external_uid,
                        "updated_at": time.time(),
                    },
                )
            now = time.monotonic()
            if now < next_typing_preview:
                return
            result = _publish_previews(
                _preview_rows(index, typed_uids),
                preview_dir,
                external,
                progress_message=typing_message,
                output_name="micrographs",
            )
            if result and result.get("progress_event_id"):
                index.set_meta("progress_event_id", result["progress_event_id"])
            typing_preview_pending = False
            next_typing_preview = now + 15

        def run_live_typing() -> None:
            nonlocal live_manifest_images
            if typing_device is None or typing_batch_size is None:
                return
            try:
                typing_manifest = write_typing_manifest_from_transfer(
                    transfer_manifest_file=transfer_path,
                    local_transfer_dir=transfer_dir,
                    inference_dir=output_dir,
                    manifest_path=typing_manifest_path,
                    dataset_id=f"{project_uid}_{workspace_uid}",
                    require_all_masks=False,
                    include_stems=[
                        Path(entries_by_uid[uid]["path"]).stem
                        for uid in completed_uids
                    ],
                )
            except ValueError as exc:
                if "No rows were written" in str(exc):
                    return
                raise
            images = int(typing_manifest.get("images") or 0)
            if images <= live_manifest_images:
                refresh_typing(phase="processing")
                return
            print(
                f"Live typing update: {images}/{expected_typed} segmented micrograph(s) ready.",
                flush=True,
            )
            run_local_typing(
                manifest_path=typing_manifest_path,
                output_dir=typing_dir,
                expected_images=expected_typed,
                workers=int(live_typing_workers or 1),
                incremental=True,
                poll_callback=lambda: refresh_typing(phase="processing"),
                poll_interval=max(1.0, float(args.poll_seconds)),
                weights_dir=(
                    Path(checkpoint_value).expanduser().resolve().parent
                    if checkpoint_value
                    else None
                ),
                typing_device=typing_device,
                sample_stride_px=int(getattr(args, "typing_sample_stride_px", 64)),
                typing_batch_size=typing_batch_size,
            )
            live_manifest_images = images
            refresh_typing(phase="processing")

        ignored_summary_stamps = {
            path: file_stamp(path)
            for path in current_summary_paths
            if path.is_file()
        }
        process = subprocess.Popen(command, start_new_session=True)
        next_log = 0.0
        next_preview = 0.0
        next_live_typing = 0.0
        while process.poll() is None:
            changed = _sync_index(
                index,
                entries_by_path=entries_by_path,
                output_dir=output_dir,
                completed_uids=completed_uids,
                summary_paths=current_summary_paths,
                ignored_summary_stamps=ignored_summary_stamps,
            )
            now = time.monotonic()
            if changed:
                index.set_meta(
                    "status",
                    {
                        "phase": "processing",
                        "expected": len(entries),
                        "segmented": len(completed_uids),
                        "typed": len(typed_uids),
                        "segmentation_pending": len(entries) - len(completed_uids),
                        "typing_pending": max(0, len(completed_uids) - len(typed_uids)),
                        "run_typing": run_typing,
                        "job_uid": external_uid,
                        "updated_at": time.time(),
                    },
                )
            message = f"Inference processing: {len(completed_uids)}/{len(entries)} micrographs complete."
            if run_typing and live_typing_devices:
                message = (
                    f"Inference processing: {len(completed_uids)}/{len(entries)} segmented; "
                    f"{len(typed_uids)}/{len(entries)} typed."
                )
            if (changed and now >= next_log) or now >= next_log + 30:
                print(message, flush=True)
                _log_progress(external, index, message)
                next_log = now
            if run_typing and live_typing_devices and changed and now >= next_live_typing:
                try:
                    run_live_typing()
                except Exception as exc:
                    print(
                        "Warning: live typing update failed; final typing will retry: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                next_live_typing = time.monotonic() + 15
            if not run_typing and changed and now >= next_preview:
                try:
                    result = _publish_previews(
                        _preview_rows(index, completed_uids),
                        preview_dir,
                        external,
                        progress_message=message,
                        output_name="micrographs",
                    )
                    if result and result.get("progress_event_id"):
                        index.set_meta("progress_event_id", result["progress_event_id"])
                except Exception as exc:
                    print(f"Inference card preview unavailable: {type(exc).__name__}: {exc}", flush=True)
                next_preview = now + 15
            time.sleep(max(0.2, float(args.poll_seconds)))

        returncode = process.wait()
        _sync_index(
            index,
            entries_by_path=entries_by_path,
            output_dir=output_dir,
            completed_uids=completed_uids,
            summary_paths=current_summary_paths,
            ignored_summary_stamps=ignored_summary_stamps,
        )
        if returncode != 0:
            raise RuntimeError(f"cryoFILTER inference exited with code {returncode}")
        if len(completed_uids) != len(entries):
            raise RuntimeError(
                f"Inference completed, but the mask card indexed {len(completed_uids)}/"
                f"{len(entries)} micrographs"
            )

        typing_summary_path: Path | None = None
        if run_typing:
            typing_manifest = write_typing_manifest_from_transfer(
                transfer_manifest_file=transfer_path,
                local_transfer_dir=transfer_dir,
                inference_dir=output_dir,
                manifest_path=typing_manifest_path,
                dataset_id=f"{project_uid}_{workspace_uid}",
                require_all_masks=True,
            )
            expected_typed = int(
                typing_manifest.get("images_expected")
                or typing_manifest.get("images")
                or len(entries)
            )
            message = (
                f"Inference typing: {len(typed_uids)}/{expected_typed} micrographs typed; "
                f"{len(completed_uids)} segmented."
            )
            print("Phase: final typing/aggregation (segmentation complete).", flush=True)
            print(message, flush=True)
            _log_progress(external, index, message)
            index.set_meta(
                "status",
                {
                    "phase": "typing",
                    "expected": len(entries),
                    "segmented": len(completed_uids),
                    "typed": len(typed_uids),
                    "segmentation_pending": 0,
                    "typing_pending": expected_typed - len(typed_uids),
                    "run_typing": True,
                    "job_uid": external_uid,
                    "updated_at": time.time(),
                },
            )
            next_typing_preview = 0.0
            if typing_device is None or typing_batch_size is None:
                typing_device, typing_batch_size, free_bytes = _typing_device_and_batch(
                    original_infer_args
                )
                if free_bytes is None:
                    print(
                        f"Typing GPU selection: {typing_device}; using batch {typing_batch_size} "
                        "(free-memory probe unavailable).",
                        flush=True,
                    )
                else:
                    print(
                        f"Typing GPU selection: {typing_device} with "
                        f"{free_bytes / 1024**3:.1f} GiB free; using batch "
                        f"{typing_batch_size}.",
                        flush=True,
                    )
            run_local_typing(
                manifest_path=typing_manifest_path,
                output_dir=typing_dir,
                expected_images=expected_typed,
                workers=_typing_worker_count(args, original_infer_args),
                incremental=True,
                poll_callback=lambda: refresh_typing(phase="typing"),
                poll_interval=max(1.0, float(args.poll_seconds)),
                weights_dir=(
                    Path(checkpoint_value).expanduser().resolve().parent
                    if checkpoint_value
                    else None
                ),
                typing_device=typing_device,
                sample_stride_px=int(getattr(args, "typing_sample_stride_px", 64)),
                typing_batch_size=typing_batch_size,
            )
            refresh_typing(phase="typing")
            if len(typed_uids) != expected_typed:
                raise RuntimeError(
                    f"Typing completed, but the mask card indexed {len(typed_uids)}/"
                    f"{expected_typed} typed micrographs"
                )
            typing_summary_path = typing_dir / "summary.json"
            if not typing_summary_path.is_file():
                raise FileNotFoundError(f"Typing summary was not written: {typing_summary_path}")

        save_output(
            external,
            "micrographs",
            dataset_filter_prefixes(source_output, output_slots),
        )
        status = {
            "phase": "completed",
            "expected": len(entries),
            "segmented": len(completed_uids),
            "typed": len(typed_uids),
            "segmentation_pending": 0,
            "typing_pending": 0,
            "run_typing": run_typing,
            "job_uid": external_uid,
            "updated_at": time.time(),
        }
        index.set_meta("status", status)
        message = f"Inference complete: {len(completed_uids)}/{len(entries)} micrographs segmented"
        if run_typing:
            message += f", {len(typed_uids)}/{len(entries)} typed"
        message += "."
        try:
            result = _publish_previews(
                _preview_rows(index, typed_uids if run_typing else completed_uids),
                preview_dir,
                external,
                progress_message=message,
                output_name="micrographs",
            )
            if result and result.get("progress_event_id"):
                index.set_meta("progress_event_id", result["progress_event_id"])
        except Exception as exc:
            print(f"Final inference card preview unavailable: {type(exc).__name__}: {exc}", flush=True)
        _log_progress(external, index, message)
        stop_external_job(external)
        print(
            f"CryoSPARC output: {project_uid}/{external_uid}:micrographs; "
            "use this job as the mask source in Filter Picks.",
            flush=True,
        )
        if typing_summary_path is not None:
            print(f"Typing summary: {typing_summary_path}", flush=True)
        return 0
    except BaseException as exc:
        if process is not None:
            _terminate_process(process)
        status = index.get_meta("status", {})
        status.update(
            phase="stopped" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error=str(exc),
            segmented=len(completed_uids),
            typed=len(typed_uids),
            updated_at=time.time(),
        )
        index.set_meta("status", status)
        try:
            stop_external_job(
                external,
                error=f"Inference interrupted: {type(exc).__name__}: {exc}",
            )
        except Exception as report_error:
            print(f"Could not update inference card status: {report_error}", flush=True)
        raise
    finally:
        if process is not None:
            _terminate_process(process)
        index.close()


__all__ = ["run_inference_card"]
