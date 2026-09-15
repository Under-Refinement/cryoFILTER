"""On-the-fly CryoSPARC integration for a shared project filesystem."""
from __future__ import annotations

import fcntl
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

from .bridge.client import make_client, server_version
from .bridge.compat import create_external_job, find_job, find_project, object_uid, stop_external_job
from .bridge.prepare import _object_dir
from .otf_source import PatchSource, TERMINAL
from .otf_state import CARD_FILE, FORMAT, Index, file_stamp, load_mask, read_card, write_json
from .otf_workers import WorkerPool, allocate_devices
from .protocol.validation import validate_job_uid, validate_project_uid, validate_workspace_uid


def _connect(config):
    from .remote.cli import _bridge_environment
    # This command is intentionally local to the shared filesystem; API host may
    # still be a separate CryoSPARC master. No staging/SSH transfer is involved.
    os.environ.update(_bridge_environment(config))
    client = make_client()
    version = server_version(client)
    if version and not str(version).lstrip("v").startswith("5."):
        raise ValueError(f"OTF currently targets CryoSPARC 5; server reports {version}")
    return client


def _options(args):
    devices = [v.strip() for v in args.gpu_devices.split(",")]
    seg, typing = allocate_devices(devices, args.run_typing)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not set(devices).issubset(visible.split(",")):
        raise ValueError("GPU IDs must belong to CUDA_VISIBLE_DEVICES; choose devices from your allocation")
    if args.num_cpus < len(devices):
        raise ValueError("Allocate at least one CPU per GPU worker")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 1:
        raise ValueError("Polling interval must be at least 1 second")
    if not math.isfinite(args.max_output_gb) or args.max_output_gb <= 0:
        raise ValueError("Output budget must be positive")
    if args.batch_forward_size < 0:
        raise ValueError("Batch size must be nonnegative")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
        raise FileNotFoundError(f"Segmentation weights not found: {checkpoint}")
    typing_checkpoint = None
    if args.run_typing:
        from cryofilter.classifier_weights import resolve_classifier_checkpoint
        typing_checkpoint = resolve_classifier_checkpoint(args.typing_checkpoint, search_dirs=(checkpoint.parent,))
        if not typing_checkpoint.is_file() or typing_checkpoint.stat().st_size < 1024:
            raise FileNotFoundError(f"Typing weights not found: {typing_checkpoint}")
    options = {"checkpoint": str(checkpoint), "threshold": args.threshold,
               "profile": args.inference_profile, "batch_size": args.batch_forward_size,
               "typing_checkpoint": str(typing_checkpoint) if typing_checkpoint else None,
               "typing_stride": args.typing_sample_stride_px}
    # Resource counts can change on resume. The model/mask recipe cannot.
    signature = dict(options, checkpoint_stamp=file_stamp(checkpoint), typing=args.run_typing,
                     typing_stamp=file_stamp(typing_checkpoint) if typing_checkpoint else None,
                     format=FORMAT)
    return devices, options, signature


def _validate_devices(devices):
    # Probe visibility in a short-lived process, without creating a CUDA context
    # in the supervisor or assuming memory-free GPUs are unallocated.
    import subprocess
    import sys
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                            env=env, capture_output=True, text=True, timeout=60, check=True)
    if int(result.stdout.strip().splitlines()[-1]) != len(devices):
        raise ValueError("One or more selected GPUs are not visible to PyTorch")


def _preview(entry, inference, typing, output, *, max_display_dim=1400, white_background=False):
    if white_background:
        # The app artifact browser excludes hidden directories. CryoSPARC has
        # its own export cache, so its white theme never replaces UI previews.
        output = output / ".cryosparc"
        output.mkdir(exist_ok=True)
    suffix = "" if max_display_dim == 1400 else f"_{max_display_dim}px"
    path = output / (entry["key"] + ("_typed" if typing else "") + "_area_v1" + suffix + "_particle_overlay.png")
    if path.exists():
        return path
    import numpy as np
    from PIL import Image
    from cryofilter.cli import _load_mrc_2d
    from cryofilter.qc_render import render_particle_overlay_png
    # Display preprocessing stays off the inference worker's critical path.
    image, _ = _load_mrc_2d(Path(entry["path"]))
    image = _average_preview_pixels(image, max_display_dim=max_display_dim)
    size = (image.shape[1], image.shape[0])
    mask = np.asarray(Image.fromarray(load_mask(inference["output_mask_npy"])).resize(size, Image.Resampling.NEAREST))
    prob = np.load(inference["output_prob_npy"], mmap_mode="r", allow_pickle=False)
    typed = np.asarray(Image.fromarray(np.load(typing["path"], mmap_mode="r", allow_pickle=False)).resize(
        size, Image.Resampling.NEAREST)) if typing else None
    temporary = path.with_name(path.stem + ".tmp.png")
    render_particle_overlay_png(image=image, mask=mask, coords_xy=np.empty((0, 2)), keep=np.empty(0, dtype=bool),
        output_path=temporary, probability_map=prob, typed_mask=typed, label=Path(entry["path"]).name,
        include_raw_panel=True, include_probability_panel=True, include_typed_mask_panel=bool(typing),
        max_display_dim=max_display_dim, white_background=white_background)
    temporary.replace(path)
    return path


def _average_preview_pixels(image, *, max_display_dim):
    """Bin the full float image before median/contrast display processing."""
    import numpy as np
    from PIL import Image
    stride = max(1, math.ceil(max(image.shape) / max_display_dim))
    if stride == 1:
        return image
    height, width = image.shape
    size = (math.ceil(width / stride), math.ceil(height / stride))
    # Area averaging uses the full image instead of aliasing high-frequency camera
    # noise through pixel skipping. Filtering/normalization then runs on the small
    # preview. BOX also retains border pixels when dimensions are not divisible.
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.float32)).resize(size, Image.Resampling.BOX))


def _preview_gallery(rows, output, *, white_background=False):
    """Compose at most ten cached previews, newest first; keep one local gallery."""
    from PIL import Image, ImageDraw
    from cryofilter.qc_render import _load_label_font

    rows = rows[:10]
    if not rows:
        raise ValueError("No completed micrographs to preview")
    panels = []
    width, gap, caption_height = 1600, 16, 32
    for entry, inference, typed in rows:
        path = _preview(entry, inference, typed, output, max_display_dim=600, white_background=white_background)
        with Image.open(path) as handle:
            panel = handle.convert("RGB")
            panel.thumbnail((width - 2 * gap, 560), Image.Resampling.LANCZOS)
            panels.append(panel)
    header_height = 44
    height = header_height + sum(panel.height + caption_height + gap for panel in panels)
    gallery = Image.new("RGB", (width, height), "white" if white_background else (25, 25, 25))
    text_color = "black" if white_background else "white"
    draw = ImageDraw.Draw(gallery)
    font = _load_label_font(20)
    draw.text((gap, 10), f"cryoFILTER OTF | Latest {len(rows)} micrographs | Newest first",
              fill=text_color, font=font)
    y = header_height
    for rank, ((entry, _, typed), panel) in enumerate(zip(rows, panels), start=1):
        stage = "segmentation + typing" if typed else "segmentation"
        draw.text((gap, y), f"{rank}. Micrograph UID {entry['uid']} | {stage}", fill=text_color, font=font)
        y += caption_height
        gallery.paste(panel, ((width - panel.width) // 2, y))
        y += panel.height + gap
    path = (output / ".cryosparc" if white_background else output) / "latest_micrographs.png"
    temporary = path.with_name(path.stem + ".tmp.png")
    # Use the same PNG format as inline CryoSPARC diagnostic plots; JPEG assets
    # appear as download links in the v5 event log.
    gallery.save(temporary, format="PNG", compress_level=3)
    temporary.replace(path)
    return path


def _publish_previews(rows, output, external, *, progress_message="OTF previews: follow latest for the current gallery."):
    """Render and upload off the supervisor/GPU path, using the public v5 API."""
    if not rows:
        return
    _preview(*rows[0], output)  # Preserve the UI's original full-size dark preview.
    try:
        external.set_tile_image(str(_preview(*rows[0], output, white_background=True)))
    except Exception as exc:
        print(f"OTF card preview unavailable: {type(exc).__name__}: {exc}", flush=True)
    _preview_gallery(rows, output)
    gallery = _preview_gallery(rows, output, white_background=True)
    from .otf_preview_log import publish_gallery
    return publish_gallery(external, gallery, rows, progress_message)


def _remember_preview(index, result):
    if result:
        index.set_meta("progress_event_id", result["progress_event_id"])


def _recent_preview_rows(records, uids):
    return [records[uid] for uid in reversed(uids) if uid in records and records[uid][1] is not None]


def _validate_cached_outputs(index):
    """Invalidate changed generations before any cached work is scheduled."""
    for entry, inference, typed in index.rows():
        if inference is None:
            continue
        try:
            valid = file_stamp(entry["path"]) == entry["stamp"] and all(
                file_stamp(inference[key]) == stamp for key, stamp in zip(
                    ("output_mask_npy", "output_prob_npy"), inference["output_stamps"], strict=True))
        except (OSError, KeyError, ValueError):
            valid = False
        if not valid:
            index.invalidate(entry["uid"])
        elif typed:
            try:
                valid = file_stamp(typed["path"]) == typed["output_stamp"]
            except (OSError, KeyError):
                valid = False
            if not valid:
                index.invalidate(entry["uid"], typing_only=True)


def run_otf(args, config):
    project_uid = validate_project_uid(args.project)
    workspace_uid = validate_workspace_uid(args.workspace)
    source_uid = validate_job_uid(args.micrographs.split(":")[0])
    devices, options, signature = _options(args)
    _validate_devices(devices)
    client = _connect(config)
    project = find_project(client, project_uid)
    source = PatchSource(find_job(client, project, project_uid, source_uid), project)
    source.refresh()
    external = None
    index = None
    lock = None
    pool = None
    previews = None
    started_external = False
    created_external = False
    signature.update(project=project_uid, source_job=source_uid, project_dir=str(source.project_dir))
    run_id = str(UUID(args.run_id)) if args.run_id else str(uuid4())
    if args.resume_job:
        from .bridge.compat import find_external_job
        uid = validate_job_uid(args.resume_job)
        external = find_external_job(client, project, project_uid, uid)
        card, path = read_card(_object_dir(external, "job_dir"), project=project_uid, job_uid=uid)
        run_dir = path.parent
    else:
        run_dir = Path(args.local_run_root).expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    try:
        lock = (run_dir / "runner.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This OTF card already has an active runner") from None
        index = Index(run_dir / "index.sqlite")
        old_signature = index.get_meta("signature")
        if args.resume_job and signature != old_signature:
            raise ValueError("Resume requires the same source job, checkpoints, threshold, profile and typing settings")
        index.set_meta("signature", signature)
        if external is None:
            external = create_external_job(project, workspace_uid, title=args.title,
                desc=f"cryoFILTER OTF follows Patch Motion Correction {source_uid}. Masks are indexed by micrograph UID.")
            created_external = True
            card = {"format": FORMAT, "project": project_uid, "workspace": workspace_uid,
                    "job_uid": object_uid(external), "source_job": source_uid,
                    "index_path": str(index.path), "run_dir": str(run_dir), "run_id": run_id}
            card_path = _object_dir(external, "job_dir") / CARD_FILE
            if card_path.exists():
                raise FileExistsError(f"OTF card metadata already exists: {card_path}")
            write_json(card_path, card)
            write_json(run_dir / CARD_FILE, card)
        external.start(status="running")
        started_external = True
        print(f"cryoFILTER OTF card: {project_uid}/{object_uid(external)}; source {source_uid}", flush=True)
        print(f"Mask index: {index.path}", flush=True)
        seg_devices, type_devices = allocate_devices(devices, args.run_typing)
        print(f"GPUs: segmentation {','.join(seg_devices)}; typing {','.join(type_devices) or 'disabled'}. CPU budget: {args.num_cpus}", flush=True)
        # Validate cached outputs without reading micrograph pixels. Unpublished
        # files left by interruption are kept and included in the storage budget.
        _validate_cached_outputs(index)
        options["output"] = str(run_dir / "inference")
        pool = WorkerPool(devices, args.run_typing, args.num_cpus, options)
        preview_dir = run_dir / "OTF_images"
        preview_dir.mkdir(exist_ok=True)
        previews = ThreadPoolExecutor(max_workers=1) if not args.no_previews else None
        supervise(source, index, pool, external, args, run_dir, previews, preview_dir)
        stop_external_job(external)
        return 0
    except BaseException as exc:
        if index is not None and (started_external or created_external):
            status = index.get_meta("status", {})
            status.update(phase="stopped" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", error=str(exc))
            index.set_meta("status", status)
        if external is not None and (started_external or created_external):
            try:
                stop_external_job(external, error=f"OTF interrupted: {type(exc).__name__}: {exc}")
            except Exception as report_error:
                print(f"Could not update OTF card status: {report_error}", flush=True)
        raise
    finally:
        if pool is not None:
            pool.close()
        if previews is not None:
            previews.shutdown(wait=True, cancel_futures=True)
        if index is not None:
            index.close()
        if lock is not None:
            lock.close()


def supervise(source, index, pool, external, args, run_dir, previews=None, preview_dir=None):
    """Single owner commits worker results; readers never see half-written masks."""
    inflight = {"segmentation": set(), "typing": set()}
    reservations = {}
    used_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    max_bytes = int(args.max_output_gb * 1024**3)
    next_scan = 0.0
    next_status = 0.0
    next_preview = 0.0
    next_card_log = 0.0
    final_since = None
    preview_future = None
    timings = {"segmentation": [], "typing": []}
    failures = 0
    waiting = []
    ready_keys = set()
    records = {entry["uid"]: (entry, inference, typed) for entry, inference, typed in index.rows()}
    preview_uids = deque(index.get_meta("preview_uids") or [
        e["uid"] for e, inference, _ in records.values() if inference
    ][-10:], maxlen=10)
    preview_pending = bool(_recent_preview_rows(records, preview_uids))
    while True:
        changed = False
        for role, entry, record, seconds, error in pool.collect():
            if error:
                raise RuntimeError(f"{role} worker failed: {error}")
            inflight[role].discard(entry["key"])
            if file_stamp(entry["path"]) != entry["stamp"]:
                raise ValueError(f"Source changed during processing: {entry['path']}; resume after motion correction finishes writing")
            if role == "segmentation":
                record["output_stamps"] = [file_stamp(record[k]) for k in ("output_mask_npy", "output_prob_npy")]
            else:
                record["output_stamp"] = file_stamp(record["path"])
            if index.complete(role, entry, record):
                original = records[entry["uid"]]
                records[entry["uid"]] = (original[0], record, original[2]) if role == "segmentation" else (original[0], original[1], record)
                timings[role].append(seconds)
                timings[role] = timings[role][-100:]
                if role == "segmentation":
                    if entry["uid"] in preview_uids:
                        preview_uids.remove(entry["uid"])
                    preview_uids.append(entry["uid"])
                    preview_pending = True
                elif entry["uid"] in preview_uids:
                    # Late typing refreshes that row without promoting an older
                    # image ahead of newly segmented micrographs.
                    preview_pending = True
            changed = True
        pool.check()
        now = time.monotonic()
        if now >= next_scan:
            try:
                source.refresh()
                rows = records.values()
                known = {e["uid"]: e["key"] for e, i, _ in rows if i}
                entries, waiting = source.scan(known)
                for entry in entries:
                    ready_keys.add(entry["key"])
                    if index.discover(entry):
                        records[entry["uid"]] = (entry, None, None)
                        if entry["uid"] in preview_uids:
                            preview_uids.remove(entry["uid"])
                            preview_pending = True
                        changed = True
                failures = 0
            except (OSError, ConnectionError, TimeoutError) as exc:
                failures += 1
                print(f"OTF source temporarily unavailable: {type(exc).__name__}: {exc}", flush=True)
                if failures >= 10:
                    raise
            next_scan = now + min(30, args.poll_seconds * (2 ** failures))
        rows = records.values()
        seg_pending = [(e, i, t) for e, i, t in rows if i is None]
        type_pending = [(e, i, t) for e, i, t in rows if i is not None and t is None] if args.run_typing else []
        for role, pending in (("segmentation", seg_pending), ("typing", type_pending)):
            for entry, inference, _ in pending:
                if len(inflight[role]) >= pool.capacity[role]:
                    break
                if entry["key"] in inflight[role]:
                    continue
                if role == "segmentation" and entry["key"] not in ready_keys:
                    continue
                if entry["key"] not in reservations:
                    # Bound packed masks (including incompressible input), preview
                    # probability data, typing and sampled PNG overhead.
                    estimate = 16 * 1024**2  # Separate UI and white CryoSPARC preview caches.
                    if role == "segmentation":
                        estimate += math.ceil(math.prod(entry["shape"]) / 8)
                    if args.run_typing:
                        estimate += math.prod(entry["shape"])
                    if used_bytes + estimate > max_bytes:
                        raise ValueError(f"OTF output budget ({args.max_output_gb:g} GiB) reached; completed masks are preserved. Resume with a larger budget.")
                    reservations[entry["key"]] = estimate
                    used_bytes += estimate
                task = dict(entry)
                if inference:
                    task["mask_path"] = inference["output_mask_npy"]
                if pool.submit(role, task):
                    inflight[role].add(entry["key"])
        n_segmented = sum(i is not None for _, i, _ in rows)
        n_typed = sum(t is not None for _, _, t in rows)
        terminal = source.status in TERMINAL
        if terminal and final_since is None:
            final_since = now
        all_final = source.final_entries is not None and all(
            uid in records and records[uid][1] is not None for uid in source.final_entries)
        drained = not seg_pending and not type_pending and not any(inflight.values())
        if terminal and waiting and final_since is not None and now - final_since > 120:
            raise ValueError(f"Motion correction stopped, but {len(waiting)} output file(s) are incomplete or unreadable")
        phase = ("draining" if terminal else "processing" if seg_pending or type_pending else "waiting_for_motion_correction")
        status = {"phase": phase, "source_status": source.status, "expected": source.expected,
                  "discovered": len(rows), "segmented": n_segmented, "typed": n_typed,
                  "segmentation_pending": len(seg_pending), "typing_pending": len(type_pending),
                  "source_files_pending": len(waiting), "reserved_output_bytes": used_bytes,
                  "source_warning": getattr(source, "last_warning", None),
                  "gpu_devices": args.gpu_devices, "run_typing": args.run_typing,
                  "job_uid": object_uid(external), "updated_at": time.time(),
                  "seconds_per_micrograph": {k: sum(v) / len(v) for k, v in timings.items() if v}}
        if now >= next_status or changed:
            index.set_meta("status", status)
            index.set_meta("preview_uids", list(preview_uids))
            next_status = now + 2
        message = f"OTF {phase}: {n_segmented} segmented, {n_typed} typed; {len(seg_pending)} awaiting segmentation, {len(type_pending)} awaiting typing. Motion correction: {source.status}."
        if previews is not None:
            message += " Follow latest in the Event Log to see the current latest-ten gallery."
        if now >= next_card_log:
            print(message, flush=True)
            try:
                event_id = external.log(message, id=index.get_meta("progress_event_id"))
                if event_id:
                    index.set_meta("progress_event_id", str(event_id))
            except Exception as exc:
                print(f"Card progress update unavailable: {type(exc).__name__}", flush=True)
            next_card_log = now + 30
        if preview_future is not None and preview_future.done():
            try:
                _remember_preview(index, preview_future.result())
            except Exception as exc:
                print(f"OTF preview unavailable: {type(exc).__name__}: {exc}", flush=True)
                preview_pending = True
            preview_future = None
        if previews is not None and preview_pending and preview_future is None and now >= next_preview:
            snapshot = _recent_preview_rows(records, preview_uids)
            preview_future = previews.submit(_publish_previews, snapshot, preview_dir, external, progress_message=message)
            preview_pending = False
            next_preview = now + 15
        if terminal and drained and not waiting and failures == 0:
            if source.status != "completed":
                raise RuntimeError(f"Motion correction {source.status}; completed masks remain available for filtering/resume")
            if all_final:
                if not source.final_entries:
                    raise ValueError("Motion correction completed without any successful micrographs")
                # Finish the latest sampled preview while the external card is
                # still running, including runs shorter than the sampling interval.
                if previews is not None:
                    if preview_future is not None:
                        try:
                            _remember_preview(index, preview_future.result())
                        except Exception as exc:
                            print(f"OTF preview unavailable: {type(exc).__name__}: {exc}", flush=True)
                            preview_pending = True
                    try:
                        snapshot = _recent_preview_rows(records, preview_uids)
                        result = previews.submit(_publish_previews, snapshot, preview_dir, external,
                            progress_message=f"OTF complete: {n_segmented} segmented, {n_typed} typed.").result()
                        _remember_preview(index, result)
                    except Exception as exc:
                        print(f"OTF final gallery unavailable: {type(exc).__name__}: {exc}", flush=True)
                status["phase"] = "completed"
                index.set_meta("status", status)
                message = f"OTF complete: {n_segmented} segmented, {n_typed} typed."
                print(message, flush=True)
                try:
                    event_id = external.log(message, id=index.get_meta("progress_event_id"))
                    if event_id:
                        index.set_meta("progress_event_id", str(event_id))
                except Exception as exc:
                    print(f"Card progress update unavailable: {type(exc).__name__}", flush=True)
                return
        time.sleep(0.2)
