"""Live galleries grouped into checkpoints using ordinary cryosparc-tools."""
from __future__ import annotations

import hashlib
import json

from .otf_state import write_json


MARKER = "cryofilter_otf_gallery"


def publish_gallery(external, gallery, rows, progress):
    """Publish one gallery per changed snapshot; recover interrupted uploads.

    Follow latest selects each new checkpoint in CryoSPARC, keeping the visible
    view to one gallery. Earlier checkpoints remain in the job's history.
    Only the single preview thread owns this small local state file.
    """
    recipe = [(entry["key"], inference.get("output_stamps"), bool(typed),
               typed.get("output_stamp") if typed else None) for entry, inference, typed in rows[:10]]
    signature = hashlib.sha256(json.dumps(["white_area_v1", recipe], sort_keys=True).encode()).hexdigest()
    context = {"project": external.project_uid, "job": external.uid, "signature": signature}
    state_path = gallery.parent / "event_log.json"
    try:
        previous = json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    same = all(previous.get(k) == v for k, v in context.items())
    state = previous if same else dict(context)

    # Normal changed snapshots need no history fetch. Only first use or recovery
    # reads history, including the case where the server saved an event but its
    # response or the local state write was lost.
    if not previous or (same and not state.get("image_event_id")):
        events = sorted(external.cs.api.jobs.get_event_logs(external.project_uid, external.uid),
                        key=lambda e: e.created_at)
        checkpoints = [i for i, e in enumerate(events) if getattr(e, "type", None) == "checkpoint"]
        if checkpoints:
            pos = checkpoints[-1]
            checkpoint = events[pos]
            if checkpoint.meta.get(MARKER) == signature:
                state["checkpoint_id"] = str(checkpoint.id)
                for event in events[pos + 1:]:
                    if getattr(event, "type", None) == "image" and MARKER in event.flags:
                        state["image_event_id"] = str(event.id)
                    elif getattr(event, "type", None) == "text" and event.text.startswith("OTF "):
                        state["progress_event_id"] = str(event.id)
    write_json(state_path, state)
    if not state.get("checkpoint_id"):
        state["checkpoint_id"] = str(external.log_checkpoint(meta={MARKER: signature}))
        write_json(state_path, state)
    # Explicit IDs avoid named-event resets at checkpoints and races with the
    # supervisor's progress updates on the same controller.
    state["progress_event_id"] = str(external.log(progress, id=state.get("progress_event_id")))
    write_json(state_path, state)
    if not state.get("image_event_id"):
        state["image_event_id"] = str(external.log_plot(
            figure=str(gallery), formats=["png"], flags=["plots", MARKER],
            text=f"cryoFILTER OTF: latest {min(len(rows), 10)} micrographs (newest first). "
                 "Raw image, contamination mask, typing when available, and probability. "
                 "Follow latest shows the current gallery; earlier checkpoints retain history.",
        ))
        write_json(state_path, state)
    return state
