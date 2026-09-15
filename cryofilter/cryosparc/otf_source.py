"""Read-only discovery of CryoSPARC 5 Patch Motion Correction outputs."""
from __future__ import annotations

import math
import os
import re
import struct
import time
from pathlib import Path

from .bridge.prepare import _dataset_fields, _object_dir
from .otf_state import file_stamp, generation_key

PATCH_FILE = re.compile(r"^(\d+)_.+_patch_aligned_doseweighted\.mrc$", re.IGNORECASE)
TERMINAL = {"completed", "failed", "killed", "aborted"}


def mrc_geometry(path: Path):
    """Read only the header; reject truncated, complex, or multi-frame files."""
    with path.open("rb") as handle:
        header = handle.read(1024)
    if len(header) != 1024 or header[208:212] != b"MAP ":
        raise ValueError("Incomplete/invalid MRC header")
    endian = ">" if header[212:214] == b"\x11\x11" else "<"
    nx, ny, nz, mode = struct.unpack_from(endian + "4i", header)
    sizes = {0: 1, 1: 2, 2: 4, 6: 2, 12: 2}
    if nx <= 0 or ny <= 0 or nz != 1 or mode not in sizes:
        raise ValueError("Expected a single real-valued micrograph")
    extra = struct.unpack_from(endian + "i", header, 92)[0]
    if extra < 0 or path.stat().st_size != 1024 + extra + nx * ny * sizes[mode]:
        raise ValueError("Micrograph payload is incomplete")
    mx = struct.unpack_from(endian + "i", header, 28)[0]
    cell_x = struct.unpack_from(endian + "f", header, 40)[0]
    px = cell_x / mx if mx > 0 else 0
    if not math.isfinite(px) or px <= 0:
        raise ValueError("Micrograph header does not contain a positive pixel size")
    return [ny, nx], px


class FileReadiness:
    def __init__(self, stable_seconds=4.0):
        self.stable_seconds = stable_seconds
        self.observed = {}

    def ready(self, path: Path, *, now=None):
        now = time.monotonic() if now is None else now
        try:
            stamp = file_stamp(path)
            old = self.observed.get(str(path))
            if old is None or old[0] != stamp:
                self.observed[str(path)] = (stamp, now)
                return None
            if now - old[1] < self.stable_seconds:
                return None
            shape, pixel_size = mrc_geometry(path)
            if file_stamp(path) != stamp:
                return None
            return stamp, shape, pixel_size
        except (OSError, ValueError, struct.error):
            return None


class PatchSource:
    def __init__(self, job, project):
        self.job = job
        self.project_dir = _object_dir(project, "project_dir")
        self.job_dir = _object_dir(job, "job_dir")
        if self.project_dir is None or self.job_dir is None or not self.job_dir.is_dir():
            raise ValueError("OTF requires direct read access to the CryoSPARC project/job directory on this machine")
        self.directory = self.job_dir / "motioncorrected"
        self.known_uids = set()
        self.status = "waiting"
        self.expected = 0
        self.final_entries = None
        self.readiness = FileReadiness()
        self.last_warning = None

    def refresh(self):
        try:
            self.job.refresh()
        except Exception as exc:
            raise ConnectionError(f"Cannot refresh motion-correction job: {exc}") from exc
        kind = str(self.job.type)
        if kind not in {"patch_motion_correction_multi", "patch_motion_correction"}:
            raise ValueError(f"OTF currently supports Patch Motion Correction; selected job type is {kind}")
        self.status = str(self.job.status)
        if not self.known_uids:
            try:
                movies = self.job.load_input("movies", slots=["movie_blob"])
                self.known_uids = {str(int(uid)) for uid in movies["uid"]}
                self.expected = len(self.known_uids)
                self.last_warning = None
            except Exception as exc:
                if self.status in TERMINAL:
                    raise ConnectionError(f"Cannot read movie input metadata: {exc}") from exc
                self.last_warning = f"Waiting for movie input metadata: {type(exc).__name__}"
        if self.status in TERMINAL and self.final_entries is None:
            try:
                data = self.job.load_output("micrographs", slots=["micrograph_blob"])
            except Exception as exc:
                # Publication may lag the terminal status; the supervisor retries
                # with bounded backoff. Failed upstream jobs can still have masks.
                if self.status == "completed":
                    raise ConnectionError(f"Waiting for final micrograph output: {exc}") from exc
            else:
                fields = _dataset_fields(data)
                if "micrograph_blob/path" not in fields:
                    raise ValueError("Final motion-correction output has no micrograph_blob/path")
                final = {}
                for uid, path in zip(data["uid"], data["micrograph_blob/path"]):
                    uid = str(int(uid))
                    raw = path.decode() if isinstance(path, bytes) else str(path)
                    final[uid] = (self.project_dir / raw).resolve()
                self.final_entries = final
                self.expected = len(final)

    def scan(self, completed_keys):
        if self.final_entries is not None:
            paths = list(self.final_entries.items())
        elif self.directory.is_dir():
            paths = []
            with os.scandir(self.directory) as listing:
                for file in listing:
                    match = PATCH_FILE.fullmatch(file.name)
                    if match and str(int(match[1])) in self.known_uids:
                        paths.append((str(int(match[1])), Path(file.path)))
        else:
            paths = []
        seen = set()
        ready = []
        waiting = []
        for uid, path in paths:
            if uid in seen:
                raise ValueError(f"Multiple dose-weighted micrographs match UID {uid}")
            seen.add(uid)
            # A finished image normally needs no further filesystem I/O. At upstream
            # completion, revalidate all stamps once to detect changed/replaced output.
            if self.final_entries is None and uid in completed_keys:
                continue
            checked = self.readiness.ready(path)
            if checked is None:
                waiting.append(uid)
                continue
            stamp, shape, pixel_size = checked
            key = generation_key(uid, stamp)
            if completed_keys.get(uid) == key:
                continue
            ready.append({"uid": uid, "key": key, "path": str(path.resolve()),
                          "source_relative_path": os.path.relpath(path, self.project_dir),
                          "stamp": stamp, "shape": shape, "pixel_size_angstrom": pixel_size})
        return ready, waiting
