"""Durable per-micrograph OTF index shared by the runner, monitor and filtering."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

FORMAT = "cryofilter-otf-v1"
CARD_FILE = "cryofilter_otf.json"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def generation_key(uid, stamp):
    return str(uid) + "_" + hashlib.sha256(json.dumps(stamp).encode()).hexdigest()[:16]


def save_mask(path: Path, mask):
    """Lossless binary masks: bit packing plus compression; publish atomically."""
    import numpy as np
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, bits=np.packbits(np.asarray(mask, dtype=bool), axis=None),
                            shape=np.asarray(mask.shape, dtype=np.int64))
    temp.replace(path)


def load_mask(path):
    import numpy as np
    if Path(path).suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(v) for v in archive["shape"])
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("Invalid packed mask shape")
        bits = archive["bits"]
        if bits.dtype != np.uint8 or bits.size != (shape[0] * shape[1] + 7) // 8:
            raise ValueError("Packed mask length does not match its geometry")
        return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)


class Index:
    def __init__(self, path: Path, *, readonly=False):
        self.path = Path(path).resolve()
        if readonly:
            self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=15)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(self.path), timeout=15)
            # Rollback journaling works on shared NFS; WAL needs same-host shared memory.
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS micrographs (uid TEXT PRIMARY KEY, entry TEXT NOT NULL, inference TEXT, typing TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS stats (uid TEXT PRIMARY KEY, pixels INTEGER, contaminated INTEGER, carbon INTEGER, crystalline INTEGER, aggregate INTEGER, ethane INTEGER, typed INTEGER DEFAULT 0)")
            self.db.commit()

    def close(self):
        self.db.close()

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, json.dumps(value)))

    def discover(self, entry):
        row = self.db.execute("SELECT entry FROM micrographs WHERE uid=?", (entry["uid"],)).fetchone()
        if row and json.loads(row[0])["key"] == entry["key"]:
            return False
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO micrographs VALUES (?,?,NULL,NULL)",
                            (entry["uid"], json.dumps(entry)))
            self.db.execute("INSERT OR REPLACE INTO stats (uid) VALUES (?)", (entry["uid"],))
        return True

    def rows(self):
        return [(json.loads(e), json.loads(i) if i else None, json.loads(t) if t else None)
                for e, i, t in self.db.execute("SELECT entry,inference,typing FROM micrographs ORDER BY rowid")]

    def complete(self, role, entry, record):
        row = self.db.execute("SELECT entry FROM micrographs WHERE uid=?", (entry["uid"],)).fetchone()
        if not row or json.loads(row[0])["key"] != entry["key"]:
            return False
        column = "inference" if role == "segmentation" else "typing"
        with self.db:
            self.db.execute(f"UPDATE micrographs SET {column}=? WHERE uid=?", (json.dumps(record), entry["uid"]))
            if role == "segmentation":
                shape = record["output_image_shape"]
                self.db.execute("UPDATE stats SET pixels=?,contaminated=? WHERE uid=?",
                    (shape[0] * shape[1], record["mask_postprocessing"]["final_mask_pixels"], entry["uid"]))
            else:
                summary = record["summary"]
                values = [int(summary.get(key + "_area_px", 0)) for key in ("carbon", "crystalline", "aggregate", "ethane")]
                self.db.execute("UPDATE stats SET carbon=?,crystalline=?,aggregate=?,ethane=?,typed=1 WHERE uid=?", (*values, entry["uid"]))
        return True

    def invalidate(self, uid, *, typing_only=False):
        clause = "typing=NULL" if typing_only else "inference=NULL, typing=NULL"
        with self.db:
            self.db.execute(f"UPDATE micrographs SET {clause} WHERE uid=?", (uid,))
            self.db.execute("UPDATE stats SET carbon=NULL,crystalline=NULL,aggregate=NULL,ethane=NULL,typed=0 WHERE uid=?", (uid,))
            if not typing_only:
                self.db.execute("UPDATE stats SET pixels=NULL,contaminated=NULL WHERE uid=?", (uid,))


def read_card(job_dir: Path, *, project: str, job_uid: str):
    card = json.loads((Path(job_dir) / CARD_FILE).read_text())
    if card.get("format") != FORMAT or card.get("project") != project or card.get("job_uid") != job_uid:
        raise ValueError("This card does not contain a matching cryoFILTER OTF index")
    path = Path(card["index_path"])
    if not path.is_file():
        raise FileNotFoundError(f"OTF masks/index are not accessible on this machine: {path}")
    return card, path


def monitor_summary(path: Path, *, mode="all", count=100):
    index = Index(path, readonly=True)
    try:
        # Small numeric rows only: monitor polling does not parse every source and
        # model record, reopen masks, or read any original micrographs.
        rows = index.db.execute("SELECT pixels,contaminated,carbon,crystalline,aggregate,ethane,typed FROM stats ORDER BY rowid").fetchall()
        status = index.get_meta("status", {})
    finally:
        index.close()
    completed = [row for row in rows if row[0] is not None]
    selected = completed[:count] if mode == "first" else completed[-count:] if mode == "last" else completed
    pixels = sum(row[0] for row in selected)
    contaminated = sum(row[1] for row in selected)
    from cryofilter.app.server import CONTAMINATION_TYPE_COLORS, CONTAMINATION_TYPE_LABELS
    types = []
    for column, label in enumerate(CONTAMINATION_TYPE_LABELS, start=2):
        area = sum(row[column] or 0 for row in selected)
        types.append({"label": label, "color": CONTAMINATION_TYPE_COLORS[label], "area_px": area})
    typed = sum(row[6] for row in rows)
    return {"ok": True, "available": bool(completed), "source": "otf", "mode": mode,
            "count": count, "n_images": len(selected), "n_images_completed": len(completed),
            "n_images_total": max(len(rows), status.get("expected", 0)),
            "n_images_typed": typed, "total_pixels": pixels, "contaminated_pixels": contaminated,
            "clean_pixels": pixels - contaminated, "contamination_fraction": contaminated / max(1, pixels),
            "types": types if typed else [], "otf": status,
            "message": "Waiting for motion-corrected micrographs."}
