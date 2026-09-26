"""Persistence: images on disk plus a SQLite database of reports for researchers."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from .detector import Detection
from .scoring import severity_for

# Cleanup workflow, in order. "dismissed" = not actionable (duplicate, not litter, private land...).
STATUSES = ["reported", "verified", "scheduled", "in_progress", "cleaned", "dismissed"]
OPEN_STATUSES = ["reported", "verified", "scheduled", "in_progress"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    captured_at     TEXT,
    waterway        TEXT,
    latitude        REAL,
    longitude       REAL,
    location_source TEXT,
    location_confirmed INTEGER NOT NULL DEFAULT 0,
    in_philadelphia INTEGER,
    notes           TEXT,
    reporter        TEXT,
    hazard_suspected INTEGER NOT NULL DEFAULT 0,
    image_path      TEXT NOT NULL,
    annotated_path  TEXT NOT NULL,
    image_width     INTEGER NOT NULL,
    image_height    INTEGER NOT NULL,
    model_name      TEXT NOT NULL,
    model_is_stand_in INTEGER NOT NULL,
    score           INTEGER NOT NULL,
    severity        TEXT NOT NULL,
    item_count      INTEGER NOT NULL,
    coverage_pct    REAL NOT NULL,
    authority_level INTEGER NOT NULL,
    recommendation_json TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'reported',
    status_updated_at TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS status_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    status      TEXT NOT NULL,
    changed_at  TEXT NOT NULL,
    changed_by  TEXT,
    note        TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_waterway ON reports(waterway);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_latlon ON reports(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_detections_report ON detections(report_id);
CREATE INDEX IF NOT EXISTS idx_history_report ON status_history(report_id);
"""

# Columns added after the first version; older databases get them via ALTER TABLE.
MIGRATIONS = {
    "location_confirmed": "INTEGER NOT NULL DEFAULT 0",
    "status": "TEXT NOT NULL DEFAULT 'reported'",
    "status_updated_at": "TEXT",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.annotated_dir = data_dir / "annotated"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(data_dir / "reports.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        existing = {r["name"] for r in self.db.execute("PRAGMA table_info(reports)")}
        for col, decl in MIGRATIONS.items():
            if col not in existing:
                self.db.execute(f"ALTER TABLE reports ADD COLUMN {col} {decl}")
        self.db.executescript(INDEXES)
        self.db.commit()

    # ---- images -------------------------------------------------------

    def save_images(self, report_id: str, image: Image.Image, detections: list[Detection]) -> tuple[str, str]:
        month = datetime.now(timezone.utc).strftime("%Y/%m")
        original = self.images_dir / month / f"{report_id}.jpg"
        annotated = self.annotated_dir / month / f"{report_id}.jpg"
        original.parent.mkdir(parents=True, exist_ok=True)
        annotated.parent.mkdir(parents=True, exist_ok=True)

        image.save(original, "JPEG", quality=92)
        drawn = image.copy()
        draw = ImageDraw.Draw(drawn)
        line = max(2, image.width // 400)
        for d in detections:
            draw.rectangle([d.x, d.y, d.x + d.w, d.y + d.h], outline=(255, 64, 64), width=line)
            draw.text((d.x + line, max(0, d.y - 12)), f"{d.label} {d.confidence:.2f}", fill=(255, 64, 64))
        drawn.save(annotated, "JPEG", quality=88)
        return (original.relative_to(self.data_dir).as_posix(), annotated.relative_to(self.data_dir).as_posix())

    def abs_path(self, rel: str) -> Path:
        return self.data_dir / rel

    # ---- reports ------------------------------------------------------

    def insert(self, report: dict, detections: list[Detection]) -> None:
        report = {**report, "status": "reported", "status_updated_at": report["created_at"]}
        cols = [c for c in report if c != "recommendation"]
        values = [report[c] for c in cols] + [json.dumps(report["recommendation"])]
        cols.append("recommendation_json")
        with self._lock, self.db:
            self.db.execute(
                f"INSERT INTO reports ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", values
            )
            self.db.executemany(
                "INSERT INTO detections (report_id, label, confidence, x, y, w, h) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(report["id"], d.label, d.confidence, d.x, d.y, d.w, d.h) for d in detections],
            )
            self.db.execute(
                "INSERT INTO status_history (report_id, status, changed_at, changed_by, note) VALUES (?, ?, ?, ?, ?)",
                (report["id"], "reported", report["created_at"], report.get("reporter"), "Report submitted"),
            )

    def set_location(self, report_id: str, lat: float, lon: float, in_philadelphia: bool) -> None:
        with self._lock, self.db:
            self.db.execute(
                "UPDATE reports SET latitude = ?, longitude = ?, location_source = 'map', "
                "location_confirmed = 1, in_philadelphia = ? WHERE id = ?",
                (lat, lon, in_philadelphia, report_id),
            )

    def set_status(self, report_id: str, status: str, changed_by: str | None, note: str | None) -> None:
        ts = now_iso()
        with self._lock, self.db:
            self.db.execute("UPDATE reports SET status = ?, status_updated_at = ? WHERE id = ?", (status, ts, report_id))
            self.db.execute(
                "INSERT INTO status_history (report_id, status, changed_at, changed_by, note) VALUES (?, ?, ?, ?, ?)",
                (report_id, status, ts, changed_by, note),
            )

    def get(self, report_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return self._hydrate(row) if row else None

    def list(self, *, limit: int = 50, offset: int = 0, sort: str = "newest", **filters) -> tuple[list[dict], int]:
        clause, args = _where(**filters)
        order = {
            "newest": "created_at DESC",
            "oldest": "created_at ASC",
            "score": "score DESC, created_at DESC",
        }[sort]
        total = self.db.execute(f"SELECT COUNT(*) FROM reports {clause}", args).fetchone()[0]
        rows = self.db.execute(
            f"SELECT * FROM reports {clause} ORDER BY {order} LIMIT ? OFFSET ?", [*args, limit, offset]
        ).fetchall()
        return [self._hydrate(r) for r in rows], total

    # ---- maps ---------------------------------------------------------

    def points(self, **filters) -> list[dict]:
        """Lightweight rows for map layers (only reports that have a location)."""
        clause, args = _where(**filters, has_location=True)
        return [dict(r) for r in self.db.execute(
            f"SELECT id, latitude, longitude, score, severity, status, waterway, created_at, item_count, "
            f"location_confirmed FROM reports {clause}", args
        ).fetchall()]

    def bubbles(self, cell_deg: float, **filters) -> list[dict]:
        """Group located reports into a lat/lon grid; one bubble per non-empty cell."""
        cells: dict[tuple[int, int], list[dict]] = {}
        for p in self.points(**filters):
            key = (math.floor(p["latitude"] / cell_deg), math.floor(p["longitude"] / cell_deg))
            cells.setdefault(key, []).append(p)
        out = []
        for (i, j), pts in cells.items():
            n = len(pts)
            avg = sum(p["score"] for p in pts) / n
            out.append({
                "cell": f"{i}:{j}",
                # Bubble sits at the average position of its reports, not the grid corner.
                "latitude": round(sum(p["latitude"] for p in pts) / n, 6),
                "longitude": round(sum(p["longitude"] for p in pts) / n, 6),
                "bounds": [round(i * cell_deg, 6), round(j * cell_deg, 6),
                           round((i + 1) * cell_deg, 6), round((j + 1) * cell_deg, 6)],
                "report_count": n,
                "open_count": sum(p["status"] in OPEN_STATUSES for p in pts),
                "avg_score": round(avg, 1),
                "max_score": max(p["score"] for p in pts),
                "avg_severity": severity_for(round(avg)),
                "waterways": sorted({p["waterway"] for p in pts if p["waterway"]}),
                "latest_report_at": max(p["created_at"] for p in pts),
            })
        return sorted(out, key=lambda b: (-b["report_count"], -b["avg_score"]))

    # ---- stats --------------------------------------------------------

    def stats(self) -> dict:
        overall = self.db.execute(
            "SELECT COUNT(*) AS reports, AVG(score) AS avg_score, SUM(item_count) AS items FROM reports"
        ).fetchone()
        by_severity = dict(self.db.execute("SELECT severity, COUNT(*) FROM reports GROUP BY severity").fetchall())
        by_status = {s: 0 for s in STATUSES}
        by_status.update(dict(self.db.execute("SELECT status, COUNT(*) FROM reports GROUP BY status").fetchall()))
        by_waterway = [
            dict(r) for r in self.db.execute(
                "SELECT COALESCE(waterway, 'Unspecified') AS waterway, COUNT(*) AS reports, "
                "ROUND(AVG(score), 1) AS avg_score, MAX(score) AS max_score, "
                f"SUM(status IN ({', '.join('?' * len(OPEN_STATUSES))})) AS open_reports "
                "FROM reports GROUP BY waterway ORDER BY avg_score DESC", OPEN_STATUSES
            ).fetchall()
        ]
        return {
            "reports": overall["reports"],
            "open_reports": sum(by_status[s] for s in OPEN_STATUSES),
            "avg_score": round(overall["avg_score"] or 0, 1),
            "items_detected": overall["items"] or 0,
            "by_severity": by_severity,
            "by_status": by_status,
            "by_waterway": by_waterway,
        }

    def _hydrate(self, row: sqlite3.Row) -> dict:
        report = dict(row)
        report["recommendation"] = json.loads(report.pop("recommendation_json"))
        for flag in ("in_philadelphia", "hazard_suspected", "model_is_stand_in", "location_confirmed"):
            if report[flag] is not None:
                report[flag] = bool(report[flag])
        report["detections"] = [
            dict(r) for r in self.db.execute(
                "SELECT label, confidence, x, y, w, h FROM detections WHERE report_id = ? ORDER BY confidence DESC",
                (report["id"],),
            ).fetchall()
        ]
        report["status_history"] = [
            dict(r) for r in self.db.execute(
                "SELECT status, changed_at, changed_by, note FROM status_history WHERE report_id = ? ORDER BY id",
                (report["id"],),
            ).fetchall()
        ]
        return report


def _where(*, waterway=None, severity=None, status=None, min_score=None, since=None, until=None,
           bbox=None, has_location=False) -> tuple[str, list]:
    """Build a WHERE clause shared by list, export and map queries.

    severity/status accept a list (any-of). status="open" expands to every not-yet-closed status.
    bbox is (min_lat, min_lon, max_lat, max_lon).
    """
    where, args = [], []
    if waterway:
        where.append("waterway = ?"); args.append(waterway)
    for col, values in (("severity", severity), ("status", status)):
        if values:
            values = [values] if isinstance(values, str) else list(values)
            if col == "status" and "open" in values:
                values = [v for v in values if v != "open"] + OPEN_STATUSES
            where.append(f"{col} IN ({', '.join('?' * len(values))})"); args.extend(values)
    if min_score is not None:
        where.append("score >= ?"); args.append(min_score)
    if since:
        where.append("created_at >= ?"); args.append(since)
    if until:
        where.append("created_at <= ?"); args.append(until)
    if has_location or bbox:
        where.append("latitude IS NOT NULL AND longitude IS NOT NULL")
    if bbox:
        where.append("latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?")
        args.extend([bbox[0], bbox[2], bbox[1], bbox[3]])
    return (f"WHERE {' AND '.join(where)}" if where else ""), args
