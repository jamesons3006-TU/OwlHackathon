"""Persistence: images on disk plus a SQLite database of reports for researchers."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from .detector import Detection
from .scoring import severity_for


# Cleanup workflow, in order.
# "dismissed" = not actionable (duplicate, not litter, private land...).
STATUSES = [
    "reported",
    "verified",
    "scheduled",
    "in_progress",
    "cleaned",
    "dismissed",
]

OPEN_STATUSES = [
    "reported",
    "verified",
    "scheduled",
    "in_progress",
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    captured_at     TEXT,
    waterway        TEXT,
    location_name   TEXT,
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
    x REAL NOT NULL,
    y REAL NOT NULL,
    w REAL NOT NULL,
    h REAL NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_reports_created
ON reports(created_at);

CREATE INDEX IF NOT EXISTS idx_reports_waterway
ON reports(waterway);

CREATE INDEX IF NOT EXISTS idx_reports_status
ON reports(status);

CREATE INDEX IF NOT EXISTS idx_reports_latlon
ON reports(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_detections_report
ON detections(report_id);

CREATE INDEX IF NOT EXISTS idx_history_report
ON status_history(report_id);
"""


# Columns added after the first version.
# Existing SQLite databases are automatically upgraded with ALTER TABLE.
MIGRATIONS = {
    "location_confirmed": "INTEGER NOT NULL DEFAULT 0",
    "status": "TEXT NOT NULL DEFAULT 'reported'",
    "status_updated_at": "TEXT",
    "location_name": "TEXT",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_store(data_dir: Path, database_url: str = ""):
    """
    Use Tiger Data / TimescaleDB when a database URL is configured.

    Otherwise use the local SQLite database.
    """
    if database_url:
        from .tigerstore import TigerStore

        return TigerStore(data_dir, database_url)

    return Store(data_dir)


class ImageStore:
    """
    Common image storage shared by the SQLite and Tiger Data backends.

    Photos remain on disk. Database implementations only control where the
    report metadata is stored.
    """

    backend = "none"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

        self.images_dir = data_dir / "images"
        self.annotated_dir = data_dir / "annotated"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

    # ---- images -------------------------------------------------------

    def save_images(
        self,
        report_id: str,
        image: Image.Image,
        detections: list[Detection],
    ) -> tuple[str, str]:

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
            draw.rectangle(
                [
                    d.x,
                    d.y,
                    d.x + d.w,
                    d.y + d.h,
                ],
                outline=(255, 64, 64),
                width=line,
            )

            draw.text(
                (
                    d.x + line,
                    max(0, d.y - 12),
                ),
                f"{d.label} {d.confidence:.2f}",
                fill=(255, 64, 64),
            )

        drawn.save(annotated, "JPEG", quality=88)

        return (
            original.relative_to(self.data_dir).as_posix(),
            annotated.relative_to(self.data_dir).as_posix(),
        )

    def abs_path(self, rel: str) -> Path:
        return self.data_dir / rel

    def close(self) -> None:
        pass


class Store(ImageStore):
    """
    SQLite storage backend.

    This is the default backend for local development and requires no external
    database configuration.
    """

    backend = "sqlite"

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)

        self._lock = threading.Lock()

        self.db = sqlite3.connect(
            data_dir / "reports.db",
            check_same_thread=False,
        )

        self.db.row_factory = sqlite3.Row

        self.db.execute("PRAGMA foreign_keys = ON")

        self.db.executescript(SCHEMA)

        # Upgrade databases created by earlier versions of RiverWatch.
        existing = {
            row["name"]
            for row in self.db.execute("PRAGMA table_info(reports)")
        }

        for column, declaration in MIGRATIONS.items():
            if column not in existing:
                self.db.execute(
                    f"ALTER TABLE reports "
                    f"ADD COLUMN {column} {declaration}"
                )

        self.db.executescript(INDEXES)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---- reports ------------------------------------------------------

    def insert(
        self,
        report: dict,
        detections: list[Detection],
    ) -> None:

        report = {
            **report,
            "status": "reported",
            "status_updated_at": report["created_at"],
        }

        # IMPORTANT:
        #
        # "detections" is included in the API report dictionary so the
        # frontend can immediately receive bounding boxes.
        #
        # It is NOT a column in the reports table. Individual detections
        # belong in the separate detections table below.
        #
        # Therefore both "recommendation" and "detections" must be excluded
        # from the dynamically generated reports INSERT.
        cols = [
            column
            for column in report
            if column not in ("recommendation", "detections")
        ]

        values = [report[column] for column in cols]

        values.append(
            json.dumps(report["recommendation"])
        )

        cols.append("recommendation_json")

        with self._lock, self.db:

            placeholders = ", ".join(
                "?" * len(cols)
            )

            self.db.execute(
                f"""
                INSERT INTO reports ({", ".join(cols)})
                VALUES ({placeholders})
                """,
                values,
            )

            self.db.executemany(
                """
                INSERT INTO detections
                    (report_id, label, confidence, x, y, w, h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        report["id"],
                        detection.label,
                        detection.confidence,
                        detection.x,
                        detection.y,
                        detection.w,
                        detection.h,
                    )
                    for detection in detections
                ],
            )

            self.db.execute(
                """
                INSERT INTO status_history
                    (
                        report_id,
                        status,
                        changed_at,
                        changed_by,
                        note
                    )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report["id"],
                    "reported",
                    report["created_at"],
                    report.get("reporter"),
                    "Report submitted",
                ),
            )

    def set_location(
        self,
        report_id: str,
        lat: float,
        lon: float,
        in_philadelphia: bool,
        location_name: str | None = None,
    ) -> None:

        with self._lock, self.db:

            self.db.execute(
                """
                UPDATE reports
                SET
                    latitude = ?,
                    longitude = ?,
                    location_source = 'map',
                    location_confirmed = 1,
                    in_philadelphia = ?,
                    location_name = COALESCE(?, location_name)
                WHERE id = ?
                """,
                (
                    lat,
                    lon,
                    in_philadelphia,
                    location_name,
                    report_id,
                ),
            )

    def set_status(
        self,
        report_id: str,
        status: str,
        changed_by: str | None,
        note: str | None,
    ) -> None:

        timestamp = now_iso()

        with self._lock, self.db:

            self.db.execute(
                """
                UPDATE reports
                SET
                    status = ?,
                    status_updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    timestamp,
                    report_id,
                ),
            )

            self.db.execute(
                """
                INSERT INTO status_history
                    (
                        report_id,
                        status,
                        changed_at,
                        changed_by,
                        note
                    )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    status,
                    timestamp,
                    changed_by,
                    note,
                ),
            )

    def get(
        self,
        report_id: str,
    ) -> dict | None:

        row = self.db.execute(
            """
            SELECT *
            FROM reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()

        return self._hydrate(row) if row else None

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        sort: str = "newest",
        **filters,
    ) -> tuple[list[dict], int]:

        clause, args = _where(**filters)

        order = {
            "newest": "created_at DESC",
            "oldest": "created_at ASC",
            "score": "score DESC, created_at DESC",
        }[sort]

        total = self.db.execute(
            f"""
            SELECT COUNT(*)
            FROM reports
            {clause}
            """,
            args,
        ).fetchone()[0]

        rows = self.db.execute(
            f"""
            SELECT *
            FROM reports
            {clause}
            ORDER BY {order}
            LIMIT ?
            OFFSET ?
            """,
            [
                *args,
                limit,
                offset,
            ],
        ).fetchall()

        return (
            [self._hydrate(row) for row in rows],
            total,
        )

    # ---- maps ---------------------------------------------------------

    def points(
        self,
        **filters,
    ) -> list[dict]:

        clause, args = _where(
            **filters,
            has_location=True,
        )

        rows = self.db.execute(
            f"""
            SELECT
                id,
                latitude,
                longitude,
                score,
                severity,
                status,
                waterway,
                created_at,
                item_count,
                location_confirmed
            FROM reports
            {clause}
            """,
            args,
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def bubbles(
        self,
        cell_deg: float,
        **filters,
    ) -> list[dict]:

        """
        Group located reports into a latitude/longitude grid.

        One bubble is returned for every non-empty cell.
        """

        cells: dict[
            tuple[int, int],
            list[dict],
        ] = {}

        for point in self.points(**filters):

            key = (
                math.floor(
                    point["latitude"] / cell_deg
                ),
                math.floor(
                    point["longitude"] / cell_deg
                ),
            )

            cells.setdefault(
                key,
                [],
            ).append(point)

        output = []

        for (i, j), points in cells.items():

            count = len(points)

            average_score = (
                sum(
                    point["score"]
                    for point in points
                )
                / count
            )

            output.append(
                {
                    "cell": f"{i}:{j}",

                    # Put the bubble at the average position of reports
                    # instead of at the grid corner.
                    "latitude": round(
                        sum(
                            point["latitude"]
                            for point in points
                        )
                        / count,
                        6,
                    ),

                    "longitude": round(
                        sum(
                            point["longitude"]
                            for point in points
                        )
                        / count,
                        6,
                    ),

                    "bounds": [
                        round(
                            i * cell_deg,
                            6,
                        ),
                        round(
                            j * cell_deg,
                            6,
                        ),
                        round(
                            (i + 1) * cell_deg,
                            6,
                        ),
                        round(
                            (j + 1) * cell_deg,
                            6,
                        ),
                    ],

                    "report_count": count,

                    "open_count": sum(
                        point["status"]
                        in OPEN_STATUSES
                        for point in points
                    ),

                    "avg_score": round(
                        average_score,
                        1,
                    ),

                    "max_score": max(
                        point["score"]
                        for point in points
                    ),

                    "avg_severity": severity_for(
                        round(average_score)
                    ),

                    "waterways": sorted(
                        {
                            point["waterway"]
                            for point in points
                            if point["waterway"]
                        }
                    ),

                    "latest_report_at": max(
                        point["created_at"]
                        for point in points
                    ),
                }
            )

        return sorted(
            output,
            key=lambda bubble: (
                -bubble["report_count"],
                -bubble["avg_score"],
            ),
        )

    # ---- stats --------------------------------------------------------

    def stats(self) -> dict:

        overall = self.db.execute(
            """
            SELECT
                COUNT(*) AS reports,
                AVG(score) AS avg_score,
                SUM(item_count) AS items
            FROM reports
            """
        ).fetchone()

        by_severity = dict(
            self.db.execute(
                """
                SELECT
                    severity,
                    COUNT(*)
                FROM reports
                GROUP BY severity
                """
            ).fetchall()
        )

        by_status = {
            status: 0
            for status in STATUSES
        }

        by_status.update(
            dict(
                self.db.execute(
                    """
                    SELECT
                        status,
                        COUNT(*)
                    FROM reports
                    GROUP BY status
                    """
                ).fetchall()
            )
        )

        placeholders = ", ".join(
            "?" * len(OPEN_STATUSES)
        )

        by_waterway = [
            dict(row)
            for row in self.db.execute(
                f"""
                SELECT
                    COALESCE(
                        waterway,
                        'Unspecified'
                    ) AS waterway,
                    COUNT(*) AS reports,
                    ROUND(
                        AVG(score),
                        1
                    ) AS avg_score,
                    MAX(score) AS max_score,
                    SUM(
                        status IN ({placeholders})
                    ) AS open_reports
                FROM reports
                GROUP BY waterway
                ORDER BY avg_score DESC
                """,
                OPEN_STATUSES,
            ).fetchall()
        ]

        return {
            "reports": overall["reports"],

            "open_reports": sum(
                by_status[status]
                for status in OPEN_STATUSES
            ),

            "avg_score": round(
                overall["avg_score"] or 0,
                1,
            ),

            "items_detected": (
                overall["items"] or 0
            ),

            "by_severity": by_severity,

            "by_status": by_status,

            "by_waterway": by_waterway,
        }

    # ---- time series --------------------------------------------------

    def trends(
        self,
        days: int = 90,
        waterway: str | None = None,
    ) -> list[dict]:

        """
        Daily litter and cleanup counts.

        SQLite calculates these directly with GROUP BY.
        Tiger Data uses continuous aggregates instead.
        """

        since = (
            datetime.now(timezone.utc)
            - timedelta(days=days)
        ).date().isoformat()

        if waterway:
            extra = " AND r.waterway = ?"
            args = [
                since,
                waterway,
            ]
        else:
            extra = ""
            args = [since]

        litter = self.db.execute(
            f"""
            SELECT
                substr(
                    r.created_at,
                    1,
                    10
                ) AS day,

                COUNT(*) AS reports,

                SUM(
                    r.score
                ) AS score_sum,

                MAX(
                    r.score
                ) AS max_score,

                SUM(
                    r.item_count
                ) AS items,

                SUM(
                    r.severity
                    IN ('high', 'severe')
                ) AS priority_reports,

                SUM(
                    r.hazard_suspected
                ) AS hazard_reports

            FROM reports r

            WHERE
                r.created_at >= ?
                {extra}

            GROUP BY day
            """,
            args,
        ).fetchall()

        cleaned = self.db.execute(
            f"""
            SELECT
                substr(
                    h.changed_at,
                    1,
                    10
                ) AS day,

                COUNT(*) AS cleaned

            FROM status_history h

            JOIN reports r
                ON r.id = h.report_id

            WHERE
                h.status = 'cleaned'
                AND h.changed_at >= ?
                {extra}

            GROUP BY day
            """,
            args,
        ).fetchall()

        return merge_days(
            [
                dict(row)
                for row in litter
            ],
            [
                dict(row)
                for row in cleaned
            ],
        )

    # River readings require Tiger Data.
    # SQLite intentionally returns empty results.

    def river_latest(self) -> list[dict]:
        return []

    def river_series(
        self,
        days: int,
        parameter: str,
        waterway: str | None = None,
    ) -> list[dict]:
        return []

    def db_info(self) -> dict:

        count = self.db.execute(
            """
            SELECT COUNT(*)
            FROM reports
            """
        ).fetchone()[0]

        return {
            "backend": self.backend,
            "reports": count,
        }

    # ---- hydration ----------------------------------------------------

    def _hydrate(
        self,
        row: sqlite3.Row,
    ) -> dict:

        report = dict(row)

        report["recommendation"] = json.loads(
            report.pop(
                "recommendation_json"
            )
        )

        for flag in (
            "in_philadelphia",
            "hazard_suspected",
            "model_is_stand_in",
            "location_confirmed",
        ):

            if report[flag] is not None:
                report[flag] = bool(
                    report[flag]
                )

        report["detections"] = [
            dict(detection)
            for detection in self.db.execute(
                """
                SELECT
                    label,
                    confidence,
                    x,
                    y,
                    w,
                    h
                FROM detections
                WHERE report_id = ?
                ORDER BY confidence DESC
                """,
                (
                    report["id"],
                ),
            ).fetchall()
        ]

        report["status_history"] = [
            dict(history)
            for history in self.db.execute(
                """
                SELECT
                    status,
                    changed_at,
                    changed_by,
                    note
                FROM status_history
                WHERE report_id = ?
                ORDER BY id
                """,
                (
                    report["id"],
                ),
            ).fetchall()
        ]

        return report


def merge_days(
    litter: list[dict],
    cleaned: list[dict],
) -> list[dict]:

    """
    Combine daily litter rows and cleanup counts into one row per day,
    ordered from oldest to newest.
    """

    days: dict[str, dict] = {}

    for row in litter:

        raw_day = row["day"]

        if isinstance(
            raw_day,
            datetime,
        ):
            key = raw_day.date().isoformat()
        else:
            key = str(raw_day)[:10]

        report_count = int(
            row["reports"]
        )

        days[key] = {
            "day": key,

            "reports": report_count,

            "avg_score": (
                round(
                    float(
                        row["score_sum"]
                    )
                    / report_count,
                    1,
                )
                if report_count
                else None
            ),

            "max_score": row["max_score"],

            "items": int(
                row["items"] or 0
            ),

            "priority_reports": int(
                row["priority_reports"] or 0
            ),

            "hazard_reports": int(
                row["hazard_reports"] or 0
            ),

            "cleaned": 0,
        }

    for row in cleaned:

        raw_day = row["day"]

        if isinstance(
            raw_day,
            datetime,
        ):
            key = raw_day.date().isoformat()
        else:
            key = str(raw_day)[:10]

        days.setdefault(
            key,
            {
                "day": key,
                "reports": 0,
                "avg_score": None,
                "max_score": None,
                "items": 0,
                "priority_reports": 0,
                "hazard_reports": 0,
                "cleaned": 0,
            },
        )

        days[key]["cleaned"] = int(
            row["cleaned"]
        )

    return [
        days[key]
        for key in sorted(days)
    ]


def _where(
    *,
    waterway=None,
    severity=None,
    status=None,
    min_score=None,
    since=None,
    until=None,
    bbox=None,
    has_location=False,
) -> tuple[str, list]:

    """
    Build a WHERE clause shared by list/export/map queries.

    severity/status accept either a single value or a list.
    status="open" expands to every not-yet-closed status.

    bbox is:
        (min_lat, min_lon, max_lat, max_lon)
    """

    where = []
    args = []

    if waterway:
        where.append(
            "waterway = ?"
        )
        args.append(
            waterway
        )

    for column, values in (
        (
            "severity",
            severity,
        ),
        (
            "status",
            status,
        ),
    ):

        if values:

            if isinstance(
                values,
                str,
            ):
                values = [values]
            else:
                values = list(values)

            if (
                column == "status"
                and "open" in values
            ):

                values = [
                    value
                    for value in values
                    if value != "open"
                ] + OPEN_STATUSES

            placeholders = ", ".join(
                "?" * len(values)
            )

            where.append(
                f"{column} IN ({placeholders})"
            )

            args.extend(values)

    if min_score is not None:
        where.append(
            "score >= ?"
        )
        args.append(
            min_score
        )

    if since:
        where.append(
            "created_at >= ?"
        )
        args.append(
            since
        )

    if until:
        where.append(
            "created_at <= ?"
        )
        args.append(
            until
        )

    if has_location or bbox:
        where.append(
            "latitude IS NOT NULL "
            "AND longitude IS NOT NULL"
        )

    if bbox:

        where.append(
            "latitude BETWEEN ? AND ? "
            "AND longitude BETWEEN ? AND ?"
        )

        args.extend(
            [
                bbox[0],
                bbox[2],
                bbox[1],
                bbox[3],
            ]
        )

    clause = (
        f"WHERE {' AND '.join(where)}"
        if where
        else ""
    )

    return clause, args