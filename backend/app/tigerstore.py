"""Tiger Data store: reports, cleanups and river sensor readings in TimescaleDB (PostgreSQL).

Used when PWW_DATABASE_URL is set, e.g. a Tiger Cloud service URL. Same interface as the SQLite
Store in storage.py, plus the time-series features the dashboard's trend charts use:

  hypertables             reports (created_at), status_history (changed_at), river_readings (time)
  relational tables       detections, river_sites (joined to the hypertables with plain SQL)
  continuous aggregates   litter_daily, cleanups_daily, river_hourly -> river_daily (hierarchical)
  compression             river_readings after 7 days, status_history after 30, reports after 90

Continuous aggregates are created with real-time aggregation on, so charts include reports filed
seconds ago while older buckets come pre-computed.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .storage import OPEN_STATUSES, STATUSES, ImageStore, merge_days, now_iso
from .scoring import severity_for

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS reports (
    id                  TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL,
    captured_at         TEXT,
    waterway            TEXT,
    location_name       TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    location_source     TEXT,
    location_confirmed  BOOLEAN NOT NULL DEFAULT FALSE,
    in_philadelphia     BOOLEAN,
    notes               TEXT,
    reporter            TEXT,
    hazard_suspected    BOOLEAN NOT NULL DEFAULT FALSE,
    image_path          TEXT NOT NULL,
    annotated_path      TEXT NOT NULL,
    image_width         INTEGER NOT NULL,
    image_height        INTEGER NOT NULL,
    model_name          TEXT NOT NULL,
    model_is_stand_in   BOOLEAN NOT NULL,
    score               SMALLINT NOT NULL,
    severity            TEXT NOT NULL,
    item_count          INTEGER NOT NULL,
    coverage_pct        REAL NOT NULL,
    authority_level     SMALLINT NOT NULL,
    recommendation      JSONB NOT NULL,
    status              TEXT NOT NULL DEFAULT 'reported',
    status_updated_at   TIMESTAMPTZ,
    PRIMARY KEY (id, created_at)
);
SELECT create_hypertable('reports', by_range('created_at', INTERVAL '30 days'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_waterway ON reports (waterway, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_score ON reports (score DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS detections (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    report_id   TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_report ON detections (report_id);

CREATE TABLE IF NOT EXISTS status_history (
    report_id   TEXT NOT NULL,
    status      TEXT NOT NULL,
    changed_at  TIMESTAMPTZ NOT NULL,
    changed_by  TEXT,
    note        TEXT,
    waterway    TEXT
);
SELECT create_hypertable('status_history', by_range('changed_at', INTERVAL '30 days'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_history_report ON status_history (report_id, changed_at);

CREATE TABLE IF NOT EXISTS river_sites (
    site_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    waterway    TEXT,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS river_readings (
    time        TIMESTAMPTZ NOT NULL,
    site_id     TEXT NOT NULL,
    parameter   TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    UNIQUE (site_id, parameter, time)
);
SELECT create_hypertable('river_readings', by_range('time', INTERVAL '7 days'), if_not_exists => TRUE);
"""

# Continuous aggregates. Sums and counts (not only averages) are kept so buckets can be combined
# correctly across waterways or rolled up again.
AGGREGATES = [
    ("litter_daily", """
        CREATE MATERIALIZED VIEW IF NOT EXISTS litter_daily
        WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
        SELECT time_bucket(INTERVAL '1 day', created_at) AS day,
               waterway,
               count(*)                                                 AS reports,
               sum(score)                                               AS score_sum,
               max(score)                                               AS max_score,
               sum(item_count)                                          AS items,
               count(*) FILTER (WHERE severity IN ('high', 'severe'))   AS priority_reports,
               count(*) FILTER (WHERE hazard_suspected)                 AS hazard_reports
        FROM reports
        GROUP BY 1, 2
        WITH NO DATA
    """, "INTERVAL '120 days'", "INTERVAL '1 hour'"),
    ("cleanups_daily", """
        CREATE MATERIALIZED VIEW IF NOT EXISTS cleanups_daily
        WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
        SELECT time_bucket(INTERVAL '1 day', changed_at) AS day,
               waterway,
               status,
               count(*) AS changes
        FROM status_history
        GROUP BY 1, 2, 3
        WITH NO DATA
    """, "INTERVAL '120 days'", "INTERVAL '1 hour'"),
    ("river_hourly", """
        CREATE MATERIALIZED VIEW IF NOT EXISTS river_hourly
        WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
        SELECT time_bucket(INTERVAL '1 hour', time) AS hour,
               site_id,
               parameter,
               sum(value)   AS value_sum,
               count(*)     AS samples,
               min(value)   AS min_value,
               max(value)   AS max_value
        FROM river_readings
        GROUP BY 1, 2, 3
        WITH NO DATA
    """, "INTERVAL '7 days'", "INTERVAL '15 minutes'"),
    # Hierarchical: a continuous aggregate built on river_hourly, not on the raw readings.
    ("river_daily", """
        CREATE MATERIALIZED VIEW IF NOT EXISTS river_daily
        WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
        SELECT time_bucket(INTERVAL '1 day', hour) AS day,
               site_id,
               parameter,
               sum(value_sum)   AS value_sum,
               sum(samples)     AS samples,
               min(min_value)   AS min_value,
               max(max_value)   AS max_value
        FROM river_hourly
        GROUP BY 1, 2, 3
        WITH NO DATA
    """, "INTERVAL '120 days'", "INTERVAL '1 hour'"),
]

# (table, segment by, order by, compress after)
COMPRESSION = [
    ("river_readings", "site_id, parameter", "time DESC", "INTERVAL '7 days'"),
    ("status_history", "status", "report_id, changed_at", "INTERVAL '30 days'"),
    ("reports", "waterway", "created_at DESC", "INTERVAL '90 days'"),
]

REPORT_COLUMNS = [
    "id", "created_at", "captured_at", "waterway", "location_name", "latitude", "longitude",
    "location_source", "location_confirmed", "in_philadelphia", "notes", "reporter", "hazard_suspected",
    "image_path", "annotated_path", "image_width", "image_height", "model_name", "model_is_stand_in",
    "score", "severity", "item_count", "coverage_pct", "authority_level", "recommendation",
    "status", "status_updated_at",
]


def _iso(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds") if isinstance(value, datetime) else value


class TigerStore(ImageStore):
    backend = "tigerdata"

    def __init__(self, data_dir: Path, database_url: str):
        super().__init__(data_dir)
        self.pool = ConnectionPool(database_url, min_size=1, max_size=8, open=True,
                                   kwargs={"row_factory": dict_row, "autocommit": True})
        self._setup()

    def close(self) -> None:
        self.pool.close()

    def _setup(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)
            for name, ddl, start, end in AGGREGATES:
                conn.execute(ddl)
                conn.execute(
                    f"SELECT add_continuous_aggregate_policy('{name}', start_offset => {start}, "
                    f"end_offset => {end}, schedule_interval => INTERVAL '15 minutes', if_not_exists => TRUE)"
                )
            enabled = {r["hypertable_name"] for r in conn.execute(
                "SELECT hypertable_name FROM timescaledb_information.hypertables WHERE compression_enabled")}
            for table, segment_by, order_by, after in COMPRESSION:
                if table not in enabled:
                    conn.execute(f"ALTER TABLE {table} SET (timescaledb.compress, "
                                 f"timescaledb.compress_segmentby = '{segment_by}', "
                                 f"timescaledb.compress_orderby = '{order_by}')")
                conn.execute(f"SELECT add_compression_policy('{table}', {after}, if_not_exists => TRUE)")

    # ---- reports ------------------------------------------------------

    def insert(self, report: dict, detections) -> None:
        report = {**report, "status": "reported", "status_updated_at": report["created_at"]}
        cols = [c for c in REPORT_COLUMNS if c in report]
        values = [json.dumps(report[c]) if c == "recommendation" else report[c] for c in cols]
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                f"INSERT INTO reports ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(values))})",
                values,
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO detections (report_id, label, confidence, x, y, w, h) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [(report["id"], d.label, d.confidence, d.x, d.y, d.w, d.h) for d in detections],
                )
            conn.execute(
                "INSERT INTO status_history (report_id, status, changed_at, changed_by, note, waterway) "
                "VALUES (%s, 'reported', %s, %s, 'Report submitted', %s)",
                (report["id"], report["created_at"], report.get("reporter"), report.get("waterway")),
            )

    def set_location(self, report_id: str, lat: float, lon: float, in_philadelphia: bool,
                     location_name: str | None = None) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE reports SET latitude = %s, longitude = %s, location_source = 'map', location_confirmed = TRUE, "
                "in_philadelphia = %s, location_name = COALESCE(%s, location_name) WHERE id = %s",
                (lat, lon, in_philadelphia, location_name, report_id),
            )

    def set_status(self, report_id: str, status: str, changed_by: str | None, note: str | None) -> None:
        ts = now_iso()
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "UPDATE reports SET status = %s, status_updated_at = %s WHERE id = %s RETURNING waterway",
                (status, ts, report_id),
            ).fetchone()
            conn.execute(
                "INSERT INTO status_history (report_id, status, changed_at, changed_by, note, waterway) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (report_id, status, ts, changed_by, note, row["waterway"] if row else None),
            )

    def get(self, report_id: str) -> dict | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM reports WHERE id = %s", (report_id,)).fetchone()
            return self._hydrate(conn, [row])[0] if row else None

    def list(self, *, limit: int = 50, offset: int = 0, sort: str = "newest", **filters) -> tuple[list[dict], int]:
        clause, args = _where(**filters)
        order = {
            "newest": "created_at DESC",
            "oldest": "created_at ASC",
            "score": "score DESC, created_at DESC",
        }[sort]
        with self.pool.connection() as conn:
            total = conn.execute(f"SELECT count(*) AS n FROM reports {clause}", args).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM reports {clause} ORDER BY {order} LIMIT %s OFFSET %s", [*args, limit, offset]
            ).fetchall()
            return self._hydrate(conn, rows), total

    def _hydrate(self, conn, rows: list[dict]) -> list[dict]:
        """Attach detections and status history with one query each, not one per report."""
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        detections: dict[str, list] = {i: [] for i in ids}
        for d in conn.execute(
            "SELECT report_id, label, confidence, x, y, w, h FROM detections WHERE report_id = ANY(%s) "
            "ORDER BY confidence DESC", (ids,)
        ):
            detections[d.pop("report_id")].append(d)
        history: dict[str, list] = {i: [] for i in ids}
        for h in conn.execute(
            "SELECT report_id, status, changed_at, changed_by, note FROM status_history WHERE report_id = ANY(%s) "
            "ORDER BY changed_at", (ids,)
        ):
            rid = h.pop("report_id")
            h["changed_at"] = _iso(h["changed_at"])
            history[rid].append(h)
        out = []
        for r in rows:
            report = {k: _iso(v) for k, v in r.items()}
            report["detections"] = detections[r["id"]]
            report["status_history"] = history[r["id"]]
            out.append(report)
        return out

    # ---- maps ---------------------------------------------------------

    def points(self, **filters) -> list[dict]:
        clause, args = _where(**filters, has_location=True)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT id, latitude, longitude, score, severity, status, waterway, created_at, item_count, "
                f"location_confirmed FROM reports {clause}", args
            ).fetchall()
        return [{k: _iso(v) for k, v in r.items()} for r in rows]

    def bubbles(self, cell_deg: float, **filters) -> list[dict]:
        """Grid the located reports in SQL; one row per non-empty cell."""
        clause, args = _where(**filters, has_location=True)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT floor(latitude / %s)::bigint AS i, floor(longitude / %s)::bigint AS j,
                       avg(latitude) AS latitude, avg(longitude) AS longitude,
                       count(*) AS report_count,
                       count(*) FILTER (WHERE status = ANY(%s)) AS open_count,
                       avg(score) AS avg_score, max(score) AS max_score,
                       array_remove(array_agg(DISTINCT waterway), NULL) AS waterways,
                       max(created_at) AS latest_report_at
                FROM reports {clause}
                GROUP BY 1, 2
                ORDER BY report_count DESC, avg_score DESC
                """,
                [cell_deg, cell_deg, OPEN_STATUSES, *args],
            ).fetchall()
        return [{
            "cell": f"{r['i']}:{r['j']}",
            "latitude": round(r["latitude"], 6),
            "longitude": round(r["longitude"], 6),
            "bounds": [round(r["i"] * cell_deg, 6), round(r["j"] * cell_deg, 6),
                       round((r["i"] + 1) * cell_deg, 6), round((r["j"] + 1) * cell_deg, 6)],
            "report_count": r["report_count"],
            "open_count": r["open_count"],
            "avg_score": round(float(r["avg_score"]), 1),
            "max_score": r["max_score"],
            "avg_severity": severity_for(round(float(r["avg_score"]))),
            "waterways": sorted(r["waterways"]),
            "latest_report_at": _iso(r["latest_report_at"]),
        } for r in rows]

    # ---- stats --------------------------------------------------------

    def stats(self) -> dict:
        with self.pool.connection() as conn:
            overall = conn.execute(
                "SELECT count(*) AS reports, avg(score) AS avg_score, sum(item_count) AS items FROM reports"
            ).fetchone()
            by_severity = {r["severity"]: r["n"] for r in conn.execute(
                "SELECT severity, count(*) AS n FROM reports GROUP BY severity")}
            by_status = {s: 0 for s in STATUSES}
            by_status.update({r["status"]: r["n"] for r in conn.execute(
                "SELECT status, count(*) AS n FROM reports GROUP BY status")})
            by_waterway = [
                {**r, "avg_score": float(r["avg_score"])} for r in conn.execute(
                    "SELECT COALESCE(waterway, 'Unspecified') AS waterway, count(*) AS reports, "
                    "round(avg(score), 1) AS avg_score, max(score) AS max_score, "
                    "count(*) FILTER (WHERE status = ANY(%s)) AS open_reports "
                    "FROM reports GROUP BY 1 ORDER BY avg_score DESC", (OPEN_STATUSES,)
                )
            ]
        return {
            "reports": overall["reports"],
            "open_reports": sum(by_status[s] for s in OPEN_STATUSES),
            "avg_score": round(float(overall["avg_score"] or 0), 1),
            "items_detected": int(overall["items"] or 0),
            "by_severity": by_severity,
            "by_status": by_status,
            "by_waterway": by_waterway,
        }

    # ---- time series --------------------------------------------------

    def trends(self, days: int = 90, waterway: str | None = None) -> list[dict]:
        """Daily litter and cleanup counts, read from the continuous aggregates."""
        where, args = "day >= now() - make_interval(days => %s)", [days]
        if waterway:
            where += " AND waterway = %s"
            args.append(waterway)
        with self.pool.connection() as conn:
            litter = conn.execute(
                f"SELECT day, sum(reports) AS reports, sum(score_sum) AS score_sum, max(max_score) AS max_score, "
                f"sum(items) AS items, sum(priority_reports) AS priority_reports, "
                f"sum(hazard_reports) AS hazard_reports FROM litter_daily WHERE {where} GROUP BY day", args
            ).fetchall()
            cleaned = conn.execute(
                f"SELECT day, sum(changes) AS cleaned FROM cleanups_daily WHERE {where} AND status = 'cleaned' "
                f"GROUP BY day", args
            ).fetchall()
        return merge_days(litter, cleaned)

    def upsert_river_sites(self, sites: list[dict]) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO river_sites (site_id, name, waterway, latitude, longitude) "
                "VALUES (%(site_id)s, %(name)s, %(waterway)s, %(latitude)s, %(longitude)s) "
                "ON CONFLICT (site_id) DO UPDATE SET name = EXCLUDED.name, waterway = EXCLUDED.waterway, "
                "latitude = COALESCE(EXCLUDED.latitude, river_sites.latitude), "
                "longitude = COALESCE(EXCLUDED.longitude, river_sites.longitude)",
                sites,
            )

    def insert_river_readings(self, readings: list[tuple]) -> int:
        """(time, site_id, parameter, value) rows; readings already stored are skipped. Returns rows added."""
        if not readings:
            return 0
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("CREATE TEMP TABLE incoming (LIKE river_readings) ON COMMIT DROP")
            with conn.cursor() as cur, cur.copy("COPY incoming (time, site_id, parameter, value) FROM STDIN") as copy:
                for row in readings:
                    copy.write_row(row)
            added = conn.execute(
                "INSERT INTO river_readings SELECT * FROM incoming ON CONFLICT DO NOTHING"
            ).rowcount
        return added

    def river_latest(self) -> list[dict]:
        """Most recent reading of each parameter at each site."""
        with self.pool.connection() as conn:
            rows = conn.execute("""
                SELECT s.site_id, s.name, s.waterway, r.parameter, r.value, r.time
                FROM river_sites s
                CROSS JOIN LATERAL (
                    SELECT DISTINCT ON (parameter) parameter, value, time
                    FROM river_readings
                    WHERE site_id = s.site_id AND time > now() - INTERVAL '3 days'
                    ORDER BY parameter, time DESC
                ) r
                ORDER BY s.site_id, r.parameter
            """).fetchall()
        return [{**r, "time": _iso(r["time"])} for r in rows]

    def river_series(self, days: int, parameter: str, waterway: str | None = None) -> list[dict]:
        """Daily mean/min/max of one parameter per site, from the hierarchical river_daily aggregate."""
        where, args = "d.day >= now() - make_interval(days => %s) AND d.parameter = %s", [days, parameter]
        if waterway:
            where += " AND s.waterway = %s"
            args.append(waterway)
        with self.pool.connection() as conn:
            rows = conn.execute(f"""
                SELECT d.day, d.site_id, s.name, s.waterway,
                       d.value_sum / NULLIF(d.samples, 0) AS mean, d.min_value AS min, d.max_value AS max
                FROM river_daily d JOIN river_sites s USING (site_id)
                WHERE {where}
                ORDER BY d.day
            """, args).fetchall()
        return [{**r, "day": r["day"].date().isoformat()} for r in rows]

    # ---- database info for the dashboard ------------------------------

    def db_info(self) -> dict:
        with self.pool.connection() as conn:
            version = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").fetchone()["extversion"]
            hypertables = []
            for h in conn.execute(
                "SELECT hypertable_name, num_chunks, compression_enabled FROM timescaledb_information.hypertables "
                "WHERE hypertable_schema = 'public' ORDER BY hypertable_name"
            ).fetchall():
                name = h["hypertable_name"]
                rows = conn.execute(f"SELECT count(*) AS n FROM {name}").fetchone()["n"]
                size = conn.execute("SELECT hypertable_size(%s::regclass) AS b", (name,)).fetchone()["b"]
                comp = conn.execute(
                    "SELECT coalesce(sum(before_compression_total_bytes), 0) AS before, "
                    "coalesce(sum(after_compression_total_bytes), 0) AS after, "
                    "count(*) FILTER (WHERE compression_status = 'Compressed') AS compressed_chunks "
                    "FROM chunk_compression_stats(%s::regclass)", (name,)
                ).fetchone()
                hypertables.append({
                    "name": name, "rows": rows, "chunks": h["num_chunks"], "total_bytes": size,
                    "compression_enabled": h["compression_enabled"],
                    "compressed_chunks": comp["compressed_chunks"],
                    "bytes_before_compression": int(comp["before"]),
                    "bytes_after_compression": int(comp["after"]),
                })
            aggregates = [r["view_name"] for r in conn.execute(
                "SELECT view_name FROM timescaledb_information.continuous_aggregates ORDER BY view_name")]

            # The dashboard trend query, timed against the aggregate and against the raw hypertable.
            start = time.perf_counter()
            conn.execute("SELECT day, sum(reports), sum(score_sum) FROM litter_daily "
                         "WHERE day >= now() - INTERVAL '90 days' GROUP BY day").fetchall()
            aggregate_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            conn.execute("SELECT time_bucket(INTERVAL '1 day', created_at) AS day, count(*), sum(score) FROM reports "
                         "WHERE created_at >= now() - INTERVAL '90 days' GROUP BY day").fetchall()
            raw_ms = (time.perf_counter() - start) * 1000
        return {
            "backend": self.backend,
            "timescaledb_version": version,
            "hypertables": hypertables,
            "continuous_aggregates": aggregates,
            "trend_query_ms": {"continuous_aggregate": round(aggregate_ms, 2), "raw_hypertable": round(raw_ms, 2)},
        }


def refresh_aggregate(conn, view: str, attempts: int = 100) -> None:
    """Refresh a continuous aggregate now, waiting out a background policy refresh running at the same moment."""
    import psycopg
    for _ in range(attempts):
        try:
            conn.execute(f"CALL refresh_continuous_aggregate('{view}', NULL, NULL)")
            return
        except psycopg.errors.LockNotAvailable:
            time.sleep(0.1)
    raise TimeoutError(f"continuous aggregate {view} is still being refreshed by a background job")


def _where(*, waterway=None, severity=None, status=None, min_score=None, since=None, until=None,
           bbox=None, has_location=False) -> tuple[str, list]:
    """PostgreSQL version of storage._where (same filters, %s placeholders)."""
    where, args = [], []
    if waterway:
        where.append("waterway = %s"); args.append(waterway)
    for col, values in (("severity", severity), ("status", status)):
        if values:
            values = [values] if isinstance(values, str) else list(values)
            if col == "status" and "open" in values:
                values = [v for v in values if v != "open"] + OPEN_STATUSES
            where.append(f"{col} = ANY(%s)"); args.append(values)
    if min_score is not None:
        where.append("score >= %s"); args.append(min_score)
    if since:
        where.append("created_at >= %s::timestamptz"); args.append(since)
    if until:
        where.append("created_at <= %s::timestamptz"); args.append(until)
    if has_location or bbox:
        where.append("latitude IS NOT NULL AND longitude IS NOT NULL")
    if bbox:
        where.append("latitude BETWEEN %s AND %s AND longitude BETWEEN %s AND %s")
        args.extend([bbox[0], bbox[2], bbox[1], bbox[3]])
    return (f"WHERE {' AND '.join(where)}" if where else ""), args
