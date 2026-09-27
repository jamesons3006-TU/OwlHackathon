"""Fill the database with synthetic demo reports, for demos and load tests.

Uses Tiger Data when PWW_DATABASE_URL is set, otherwise the local SQLite file.

    python -m app.seed                      # 400 reports over the last 120 days
    python -m app.seed --reports 200000     # load test: bulk COPY, then compare aggregate vs raw query time
    python -m app.seed --clear              # remove every seeded report

Seeded reports are clearly marked: model_name "demo-seed", model_is_stand_in true, a note saying they are
synthetic, and a shared placeholder image. They are made-up data; never present them as real sightings.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

from PIL import Image, ImageDraw

from . import config
from .recommendations import recommend
from .scoring import severity_for
from .storage import open_store

SEED_MODEL = "demo-seed"
NOTE = "Synthetic demo report (python -m app.seed). Not a real sighting."

# Public places along Philadelphia waterways: (name, waterway, lat, lon, typical score)
SPOTS = [
    ("Schuylkill Banks", "Schuylkill River", 39.9526, -75.1805, 55),
    ("Bartram's Garden", "Schuylkill River", 39.9320, -75.2110, 60),
    ("Grays Ferry Crescent", "Schuylkill River", 39.9395, -75.1990, 65),
    ("Fairmount Water Works", "Schuylkill River", 39.9660, -75.1830, 30),
    ("Boathouse Row", "Schuylkill River", 39.9690, -75.1860, 35),
    ("Kelly Drive", "Schuylkill River", 39.9800, -75.1920, 30),
    ("Manayunk Towpath", "Schuylkill River", 40.0255, -75.2230, 40),
    ("Penn's Landing", "Delaware River", 39.9455, -75.1400, 55),
    ("Race Street Pier", "Delaware River", 39.9530, -75.1390, 45),
    ("Spruce Street Harbor Park", "Delaware River", 39.9445, -75.1415, 50),
    ("Washington Avenue Pier", "Delaware River", 39.9330, -75.1420, 50),
    ("Penn Treaty Park", "Delaware River", 39.9660, -75.1280, 35),
    ("Pennypack on the Delaware", "Delaware River", 40.0290, -75.0250, 60),
    ("Valley Green", "Wissahickon Creek", 40.0550, -75.2190, 20),
    ("Wissahickon at Ridge Ave", "Wissahickon Creek", 40.0140, -75.2060, 35),
    ("Pennypack Park", "Pennypack Creek", 40.0620, -75.0470, 30),
    ("Cobbs Creek Park", "Cobbs Creek", 39.9460, -75.2470, 55),
    ("Tacony Creek Park", "Tacony-Frankford Creek", 40.0330, -75.1110, 60),
    ("Poquessing Creek Park", "Poquessing Creek", 40.0850, -74.9830, 30),
    ("Darby Creek at Eastwick", "Darby Creek", 39.9030, -75.2470, 50),
]
REPORTERS = ["Temple Eco Club", "Ana R.", "Jordan K.", "Priya S.", "Dev M.", "Sam T.",
             "Schuylkill cleanup crew", "Delaware watershed volunteers", None]


def placeholder(data_dir) -> str:
    rel = "demo/placeholder.jpg"
    path = data_dir / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (960, 540), (8, 41, 54))
        ImageDraw.Draw(img).text((380, 260), "Synthetic demo report", fill=(135, 170, 184))
        img.save(path, "JPEG", quality=85)
    return rel


def make_report(rng: random.Random, now: datetime, days: int, image: str) -> tuple[dict, list[tuple], list[tuple]]:
    name, waterway, lat, lon, typical = rng.choice(SPOTS)
    created = now - timedelta(seconds=rng.uniform(0, days * 86400))
    score = max(0, min(100, round(rng.gauss(typical, 15))))
    severity = severity_for(score)
    items = 0 if score == 0 else max(1, round(score / 7 + rng.uniform(-2, 2)))
    rec = recommend(severity)
    rid = uuid.uuid4().hex

    # Older reports are more likely to have moved through the cleanup workflow.
    age_days = (now - created).days
    history = [(rid, "reported", created, None, "Report submitted", waterway)]
    status = "reported"
    if age_days > 3 and rng.random() < min(0.85, age_days / 40):
        for step in ("verified", "scheduled", "cleaned"):
            status = step
            # Never in the future.
            changed = min(now, created + timedelta(days=len(history) * rng.uniform(1, 4)))
            history.append((rid, step, changed, "Demo cleanup team", None, waterway))
            if rng.random() < 0.25:
                break
    report = {
        "id": rid, "created_at": created, "waterway": waterway, "location_name": name,
        "latitude": lat + rng.uniform(-0.002, 0.002), "longitude": lon + rng.uniform(-0.002, 0.002),
        "location_source": "map", "location_confirmed": True, "in_philadelphia": True,
        "notes": NOTE, "reporter": rng.choice(REPORTERS), "hazard_suspected": rng.random() < 0.02,
        "image_path": image, "annotated_path": image, "image_width": 960, "image_height": 540,
        "model_name": SEED_MODEL, "model_is_stand_in": True, "score": score, "severity": severity,
        "item_count": items, "coverage_pct": round(score / 12, 2), "authority_level": rec["authority_level"],
        "recommendation": json.dumps(rec), "status": status, "status_updated_at": history[-1][2],
    }
    boxes = [(rid, "garbage", round(rng.uniform(0.3, 0.95), 3), rng.uniform(0, 900), rng.uniform(0, 500),
              rng.uniform(10, 60), rng.uniform(10, 60)) for _ in range(items)]
    return report, boxes, history


def seed(store, count: int, days: int, rng_seed: int | None = None) -> dict:
    rng = random.Random(rng_seed)
    now = datetime.now(timezone.utc)
    image = placeholder(store.data_dir)
    start = time.perf_counter()
    cols = None
    with store.pool.connection() as conn, conn.cursor() as cur:
        batch = 20_000
        for offset in range(0, count, batch):
            rows = [make_report(rng, now, days, image) for _ in range(min(batch, count - offset))]
            cols = cols or list(rows[0][0])
            with cur.copy(f"COPY reports ({', '.join(cols)}) FROM STDIN") as copy:
                for report, _, _ in rows:
                    copy.write_row([report[c] for c in cols])
            with cur.copy("COPY detections (report_id, label, confidence, x, y, w, h) FROM STDIN") as copy:
                for _, boxes, _ in rows:
                    for box in boxes:
                        copy.write_row(box)
            with cur.copy("COPY status_history (report_id, status, changed_at, changed_by, note, waterway) "
                          "FROM STDIN") as copy:
                for _, _, history in rows:
                    for entry in history:
                        copy.write_row(entry)
        load_s = time.perf_counter() - start
    from .tigerstore import refresh_aggregate
    with store.pool.connection() as conn:
        for view in ("litter_daily", "cleanups_daily"):
            refresh_aggregate(conn, view)
    return {"reports_added": count, "load_seconds": round(load_s, 2),
            "reports_per_second": round(count / load_s) if load_s else None}


def clear(store) -> int:
    with store.pool.connection() as conn, conn.transaction():
        ids = "SELECT id FROM reports WHERE model_name = %s"
        conn.execute(f"DELETE FROM detections WHERE report_id IN ({ids})", (SEED_MODEL,))
        conn.execute(f"DELETE FROM status_history WHERE report_id IN ({ids})", (SEED_MODEL,))
        removed = conn.execute("DELETE FROM reports WHERE model_name = %s", (SEED_MODEL,)).rowcount
    from .tigerstore import refresh_aggregate
    with store.pool.connection() as conn:
        for view in ("litter_daily", "cleanups_daily"):
            refresh_aggregate(conn, view)
    return removed


def seed_local(store, count: int, days: int, rng_seed: int | None = None) -> dict:
    """seed() for the local SQLite store."""
    rng = random.Random(rng_seed)
    now = datetime.now(timezone.utc)
    image = placeholder(store.data_dir)
    start = time.perf_counter()

    def iso(value):
        return value.isoformat(timespec="seconds") if isinstance(value, datetime) else value

    with store._lock, store.db:
        for _ in range(count):
            report, boxes, history = make_report(rng, now, days, image)
            row = {key: iso(value) for key, value in report.items()}
            row["recommendation_json"] = row.pop("recommendation")
            for flag in ("location_confirmed", "in_philadelphia", "hazard_suspected", "model_is_stand_in"):
                row[flag] = int(row[flag])
            cols = list(row)
            store.db.execute(
                f"INSERT INTO reports ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols],
            )
            store.db.executemany(
                "INSERT INTO detections (report_id, label, confidence, x, y, w, h) VALUES (?, ?, ?, ?, ?, ?, ?)",
                boxes,
            )
            store.db.executemany(
                "INSERT INTO status_history (report_id, status, changed_at, changed_by, note) VALUES (?, ?, ?, ?, ?)",
                [(rid, status, iso(at), by, note) for rid, status, at, by, note, _ in history],
            )
    load_s = time.perf_counter() - start
    return {"reports_added": count, "load_seconds": round(load_s, 2)}


def clear_local(store) -> int:
    """clear() for the local SQLite store."""
    with store._lock, store.db:
        ids = "SELECT id FROM reports WHERE model_name = ?"
        store.db.execute(f"DELETE FROM detections WHERE report_id IN ({ids})", (SEED_MODEL,))
        store.db.execute(f"DELETE FROM status_history WHERE report_id IN ({ids})", (SEED_MODEL,))
        return store.db.execute("DELETE FROM reports WHERE model_name = ?", (SEED_MODEL,)).rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic demo reports (Tiger Data or the local SQLite file)")
    parser.add_argument("--reports", type=int, default=400)
    parser.add_argument("--days", type=int, default=120, help="spread reports over this many past days")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for repeatable data")
    parser.add_argument("--clear", action="store_true", help="remove all seeded reports and exit")
    parser.add_argument("--if-empty", action="store_true", help="only add reports if the database has none")
    args = parser.parse_args()
    store = open_store(config.DATA_DIR, config.DATABASE_URL)
    tiger = store.backend == "tigerdata"
    try:
        if args.if_empty and store.list(limit=1)[1]:
            print("The database already has reports; not adding demo data.")
            return
        if args.clear:
            print({"reports_removed": clear(store) if tiger else clear_local(store)})
            return
        if tiger:
            print(seed(store, args.reports, args.days, args.seed))
            print({"trend_query_ms": store.db_info()["trend_query_ms"]})
        else:
            print(seed_local(store, args.reports, args.days, args.seed))
    finally:
        store.close()


if __name__ == "__main__":
    main()