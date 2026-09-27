"""Trends, river readings and Tiger Data internals. Tiger-only tests need PWW_TEST_DATABASE_URL."""
import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import river
from app.detector import Detection
from app.main import create_app
from conftest import requires_tiger


class FixedDetector:
    name = "fixed"
    is_stand_in = False

    def detect(self, image):
        n = image.getpixel((0, 0))[0] // 10
        return [Detection("garbage", 0.9, x=i * 60, y=10, w=50, h=50) for i in range(n)]


def _jpeg(items):
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (items * 10 + 5, 90, 140)).save(buf, "JPEG", quality=100)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(detector=FixedDetector(), data_dir=tmp_path)) as c:
        yield c


def _upload(client, items, waterway):
    r = client.post("/api/reports", files={"image": ("p.jpg", _jpeg(items), "image/jpeg")},
                    data={"waterway": waterway, "latitude": "39.95", "longitude": "-75.18"})
    assert r.status_code == 201, r.text
    return r.json()


def test_trends_by_day_and_waterway(client):
    a = _upload(client, 2, "Schuylkill River")
    b = _upload(client, 8, "Schuylkill River")
    _upload(client, 4, "Delaware River")
    client.patch(f"/api/reports/{a['id']}/status", json={"status": "cleaned"})

    today = datetime.now(timezone.utc).date().isoformat()
    body = client.get("/api/trends?days=7").json()
    assert [d["day"] for d in body["days"]] == [today]
    day = body["days"][0]
    assert day["reports"] == 3 and day["cleaned"] == 1
    assert day["max_score"] == b["score"]

    schuylkill = client.get("/api/trends", params={"days": 7, "waterway": "Schuylkill River"}).json()["days"][0]
    assert schuylkill["reports"] == 2
    assert schuylkill["avg_score"] == round((a["score"] + b["score"]) / 2, 1)
    assert schuylkill["cleaned"] == 1
    assert client.get("/api/trends", params={"waterway": "Cobbs Creek"}).json()["days"] == []


def test_river_endpoints_without_tiger(client, database):
    if database:
        pytest.skip("SQLite behavior")
    assert client.get("/api/river/latest").json() == {"available": False, "sites": [], "source": "USGS provisional data"}
    assert client.get("/api/river/daily").json()["series"] == []
    assert client.get("/api/river/daily?parameter=nope").status_code == 422
    assert client.get("/api/db").json()["backend"] == "sqlite"


def _usgs_payload(start: datetime, hours: int):
    """Minimal USGS instantaneous-values JSON: 15-minute discharge readings plus one missing value."""
    values = [{"value": str(1000 + i), "qualifiers": ["P"], "dateTime": (start + timedelta(minutes=15 * i)).isoformat()}
              for i in range(hours * 4)]
    values.append({"value": "-999999", "qualifiers": ["P"], "dateTime": start.isoformat()})
    return {"value": {"timeSeries": [
        {
            "sourceInfo": {"siteName": "SCHUYLKILL RIVER AT PHILADELPHIA, PA",
                           "siteCode": [{"value": "01474500"}],
                           "geoLocation": {"geogLocation": {"latitude": 39.9676, "longitude": -75.1885}}},
            "variable": {"variableCode": [{"value": "00060"}], "noDataValue": -999999.0},
            "values": [{"value": values}],
        },
        {   # a parameter we don't track is ignored
            "sourceInfo": {"siteCode": [{"value": "01474500"}]},
            "variable": {"variableCode": [{"value": "99999"}]},
            "values": [{"value": [{"value": "1", "dateTime": start.isoformat()}]}],
        },
    ]}}


def test_parse_usgs():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    sites, readings = river.parse(_usgs_payload(start, 2))
    assert sites == [{"site_id": "01474500", "name": "Schuylkill River at Philadelphia",
                      "waterway": "Schuylkill River", "latitude": 39.9676, "longitude": -75.1885}]
    assert len(readings) == 8  # missing value dropped, unknown parameter ignored
    assert readings[0] == (start, "01474500", "discharge_cfs", 1000.0)
    assert river.parse({}) == ([], [])


class FakeUSGS:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, params):
        import httpx
        return httpx.Response(200, json=self.payload, request=httpx.Request("GET", url))


@requires_tiger
def test_river_sync_and_aggregates(client):
    from app.tigerstore import refresh_aggregate
    store = client.app.state.store
    start = (datetime.now(timezone.utc) - timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
    fake = FakeUSGS(_usgs_payload(start, 48))
    first = river.sync(store, days=3, client=fake)
    assert first == {"sites": 1, "readings_fetched": 192, "readings_added": 192}
    assert river.sync(store, days=3, client=fake)["readings_added"] == 0  # repeat sync adds nothing

    latest = client.get("/api/river/latest").json()
    assert latest["available"] is True
    assert latest["sites"][0]["readings"]["discharge_cfs"]["value"] == 1191.0

    # Materialize the hourly aggregate, then the daily one built on it; results match the raw readings.
    with store.pool.connection() as conn:
        refresh_aggregate(conn, "river_hourly")
        refresh_aggregate(conn, "river_daily")
        raw = conn.execute(
            "SELECT time_bucket('1 day', time)::date AS day, avg(value) AS mean, min(value) AS min, max(value) AS max "
            "FROM river_readings GROUP BY 1 ORDER BY 1").fetchall()
    series = client.get("/api/river/daily", params={"days": 5, "waterway": "Schuylkill River"}).json()["series"]
    assert [(s["day"], s["mean"], s["min"], s["max"]) for s in series] == \
           [(r["day"].isoformat(), float(r["mean"]), r["min"], r["max"]) for r in raw]
    assert client.get("/api/river/daily", params={"waterway": "Delaware River"}).json()["series"] == []


@requires_tiger
def test_compression_and_db_info(client):
    store = client.app.state.store
    old = datetime.now(timezone.utc) - timedelta(days=60)
    store.upsert_river_sites([{"site_id": "01474500", "name": "Schuylkill", "waterway": "Schuylkill River",
                               "latitude": None, "longitude": None}])
    store.insert_river_readings([(old + timedelta(minutes=15 * i), "01474500", "discharge_cfs", 900.0 + i % 50)
                                 for i in range(4 * 24 * 14)])
    with store.pool.connection() as conn:
        conn.execute("SELECT compress_chunk(c, if_not_compressed => true) "
                     "FROM show_chunks('river_readings', older_than => INTERVAL '7 days') c")

    _upload(client, 3, "Schuylkill River")
    info = client.get("/api/db").json()
    assert info["backend"] == "tigerdata"
    tables = {h["name"]: h for h in info["hypertables"]}
    assert set(tables) >= {"reports", "status_history", "river_readings"}
    readings = tables["river_readings"]
    assert readings["rows"] == 4 * 24 * 14 and readings["compressed_chunks"] >= 1
    assert readings["bytes_after_compression"] < readings["bytes_before_compression"]
    assert set(info["continuous_aggregates"]) == {"litter_daily", "cleanups_daily", "river_hourly", "river_daily"}
    assert info["trend_query_ms"]["continuous_aggregate"] >= 0

    # Compressed history is still readable through the aggregates.
    assert client.get("/api/river/daily", params={"days": 90}).json()["series"]


@requires_tiger
def test_seed_and_clear(client):
    from app import seed
    store = client.app.state.store
    _upload(client, 3, "Schuylkill River")  # a real report, which --clear must keep
    result = seed.seed(store, 60, days=30, rng_seed=1)
    assert result["reports_added"] == 60

    days = client.get("/api/trends?days=31").json()["days"]
    assert sum(d["reports"] for d in days) == 61
    listed = client.get("/api/reports?limit=500").json()
    assert listed["total"] == 61
    demo = next(r for r in listed["items"] if r["model_name"] == "demo-seed")
    assert demo["model_is_stand_in"] is True and "Synthetic" in demo["notes"]
    assert client.get(f"/api/reports/{demo['id']}/image").status_code == 200

    assert seed.clear(store) == 60
    assert client.get("/api/reports").json()["total"] == 1
    assert sum(d["reports"] for d in client.get("/api/trends?days=31").json()["days"]) == 1
