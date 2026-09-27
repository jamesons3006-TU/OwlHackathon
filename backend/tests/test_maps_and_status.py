import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.detector import Detection
from app.main import create_app


class FixedDetector:
    """Returns `n` boxes, where n is set per upload through the image's red channel."""
    name = "fixed"
    is_stand_in = False

    def detect(self, image):
        n = image.getpixel((0, 0))[0] // 10
        return [Detection("garbage", 0.9, x=i * 60, y=10, w=50, h=50) for i in range(n)]


def _jpeg(items: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (items * 10 + 5, 90, 140)).save(buf, "JPEG", quality=100)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(detector=FixedDetector(), data_dir=tmp_path)) as c:
        yield c


def _upload(client, items, lat=None, lon=None, waterway=None):
    data = {k: str(v) for k, v in {"latitude": lat, "longitude": lon, "waterway": waterway}.items() if v is not None}
    r = client.post("/api/reports", files={"image": ("p.jpg", _jpeg(items), "image/jpeg")}, data=data)
    assert r.status_code == 201, r.text
    return r.json()


def test_new_report_defaults(client):
    r = _upload(client, 2)
    assert r["status"] == "reported"
    assert r["location_confirmed"] is False
    assert [h["status"] for h in r["status_history"]] == ["reported"]
    assert "not chemical pollution or water safety" in r["score_note"]


def test_confirm_location_on_map(client):
    r = _upload(client, 2, lat=39.0, lon=-75.0)  # outside Philadelphia
    assert r["in_philadelphia"] is False
    moved = client.patch(f"/api/reports/{r['id']}/location", json={"latitude": 39.9656, "longitude": -75.181}).json()
    assert (moved["latitude"], moved["longitude"]) == (39.9656, -75.181)
    assert moved["location_source"] == "map" and moved["location_confirmed"] is True
    assert moved["in_philadelphia"] is True
    assert client.patch(f"/api/reports/{r['id']}/location", json={"latitude": 200, "longitude": 0}).status_code == 422


def test_status_workflow_and_history(client):
    r = _upload(client, 3)
    client.patch(f"/api/reports/{r['id']}/status", json={"status": "scheduled", "changed_by": "Schuylkill cleanup crew",
                                                         "note": "Saturday 9am"})
    done = client.patch(f"/api/reports/{r['id']}/status", json={"status": "cleaned"}).json()
    assert done["status"] == "cleaned"
    assert [h["status"] for h in done["status_history"]] == ["reported", "scheduled", "cleaned"]
    assert done["status_history"][1]["changed_by"] == "Schuylkill cleanup crew"
    assert client.patch(f"/api/reports/{r['id']}/status", json={"status": "bogus"}).status_code == 422
    assert client.patch("/api/reports/nope/status", json={"status": "cleaned"}).status_code == 404


def test_dashboard_filters_and_priority_sort(client):
    low = _upload(client, 1)
    high = _upload(client, 12)
    mid = _upload(client, 4)
    client.patch(f"/api/reports/{mid['id']}/status", json={"status": "cleaned"})

    by_score = client.get("/api/reports", params={"sort": "score"}).json()["items"]
    assert [i["id"] for i in by_score] == [high["id"], mid["id"], low["id"]]

    open_ids = {i["id"] for i in client.get("/api/reports", params={"status": "open"}).json()["items"]}
    assert open_ids == {low["id"], high["id"]}

    sev = client.get("/api/reports", params=[("severity", low["severity"]), ("severity", high["severity"])]).json()
    assert {i["id"] for i in sev["items"]} >= {low["id"], high["id"]}

    stats = client.get("/api/stats").json()
    assert stats["open_reports"] == 2 and stats["by_status"]["cleaned"] == 1


def test_heat_map_points_geojson(client):
    a = _upload(client, 5, lat=39.9656, lon=-75.1810)
    _upload(client, 5)  # no location: left off the map
    fc = client.get("/api/map/points").json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 1
    f = fc["features"][0]
    assert f["geometry"]["coordinates"] == [-75.181, 39.9656]  # GeoJSON is [lon, lat]
    assert f["properties"]["id"] == a["id"] and f["properties"]["weight"] == a["score"] / 100


def test_bubbles_group_nearby_reports(client):
    # Two reports ~100 m apart near Boathouse Row, one on the Delaware ~3 km away
    near1 = _upload(client, 2, lat=39.9690, lon=-75.1860, waterway="Schuylkill River")
    near2 = _upload(client, 10, lat=39.9696, lon=-75.1852, waterway="Schuylkill River")
    far = _upload(client, 6, lat=39.9500, lon=-75.1400, waterway="Delaware River")

    body = client.get("/api/map/bubbles", params={"cell_m": 500}).json()
    assert body["count"] == 2 and body["max_report_count"] == 2
    big, small = body["bubbles"]
    assert big["report_count"] == 2 and small["report_count"] == 1
    assert big["avg_score"] == round((near1["score"] + near2["score"]) / 2, 1)
    assert big["max_score"] == near2["score"]
    assert big["waterways"] == ["Schuylkill River"]
    assert 39.969 <= big["latitude"] <= 39.9696
    assert small["avg_score"] == far["score"]

    # viewport filter keeps only the Delaware report
    body = client.get("/api/map/bubbles", params={"bbox": "39.94,-75.15,39.96,-75.13"}).json()
    assert body["count"] == 1 and body["bubbles"][0]["waterways"] == ["Delaware River"]
    assert client.get("/api/map/bubbles", params={"bbox": "1,2,3"}).status_code == 422


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    from app.storage import MIGRATIONS, SCHEMA, Store

    # v0.1 schema = today's schema without the migrated columns
    v1 = "\n".join(line for line in SCHEMA.splitlines() if line.strip().split(" ")[0] not in MIGRATIONS)
    v1 = v1.replace("recommendation_json TEXT NOT NULL,", "recommendation_json TEXT NOT NULL")
    db = sqlite3.connect(tmp_path / "reports.db")
    db.executescript(v1)
    assert "status" not in {r[1] for r in db.execute("PRAGMA table_info(reports)")}
    db.execute("INSERT INTO reports (id, created_at, image_path, annotated_path, image_width, image_height, model_name, "
               "model_is_stand_in, score, severity, item_count, coverage_pct, authority_level, recommendation_json) "
               "VALUES ('old', '2026-09-01T00:00:00+00:00', 'a', 'b', 1, 1, 'm', 0, 10, 'low', 1, 0.1, 1, '{}')")
    db.commit(); db.close()

    store = Store(tmp_path)
    row = store.db.execute("SELECT status, location_confirmed FROM reports WHERE id = 'old'").fetchone()
    assert tuple(row) == ("reported", 0)


def test_location_name_on_upload_and_confirm(client):
    r = client.post("/api/reports", files={"image": ("p.jpg", _jpeg(2), "image/jpeg")},
                    data={"location_name": "Schuylkill Banks"}).json()
    assert r["location_name"] == "Schuylkill Banks"
    # Confirming without a name keeps the old one; a new name replaces it.
    kept = client.patch(f"/api/reports/{r['id']}/location", json={"latitude": 39.95, "longitude": -75.18}).json()
    assert kept["location_name"] == "Schuylkill Banks"
    renamed = client.patch(f"/api/reports/{r['id']}/location",
                           json={"latitude": 39.95, "longitude": -75.18, "location_name": "Bartram's Garden"}).json()
    assert renamed["location_name"] == "Bartram's Garden"
    assert "location_name" in client.get("/api/export?format=csv").text.splitlines()[0]


def test_serves_front_end(client, tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "FRONTEND_DIR", tmp_path / "missing")
    assert client.get("/").status_code == 404
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "index.html").write_text("<!doctype html><title>t</title>")
    monkeypatch.setattr(config, "FRONTEND_DIR", tmp_path / "site")
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
