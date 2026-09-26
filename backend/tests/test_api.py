import csv
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.detector import Detection
from app.main import create_app
from app.recommendations import recommend
from app.scoring import score_detections, severity_for


class StubDetector:
    name = "stub.pt"
    is_stand_in = False

    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections


def _box(n, size=40):
    return [Detection("garbage", 0.9, x=i * 50, y=10, w=size, h=size) for i in range(n)]


def _jpeg(w=640, h=480):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (30, 90, 140)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client_factory(tmp_path):
    def make(detections):
        return TestClient(create_app(detector=StubDetector(detections), data_dir=tmp_path))
    return make


# ---- scoring ------------------------------------------------------------

def test_no_detections_scores_zero():
    s = score_detections([], 640, 480)
    assert (s.score, s.severity) == (0, "none")


def test_score_increases_with_litter():
    few = score_detections(_box(1), 640, 480).score
    many = score_detections(_box(10), 640, 480).score
    assert 0 < few < many <= 100


def test_overlapping_boxes_not_double_counted():
    same = [Detection("garbage", 0.9, 0, 0, 64, 48)] * 5
    assert score_detections(same, 640, 480).coverage == pytest.approx(0.01, abs=0.002)


@pytest.mark.parametrize("score,severity", [(0, "none"), (1, "low"), (24, "low"), (25, "moderate"),
                                            (50, "high"), (75, "severe"), (100, "severe")])
def test_severity_bands(score, severity):
    assert severity_for(score) == severity


# ---- recommendations ------------------------------------------------------

def test_authority_level_rises_with_severity():
    levels = [recommend(s)["authority_level"] for s in ("none", "low", "moderate", "high", "severe")]
    assert levels == sorted(levels) and levels[0] == 0 and levels[-1] == 4


def test_hazard_escalates_to_federal():
    rec = recommend("low", hazard_suspected=True)
    assert rec["authority_level"] == 5
    assert rec["primary_authority"]["key"] == "federal"


# ---- API ------------------------------------------------------------------

def test_upload_scores_saves_and_exports(client_factory):
    with client_factory(_box(6, size=80)) as client:
        r = client.post(
            "/api/reports",
            files={"image": ("river.jpg", _jpeg(), "image/jpeg")},
            data={"waterway": "Schuylkill River", "latitude": "39.9656", "longitude": "-75.1810", "notes": "near boathouse"},
        )
        assert r.status_code == 201, r.text
        report = r.json()
        assert report["item_count"] == 6
        assert report["severity"] in ("moderate", "high", "severe")
        assert report["in_philadelphia"] is True
        assert report["recommendation"]["authorities"]
        assert len(report["detections"]) == 6

        assert client.get(report["image_url"]).headers["content-type"] == "image/jpeg"
        assert client.get(report["annotated_image_url"]).status_code == 200
        assert client.get(f"/api/reports/{report['id']}").json()["id"] == report["id"]

        listing = client.get("/api/reports", params={"waterway": "Schuylkill River"}).json()
        assert listing["total"] == 1

        rows = list(csv.DictReader(io.StringIO(client.get("/api/export?format=csv").text)))
        assert rows[0]["id"] == report["id"] and rows[0]["waterway"] == "Schuylkill River"

        coco = client.get("/api/export?format=coco").json()
        assert len(coco["images"]) == 1 and len(coco["annotations"]) == 6

        z = zipfile.ZipFile(io.BytesIO(client.get("/api/export?format=zip").content))
        assert {f"images/{report['id']}.jpg", f"annotated/{report['id']}.jpg", "reports.csv"} <= set(z.namelist())
        assert json.loads(z.read("reports.json"))[0]["id"] == report["id"]

        stats = client.get("/api/stats").json()
        assert stats["reports"] == 1 and stats["items_detected"] == 6


def test_clean_photo_needs_no_report(client_factory):
    with client_factory([]) as client:
        report = client.post("/api/reports", files={"image": ("clean.jpg", _jpeg(), "image/jpeg")}).json()
        assert report["score"] == 0 and report["severity"] == "none"
        assert report["recommendation"]["authorities"] == []
        assert report["latitude"] is None and report["in_philadelphia"] is None


def test_rejects_non_image(client_factory):
    with client_factory([]) as client:
        r = client.post("/api/reports", files={"image": ("x.txt", b"not an image", "text/plain")})
        assert r.status_code == 415


def test_model_down_returns_503_and_saves_nothing(tmp_path):
    from app.detector import ModelUnavailable

    class DownDetector(StubDetector):
        def detect(self, image):
            raise ModelUnavailable("connection refused")

    with TestClient(create_app(detector=DownDetector([]), data_dir=tmp_path)) as client:
        r = client.post("/api/reports", files={"image": ("x.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 503 and "connection refused" in r.json()["detail"]
        assert client.get("/api/reports").json()["total"] == 0


def test_missing_report_404(client_factory):
    with client_factory([]) as client:
        assert client.get("/api/reports/nope").status_code == 404
