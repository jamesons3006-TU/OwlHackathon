"""Philly Water Watch API: upload waterway photos, score visible litter, map hotspots, track cleanups."""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config
from .detector import Detector, ModelUnavailable, load_detector
from .recommendations import AUTHORITIES, recommend
from .scoring import SCORE_NOTE, SEVERITY_BANDS, score_detections
from .river import PARAMETERS, Poller
from .storage import OPEN_STATUSES, STATUSES, Store, now_iso, open_store

WATERWAYS = [
    "Schuylkill River",
    "Delaware River",
    "Wissahickon Creek",
    "Pennypack Creek",
    "Cobbs Creek",
    "Tacony-Frankford Creek",
    "Poquessing Creek",
    "Darby Creek",
    "Other",
]

Severity = Literal["none", "low", "moderate", "high", "severe"]
Status = Literal["reported", "verified", "scheduled", "in_progress", "cleaned", "dismissed"]


@dataclass
class ReportFilters:
    """Query filters shared by the report list, map layers and exports."""
    waterway: str | None = Query(None)
    severity: list[Severity] | None = Query(None, description="Repeat to match any: ?severity=high&severity=severe")
    status: list[Status | Literal["open"]] | None = Query(
        None, description="Repeat to match any. `open` = reported, verified, scheduled or in_progress")
    min_score: int | None = Query(None, ge=0, le=100)
    since: str | None = Query(None, description="ISO date/time, e.g. 2026-09-01")
    until: str | None = Query(None, description="ISO date/time")
    bbox: str | None = Query(None, description="Map viewport: min_lat,min_lon,max_lat,max_lon")

    def as_kwargs(self) -> dict:
        bbox = None
        if self.bbox:
            try:
                bbox = tuple(float(v) for v in self.bbox.split(","))
                assert len(bbox) == 4
            except (ValueError, AssertionError):
                raise HTTPException(422, "bbox must be min_lat,min_lon,max_lat,max_lon")
        return {"waterway": self.waterway, "severity": self.severity, "status": self.status,
                "min_score": self.min_score, "since": self.since, "until": self.until, "bbox": bbox}


class LocationUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location_name: str | None = Field(None, max_length=120, description="Place name, e.g. Schuylkill Banks")


class StatusUpdate(BaseModel):
    status: Status
    note: str | None = Field(None, max_length=1000)
    changed_by: str | None = Field(None, max_length=200, description="Person or organization making the change")


def create_app(detector: Detector | None = None, data_dir: Path | None = None,
               database_url: str | None = None, river_sync: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = open_store(data_dir or config.DATA_DIR, config.DATABASE_URL if database_url is None else database_url)
        app.state.store = store
        app.state.detector = detector or load_detector()
        app.state.river = None
        if river_sync and store.backend == "tigerdata" and config.RIVER_SYNC_MINUTES > 0:
            app.state.river = Poller(store, config.RIVER_SYNC_MINUTES)
            app.state.river.start()
        yield
        if app.state.river:
            app.state.river.stop()
        store.close()

    app = FastAPI(
        title="Philly Water Watch API",
        description="Scores visible litter in photos of Philadelphia waterways, maps hotspots, "
                    "recommends who to contact and tracks cleanup status. " + SCORE_NOTE,
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    # ---- info ---------------------------------------------------------

    @app.get("/api/health")
    def health(request: Request):
        det = request.app.state.detector
        model = det.health() if hasattr(det, "health") else {}
        status = "ok" if model.get("reachable", True) else "degraded"
        return {"status": status, "model": det.name, "model_is_stand_in": det.is_stand_in, "model_status": model,
                "database": request.app.state.store.backend}

    @app.get("/api/meta")
    def meta():
        """Reference data for the front end: waterways, severity bands, statuses, authority tiers."""
        return {
            "score_note": SCORE_NOTE,
            "waterways": WATERWAYS,
            "severity_bands": [{"severity": s, "min_score": m} for m, s in sorted(SEVERITY_BANDS)],
            "statuses": STATUSES,
            "open_statuses": OPEN_STATUSES,
            "authorities": [dict(key=k, **v) for k, v in AUTHORITIES.items()],
            "philadelphia_bounds": config.PHILLY_BOUNDS,
        }

    # ---- reports ------------------------------------------------------

    @app.post("/api/reports", status_code=201)
    async def create_report(
        request: Request,
        image: UploadFile = File(..., description="Photo of the waterway (JPEG/PNG/WebP)"),
        waterway: str | None = Form(None),
        location_name: str | None = Form(None, max_length=120, description="Place name, e.g. Schuylkill Banks"),
        latitude: float | None = Form(None, ge=-90, le=90),
        longitude: float | None = Form(None, ge=-180, le=180),
        notes: str | None = Form(None, max_length=2000),
        reporter: str | None = Form(None, max_length=200),
        hazard_suspected: bool = Form(False, description="Reporter sees oil sheen, chemical smell, sewage, dead fish, etc. "
                                                         "Does not change the litter score; escalates who to contact."),
    ):
        data = await image.read(config.MAX_UPLOAD_BYTES + 1)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Image larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            img = Image.open(io.BytesIO(data))
            exif = _read_exif(img)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(415, "File is not a readable image")

        location_source = "reporter" if latitude is not None and longitude is not None else None
        if location_source is None and exif.get("gps"):
            latitude, longitude = exif["gps"]
            location_source = "exif"
        if location_source is None:
            latitude = longitude = None

        store: Store = request.app.state.store
        det: Detector = request.app.state.detector
        try:
            detections = await run_in_threadpool(det.detect, img)
        except ModelUnavailable as e:
            raise HTTPException(503, f"Detection model unavailable: {e}")
        score = score_detections(detections, img.width, img.height)
        rec = recommend(score.severity, hazard_suspected)

        report_id = uuid.uuid4().hex
        image_path, annotated_path = store.save_images(report_id, img, detections)
        report = {
            "id": report_id,
            "created_at": now_iso(),
            "captured_at": exif.get("captured_at"),
            "waterway": waterway,
            "location_name": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "location_source": location_source,
            "in_philadelphia": _in_philly(latitude, longitude),
            "notes": notes,
            "reporter": reporter,
            "hazard_suspected": hazard_suspected,
            "image_path": image_path,
            "annotated_path": annotated_path,
            "image_width": img.width,
            "image_height": img.height,
            "model_name": det.name,
            "model_is_stand_in": det.is_stand_in,
            "score": score.score,
            "severity": score.severity,
            "item_count": score.item_count,
            "coverage_pct": round(score.coverage * 100, 2),
            "authority_level": rec["authority_level"],
            "recommendation": rec,
        }
        store.insert(report, detections)
        return _public(store.get(report_id), request)

    @app.get("/api/reports")
    def list_reports(
        request: Request,
        filters: ReportFilters = Depends(),
        sort: Literal["newest", "oldest", "score"] = Query("newest", description="`score` = highest severity first"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        items, total = request.app.state.store.list(limit=limit, offset=offset, sort=sort, **filters.as_kwargs())
        return {"total": total, "limit": limit, "offset": offset, "items": [_public(r, request) for r in items]}

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: str, request: Request):
        return _public(_require(request, report_id), request)

    @app.patch("/api/reports/{report_id}/location")
    def confirm_location(report_id: str, body: LocationUpdate, request: Request):
        """Set the location the user confirmed (or dragged the pin to) on the map."""
        _require(request, report_id)
        store: Store = request.app.state.store
        store.set_location(report_id, body.latitude, body.longitude, _in_philly(body.latitude, body.longitude),
                           body.location_name.strip() if body.location_name and body.location_name.strip() else None)
        return _public(store.get(report_id), request)

    @app.patch("/api/reports/{report_id}/status")
    def update_status(report_id: str, body: StatusUpdate, request: Request):
        """Move a report through the cleanup workflow. Every change is kept in status_history."""
        _require(request, report_id)
        store: Store = request.app.state.store
        store.set_status(report_id, body.status, body.changed_by, body.note)
        return _public(store.get(report_id), request)

    @app.get("/api/reports/{report_id}/image")
    def get_image(report_id: str, request: Request, annotated: bool = False):
        report = _require(request, report_id)
        path = request.app.state.store.abs_path(report["annotated_path" if annotated else "image_path"])
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/stats")
    def stats(request: Request):
        return request.app.state.store.stats()

    # ---- maps ---------------------------------------------------------

    @app.get("/api/map/points")
    def map_points(request: Request, filters: ReportFilters = Depends()):
        """Heat-map layer: GeoJSON points weighted by score (0-1). Only reports with a location."""
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["longitude"], p["latitude"]]},
                "properties": {
                    "id": p["id"], "score": p["score"], "weight": round(p["score"] / 100, 3),
                    "severity": p["severity"], "status": p["status"], "waterway": p["waterway"],
                    "item_count": p["item_count"], "created_at": p["created_at"],
                    "location_confirmed": bool(p["location_confirmed"]),
                },
            }
            for p in request.app.state.store.points(**filters.as_kwargs())
        ]
        return {"type": "FeatureCollection", "features": features}

    @app.get("/api/map/bubbles")
    def map_bubbles(
        request: Request,
        filters: ReportFilters = Depends(),
        cell_m: int = Query(500, ge=50, le=10000, description="Grid cell size in meters"),
    ):
        """Bubble-map layer: reports grouped into grid cells.

        Color by `avg_score` / `avg_severity`, size by `report_count`.
        """
        cell_deg = cell_m / 111_000  # meters to degrees of latitude; fine at Philadelphia's scale
        bubbles = request.app.state.store.bubbles(cell_deg, **filters.as_kwargs())
        return {"cell_m": cell_m, "count": len(bubbles),
                "max_report_count": max((b["report_count"] for b in bubbles), default=0), "bubbles": bubbles}

    # ---- time series (continuous aggregates on Tiger Data) -------------

    @app.get("/api/trends")
    def trends(request: Request, days: int = Query(90, ge=1, le=3650), waterway: str | None = None):
        """Daily reports, average score, priority reports and cleanups. Days with no activity are omitted."""
        start = time.perf_counter()
        rows = request.app.state.store.trends(days=days, waterway=waterway)
        return {"database": request.app.state.store.backend, "days": rows,
                "query_ms": round((time.perf_counter() - start) * 1000, 2)}

    @app.get("/api/river/latest")
    def river_latest(request: Request):
        """Latest USGS reading of each parameter at each gauge (needs Tiger Data)."""
        store = request.app.state.store
        sites: dict[str, dict] = {}
        for r in store.river_latest():
            site = sites.setdefault(r["site_id"], {"site_id": r["site_id"], "name": r["name"],
                                                   "waterway": r["waterway"], "readings": {}})
            site["readings"][r["parameter"]] = {"value": r["value"], "time": r["time"]}
        return {"available": store.backend == "tigerdata", "sites": list(sites.values()),
                "source": "USGS provisional data"}

    @app.get("/api/river/daily")
    def river_daily(
        request: Request,
        parameter: str = Query("discharge_cfs", description=", ".join(PARAMETERS.values())),
        days: int = Query(90, ge=1, le=3650),
        waterway: str | None = None,
    ):
        """Daily mean/min/max per gauge, from the hierarchical river_daily continuous aggregate."""
        if parameter not in PARAMETERS.values():
            raise HTTPException(422, f"parameter must be one of: {', '.join(PARAMETERS.values())}")
        store = request.app.state.store
        return {"available": store.backend == "tigerdata", "parameter": parameter,
                "series": store.river_series(days, parameter, waterway)}

    @app.get("/api/db")
    def db_info(request: Request):
        """Which database is in use and, on Tiger Data, hypertables, compression and aggregate timings."""
        info = request.app.state.store.db_info()
        poller = request.app.state.river
        if poller:
            info["river_sync"] = {"every_minutes": poller.minutes, "last_result": poller.last_result,
                                  "last_error": poller.last_error}
        return info

    # ---- front end ----------------------------------------------------

    @app.get("/", include_in_schema=False)
    def frontend():
        index = config.FRONTEND_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(404, "Front end not found. The API docs are at /docs")
        return FileResponse(index, media_type="text/html")

    # ---- researcher export --------------------------------------------

    @app.get("/api/export")
    def export(
        request: Request,
        filters: ReportFilters = Depends(),
        format: Literal["csv", "json", "coco", "zip"] = "csv",
    ):
        """Download reports for research. `zip` bundles images, annotated images, CSV, JSON and COCO labels."""
        store: Store = request.app.state.store
        reports, _ = store.list(limit=1_000_000, **filters.as_kwargs())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if format == "csv":
            return Response(_to_csv(reports), media_type="text/csv",
                            headers=_attachment(f"philly-water-reports-{stamp}.csv"))
        if format == "json":
            return Response(json.dumps(reports, indent=2), media_type="application/json",
                            headers=_attachment(f"philly-water-reports-{stamp}.json"))
        if format == "coco":
            return Response(json.dumps(_to_coco(reports), indent=2), media_type="application/json",
                            headers=_attachment(f"philly-water-coco-{stamp}.json"))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", SCORE_NOTE + "\nDetections are model predictions, not human-verified labels.\n")
            zf.writestr("reports.csv", _to_csv(reports))
            zf.writestr("reports.json", json.dumps(reports, indent=2))
            zf.writestr("annotations_coco.json", json.dumps(_to_coco(reports), indent=2))
            for r in reports:
                zf.write(store.abs_path(r["image_path"]), f"images/{r['id']}.jpg")
                zf.write(store.abs_path(r["annotated_path"]), f"annotated/{r['id']}.jpg")
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers=_attachment(f"philly-water-dataset-{stamp}.zip"))

    return app


# ---- helpers ----------------------------------------------------------

def _require(request: Request, report_id: str) -> dict:
    report = request.app.state.store.get(report_id)
    if report is None:
        raise HTTPException(404, "Report not found")
    return report


def _public(report: dict, request: Request) -> dict:
    """Swap internal file paths for API URLs."""
    out = {k: v for k, v in report.items() if k not in ("image_path", "annotated_path")}
    base = str(request.url_for("get_image", report_id=report["id"]))
    out["image_url"] = base
    out["annotated_image_url"] = base + "?annotated=true"
    out["score_note"] = SCORE_NOTE
    return out


def _attachment(filename: str) -> dict:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


def _in_philly(lat: float | None, lon: float | None) -> bool | None:
    if lat is None or lon is None:
        return None
    b = config.PHILLY_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


def _read_exif(img: Image.Image) -> dict:
    """Pull capture time and GPS coordinates from photo metadata when present."""
    out: dict = {}
    try:
        exif = img.getexif()
    except Exception:
        return out
    taken = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    if taken:
        try:
            out["captured_at"] = datetime.strptime(str(taken), "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            pass
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    try:
        lat = _dms(gps[ExifTags.GPS.GPSLatitude]) * (-1 if gps.get(ExifTags.GPS.GPSLatitudeRef) == "S" else 1)
        lon = _dms(gps[ExifTags.GPS.GPSLongitude]) * (-1 if gps.get(ExifTags.GPS.GPSLongitudeRef) == "W" else 1)
        out["gps"] = (round(lat, 6), round(lon, 6))
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        pass
    return out


def _dms(value) -> float:
    d, m, s = (float(v) for v in value)
    return d + m / 60 + s / 3600


CSV_FIELDS = [
    "id", "created_at", "captured_at", "waterway", "location_name", "latitude", "longitude", "location_source", "location_confirmed",
    "in_philadelphia", "score", "severity", "item_count", "coverage_pct", "status", "status_updated_at",
    "authority_level", "primary_authority", "hazard_suspected", "model_name", "model_is_stand_in",
    "image_width", "image_height", "reporter", "notes",
]


def _to_csv(reports: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for r in reports:
        primary = r["recommendation"].get("primary_authority")
        writer.writerow({**r, "primary_authority": primary["name"] if primary else ""})
    return buf.getvalue()


def _to_coco(reports: list[dict]) -> dict:
    labels = sorted({d["label"] for r in reports for d in r["detections"]})
    cat_ids = {label: i + 1 for i, label in enumerate(labels)}
    images, annotations = [], []
    for img_id, r in enumerate(reports, start=1):
        images.append({
            "id": img_id, "file_name": f"images/{r['id']}.jpg", "width": r["image_width"], "height": r["image_height"],
            "report_id": r["id"], "waterway": r["waterway"], "latitude": r["latitude"], "longitude": r["longitude"],
            "date_captured": r["captured_at"] or r["created_at"], "score": r["score"], "severity": r["severity"],
        })
        for d in r["detections"]:
            annotations.append({
                "id": len(annotations) + 1, "image_id": img_id, "category_id": cat_ids[d["label"]],
                "bbox": [round(d["x"], 2), round(d["y"], 2), round(d["w"], 2), round(d["h"], 2)],
                "area": round(d["w"] * d["h"], 2), "iscrowd": 0, "score": round(d["confidence"], 4),
            })
    return {
        "info": {"description": "Philly Water Watch citizen reports (model predictions, not human-verified). " + SCORE_NOTE,
                 "date_created": now_iso()},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in cat_ids.items()],
    }


app = create_app()
