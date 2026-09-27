"""FastAPI application for RiverWatch / Philly Water Watch."""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config
from .detector import Detector, load_detector
from .recommendations import AUTHORITIES, recommend
from .river import PARAMETERS, Poller
from .scoring import SCORE_NOTE, SEVERITY_BANDS, score_detections
from .storage import OPEN_STATUSES, STATUSES, now_iso, open_store


WATERWAYS = [
    "Delaware River",
    "Schuylkill River",
    "Wissahickon Creek",
    "Pennypack Creek",
    "Cobbs Creek",
    "Tacony-Frankford Creek",
    "Poquessing Creek",
    "Other",
]

Severity = Literal["low", "moderate", "high", "severe"]
Status = Literal[
    "reported",
    "verified",
    "scheduled",
    "in_progress",
    "cleaned",
    "dismissed",
]


class Filters(BaseModel):
    waterway: str | None = None
    severity: list[str] | None = None
    status: list[str] | None = None
    min_score: int | None = Field(None, ge=0, le=100)
    since: str | None = None
    until: str | None = None
    bbox: str | None = None

    def as_kwargs(self) -> dict:
        bbox = None

        if self.bbox:
            try:
                bbox = tuple(
                    float(value)
                    for value in self.bbox.split(",")
                )
                assert len(bbox) == 4
            except (ValueError, AssertionError):
                raise HTTPException(
                    422,
                    "bbox must be min_lat,min_lon,max_lat,max_lon",
                )

        return {
            "waterway": self.waterway,
            "severity": self.severity,
            "status": self.status,
            "min_score": self.min_score,
            "since": self.since,
            "until": self.until,
            "bbox": bbox,
        }


class LocationUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location_name: str | None = Field(
        None,
        max_length=120,
        description="Place name, e.g. Schuylkill Banks",
    )


class StatusUpdate(BaseModel):
    status: Status
    note: str | None = Field(
        None,
        max_length=1000,
    )
    changed_by: str | None = Field(
        None,
        max_length=200,
        description="Person or organization making the change",
    )


def create_app(
    detector: Detector | None = None,
    data_dir: Path | None = None,
    database_url: str | None = None,
    river_sync: bool = True,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        store = open_store(
            data_dir or config.DATA_DIR,
            config.DATABASE_URL
            if database_url is None
            else database_url,
        )

        app.state.store = store
        app.state.detector = detector or load_detector()
        app.state.river = None

        if (
            river_sync
            and store.backend == "tigerdata"
            and config.RIVER_SYNC_MINUTES > 0
        ):
            app.state.river = Poller(
                store,
                config.RIVER_SYNC_MINUTES,
            )
            app.state.river.start()

        yield

        if app.state.river:
            app.state.river.stop()

        store.close()

    app = FastAPI(
        title="Philly Water Watch API",
        description=(
            "Scores visible litter in photos of Philadelphia waterways, "
            "maps hotspots, recommends who to contact and tracks cleanup "
            "status. "
            + SCORE_NOTE
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health(request: Request):

        det = request.app.state.detector

        model = (
            det.health()
            if hasattr(det, "health")
            else {}
        )

        status = (
            "ok"
            if model.get("reachable", True)
            else "degraded"
        )

        return {
            "status": status,
            "model": det.name,
            "model_is_stand_in": det.is_stand_in,
            "model_status": model,
            "database": request.app.state.store.backend,
        }

    @app.get("/api/meta")
    def meta():
        """Reference information used by the frontend."""

        return {
            "score_note": SCORE_NOTE,
            "waterways": WATERWAYS,
            "severity_bands": [
                {
                    "severity": severity,
                    "min_score": minimum,
                }
                for minimum, severity
                in sorted(SEVERITY_BANDS)
            ],
            "statuses": STATUSES,
            "open_statuses": OPEN_STATUSES,
            "authorities": [
                dict(
                    key=key,
                    **value,
                )
                for key, value
                in AUTHORITIES.items()
            ],
            "philadelphia_bounds": config.PHILLY_BOUNDS,
        }

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @app.post(
        "/api/reports",
        status_code=201,
    )
    async def create_report(
        request: Request,

        image: UploadFile = File(
            ...,
            description="Photo of the waterway (JPEG/PNG/WebP)",
        ),

        waterway: str | None = Form(None),

        location_name: str | None = Form(
            None,
            max_length=120,
        ),

        latitude: float | None = Form(
            None,
            ge=-90,
            le=90,
        ),

        longitude: float | None = Form(
            None,
            ge=-180,
            le=180,
        ),

        notes: str | None = Form(
            None,
            max_length=2000,
        ),

        reporter: str | None = Form(
            None,
            max_length=200,
        ),

        hazard_suspected: bool = Form(
            False,
            description=(
                "Reporter sees oil sheen, chemical smell, sewage, "
                "dead fish, etc. Does not change the litter score; "
                "escalates who to contact."
            ),
        ),
    ):

        data = await image.read(
            config.MAX_UPLOAD_BYTES + 1
        )

        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                (
                    "Image larger than "
                    f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                ),
            )

        try:
            img = Image.open(
                io.BytesIO(data)
            )

            exif = _read_exif(img)

            img = (
                ImageOps.exif_transpose(img)
                .convert("RGB")
            )

        except (
            UnidentifiedImageError,
            OSError,
        ):
            raise HTTPException(
                415,
                "Unsupported or corrupt image",
            )

        if waterway and waterway not in WATERWAYS:
            raise HTTPException(
                422,
                f"Unknown waterway: {waterway}",
            )

        # --------------------------------------------------------------
        # Location
        # --------------------------------------------------------------

        location_source = None
        location_confirmed = False

        if (
            latitude is not None
            and longitude is not None
        ):
            location_source = "form"
            location_confirmed = True

        elif (
            exif.get("latitude") is not None
            and exif.get("longitude") is not None
        ):
            latitude = exif["latitude"]
            longitude = exif["longitude"]
            location_source = "exif"
            location_confirmed = False

        in_philadelphia = None

        if (
            latitude is not None
            and longitude is not None
        ):
            in_philadelphia = _in_philadelphia(
                latitude,
                longitude,
            )

        # --------------------------------------------------------------
        # Model inference
        # --------------------------------------------------------------

        detector_impl = request.app.state.detector

        try:
            detections = detector_impl.detect(img)

        except Exception as exc:
            raise HTTPException(
                503,
                f"Model inference failed: {exc}",
            ) from exc

        # --------------------------------------------------------------
        # Score
        # --------------------------------------------------------------

        score = score_detections(
            detections,
            img.width,
            img.height,
        )

        recommendation = recommend(
            score.severity,
            hazard_suspected,
        )

        report_id = str(
            uuid.uuid4()
        )

        created_at = now_iso()

        captured_at = exif.get(
            "captured_at"
        )

        # --------------------------------------------------------------
        # Save original + annotated images
        # --------------------------------------------------------------

        store = request.app.state.store

        image_path, annotated_path = (
            store.save_images(
                report_id,
                img,
                detections,
            )
        )

        # --------------------------------------------------------------
        # Build report
        # --------------------------------------------------------------

        report = {
            "id": report_id,
            "created_at": created_at,
            "captured_at": captured_at,
            "waterway": waterway,
            "location_name": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "location_source": location_source,
            "location_confirmed": location_confirmed,
            "in_philadelphia": in_philadelphia,
            "notes": notes,
            "reporter": reporter,
            "hazard_suspected": hazard_suspected,
            "image_path": image_path,
            "annotated_path": annotated_path,
            "image_width": img.width,
            "image_height": img.height,
            "model_name": detector_impl.name,
            "model_is_stand_in": detector_impl.is_stand_in,
            "score": score.score,
            "severity": score.severity,
            "item_count": score.item_count,

            # IMPORTANT:
            # Keep the detections in the API report.
            #
            # storage.py deliberately excludes this field from the reports
            # table and writes each detection into the detections table.
            "detections": [
                detection.to_dict()
                for detection in detections
            ],

            "coverage_pct": round(score.coverage * 100, 2),
            "authority_level": recommendation["authority_level"],
            "recommendation": recommendation,
        }

        store.insert(
            report,
            detections,
        )

        # Re-fetch so status/status_history and database-normalized values
        # are included in the response.
        stored = store.get(report_id)

        return _public(
            stored or report
        )

    @app.get("/api/reports")
    def list_reports(
        request: Request,

        waterway: str | None = Query(None),

        severity: list[str] | None = Query(None),

        status: list[str] | None = Query(None),

        min_score: int | None = Query(
            None,
            ge=0,
            le=100,
        ),

        since: str | None = Query(None),

        until: str | None = Query(None),

        bbox: str | None = Query(None),

        limit: int = Query(
            50,
            ge=1,
            le=500,
        ),

        offset: int = Query(
            0,
            ge=0,
        ),

        sort: Literal[
            "newest",
            "oldest",
            "score",
        ] = "newest",
    ):

        filters = Filters(
            waterway=waterway,
            severity=severity,
            status=status,
            min_score=min_score,
            since=since,
            until=until,
            bbox=bbox,
        )

        rows, total = request.app.state.store.list(
            limit=limit,
            offset=offset,
            sort=sort,
            **filters.as_kwargs(),
        )

        return {
            "items": [
                _public(row)
                for row in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.get(
        "/api/reports/{report_id}"
    )
    def get_report(
        report_id: str,
        request: Request,
    ):

        report = request.app.state.store.get(
            report_id
        )

        if report is None:
            raise HTTPException(
                404,
                "Report not found",
            )

        return _public(report)

    @app.patch(
        "/api/reports/{report_id}/location"
    )
    def update_location(
        report_id: str,
        body: LocationUpdate,
        request: Request,
    ):

        store = request.app.state.store

        if store.get(report_id) is None:
            raise HTTPException(
                404,
                "Report not found",
            )

        in_philadelphia = _in_philadelphia(
            body.latitude,
            body.longitude,
        )

        store.set_location(
            report_id,
            body.latitude,
            body.longitude,
            in_philadelphia,
            body.location_name,
        )

        return _public(
            store.get(report_id)
        )

    @app.patch(
        "/api/reports/{report_id}/status"
    )
    def update_status(
        report_id: str,
        body: StatusUpdate,
        request: Request,
    ):

        store = request.app.state.store

        if store.get(report_id) is None:
            raise HTTPException(
                404,
                "Report not found",
            )

        store.set_status(
            report_id,
            body.status,
            body.changed_by,
            body.note,
        )

        return _public(
            store.get(report_id)
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    @app.get(
        "/api/reports/{report_id}/image"
    )
    def report_image(
        report_id: str,
        request: Request,
        annotated: bool = Query(False),
    ):

        store = request.app.state.store

        report = store.get(
            report_id
        )

        if report is None:
            raise HTTPException(
                404,
                "Report not found",
            )

        rel = (
            report["annotated_path"]
            if annotated
            else report["image_path"]
        )

        path = store.abs_path(rel)

        if not path.is_file():
            raise HTTPException(
                404,
                "Image not found",
            )

        return FileResponse(
            path,
            media_type="image/jpeg",
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @app.get("/api/stats")
    def stats(request: Request):
        return request.app.state.store.stats()

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------

    @app.get("/api/map/points")
    def map_points(
        request: Request,

        waterway: str | None = Query(None),

        severity: list[str] | None = Query(None),

        status: list[str] | None = Query(None),

        min_score: int | None = Query(
            None,
            ge=0,
            le=100,
        ),

        since: str | None = Query(None),

        until: str | None = Query(None),

        bbox: str | None = Query(None),
    ):

        filters = Filters(
            waterway=waterway,
            severity=severity,
            status=status,
            min_score=min_score,
            since=since,
            until=until,
            bbox=bbox,
        )

        points = request.app.state.store.points(
            **filters.as_kwargs()
        )

        # Frontend expects GeoJSON:
        #
        # {
        #   "features": [
        #       {
        #           "type": "Feature",
        #           "geometry": {
        #               "type": "Point",
        #               "coordinates": [longitude, latitude]
        #           },
        #           "properties": {...}
        #       }
        #   ]
        # }

        features = []

        for point in points:
            features.append(
                {
                    "type": "Feature",

                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            point["longitude"],
                            point["latitude"],
                        ],
                    },

                    "properties": {
                        key: value
                        for key, value in point.items()
                        if key not in (
                            "latitude",
                            "longitude",
                        )
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
        }


    @app.get("/api/map/bubbles")
    def map_bubbles(
        request: Request,

        # Frontend sends the desired grid size in meters.
        cell_m: float = Query(
            500,
            gt=0,
            le=10000,
        ),

        waterway: str | None = Query(None),

        severity: list[str] | None = Query(None),

        status: list[str] | None = Query(None),

        min_score: int | None = Query(
            None,
            ge=0,
            le=100,
        ),

        since: str | None = Query(None),

        until: str | None = Query(None),

        bbox: str | None = Query(None),
    ):

        filters = Filters(
            waterway=waterway,
            severity=severity,
            status=status,
            min_score=min_score,
            since=since,
            until=until,
            bbox=bbox,
        )

        # storage.py currently groups reports using degrees.
        #
        # Around Philadelphia:
        #     1 degree latitude ≈ 111,000 meters
        #
        # This conversion is sufficient for the visualization grid.
        cell_deg = cell_m / 111_000.0

        bubbles = request.app.state.store.bubbles(
            cell_deg,
            **filters.as_kwargs(),
        )

        return {
            "cell_m": cell_m,
            "bubbles": bubbles,
        }

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------

    @app.get("/api/trends")
    def trends(
        request: Request,

        days: int = Query(
            90,
            ge=1,
            le=3650,
        ),

        waterway: str | None = Query(None),
    ):

        return request.app.state.store.trends(
            days=days,
            waterway=waterway,
        )

    # ------------------------------------------------------------------
    # River / USGS data
    # ------------------------------------------------------------------

    @app.get("/api/river/latest")
    def river_latest(
        request: Request,
    ):

        return {
            "parameters": PARAMETERS,
            "readings": (
                request.app.state.store
                .river_latest()
            ),
        }

    @app.get("/api/river/daily")
    def river_daily(
        request: Request,

        days: int = Query(
            30,
            ge=1,
            le=3650,
        ),

        parameter: str = Query(
            "discharge_cfs"
        ),

        waterway: str | None = Query(None),
    ):

        if parameter not in PARAMETERS.values():
            raise HTTPException(
                422,
                (
                    "Unknown river parameter. "
                    f"Choose one of: {', '.join(PARAMETERS.values())}"
                ),
            )

        return {
            "parameter": parameter,
            "days": days,
            "waterway": waterway,
            "readings": (
                request.app.state.store
                .river_series(
                    days,
                    parameter,
                    waterway,
                )
            ),
        }

    # ------------------------------------------------------------------
    # Database diagnostics
    # ------------------------------------------------------------------

    @app.get("/api/db")
    def db_info(
        request: Request,
    ):

        result = (
            request.app.state.store
            .db_info()
        )

        poller = request.app.state.river

        if poller is not None:
            result["river_sync"] = {
                "enabled": True,
                "interval_minutes": (
                    config.RIVER_SYNC_MINUTES
                ),
                "last_result": (
                    poller.last_result
                ),
                "last_error": (
                    poller.last_error
                ),
            }
        else:
            result["river_sync"] = {
                "enabled": False,
            }

        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @app.get("/api/export")
    def export_reports(
        request: Request,

        format: Literal[
            "csv",
            "json",
        ] = Query("csv"),

        waterway: str | None = Query(None),

        severity: list[str] | None = Query(None),

        status: list[str] | None = Query(None),

        min_score: int | None = Query(
            None,
            ge=0,
            le=100,
        ),

        since: str | None = Query(None),

        until: str | None = Query(None),

        bbox: str | None = Query(None),
    ):

        filters = Filters(
            waterway=waterway,
            severity=severity,
            status=status,
            min_score=min_score,
            since=since,
            until=until,
            bbox=bbox,
        )

        rows, _ = request.app.state.store.list(
            limit=100000,
            offset=0,
            sort="newest",
            **filters.as_kwargs(),
        )

        public_rows = [
            _public(row)
            for row in rows
        ]

        if format == "json":

            payload = json.dumps(
                public_rows,
                indent=2,
                default=str,
            )

            return StreamingResponse(
                iter([payload]),
                media_type="application/json",
                headers={
                    "Content-Disposition":
                        'attachment; filename="riverwatch-reports.json"'
                },
            )

        output = io.StringIO()

        if public_rows:

            fieldnames = [
                "id",
                "created_at",
                "captured_at",
                "waterway",
                "location_name",
                "latitude",
                "longitude",
                "location_source",
                "location_confirmed",
                "in_philadelphia",
                "notes",
                "reporter",
                "hazard_suspected",
                "model_name",
                "model_is_stand_in",
                "score",
                "severity",
                "item_count",
                "coverage_pct",
                "authority_level",
                "status",
                "status_updated_at",
            ]

            writer = csv.DictWriter(
                output,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in public_rows:
                writer.writerow(row)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="riverwatch-reports.csv"'
            },
        )

    # ------------------------------------------------------------------
    # Frontend
    # ------------------------------------------------------------------

    @app.get(
        "/",
        include_in_schema=False,
    )
    def frontend():

        index = (
            config.FRONTEND_DIR
            / "index.html"
        )

        if not index.is_file():
            raise HTTPException(
                404,
                (
                    "Front end not found. "
                    "The API docs are at /docs"
                ),
            )

        return FileResponse(
            index,
            media_type="text/html",
        )

    return app


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _public(
    report: dict | None,
) -> dict | None:

    if report is None:
        return None

    result = dict(report)

    report_id = result["id"]

    # Don't expose internal filesystem paths.
    result.pop(
        "image_path",
        None,
    )

    result.pop(
        "annotated_path",
        None,
    )

    result["image_url"] = (
        f"/api/reports/{report_id}/image"
    )

    result["annotated_image_url"] = (
        f"/api/reports/{report_id}/image"
        "?annotated=true"
    )

    # Every report carries what the score does and doesn't mean.
    result["score_note"] = SCORE_NOTE

    return result


def _in_philadelphia(
    latitude: float,
    longitude: float,
) -> bool:

    bounds = config.PHILLY_BOUNDS

    return (
        bounds["lat_min"]
        <= latitude
        <= bounds["lat_max"]
        and bounds["lon_min"]
        <= longitude
        <= bounds["lon_max"]
    )


def _read_exif(
    image: Image.Image,
) -> dict:

    """
    Extract capture time and GPS coordinates when present.

    EXIF is optional. Failure to parse EXIF should never prevent a report
    from being submitted.
    """

    result = {
        "captured_at": None,
        "latitude": None,
        "longitude": None,
    }

    try:
        exif = image.getexif()
    except Exception:
        return result

    if not exif:
        return result

    # --------------------------------------------------------------
    # Capture time
    # --------------------------------------------------------------

    for tag_id, value in exif.items():

        tag = ExifTags.TAGS.get(
            tag_id,
            tag_id,
        )

        if tag in (
            "DateTimeOriginal",
            "DateTimeDigitized",
            "DateTime",
        ):
            try:
                # Common EXIF format:
                # YYYY:MM:DD HH:MM:SS
                parsed = time.strptime(
                    str(value),
                    "%Y:%m:%d %H:%M:%S",
                )

                result["captured_at"] = (
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%S",
                        parsed,
                    )
                )

                break

            except (
                ValueError,
                TypeError,
            ):
                pass

    # --------------------------------------------------------------
    # GPS
    # --------------------------------------------------------------

    try:
        gps_ifd = exif.get_ifd(
            ExifTags.IFD.GPSInfo
        )
    except Exception:
        gps_ifd = None

    if not gps_ifd:
        return result

    gps = {
        ExifTags.GPSTAGS.get(
            key,
            key,
        ): value
        for key, value
        in gps_ifd.items()
    }

    try:
        latitude = _gps_decimal(
            gps["GPSLatitude"]
        )

        longitude = _gps_decimal(
            gps["GPSLongitude"]
        )

        if gps.get(
            "GPSLatitudeRef",
            "N",
        ) == "S":
            latitude = -latitude

        if gps.get(
            "GPSLongitudeRef",
            "E",
        ) == "W":
            longitude = -longitude

        result["latitude"] = latitude
        result["longitude"] = longitude

    except (
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        pass

    return result


def _gps_decimal(
    values,
) -> float:

    degrees = float(
        values[0]
    )

    minutes = float(
        values[1]
    )

    seconds = float(
        values[2]
    )

    return (
        degrees
        + minutes / 60.0
        + seconds / 3600.0
    ) 


app = create_app()