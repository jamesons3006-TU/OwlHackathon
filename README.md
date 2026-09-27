# OwlHackathon: Philly Water Watch

A Temple hackathon project that helps people find and report visible litter in Philadelphia's waterways. A user uploads a photo, sees the detected trash boxed, and confirms the location on a map. The app computes an experimental 0–100 visible-litter severity score, recommends who to contact, and saves the report for research and cleanup planning. A heat map and a bubble map show hotspots, and a cleanup dashboard lets community groups and city teams filter reports and track their status.

```
photo ──► detector (fine-tuned Faster R-CNN) ──► boxes ──► 0–100 visible-litter score ──► who to contact
                                                     │
          map pin confirmed ──► SQLite + photo archive ──┬──► heat map / bubble map
                                                         ├──► cleanup dashboard (status tracking)
                                                         └──► research export (CSV / JSON / COCO / ZIP)
```

> **What the score means:** it describes litter *visible in the photo*. It says nothing about chemical pollution, bacteria or whether the water is safe. Every API response carries this as `score_note`. The optional `hazard_suspected` checkbox (oil sheen, chemical smell, sewage) never changes the score; it only moves emergency contacts (PA DEP, National Response Center) to the top of the recommendations.

## Run the backend

Use PowerShell from the repository root, or `cd` into `backend` first. The key is that the virtual environment and app module must be resolved from the backend folder.

```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If you are already inside `backend`, omit the first `cd` and keep the same `.venv\Scripts\python.exe -m ...` commands.

Open the dashboard at http://127.0.0.1:8000/ (the backend serves [`frontend/index.html`](frontend/index.html)). Interactive API docs: http://127.0.0.1:8000/docs. For front-end work without a model, set `$env:PWW_DETECTOR="mock"` first.

Run the tests with `.\.venv\Scripts\python.exe -m pytest` from the `backend` directory.

## Front end

[`frontend/index.html`](frontend/index.html) is the Riverwatch dashboard: one HTML file with no build step (Leaflet and the font load from CDNs).

- **Overview:** stats, bubble map / heat map, priority hotspots and the report table.
- **Report a sighting:** upload a photo, see the boxed detections, score and who to contact, then drop a pin on the map to confirm the location. GPS in the photo places the pin automatically.
- **Reports / Cleanups:** filter by severity and status, open a report, move it through the cleanup workflow, export CSV or the research ZIP.
- **My impact:** your reports compared with the community. The "Reporting as" name is sent as `reporter` on uploads.

It calls the API on the same origin when the backend serves it. Opened straight from disk it uses `http://127.0.0.1:8000`; anywhere else, add `?api=http://host:port` to the URL. Set `PWW_FRONTEND_DIR` to serve it from another folder.

## Plugging in the model

The backend never calls a model directly. It goes through a detector chosen by `PWW_DETECTOR`, so the trained model can be connected in whichever form is easiest:

| Option | Settings | When to use |
|---|---|---|
| **A. Faster R-CNN checkpoint** | Put `garbage.pth` in `backend/models/`. Set `PWW_FRCNN_ARCH` if not `fasterrcnn_resnet50_fpn`, and `PWW_CLASS_NAMES` if not just `garbage` | Model runs on the same machine as the API |
| **B. Model as its own API** | `PWW_MODEL_URL=http://host:9000/predict` (+ `PWW_MODEL_API_KEY`) | Model runs on a GPU machine or Colab |
| **C. YOLO weights** | `garbage.pt` / `.onnx` in `backend/models/` | Ultralytics model |
| **D. Python class** | `PWW_DETECTOR=my_module:MyDetector` | Anything else; class needs `name`, `is_stand_in` and `detect(pil_image)` |
| **E. Mock** | `PWW_DETECTOR=mock` | Front-end work without any model |

With the default `PWW_DETECTOR=auto`, the backend picks the first that exists: `PWW_MODEL_URL`, then `models/garbage.pth`, then `models/garbage.pt`, then the general-purpose `yolo11n.pt` stand-in.

### Faster R-CNN checkpoints (option A)

Any of the usual ways of saving from a torchvision training script works:

```python
torch.save(model.state_dict(), "garbage.pth")                                   # plain state_dict
torch.save({"model_state_dict": model.state_dict(), "epoch": e, ...}, "garbage.pth")  # training checkpoint
torch.save(model, "garbage.pth")                                                # whole model
```

The number of classes is read from the checkpoint. Tell the backend two things:

- `PWW_FRCNN_ARCH`: the torchvision builder used in training: `fasterrcnn_resnet50_fpn` (default), `fasterrcnn_resnet50_fpn_v2`, `fasterrcnn_mobilenet_v3_large_fpn` or `fasterrcnn_mobilenet_v3_large_320_fpn`.
- `PWW_CLASS_NAMES`: class names in training order, **without** background, e.g. `garbage` or `bottle,bag,can`.

Only load checkpoints from your own team: whole-model files are unpickled, which can run code.

### Model API contract (option B)

```
POST /predict            multipart/form-data, field "image" (JPEG)
Authorization: Bearer <PWW_MODEL_API_KEY>      (only if a key is set)

200 OK
{
  "model": "garbage-frcnn-v2",         // optional, saved on each report
  "bbox_format": "xyxy",               // optional: xyxy | xywh | cxcywh | cxcywhn | xyxyn
  "detections": [
    {"label": "garbage", "confidence": 0.91, "bbox": [812.4, 301.0, 866.2, 330.7]}
  ]
}
```

Pixel formats: `xyxy` corners (what torchvision returns), `xywh` COCO-style, `cxcywh` center. Normalized 0–1: `cxcywhn` (YOLO `.txt` style), `xyxyn`. Aliases are accepted (`class`/`name` for label, `score`/`conf` for confidence, `box` for bbox), and so is a bare list of detections. The backend applies its own confidence cutoff (`PWW_CONFIDENCE`), so the model can return everything.

If the model service is down or returns junk, uploads get **HTTP 503**, nothing is saved, and `/api/health` reports `"status": "degraded"`.

**Ready-made service:** [`model_server/server.py`](model_server/server.py) implements this contract for Faster R-CNN (default) or YOLO, and uses a GPU when one is available:

```powershell
cd model_server
pip install -r requirements.txt
$env:MODEL_WEIGHTS="garbage.pth"; $env:FRCNN_ARCH="fasterrcnn_resnet50_fpn"; $env:CLASS_NAMES="garbage"
$env:MODEL_API_KEY="choose-a-secret"
uvicorn server:app --host 0.0.0.0 --port 9000
```

Every report stores `model_name` and `model_is_stand_in`, so researchers can tell which model scored it.

## API

### Reporting

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/reports` | Upload a photo (multipart). Returns detections, score, severity and recommendations |
| `PATCH` | `/api/reports/{id}/location` | Save the location the user confirmed on the map. JSON `{"latitude", "longitude", "location_name"}` (name optional) |
| `GET` | `/api/reports/{id}` | One report, including its status history |
| `GET` | `/api/reports/{id}/image?annotated=true` | Original photo, or the copy with detection boxes drawn |

`POST /api/reports` form fields: `image` (required), `waterway`, `location_name`, `latitude`, `longitude`, `notes`, `reporter`, `hazard_suspected`. If latitude/longitude are omitted, GPS is read from the photo's EXIF data when present, so the map can open on that spot. `location_source` records where the position came from (`reporter`, `exif` or `map`), and `location_confirmed` becomes `true` once the user confirms it.

### Maps

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/map/points` | **Heat map:** GeoJSON `FeatureCollection` of located reports. `properties.weight` = score / 100 |
| `GET` | `/api/map/bubbles?cell_m=500` | **Bubble map:** reports grouped into grid cells (50–10,000 m). Color by `avg_score` / `avg_severity`, size by `report_count`. Also `max_score`, `open_count`, `waterways`, `latest_report_at` |

### Cleanup dashboard

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/reports?sort=score&status=open` | Report list. `sort`: `newest`, `oldest`, `score` (highest first, for prioritizing) |
| `PATCH` | `/api/reports/{id}/status` | JSON `{"status", "note", "changed_by"}`. Every change is logged in `status_history` |
| `GET` | `/api/stats` | Totals, open reports, counts by severity and status, per-waterway averages |

Statuses: `reported` → `verified` → `scheduled` → `in_progress` → `cleaned`, or `dismissed`. `status=open` matches the first four.

### Shared filters and other endpoints

The list, both map layers and the export accept the same filters: `waterway`, `severity` (repeatable), `status` (repeatable, or `open`), `min_score`, `since`, `until`, and `bbox=min_lat,min_lon,max_lat,max_lon` for the current map viewport.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/export?format=csv\|json\|coco\|zip` | Research export. `zip` bundles photos, annotated photos, CSV, JSON and COCO labels |
| `GET` | `/api/meta` | Waterways, severity bands, statuses, authority tiers, score note (for dropdowns and legends) |
| `GET` | `/api/health` | Status and which model is loaded |

## Scoring

```
score = 60 × (1 − e^(−confidence-weighted item count / 5))  +  40 × min(1, frame coverage / 10%)
```

| Score | Severity | Authority level | Recommended contact |
|---|---|---|---|
| 0 | none | 0 | none |
| 1–24 | low | 1 | Community / volunteer cleanup |
| 25–49 | moderate | 2 | Philly311 |
| 50–74 | high | 3 | Philadelphia Water Department |
| 75–100 | severe | 4 | PA DEP Southeast Region |
| any + `hazard_suspected` | (unchanged) | 5 | National Response Center + PA DEP |

The score is experimental: the weights are a starting point to tune against real photos in [`backend/app/scoring.py`](backend/app/scoring.py). Contacts live in [`backend/app/recommendations.py`](backend/app/recommendations.py). **Verify phone numbers before a public launch.**

## Storage

Everything lives in `backend/data/` (git-ignored): `reports.db` (SQLite), `images/YYYY/MM/<id>.jpg` and `annotated/YYYY/MM/<id>.jpg`. Set `PWW_DATA_DIR` to store it elsewhere. Databases from earlier versions are upgraded automatically on startup.

Other settings: `PWW_DETECTOR`, `PWW_FRCNN_MODEL`, `PWW_FRCNN_ARCH`, `PWW_CLASS_NAMES`, `PWW_MODEL_URL`, `PWW_MODEL_API_KEY`, `PWW_MODEL_TIMEOUT` (default 60 s), `PWW_MODEL`, `PWW_CONFIDENCE` (default 0.25), `PWW_MAX_UPLOAD_MB` (default 20), `PWW_CORS_ORIGINS` (default `*`), `PWW_FRONTEND_DIR` (default `frontend/`).

## Training notebook

[`training/train_colab.ipynb`](training/train_colab.ipynb) trains a YOLO model on `WaterDataset.zip`. It was an early alternative; the team is fine-tuning Faster R-CNN separately.
