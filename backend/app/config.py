"""Runtime settings. Override any of these with environment variables."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("PWW_DATA_DIR", BACKEND_DIR / "data"))
MODELS_DIR = Path(os.getenv("PWW_MODELS_DIR", BACKEND_DIR / "models"))

# Which model backend to use (see app/detector.py and README "Plugging in the model"):
#   auto   - PWW_MODEL_URL if set, else local weights if present, else pretrained stand-in
#   http   - call a separate model service at PWW_MODEL_URL
#   frcnn  - load a fine-tuned torchvision Faster R-CNN checkpoint (.pth) from PWW_FRCNN_MODEL
#   yolo   - load local Ultralytics weights (.pt / .onnx) from PWW_MODEL
#   mock   - fake but deterministic detections, for front-end work without a model
#   package.module:ClassName - any Python class with a detect(image) method
DETECTOR = os.getenv("PWW_DETECTOR", "auto")

# Faster R-CNN checkpoint: a state_dict, a {"model_state_dict": ...} training checkpoint, or a whole saved model.
FRCNN_MODEL_PATH = Path(os.getenv("PWW_FRCNN_MODEL", MODELS_DIR / "garbage.pth"))
# torchvision builder used to rebuild the network before loading a state_dict:
# fasterrcnn_resnet50_fpn | fasterrcnn_resnet50_fpn_v2 | fasterrcnn_mobilenet_v3_large_fpn | fasterrcnn_mobilenet_v3_large_320_fpn
FRCNN_ARCH = os.getenv("PWW_FRCNN_ARCH", "fasterrcnn_resnet50_fpn")
# Class names in training order, WITHOUT the background class (index 0 in torchvision).
CLASS_NAMES = [c.strip() for c in os.getenv("PWW_CLASS_NAMES", "garbage").split(",") if c.strip()]

# YOLO weights. If no model is found at all, the API falls back to a general-purpose pretrained model.
GARBAGE_MODEL_PATH = Path(os.getenv("PWW_MODEL", MODELS_DIR / "garbage.pt"))
FALLBACK_MODEL = os.getenv("PWW_FALLBACK_MODEL", "yolo11n.pt")

# Remote model service
MODEL_URL = os.getenv("PWW_MODEL_URL", "")
MODEL_API_KEY = os.getenv("PWW_MODEL_API_KEY", "")
MODEL_TIMEOUT = float(os.getenv("PWW_MODEL_TIMEOUT", "60"))

CONFIDENCE_THRESHOLD = float(os.getenv("PWW_CONFIDENCE", "0.25"))
MAX_UPLOAD_BYTES = int(os.getenv("PWW_MAX_UPLOAD_MB", "20")) * 1024 * 1024

# Comma-separated list of front-end origins allowed to call the API.
CORS_ORIGINS = [o.strip() for o in os.getenv("PWW_CORS_ORIGINS", "*").split(",") if o.strip()]

# Rough bounding box around the City of Philadelphia, used to flag
# reports whose coordinates fall outside the city.
PHILLY_BOUNDS = {"lat_min": 39.86, "lat_max": 40.14, "lon_min": -75.29, "lon_max": -74.95}
