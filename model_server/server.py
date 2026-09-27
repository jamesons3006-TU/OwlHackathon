"""Model service template: wrap the trained detector in the API the backend expects.

Run:   uvicorn server:app --host 0.0.0.0 --port 9000
Then start the backend with:   PWW_MODEL_URL=http://<host>:9000/predict

Settings (environment variables):
  MODEL_TYPE     frcnn (torchvision Faster R-CNN, default) or yolo (Ultralytics)
  MODEL_WEIGHTS  checkpoint path, default ../backend/models/model.pth
  FRCNN_ARCH     torchvision builder, default fasterrcnn_resnet50_fpn
  CLASS_NAMES    comma-separated class names without background, default "garbage"
  MODEL_API_KEY  optional shared secret; the backend sends it as a Bearer token

Only `predict()` needs to change for a different kind of model.
Contract (see backend/app/detector.py, HttpDetector):
    POST /predict   multipart field "image"
    -> {"model": str, "bbox_format": "xyxy",
        "detections": [{"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]}
"""
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

MODEL_TYPE = os.getenv("MODEL_TYPE", "frcnn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "backend" / "models" / "model.pth"

WEIGHTS = os.getenv("MODEL_WEIGHTS", str(DEFAULT_WEIGHTS))
FRCNN_ARCH = os.getenv("FRCNN_ARCH", "fasterrcnn_resnet50_fpn")
CLASS_NAMES = ["__background__"] + [c.strip() for c in os.getenv("CLASS_NAMES", "garbage").split(",") if c.strip()]
MODEL_NAME = os.getenv("MODEL_NAME", os.path.basename(WEIGHTS))
API_KEY = os.getenv("MODEL_API_KEY", "")

app = FastAPI(title="Garbage detector model service")
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    if MODEL_TYPE == "yolo":
        from ultralytics import YOLO
        _model = YOLO(WEIGHTS)
        return _model

    import torch
    import torchvision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(WEIGHTS, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)  # whole pickled model
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        state = next((ckpt[k] for k in ("model_state_dict", "state_dict", "model") if isinstance(ckpt.get(k), dict)), ckpt)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        num_classes = state["roi_heads.box_predictor.cls_score.weight"].shape[0]
        model = getattr(torchvision.models.detection, FRCNN_ARCH)(weights=None, weights_backbone=None,
                                                                   num_classes=num_classes)
        model.load_state_dict(state)
    _model = model.to(device).eval()
    _model.device_name = device
    return _model


def predict(image: Image.Image) -> list[dict]:
    """Return detections as [{"label", "confidence", "bbox": [x1, y1, x2, y2]}] in pixels.

    The backend applies its own confidence cutoff, so return everything above a low floor.
    """
    model = load_model()
    if MODEL_TYPE == "yolo":
        r = model.predict(image, conf=0.01, verbose=False)[0]
        boxes, scores, labels = r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
        names = lambda i: r.names[int(i)]
    else:
        import torch
        from torchvision.transforms.functional import to_tensor
        with torch.inference_mode():
            out = model([to_tensor(image).to(model.device_name)])[0]
        boxes, scores, labels = out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()
        names = lambda i: CLASS_NAMES[int(i)] if int(i) < len(CLASS_NAMES) else f"class_{int(i)}"
    return [
        {"label": names(c), "confidence": round(float(p), 4), "bbox": [round(v, 2) for v in box]}
        for box, p, c in zip(boxes, scores, labels) if p >= 0.01
    ]


@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...), authorization: str = Header("")):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Bad or missing API key")
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except OSError:
        raise HTTPException(415, "Not an image")
    return {"model": MODEL_NAME, "bbox_format": "xyxy", "detections": predict(img)}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "type": MODEL_TYPE, "loaded": _model is not None}
