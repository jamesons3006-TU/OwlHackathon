"""Pluggable object detectors.

Every detector exposes the same interface:

    name: str               - recorded on every report
    is_stand_in: bool       - True when results don't come from the trained garbage model
    detect(image) -> list[Detection]
    health() -> dict        - optional status info for /api/health

Pick one with the PWW_DETECTOR setting (see config.py).
"""
from __future__ import annotations

import hashlib
import importlib
import io
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol

from PIL import Image

from . import config

# When running the general-purpose COCO model as a stand-in, only these
# classes are treated as litter.
COCO_LITTER_CLASSES = {
    "bottle", "cup", "wine glass", "bowl", "fork", "knife", "spoon",
    "handbag", "backpack", "suitcase", "umbrella", "sports ball", "frisbee",
}


class ModelUnavailable(RuntimeError):
    """The model could not produce a prediction (service down, bad response...)."""


@dataclass
class Detection:
    label: str
    confidence: float
    # Box in pixels: x, y (top-left), width, height
    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


class Detector(Protocol):
    name: str
    is_stand_in: bool

    def detect(self, image: Image.Image) -> list[Detection]: ...


# ---- local Ultralytics weights ------------------------------------------------

class YoloDetector:
    def __init__(self, weights: str | Path, is_stand_in: bool):
        from ultralytics import YOLO  # heavy import, keep it lazy

        self.model = YOLO(str(weights))
        self.name = Path(str(weights)).name
        self.is_stand_in = is_stand_in

    def detect(self, image: Image.Image) -> list[Detection]:
        result = self.model.predict(image, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
        names = result.names
        detections = []
        for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            label = names[int(cls)]
            if self.is_stand_in and label not in COCO_LITTER_CLASSES:
                continue
            x1, y1, x2, y2 = box
            detections.append(Detection(label=label, confidence=float(conf), x=x1, y=y1, w=x2 - x1, h=y2 - y1))
        return detections

    def health(self) -> dict:
        return {"type": "yolo", "reachable": True}


# ---- torchvision Faster R-CNN ------------------------------------------------

class FasterRCNNDetector:
    """Fine-tuned torchvision Faster R-CNN garbage detector."""

    def __init__(
        self,
        path: str | Path,
        arch: str = "fasterrcnn_resnet50_fpn",
        class_names: list[str] | None = None,
    ):
        import torch
        import torchvision

        self.torch = torch
        self.name = Path(path).name
        self.is_stand_in = False

        # Use GPU when available, otherwise CPU.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception:
            # Only use weights_only=False for checkpoints you trust.
            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        # Checkpoint may be an entire saved PyTorch model.
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt

        else:
            state = ckpt

            # Support common training checkpoint formats.
            for key in ("model_state_dict", "state_dict", "model"):
                if (
                    isinstance(ckpt, dict)
                    and isinstance(ckpt.get(key), dict)
                ):
                    state = ckpt[key]
                    break

            # Handle checkpoints saved with DataParallel.
            state = {
                k.removeprefix("module."): v
                for k, v in state.items()
            }

            try:
                num_classes = (
                    state[
                        "roi_heads.box_predictor.cls_score.weight"
                    ].shape[0]
                )
            except KeyError as e:
                raise RuntimeError(
                    f"{path} doesn't look like a Faster R-CNN checkpoint"
                ) from e

            builder = getattr(
                torchvision.models.detection,
                arch,
            )

            model = builder(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )

            model.load_state_dict(state)

        self.model = model.to(self.device)
        self.model.eval()

        # torchvision uses class 0 for background.
        self.class_names = (
            ["__background__"]
            + list(class_names or ["garbage"])
        )

    def _label(self, idx: int) -> str:
        if idx < len(self.class_names):
            return self.class_names[idx]

        return f"class_{idx}"

    def detect(
        self,
        image: Image.Image,
    ) -> list[Detection]:

        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(self.device)

        with self.torch.inference_mode():
            prediction = self.model([tensor])[0]

        boxes = prediction["boxes"].cpu().tolist()
        scores = prediction["scores"].cpu().tolist()
        labels = prediction["labels"].cpu().tolist()

        detections = []

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            # Our trained model:
            # 0 = background
            # 1 = garbage
            if int(label) != 1:
                continue

            if score < config.CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box

            # Keep coordinates inside the uploaded image.
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)

            x2 = min(
                float(image.width),
                x2,
            )

            y2 = min(
                float(image.height),
                y2,
            )

            # Ignore invalid boxes.
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    label="garbage",
                    confidence=float(score),
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                )
            )

        return detections

    def health(self) -> dict:
        return {
            "type": "frcnn",
            "reachable": True,
            "model": self.name,
            "device": str(self.device),
            "classes": self.class_names[1:],
        }
# ---- remote model service -----------------------------------------------------

class HttpDetector:
    """Sends the image to a model service and parses its JSON reply.

    Request:  POST <url>  multipart/form-data, field "image" (JPEG)
              header "Authorization: Bearer <key>" when PWW_MODEL_API_KEY is set
    Response: {
                "model": "garbage-yolo11n-v2",            (optional)
                "bbox_format": "xyxy",                    (optional, default "xyxy")
                "detections": [
                  {"label": "garbage", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
                ]
              }

    bbox_format is one of:
      xyxy       - pixel corners [x1, y1, x2, y2]
      xywh       - pixel top-left + size [x, y, w, h]   (COCO style)
      cxcywh     - pixel center + size [cx, cy, w, h]
      cxcywhn    - normalized 0-1 center + size          (YOLO .txt style)
      xyxyn      - normalized 0-1 corners
    Aliases accepted per detection: label/class/name, confidence/score/conf, bbox/box.
    """

    FORMATS = {"xyxy", "xywh", "cxcywh", "cxcywhn", "xyxyn"}

    def __init__(self, url: str, api_key: str = "", timeout: float = 60, transport=None):
        import httpx

        self.url = url
        self.name = url
        self.is_stand_in = False
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers, transport=transport)

    def detect(self, image: Image.Image) -> list[Detection]:
        import httpx

        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=92)
        try:
            resp = self.client.post(self.url, files={"image": ("image.jpg", buf.getvalue(), "image/jpeg")})
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as e:
            raise ModelUnavailable(f"Model service returned HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise ModelUnavailable(f"Could not reach model service: {e}") from e
        except ValueError as e:
            raise ModelUnavailable("Model service did not return JSON") from e
        if payload.get("model"):
            self.name = str(payload["model"])
        return parse_detections(payload, image.width, image.height)

    def health(self) -> dict:
        try:
            probe = Image.new("RGB", (64, 64))
            self.detect(probe)
            return {"type": "http", "url": self.url, "reachable": True}
        except ModelUnavailable as e:
            return {"type": "http", "url": self.url, "reachable": False, "error": str(e)}


def parse_detections(payload: dict | list, width: int, height: int) -> list[Detection]:
    """Normalize a model's JSON output into Detection objects (pixel xywh)."""
    if isinstance(payload, list):
        payload = {"detections": payload}
    fmt = str(payload.get("bbox_format", "xyxy")).lower()
    if fmt not in HttpDetector.FORMATS:
        raise ModelUnavailable(f"Unknown bbox_format '{fmt}'")
    out = []
    try:
        for d in payload.get("detections", []):
            label = str(d.get("label", d.get("class", d.get("name", "garbage"))))
            conf = float(d.get("confidence", d.get("score", d.get("conf", 1.0))))
            a, b, c, e = (float(v) for v in d.get("bbox", d.get("box")))
            if fmt == "xyxy":
                x, y, w, h = a, b, c - a, e - b
            elif fmt == "xywh":
                x, y, w, h = a, b, c, e
            elif fmt == "cxcywh":
                x, y, w, h = a - c / 2, b - e / 2, c, e
            elif fmt == "cxcywhn":
                x, y, w, h = (a - c / 2) * width, (b - e / 2) * height, c * width, e * height
            else:  # xyxyn
                x, y, w, h = a * width, b * height, (c - a) * width, (e - b) * height
            if conf >= config.CONFIDENCE_THRESHOLD and w > 0 and h > 0:
                out.append(Detection(label=label, confidence=conf, x=x, y=y, w=w, h=h))
    except (TypeError, ValueError, AttributeError) as e:
        raise ModelUnavailable(f"Malformed detection in model response: {e}") from e
    return out


# ---- mock ---------------------------------------------------------------------

class MockDetector:
    """Plausible fake detections, identical for identical images. For UI work only."""

    name = "mock"
    is_stand_in = True

    def detect(self, image: Image.Image) -> list[Detection]:
        seed = hashlib.sha1(image.resize((32, 32)).tobytes()).digest()
        rng = random.Random(seed)
        out = []
        for _ in range(rng.choice([0, 1, 2, 3, 3, 5, 8, 12])):
            w, h = rng.uniform(0.02, 0.08) * image.width, rng.uniform(0.02, 0.06) * image.height
            x, y = rng.uniform(0, image.width - w), rng.uniform(0.2, 0.9) * (image.height - h)
            out.append(Detection("garbage", round(rng.uniform(0.3, 0.97), 3), x, y, w, h))
        return out

    def health(self) -> dict:
        return {"type": "mock", "reachable": True}


# ---- factory ------------------------------------------------------------------

def load_detector() -> Detector:
    choice = config.DETECTOR.strip()
    if choice == "auto":
        if config.MODEL_URL:
            choice = "http"
        elif config.FRCNN_MODEL_PATH.exists():
            choice = "frcnn"
        elif config.GARBAGE_MODEL_PATH.exists():
            choice = "yolo"
        else:
            return YoloDetector(config.FALLBACK_MODEL, is_stand_in=True)

    if choice == "http":
        if not config.MODEL_URL:
            raise RuntimeError("PWW_DETECTOR=http requires PWW_MODEL_URL")
        return HttpDetector(config.MODEL_URL, config.MODEL_API_KEY, config.MODEL_TIMEOUT)
    if choice == "frcnn":
        if not config.FRCNN_MODEL_PATH.exists():
            raise RuntimeError(f"Faster R-CNN checkpoint not found: {config.FRCNN_MODEL_PATH}")
        return FasterRCNNDetector(config.FRCNN_MODEL_PATH, config.FRCNN_ARCH, config.CLASS_NAMES)
    if choice == "yolo":
        if not config.GARBAGE_MODEL_PATH.exists():
            raise RuntimeError(f"Model weights not found: {config.GARBAGE_MODEL_PATH}")
        return YoloDetector(config.GARBAGE_MODEL_PATH, is_stand_in=False)
    if choice == "mock":
        return MockDetector()
    if ":" in choice:
        module_name, class_name = choice.split(":", 1)
        return getattr(importlib.import_module(module_name), class_name)()
    raise RuntimeError(f"Unknown PWW_DETECTOR '{choice}'")
