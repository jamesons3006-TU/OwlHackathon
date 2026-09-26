import httpx
import pytest
from PIL import Image

from app.detector import HttpDetector, MockDetector, ModelUnavailable, parse_detections


@pytest.mark.parametrize("fmt,bbox", [
    ("xyxy", [100, 50, 300, 150]),
    ("xywh", [100, 50, 200, 100]),
    ("cxcywh", [200, 100, 200, 100]),
    ("cxcywhn", [0.5, 0.25, 0.5, 0.25]),
    ("xyxyn", [0.25, 0.125, 0.75, 0.375]),
])
def test_bbox_formats_normalize_to_pixel_xywh(fmt, bbox):
    [d] = parse_detections({"bbox_format": fmt, "detections": [{"label": "garbage", "confidence": 0.9, "bbox": bbox}]},
                           width=400, height=400)
    assert (d.x, d.y, d.w, d.h) == pytest.approx((100, 50, 200, 100))


def test_aliases_bare_list_and_threshold():
    dets = parse_detections([
        {"class": "bottle", "score": 0.8, "box": [0, 0, 10, 10]},
        {"name": "bag", "conf": 0.05, "box": [0, 0, 10, 10]},  # below confidence threshold
    ], 100, 100)
    assert [(d.label, d.confidence) for d in dets] == [("bottle", 0.8)]


def test_malformed_response_raises():
    with pytest.raises(ModelUnavailable):
        parse_detections({"detections": [{"bbox": [1, 2]}]}, 100, 100)


def _http_detector(handler):
    return HttpDetector("http://model/predict", api_key="k", transport=httpx.MockTransport(handler))


def test_http_detector_round_trip():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["multipart"] = b'name="image"' in request.content
        return httpx.Response(200, json={"model": "garbage-v2", "detections": [
            {"label": "garbage", "confidence": 0.9, "bbox": [10, 10, 50, 40]}]})

    det = _http_detector(handler)
    [d] = det.detect(Image.new("RGB", (100, 100)))
    assert (d.x, d.y, d.w, d.h) == (10, 10, 40, 30)
    assert det.name == "garbage-v2"
    assert seen == {"auth": "Bearer k", "multipart": True}


def test_http_detector_errors_become_model_unavailable():
    det = _http_detector(lambda r: httpx.Response(500))
    with pytest.raises(ModelUnavailable):
        det.detect(Image.new("RGB", (10, 10)))
    assert det.health()["reachable"] is False


@pytest.mark.parametrize("save_as", ["state_dict", "checkpoint", "whole_model"])
def test_faster_rcnn_checkpoint_formats(tmp_path, save_as):
    import torch
    import torchvision
    from app import config
    from app.detector import FasterRCNNDetector

    arch = "fasterrcnn_mobilenet_v3_large_320_fpn"  # small and quick to build on CPU
    model = getattr(torchvision.models.detection, arch)(weights=None, weights_backbone=None, num_classes=3)
    path = tmp_path / "garbage.pth"
    torch.save({"state_dict": model.state_dict(), "whole_model": model,
                "checkpoint": {"model_state_dict": model.state_dict(), "epoch": 12}}[save_as], path)

    det = FasterRCNNDetector(path, arch=arch, class_names=["bottle", "bag"])
    assert det.class_names == ["__background__", "bottle", "bag"]
    assert det._label(1) == "bottle" and det._label(7) == "class_7"

    # Untrained weights give arbitrary boxes; check the plumbing, not the predictions.
    config_threshold = config.CONFIDENCE_THRESHOLD
    config.CONFIDENCE_THRESHOLD = 0.0
    try:
        dets = det.detect(Image.new("RGB", (320, 240), (40, 90, 120)))
    finally:
        config.CONFIDENCE_THRESHOLD = config_threshold
    assert all(d.label in ("bottle", "bag") and d.w >= 0 and d.h >= 0 for d in dets)


def test_faster_rcnn_rejects_other_checkpoints(tmp_path):
    import torch
    from app.detector import FasterRCNNDetector

    path = tmp_path / "not_frcnn.pth"
    torch.save({"conv.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="Faster R-CNN"):
        FasterRCNNDetector(path)


def test_mock_is_deterministic():
    img = Image.new("RGB", (640, 480), (20, 80, 120))
    assert MockDetector().detect(img) == MockDetector().detect(img)
