"""
Predict floating litter using the trained Faster R-CNN model.

Project structure:
fml-main/
├── images/
├── fasterrcnn_eval.py
├── fasterrcnn_fml.pth
├── fasterrcnn_training.py
├── pic1.jpg
├── predict_litter.py
└── ...

Usage:
    python3 predict_litter.py --image pic1.jpg

Or:
    python3 predict_litter.py
    # Then enter the image path when prompted.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "fasterrcnn_fml.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "predictions"


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model(weights_path, device):
    print(f"Loading weights from: {weights_path}")

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        features,
        2  # background + garbage
    )

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(model, image, device, threshold):
    tensor = to_tensor(image).to(device)

    with torch.inference_mode():
        prediction = model([tensor])[0]

    detections = []

    boxes = prediction["boxes"].cpu().tolist()
    scores = prediction["scores"].cpu().tolist()
    labels = prediction["labels"].cpu().tolist()

    for box, score, label in zip(boxes, scores, labels):

        # Class 1 = garbage
        if label != 1:
            continue

        if score < threshold:
            continue

        x1, y1, x2, y2 = box

        # Keep coordinates inside image
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(image.width), x2)
        y2 = min(float(image.height), y2)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "label": "garbage",
            "confidence": score,
            "box_xyxy": [x1, y1, x2, y2]
        })

    return detections


# --------------------------------------------------
# Draw bounding boxes
# --------------------------------------------------

def annotate(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    line_width = max(
        2,
        round(min(image.size) / 250)
    )

    for index, detection in enumerate(detections, start=1):

        box = detection["box_xyxy"]
        confidence = detection["confidence"]

        draw.rectangle(
            box,
            outline="#ff3b30",
            width=line_width
        )

        text = f"{index}: garbage {confidence:.0%}"

        bounds = draw.textbbox((0, 0), text)

        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]

        x = max(
            0,
            min(
                int(box[0]),
                image.width - text_width - 8
            )
        )

        y = max(
            0,
            int(box[1]) - text_height - 10
        )

        draw.rectangle(
            (
                x,
                y,
                x + text_width + 8,
                y + text_height + 8
            ),
            fill="#ff3b30"
        )

        draw.text(
            (
                x + 4,
                y + 4 - bounds[1]
            ),
            text,
            fill="white"
        )

    return annotated


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Detect floating litter using Faster R-CNN."
    )

    parser.add_argument(
        "--image",
        help="Path to image"
    )

    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help=f"Model weights (default: {DEFAULT_WEIGHTS})"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for predictions"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Open annotated image after prediction"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Validate threshold
    # --------------------------------------------------

    if not 0 <= args.threshold <= 1:
        parser.error(
            "--threshold must be between 0 and 1"
        )

    # --------------------------------------------------
    # Image path
    # --------------------------------------------------

    if args.image:
        image_path = Path(args.image).expanduser()
    else:
        raw_path = input(
            "Enter image path: "
        ).strip().strip("\"'")

        image_path = Path(raw_path).expanduser()

    # If only a filename is supplied, look inside project
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path

    image_path = image_path.resolve()

    # --------------------------------------------------
    # Weights path
    # --------------------------------------------------

    weights_path = Path(args.weights).expanduser()

    if not weights_path.is_absolute():
        weights_path = PROJECT_DIR / weights_path

    weights_path = weights_path.resolve()

    # --------------------------------------------------
    # Check files
    # --------------------------------------------------

    if not image_path.is_file():
        parser.error(
            f"Image not found: {image_path}"
        )

    if not weights_path.is_file():
        parser.error(
            f"Model weights not found: {weights_path}"
        )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    if args.device == "cuda":

        if not torch.cuda.is_available():
            parser.error(
                "CUDA is unavailable. Use --device cpu."
            )

        device = torch.device("cuda")

    elif args.device == "cpu":

        device = torch.device("cpu")

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    try:

        with Image.open(image_path) as source:

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

    except OSError as exc:

        parser.error(
            f"Cannot open image: {exc}"
        )

    print(f"Image: {image_path}")
    print(f"Resolution: {image.width}x{image.height}")

    # --------------------------------------------------
    # Run model
    # --------------------------------------------------

    model = load_model(
        weights_path,
        device
    )

    print("Running detection...")

    detections = predict_image(
        model,
        image,
        device,
        args.threshold
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = Path(
        args.output_dir
    ).expanduser()

    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Annotated image
    # --------------------------------------------------

    image_output = (
        output_dir /
        f"{image_path.stem}_detections.jpg"
    )

    annotated = annotate(
        image,
        detections
    )

    annotated.save(
        image_output,
        quality=95
    )

    # --------------------------------------------------
    # JSON results
    # --------------------------------------------------

    json_output = (
        output_dir /
        f"{image_path.stem}_detections.json"
    )

    result = {

        "image": str(image_path),

        "image_width": image.width,

        "image_height": image.height,

        "threshold": args.threshold,

        "detected_count": len(detections),

        "detections": detections
    }

    json_output.write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    # --------------------------------------------------
    # Terminal results
    # --------------------------------------------------

    print()
    print("------------------------------")
    print("DETECTION RESULTS")
    print("------------------------------")

    if detections:

        print(
            f"{len(detections)} litter item(s) detected."
        )

        for index, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"{index}. garbage "
                f"{detection['confidence']:.1%}"
            )

    else:

        print(
            "No litter detected at this threshold."
        )

    print()
    print(
        f"Annotated image: {image_output}"
    )

    print(
        f"JSON results: {json_output}"
    )

    # --------------------------------------------------
    # Show image
    # --------------------------------------------------

    if args.show:
        annotated.show()


if __name__ == "__main__":
    main()