"""Run an Ultralytics pretrained YOLO model on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLO inference on a single image."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the image to inspect.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        default=0.25,
        type=float,
        help="Minimum confidence for a detection (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device such as cpu, mps, or 0. Defaults to Ultralytics auto-select.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/inference"),
        type=Path,
        help="Directory for the annotated image and JSON result.",
    )
    return parser.parse_args()


def serialize_detections(result) -> list[dict[str, float | int | str | list[float]]]:
    """Convert YOLO bounding boxes into JSON-friendly dictionaries."""
    if result.boxes is None:
        return []

    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            }
        )
    return detections


def main() -> None:
    args = parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Image not found: {args.source}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(
        source=str(args.source),
        conf=args.confidence,
        device=args.device,
        save=False,
        verbose=False,
    )
    result = results[0]
    detections = serialize_detections(result)

    annotated_image = args.output_dir / f"annotated_{args.source.stem}.jpg"
    result.save(filename=str(annotated_image))

    json_result = {
        "source": str(args.source.resolve()),
        "model": args.model,
        "confidence_threshold": args.confidence,
        "image_shape": list(result.orig_shape),
        "detection_count": len(detections),
        "detections": detections,
    }
    json_path = args.output_dir / f"detections_{args.source.stem}.json"
    json_path.write_text(
        json.dumps(json_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Detections: {len(detections)}")
    print(f"Annotated image: {annotated_image}")
    print(f"JSON result: {json_path}")


if __name__ == "__main__":
    main()
