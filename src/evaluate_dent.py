"""Create qualitative dent predictions for the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse a trained checkpoint and a directory of held-out test images."""
    parser = argparse.ArgumentParser(
        description="Save annotated dent predictions and a JSON evidence report."
    )
    parser.add_argument("--model", required=True, type=Path, help="YOLO checkpoint path.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/coin-dent-v1/images/test"),
        help="Directory of held-out test images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dent-evaluation"),
        help="Directory for annotated predictions and JSON.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Prediction confidence threshold from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO input size (default: 640).",
    )
    parser.add_argument("--device", default="cpu", help="YOLO device (default: cpu).")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1.")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32 and at least 32.")
    return args


def serialize_detections(result) -> list[dict[str, float | str | list[float]]]:
    """Convert Ultralytics boxes to a portable evidence format."""
    detections: list[dict[str, float | str | list[float]]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append(
            {
                "class_name": result.names[class_id],
                "confidence": float(box.conf.item()),
                "xyxy": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    return detections


def main() -> None:
    """Run fixed-threshold inference once across the held-out dent test images."""
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Test image directory not found: {args.source_dir}")

    images = sorted(
        path
        for path in args.source_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    report_images: list[dict[str, object]] = []

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )[0]
        annotated_path = args.output_dir / f"pred_{image_path.stem}.jpg"
        result.save(filename=str(annotated_path))
        report_images.append(
            {
                "source": str(image_path),
                "annotated_image": str(annotated_path),
                "detections": serialize_detections(result),
            }
        )

    report = {
        "model": str(args.model),
        "source_dir": str(args.source_dir),
        "confidence_threshold": args.confidence,
        "imgsz": args.imgsz,
        "device": args.device,
        "images": report_images,
    }
    report_path = args.output_dir / "detections.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Evaluated {len(report_images)} held-out images.")
    print(f"Evidence report: {report_path}")


if __name__ == "__main__":
    main()
