"""Evaluate the fixed v13 three-image smoke gate and preserve visual evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from v13_pipeline import (
    IMAGE_EXTENSIONS,
    load_dataset_descriptor,
    match_detections,
    normalized_xywh_to_xyxy,
    parse_label_file,
    smoke_image_passed,
)


EXPECTED_TEST_CASES = {"C005_03": 1, "C013_01": 0, "C006_01": None}


def test_case_id(image_path: Path) -> str:
    """Extract the stable capture ID from a Roboflow-exported filename."""
    match = re.search(r"self_(C\d{3}_\d{2})_", image_path.name)
    if not match:
        raise ValueError(f"Cannot identify fixed v13 test case: {image_path.name}")
    return match.group(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-config", type=Path, default=Path("datasets/coin-defect-v13/data.yaml"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-failure", action="store_true", help="Write diagnostics without a failing exit code.")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if not 0 <= args.match_iou <= 1:
        parser.error("--match-iou must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    _, split_dirs, names = load_dataset_descriptor(args.dataset_config)
    test_dir = split_dirs["test"]
    images = sorted(path for path in test_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    case_ids = {test_case_id(image_path) for image_path in images}
    if case_ids != set(EXPECTED_TEST_CASES):
        raise ValueError(
            f"Expected fixed test cases {sorted(EXPECTED_TEST_CASES)}, got {sorted(case_ids)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    image_reports: list[dict[str, object]] = []

    for image_path in images:
        case_id = test_case_id(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = parse_label_file(test_dir.parent / "labels" / f"{image_path.stem}.txt", len(names))
        ground_truth = [
            {
                "class_id": item.class_id,
                "class_name": names[item.class_id],
                "xyxy": list(normalized_xywh_to_xyxy(item.xywh, width, height)),
                "source_format": item.source_format,
            }
            for item in annotations
        ]
        expected_class = EXPECTED_TEST_CASES[case_id]
        ground_truth_classes = [item["class_id"] for item in ground_truth]
        expected_classes = [] if expected_class is None else [expected_class]
        if ground_truth_classes != expected_classes:
            raise ValueError(
                f"{case_id} expected GT classes {expected_classes}, got {ground_truth_classes}"
            )
        result = model.predict(
            source=str(image_path),
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )[0]
        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "confidence": float(box.conf.item()),
                        "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    }
                )
        match_result = match_detections(ground_truth, detections, args.match_iou)
        passed = smoke_image_passed(ground_truth, match_result)
        annotated_path = args.output_dir / f"pred_{image_path.stem}.jpg"
        result.save(filename=str(annotated_path))
        image_report = {
            "case_id": case_id,
            "source": str(image_path),
            "annotated_image": str(annotated_path),
            "ground_truth": ground_truth,
            "detections": detections,
            **match_result,
            "passed": passed,
        }
        image_reports.append(image_report)
        image_report_path = args.output_dir / f"result_{image_path.stem}.json"
        image_report_path.write_text(
            json.dumps(image_report, ensure_ascii=False, indent=2) + "\n"
        )

    passed = len(image_reports) == 3 and all(bool(item["passed"]) for item in image_reports)
    report = {
        "status": "passed" if passed else "failed",
        "model": str(args.model),
        "dataset_config": str(args.dataset_config),
        "confidence_threshold": args.confidence,
        "matching_iou_threshold": args.match_iou,
        "acceptance_rule": "all positive GT matched by class and IoU; no detections on normal image",
        "images": image_reports,
    }
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"v13 smoke gate: {report['status']}")
    print(f"Evidence report: {report_path}")
    if not passed and not args.allow_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
