"""Shared validation and matching helpers for the v13 YOLO smoke run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ultralytics.utils import YAML


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_SPLIT_COUNTS = {"train": 93, "val": 6, "test": 3}


@dataclass(frozen=True)
class Annotation:
    """One normalized YOLO detection annotation."""

    class_id: int
    xywh: tuple[float, float, float, float]
    source_format: str


def load_dataset_descriptor(
    config_path: Path, repository_root: Path | None = None
) -> tuple[Path, dict[str, Path], list[str]]:
    """Resolve a project-relative dataset descriptor without YOLO's path fallback."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    data = YAML.load(config_path)
    names_value = data.get("names")
    if isinstance(names_value, dict):
        names = [str(names_value[index]) for index in range(len(names_value))]
    elif isinstance(names_value, list):
        names = [str(name) for name in names_value]
    else:
        raise ValueError(f"Dataset names must be a list or ID mapping: {config_path}")

    root_value = Path(str(data.get("path", config_path.parent)))
    if root_value.is_absolute():
        dataset_root = root_value.resolve()
    else:
        base = (repository_root or Path.cwd()).resolve()
        dataset_root = (base / root_value).resolve()

    split_dirs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if not isinstance(split_value, str) or not split_value:
            raise ValueError(f"Dataset descriptor is missing a string '{split}' path")
        split_dirs[split] = (dataset_root / split_value).resolve()
    return dataset_root, split_dirs, names


def polygon_to_box(coordinates: Iterable[float]) -> tuple[float, float, float, float]:
    """Convert normalized polygon x/y coordinates to normalized YOLO xywh."""
    values = tuple(float(value) for value in coordinates)
    if len(values) < 6 or len(values) % 2:
        raise ValueError("A polygon needs at least three x/y points")
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        x_max - x_min,
        y_max - y_min,
    )


def parse_label_file(label_path: Path, class_count: int) -> list[Annotation]:
    """Load YOLO boxes and convert polygon rows to enclosing detection boxes."""
    annotations: list[Annotation] = []
    formats: set[str] = set()
    for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            values = [float(value) for value in raw_line.split()]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: non-numeric label") from exc
        if len(values) < 5:
            raise ValueError(f"{label_path}:{line_number}: expected a box or polygon")
        if not values[0].is_integer():
            raise ValueError(f"{label_path}:{line_number}: class ID must be an integer")
        class_id = int(values[0])
        if not 0 <= class_id < class_count:
            raise ValueError(
                f"{label_path}:{line_number}: class ID {class_id} is outside 0..{class_count - 1}"
            )

        if len(values) == 5:
            xywh = tuple(values[1:5])
            source_format = "box"
        else:
            xywh = polygon_to_box(values[1:])
            source_format = "polygon"
        if any(value < 0 or value > 1 for value in xywh):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be normalized")
        if xywh[2] <= 0 or xywh[3] <= 0:
            raise ValueError(f"{label_path}:{line_number}: box width and height must be positive")
        formats.add(source_format)
        annotations.append(Annotation(class_id, xywh, source_format))

    if len(formats) > 1:
        raise ValueError(f"{label_path}: cannot mix box and polygon rows in one label file")
    return annotations


def validate_dataset(
    config_path: Path,
    expected_counts: dict[str, int] | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Validate split paths, one-to-one image labels, counts, and annotations."""
    expected_counts = expected_counts or EXPECTED_SPLIT_COUNTS
    dataset_root, split_dirs, names = load_dataset_descriptor(config_path, repository_root)
    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "classes": names,
        "splits": {},
        "polygon_rows_converted": 0,
    }
    split_report: dict[str, dict[str, int | str]] = {}
    polygon_count = 0
    for split, image_dir in split_dirs.items():
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Missing {split} image directory: {image_dir}")
        if not label_dir.is_dir():
            raise NotADirectoryError(f"Missing {split} label directory: {label_dir}")
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            missing_images = sorted(label_stems - image_stems)
            raise ValueError(
                f"{split} image/label mismatch; missing labels={missing_labels}, "
                f"missing images={missing_images}"
            )
        expected = expected_counts.get(split)
        if expected is not None and len(images) != expected:
            raise ValueError(f"{split} has {len(images)} images; expected {expected}")
        annotation_count = 0
        for label_path in labels:
            annotations = parse_label_file(label_path, len(names))
            annotation_count += len(annotations)
            polygon_count += sum(item.source_format == "polygon" for item in annotations)
        split_report[split] = {
            "image_dir": str(image_dir),
            "images": len(images),
            "labels": len(labels),
            "annotations": annotation_count,
        }
    report["splits"] = split_report
    report["polygon_rows_converted"] = polygon_count
    return report


def normalized_xywh_to_xyxy(
    xywh: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a normalized YOLO box to pixel xyxy coordinates."""
    x_center, y_center, box_width, box_height = xywh
    return (
        (x_center - box_width / 2) * width,
        (y_center - box_height / 2) * height,
        (x_center + box_width / 2) * width,
        (y_center + box_height / 2) * height,
    )


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Calculate IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_detections(
    ground_truth: list[dict[str, object]],
    detections: list[dict[str, object]],
    iou_threshold: float,
) -> dict[str, object]:
    """Greedily match each GT to the best unused detection of the same class."""
    matches: list[dict[str, object]] = []
    unmatched_detection_ids = set(range(len(detections)))
    unmatched_ground_truth_ids: list[int] = []
    for ground_truth_id, truth in enumerate(ground_truth):
        candidates = [
            (box_iou(truth["xyxy"], detection["xyxy"]), detection_id)
            for detection_id, detection in enumerate(detections)
            if detection_id in unmatched_detection_ids
            and detection["class_id"] == truth["class_id"]
        ]
        best_iou, best_detection_id = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            unmatched_detection_ids.remove(best_detection_id)
            matches.append(
                {
                    "ground_truth_index": ground_truth_id,
                    "detection_index": best_detection_id,
                    "iou": best_iou,
                }
            )
        else:
            unmatched_ground_truth_ids.append(ground_truth_id)
    return {
        "matches": matches,
        "unmatched_ground_truth_indices": unmatched_ground_truth_ids,
        "unmatched_detection_indices": sorted(unmatched_detection_ids),
    }


def smoke_image_passed(
    ground_truth: list[dict[str, object]], match_result: dict[str, object]
) -> bool:
    """Require all positive GTs to match, and no detections on a normal image."""
    if ground_truth:
        return not match_result["unmatched_ground_truth_indices"]
    return not match_result["unmatched_detection_indices"]
