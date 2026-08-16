"""Validate a local YOLO export and its Coin-AOI data manifest."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = ("dent", "scratch", "stain_corrosion")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = {
    "image_id",
    "coin_id",
    "source_type",
    "source_url",
    "license",
    "capture_setup",
    "defect_classes",
    "split",
    "annotation_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Coin-AOI YOLO labels, image pairing, and data splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/coin-defect-hybrid"),
        help="YOLO export root containing images/ and labels/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="CSV created from data/manifest_template.csv.",
    )
    return parser.parse_args()


def read_manifest(path: Path, errors: list[str]) -> dict[str, dict[str, str]]:
    if not path.is_file():
        errors.append(
            f"Manifest not found: {path}. Copy data/manifest_template.csv to this path."
        )
        return {}

    with path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            errors.append(
                f"Manifest is missing columns: {', '.join(sorted(missing_columns))}"
            )
            return {}

        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            image_id = row["image_id"].strip()
            coin_id = row["coin_id"].strip()
            split = row["split"].strip()

            if not image_id:
                errors.append(f"Manifest line {line_number}: image_id is empty.")
                continue
            if image_id in rows:
                errors.append(f"Manifest line {line_number}: duplicate image_id {image_id}.")
            if not coin_id:
                errors.append(f"Manifest line {line_number}: coin_id is empty.")
            if split not in SPLITS:
                errors.append(
                    f"Manifest line {line_number}: split must be one of {SPLITS}, got {split!r}."
                )
            rows[image_id] = {key: value.strip() for key, value in row.items()}
    return rows


def validate_manifest_splits(
    manifest: dict[str, dict[str, str]], errors: list[str]
) -> None:
    coin_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest.values():
        coin_splits[row["coin_id"]].add(row["split"])

    for coin_id, splits in sorted(coin_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"Data leakage risk: coin_id {coin_id!r} appears in splits "
                f"{', '.join(sorted(splits))}."
            )


def validate_label(label_path: Path, errors: list[str]) -> None:
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = line.split()
        if len(values) != 5:
            errors.append(
                f"{label_path}:{line_number}: expected 5 YOLO values, got {len(values)}."
            )
            continue

        try:
            class_id = int(values[0])
            x_center, y_center, width, height = (float(value) for value in values[1:])
        except ValueError:
            errors.append(f"{label_path}:{line_number}: values must be numeric.")
            continue

        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(
                f"{label_path}:{line_number}: class_id {class_id} is not in "
                f"0..{len(CLASS_NAMES) - 1}."
            )
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            errors.append(
                f"{label_path}:{line_number}: box center must be within 0..1."
            )
        if not 0 < width <= 1 or not 0 < height <= 1:
            errors.append(
                f"{label_path}:{line_number}: box width and height must be within (0, 1]."
            )
        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            errors.append(
                f"{label_path}:{line_number}: bounding box extends outside the image."
            )


def manifest_image_id(exported_image_id: str) -> str:
    """Map Roboflow's exported filename back to its original image ID."""
    source_name = exported_image_id.partition(".rf.")[0]
    return re.sub(r"_(?:jpe?g|png)$", "", source_name, flags=re.IGNORECASE)


def validate_export(
    dataset_root: Path,
    manifest: dict[str, dict[str, str]],
    errors: list[str],
) -> int:
    image_ids: set[str] = set()

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"Missing image directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"Missing label directory: {label_dir}")
            continue

        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in images:
            exported_image_id = image_path.stem
            image_id = manifest_image_id(exported_image_id)
            image_ids.add(image_id)
            label_path = label_dir / f"{exported_image_id}.txt"

            if image_id not in manifest:
                errors.append(f"{image_path}: missing manifest row for {image_id}.")
            elif manifest[image_id]["split"] != split:
                errors.append(
                    f"{image_path}: manifest split is {manifest[image_id]['split']!r}, "
                    f"not {split!r}."
                )

            if not label_path.is_file():
                errors.append(
                    f"{image_path}: missing label file {label_path.name}. "
                    "Normal images require an empty label file."
                )
                continue
            validate_label(label_path, errors)

        for label_path in label_dir.glob("*.txt"):
            if not any(image_path.stem == label_path.stem for image_path in images):
                errors.append(f"{label_path}: has no matching image in {image_dir}.")

    for image_id in manifest:
        if image_id not in image_ids:
            errors.append(f"Manifest image_id {image_id!r} has no exported image.")

    return len(image_ids)


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    manifest = read_manifest(args.manifest, errors)
    validate_manifest_splits(manifest, errors)
    image_count = validate_export(args.dataset_root, manifest, errors)

    if errors:
        print(f"Dataset validation failed with {len(errors)} issue(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(
        "Dataset validation passed: "
        f"{image_count} images, {len(manifest)} manifest rows, "
        f"classes={', '.join(CLASS_NAMES)}."
    )


if __name__ == "__main__":
    main()
