"""Build a clean single-class dent dataset from the validated hybrid export."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse paths for the derived dent-only dataset."""
    parser = argparse.ArgumentParser(
        description="Build a dent-only YOLO dataset from the hybrid local export."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/coin-defect-hybrid"),
        help="Validated hybrid YOLO dataset root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Full hybrid dataset manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/coin-dent-v1"),
        help="Derived dent-only YOLO dataset root.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/dent_dataset_manifest.csv"),
        help="Selected dent-only manifest output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/dent_dataset_report.json"),
        help="Derived dataset selection report output.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace generated images and labels in the output root.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read the manifest while retaining its documented column order."""
    with path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        return reader.fieldnames, list(reader)


def manifest_image_id(exported_image_id: str) -> str:
    """Map a Roboflow export filename back to the original manifest image ID."""
    source_name = exported_image_id.partition(".rf.")[0]
    return re.sub(r"_(?:jpe?g|png)$", "", source_name, flags=re.IGNORECASE)


def source_images(source_root: Path) -> dict[str, Path]:
    """Index source images by their manifest image ID."""
    images: dict[str, Path] = {}
    for split in SPLITS:
        for image_path in (source_root / "images" / split).iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = manifest_image_id(image_path.stem)
            if image_id in images:
                raise ValueError(f"Duplicate source export for {image_id}.")
            images[image_id] = image_path
    return images


def is_clean_dent_or_normal(row: dict[str, str]) -> bool:
    """Keep dent images and empty-label normal images only."""
    defect_classes = {
        value.strip() for value in row["defect_classes"].split(",") if value.strip()
    }
    return not defect_classes or defect_classes == {"dent"}


def dent_label_content(
    label_path: Path, image_id: str, defect_classes: str
) -> str:
    """Keep class-0 dent boxes while rejecting ambiguous source annotations."""
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    is_normal = not defect_classes.strip()

    if is_normal:
        if lines:
            raise ValueError(f"{image_id} is marked normal but has YOLO labels.")
        return ""

    dent_lines: list[str] = []
    for line in lines:
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}: expected 5 YOLO values, got {len(values)}.")
        if values[0] != "0":
            raise ValueError(
                f"{image_id} is selected as dent-only but has non-dent class {values[0]}."
            )
        dent_lines.append(line)

    if not dent_lines:
        raise ValueError(f"{image_id} is marked dent but has no dent YOLO label.")
    return "\n".join(dent_lines) + "\n"


def prepare_output(output_root: Path, replace: bool) -> None:
    """Create output directories without deleting tracked configuration files."""
    generated_directories = (output_root / "images", output_root / "labels")
    if any(path.exists() for path in generated_directories) and not replace:
        raise FileExistsError(
            f"Generated dataset already exists at {output_root}. Use --replace to rebuild it."
        )

    for path in generated_directories:
        if path.exists():
            shutil.rmtree(path)
        for split in SPLITS:
            (path / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Select clean dent/normal samples and write a single-class YOLO dataset."""
    args = parse_args()
    columns, manifest = read_manifest(args.manifest)
    source_by_image_id = source_images(args.source_root)
    selected_rows = [row for row in manifest if is_clean_dent_or_normal(row)]

    missing_images = [
        row["image_id"] for row in selected_rows if row["image_id"] not in source_by_image_id
    ]
    if missing_images:
        raise FileNotFoundError(
            "Selected manifest images are missing from the source dataset: "
            + ", ".join(missing_images)
        )

    prepare_output(args.output_root, args.replace)
    split_counts: Counter[str] = Counter()
    coin_ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    for row in selected_rows:
        image_id = row["image_id"]
        split = row["split"]
        image_path = source_by_image_id[image_id]
        source_label = (
            image_path.parent.parent.parent / "labels" / split / f"{image_path.stem}.txt"
        )
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing YOLO label file: {source_label}")

        label_content = dent_label_content(
            source_label, image_id, row["defect_classes"]
        )
        output_image = args.output_root / "images" / split / image_path.name
        output_label = args.output_root / "labels" / split / source_label.name
        shutil.copy2(image_path, output_image)
        output_label.write_text(label_content, encoding="utf-8")

        split_counts[split] += 1
        coin_ids_by_split[split].add(row["coin_id"])

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected_rows)

    report = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "class_names": ["dent"],
        "selected_images": len(selected_rows),
        "split_image_counts": dict(split_counts),
        "split_coin_ids": {
            split: sorted(coin_ids) for split, coin_ids in coin_ids_by_split.items()
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        "Dent dataset built: "
        f"{len(selected_rows)} images, "
        + ", ".join(f"{split}={split_counts[split]}" for split in SPLITS)
    )


if __name__ == "__main__":
    main()
