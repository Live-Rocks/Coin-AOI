"""Build the reviewed rim-only dent dataset from the dent-v1 export."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from experiments.legacy.build_dent_dataset import (
    SPLITS,
    dent_label_content,
    prepare_output,
    read_manifest,
    source_images,
)


ALLOWED_DECISIONS = {"keep_rim", "exclude_face", "exclude_design"}
REVIEW_COLUMNS = {"image_id", "decision", "reason"}
VALIDATION_COIN_IDS = {"C011"}


def parse_args() -> argparse.Namespace:
    """Parse the reviewed source and generated dataset paths."""
    parser = argparse.ArgumentParser(
        description="Build a reviewed rim-only YOLO dataset with C011 validation."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/coin-dent-v1"),
        help="Validated dent-v1 YOLO dataset root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dent_dataset_manifest.csv"),
        help="Manifest paired with the dent-v1 source dataset.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("data/rim_dent_v2_review.csv"),
        help="Human review decisions for every source dent annotation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/coin-rim-dent-v2"),
        help="Generated rim-only YOLO dataset root.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/rim_dent_v2_manifest.csv"),
        help="Generated rim-only dataset manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/rim_dent_v2_report.json"),
        help="Generated selection and split report.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace generated images and labels in the output root.",
    )
    return parser.parse_args()


def read_review(path: Path) -> dict[str, dict[str, str]]:
    """Read and validate one review decision per source positive image."""
    with path.open(encoding="utf-8", newline="") as review_file:
        reader = csv.DictReader(review_file)
        missing_columns = REVIEW_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Review is missing columns: " + ", ".join(sorted(missing_columns))
            )

        decisions: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            image_id = row["image_id"].strip()
            decision = row["decision"].strip()
            reason = row["reason"].strip()
            if not image_id:
                raise ValueError(f"Review line {line_number}: image_id is empty.")
            if image_id in decisions:
                raise ValueError(
                    f"Review line {line_number}: duplicate image_id {image_id}."
                )
            if decision not in ALLOWED_DECISIONS:
                raise ValueError(
                    f"Review line {line_number}: unknown decision {decision!r}."
                )
            if not reason:
                raise ValueError(f"Review line {line_number}: reason is empty.")
            decisions[image_id] = {
                "decision": decision,
                "reason": reason,
            }
    return decisions


def output_split(row: dict[str, str]) -> str:
    """Move the complete C011 coin group to validation."""
    if row["coin_id"] in VALIDATION_COIN_IDS:
        return "val"
    return row["split"]


def validate_contract(
    manifest: list[dict[str, str]],
    review: dict[str, dict[str, str]],
    source_by_image_id: dict[str, Path],
) -> None:
    """Reject incomplete review, missing images, or split leakage before writing."""
    manifest_ids = {row["image_id"] for row in manifest}
    positive_ids = {
        row["image_id"] for row in manifest if row["defect_classes"].strip() == "dent"
    }
    unexpected_source_classes = {
        row["image_id"]
        for row in manifest
        if row["defect_classes"].strip() not in {"", "dent"}
    }
    if unexpected_source_classes:
        raise ValueError(
            "Source manifest contains non-dent classes: "
            + ", ".join(sorted(unexpected_source_classes))
        )
    if set(review) != positive_ids:
        missing = sorted(positive_ids - set(review))
        extra = sorted(set(review) - positive_ids)
        details = []
        if missing:
            details.append("missing positive reviews: " + ", ".join(missing))
        if extra:
            details.append("reviews for non-positive images: " + ", ".join(extra))
        raise ValueError("; ".join(details))

    missing_images = sorted(manifest_ids - set(source_by_image_id))
    extra_images = sorted(set(source_by_image_id) - manifest_ids)
    if missing_images or extra_images:
        details = []
        if missing_images:
            details.append("missing source images: " + ", ".join(missing_images))
        if extra_images:
            details.append("source images without manifest rows: " + ", ".join(extra_images))
        raise ValueError("; ".join(details))

    coin_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        split = output_split(row)
        if split not in SPLITS:
            raise ValueError(f"Invalid output split {split!r} for {row['image_id']}.")
        coin_splits[row["coin_id"]].add(split)
    leaked_coins = {
        coin_id: splits for coin_id, splits in coin_splits.items() if len(splits) > 1
    }
    if leaked_coins:
        details = ", ".join(
            f"{coin_id}={sorted(splits)}"
            for coin_id, splits in sorted(leaked_coins.items())
        )
        raise ValueError(f"Output split leakage: {details}")


def source_label_for(image_path: Path) -> Path:
    """Return the source label paired with an indexed source image."""
    split = image_path.parent.name
    return (
        image_path.parent.parent.parent
        / "labels"
        / split
        / f"{image_path.stem}.txt"
    )


def main() -> None:
    """Build the reviewed dataset and write its reproducibility metadata."""
    args = parse_args()
    columns, manifest = read_manifest(args.manifest)
    review = read_review(args.review)
    source_by_image_id = source_images(args.source_root)
    validate_contract(manifest, review, source_by_image_id)

    prepared_rows: list[dict[str, str]] = []
    prepared_labels: dict[str, str] = {}
    decision_counts: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    split_positive_counts: Counter[str] = Counter()
    split_coin_ids: dict[str, set[str]] = {split: set() for split in SPLITS}

    for source_row in manifest:
        row = dict(source_row)
        image_id = row["image_id"]
        source_label = source_label_for(source_by_image_id[image_id])
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing YOLO label file: {source_label}")

        source_classes = row["defect_classes"].strip()
        if source_classes == "dent":
            source_content = dent_label_content(source_label, image_id, source_classes)
            decision = review[image_id]["decision"]
            label_content = source_content if decision == "keep_rim" else ""
        else:
            label_content = dent_label_content(source_label, image_id, source_classes)
            decision = "source_normal"

        split = output_split(row)
        row["source_defect_classes"] = source_classes
        row["rim_review_decision"] = decision
        row["defect_classes"] = "rim_dent" if decision == "keep_rim" else ""
        row["split"] = split
        prepared_rows.append(row)
        prepared_labels[image_id] = label_content
        decision_counts[decision] += 1
        split_image_counts[split] += 1
        split_positive_counts[split] += int(bool(label_content))
        split_coin_ids[split].add(row["coin_id"])

    prepare_output(args.output_root, args.replace)
    for row in prepared_rows:
        image_id = row["image_id"]
        split = row["split"]
        source_image = source_by_image_id[image_id]
        source_label = source_label_for(source_image)
        output_image = args.output_root / "images" / split / source_image.name
        output_label = args.output_root / "labels" / split / source_label.name
        shutil.copy2(source_image, output_image)
        output_label.write_text(prepared_labels[image_id], encoding="utf-8")

    output_columns = columns + ["source_defect_classes", "rim_review_decision"]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(prepared_rows)

    report = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "review": str(args.review),
        "class_names": ["rim_dent"],
        "selected_images": len(prepared_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "split_image_counts": {
            split: split_image_counts[split] for split in SPLITS
        },
        "split_positive_counts": {
            split: split_positive_counts[split] for split in SPLITS
        },
        "split_negative_counts": {
            split: split_image_counts[split] - split_positive_counts[split]
            for split in SPLITS
        },
        "split_coin_ids": {
            split: sorted(split_coin_ids[split]) for split in SPLITS
        },
        "validation_coin_ids": sorted(VALIDATION_COIN_IDS),
        "sealed_test_coin_ids": ["C009"],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        "Rim dent v2 dataset built: "
        + ", ".join(
            f"{split}={split_image_counts[split]} "
            f"({split_positive_counts[split]} positive)"
            for split in SPLITS
        )
    )


if __name__ == "__main__":
    main()
