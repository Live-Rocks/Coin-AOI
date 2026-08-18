"""Train a reproducible CPU YOLO11n run on the v13 cloud-augmented dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from v13_pipeline import EXPECTED_SPLIT_COUNTS, validate_dataset


DATASET_CONFIG = Path("datasets/coin-defect-v13/data.yaml")
RUN_PROJECT = Path("runs/detect/v13").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="Starting model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--run-name", required=True, help="Unique directory name under runs/detect/v13.")
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG)
    parser.add_argument("--test-after-train", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset_config, EXPECTED_SPLIT_COUNTS)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    model = YOLO(args.model)
    model.train(
        data=str(args.dataset_config.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        seed=0,
        deterministic=True,
        cache=False,
        optimizer="auto",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        close_mosaic=0,
        project=str(RUN_PROJECT),
        name=args.run_name,
        exist_ok=False,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    for checkpoint in (best_weights, last_weights):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
    print(f"best checkpoint: {best_weights}")
    print(f"last checkpoint: {last_weights}")

    if args.test_after_train:
        YOLO(str(best_weights)).val(
            data=str(args.dataset_config.resolve()),
            split="test",
            imgsz=640,
            batch=2,
            device="cpu",
            workers=0,
            project=str(RUN_PROJECT),
            name=f"{args.run_name}-test",
            exist_ok=False,
        )


if __name__ == "__main__":
    main()
