"""Run a reproducible CPU YOLO baseline on a validated local dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from validate_dataset import main as validate_dataset


DATASET_CONFIG = Path("datasets/coin-defect-hybrid/data.yaml")
MODEL_NAME = "yolo11n.pt"
RUN_PROJECT = "baseline"
DEFAULT_EPOCHS = 5
DEFAULT_IMGSZ = 640


def parse_args() -> argparse.Namespace:
    """Parse the experiment settings while preserving the five-epoch default."""
    parser = argparse.ArgumentParser(
        description="Train and test a reproducible CPU YOLO baseline."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DATASET_CONFIG,
        help=f"YOLO dataset YAML path (default: {DATASET_CONFIG}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Manifest paired with the selected dataset.",
    )
    parser.add_argument(
        "--class-names",
        default="dent,scratch,stain_corrosion",
        help="Comma-separated class names in YOLO ID order.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Square YOLO input size in pixels (default: {DEFAULT_IMGSZ}).",
    )
    parser.add_argument(
        "--run-name",
        help="Optional output directory name under runs/detect/baseline/.",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="YOLO mosaic augmentation probability from 0.0 to 1.0 (default: 1.0).",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32 and at least 32.")
    if not 0.0 <= args.mosaic <= 1.0:
        parser.error("--mosaic must be between 0.0 and 1.0.")
    return args


def main() -> None:
    """Validate the dataset, then run a reproducible CPU baseline."""
    args = parse_args()
    validate_dataset(
        [
            "--dataset-root",
            str(args.dataset_config.parent),
            "--manifest",
            str(args.manifest),
            "--class-names",
            args.class_names,
        ]
    )

    if not args.dataset_config.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {args.dataset_config}"
        )

    run_name = args.run_name or f"yolo11n-cpu-e{args.epochs}-s0-i{args.imgsz}"
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(args.dataset_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=2,
        device="cpu",
        seed=0,
        deterministic=True,
        workers=0,
        cache=False,
        mosaic=args.mosaic,
        project=RUN_PROJECT,
        name=run_name,
        exist_ok=False,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {best_weights}")

    YOLO(str(best_weights)).val(
        data=str(args.dataset_config),
        split="test",
        imgsz=args.imgsz,
        batch=2,
        device="cpu",
        workers=0,
        project=RUN_PROJECT,
        name=f"{run_name}-test",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
