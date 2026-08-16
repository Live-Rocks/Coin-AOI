"""Run one reproducible YOLO fine-tuning smoke test."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from validate_dataset import main as validate_dataset


DATASET_CONFIG = Path("datasets/coin-defect-hybrid/data.yaml")
MODEL_NAME = "yolo11n.pt"
RUN_PROJECT = "runs/smoke"
RUN_NAME = "yolo11n-mps-s0"


def main() -> None:
    """Validate the local dataset, then complete one MPS training epoch."""
    validate_dataset()

    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG}")

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(DATASET_CONFIG),
        epochs=1,
        imgsz=320,
        batch=2,
        device="mps",
        seed=0,
        deterministic=True,
        workers=0,
        cache=False,
        project=RUN_PROJECT,
        name=RUN_NAME,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
