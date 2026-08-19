"""Image validation and single-image YOLO inference for the web demo."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT
    / "runs"
    / "detect"
    / "v13"
    / "v13-yolo11n-cpu-e100-i640-s0"
    / "weights"
    / "best.pt"
)


class InvalidImageError(ValueError):
    """Raised when an uploaded payload cannot be safely decoded as an image."""


class ModelConfigurationError(RuntimeError):
    """Raised when the configured YOLO checkpoint is unavailable."""


@dataclass(frozen=True)
class Prediction:
    """JSON-friendly result produced by one YOLO prediction."""

    model: str
    confidence_threshold: float
    image_shape: list[int]
    detections: list[dict[str, float | int | str | list[float]]]
    annotated_image_data_url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "confidence_threshold": self.confidence_threshold,
            "image_shape": self.image_shape,
            "detection_count": len(self.detections),
            "detections": self.detections,
            "annotated_image_data_url": self.annotated_image_data_url,
        }


def serialize_detections(result: Any) -> list[dict[str, float | int | str | list[float]]]:
    """Convert Ultralytics result boxes into the project's existing JSON schema."""
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


def decode_image(contents: bytes) -> Image.Image:
    """Decode an upload after Pillow has verified its image structure."""
    try:
        with Image.open(BytesIO(contents)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(contents)) as candidate:
            return candidate.convert("RGB")
    except (ImportError, UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("The uploaded file is not a valid image.") from error


class YoloService:
    """Lazily load one checkpoint and run inference without persisting uploads."""

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
        model_loader: Callable[[str], Any] = YOLO,
    ) -> None:
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self.confidence = confidence
        self.image_size = image_size
        self.device = device if device is not None else os.getenv("YOLO_DEVICE", "cpu")
        self._model_loader = model_loader
        self._model: Any | None = None
        self._model_lock = Lock()
        self._prediction_lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise ModelConfigurationError(
                        "YOLO checkpoint not found. Set MODEL_PATH to a mounted best.pt file."
                    )
                self._model = self._model_loader(str(self.model_path))
        return self._model

    def ensure_ready(self) -> None:
        """Load the configured checkpoint once so readiness reflects model access."""
        self._get_model()

    def predict_bytes(self, contents: bytes) -> Prediction:
        """Run YOLO against image bytes and encode the annotated result for JSON."""
        image = decode_image(contents)
        model = self._get_model()
        with self._prediction_lock:
            result = model.predict(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                save=False,
                verbose=False,
            )[0]
        annotated_bgr = result.plot()
        annotated_rgb = Image.fromarray(annotated_bgr[..., ::-1])
        output = BytesIO()
        annotated_rgb.save(output, format="JPEG", quality=90)
        annotated_image_data_url = (
            "data:image/jpeg;base64,"
            + base64.b64encode(output.getvalue()).decode("ascii")
        )

        return Prediction(
            model=self.model_path.name,
            confidence_threshold=self.confidence,
            image_shape=list(result.orig_shape),
            detections=serialize_detections(result),
            annotated_image_data_url=annotated_image_data_url,
        )
