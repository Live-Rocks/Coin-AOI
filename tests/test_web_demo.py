"""Tests for the portfolio web demo without loading a real YOLO checkpoint."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app  # noqa: E402
from app.yolo_service import ModelConfigurationError, Prediction, YoloService  # noqa: E402


def image_bytes() -> bytes:
    """Make a small valid JPEG payload for upload tests."""
    image = Image.new("RGB", (12, 8), color=(100, 120, 140))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class FakeService:
    def __init__(self, readiness_error: Exception | None = None) -> None:
        self.calls: list[bytes] = []
        self.readiness_calls = 0
        self.readiness_error = readiness_error

    def ensure_ready(self) -> None:
        self.readiness_calls += 1
        if self.readiness_error is not None:
            raise self.readiness_error

    def predict_bytes(self, contents: bytes) -> Prediction:
        self.calls.append(contents)
        return Prediction(
            model="best.pt",
            confidence_threshold=0.25,
            image_shape=[8, 12],
            detections=[
                {
                    "class_id": 0,
                    "class_name": "dent",
                    "confidence": 0.92,
                    "xyxy": [1.0, 1.0, 8.0, 6.0],
                }
            ],
            annotated_image_data_url="data:image/jpeg;base64,ZmFrZQ==",
        )


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Coordinates:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeBox:
    cls = Scalar(0)
    conf = Scalar(0.92)
    xyxy = [Coordinates([1.0, 1.0, 8.0, 6.0])]


class FakeResult:
    boxes = [FakeBox()]
    names = {0: "dent"}
    orig_shape = (8, 12)

    def plot(self) -> np.ndarray:
        return np.zeros((8, 12, 3), dtype=np.uint8)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [FakeResult()]


class BlockingModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self.active_lock = threading.Lock()

    def predict(self, **kwargs: object) -> list[FakeResult]:
        with self.active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        self.release.wait(timeout=2)
        with self.active_lock:
            self.active -= 1
        return super().predict(**kwargs)


class WebDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeService()
        self.client = TestClient(create_app(service=self.service))

    def test_home_and_health_check_are_available(self) -> None:
        home = self.client.get("/")
        health = self.client.get("/healthz")

        self.assertEqual(home.status_code, 200)
        self.assertIn("Coin-AOI", home.text)
        self.assertEqual(health.json(), {"status": "ok"})

    def test_readiness_reports_model_configuration_errors(self) -> None:
        self.assertEqual(self.client.get("/readyz").json(), {"status": "ready"})
        self.assertEqual(self.service.readiness_calls, 1)

        unavailable = TestClient(
            create_app(
                service=FakeService(
                    ModelConfigurationError("YOLO checkpoint not found.")
                )
            )
        )
        self.assertEqual(unavailable.get("/healthz").status_code, 200)
        ready = unavailable.get("/readyz")
        self.assertEqual(ready.status_code, 503)
        self.assertIn("checkpoint not found", ready.json()["detail"])

    def test_upload_returns_serialized_prediction(self) -> None:
        response = self.client.post(
            "/api/predict",
            files={"image": ("coin.jpg", image_bytes(), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detection_count"], 1)
        self.assertEqual(response.json()["detections"][0]["class_name"], "dent")
        self.assertEqual(len(self.service.calls), 1)

    def test_bundled_example_uses_the_same_prediction_service(self) -> None:
        response = self.client.post("/api/predict-example")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "best.pt")
        self.assertEqual(len(self.service.calls), 1)

    def test_rejects_unsupported_or_invalid_uploads(self) -> None:
        unsupported = self.client.post(
            "/api/predict",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(unsupported.status_code, 415)

        invalid_image_client = TestClient(create_app())
        invalid = invalid_image_client.post(
            "/api/predict",
            files={"image": ("broken.jpg", b"not a JPEG", "image/jpeg")},
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("not a valid image", invalid.json()["detail"])

    def test_rejects_files_larger_than_limit(self) -> None:
        response = self.client.post(
            "/api/predict",
            files={
                "image": (
                    "large.jpg",
                    b"x" * (10 * 1024 * 1024 + 1),
                    "image/jpeg",
                )
            },
        )

        self.assertEqual(response.status_code, 413)

    def test_prediction_rate_limit_and_busy_slot_reject_excess_requests(self) -> None:
        limited_service = FakeService()
        limited_client = TestClient(create_app(service=limited_service))
        responses = [limited_client.post("/api/predict-example") for _ in range(6)]
        self.assertEqual([response.status_code for response in responses[:5]], [200] * 5)
        self.assertEqual(responses[5].status_code, 429)

        busy_app = create_app(service=FakeService())
        busy_app.state.inference_slot.acquire()
        try:
            busy = TestClient(busy_app).post("/api/predict-example")
        finally:
            busy_app.state.inference_slot.release()
        self.assertEqual(busy.status_code, 429)
        self.assertIn("busy", busy.json()["detail"])

    def test_yolo_service_serializes_a_mock_model_once(self) -> None:
        fake_model = FakeModel()
        loader_calls: list[str] = []

        def loader(path: str) -> FakeModel:
            loader_calls.append(path)
            return fake_model

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.touch()
            service = YoloService(model_path=checkpoint, model_loader=loader)
            first = service.predict_bytes(image_bytes())
            second = service.predict_bytes(image_bytes())

        self.assertEqual(loader_calls, [str(checkpoint)])
        self.assertEqual(first.detections[0]["class_name"], "dent")
        self.assertTrue(first.annotated_image_data_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(second.image_shape, [8, 12])
        self.assertEqual(len(fake_model.calls), 2)

    def test_yolo_service_serializes_concurrent_model_predictions(self) -> None:
        blocking_model = BlockingModel()
        results: list[Prediction] = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.touch()
            service = YoloService(
                model_path=checkpoint, model_loader=lambda _path: blocking_model
            )
            first = threading.Thread(
                target=lambda: results.append(service.predict_bytes(image_bytes()))
            )
            second = threading.Thread(
                target=lambda: results.append(service.predict_bytes(image_bytes()))
            )
            first.start()
            self.assertTrue(blocking_model.entered.wait(timeout=1))
            second.start()
            time.sleep(0.05)
            self.assertEqual(blocking_model.max_active, 1)
            blocking_model.release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(blocking_model.max_active, 1)

    def test_bundled_example_is_a_small_webp(self) -> None:
        example = Path(__file__).resolve().parents[1] / "app/static/examples/coin-dent.webp"
        with Image.open(example) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertEqual(image.size, (768, 1024))
        self.assertLess(example.stat().st_size, 500 * 1024)


if __name__ == "__main__":
    unittest.main()
