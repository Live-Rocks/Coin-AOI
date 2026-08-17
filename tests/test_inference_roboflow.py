"""Tests for the dependency-free Roboflow Workflow client."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from src.inference_roboflow import (
    RoboflowHTTPError,
    RoboflowResponseError,
    build_request,
    detection_count,
    externalize_base64_outputs,
    run_workflow,
    source_metadata,
    validate_acceptance,
    validate_response,
)


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class RoboflowInferenceTests(unittest.TestCase):
    def test_build_request_keeps_api_key_out_of_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image = Path(temporary_directory) / "coin.jpg"
            image.write_bytes(b"coin-image")

            request = build_request(
                image,
                "private-key",
                "https://serverless.roboflow.com",
                "-dyayq",
                "coin-workflow",
            )
            payload = json.loads(request.data)

        self.assertNotIn("private-key", request.full_url)
        self.assertEqual(payload["api_key"], "private-key")
        self.assertEqual(payload["inputs"]["image"]["type"], "base64")

    @mock.patch("src.inference_roboflow.time.sleep")
    @mock.patch("src.inference_roboflow.create_ssl_context")
    @mock.patch("src.inference_roboflow.urllib.request.urlopen")
    def test_run_workflow_retries_transient_http_failure(
        self,
        urlopen: mock.Mock,
        create_ssl_context: mock.Mock,
        sleep: mock.Mock,
    ) -> None:
        error = urllib.error.HTTPError(
            "https://example.test",
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b"temporary failure"),
        )
        urlopen.side_effect = [
            error,
            FakeResponse(b'[{"dynamic_output": []}]'),
        ]
        request = mock.Mock()

        result = run_workflow(request, retries=1, backoff_seconds=0)

        self.assertEqual(result, [{"dynamic_output": []}])
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0)
        create_ssl_context.assert_called_once()

    @mock.patch("src.inference_roboflow.create_ssl_context")
    @mock.patch("src.inference_roboflow.urllib.request.urlopen")
    def test_run_workflow_raises_typed_non_retryable_error(
        self,
        urlopen: mock.Mock,
        create_ssl_context: mock.Mock,
    ) -> None:
        urlopen.side_effect = urllib.error.HTTPError(
            "https://example.test",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"invalid key"),
        )

        with self.assertRaises(RoboflowHTTPError) as raised:
            run_workflow(mock.Mock(), retries=2)

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(urlopen.call_count, 1)
        create_ssl_context.assert_called_once()

    def test_validate_response_uses_documented_list_shape(self) -> None:
        response = [{"first_output": [], "second_output": {"value": 1}}]
        self.assertEqual(validate_response(response), response)
        with self.assertRaises(RoboflowResponseError):
            validate_response({"first_output": []})

    def test_externalize_base64_output_without_known_output_name(self) -> None:
        png = b"\x89PNG\r\n\x1a\nimage-content"
        response = [
            {
                "arbitrary_visualization": {
                    "type": "base64",
                    "value": base64.b64encode(png).decode("ascii"),
                }
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            compact = externalize_base64_outputs(response, output_dir, "coin")
            artifact = Path(compact[0]["arbitrary_visualization"]["path"])

            self.assertEqual(artifact.suffix, ".png")
            self.assertEqual(artifact.read_bytes(), png)
            self.assertNotIn("value", compact[0]["arbitrary_visualization"])

    def test_detection_count_does_not_require_output_name(self) -> None:
        response = [
            {
                "anything": [
                    {"class": "scratch", "confidence": 0.91},
                    {"class": "dent", "confidence": 0.74},
                ]
            }
        ]
        self.assertEqual(detection_count(response), 2)

    def test_acceptance_gate_reports_missing_outputs_and_classes(self) -> None:
        response = [
            {
                "predictions": [
                    {"class": "scratch", "confidence": 0.91},
                ]
            }
        ]

        passed = validate_acceptance(response, ["predictions"], ["scratch"])
        failed = validate_acceptance(response, ["visualization"], ["dent"])

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["missing_outputs"], ["visualization"])
        self.assertEqual(failed["missing_classes"], ["dent"])

    def test_source_metadata_has_reproducible_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "coin.png"
            source.write_bytes(b"fixed-coin")

            metadata = source_metadata(source, "https://example.test/coin.png")

        self.assertEqual(
            metadata["sha256"],
            "6984d09f98092e9274c7422120d150448702445a476a39b250f7ca645abc5e64",
        )
        self.assertEqual(metadata["size_bytes"], 10)
        self.assertEqual(metadata["url"], "https://example.test/coin.png")


@unittest.skipUnless(
    os.environ.get("RUN_ROBOFLOW_LIVE") == "1",
    "set RUN_ROBOFLOW_LIVE=1 to use Roboflow credits",
)
class RoboflowLiveSmokeTests(unittest.TestCase):
    def test_c005_03_scratch_workflow(self) -> None:
        source = Path("data/local/roboflow-smoke/C005_03.png")
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        self.assertTrue(api_key, "ROBOFLOW_API_KEY must be set")
        self.assertTrue(source.is_file(), f"smoke image missing: {source}")
        self.assertEqual(
            source_metadata(source, None)["sha256"],
            "d1ac6fdf0170266bf02bf89b91176e9c64cee415eb303e397e4857167a34aec0",
            "C005_03 smoke input changed",
        )

        request = build_request(
            source,
            api_key,
            "https://serverless.roboflow.com",
            "-dyayq",
            "coin-defect-hybrid-vcoin-defect-hybrid-0zuac-9-yolo11n-t2-logic",
        )
        response = validate_response(run_workflow(request))
        acceptance = validate_acceptance(response, ["predictions"], ["scratch"])

        self.assertTrue(acceptance["passed"], acceptance)


if __name__ == "__main__":
    unittest.main()
