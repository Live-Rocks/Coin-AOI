"""Run the published Coin-AOI Roboflow Workflow on one local image."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_WORKSPACE = "-dyayq"
DEFAULT_WORKFLOW = "coin-defect-hybrid-vcoin-defect-hybrid-0zuac-9-yolo11n-t2-logic"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RoboflowInferenceError(RuntimeError):
    """Base error for hosted Roboflow inference failures."""


class RoboflowHTTPError(RoboflowInferenceError):
    """Roboflow returned a non-success HTTP response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Roboflow returned HTTP {status_code}: {detail}")


class RoboflowConnectionError(RoboflowInferenceError):
    """The Roboflow endpoint could not be reached."""


class RoboflowResponseError(RoboflowInferenceError):
    """Roboflow returned an unexpected response shape."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local image through the published Roboflow v9 Workflow."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the local image to inspect.",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Roboflow workspace URL slug (default: {DEFAULT_WORKSPACE}).",
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help="Published Roboflow Workflow URL slug.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Roboflow inference host (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/roboflow-inference"),
        type=Path,
        help="Directory for the JSON result.",
    )
    parser.add_argument(
        "--timeout",
        default=120.0,
        type=float,
        help="Request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--retries",
        default=2,
        type=int,
        help="Retries for transient failures after the first request (default: 2).",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional original image URL to include in output metadata.",
    )
    parser.add_argument(
        "--expect-output",
        action="append",
        default=[],
        help="Required Workflow output key; repeat for multiple keys.",
    )
    parser.add_argument(
        "--expect-class",
        action="append",
        default=[],
        help="Required detected class; repeat for multiple classes.",
    )
    return parser.parse_args()


def encode_image(path: Path) -> dict[str, str]:
    image_bytes = path.read_bytes()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(image_bytes)} bytes; Roboflow serverless accepts at most "
            f"{MAX_IMAGE_BYTES} bytes."
        )
    return {
        "type": "base64",
        "value": base64.b64encode(image_bytes).decode("ascii"),
    }


def build_request(
    source: Path,
    api_key: str,
    api_url: str,
    workspace: str,
    workflow: str,
) -> urllib.request.Request:
    endpoint = (
        f"{api_url.rstrip('/')}/{workspace.strip('/')}/workflows/"
        f"{workflow.strip('/')}"
    )
    payload = {
        "api_key": api_key,
        "inputs": {"image": encode_image(source)},
    }
    return urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def create_ssl_context() -> ssl.SSLContext:
    """Use certifi when available without disabling certificate verification."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def run_workflow(
    request: urllib.request.Request,
    timeout: float = 120,
    retries: int = 2,
    backoff_seconds: float = 1,
) -> Any:
    """Execute a Workflow request with bounded retries for transient failures."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")
    if retries < 0:
        raise ValueError("retries cannot be negative.")

    ssl_context = create_ssl_context()
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
                context=ssl_context,
            ) as response:
                try:
                    return json.load(response)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise RoboflowResponseError(
                        "Roboflow returned a non-JSON response."
                    ) from error
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            detail = body[:1000] if body else str(error.reason)
            retryable = error.code in RETRYABLE_STATUS_CODES
            if retryable and attempt < retries:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise RoboflowHTTPError(error.code, detail) from error
        except urllib.error.URLError as error:
            if attempt < retries:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise RoboflowConnectionError(
                f"Could not reach Roboflow: {error.reason}"
            ) from error

    raise AssertionError("Workflow retry loop exited unexpectedly.")


def validate_response(result: Any, expected_items: int = 1) -> list[dict[str, Any]]:
    """Validate the documented one-result-per-input Workflow response contract."""
    if not isinstance(result, list):
        raise RoboflowResponseError(
            f"Expected a list response, received {type(result).__name__}."
        )
    if len(result) != expected_items:
        raise RoboflowResponseError(
            f"Expected {expected_items} result item(s), received {len(result)}."
        )
    if not all(isinstance(item, dict) for item in result):
        raise RoboflowResponseError("Every Workflow result item must be an object.")
    return result


def _artifact_extension(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def externalize_base64_outputs(
    value: Any,
    output_dir: Path,
    source_stem: str,
    path: tuple[str, ...] = (),
) -> Any:
    """Write base64-shaped Workflow outputs to files and return small references."""
    if (
        isinstance(value, dict)
        and value.get("type") == "base64"
        and isinstance(value.get("value"), str)
    ):
        try:
            content = base64.b64decode(value["value"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise RoboflowResponseError("Invalid base64 output from Roboflow.") from error
        safe_parts = [
            "".join(character if character.isalnum() else "_" for character in part)
            for part in path
        ]
        suffix = "_".join(part for part in safe_parts if part) or "output"
        artifact_path = output_dir / (
            f"workflow_{source_stem}_{suffix}{_artifact_extension(content)}"
        )
        artifact_path.write_bytes(content)
        return {"type": "file", "path": str(artifact_path)}
    if isinstance(value, dict):
        return {
            key: externalize_base64_outputs(
                child,
                output_dir,
                source_stem,
                path + (str(key),),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            externalize_base64_outputs(
                child,
                output_dir,
                source_stem,
                path + (str(index),),
            )
            for index, child in enumerate(value)
        ]
    return value


def detection_count(result: Any) -> int | None:
    """Find an object-detection list without assuming a Workflow output name."""
    if isinstance(result, dict):
        for value in result.values():
            count = detection_count(value)
            if count is not None:
                return count
    if isinstance(result, list):
        if result and all(
            isinstance(item, dict)
            and ("class" in item or "class_name" in item)
            and "confidence" in item
            for item in result
        ):
            return len(result)
        for value in result:
            count = detection_count(value)
            if count is not None:
                return count
    return None


def detected_classes(result: Any) -> set[str]:
    """Collect class names from object-detection-shaped dictionaries."""
    classes: set[str] = set()
    if isinstance(result, dict):
        class_name = result.get("class", result.get("class_name"))
        if isinstance(class_name, str) and "confidence" in result:
            classes.add(class_name)
        for value in result.values():
            classes.update(detected_classes(value))
    elif isinstance(result, list):
        for value in result:
            classes.update(detected_classes(value))
    return classes


def validate_acceptance(
    response: list[dict[str, Any]],
    expected_outputs: list[str],
    expected_classes: list[str],
) -> dict[str, Any]:
    """Return an explicit, serializable smoke-test acceptance result."""
    output_keys = sorted(response[0])
    classes = sorted(detected_classes(response))
    missing_outputs = sorted(set(expected_outputs) - set(output_keys))
    missing_classes = sorted(set(expected_classes) - set(classes))
    return {
        "passed": not missing_outputs and not missing_classes,
        "expected_outputs": expected_outputs,
        "observed_outputs": output_keys,
        "missing_outputs": missing_outputs,
        "expected_classes": expected_classes,
        "observed_classes": classes,
        "missing_classes": missing_classes,
    }


def source_metadata(source: Path, source_url: str | None) -> dict[str, Any]:
    """Record the immutable input identity without embedding image bytes."""
    return {
        "path": str(source.resolve()),
        "url": source_url,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_failure_evidence(
    output_dir: Path,
    source: Path,
    source_url: str | None,
    workspace: str,
    workflow: str,
    error: RoboflowInferenceError,
) -> Path:
    """Persist a secret-free record when hosted execution fails."""
    output_dir.mkdir(parents=True, exist_ok=True)
    error_details: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, RoboflowHTTPError):
        error_details["status_code"] = error.status_code
    evidence = {
        "requested_at": utc_timestamp(),
        "source": source_metadata(source, source_url),
        "workspace": workspace,
        "workflow": workflow,
        "status": "hosted_execution_failed",
        "error": error_details,
    }
    evidence_path = output_dir / f"failure_{source.stem}.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence_path


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Image not found: {args.source}")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero.")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative.")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RoboflowInferenceError(
            "ROBOFLOW_API_KEY is not set. Store it in your shell environment; "
            "do not commit it to Git."
        )

    request = build_request(
        source=args.source,
        api_key=api_key,
        api_url=args.api_url,
        workspace=args.workspace,
        workflow=args.workflow,
    )
    try:
        response = validate_response(
            run_workflow(
                request,
                timeout=args.timeout,
                retries=args.retries,
            )
        )
    except RoboflowInferenceError as error:
        evidence_path = write_failure_evidence(
            args.output_dir,
            args.source,
            args.source_url,
            args.workspace,
            args.workflow,
            error,
        )
        raise SystemExit(
            f"Roboflow inference failed: {error}\nFailure evidence: {evidence_path}"
        ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_response = externalize_base64_outputs(
        response,
        args.output_dir,
        args.source.stem,
    )
    output_path = args.output_dir / f"predictions_{args.source.stem}.json"
    acceptance = validate_acceptance(
        response,
        args.expect_output,
        args.expect_class,
    )
    output = {
        "requested_at": utc_timestamp(),
        "source": source_metadata(args.source, args.source_url),
        "workspace": args.workspace,
        "workflow": args.workflow,
        "acceptance": acceptance,
        "response": compact_response,
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    count = detection_count(compact_response)
    print(f"Detections: {count if count is not None else 'unknown response shape'}")
    print(f"Output keys: {', '.join(acceptance['observed_outputs'])}")
    print(f"Acceptance: {'passed' if acceptance['passed'] else 'failed'}")
    print(f"JSON result: {output_path}")
    if not acceptance["passed"]:
        raise SystemExit(
            "Workflow response did not satisfy the requested acceptance gate. "
            f"See {output_path}"
        )


if __name__ == "__main__":
    main()
