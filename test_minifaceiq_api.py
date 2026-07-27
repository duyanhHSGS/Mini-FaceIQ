"""
Mini-FaceIQ OpenAPI + Live API Test Runner

What it tests:
1. The OpenAPI YAML file can be parsed.
2. The OpenAPI document has the expected version and paths.
3. Optional strict OpenAPI validation, if openapi-spec-validator is installed.
4. The running Flask server exposes the expected endpoints.
5. Upload endpoints reject missing images correctly.
6. Metric endpoints reject missing landmarks correctly.
7. Optional real-image tests for automatic landmarks and features rating.
8. Optional real-payload tests for front and side metrics.

Usage examples:

    python test_minifaceiq_api.py --base-url http://127.0.0.1:7860 --front-image "examples\chicofront.jpg" --side-image "examples\chicoside.jpg" --front-landmarks-json "examples\front.json" --side-landmarks-json "examples\side.json" --gender male  --ethnicity caucasian  

Recommended dependencies:

    pip install requests pyyaml openapi-spec-validator

Only requests and PyYAML are required. Strict OpenAPI validation is skipped when
openapi-spec-validator is not installed.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: requests\n"
        "Install it with: pip install requests"
    ) from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: PyYAML\n"
        "Install it with: pip install pyyaml"
    ) from exc


EXPECTED_PATHS: dict[str, set[str]] = {
    "/": {"get"},
    "/api/front-landmarks": {"get"},
    "/api/side-landmarks": {"get"},
    "/api/front-autolandmarks": {"post"},
    "/api/side-autolandmarks": {"post"},
    "/api/features-rating": {"post"},
    "/api/front-metrics": {"post"},
    "/api/side-metrics": {"post"},
}


@dataclass(slots=True)
class TestResult:
    name: str
    passed: bool
    details: str = ""
    skipped: bool = False


class TestRunner:
    def __init__(self) -> None:
        self.results: list[TestResult] = []

    def run(self, name: str, test: Callable[[], str | None]) -> None:
        try:
            details = test() or ""
            self.results.append(TestResult(name=name, passed=True, details=details))
            print(f"[PASS] {name}")
            if details:
                print(f"       {details}")
        except SkipTest as exc:
            self.results.append(
                TestResult(name=name, passed=False, skipped=True, details=str(exc))
            )
            print(f"[SKIP] {name}")
            if str(exc):
                print(f"       {exc}")
        except Exception as exc:
            self.results.append(
                TestResult(name=name, passed=False, details=f"{type(exc).__name__}: {exc}")
            )
            print(f"[FAIL] {name}")
            print(f"       {type(exc).__name__}: {exc}")

    def summary(self) -> int:
        passed = sum(result.passed for result in self.results)
        skipped = sum(result.skipped for result in self.results)
        failed = len(self.results) - passed - skipped

        print("\n" + "=" * 68)
        print("Mini-FaceIQ test summary")
        print("=" * 68)
        print(f"Passed : {passed}")
        print(f"Failed : {failed}")
        print(f"Skipped: {skipped}")
        print(f"Total  : {len(self.results)}")

        if failed:
            print("\nFailed tests:")
            for result in self.results:
                if not result.passed and not result.skipped:
                    print(f"  - {result.name}: {result.details}")

        return 1 if failed else 0


class SkipTest(Exception):
    """Raised when a test cannot run because optional input is absent."""


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    assert_true(path.exists(), f"OpenAPI file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)

    assert_true(isinstance(document, dict), "OpenAPI root must be a YAML object")
    return document


def decode_json_response(response: requests.Response) -> dict[str, Any]:
    content_type = response.headers.get("Content-Type", "")
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        preview = response.text[:500].replace("\n", "\\n")
        raise AssertionError(
            f"Expected JSON but received {content_type or 'unknown content type'}: "
            f"{preview!r}"
        ) from exc

    assert_true(isinstance(payload, dict), "JSON response must be an object")
    return payload


def assert_success_envelope(payload: dict[str, Any]) -> None:
    assert_true(payload.get("success") is True, f"Expected success=true, got: {payload}")


def assert_error_envelope(payload: dict[str, Any]) -> None:
    assert_true(payload.get("success") is False, f"Expected success=false, got: {payload}")
    assert_true(
        isinstance(payload.get("error"), str) and payload["error"].strip(),
        f"Expected a non-empty error string, got: {payload}",
    )


def request(
    method: str,
    url: str,
    *,
    timeout: float,
    **kwargs: Any,
) -> requests.Response:
    try:
        return requests.request(method, url, timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise AssertionError(
            f"Could not connect to {url}. Is Flask running and listening on the "
            "correct host? For phone/LAN access, use host='0.0.0.0'."
        ) from exc
    except requests.Timeout as exc:
        raise AssertionError(f"Request timed out after {timeout} seconds: {url}") from exc


def load_json_file(path: Path) -> Any:
    assert_true(path.exists(), f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_landmarks(payload: Any) -> list[dict[str, Any]]:
    """
    Accept either:
      - a raw landmark array
      - {"landmarks": [...]}
      - {"success": true, "landmarks": [...]}
    """
    if isinstance(payload, list):
        landmarks = payload
    elif isinstance(payload, dict):
        landmarks = payload.get("landmarks")
    else:
        landmarks = None

    assert_true(isinstance(landmarks, list), "Could not find a landmark array")
    assert_true(
        all(isinstance(item, dict) for item in landmarks),
        "Every landmark must be a JSON object",
    )
    return landmarks


def open_image_file(path: Path):
    assert_true(path.exists(), f"Image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.open("rb"), mime_type


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Mini-FaceIQ OpenAPI and test the live Flask API."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("openapi.yaml"),
        help="Path to the OpenAPI YAML file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:7860",
        help="Base URL of the running Flask server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument("--front-image", type=Path)
    parser.add_argument("--side-image", type=Path)
    parser.add_argument("--features-image", type=Path)
    parser.add_argument("--front-landmarks-json", type=Path)
    parser.add_argument("--side-landmarks-json", type=Path)
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--ethnicity", default="asian")
    parser.add_argument("--front-aspect", type=float, default=1.0)
    parser.add_argument("--side-aspect", type=float, default=1.0)
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Print full tracebacks for unexpected top-level failures.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    runner = TestRunner()
    document_holder: dict[str, dict[str, Any]] = {}

    def test_parse_spec() -> str:
        document = load_yaml(args.spec)
        document_holder["document"] = document
        return f"Loaded {args.spec}"

    runner.run("Parse OpenAPI YAML", test_parse_spec)

    def test_openapi_version() -> str:
        document = document_holder.get("document") or load_yaml(args.spec)
        version = document.get("openapi")
        assert_true(
            isinstance(version, str) and version.startswith("3.0."),
            f"Expected OpenAPI 3.0.x, got {version!r}",
        )
        return f"OpenAPI version {version}"

    runner.run("Check OpenAPI version", test_openapi_version)

    def test_expected_paths() -> str:
        document = document_holder.get("document") or load_yaml(args.spec)
        paths = document.get("paths")
        assert_true(isinstance(paths, dict), "OpenAPI document has no paths object")

        errors: list[str] = []
        for path, methods in EXPECTED_PATHS.items():
            path_item = paths.get(path)
            if not isinstance(path_item, dict):
                errors.append(f"missing path {path}")
                continue

            existing_methods = {key.lower() for key in path_item}
            for method in methods:
                if method not in existing_methods:
                    errors.append(f"missing {method.upper()} {path}")

        assert_true(not errors, "; ".join(errors))
        return f"Found all {len(EXPECTED_PATHS)} expected paths"

    runner.run("Check required API paths", test_expected_paths)

    def test_unique_operation_ids() -> str:
        document = document_holder.get("document") or load_yaml(args.spec)
        paths = document.get("paths", {})
        operation_ids: dict[str, str] = {}

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            for method, operation in path_item.items():
                if method.lower() not in {
                    "get", "post", "put", "patch", "delete", "head", "options", "trace"
                }:
                    continue
                if not isinstance(operation, dict):
                    continue

                operation_id = operation.get("operationId")
                if not operation_id:
                    continue

                location = f"{method.upper()} {path}"
                if operation_id in operation_ids:
                    raise AssertionError(
                        f"Duplicate operationId {operation_id!r}: "
                        f"{operation_ids[operation_id]} and {location}"
                    )
                operation_ids[operation_id] = location

        return f"Checked {len(operation_ids)} operationId values"

    runner.run("Check operationId uniqueness", test_unique_operation_ids)

    def test_strict_openapi_validation() -> str:
        try:
            from openapi_spec_validator import validate
        except ImportError as exc:
            raise SkipTest(
                "Install optional validator with: pip install openapi-spec-validator"
            ) from exc

        document = document_holder.get("document") or load_yaml(args.spec)
        validate(document)
        return "Strict OpenAPI validation passed"

    runner.run("Strict OpenAPI validation", test_strict_openapi_validation)

    def test_root() -> str:
        response = request("GET", f"{base_url}/", timeout=args.timeout)
        assert_true(response.status_code == 200, f"Expected 200, got {response.status_code}")
        content_type = response.headers.get("Content-Type", "")
        assert_true(
            "text/html" in content_type,
            f"Expected text/html, got {content_type or 'no Content-Type'}",
        )
        return f"HTTP {response.status_code}, {content_type}"

    runner.run("GET /", test_root)

    landmark_definitions: dict[str, list[dict[str, Any]]] = {}

    def test_landmark_definitions(kind: str) -> Callable[[], str]:
        def inner() -> str:
            response = request(
                "GET",
                f"{base_url}/api/{kind}-landmarks",
                timeout=args.timeout,
            )
            assert_true(
                response.status_code == 200,
                f"Expected 200, got {response.status_code}",
            )
            payload = decode_json_response(response)
            assert_success_envelope(payload)
            landmarks = payload.get("landmarks")
            assert_true(isinstance(landmarks, list), "landmarks must be an array")
            assert_true(len(landmarks) > 0, "landmarks array must not be empty")
            assert_true(
                all(isinstance(item, dict) and "id" in item for item in landmarks),
                "Every landmark definition must be an object containing id",
            )
            landmark_definitions[kind] = landmarks

            id_types = sorted({type(item["id"]).__name__ for item in landmarks})
            return f"{len(landmarks)} definitions; ID type(s): {', '.join(id_types)}"

        return inner

    runner.run(
        "GET /api/front-landmarks",
        test_landmark_definitions("front"),
    )
    runner.run(
        "GET /api/side-landmarks",
        test_landmark_definitions("side"),
    )

    def test_upload_without_image(endpoint: str) -> Callable[[], str]:
        def inner() -> str:
            response = request(
                "POST",
                f"{base_url}{endpoint}",
                timeout=args.timeout,
                files={},
            )
            assert_true(
                response.status_code == 400,
                f"Expected 400, got {response.status_code}",
            )
            payload = decode_json_response(response)
            assert_error_envelope(payload)
            assert_true(
                payload["error"] == "No image file uploaded",
                f"Unexpected error message: {payload['error']!r}",
            )
            return payload["error"]

        return inner

    for endpoint in (
        "/api/front-autolandmarks",
        "/api/side-autolandmarks",
        "/api/features-rating",
    ):
        runner.run(
            f"POST {endpoint} without image",
            test_upload_without_image(endpoint),
        )

    def test_invalid_extension(endpoint: str) -> Callable[[], str]:
        def inner() -> str:
            response = request(
                "POST",
                f"{base_url}{endpoint}",
                timeout=args.timeout,
                files={
                    "image": (
                        "not_an_image.txt",
                        b"Mini-FaceIQ test payload",
                        "text/plain",
                    )
                },
            )
            assert_true(
                response.status_code == 400,
                f"Expected 400, got {response.status_code}",
            )
            payload = decode_json_response(response)
            assert_error_envelope(payload)
            assert_true(
                payload["error"] == "Use jpg, jpeg, png, or webp",
                f"Unexpected error message: {payload['error']!r}",
            )
            return payload["error"]

        return inner

    for endpoint in (
        "/api/front-autolandmarks",
        "/api/side-autolandmarks",
        "/api/features-rating",
    ):
        runner.run(
            f"POST {endpoint} with invalid extension",
            test_invalid_extension(endpoint),
        )

    def test_missing_metrics(kind: str) -> Callable[[], str]:
        def inner() -> str:
            aspect_key = "frontAspect" if kind == "front" else "sideAspect"
            response = request(
                "POST",
                f"{base_url}/api/{kind}-metrics",
                timeout=args.timeout,
                json={
                    "landmarks": [],
                    "gender": args.gender,
                    "ethnicity": args.ethnicity,
                    aspect_key: 1.0,
                },
            )
            assert_true(
                response.status_code == 400,
                f"Expected 400, got {response.status_code}",
            )
            payload = decode_json_response(response)
            assert_error_envelope(payload)
            missing = payload.get("missing")
            assert_true(isinstance(missing, list), "Expected missing to be an array")

            expected_defs = landmark_definitions.get(kind)
            if expected_defs is not None:
                expected_ids = {item["id"] for item in expected_defs}
                assert_true(
                    set(missing) == expected_ids,
                    f"Missing IDs differ from /api/{kind}-landmarks definitions",
                )

            return f"Server reported {len(missing)} missing landmarks"

        return inner

    runner.run("POST /api/front-metrics with no landmarks", test_missing_metrics("front"))
    runner.run("POST /api/side-metrics with no landmarks", test_missing_metrics("side"))

    def test_real_image(endpoint: str, image_path: Path | None) -> Callable[[], str]:
        def inner() -> str:
            if image_path is None:
                raise SkipTest("No image path supplied")

            file_handle, mime_type = open_image_file(image_path)
            try:
                response = request(
                    "POST",
                    f"{base_url}{endpoint}",
                    timeout=args.timeout,
                    files={"image": (image_path.name, file_handle, mime_type)},
                )
            finally:
                file_handle.close()

            payload = decode_json_response(response)
            assert_true(
                response.status_code == 200,
                f"Expected 200, got {response.status_code}: {payload}",
            )
            assert_success_envelope(payload)

            if endpoint.endswith("autolandmarks"):
                landmarks = payload.get("landmarks")
                assert_true(isinstance(landmarks, list), "landmarks must be an array")
                assert_true(len(landmarks) > 0, "landmarks array must not be empty")
                return f"Detected {len(landmarks)} landmarks"

            data = payload.get("data")
            assert_true(isinstance(data, dict), "features data must be an object")
            return f"Returned data keys: {', '.join(sorted(data)[:12]) or '(none)'}"

        return inner

    runner.run(
        "Real front autolandmark detection",
        test_real_image("/api/front-autolandmarks", args.front_image),
    )
    runner.run(
        "Real side autolandmark detection",
        test_real_image("/api/side-autolandmarks", args.side_image),
    )
    runner.run(
        "Real features-rating inference",
        test_real_image(
            "/api/features-rating",
            args.features_image or args.front_image,
        ),
    )

    def test_real_metrics(
        kind: str,
        json_path: Path | None,
    ) -> Callable[[], str]:
        def inner() -> str:
            if json_path is None:
                raise SkipTest("No landmark JSON path supplied")

            raw = load_json_file(json_path)
            landmarks = extract_landmarks(raw)
            aspect_key = "frontAspect" if kind == "front" else "sideAspect"
            aspect_value = args.front_aspect if kind == "front" else args.side_aspect

            response = request(
                "POST",
                f"{base_url}/api/{kind}-metrics",
                timeout=args.timeout,
                json={
                    "landmarks": landmarks,
                    "gender": args.gender,
                    "ethnicity": args.ethnicity,
                    aspect_key: aspect_value,
                },
            )
            payload = decode_json_response(response)
            assert_true(
                response.status_code == 200,
                f"Expected 200, got {response.status_code}: {payload}",
            )
            assert_success_envelope(payload)
            data = payload.get("data")
            assert_true(isinstance(data, dict), "data must be an object")
            return f"Returned data keys: {', '.join(sorted(data)[:15]) or '(none)'}"

        return inner

    runner.run(
        "Real front metric calculation",
        test_real_metrics("front", args.front_landmarks_json),
    )
    runner.run(
        "Real side metric calculation",
        test_real_metrics("side", args.side_landmarks_json),
    )

    return runner.summary()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
    except Exception:
        print("\nUnexpected fatal error:")
        traceback.print_exc()
        raise SystemExit(2)
