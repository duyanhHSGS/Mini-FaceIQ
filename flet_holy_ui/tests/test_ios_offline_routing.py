"""iOS offline-routing smoke tests (Linux FakePage; not an IPA build)."""

from __future__ import annotations

import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flet as ft

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
try:
    app_module = importlib.import_module("main")
finally:
    sys.path.remove(str(SRC))


class FakePage:
    def __init__(self) -> None:
        self.width = 393
        self.height = 852
        self.platform = ft.PagePlatform.IOS
        self.window = SimpleNamespace()
        self.services: list = []
        self.controls: list = []
        self.navigation_bar = None
        self.scheduled_tasks: list = []

    def run_task(self, handler) -> None:
        self.scheduled_tasks.append(handler)

    def add(self, *controls) -> None:
        self.controls.extend(controls)

    def update(self, *controls) -> None:
        pass


def walk(control):
    if isinstance(control, ft.Control):
        yield control
        for name in ("content", "controls", "title"):
            item = getattr(control, name, None)
            if isinstance(item, ft.Control):
                yield from walk(item)
            elif isinstance(item, list):
                for child in item:
                    yield from walk(child)


def png_image() -> bytes:
    # Real PNG stream with an IHDR, enough for image dimensions and UI fallback.
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
    )


class IOSRoutingTests(unittest.TestCase):
    def test_native_front_offline_bridge_without_desktop_ml(self) -> None:
        class FakeFaceLandmarker:
            def __init__(self) -> None:
                self.calls = []

            async def detect(self, image: bytes, model: bytes) -> list[list[float]]:
                self.calls.append((image, model))
                return [[0.5, 0.5] for _ in range(478)]

        module = SimpleNamespace(FletFaceLandmarker=FakeFaceLandmarker)
        page = FakePage()
        with patch.dict(sys.modules, {"flet_face_landmarker": module}):
            app_module.main(page)

        self.assertEqual(len(page.services), 4)
        landmarker = page.services[0]
        picker = page.services[1]
        photo = png_image()
        picker.on_result(SimpleNamespace(files=[SimpleNamespace(
            name="face.png", bytes=photo
        )]))
        buttons = [
            c for c in walk(page.controls[0])
            if isinstance(c, ft.Button) and
            c.content == "Detect front · manually align side"
        ]
        self.assertEqual(len(buttons), 1)
        with patch.dict(sys.modules, {
            "mobile_front": SimpleNamespace(get_front_task_model=lambda: b"local-task")
        }), patch.object(app_module, "analyze_mobile_front_points", return_value={
            "mode": "front",
            "data": {"overallScore": 7.3},
            "landmarks": [],
            "landmark_count": 478,
        }):
            asyncio.run(buttons[0].on_click(None))

        self.assertEqual(landmarker.calls, [(photo, b"local-task")])
        self.assertTrue(any(
            isinstance(c, ft.Text) and "All loaded profiles calculated offline" in str(c.value)
            for c in walk(page.controls[0])
        ))

    def test_heic_is_converted_locally_before_scoring(self) -> None:
        class NativeService:
            async def convert_to_jpeg(self, photo):
                self.input = photo
                return png_image()

        page = FakePage()
        with patch.dict(sys.modules, {"flet_face_landmarker": SimpleNamespace(
            FletFaceLandmarker=NativeService
        )}):
            app_module.main(page)
        native = page.services[0]
        page.services[1].on_result(SimpleNamespace(files=[SimpleNamespace(
            name="iphone16.heic", bytes=b"original HEIC bytes"
        )]))
        self.assertEqual(len(page.scheduled_tasks), 1)
        asyncio.run(page.scheduled_tasks[0]())
        self.assertEqual(native.input, b"original HEIC bytes")
        self.assertTrue(any(
            isinstance(c, ft.Text) and "iphone16.jpg loaded" in str(c.value)
            for c in walk(page.controls[0])
        ))

    def test_ios_missing_native_plugin_is_manual_not_linux_fallback(self) -> None:
        page = FakePage()
        with patch.dict(sys.modules, {"flet_face_landmarker": None}):
            app_module.main(page)
        self.assertEqual(len(page.services), 3)
        page.services[0].on_result(SimpleNamespace(
            files=[SimpleNamespace(name="face.png", bytes=png_image())]
        ))
        nav = page.navigation_bar
        self.assertIsNotNone(nav)
        nav.on_change(SimpleNamespace(control=SimpleNamespace(selected_index=1)))
        buttons = [
            c for c in walk(page.controls[0])
            if isinstance(c, ft.Button) and c.content == "Detect front automatically"
        ]
        self.assertEqual(len(buttons), 1)
        with patch.object(app_module, "analyze_photo", side_effect=AssertionError(
            "Desktop inference must never run on iOS"
        )):
            asyncio.run(buttons[0].on_click(None))
        self.assertTrue(any(
            isinstance(c, ft.Text) and "extension is missing" in str(c.value)
            for c in walk(page.controls[0])
        ))


if __name__ == "__main__":
    unittest.main()
