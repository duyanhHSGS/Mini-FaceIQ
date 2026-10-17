"""Smoke-test the Android Material 3 landmark editor without launching Flutter."""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
try:
    app_main = importlib.import_module("main").__dict__["main"]
finally:
    sys.path.remove(str(SRC))


class FakePage:
    def __init__(self) -> None:
        self.width = 393
        self.height = 830
        self.platform = ft.PagePlatform.ANDROID
        self.window = SimpleNamespace()
        self.services: list = []
        self.controls: list = []
        self.navigation_bar = None
        self.update_count = 0

    def add(self, *controls) -> None:
        self.controls.extend(controls)

    def update(self, *controls) -> None:
        self.update_count += 1


def walk(control):
    if isinstance(control, ft.Control):
        yield control
        for key in ("content", "controls", "title"):
            value = getattr(control, key, None)
            if isinstance(value, ft.Control):
                yield from walk(value)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(child)


class AndroidManualUITests(unittest.TestCase):
    def test_photo_then_editor_tap(self):
        page = FakePage()
        app_main(page)
        self.assertEqual(len(page.services), 3)
        nav = page.navigation_bar
        self.assertIsInstance(nav, ft.NavigationBar)
        assert isinstance(nav, ft.NavigationBar)
        self.assertEqual(nav.selected_index, 0)

        image = (
            bytes.fromhex("89504e470d0a1a0a") + bytes(8)
            + (300).to_bytes(4, "big") + (400).to_bytes(4, "big")
        )
        page.services[0].on_result(
            SimpleNamespace(files=[SimpleNamespace(name="front.png", bytes=image)])
        )
        nav_handler = nav.on_change
        assert nav_handler is not None
        nav_handler(cast(Any, SimpleNamespace(control=SimpleNamespace(selected_index=1))))
        editors = [
            control for control in walk(page.controls[0])
            if isinstance(control, ft.GestureDetector) and control.on_tap_down is not None
        ]
        self.assertTrue(editors, "Android Align tab must have a touch-enabled editor.")
        self.assertTrue(
            any(isinstance(c, ft.Dropdown) and c.label == "Point to place or reposition"
                for c in walk(page.controls[0]))
        )
        tap_handler = editors[0].on_tap_down
        assert tap_handler is not None
        tap_handler(cast(Any, SimpleNamespace(local_position=SimpleNamespace(x=60, y=80))))  # pyright: ignore[reportCallIssue]
        texts = [str(c.value) for c in walk(page.controls[0]) if isinstance(c, ft.Text)]
        self.assertTrue(any("1 / " in value and "landmarks placed" in value for value in texts))


if __name__ == "__main__":
    unittest.main()
