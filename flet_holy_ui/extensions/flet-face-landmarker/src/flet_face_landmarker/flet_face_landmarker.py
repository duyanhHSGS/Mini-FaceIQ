"""Flet Service bridge for the native Android MediaPipe Face Landmarker."""

from __future__ import annotations

import base64

import flet as ft


@ft.control("FletFaceLandmarker")
class FletFaceLandmarker(ft.Service):
    """Detect normalized face landmarks using the bundled .task model on Android."""

    async def detect(self, image: bytes, model: bytes) -> list[list[float]]:
        result = await self._invoke_method(
            "detect",
            {
                "image": base64.b64encode(image).decode("ascii"),
                "model": base64.b64encode(model).decode("ascii"),
            },
            timeout=120,
        )
        if not isinstance(result, list):
            raise RuntimeError("MediaPipe returned an invalid landmark list")
        return result
