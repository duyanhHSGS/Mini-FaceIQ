"""Flet Service bridge for native Android and iOS MediaPipe Face Landmarker."""

from __future__ import annotations

import base64

import flet as ft


@ft.control("FletFaceLandmarker")
class FletFaceLandmarker(ft.Service):
    """Detect normalized face landmarks offline on Android or iOS."""

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

    async def convert_to_jpeg(self, image: bytes) -> bytes:
        """Convert iPhone HEIC/HEIF photos locally; no files leave the device."""
        result = await self._invoke_method(
            "convertToJpeg",
            {"image": base64.b64encode(image).decode("ascii")},
            timeout=120,
        )
        if not isinstance(result, str):
            raise RuntimeError("Native iOS converter returned invalid image bytes")
        return base64.b64decode(result, validate=True)
