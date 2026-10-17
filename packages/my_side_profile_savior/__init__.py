"""Experimental side-profile landmark model factory.

This package is intentionally isolated from Mini-FaceIQ's production adapter.
Nothing here is imported by ``main.py`` or ``packages.adapter_side``. Public
symbols are lazy so importing the package does not initialize Torch/torchvision.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mapping import LandmarkMapping, load_landmark_mapping
    from .model import ProfileLandmarkModel

__all__ = [
    "LandmarkMapping",
    "ProfileLandmarkModel",
    "load_landmark_mapping",
]


def __getattr__(name: str) -> object:
    if name in {"LandmarkMapping", "load_landmark_mapping"}:
        from .mapping import LandmarkMapping, load_landmark_mapping

        return {
            "LandmarkMapping": LandmarkMapping,
            "load_landmark_mapping": load_landmark_mapping,
        }[name]
    if name == "ProfileLandmarkModel":
        from .model import ProfileLandmarkModel

        return ProfileLandmarkModel
    raise AttributeError(name)
