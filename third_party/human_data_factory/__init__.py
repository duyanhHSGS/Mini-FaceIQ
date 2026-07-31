"""Sir FaceIQ Human Data Factory.

This package is a local human-annotation tool. It deliberately contains no
model, prediction, training, benchmark, or production-provider integration.
"""

from .schema import LANDMARKS, SCHEMA_ID

__all__ = ["LANDMARKS", "SCHEMA_ID"]
