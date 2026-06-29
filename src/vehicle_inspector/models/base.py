"""Common interface every segmentation model must implement.

The point of this abstraction: training, evaluation, inference, and serving never need to know
*which* model they are talking to. Adding a model = one new subclass + one config file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Detection:
    """A single predicted damage instance."""

    class_name: str
    score: float
    # axis-aligned box in pixels: (x1, y1, x2, y2)
    box: tuple[float, float, float, float]
    # optional binary instance mask, HxW (same size as the input image)
    mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def area_px(self) -> float:
        """Mask area if available, else box area — used by the severity heuristic."""
        if self.mask is not None:
            return float(self.mask.sum())
        x1, y1, x2, y2 = self.box
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


class SegModel(ABC):
    """Abstract segmentation/detection model."""

    name: str

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        """Run inference on a single RGB image (HxWx3 uint8) and return detections."""

    def train(self, data: str | Path, **kwargs: Any) -> Any:  # noqa: D401
        """Train/fine-tune the model. Open-vocab baselines may not support this."""
        raise NotImplementedError(f"{self.name} does not support training")

    def evaluate(self, data: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Return a metrics dict (mAP, mIoU, per-class). Optional for baselines."""
        raise NotImplementedError(f"{self.name} does not implement evaluate()")
