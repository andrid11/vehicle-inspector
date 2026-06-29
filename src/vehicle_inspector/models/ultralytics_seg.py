"""Ultralytics-backed supervised segmenter (YOLOv8-seg, YOLOv11-seg, ...).

Wraps the Ultralytics `YOLO` API behind our `SegModel` interface. `ultralytics` is imported
lazily so the rest of the package (config, schemas, tests) works without the heavy dep installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import Detection, SegModel


class UltralyticsSeg(SegModel):
    def __init__(self, weights: str = "yolov8s-seg.pt", name: str = "ultralytics_seg"):
        self.name = name
        self.weights = weights
        self._model = None  # lazy

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "ultralytics is required for this model. Install with `pip install ultralytics`."
                ) from e
            self._model = YOLO(self.weights)
        return self._model

    def train(self, data: str | Path, **kwargs: Any):
        """Fine-tune on a CarDD YOLO dataset. kwargs map to Ultralytics train args."""
        return self.model.train(data=str(data), **kwargs)

    def evaluate(self, data: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Validate and return a flat metrics dict (segmentation mAP)."""
        metrics = self.model.val(data=str(data), **kwargs)
        seg = getattr(metrics, "seg", None)
        out: dict[str, Any] = {}
        if seg is not None:
            out["map50"] = float(getattr(seg, "map50", float("nan")))
            out["map"] = float(getattr(seg, "map", float("nan")))
        # per-class mAP50 if available
        try:
            names = self.model.names
            per_class = getattr(seg, "ap50", None)
            if per_class is not None:
                out["per_class_map50"] = {
                    names[i]: float(v) for i, v in enumerate(per_class)
                }
        except Exception:  # pragma: no cover
            pass
        return out

    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        results = self.model.predict(image, conf=conf, verbose=False)
        dets: list[Detection] = []
        if not results:
            return dets
        r = results[0]
        names = r.names
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        mask_arr = masks.data.cpu().numpy() if masks is not None else None
        for i in range(len(xyxy)):
            mask = None
            if mask_arr is not None and i < len(mask_arr):
                mask = (mask_arr[i] > 0.5).astype(np.uint8)
            dets.append(
                Detection(
                    class_name=names[clss[i]],
                    score=float(scores[i]),
                    box=tuple(float(v) for v in xyxy[i]),
                    mask=mask,
                )
            )
        return dets
