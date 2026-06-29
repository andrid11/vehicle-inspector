"""YOLOE open-vocabulary baseline (zero-shot, text-prompted).

No CarDD training: we set the damage class names (or richer phrases) as text prompts and measure
how far an off-the-shelf open-vocab model gets. This is the comparison's "free baseline".
"""

from __future__ import annotations

import numpy as np

from .base import Detection, SegModel


class YOLOEBaseline(SegModel):
    def __init__(
        self,
        weights: str = "yoloe-v8s-seg.pt",
        prompts: dict[str, str] | None = None,
        name: str = "yoloe_zeroshot",
    ):
        self.name = name
        self.weights = weights
        # map: canonical class name -> prompt phrase
        self.prompts = prompts or {}
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLOE
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "ultralytics>=8.3 with YOLOE support is required. `pip install -U ultralytics`."
                ) from e
            self._model = YOLOE(self.weights)
            # register text prompts as the model's vocabulary
            classes = list(self.prompts.keys())
            phrases = [self.prompts[c] for c in classes]
            self._model.set_classes(phrases, self._model.get_text_pe(phrases))
            self._prompt_classes = classes
        return self._model

    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        model = self.model
        results = model.predict(image, conf=conf, verbose=False)
        dets: list[Detection] = []
        if not results:
            return dets
        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        mask_arr = masks.data.cpu().numpy() if masks is not None else None
        for i in range(len(xyxy)):
            # map prompt index back to the canonical CarDD class name
            idx = clss[i]
            class_name = (
                self._prompt_classes[idx] if idx < len(self._prompt_classes) else str(idx)
            )
            mask = None
            if mask_arr is not None and i < len(mask_arr):
                mask = (mask_arr[i] > 0.5).astype(np.uint8)
            dets.append(
                Detection(
                    class_name=class_name,
                    score=float(scores[i]),
                    box=tuple(float(v) for v in xyxy[i]),
                    mask=mask,
                )
            )
        return dets
