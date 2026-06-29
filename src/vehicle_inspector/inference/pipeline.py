"""Inference pipeline: assemble model outputs into an InspectionReport.

Phase 1 wires the damage segmenter only. Parts segmentation, make/model recognition, and a
calibrated severity model slot in later behind the same `InspectionPipeline.run` call — missing
optional models are skipped, never fatal (see ROADMAP design rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vehicle_inspector.config import load_config
from vehicle_inspector.models import SegModel, build_model
from vehicle_inspector.reporting import (
    DamageInstance,
    InspectionReport,
    VehicleInfo,
    roll_up_severity,
    severity_for,
)


class InspectionPipeline:
    def __init__(
        self,
        damage_model: SegModel,
        parts_model: SegModel | None = None,
        make_model_clf: Any | None = None,
    ):
        self.damage_model = damage_model
        self.parts_model = parts_model          # Phase 3
        self.make_model_clf = make_model_clf    # Phase 3

    @classmethod
    def from_config(cls, damage_config: str | Path, weights: str | None = None) -> "InspectionPipeline":
        cfg = load_config(damage_config)
        if weights:
            cfg["weights"] = weights
        return cls(damage_model=build_model(cfg))

    def run(self, image: np.ndarray, image_id: str = "image", conf: float = 0.25) -> InspectionReport:
        """Run inference and return just the structured report."""
        report, _ = self._predict_and_report(image, image_id, conf)
        return report

    def run_with_overlay(
        self, image: np.ndarray, image_id: str = "image", conf: float = 0.25
    ) -> tuple[InspectionReport, np.ndarray]:
        """Run inference and also return the image with masks/boxes/labels drawn on it."""
        from .annotate import draw_detections

        report, detections = self._predict_and_report(image, image_id, conf)
        overlay = draw_detections(image, detections)
        return report, overlay

    def _predict_and_report(self, image, image_id, conf):
        """Shared core: predict once, build the report, return (report, detections)."""
        h, w = image.shape[:2]
        img_area = float(h * w) or 1.0

        detections = self.damage_model.predict(image, conf=conf)
        damages: list[DamageInstance] = []
        for det in detections:
            area_fraction = det.area_px / img_area
            part = None
            if self.parts_model is not None:
                part = self._assign_part(det, image)  # Phase 3
            damages.append(
                DamageInstance(
                    type=det.class_name,
                    confidence=round(det.score, 4),
                    box=det.box,
                    area_fraction=round(area_fraction, 5),
                    part=part,
                    severity=severity_for(det.class_name, area_fraction),
                )
            )

        vehicle = VehicleInfo()
        if self.make_model_clf is not None:
            vehicle = self._classify_vehicle(image)  # Phase 3

        report = InspectionReport(
            image_id=image_id,
            width=w,
            height=h,
            vehicle=vehicle,
            damages=damages,
            overall_severity=roll_up_severity(damages),
        )
        if not damages:
            report.notes.append("No damage detected above the confidence threshold.")
        return report, detections

    # --- Phase 3 hooks (kept explicit so the extension points are obvious) ---
    def _assign_part(self, det, image) -> str | None:  # pragma: no cover - Phase 3
        raise NotImplementedError

    def _classify_vehicle(self, image) -> VehicleInfo:  # pragma: no cover - Phase 3
        raise NotImplementedError
