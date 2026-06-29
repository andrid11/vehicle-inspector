"""Tests for the annotation overlay + pipeline.run_with_overlay (no heavy CV deps)."""

import numpy as np

from vehicle_inspector.inference import InspectionPipeline
from vehicle_inspector.inference.annotate import draw_detections
from vehicle_inspector.models.base import Detection, SegModel


def test_draw_box_only_returns_same_shape():
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    dets = [Detection(class_name="dent", score=0.9, box=(10, 10, 60, 50))]
    out = draw_detections(img, dets)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # something was drawn (no longer all zeros)
    assert out.sum() > 0


def test_draw_with_mask_shades_region():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    dets = [Detection(class_name="scratch", score=0.8, box=(10, 10, 20, 20), mask=mask)]
    out = draw_detections(img, dets)
    assert out[15, 15].sum() > 0  # mask region got colored


class _FakeModel(SegModel):
    name = "fake"

    def predict(self, image, conf=0.25):
        return [Detection(class_name="dent", score=0.9, box=(5, 5, 25, 25))]


def test_run_with_overlay():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    report, overlay = InspectionPipeline(damage_model=_FakeModel()).run_with_overlay(img)
    assert report.damage_count == 1
    assert overlay.shape == img.shape
