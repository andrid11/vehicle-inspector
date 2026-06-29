"""Smoke tests that exercise the pipeline with a FAKE model (no torch/ultralytics needed).

Proves the wiring — registry interface, pipeline assembly, report shape — independent of the
heavy CV backends, which is what we can verify in CI before any training happens.
"""

import numpy as np

from vehicle_inspector.models.base import Detection, SegModel
from vehicle_inspector.inference import InspectionPipeline


class FakeModel(SegModel):
    name = "fake"

    def predict(self, image, conf=0.25):
        return [
            Detection(class_name="dent", score=0.91, box=(10, 10, 60, 50)),
            Detection(class_name="glass_shatter", score=0.77, box=(70, 20, 120, 80)),
        ]


def test_pipeline_builds_report():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    pipe = InspectionPipeline(damage_model=FakeModel())
    report = pipe.run(img, image_id="unit")
    assert report.damage_count == 2
    types = {d.type for d in report.damages}
    assert types == {"dent", "glass_shatter"}
    assert report.overall_severity in ("minor", "moderate", "severe")


def test_pipeline_no_detections_notes():
    class Empty(SegModel):
        name = "empty"

        def predict(self, image, conf=0.25):
            return []

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    report = InspectionPipeline(damage_model=Empty()).run(img)
    assert report.damage_count == 0
    assert any("No damage" in n for n in report.notes)


def test_registry_lists_backends():
    from vehicle_inspector.models import available_backends

    backends = available_backends()
    assert "ultralytics" in backends and "yoloe" in backends
