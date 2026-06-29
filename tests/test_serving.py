"""Serving-path tests: exercise /health and /inspect end-to-end with a fake detector,
so the report assembly + overlay + base64 + endpoint wiring are covered without needing
torch/weights (the model itself is tested separately)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from vehicle_inspector.models.base import Detection, SegModel


class _FakeDamageModel(SegModel):
    name = "fake"

    def predict(self, image, conf: float = 0.25):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[5:25, 5:30] = 1
        return [
            Detection("dent", 0.87, (5, 5, 30, 25), mask),
            Detection("glass_shatter", 0.93, (40, 40, 90, 80), None),  # box-only path
        ]


@pytest.fixture()
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    pytest.importorskip("multipart")  # python-multipart, required for UploadFile
    from vehicle_inspector.inference import InspectionPipeline
    import vehicle_inspector.serving.app as appmod

    appmod._PIPELINE = InspectionPipeline(damage_model=_FakeDamageModel())
    appmod._MODEL_NAME = "fake"
    return fastapi_testclient.TestClient(appmod.app)


def _png_bytes(h=96, w=128):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 200, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_inspect_returns_report_and_overlay(client):
    r = client.post("/inspect", files={"file": ("car.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["report"]["width"] == 128 and d["report"]["height"] == 96
    types = {dmg["type"] for dmg in d["report"]["damages"]}
    assert {"dent", "glass_shatter"} <= types
    assert d["report"]["overall_severity"] == "severe"  # glass_shatter rolls up
    assert d["annotated_image_b64"]  # overlay PNG returned
    assert d["latency_ms"] >= 0


def test_inspect_rejects_non_image(client):
    r = client.post("/inspect", files={"file": ("x.txt", b"not-an-image", "text/plain")})
    assert r.status_code == 415
