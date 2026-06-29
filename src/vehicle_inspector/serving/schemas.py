"""API response schema. The inspection report is the payload; this wraps it with meta."""

from __future__ import annotations

from pydantic import BaseModel

from vehicle_inspector.reporting import InspectionReport


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class InspectResponse(BaseModel):
    report: InspectionReport
    annotated_image_b64: str | None = None  # PNG, base64 (optional)
    latency_ms: float
