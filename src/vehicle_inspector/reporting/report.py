"""The inspection-report contract: the structured output of the whole system.

Pydantic models so the same schema validates the pipeline output and the API response.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Coarse severity bands derived from damage type and relative area.
SEVERITY_BANDS = ("minor", "moderate", "severe")

# Rough per-type weighting for the severity heuristic (Phase 1 placeholder;
# replaced by a learned/calibrated estimator in Phase 4).
_TYPE_WEIGHT = {
    "scratch": 0.3,
    "dent": 0.6,
    "crack": 0.7,
    "lamp_broken": 0.8,
    "glass_shatter": 0.9,
    "tire_flat": 0.9,
}


class DamageInstance(BaseModel):
    type: str = Field(..., description="damage class, e.g. 'dent'")
    confidence: float
    box: tuple[float, float, float, float] = Field(..., description="x1,y1,x2,y2 in pixels")
    area_fraction: float = Field(..., description="mask/box area as a fraction of the image")
    part: str | None = Field(None, description="localized part (Phase 3), e.g. 'front-left door'")
    severity: str = Field(..., description="one of SEVERITY_BANDS")


class VehicleInfo(BaseModel):
    make: str | None = None
    model: str | None = None
    confidence: float | None = None


class InspectionReport(BaseModel):
    image_id: str
    width: int
    height: int
    vehicle: VehicleInfo = Field(default_factory=VehicleInfo)
    damages: list[DamageInstance] = Field(default_factory=list)
    overall_severity: str = "minor"
    notes: list[str] = Field(default_factory=list)

    @property
    def damage_count(self) -> int:
        return len(self.damages)


def severity_for(damage_type: str, area_fraction: float) -> str:
    """Heuristic: combine a per-type weight with relative size into a band."""
    w = _TYPE_WEIGHT.get(damage_type, 0.5)
    score = w * (1.0 + min(area_fraction, 0.5) * 4.0)  # area boosts up to ~3x
    if score < 0.6:
        return "minor"
    if score < 1.2:
        return "moderate"
    return "severe"


def roll_up_severity(damages: list[DamageInstance]) -> str:
    """Overall = worst individual severity."""
    order = {b: i for i, b in enumerate(SEVERITY_BANDS)}
    worst = 0
    for d in damages:
        worst = max(worst, order.get(d.severity, 0))
    return SEVERITY_BANDS[worst]
