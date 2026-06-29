"""Unit tests for the reporting schema + severity heuristic (no heavy deps)."""

from vehicle_inspector.reporting import (
    DamageInstance,
    InspectionReport,
    roll_up_severity,
    severity_for,
)


def test_severity_bands_monotonic_in_area():
    small = severity_for("dent", 0.01)
    big = severity_for("dent", 0.5)
    order = {"minor": 0, "moderate": 1, "severe": 2}
    assert order[big] >= order[small]


def test_severe_types_rank_high():
    assert severity_for("glass_shatter", 0.2) in ("moderate", "severe")
    assert severity_for("scratch", 0.001) == "minor"


def test_rollup_picks_worst():
    damages = [
        DamageInstance(type="scratch", confidence=0.9, box=(0, 0, 10, 10),
                       area_fraction=0.001, severity="minor"),
        DamageInstance(type="glass_shatter", confidence=0.8, box=(0, 0, 50, 50),
                       area_fraction=0.2, severity="severe"),
    ]
    assert roll_up_severity(damages) == "severe"


def test_report_serializes():
    rep = InspectionReport(image_id="x", width=100, height=80)
    d = rep.model_dump()
    assert d["image_id"] == "x"
    assert rep.damage_count == 0
