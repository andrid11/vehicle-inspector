"""Build a model from a config dict, selected by `backend`.

This is the single place that knows which wrapper implements which backend, so the rest of the
codebase stays model-agnostic. Add a model by registering one builder here.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import SegModel


def _build_ultralytics(cfg: dict[str, Any]) -> SegModel:
    from .ultralytics_seg import UltralyticsSeg

    return UltralyticsSeg(weights=cfg.get("weights", "yolov8s-seg.pt"), name=cfg["name"])


def _build_yoloe(cfg: dict[str, Any]) -> SegModel:
    from .yoloe_baseline import YOLOEBaseline

    return YOLOEBaseline(
        weights=cfg.get("weights", "yoloe-v8s-seg.pt"),
        prompts=cfg.get("prompts", {}),
        name=cfg["name"],
    )


# backend id -> builder
_BUILDERS: dict[str, Callable[[dict[str, Any]], SegModel]] = {
    "ultralytics": _build_ultralytics,
    "yoloe": _build_yoloe,
    # "detectron2": _build_detectron2,   # Phase 2: Mask R-CNN
    # "rtdetr": _build_rtdetr,           # Phase 2: RT-DETR-seg
    # "sam2": _build_sam2,               # Phase 2: promptable baseline
}


def available_backends() -> list[str]:
    return sorted(_BUILDERS)


def build_model(cfg: dict[str, Any]) -> SegModel:
    """Instantiate a SegModel from a parsed model config dict."""
    backend = cfg.get("backend")
    if backend not in _BUILDERS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {available_backends()}"
        )
    return _BUILDERS[backend](cfg)
