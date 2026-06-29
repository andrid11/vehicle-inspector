"""Draw predicted damage (masks + boxes + labels) onto an image for the demo overlay."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from vehicle_inspector.models.base import Detection

# distinct RGB color per damage class
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "dent": (255, 99, 71),
    "scratch": (255, 193, 7),
    "crack": (156, 39, 176),
    "glass_shatter": (33, 150, 243),
    "lamp_broken": (76, 175, 80),
    "tire_flat": (244, 67, 54),
}
_DEFAULT_COLOR = (0, 200, 200)


def _color(name: str) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(name, _DEFAULT_COLOR)


def draw_detections(image: np.ndarray, detections: list[Detection], alpha: float = 0.45) -> np.ndarray:
    """Return a copy of `image` (HxWx3 uint8) with masks shaded and boxes/labels drawn."""
    h, w = image.shape[:2]
    out = image.astype(np.float32).copy()

    # 1) shade instance masks
    for d in detections:
        if d.mask is None:
            continue
        m = d.mask
        if m.shape != (h, w):
            m = np.asarray(
                Image.fromarray((m > 0).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127
        m = m.astype(bool)
        color = np.asarray(_color(d.class_name), dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color

    # 2) draw boxes + labels on top
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for d in detections:
        color = _color(d.class_name)
        x1, y1, x2, y2 = (float(v) for v in d.box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{d.class_name} {d.score:.2f}"
        ty = max(0.0, y1 - 12)
        # simple text background for legibility
        draw.rectangle([x1, ty, x1 + 7 * len(label), ty + 12], fill=color)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0))
    return np.asarray(img)
