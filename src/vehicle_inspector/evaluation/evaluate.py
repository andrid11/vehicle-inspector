"""Model-agnostic evaluation + cross-model comparison table.

Single model:
    python -m vehicle_inspector.evaluation.evaluate --config configs/models/yolov8_seg.yaml \
        --weights runs/segment/yolov8s_cardd/weights/best.pt

Compare several (writes a markdown table):
    python -m vehicle_inspector.evaluation.evaluate --compare \
        configs/models/yolov8_seg.yaml configs/models/yolov11_seg.yaml --out docs/results.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_inspector.config import load_config
from vehicle_inspector.models import build_model


def eval_one(config_path: Path, weights: str | None = None) -> dict:
    cfg = load_config(config_path)
    if weights:
        cfg["weights"] = weights
    model = build_model(cfg)
    metrics = model.evaluate(data=cfg["data"])
    return {"name": cfg["name"], **metrics}


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Model | mAP@50 | mAP@[.5:.95] |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| {r.get('name','?')} | {r.get('map50', float('nan')):.3f} "
            f"| {r.get('map', float('nan')):.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate / compare segmentation models")
    ap.add_argument("--config", type=Path, help="single model config")
    ap.add_argument("--weights", type=str, default=None, help="override weights path")
    ap.add_argument("--compare", nargs="+", type=Path, help="several configs to compare")
    ap.add_argument("--out", type=Path, default=None, help="write markdown table here")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.compare:
        for c in args.compare:
            print(f"Evaluating {c} ...")
            rows.append(eval_one(c))
    elif args.config:
        rows.append(eval_one(args.config, args.weights))
    else:
        ap.error("provide --config or --compare")

    table = to_markdown(rows)
    print("\n" + table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
