"""Config-driven training entry point.

    python -m vehicle_inspector.training.train --config configs/models/yolov8_seg.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_inspector.config import load_config, project_root
from vehicle_inspector.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a damage segmentation model")
    ap.add_argument("--config", type=Path, required=True, help="path to a model config YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("trainable") is False:
        raise SystemExit(f"Model {cfg['name']!r} is a zero-shot baseline and is not trainable.")

    model = build_model(cfg)
    train_args = dict(cfg.get("train", {}))
    data = cfg["data"]

    # Pin the output dir to an ABSOLUTE, repo-rooted path: <repo>/runs/<task>/<name>.
    # Ultralytics resolves a *relative* `project` against its global `runs_dir` setting,
    # which previously double-nested results into runs/segment/runs/segment/... .
    # An absolute path is used as-is, so the layout is deterministic regardless of settings.
    train_args.setdefault(
        "project", str(project_root() / "runs" / cfg.get("task", "segment"))
    )

    print(f"Training {cfg['name']} ({cfg['backend']}) on {data}")
    results = model.train(data=data, **train_args)
    print("Training complete.")
    # Ultralytics returns a results object exposing the run dir / best weights.
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"Best weights under: {save_dir}")


if __name__ == "__main__":
    main()
