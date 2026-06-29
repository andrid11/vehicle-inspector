# Vehicle Inspector — Vision-Based Car Damage & Inspection System

Upload photos of a used or salvage vehicle and get back a structured **inspection report**:
which parts are damaged, what kind of damage, how severe, and the vehicle's make/model — with
the damage segmented and annotated on the image.

This is a multi-task computer-vision system built as a benchmark **and** a deployable product:

- **Damage instance segmentation** (core task) on the [CarDD](https://cardd-ustc.github.io) dataset.
- A rigorous **model comparison**: supervised segmenters (YOLOv8-seg, YOLOv11-seg, Mask R-CNN,
  RT-DETR-seg) vs. **open-vocabulary baselines** (YOLOE zero-shot, SAM2 promptable).
- **Car-part segmentation** + **make/model recognition** + **severity estimation** layered on top
  to produce a full report.
- A **FastAPI + Docker** service with an upload UI, deployable to any Linux/VPS host.

> **Status:** Phase 1 complete — a supervised damage model trained on CarDD, end-to-end FastAPI
> serving, and a Docker image, all runnable. Later models slot into the same registry/eval interfaces.

---

## Architecture

```
              ┌───────────────┐
  images ───▶ │  Inference    │ ──▶ InspectionReport (JSON + annotated image)
              │  Pipeline     │
              └──────┬────────┘
                     │ calls, via the model registry:
   ┌─────────────────┼───────────────────────────────────────────┐
   ▼                 ▼                 ▼                ▼
 damage seg      parts seg        make/model        severity
 (CarDD)         (parts set)      (Stanford Cars)   (rule/regression)
   ▲
   │  benchmarked against:
   └─ YOLOv8-seg · YOLOv11-seg · Mask R-CNN · RT-DETR-seg · YOLOE (zero-shot) · SAM2
```

Every model implements the same `SegModel` interface (`src/vehicle_inspector/models/base.py`)
and is created through the **registry**, so training, evaluation, and inference are model-agnostic
and adding a model is a single new wrapper + config.

## Repo layout

```
configs/                  dataset + per-model YAML configs
src/vehicle_inspector/
  data/        download + CarDD COCO→YOLO conversion
  models/      base interface, registry, Ultralytics + YOLOE wrappers
  training/    config-driven training entry point
  evaluation/  mAP / mask-IoU per class, cross-model comparison table
  inference/   pipeline that assembles models into a report
  reporting/   InspectionReport schema (Pydantic)
  serving/     FastAPI app + request/response schemas
scripts/       helper scripts (run API, etc.)
docker/        Dockerfile + docker-compose
tests/         smoke + unit tests
```

## Quickstart

From a fresh clone, with Python 3.10+:

```bash
# 1. install
python -m pip install -e ".[dev]"

# 2. verify the wiring (no data needed) -> "7 passed"
pytest -q

# 3. get CarDD, then convert to YOLO format
python -m vehicle_inspector.data.download            # prints download steps
python -m vehicle_inspector.data.prepare_cardd --cardd-root /path/to/CarDD_release

# 4. train, evaluate, serve
python -m vehicle_inspector.training.train --config configs/models/yolov8_seg.yaml
python -m vehicle_inspector.evaluation.evaluate --config configs/models/yolov8_seg.yaml --weights runs/segment/yolov8s_cardd/weights/best.pt
uvicorn vehicle_inspector.serving.app:app --port 8000   # open http://localhost:8000
```

## Datasets

| Task                | Dataset                                   | Notes |
| ------------------- | ----------------------------------------- | ----- |
| Damage segmentation | [CarDD](https://cardd-ustc.github.io)     | 4,000 imgs, 9,163 instances, 6 classes, COCO-style |
| Part segmentation   | Car-parts segmentation set (Roboflow)     | added in Phase 3 |
| Make/model          | Stanford Cars / CompCars                  | added in Phase 3 |

## Ethics & data sourcing

This system runs on **uploaded images** and public research datasets. It is **not** wired to scrape
any auction or marketplace site (e.g. Copart/IAAI), which their terms prohibit. To demo on
real-world listings, download a few images manually and pass them to the API.

## License

MIT (see `LICENSE`).
