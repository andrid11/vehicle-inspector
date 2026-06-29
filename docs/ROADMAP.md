# Roadmap

The project is built in layers. **Each phase ends with something runnable and demoable** — so there
is always a live artifact, and complexity is added without risking "never ships".

## Phase 1 — MVP (damage segmentation, end-to-end)  ← scaffold targets this
- [ ] Convert CarDD (COCO) → YOLO-seg format (`data/prepare_cardd.py`)
- [ ] Train one supervised model: YOLOv8-seg (`training/train.py`)
- [ ] Evaluate: mAP@50, mask mIoU, per-class (`evaluation/evaluate.py`)
- [ ] FastAPI upload → annotated image + JSON report (`serving/app.py`)
- [ ] Dockerize + deploy to VPS (second live demo alongside ecg-andrid.xyz)

## Phase 2 — Model comparison (the rigor layer)
- [ ] Add YOLOv11-seg, Mask R-CNN (Detectron2), RT-DETR-seg wrappers
- [ ] Add open-vocabulary baselines: **YOLOE** (zero-shot text prompts), **SAM2** (promptable)
- [ ] Unified comparison table + per-class error analysis (where zero-shot holds vs collapses)
- [ ] Write up results in `docs/REPORT_OUTLINE.md` (thesis style)

## Phase 3 — Multi-task inspection report
- [ ] Car-part segmentation model → localize damage to a part
- [ ] Make/model recognition (fine-grained classification)
- [ ] Assemble full `InspectionReport` (vehicle + parts + damages)

## Phase 4 — Severity + knowledge bridge
- [ ] Severity / repair-cost estimate (area × type × part)
- [ ] Map detected part+issue → CarOS knowledge base (known issues / repair notes)

## Phase 5 — Polish
- [ ] README + thesis-style report with figures
- [ ] Demo gallery on real (manually sourced) auction images
- [ ] CI (lint + tests), model card, reproducibility notes

## Design rules
1. Every model implements `models/base.SegModel` and is built via `models/registry`.
2. Configs are YAML; no hardcoded paths or hyperparameters in code.
3. Evaluation is model-agnostic so the comparison table is apples-to-apples.
4. The inference pipeline degrades gracefully — missing optional models are skipped, not fatal.
