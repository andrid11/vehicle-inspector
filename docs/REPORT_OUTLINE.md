# Report outline (thesis-style)

Mirror the structure that made the ECG thesis strong: a clear problem, a rigorous multi-model
comparison under an honest evaluation protocol, and explicit statements of where the system is and
is not reliable.

1. **Introduction** — automated vehicle inspection; use cases (insurance, rental, salvage auctions);
   why segmentation (not just classification).
2. **Related work** — car damage detection datasets/methods; open-vocabulary detection (YOLOE,
   YOLO-World); promptable segmentation (SAM/SAM2).
3. **Data** — CarDD (4,000 imgs, 9,163 instances, 6 classes); splits; class imbalance; part &
   make/model datasets for the report layer.
4. **Method** — model registry; supervised segmenters vs open-vocab baselines; the multi-task
   inference pipeline; severity heuristic.
5. **Evaluation protocol** — mAP@50/[.5:.95], mask mIoU, per-class; same val set for all models;
   inference latency; (optional) a small cross-source test set of real listing images.
6. **Results**
   - 6.1 Supervised comparison table (YOLOv8/v11-seg, Mask R-CNN, RT-DETR-seg).
   - 6.2 Open-vocab baselines (YOLOE zero-shot, SAM2) — quantify the gap.
   - 6.3 Per-class behaviour — which damage types are easy (glass shatter, tire flat) vs hard
     (subtle scratch/dent); where zero-shot collapses.
   - 6.4 Latency vs accuracy trade-off; deployment choice.
7. **Reliability** — when the system works well / poorly; failure cases; honest limitations.
8. **Deployment** — FastAPI/Docker architecture; the inspection-report contract.
9. **Conclusions & future work** — severity calibration, more parts, domain adaptation to auction
   photo styles.
