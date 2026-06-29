# Results — Phase 1 (YOLOv8s-seg on CarDD)

Supervised damage **instance segmentation**, the Phase 1 baseline. Numbers below are the model's
performance on the **CarDD validation split (810 images)**, taken from the completed training run
(`runs/segment/yolov8s_cardd/`).

| Setting        | Value                                  |
| -------------- | -------------------------------------- |
| Model          | YOLOv8s-seg (Ultralytics), fine-tuned  |
| Pretrained from| COCO `yolov8s-seg.pt`                   |
| Epochs         | 100                                    |
| Image size     | 640                                    |
| Classes        | dent, scratch, crack, glass_shatter, lamp_broken, tire_flat |
| Eval split     | CarDD val (810 images)                 |

## Headline metrics

| Task                  | Precision | Recall | mAP@50 | mAP@[.5:.95] |
| --------------------- | --------- | ------ | ------ | ------------ |
| **Box** (detection)   | 0.791     | 0.715  | 0.745  | 0.591        |
| **Mask** (segmentation) | 0.783   | 0.707  | 0.733  | 0.559        |

## Per-class mAP@50

| Class           | Box mAP@50 | Mask mAP@50 | Recall (diag) | Missed → background |
| --------------- | ---------- | ----------- | ------------- | ------------------- |
| glass_shatter   | 0.995      | 0.995       | 0.99          | 0.01                |
| tire_flat       | 0.944      | 0.944       | 0.94          | 0.06                |
| lamp_broken     | 0.874      | 0.879       | 0.82          | 0.17                |
| dent            | 0.626      | 0.599       | 0.62          | 0.31                |
| scratch         | 0.548      | 0.532       | 0.60          | 0.38                |
| crack           | 0.483      | 0.458       | 0.59          | 0.41                |
| **all**         | **0.745**  | **0.733**   | —             | —                   |

(Per-class mAP@50 from the run's PR-curve legends; recall and miss rate from the normalized
confusion matrix. Box and mask numbers track closely, so localization quality is not the
bottleneck — detection is.)

## Analysis

There is a sharp split between two groups of damage types. **glass_shatter, tire_flat, and
lamp_broken are nearly solved** (mAP@50 0.88–0.995): these are large, high-contrast, distinctly
shaped failures, and the model finds them almost every time. **scratch, crack, and dent are the hard
cases** (mAP@50 0.46–0.63): they are thin, low-contrast, and have ambiguous boundaries, so the model
both misses a large fraction of them (31% of dents, 38% of scratches, 41% of cracks are predicted as
background) and over-fires on clean background (background regions are predicted as scratch 49% of
the time, dent 26%, crack 19%).

Crucially, the errors are almost entirely **class-vs-background**, not class-vs-class: inter-damage
confusion (e.g. dent mistaken for scratch) is ≤1%. So the model is not confused about *what* kind of
damage it sees — it struggles to reliably *detect the presence* of subtle damage at all. That points
the next improvements at recall on small/subtle instances (higher-resolution inputs, tiling, targeted
augmentation, or class-balanced sampling) rather than at the classifier.

## Reproduce

```bash
python -m vehicle_inspector.evaluation.evaluate \
  --config configs/models/yolov8_seg.yaml \
  --weights runs/segment/yolov8s_cardd/weights/best.pt
```

> Source figures for this writeup live in `runs/segment/yolov8s_cardd/`
> (`results.csv`, `BoxPR_curve.png`, `MaskPR_curve.png`, `confusion_matrix_normalized.png`).
> That folder is gitignored, so regenerate it by training, or keep a copy for the report.
