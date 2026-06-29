#!/usr/bin/env bash
# Launch the inspection API locally. Set VI_DAMAGE_WEIGHTS to your trained best.pt.
set -euo pipefail

export VI_DAMAGE_CONFIG="${VI_DAMAGE_CONFIG:-configs/models/yolov8_seg.yaml}"
# export VI_DAMAGE_WEIGHTS=runs/segment/yolov8s_cardd/weights/best.pt

exec uvicorn vehicle_inspector.serving.app:app --host 0.0.0.0 --port 8000 "$@"
