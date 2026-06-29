# CPU image for serving. For training/GPU, base on an official CUDA + PyTorch image instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV / Pillow runtime libs needed by ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000
# Mount trained weights at runtime and point VI_DAMAGE_WEIGHTS at them.
ENV VI_DAMAGE_CONFIG=configs/models/yolov8_seg.yaml
CMD ["uvicorn", "vehicle_inspector.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
