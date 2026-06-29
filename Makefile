# Convenience targets. On Windows, run these in Git Bash, or copy the underlying
# command from GETTING_STARTED.md into PowerShell.

.PHONY: install dev test lint prepare train serve docker-build docker-up clean

install:        ## install runtime deps (editable)
	pip install -e .

dev:            ## install with dev tools
	pip install -e ".[dev]"

test:           ## run the smoke + unit tests (no heavy deps needed)
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests

prepare:        ## convert CarDD COCO -> YOLO  (CARDD_ROOT=/path/to/CarDD_release)
	python -m vehicle_inspector.data.prepare_cardd --cardd-root $(CARDD_ROOT)

train:          ## train the damage segmenter on CarDD
	python -m vehicle_inspector.training.train --config configs/models/yolov8_seg.yaml

serve:          ## run the API locally on :8000
	uvicorn vehicle_inspector.serving.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t vehicle-inspector:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

clean:
	rm -rf runs .pytest_cache **/__pycache__
