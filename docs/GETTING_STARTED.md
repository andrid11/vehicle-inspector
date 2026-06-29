# Getting started

Step-by-step, copy-paste commands. Windows **PowerShell** is shown first (you're on Windows);
Linux/macOS equivalents follow where they differ. Anything in `<...>` you replace.

---

## 0. Prerequisites

- **Python 3.10–3.12** (`python --version`)
- **git** (you already ran `git init`)
- For training: an **NVIDIA GPU + CUDA** is strongly recommended. CPU works for inference, but
  CarDD training on CPU is very slow.
- For deployment: **Docker** (optional but recommended).

---

## 1. Create a virtual environment & install

### Windows (PowerShell)
```powershell
cd D:\VSCodeProjects\vehicle-inspector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

> If PowerShell blocks the activate script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Linux / macOS
```bash
cd vehicle-inspector
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### GPU note
`pip install -e .` pulls a default PyTorch (often CPU). For CUDA, install the matching torch
build first, then install the project:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```
Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 2. Verify the install (no data needed)

```powershell
pytest -q
```
You should see `7 passed`. These are smoke tests using a fake model — they prove the wiring works
before any training.

---

## 3. CarDD dataset

### 3a. Download
CarDD is gated behind a request form (Google Drive). Run the helper to see the steps:
```powershell
python -m vehicle_inspector.data.download
```
Download + unzip `CarDD_release`. Expected layout:
```
CarDD_release/CarDD_COCO/
  annotations/instances_train2017.json, instances_val2017.json, instances_test2017.json
  train2017/  val2017/  test2017/
```
> If your release uses different folder names, adjust `_SPLITS` in
> `src/vehicle_inspector/data/prepare_cardd.py`.

### 3b. Convert COCO → YOLO-seg
```powershell
python -m vehicle_inspector.data.prepare_cardd --cardd-root D:\path\to\CarDD_release
```
This writes `datasets/cardd_yolo/{images,labels}/{train,val,test}`, which
`configs/dataset_cardd.yaml` already points to.

---

## 4. Train the model

```powershell
python -m vehicle_inspector.training.train --config configs/models/yolov8_seg.yaml
```
Edit `configs/models/yolov8_seg.yaml` to change epochs/batch/image size. Output (weights, curves,
confusion matrix) lands in `runs/segment/yolov8s_cardd/`.

---

## 5. Evaluate

Single model:
```powershell
python -m vehicle_inspector.evaluation.evaluate `
  --config configs/models/yolov8_seg.yaml `
  --weights runs/segment/yolov8s_cardd/weights/best.pt
```
Compare several (writes a markdown table for your report):
```powershell
python -m vehicle_inspector.evaluation.evaluate --compare `
  configs/models/yolov8_seg.yaml configs/models/yolov11_seg.yaml `
  --out docs/results.md
```

---

## 6. Serve the trained model

```powershell
$env:VI_DAMAGE_CONFIG="configs/models/yolov8_seg.yaml"
$env:VI_DAMAGE_WEIGHTS="runs/segment/yolov8s_cardd/weights/best.pt"
uvicorn vehicle_inspector.serving.app:app --host 0.0.0.0 --port 8000
```
- UI: <http://localhost:8000>
- API docs: <http://localhost:8000/docs>
- Programmatic:
  ```bash
  curl -F "file=@assets/sample_inputs/car.jpg" "http://localhost:8000/inspect?conf=0.25"
  ```

---

## 7. Docker (for the VPS)

Build and run locally:
```powershell
docker build -f docker/Dockerfile -t vehicle-inspector:latest .
# put your trained best.pt in .\weights\best.pt, then:
docker compose -f docker/docker-compose.yml up --build
```
`docker-compose.yml` mounts `./weights/best.pt` into the container and sets `VI_DAMAGE_WEIGHTS`.

### Deploy to your VPS (same pattern as ecg-andrid.xyz)
1. Copy the repo + your `weights/best.pt` to the server.
2. `docker compose -f docker/docker-compose.yml up -d --build`
3. Point Nginx at `127.0.0.1:8000` and add TLS (Let's Encrypt), exactly as you did for ECG.

---

## 8. Push to GitHub

```powershell
git add .
git commit -m "Initial scaffold: multi-model vehicle damage inspection system"
git branch -M main
git remote add origin https://github.com/andrid11/vehicle-inspector.git
git push -u origin main
```
CI (`.github/workflows/ci.yml`) runs lint + tests on every push.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `python-multipart` error on `/inspect` | `pip install python-multipart` (in `.[dev]` already) |
| `torch.cuda.is_available()` is False | install the CUDA torch wheel (step 1 GPU note) |
| Ultralytics can't download weights | check network/proxy, or pre-place the `.pt` next to the project |
| CarDD category names unmapped | check the printed warning; update `_ALIASES` in `prepare_cardd.py` |
| PowerShell won't activate venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
