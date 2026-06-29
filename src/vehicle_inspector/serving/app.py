"""FastAPI service: upload an image -> InspectionReport (+ optional annotated PNG).

Run:
    export VI_DAMAGE_CONFIG=configs/models/yolov8_seg.yaml
    export VI_DAMAGE_WEIGHTS=runs/segment/yolov8s_cardd/weights/best.pt
    uvicorn vehicle_inspector.serving.app:app --host 0.0.0.0 --port 8000

The model is loaded lazily on first request, so the app boots even before weights exist
(useful for health checks / CI). A minimal HTML upload form is served at '/'.
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from vehicle_inspector.serving.schemas import HealthResponse, InspectResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional warmup: set VI_WARMUP=1 to load weights + run one dummy inference at startup,
    # so the first real request isn't slowed by lazy model loading / CUDA init.
    if os.environ.get("VI_WARMUP") == "1":
        try:
            _get_pipeline().run(np.zeros((640, 640, 3), dtype=np.uint8))
        except Exception as e:  # best effort — never block startup
            print(f"[warmup] skipped: {e}")
    yield


app = FastAPI(title="Vehicle Inspector", version="0.1.0", lifespan=lifespan)

_PIPELINE = None
_MODEL_NAME = None


def _default_weights() -> str | None:
    """Canonical trained weights, used when VI_DAMAGE_WEIGHTS is not set."""
    from vehicle_inspector.config import project_root

    w = project_root() / "runs" / "segment" / "yolov8s_cardd" / "weights" / "best.pt"
    return str(w) if w.exists() else None


def _get_pipeline():
    """Lazy-load the inference pipeline from env-configured paths."""
    global _PIPELINE, _MODEL_NAME
    if _PIPELINE is None:
        from vehicle_inspector.config import load_config
        from vehicle_inspector.inference import InspectionPipeline

        cfg_path = os.environ.get("VI_DAMAGE_CONFIG", "configs/models/yolov8_seg.yaml")
        weights = os.environ.get("VI_DAMAGE_WEIGHTS")  # best.pt after training
        if not weights:
            # Fall back to the trained model if it exists, so the demo works without
            # remembering the env var. Otherwise the config's *pretrained COCO* weights
            # load instead and silently predict the wrong classes.
            weights = _default_weights()
            if weights:
                print(f"[serving] VI_DAMAGE_WEIGHTS not set; using trained weights at {weights}")
            else:
                print(
                    "[serving] WARNING: no trained weights found and VI_DAMAGE_WEIGHTS unset; "
                    "loading pretrained base weights from config. Predictions will NOT be damage "
                    "classes. Train a model or set VI_DAMAGE_WEIGHTS."
                )
        _MODEL_NAME = load_config(cfg_path).get("name")
        _PIPELINE = InspectionPipeline.from_config(cfg_path, weights=weights)
    return _PIPELINE


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_PIPELINE is not None,
        model_name=_MODEL_NAME,
    )


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...), conf: float = 0.25) -> InspectResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e

    arr = np.asarray(img)
    t0 = time.perf_counter()
    pipeline = _get_pipeline()
    report, overlay = pipeline.run_with_overlay(arr, image_id=file.filename or "upload", conf=conf)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return InspectResponse(
        report=report,
        annotated_image_b64=_b64_png(overlay),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html><html><head><meta charset="utf-8"><title>Vehicle Inspector</title>
    <style>
      body{font-family:system-ui;max-width:820px;margin:2.5rem auto;padding:0 1rem;color:#1f2a44}
      code{background:#f3f3f3;padding:.1rem .3rem;border-radius:4px}
      #out{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:1.5rem}
      #out img{max-width:480px;border:1px solid #ddd;border-radius:8px}
      pre{background:#0f172a;color:#e2e8f0;padding:1rem;border-radius:8px;overflow:auto;max-width:300px}
      .sev-severe{color:#dc2626;font-weight:600}.sev-moderate{color:#d97706;font-weight:600}
      .sev-minor{color:#16a34a;font-weight:600}
      button{padding:.5rem 1rem;border:0;background:#1f2a44;color:#fff;border-radius:6px;cursor:pointer}
    </style></head>
    <body>
      <h1>Vehicle Inspector</h1>
      <p>Upload a car photo to get a damage inspection report with the damage highlighted.</p>
      <form id="f">
        <input type="file" id="file" accept="image/*" required>
        <label>conf <input type="number" id="conf" value="0.25" step="0.05" min="0" max="1" style="width:5rem"></label>
        <button type="submit">Inspect</button>
        <span id="status"></span>
      </form>
      <div id="out"></div>
      <script>
        const f=document.getElementById('f'), out=document.getElementById('out'),
              st=document.getElementById('status');
        f.addEventListener('submit', async (e)=>{
          e.preventDefault();
          const file=document.getElementById('file').files[0];
          if(!file) return;
          const conf=document.getElementById('conf').value;
          st.textContent=' inspecting...'; out.innerHTML='';
          const fd=new FormData(); fd.append('file', file);
          const r=await fetch('/inspect?conf='+conf, {method:'POST', body:fd});
          st.textContent='';
          if(!r.ok){ out.innerHTML='<p style="color:#dc2626">Error: '+r.status+'</p>'; return; }
          const d=await r.json();
          const sev=d.report.overall_severity;
          const img=d.annotated_image_b64
            ? '<img src="data:image/png;base64,'+d.annotated_image_b64+'">' : '';
          out.innerHTML = img +
            '<div><p>Overall: <span class="sev-'+sev+'">'+sev+'</span> &middot; '+
            d.report.damages.length+' damage(s) &middot; '+Math.round(d.latency_ms)+' ms</p>'+
            '<pre>'+JSON.stringify(d.report, null, 2)+'</pre></div>';
        });
      </script>
      <p style="margin-top:2rem;color:#666">API: <code>POST /inspect</code> &middot;
         <code>GET /health</code> &middot; docs at <code>/docs</code></p>
    </body></html>
    """


def _b64_png(arr: np.ndarray) -> str:
    """Helper for later: encode an annotated image array as base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
