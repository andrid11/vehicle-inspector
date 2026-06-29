# Daily tasks — one chunk per peak session (11:00–14:00)

Each task is sized for a single focused block. Pick tomorrow's task the night before so the
peak window starts with code, not with "what do I do". One task = one demoable win.

Rule: a task is **done** only when its "Done when" line is true. If you finish early, you may
pull the next one — but never start a session without a task already chosen.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Now: finish Phase 1 (ship the MVP)

### Day 1 — Lock the work into git  ⟵ do this first, it protects everything else
- [x] Add `CarDD_release/` to `.gitignore` (raw dataset must not be committed).
- [x] `git status` → confirm no `*.pt`, no `datasets/`, no `runs/`, no `CarDD_release/`, no `.venv/`. (clean: 43 files staged)
- [x] First commit `83545ef` — "Phase 1 scaffold: CarDD data prep, YOLOv8-seg training, evaluation, FastAPI serving".
- [x] Deleted leftover lock; created GitHub repo and pushed.
- [x] **Live:** https://github.com/andrid11/vehicle-inspector — public, 1 commit, no data/weights leaked. ✅ Day 1 DONE.
- **Why:** a full day of work currently exists only in your working folder. This is the highest
  risk-to-effort task in the whole project.
- **Done when:** the repo is on GitHub, clone is < ~5 MB (no data/weights), and `git log` shows a commit.

### Day 2 — Make the run dir sane + verify weights path  ✅ DONE
- [x] **Root cause:** training `args.yaml` stored `project: runs/segment` as a *relative* path;
  Ultralytics resolves a relative project against its global `runs_dir` setting (which on this
  machine is `runs/segment`), so it got prepended twice → `runs/segment/runs/segment/...`.
- [x] **Fix (code):** `train.py` and `evaluate.py` now force an **absolute** repo-rooted project
  path (`<repo>/runs/<task>`) via `project_root()`. Absolute paths are used as-is, so the layout is
  deterministic regardless of the global Ultralytics setting. Tests still pass (10/10).
- [x] **Keeper identified + promoted:** two runs existed — `yolov8s_cardd` was *interrupted* at
  epoch 28 (91 MB weights, optimizer not stripped); `yolov8s_cardd-2` ran the **full 100 epochs**
  (23 MB weights, all val curves, mask mAP@50 0.73 vs 0.67). The full run was moved to the canonical
  `runs/segment/yolov8s_cardd/weights/best.pt`.
- [x] README / `app.py` / `GETTING_STARTED.md` already reference that exact path — verified, no edits needed.
- [ ] **YOU (mount can't delete files):** remove the leftover junk to reclaim ~180 MB:
  `Remove-Item -Recurse -Force runs\segment\runs, runs\segment\_dstparent`
- **Done when:** `runs/segment/yolov8s_cardd/weights/best.pt` exists (✅ 23 MB) and future runs land
  at `runs/segment/<name>` with no double-nesting.

> Optional belt-and-braces: also reset the global setting on your machine — `yolo settings runs_dir=runs`
> — so even ad-hoc `yolo` CLI calls outside this repo behave. Not required; the code fix already handles it.

### Day 3 — Prove the service works end-to-end
- [ ] Put 2–3 real car photos in `assets/sample_inputs/`.
- [ ] `uvicorn vehicle_inspector.serving.app:app --port 8000` with the env weights set; open `/`.
- [ ] Upload an image, confirm you get an annotated overlay + JSON `InspectionReport` back.
- [ ] Fix whatever breaks (the pipeline's `run_with_overlay` path is the most likely first failure).
- [ ] Save one screenshot of a working result into `docs/` for the README later.
- **Done when:** a browser upload returns a damage report + highlighted image, and you have the screenshot.

### Day 4 — Clean metrics artifact for the report
- [ ] Run `evaluation/evaluate.py` against the canonical weights; capture per-class mAP@50,
  mAP@[.5:.95], and mask IoU.
- [ ] Write the numbers into a small `docs/RESULTS.md` table (this becomes section 6.1 of the report).
- [ ] Note which classes are strong vs weak (you already have the confusion matrix — read it).
- **Done when:** `docs/RESULTS.md` has a per-class table and 2–3 sentences of honest analysis.

### Day 5 — Dockerize
- [ ] Build the image from `docker/Dockerfile`; fix any missing system deps (OpenCV usually needs
  `libgl1`/`libglib2.0-0`).
- [ ] `docker compose up`, hit `/health` then `/inspect` from the host.
- **Done when:** a fresh container serves a correct `/inspect` response on a sample image.

### Day 6 — Deploy to the VPS (second live demo)
- [ ] Push the image / pull on the VPS; run it behind your existing reverse proxy
  (alongside ecg-andrid.xyz).
- [ ] Smoke-test the public URL from your phone.
- **Done when:** a public URL accepts an upload and returns a report. Phase 1 = DONE.

### Day 7 — Portfolio polish (this is the resume payload)
- [ ] Update README status to "Phase 1 live", add the public demo link + the screenshot from Day 3.
- [ ] Add a 3–4 image demo gallery (input → annotated output) to `assets/` / README.
- **Done when:** someone landing on the GitHub repo understands what it does and can click a live demo
  in under 30 seconds. Drop this link into your CV.

---

## Next: Phase 2 — model comparison (the rigor layer)

Lighter chunks; start only after Phase 1 is live. Each model is "one wrapper + one config + one
training run + one row in the table" — naturally one session each.

- [ ] **Day 8** — YOLOv11-seg: add config `configs/models/yolov11_seg.yaml` (exists?), confirm the
  Ultralytics wrapper handles it with no code change, kick off training.
- [ ] **Day 9** — Evaluate YOLOv11-seg, add its row to `docs/RESULTS.md`. Now you have a real
  comparison table (v8 vs v11).
- [ ] **Day 10** — YOLOE zero-shot baseline (`models/yoloe_baseline.py` exists): run it on the val
  set with text prompts, quantify the gap vs supervised. This is the interesting result.
- [ ] **Day 11** — Per-class error analysis writeup: where zero-shot collapses (subtle scratch/dent)
  vs holds (glass shatter, tire flat). Feeds report section 6.3.
- [ ] *(stretch)* Mask R-CNN / RT-DETR-seg wrappers — only if you still want more after the above.

---

## Parking lot (don't start until Phases 1–2 are solid)
- Phase 3: part segmentation + make/model recognition → assemble full `InspectionReport`
  (the `pipeline.py` `NotImplementedError` branches).
- Phase 4: severity / repair-cost estimate; CarOS knowledge-base bridge.
- Phase 5: thesis-style report from `docs/REPORT_OUTLINE.md`.
