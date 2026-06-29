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
- [x] **Leftover junk removed** (`runs\segment\runs`, `runs\segment\_dstparent`) — clean single
  `runs/segment/yolov8s_cardd/`.
- **Done when:** `runs/segment/yolov8s_cardd/weights/best.pt` exists (✅ 23 MB) and future runs land
  at `runs/segment/<name>` with no double-nesting.

> Optional belt-and-braces: also reset the global setting on your machine — `yolo settings runs_dir=runs`
> — so even ad-hoc `yolo` CLI calls outside this repo behave. Not required; the code fix already handles it.

### Day 3 — Prove the service works end-to-end  ✅ DONE (live demo confirmed)
- [x] **Traced the full request path** (serving → pipeline → model → annotate → report): the
  `Detection` contract (`area_px`, mask/box), report exports, and severity roll-up all line up.
- [x] **Verified everything except the GPU model** with a fake detector: `/health` and `/inspect`
  both return 200, report assembles with correct severities, mask + box overlay render, base64 PNG
  comes back. Locked it in as `tests/test_serving.py` — full suite now **13 passed**.
- [x] **Robustness fix:** `app.py` now auto-uses the trained weights at the canonical path when
  `VI_DAMAGE_WEIGHTS` is unset (was silently loading pretrained COCO weights → wrong classes).
- [x] **Live demo confirmed:** ran the server, uploaded an image — got back a damage report +
  annotated overlay. Trained model loads and predicts (correctly localized a dent even on an
  out-of-domain GTA-5 render). Screenshots saved in `docs/`.
- [x] **CI fixed + green:** latest `starlette` TestClient needs `httpx`, which CI's deps lacked →
  serving tests errored. Added `httpx` to `ci.yml` + dev extras; fixture now skips cleanly if absent.
  Suite: 13 passed (3 skip without httpx). Badge is green.
- **Done when:** a browser upload returns a damage report + highlighted image. ✅ Confirmed.

### Day 4 — Clean metrics artifact for the report  ✅ DONE
- [x] Pulled the real validation metrics from the completed run (`results.csv`, PR-curve legends,
  normalized confusion matrix) — no re-run needed.
- [x] Wrote `docs/RESULTS.md`: headline box/mask mAP@50 + mAP@[.5:.95], a per-class table
  (mAP@50, recall, miss-rate), and the reproduce command. Becomes report section 6.1.
- [x] Strong-vs-weak analysis grounded in real numbers: glass_shatter/tire_flat/lamp_broken near
  solved (0.88–0.995); scratch/crack/dent are the hard, subtle cases (0.46–0.63), failing as
  class-vs-background (missed + false positives), not class-vs-class confusion.
- **Done when:** `docs/RESULTS.md` has a per-class table and honest analysis. ✅
- *Headline:* Box mAP@50 **0.745**, Mask mAP@50 **0.733** (CarDD val, 810 imgs, 100 epochs).

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
