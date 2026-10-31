# floorplan2bim

A residential floor-plan image goes in; a 3D/BIM model (IFC), a quantity takeoff and a cost estimate come out,
together with a hint on whether the estimate should be checked by hand. Accuracy is measured against the
CubiCasa5K ground truth.

```
plan image ─► recognition ─► vectorization ─► scale ─► BIM model ─► takeoff ─► cost ─► review hint
             SegFormer,      walls, doors,     px per m   IFC4 + 3D    lengths,          error model
             fine-tuned at   windows, rooms               web view     areas,            (RQ4)
             full resolution                                           counts
```

Scope: single-storey plans in the CubiCasa5K style (scans at about 1 px = 1 cm). Cost covers walls, floor finish,
doors and windows only, with placeholder unit rates.

Full plan, decisions, results and task list: **PROJECT_PLAN.md** (section 15 = current state).
Working rules for Claude Code: **CLAUDE.md**.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build for an RTX GPU
pip install -r requirements.txt
# In the repo: models/segformer_native.pt (our recognition model), models/segformer_best.pt (the public model it
# started from), samples/. Not in the repo (~5 GB): CubiCasa5K - Zenodo record 2613548, unzip to data/cubicasa5k/
```

## Use it
```bash
python web/app.py                                               # upload a plan, see image | recognition | 3D
python scripts/run_pipeline.py --image plan.png                 # one plan -> outputs/plans/plan/ (json, svg, ifc)
python scripts/run_pipeline.py --image F1_scaled.png --gt-svg model.svg   # + "vs ground truth" view
python scripts/build_demo.py                                    # static demo page -> outputs/web/index.html
```

## Evaluate and train
```bash
python evaluation/evaluate_cubicasa.py --split val --all-floors          # conditions A-E vs ground truth
python evaluation/compare_models.py outputs/run_a outputs/run_b         # models side by side (gross/net cost error, BIM geometry)
python evaluation/calibrate_scale.py outputs/run                        # scale calibration (validation split only)
python evaluation/error_model.py --train outputs/rq4/... --test outputs/evaluation   # RQ4 error model + review curve
python evaluation/recost.py outputs/evaluation --sensitivity            # new unit rates without rerunning recognition
python recognition/train_segformer.py native-prepare / native-train     # fine-tune the recognition model
```
Rules: tune on the validation split only; the test split is run once for the baseline and once for the final
system (see CLAUDE.md).

## Folders
| folder | what it does |
|---|---|
| `recognition/` | SegFormer segmentation (5 classes, tiled at full resolution), fine-tuning script, OCR of printed room areas |
| `vectorization/` | mask → walls, doors, windows, rooms; exterior walls, balconies, railings; SVG read/write |
| `bim_generation/` | scale estimation (doors, wall thickness, room labels), metric model, 3D viewer data, IFC export |
| `takeoff_cost/` | quantities and cost |
| `review/` | run-time signals and the review hint (RQ4) |
| `evaluation/` | ground-truth checks, A–E evaluation, calibration, model comparison, error model, AACE, figures |
| `web/` | demo page and live upload app |
| `samples/`, `scripts/` | synthetic test plan, demo images, command-line entry points |
| `outputs/` | results kept in git (evaluation runs, priors, figures, demo page) |
