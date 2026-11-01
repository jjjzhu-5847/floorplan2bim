# floorplan2bim

**What it does:** you give it a picture of a residential floor plan. It gives back a 3D building model (BIM / IFC),
the quantities an estimator needs (wall lengths and areas, floor area, doors, windows), a cost estimate, and a
range for the actual cost that also says whether an estimator should check the plan by hand.
Every number is checked against the CubiCasa5K ground truth.

The project has two parts:
- **Part 1: plan image → 3D model → quantities → cost** (steps 1–7 below). Goal: make it as accurate as possible.
- **Part 2: how far off is this estimate?** (step 8). No estimate is perfect, so the system predicts, for each plan,
  how far the actual cost could be from its estimate, and flags the plans a person should check.

**Try it:** `python web/app.py`, open http://127.0.0.1:5000 and upload a plan image (setup under "How to run").

Full plan, decisions and the results log: `PROJECT_PLAN.md` (section 9 = results, section 10 = what is left).

---

## The data
**CubiCasa5K**: 5,000 Finnish floor plans, each with a scanned image (`F1_scaled.png`) and a hand-drawn vector
ground truth (`model.svg`: every wall, door, window and room). We checked that 1 px in the image = 1 cm and
100 SVG units = 1 m, and that the ground truth lines up with the image (`evaluation/check_ground_truth.py`).

How the plans are used (no plan is used for two jobs):

| split | plans | used for |
|---|---|---|
| train | 3,000 | training our recognition model (step 1) |
| train | 1,200 (955 single-storey) | never seen by any recognition model; training the part-2 model (step 8) |
| validation | 400 (319 single-storey) | all tuning and calibration |
| test | 400 (301 single-storey) | run only twice: the baseline at the start, the final system at the end |

Scope: single-storey plans. Multi-storey sheets draw several floors side by side; they are reported as a limitation.

---

## The pipeline, step by step
One plan goes through these eight steps (`pipeline.py`). Every step below says **what** it does, **how**, **why**
it is done that way, the **result** of the final system on the test split (301 single-storey plans), and where the
**code** is.

### Step 1. Recognition: which pixels are wall, door, window, room?
- **What:** a neural network colours every pixel of the image as background, room, wall, door or window.
- **How:** SegFormer-B2 (a segmentation network pretrained on general images), fine-tuned by us on 3,000 training
  plans at the original resolution: 512 × 512 crops, randomly rescaled ×0.6–1.6 so it cannot memorise "a door is
  80 px wide", mostly centred on walls, doors and windows; 12 epochs on one RTX GPU. At run time the whole image is
  cut into overlapping 512-px tiles and the results are averaged.
- **Why:** the public model we started from shrinks the whole plan to 512 px, where a 10 cm wall is only 2–3 px wide,
  and it found doors poorly. Doors matter twice: they are counted, and step 4 measures the scale from them. On the
  validation split our B2 beat our smaller B0 on every measure (gross cost error 24.2% → 21.6%).
- **Result:** IoU wall 0.73, door 0.52, window 0.66, room 0.89 (public model: 0.60, 0.25, 0.49, 0.78).
- **Code:** `recognition/segformer.py` (run), `recognition/train_segformer.py` (train); model
  `models/segformer_native_b2.pt`.

### Step 2. Read the printed room areas (OCR)
- **What:** many plans print room sizes such as "12.5 m²". The system reads them.
- **How:** RapidOCR finds text in the image; numbers written like 12.5 or 12,5 (optionally with m²) are kept.
  Step 4 compares them with the room areas it measures.
- **Why:** a printed area measures the scale directly, independent of how well walls and doors were recognised.
- **Result:** area labels found on 30% of plans and used on 21% (the rest disagreed too much with doors and walls);
  where used, the scale from labels alone is off by 7.9% (median).
- **Code:** `recognition/ocr.py`.

### Step 3. Vectorization: from pixels to building elements
- **What:** turns the coloured pixels into what a BIM model is made of: wall centre lines with a thickness, doors
  and windows as openings in the walls, and rooms as polygons.
- **How:** rules, tuned on the validation split:
  - walls: thin the wall pixels to centre lines, keep straight horizontal / vertical runs, measure each wall's
    thickness across it, join the ends at corners and T-junctions;
  - doors / windows: door or window pixels that sit inside a wall become an opening of that wall;
  - rooms: the building outline minus the walls;
  - exterior walls: walls on the building outline. Balconies and terraces (open to the outside) are left out of the
    outline; balcony railings are dropped;
  - clean-up: short walls attached to nothing are dropped, and so are thin lines with the same room on both sides
    (usually furniture or kitchen counters, not walls).
- **Why:** a BIM model and a takeoff need elements (a wall with a length, a thickness and openings), not pixels.
  Rules are transparent and can be checked and tuned without retraining the network.
- **Result:** given perfect pixels (condition C below) the rules find the walls almost exactly (precision 1.00,
  recall 0.98) and the cost is off by only 3.2%; most of the remaining error comes from recognition and scale.
- **Code:** `vectorization/vectorize.py`.

### Step 4. Scale: how many pixels is one metre?
- **What:** converts the drawing from pixels to metres.
- **How:** four independent estimates, each from something whose real size we know on average:
  1. door widths (a typical door is 0.80 m),
  2. exterior wall thickness,
  3. interior wall thickness,
  4. the printed room areas from step 2, compared with the measured room areas.
  The typical sizes come from the training plans' ground truth. How our system measures them is calibrated on the
  validation plans (`evaluation/calibrate_scale.py`). The estimates are combined by a weighted average: the
  steadier a method, the more weight.
- **Why:** every quantity depends on it. A 10% scale error is about a 10% error on lengths and about 20% on areas.
  Several methods agree more often than one: when one is off (few doors, odd walls), the others pull it back.
- **Result:** median scale error 6.5% (doors alone 7.1%, interior walls 9.1%, exterior walls 18.8%). The baseline,
  which measured doors recognised by the public model, was off by 38%.
- **Code:** `bim_generation/scale.py`; values in `config.py` (`SCALE_PRIORS`, `SCALE_CALIBRATED`).

### Step 5. BIM model: from pixels to metres, then 3D
- **What:** builds the 3D building model.
- **How:** the elements are converted to metres and extruded with standard heights: walls 2.7 m, doors 2.1 m,
  windows from 0.9 m to 2.1 m (assumptions in `config.py`). Walls are typed exterior or interior; rooms become spaces.
  Output: an IFC4 file (opens in any BIM tool, e.g. BIM Vision or Revit) and the 3D view on the website.
- **Why:** BIM / IFC is how the industry exchanges building models, and measuring quantities from the model is how
  BIM-based takeoff works.
- **Result:** 94% of the predicted wall centre lines are real walls and 95% of the real walls are found (within
  15 cm); 60% of the rooms are found. Even perfect pixels give only about 62%, because open-plan spaces have no walls
  between them.
- **Code:** `bim_generation/geometry.py`, `bim_generation/ifc_export.py`, `bim_generation/viewer_export.py`.

### Step 6. Quantity takeoff
- **What:** measures what an estimator would measure by hand.
- **How:** from the 3D model: wall length = wall footprint area ÷ thickness (so corners are counted once), for
  exterior and interior walls separately; net wall face area = length × 2.7 m − openings; floor area of the rooms;
  doors in exterior walls are entrance doors, the rest interior doors; window count and area (width × 1.2 m).
- **Why:** these are the items for walls, finishes and openings, and each can be compared with the ground truth
  one by one, which shows where the errors are.
- **Result (median error per item):** exterior wall length −5%, interior wall length +10%, floor area −6%, window
  area −1%, interior doors +17%, entrance doors 0%, windows 0%. Interior walls are over-counted because some
  furniture lines survive and some sheets also draw the neighbouring flat.
- **Code:** `takeoff_cost/quantities.py`.

### Step 7. Cost
- **What:** prices the quantities.
- **How:** quantity × unit rate for six items: exterior wall and interior partition (per m² of net face), floor
  finish (per m²), interior and entrance doors (each), windows (per m²). The unit rates are US public data: the
  State of Michigan 2025 Assessor's Manual (Vol. II, unit-in-place costs), midpoint of each range, with walls built
  up layer by layer as the manual does (exterior wall = 2×6 studs + sheathing + vinyl siding + insulation + gypsum
  board + paint). They are converted to Houston with the DoD area cost factors (Texas 0.82 / Michigan 1.03 = 0.796;
  Houston has no row of its own): exterior wall 146 USD/m², partition 81 USD/m², floor finish 78 USD/m², interior
  door 371 USD, entrance door 634 USD, window 358 USD/m² (derivation in `config.py`). One factor scales every rate,
  so no percentage result depends on the location. `evaluation/recost.py --write` reprices all saved results when
  the rates change, without rerunning the recognition.
- **Why:** pricing weights each quantity error by what it costs, and it places the result against the AACE estimate
  classes, the industry standard for how accurate an estimate is at each project stage.
- **Result:** median cost error 12.5% (net, total vs total) and 23.2% (gross: item errors added without offsetting);
  72% of plans within ±20%; AACE Class 5 (80% of actual costs within −24% / +28% of the estimate). The baseline:
  33.1% / 40.7%, no AACE class. With every unit rate varied at random by ±30% (1,000 draws) the median error stays
  between 12.0% and 12.9% and the class stays 5, so the conclusions do not hinge on the exact rates.
- **Code:** `takeoff_cost/cost.py`.

### Step 8. Cost range and review hint (part 2)
- **What:** for each plan, an 80% range for the actual cost, its AACE class, and whether an estimator should check it.
- **How:**
  - inputs: 24 "run-time signals" the system sees for itself while running steps 1–7: from the 3D model and
    quantities (number of walls, doors, windows and rooms, floor area, wall metres per m², window-to-wall ratio,
    cost per m²), from the scale (how much the scale estimates disagree, how much the door widths vary), from the
    recognition (confidence, how much of the image is wall, door and window) and from the image (size, shape);
  - model: on the 955 held-out training plans we know both the signals and the true cost error. Gradient-boosted
    quantile regression predicts the 10th and 90th percentile of the error; a conformal correction widens the
    range until 80% of the (cross-validated) training plans fall inside. It was chosen over "one range for every
    plan" and "three risk groups" by cross-validated interval score;
  - output: e.g. for test plan 6059, estimated at 25,000 USD: "likely actual cost 21,500 – 29,700 USD (−14% / +19%),
    accuracy of an AACE Class 3 estimate" (its true cost is 29,300 USD).
    **If the range is wider than AACE Class 5, the website says: have an estimator check this plan.**
- **Why:** the user has no ground truth. One accuracy statement for the whole system hides that some plans are much
  worse than others; a range per plan says how far to trust this plan, in the industry's own terms.
- **Result:** the per-plan range scores 27% better than one range for every plan (interval score 76 vs 104) and is
  narrower (median width 39% vs 45%); 79% of test plans fall inside their 80% range. The 11% of plans flagged for
  checking have a median cost error of 34%, the others 12%; the plans not flagged reach AACE Class 4 (−19% / +25%).
- **Code:** `review/signals.py` (signals), `review/intervals.py` (range models), `review/risk.py` (one plan),
  `evaluation/error_model.py` (training and testing); model `models/error_model.joblib`.

---

## How accuracy is measured
To see **which step causes how much error**, every plan is run five ways (`evaluation/evaluate_cubicasa.py`):

| condition | geometry from | scale | isolates |
|---|---|---|---|
| A reference | ground truth | true | the "correct" quantities and cost |
| B scale only | ground truth | estimated (step 4) | the error of the scale |
| C vectorization only | ground-truth pixels → step 3 | true | the error of the vectorization rules |
| D recognition + vectorization | our model → step 3 | true | the error when the scale is known |
| E full system | our model → step 3 | estimated | what a user actually gets |

Metrics:
- **Cost error, gross** (main metric for choosing between versions): each item's quantity error is priced and the
  absolute values are added, so over- and under-counts cannot cancel. **Net** cost error (total vs total) is what a
  client sees and is reported too. Example: +10% walls and −10% floor can give a net error near 0% but a gross error
  of 20%.
- **3D geometry:** wall precision / recall, footprint overlap, share of rooms found.
- **AACE class:** which class's accuracy range holds 80% of the plans (10th to 90th percentile).

## Results

**Final system vs baseline, test split** (301 single-storey plans; `outputs/evaluation/test_comparison.csv`):

| | baseline: public model + first pipeline | **final system** |
|---|---|---|
| recognition IoU wall / door / window / room | 0.60 / 0.25 / 0.49 / 0.78 | **0.73 / 0.52 / 0.66 / 0.89** |
| 3D walls precision / recall | – | **0.94 / 0.95** |
| scale error (median) | 38.3% | **6.5%** |
| cost error by condition, net: B / C / D / **E** | 13.7% / 8.1% / 11.4% / **33.1%** | 5.4% / 3.2% / 8.1% / **12.5%** |
| cost error E, gross | 40.7% | **23.2%** |
| plans within ±20% cost error | 26% | **72%** |
| AACE class | none (−12% / +125%) | **Class 5 (−24% / +28%)** |

**Improvement path, validation split** (319 single-storey plans; each row adds to the one above; from row 2 on, E uses
each model's own scale calibration, `compare_models.py`):

| version | E net | E gross | run in `outputs/` |
|---|---|---|---|
| public model, first pipeline | 30.4% | 41.5% | `evaluation_val_baseline` |
| + vectorization fixes, scale priors and calibration, OCR | 19.2% | 29.3% | `evaluation_val_fullres` |
| + our model fine-tuned at 512 px | 17.2% | 27.7% | `evaluation_val_m512` |
| + our B0 at full resolution, balcony / railing / furniture rules | 14.0% | 24.2% | `evaluation_val_dev_sep3` |
| **+ our B2 at full resolution (final)** | **11.7%** | **21.6%** | `evaluation_val_b2` |

**Part 2, test split** (`outputs/rq4/`):

| range method | coverage (target 80%) | median width | interval score (lower is better) |
|---|---|---|---|
| one range for every plan | 74% | 45% | 104 |
| three risk groups | 77% | 47% | 99 |
| **per-plan range (chosen)** | **79%** | **39%** | **76** |

---

## How to run
```bash
# setup
python -m venv .venv && .venv\Scripts\activate             # macOS / Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# CubiCasa5K (~5 GB, Zenodo record 2613548) -> unzip to data/cubicasa5k/  (not in git)

# the website: open http://127.0.0.1:5000, upload a plan image (e.g. data/cubicasa5k/.../F1_scaled.png)
python web/app.py
# one plan from the command line -> outputs/plans/<name>/ (plan.json, plan.svg, model.ifc)
python scripts/run_pipeline.py --image F1_scaled.png --gt-svg model.svg

# evaluate (validation only for tuning; the test split has been used its two times)
python evaluation/evaluate_cubicasa.py --split val --all-floors --workers 4 --mask-cache data/seg_cache/pred_b2_native
python evaluation/compare_models.py outputs/evaluation_val_dev_sep3 outputs/evaluation_val_b2
python evaluation/test_results.py outputs/evaluation_baseline outputs/evaluation     # the test table above
python evaluation/error_model.py --train outputs/rq4/eval_train_heldout --test outputs/evaluation   # part 2
python evaluation/recost.py outputs/evaluation --sensitivity       # unit rates ±30%; --write reprices a run in place

# continue training from our model (needs CubiCasa5K and an NVIDIA GPU; ours: RTX 4070 Super, ~9 min per epoch)
python recognition/train_segformer.py native-prepare        # training crops -> data/seg_cache/ (~12 GB disk, ~10 min)
python recognition/train_segformer.py native-train models/segformer_native_b2.pt my_model   # -> models/my_model.pt
```
Continuing starts from our weights, but the optimizer state is not saved, so it is a new fine-tuning run (12 epochs,
fresh learning-rate schedule), not an exact resume. To use a new model, point `config.SEGFORMER_CKPT` to it and
recalibrate the scale on the validation split (`evaluation/calibrate_scale.py`), as in PROJECT_PLAN section 5.
Quick check after changing code: `python scripts/run_pipeline.py --svg samples/sample_plan/model.svg --name sample
--px-per-m 100` must give exterior walls 36.00 m, floor area 72.79 m², window area 13.56 m².

## Where things are
| folder | contents |
|---|---|
| `recognition/` | steps 1–2: SegFormer, its training script, OCR |
| `vectorization/` | step 3: mask → walls, openings, rooms; SVG read/write |
| `bim_generation/` | steps 4–5: scale, metric model, IFC, 3D view data |
| `takeoff_cost/` | steps 6–7: quantities and cost |
| `review/` | step 8: run-time signals, cost ranges, AACE class / review hint |
| `evaluation/` | conditions A–E, scale calibration, model comparison, test table, part-2 model, AACE, figures |
| `web/` | the website (`app.py`): upload → whole pipeline → image, recognition, 3D, takeoff, cost (quantity × unit rate; unit rates shown), cost range |
| `scripts/` | `run_pipeline.py`: one plan from the command line |
| `models/` | our recognition model (+ training curve), the part-2 model |
| `outputs/evaluation/`, `outputs/evaluation_baseline/` | test split: final system / baseline |
| `outputs/evaluation_val_*` | validation runs along the improvement path (table above) and the resolution study (`_r512_t0`, `_r768_t0`, `_r1024_t0`) |
| `outputs/rq4/` | plan lists, the held-out evaluation, part-2 results |
| `outputs/priors/`, `outputs/checks/` | scale priors and calibration; ground-truth checks |
| `config.py` | every assumption and parameter (heights, unit rates, scale calibration, AACE ranges) |
