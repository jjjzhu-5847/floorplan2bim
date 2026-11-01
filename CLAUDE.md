# Working agreement for Claude Code

Read `PROJECT_PLAN.md` before doing anything: section 9 is the results so far, section 10 what is left. It also
holds the goals, decisions, assumptions, evaluation design and the report and presentation outlines.

## How to work in this repo
- Talk to the user in Chinese (简体中文). Code, comments, figure labels and file names stay in English.
- Run every script from the repo root, e.g. `python evaluation/evaluate_cubicasa.py --split val --limit 30`.
  Each module adds the repo root to `sys.path`, so no package install is needed.
- All parameters and assumptions live in `config.py`. Do not hard-code heights, rates, paths or class ids elsewhere.
- The "raw plan" dict (walls / openings / spaces as shapely polygons in drawing units) is the interface
  between steps; see `vectorization/__init__.py`. Keep it stable.
- Only `data/cubicasa5k/` (~5 GB) and `data/seg_cache/` (training caches) are kept out of git. Model weights
  (`models/segformer_*.pt` that are in use) and results (`outputs/`) are committed so the team shares them.
- Anything longer than about a minute (evaluation, training, caching) runs in the background with progress
  written to a log (`python -u ... > log`) and a monitor on it; report progress to the user as it comes.
- One GPU process at a time: a training run and an evaluation together exceed this machine's memory. Use
  `--mask-cache` (or `CUDA_VISIBLE_DEVICES=""`) for vectorizer work while the GPU is busy.

## Rules for experiments
- Tune recognition, vectorization and scale on the **validation** split only. The **test** split has been used both
  times it may be used (baseline `outputs/evaluation_baseline`, final system `outputs/evaluation`, 2026-10-31).
  Do not run it again or tune anything on it.
- Decision metric: the GROSS cost error of condition D/E (item quantity errors priced and added without offsetting,
  `evaluation/compare_models.py`), together with the BIM geometry metrics (wall precision / recall, footprint IoU,
  rooms found). The user ranks detection and 3D accuracy first. Report NET cost error and the AACE class too.
- Every number that goes into the report must come from a file under `outputs/`. Record key numbers in
  `PROJECT_PLAN.md` section 9 as you go.
- If a check fails, stop and fix that first; everything downstream depends on it.
- Unit rates in `config.py` come from the State of Michigan 2025 Assessor's Manual (derivation in the comments). If
  they change, run `python evaluation/recost.py <run> --write` on every saved run, then rerun `error_model.py`.

## Quick checks after changing code
- `python scripts/run_pipeline.py --svg samples/sample_plan/model.svg --name sample --px-per-m 100`
  must still give: exterior wall length 36.00 m, floor area 72.79 m2, window area 13.56 m2
  (TOTAL 36,640 USD with the current unit rates).
- `python evaluation/evaluate_cubicasa.py --split val --limit 10` must finish with an empty `failed.csv`.
