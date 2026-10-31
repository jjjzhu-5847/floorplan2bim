# Working agreement for Claude Code

Read `PROJECT_PLAN.md` in full before doing anything. It holds the goals, decisions, assumptions,
the day-by-day task list with acceptance criteria, and the report and presentation outlines.

## How to work in this repo
- Talk to the user in Chinese (简体中文). Code, comments, figure labels and file names stay in English.
- Run every script from the repo root, e.g. `python evaluation/evaluate_cubicasa.py --split val --limit 30`.
  Each module adds the repo root to `sys.path`, so no package install is needed.
- All parameters and assumptions live in `config.py`. Do not hard-code heights, rates, paths or class ids elsewhere.
- The "raw plan" dict (walls / openings / spaces as shapely polygons in drawing units) is the interface
  between steps; see `vectorization/__init__.py`. Keep it stable.
- Only `data/cubicasa5k/` (~5 GB) is kept out of git; each person downloads it. Model weights (`models/`),
  price data (`data/ppi/`) and results (`outputs/`) are committed so the whole team works from the same files.

## Rules for experiments
- Tune vectorization and scale methods on the **validation** split only. Run the **test** split once for the
  baseline (task 1.2) and once after improvements (task 2.5). Keep both results.
- Every number that goes into the report must come from a file under `outputs/`. Record key numbers in
  `PROJECT_PLAN.md` section 14 as you go, and mark finished tasks with ✅ in section 8.
- If a check fails (e.g. task 0.3 shows the SVG scale is not 100 units = 1 m), stop and fix that first;
  everything downstream depends on it.
- Unit rates and AACE ranges in `config.py` are placeholders / from memory. Replace them with cited values
  (tasks 3.1 and 3.4) before any cost figure goes into the report.

## Quick checks after changing code
- `python scripts/run_pipeline.py --svg samples/sample_plan/model.svg --name sample --px-per-m 100`
  must still give: exterior wall length 36.00 m, floor area 72.79 m2, window area 13.56 m2
  (TOTAL 33,943 USD with the placeholder rates).
- `python evaluation/evaluate_cubicasa.py --split val --limit 10` must finish with an empty `failed.csv`.
