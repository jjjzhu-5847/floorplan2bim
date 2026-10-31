# floorplan2bim

Floor plan image → recognition (SegFormer) → vector walls/doors/windows/rooms → scale → 3D + IFC (BIM)
→ quantity takeoff → cost → cost range by construction start month. Evaluated against CubiCasa5K ground truth.

Full plan, decisions and task list: **PROJECT_PLAN.md**. Working rules for Claude Code: **CLAUDE.md**.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build for an RTX GPU
pip install -r requirements.txt
# Already in the repo: models/segformer_best.pt, samples/demo_images/, data/ppi/ppi_monthly.csv
# Not in the repo (~5 GB): CubiCasa5K - download Zenodo record 2613548 and unzip to data/cubicasa5k/
```

## Common commands
```bash
python scripts/build_demo.py                                   # demo page -> outputs/web/index.html
python scripts/run_pipeline.py --image plan.png --band         # one plan -> outputs/plans/plan/
python evaluation/evaluate_cubicasa.py --split val --limit 30  # accuracy vs ground truth
python evaluation/make_figures.py                              # report figures -> outputs/figures/
python price_forecast/backtest.py                              # forecast backtest
python web/app.py                                              # live upload demo (pip install flask)
```

## Folders
| folder | step |
|---|---|
| `recognition/` | 1. image → 5-class pixel mask |
| `vectorization/` | 2. mask → walls, openings, rooms; SVG read/write |
| `bim_generation/` | 3–4. scale, metric model, 3D viewer data, IFC export |
| `takeoff_cost/` | 5. quantities and cost |
| `price_forecast/` | 6. price indices, backtest, cost by start month |
| `evaluation/` | accuracy vs ground truth, error attribution, AACE, figures |
| `web/` | demo page and live upload app |
| `samples/`, `scripts/` | test plan, demo images, command-line entry points |
