# Trail bridge feasibility screening

A machine learning model that predicts, from free remote-sensing data alone, whether a prospective trail bridge site is technically feasible. Built for Fika (formerly Bridges to Prosperity) so survey engineers can visit likely-feasible sites first. `REPORT.md` holds the full write-up, results, and honest limitations.

## Results in brief

- Rwanda spatial cross-validation: AUC ~0.78; precision@100 of 0.66-0.70 against a 0.24 base rate.
- Uganda transfer (frozen model, sites never seen in training): AUC 0.77, precision@20 of 0.75.
- A national screening surface: the model scored at 122,347 channel cells along every Rwandan stream draining at least 1 km² (`figures/rwanda_feasibility_network_diverging.png`).

## Data sensitivity

Unsurveyed site coordinates could identify communities, so raw site data never enters this repository. `data/` is gitignored and `figures/private/` (site-level maps) is gitignored. Everything committed here derives from public sources (Copernicus DEM, SoilGrids, published RCT bridges) or is aggregated past the point of disclosing any site.

## Layout

- `src/b2p/`: the pipeline. Feature extraction (`features.py`, `extract_*.py`), training and spatial-CV evaluation (`train_model.py`), transfer and ablation tests (`test_uganda.py`, `test_fabdem.py`), scoring (`score_backlog.py`, `score_network.py`), and map rendering (`make_map.py`, `make_network_map.py`, `make_explorer.py`).
- `figures/`: shareable outputs only.
- `REPORT.md`: the deliverable write-up for Fika.

## Running

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). Install with `uv sync`, then run pipeline stages as modules, for example:

```
uv run python -m b2p.train_model
uv run python -m b2p.score_network
uv run python -m b2p.make_network_map
```

Extraction scripts expect Copernicus DEM tiles under `data/dem_tiles/` and the site CSVs under `data/`, which are not distributed.
