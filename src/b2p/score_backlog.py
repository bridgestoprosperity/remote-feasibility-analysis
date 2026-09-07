"""Score the Identified (unsurveyed) Rwanda-box sites with the trained model.

Extracts the same terrain/soil/hydrology features used in training,
then writes data/backlog_scored.csv ranked by P(feasible).
Run: uv run python -m b2p.score_backlog
"""

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

from b2p.extract_flowacc import TMP as DEM90
from b2p.extract_soil import LAYERS
from b2p.features import site_features


def main():
    df = pd.read_csv("data/all_sites_09032026.csv", encoding="utf-8-sig")
    df = df.drop_duplicates(subset="Project Code")
    lat, lon = df["GPS (Latitude)"], df["GPS (Longitude)"]
    box = df[
        (df.Stage == "Identified") & lat.between(-3, -1) & lon.between(28, 31)
    ].copy()
    print(f"{len(box)} Identified sites in Rwanda box")

    rows = []
    for i, r in enumerate(box.itertuples()):
        la, lo = getattr(r, "_5"), getattr(r, "_6")
        feats = site_features(la, lo)
        feats.update(project_code=getattr(r, "_1"), lat=la, lon=lo)
        rows.append(feats)
        if (i + 1) % 500 == 0:
            print(f"terrain {i + 1}/{len(box)}")
    out = pd.DataFrame(rows)

    lons, lats = out.lon.tolist(), out.lat.tolist()
    for name, url in LAYERS.items():
        with rasterio.open(url) as src:
            if src.crs.to_epsg() == 4326:
                coords = list(zip(lons, lats))
            else:
                xs, ys = warp_transform("EPSG:4326", src.crs, lons, lats)
                coords = list(zip(xs, ys))
            vals = [v[0] for v in src.sample(coords)]
        nodata = {None, -32768, 0} if name != "bedrock_depth_cm" else {None, -32768}
        out[name] = [float(v) if v not in nodata else float("nan") for v in vals]
        print("soil layer done:", name)

    with rasterio.open(DEM90) as src:
        pass  # flow accumulation grid gets rebuilt below only if needed
    from b2p.extract_flowacc import Grid, build_mosaic
    grid = Grid.from_raster(DEM90)
    dem = grid.read_raster(DEM90)
    filled = grid.fill_depressions(grid.fill_pits(dem))
    fdir = grid.flowdir(grid.resolve_flats(filled))
    acc = grid.accumulation(fdir)
    inv = ~grid.affine
    cell_km2 = (90 * 90) / 1e6
    vals = []
    for la, lo in zip(out.lat, out.lon):
        col, row = inv * (lo, la)
        r_, c_ = int(row), int(col)
        if 0 <= r_ < acc.shape[0] and 0 <= c_ < acc.shape[1]:
            win = acc[max(0, r_ - 1):r_ + 2, max(0, c_ - 1):c_ + 2]
            vals.append(float(np.nanmax(win)) * cell_km2)
        else:
            vals.append(float("nan"))
    out["upstream_km2"] = vals
    out["log_upstream"] = np.log10(out.upstream_km2 + 0.01)

    bundle = joblib.load("data/feasibility_model.joblib")
    feats = bundle["features"]
    valid = out.dropna(subset=feats)
    valid = valid.assign(p_feasible=bundle["model"].predict_proba(valid[feats].values)[:, 1])
    valid = valid.sort_values("p_feasible", ascending=False)
    valid.to_csv("data/backlog_scored.csv", index=False)
    print(f"scored {len(valid)} of {len(out)} sites -> data/backlog_scored.csv")
    print(valid[["project_code", "p_feasible"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
