"""Frozen-model generalization test on Uganda labels.

The model never saw a Ugandan site. Extract identical features
(terrain, soil, flow accumulation) and evaluate the saved Rwanda
model on Uganda's 96 labels. Flow accumulation runs on two regional
mosaics (western and eastern clusters); accumulation truncates at
mosaic edges, an accepted caveat noted in the report.
Run: uv run python -m b2p.test_uganda
"""

import math

import joblib
import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import pandas as pd
import rasterio
from pysheds.grid import Grid
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import transform as warp_transform

from b2p.extract_soil import LAYERS
from b2p.features import TILE_DIR, site_features

CLUSTERS = {
    "west": ["S01_00_E029", "S01_00_E030", "N00_00_E029", "N00_00_E030",
             "N00_00_E031", "N01_00_E030", "N01_00_E031"],
    "east": ["N00_00_E034", "N01_00_E034"],
}


def flowacc_for_cluster(tiles: list[str], out_path: str):
    paths = [TILE_DIR / f"Copernicus_DSM_COG_10_{t}_00_DEM.tif" for t in tiles]
    srcs = [rasterio.open(p) for p in paths]
    data, transform = merge(srcs)
    h, w = data.shape[1] // 3, data.shape[2] // 3
    with rasterio.open(
        "/vsimem/full.tif", "w", driver="GTiff", height=data.shape[1],
        width=data.shape[2], count=1, dtype="float32", crs=srcs[0].crs,
        transform=transform,
    ) as tmp:
        tmp.write(data[0].astype("float32"), 1)
    with rasterio.open("/vsimem/full.tif") as tmp:
        out = tmp.read(out_shape=(1, h, w), resampling=Resampling.average)
        ct = tmp.transform * tmp.transform.scale(tmp.width / w, tmp.height / h)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs=srcs[0].crs, transform=ct, nodata=-9999,
    ) as dst:
        dst.write(out[0], 1)
    for s in srcs:
        s.close()
    grid = Grid.from_raster(out_path)
    dem = grid.read_raster(out_path)
    fdir = grid.flowdir(grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem))))
    return grid, grid.accumulation(fdir)


def main():
    df = pd.read_csv("data/labeled_clean.csv")
    lat, lon = df["GPS (Latitude)"], df["GPS (Longitude)"]
    ug = (
        lat.between(-1.5, 4.5) & lon.between(29, 35.5)
        & ~(lat.between(-3, -1) & lon.between(28, 31))
    )
    u = df[ug].copy()
    print(len(u), "Uganda labels")

    rows = []
    for r in u.itertuples():
        la, lo = getattr(r, "_5"), getattr(r, "_6")
        feats = site_features(la, lo)
        feats.update(lat=la, lon=lo, label=r.label)
        rows.append(feats)
    out = pd.DataFrame(rows)
    print("terrain done")

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
    print("soil done")

    out["upstream_km2"] = np.nan
    cell_km2 = (90 * 90) / 1e6
    for cname, tiles in CLUSTERS.items():
        grid, acc = flowacc_for_cluster(tiles, f"data/uganda_dem90_{cname}.tif")
        inv = ~grid.affine
        for i, (la, lo) in enumerate(zip(out.lat, out.lon)):
            col, row = inv * (lo, la)
            r_, c_ = int(row), int(col)
            if 0 <= r_ < acc.shape[0] and 0 <= c_ < acc.shape[1]:
                win = acc[max(0, r_ - 1):r_ + 2, max(0, c_ - 1):c_ + 2]
                out.loc[out.index[i], "upstream_km2"] = float(np.nanmax(win)) * cell_km2
        print(f"flowacc {cname} done")
    out["log_upstream"] = np.log10(out.upstream_km2 + 0.01)

    out.to_csv("data/uganda_features.csv", index=False)
    bundle = joblib.load("data/feasibility_model.joblib")
    feats = bundle["features"]
    valid = out.dropna(subset=feats + ["label"])
    p = bundle["model"].predict_proba(valid[feats].values)[:, 1]
    from sklearn.metrics import roc_auc_score
    y = valid.label.values
    top = np.argsort(-p)
    for k in [20, 40]:
        print(f"P@{k}: {y[top[:k]].mean():.2f}")
    print(f"Uganda: n={len(valid)} base rate={y.mean():.2f} AUC={roc_auc_score(y, p):.3f}")


if __name__ == "__main__":
    main()
