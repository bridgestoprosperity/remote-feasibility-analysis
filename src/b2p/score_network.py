"""Score the model along Rwanda's entire river network.

Every feature derives from public data (Copernicus DEM, SoilGrids), so
the output contains no Fika site coordinates and is shareable. Cells
come from the 90 m flow-accumulation grid: every channel cell with at
least 1 km2 upstream (the 10th percentile of training sites) inside a
buffered Rwanda outline.

Caveat recorded in NOTES/REPORT: on-channel cells have HAND = 0 by
definition. Two thirds of training sites also sit at 0, so the surface
scores at the training mode for that feature.

Run: uv run python -m b2p.score_network  (~1 h; writes data/network_scored.csv)
"""

import time

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import geopandas as gpd
import joblib
import pandas as pd
import rasterio
from pysheds.grid import Grid
from rasterio.features import geometry_mask
from rasterio.warp import transform as warp_transform

from b2p.extract_flowacc import TMP as DEM90
from b2p.extract_soil import LAYERS
from b2p.features import site_features

MIN_UPSTREAM_KM2 = 1.0
CHANNEL_MIN_CELLS = 25  # HAND drainage definition, same as extract_tier1
CELL_KM2 = (90 * 90) / 1e6
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
OFFSETS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
           4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def channel_slope(dem_a, fdir_a, acc_a, r, c, steps=5):
    """Drop per meter over ~10 cells of flow path through (r, c)."""
    down = [(r, c)]
    rr, cc = r, c
    for _ in range(steps):
        d = int(fdir_a[rr, cc])
        if d not in OFFSETS:
            break
        dr, dc = OFFSETS[d]
        rr, cc = rr + dr, cc + dc
        if not (0 <= rr < dem_a.shape[0] and 0 <= cc < dem_a.shape[1]):
            break
        down.append((rr, cc))
    up = [(r, c)]
    rr, cc = r, c
    for _ in range(steps):
        best = None
        for d, (dr, dc) in OFFSETS.items():
            ur, uc = rr - dr, cc - dc
            if not (0 <= ur < dem_a.shape[0] and 0 <= uc < dem_a.shape[1]):
                continue
            if int(fdir_a[ur, uc]) == d:
                if best is None or acc_a[ur, uc] > acc_a[best]:
                    best = (ur, uc)
        if best is None:
            break
        rr, cc = best
        up.append((rr, cc))
    dist = 90.0 * (len(up) + len(down) - 2)
    if dist <= 0:
        return np.nan
    drop = dem_a[up[-1]] - dem_a[down[-1]]
    return max(float(drop), 0.0) / dist


def main():
    grid = Grid.from_raster(DEM90)
    dem = grid.read_raster(DEM90)
    filled = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(filled)
    acc = grid.accumulation(fdir)
    channels = acc > CHANNEL_MIN_CELLS
    hand = grid.compute_hand(fdir, filled, channels, dirmap=DIRMAP)
    print("hydrology grids done", flush=True)

    dem_a, fdir_a = np.asarray(filled), np.asarray(fdir)
    acc_a, hand_a = np.asarray(acc), np.asarray(hand)

    rw = gpd.read_file("zip://data/ne_countries.zip")
    rw = rw[rw.NAME == "Rwanda"]
    with rasterio.open(DEM90) as src:
        inside = ~geometry_mask(
            rw.geometry.buffer(0.01), out_shape=src.shape, transform=src.transform
        )
    keep = (acc_a * CELL_KM2 >= MIN_UPSTREAM_KM2) & inside
    ys, xs = np.where(keep)
    print(f"{len(ys)} network cells to score", flush=True)

    aff = grid.affine
    lons, lats = aff * (xs + 0.5, ys + 0.5)

    rows = []
    t0 = time.time()
    for i, (la, lo, r, c) in enumerate(zip(lats, lons, ys, xs)):
        try:
            feats = site_features(la, lo)
        except Exception:
            feats = {}
        feats.update(
            lat=la, lon=lo,
            upstream_km2=float(acc_a[r, c]) * CELL_KM2,
            hand_m=float(hand_a[r, c]) if np.isfinite(hand_a[r, c]) else np.nan,
            channel_slope=channel_slope(dem_a, fdir_a, acc_a, r, c),
        )
        rows.append(feats)
        if (i + 1) % 20000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"terrain {i + 1}/{len(ys)} ({rate:.0f}/s)", flush=True)
    out = pd.DataFrame(rows)
    out["log_upstream"] = np.log10(out.upstream_km2 + 0.01)
    print("terrain done", flush=True)

    pt_lons, pt_lats = out.lon.tolist(), out.lat.tolist()
    for name, url in LAYERS.items():
        with rasterio.open(url) as src:
            if src.crs.to_epsg() == 4326:
                coords = list(zip(pt_lons, pt_lats))
            else:
                sx, sy = warp_transform("EPSG:4326", src.crs, pt_lons, pt_lats)
                coords = list(zip(sx, sy))
            vals = [v[0] for v in src.sample(coords)]
        nodata = {None, -32768, 0} if name != "bedrock_depth_cm" else {None, -32768}
        out[name] = [float(v) if v not in nodata else float("nan") for v in vals]
        print("soil layer done:", name, flush=True)

    bundle = joblib.load("data/feasibility_model.joblib")
    feats = bundle["features"]
    valid = out.dropna(subset=feats).copy()
    valid["p_feasible"] = bundle["model"].predict_proba(valid[feats].values)[:, 1]
    slim = valid[["lat", "lon", "upstream_km2", "p_feasible"]]
    slim.to_csv("data/network_scored.csv", index=False)
    print(f"scored {len(slim)} of {len(out)} cells -> data/network_scored.csv")
    print(slim.p_feasible.describe(percentiles=[0.1, 0.5, 0.9]).round(3))


if __name__ == "__main__":
    main()
