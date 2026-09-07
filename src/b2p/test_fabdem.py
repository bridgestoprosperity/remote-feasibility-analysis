"""FABDEM swap test: does bare-earth elevation beat the Copernicus DSM?

Copernicus measures surface height including tree canopy; FABDEM is
the same data with forests/buildings machine-removed (CC-BY-NC-SA,
non-commercial use). Rebuild every DEM-derived feature from FABDEM
for the Rwanda labeled sites, then paired-bootstrap the full models.
Run: uv run python -m b2p.test_fabdem
"""

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pathlib import Path

import pandas as pd
import rasterio
from pysheds.grid import Grid
from rasterio.enums import Resampling
from rasterio.merge import merge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

import b2p.features as features
from b2p.extract_tier1 import CHANNEL_MIN_CELLS, DIRMAP, OFFSETS

FAB_DIR = Path("data/fabdem")
FAB_90 = "data/fabdem_rwanda_90m.tif"


def fab_tile(lat, lon):
    import math
    la, lo = math.floor(lat), math.floor(lon)
    las = f"N{la:02d}" if la >= 0 else f"S{abs(la):02d}"
    los = f"E{lo:03d}" if lo >= 0 else f"W{abs(lo):03d}"
    return FAB_DIR / f"{las}{los}_FABDEM_V1-2.tif"


def build_90m():
    srcs = [rasterio.open(p) for p in sorted(FAB_DIR.glob("*.tif"))]
    data, transform = merge(srcs)
    h, w = data.shape[1] // 3, data.shape[2] // 3
    with rasterio.open("/vsimem/fab.tif", "w", driver="GTiff", height=data.shape[1],
                       width=data.shape[2], count=1, dtype="float32",
                       crs=srcs[0].crs, transform=transform) as tmp:
        tmp.write(data[0].astype("float32"), 1)
    with rasterio.open("/vsimem/fab.tif") as tmp:
        out = tmp.read(out_shape=(1, h, w), resampling=Resampling.average)
        ct = tmp.transform * tmp.transform.scale(tmp.width / w, tmp.height / h)
    with rasterio.open(FAB_90, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs=srcs[0].crs, transform=ct,
                       nodata=-9999) as dst:
        dst.write(out[0], 1)
    for s in srcs:
        s.close()


def main():
    df = pd.read_csv("data/rwanda_features.csv")

    # terrain features from FABDEM 30 m, reusing features.py by
    # pointing its tile lookup at the FABDEM naming scheme
    features.local_tile = lambda lat, lon: fab_tile(lat, lon)
    rows = []
    for i, (la, lo) in enumerate(zip(df.lat, df.lon)):
        f = features.site_features(la, lo)
        rows.append({f"fab_{k}": v for k, v in f.items()})
        if (i + 1) % 300 == 0:
            print(f"terrain {i + 1}/{len(df)}")
    fab = pd.DataFrame(rows, index=df.index)

    # hydrology from FABDEM 90 m
    build_90m()
    grid = Grid.from_raster(FAB_90)
    dem = grid.read_raster(FAB_90)
    filled = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(filled)
    acc = grid.accumulation(fdir)
    hand = grid.compute_hand(fdir, filled, acc > CHANNEL_MIN_CELLS, dirmap=DIRMAP)
    dem_a, fdir_a, acc_a, hand_a = map(np.asarray, (filled, fdir, acc, hand))
    inv = ~grid.affine
    ups, hands, slopes = [], [], []
    for la, lo in zip(df.lat, df.lon):
        col, row = inv * (lo, la)
        r, c = int(row), int(col)
        if not (0 <= r < acc_a.shape[0] and 0 <= c < acc_a.shape[1]):
            ups.append(np.nan); hands.append(np.nan); slopes.append(np.nan)
            continue
        win = [(rr, cc) for rr in range(max(0, r - 1), min(acc_a.shape[0], r + 2))
               for cc in range(max(0, c - 1), min(acc_a.shape[1], c + 2))]
        sr, sc = max(win, key=lambda rc: acc_a[rc])
        ups.append(float(acc_a[sr, sc]) * (90 * 90) / 1e6)
        h = hand_a[r, c]
        hands.append(float(h) if np.isfinite(h) else np.nan)
        rr, cc = sr, sc
        down = [(sr, sc)]
        for _ in range(5):
            dd = int(fdir_a[rr, cc])
            if dd not in OFFSETS:
                break
            dr, dc = OFFSETS[dd]
            rr, cc = rr + dr, cc + dc
            if not (0 <= rr < dem_a.shape[0] and 0 <= cc < dem_a.shape[1]):
                break
            down.append((rr, cc))
        rr, cc = sr, sc
        up = [(sr, sc)]
        for _ in range(5):
            best = None
            for dd, (dr, dc) in OFFSETS.items():
                ur, uc = rr - dr, cc - dc
                if (0 <= ur < dem_a.shape[0] and 0 <= uc < dem_a.shape[1]
                        and int(fdir_a[ur, uc]) == dd):
                    if best is None or acc_a[ur, uc] > acc_a[best]:
                        best = (ur, uc)
            if best is None:
                break
            rr, cc = best
            up.append((rr, cc))
        dist = 90.0 * (len(up) + len(down) - 2)
        slopes.append(max(float(dem_a[up[-1]] - dem_a[down[-1]]), 0.0) / dist
                      if dist > 0 else np.nan)
    fab["fab_log_upstream"] = np.log10(np.array(ups) + 0.01)
    fab["fab_hand_m"] = hands
    fab["fab_channel_slope"] = slopes
    print("hydrology done")

    both = pd.concat([df, fab], axis=1)
    both.to_csv("data/rwanda_features_fabdem.csv", index=False)

    terrain = ["elev_m", "relief_m", "valley_depth_m", "local_slope_deg",
               "area_slope_deg", "gap_width_m", "roughness_m"]
    soil = ["clay_pct", "sand_pct", "bulk_density", "bedrock_depth_cm"]
    both["log_upstream"] = np.log10(both.upstream_km2 + 0.01)
    cop = terrain + soil + ["log_upstream", "hand_m", "channel_slope"]
    fabc = [f"fab_{t}" for t in terrain] + soil + [
        "fab_log_upstream", "fab_hand_m", "fab_channel_slope"]
    d = both.dropna(subset=cop + fabc + ["label"])
    y = d.label.values
    g = ((d.lat // 0.25).astype(int).astype(str) + "_"
         + (d.lon // 0.25).astype(int).astype(str))
    gb = lambda: HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=30, random_state=0)
    p_cop = cross_val_predict(gb(), d[cop].values, y, cv=GroupKFold(5),
                              groups=g, method="predict_proba")[:, 1]
    p_fab = cross_val_predict(gb(), d[fabc].values, y, cv=GroupKFold(5),
                              groups=g, method="predict_proba")[:, 1]
    print(f"n={len(d)}")
    print(f"Copernicus AUC={roc_auc_score(y, p_cop):.3f}")
    print(f"FABDEM     AUC={roc_auc_score(y, p_fab):.3f}")
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p_fab[i]) - roc_auc_score(y[i], p_cop[i]))
    diffs = np.array(diffs)
    print(f"FABDEM - Copernicus: {diffs.mean():.3f}, "
          f"CI [{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}], "
          f"P(>0)={np.mean(diffs > 0):.2f}")


if __name__ == "__main__":
    main()
